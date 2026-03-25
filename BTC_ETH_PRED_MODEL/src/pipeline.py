from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import time
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

SEED = 42
WINDOW = 7
START_DATE = "2023-03-01"



TARGET_COLUMN = "y_log_price"
ASSET_TICKERS = {
    "btc": "BTC-USD",
    "eth": "ETH-USD",
    "gold": "GC=F",
    "sp500": "^GSPC",
    "usd": "DX-Y.NYB",
}

ASSET_PREFIX = {
    "eth": "Eth",
    "btc": "Btc",
}

PREDICTION_LOG_COLUMNS = [
    "logged_at",
    "model_name",
    "asset",
    "last_feature_date",
    "prediction_for_date",
    "last_close",
    "pred_log_price",
    "pred_close_price",
    "pred_change_pct",
]


@dataclass
class Artifacts:
    model: keras.Model
    scaler: MinMaxScaler
    metadata: Dict

def get_asset_prefix(asset: str) -> str:
    asset_key = asset.lower()
    if asset_key not in ASSET_PREFIX:
        raise ValueError(f"Asset tidak didukung: {asset}")
    return ASSET_PREFIX[asset_key]

def get_feature_columns(asset: str = "eth") -> List[str]:
    prefix = get_asset_prefix(asset)
    return [
        f"{prefix}_Close", f"{prefix}_Volume", f"{prefix}_Return",
        f"{prefix}_MA_7", f"{prefix}_RSI_14", f"{prefix}_MACD", f"{prefix}_Volatility_7", f"{prefix}_Range_Pct",
        f"{prefix}_Close_Lag1", f"{prefix}_Close_Lag3", f"{prefix}_Close_Lag7",
        f"{prefix}_ret_lag1", f"{prefix}_ret_lag3", f"{prefix}_ret_lag7",
        f"{prefix}_Range_Pct_Lag1", f"{prefix}_Range_Pct_Lag3", f"{prefix}_Range_Pct_Lag7",
        "Gold_ret_lag1", "Gold_ret_lag3", "Gold_ret_lag7",
        "USDIndex_ret_lag1", "USDIndex_ret_lag3", "USDIndex_ret_lag7",
        "SNP500_ret_lag1", "SNP500_ret_lag3", "SNP500_ret_lag7",
    ]

FEATURE_COLUMNS = get_feature_columns("eth")

def standardize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.to_datetime(df.index, utc=True).normalize()
    df.index = idx.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _drop_partial_current_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    today_utc = pd.Timestamp.utcnow().tz_localize(None).normalize()
    if df.index[-1] >= today_utc:
        return df.iloc[:-1].copy()
    return df

def _clean_downloaded_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.drop(
        columns=["Adj Close","Dividends", "Stock Splits", "Capital Gains"],
        errors="ignore",
    )

    df = standardize_daily_index(df)
    df = _drop_partial_current_day(df)
    return df

def _download_one_ticker(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
    max_enddate_lookback: int = 0,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:

    if end_date is None:
        current_end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    else:
        current_end = pd.Timestamp(end_date).tz_localize(None).normalize()

    last_error: Exception | None = None

    for _ in range(max_enddate_lookback + 1):
        try:
            yf_end = (current_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            df = yf.download(
                ticker,
                start=start_date,
                end=yf_end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            df = _clean_downloaded_df(df)
            if not df.empty:
                return df

            last_error = ValueError(
                f"Tidak ada data untuk ticker {ticker} sampai {current_end.date()}."
            )
        except Exception as exc:
            last_error = exc

        current_end = current_end - pd.Timedelta(days=1)
        time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Gagal download ticker {ticker} setelah mundurkan end_date sampai {max_enddate_lookback} hari."
    ) from last_error


def download_market_data(
    asset: str = "eth",
    start_date: str = START_DATE,
    end_date: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    asset_key = asset.lower()
    if asset_key not in ASSET_TICKERS:
        raise ValueError(f"Asset tidak didukung: {asset}")
    if asset_key not in ASSET_PREFIX:
        raise ValueError(f"Asset target tidak didukung untuk prediksi: {asset}")

    df_target = _download_one_ticker(
        ASSET_TICKERS[asset_key],
        start_date=start_date,
        end_date=end_date,
        max_enddate_lookback=3,
    )
    df_gold = _download_one_ticker(
        ASSET_TICKERS["gold"],
        start_date=start_date,
        end_date=end_date,
        max_enddate_lookback=10,
    )
    df_sp500 = _download_one_ticker(
        ASSET_TICKERS["sp500"],
        start_date=start_date,
        end_date=end_date,
        max_enddate_lookback=10,
    )
    df_usd = _download_one_ticker(
        ASSET_TICKERS["usd"],
        start_date=start_date,
        end_date=end_date,
        max_enddate_lookback=10,
    )

    return df_target, df_gold, df_sp500, df_usd


def _rename_target_columns(df_target: pd.DataFrame, asset: str) -> pd.DataFrame:
    prefix = get_asset_prefix(asset)
    return df_target.rename(columns={
        "Open": f"{prefix}_Open",
        "High": f"{prefix}_High",
        "Low": f"{prefix}_Low",
        "Close": f"{prefix}_Close",
        "Volume": f"{prefix}_Volume",
    })


def build_feature_frame(
    df_target: pd.DataFrame,
    df_gold: pd.DataFrame,
    df_sp500: pd.DataFrame,
    df_usd: pd.DataFrame,
    asset: str = "eth",
    inference: bool = False,
    feature_columns: List[str] | None = None,
) -> pd.DataFrame:
    import ta

    asset_key = asset.lower()
    prefix = get_asset_prefix(asset_key)
    feature_columns = feature_columns or get_feature_columns(asset_key)

    df_target = _rename_target_columns(df_target, asset_key)
    df_gold = df_gold.rename(columns={
        "Open": "Gold_Open",
        "High": "Gold_High",
        "Low": "Gold_Low",
        "Close": "Gold_Close",
        "Volume": "Gold_Volume",
    })
    df_usd = df_usd.rename(columns={
        "Open": "USDIndex_Open",
        "High": "USDIndex_High",
        "Low": "USDIndex_Low",
        "Close": "USDIndex_Close",
        "Volume": "USDIndex_Volume",
    })
    df_sp500 = df_sp500.rename(columns={
        "Open": "SNP500_Open",
        "High": "SNP500_High",
        "Low": "SNP500_Low",
        "Close": "SNP500_Close",
        "Volume": "SNP500_Volume",
    })

    cal = df_target.index
    df_gold_ff = df_gold.reindex(cal).ffill()
    df_usd_ff = df_usd.reindex(cal).ffill()
    df_sp500_ff = df_sp500.reindex(cal).ffill()

    df = df_target.join(df_gold_ff, how="left").join(df_usd_ff, how="left").join(df_sp500_ff, how="left")
    df = df.sort_index()

    close_col = f"{prefix}_Close"
    high_col = f"{prefix}_High"
    low_col = f"{prefix}_Low"
    return_col = f"{prefix}_Return"
    range_col = f"{prefix}_Range_Pct"

    df[range_col] = (df[high_col] - df[low_col]) / df[close_col]
    df[return_col] = df[close_col].pct_change()
    df["Gold_Return"] = df["Gold_Close"].pct_change()
    df["USDIndex_Return"] = df["USDIndex_Close"].pct_change()
    df["SNP500_Return"] = df["SNP500_Close"].pct_change()

    df[f"{prefix}_MA_7"] = df[close_col].shift(1).rolling(7).mean()
    df[f"{prefix}_RSI_14"] = ta.momentum.RSIIndicator(df[close_col].shift(1), window=14).rsi()
    df[f"{prefix}_MACD"] = ta.trend.MACD(df[close_col].shift(1)).macd()
    df[f"{prefix}_Volatility_7"] = df[return_col].shift(1).rolling(7).std()

    for lag in [1, 3, 7]:
        df[f"{prefix}_ret_lag{lag}"] = df[return_col].shift(lag)
        df[f"{prefix}_Range_Pct_Lag{lag}"] = df[range_col].shift(lag)
        df[f"{prefix}_Close_Lag{lag}"] = df[close_col].shift(lag)
        df[f"SNP500_ret_lag{lag}"] = df["SNP500_Return"].shift(lag)
        df[f"USDIndex_ret_lag{lag}"] = df["USDIndex_Return"].shift(lag)
        df[f"Gold_ret_lag{lag}"] = df["Gold_Return"].shift(lag)

    df = df.replace([np.inf, -np.inf], np.nan)

    if inference:
        df = df.dropna(subset=feature_columns).copy()
    else:
        df[TARGET_COLUMN] = np.log(df[close_col].shift(-1))
        df = df.dropna(subset=feature_columns + [TARGET_COLUMN]).copy()

    return df


def load_artifacts(artifacts_dir: str | Path, asset: str) -> Artifacts:
    artifacts_path = Path(artifacts_dir)

    model_path = artifacts_path / f"gru_{asset}_next_log_price.keras"
    scaler_path = artifacts_path / "minmax_scaler.pkl"
    meta_path = artifacts_path / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model belum ditemukan: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler belum ditemukan: {scaler_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata belum ditemukan: {meta_path}")

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return Artifacts(model=model, scaler=scaler, metadata=metadata)

def build_latest_sequence(
    df_features: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_columns: List[str] | None = None,
    window: int = WINDOW,
) -> Tuple[np.ndarray, pd.DataFrame]:
    feature_columns = feature_columns or FEATURE_COLUMNS
    missing_cols = [col for col in feature_columns if col not in df_features.columns]
    if missing_cols:
        raise ValueError(f"Kolom fitur hilang: {missing_cols}")

    if len(df_features) < window:
        raise ValueError(f"Data fitur kurang dari {window} hari, belum bisa membuat sequence.")

    latest = df_features[feature_columns].tail(window).copy()
    latest_scaled = scaler.transform(latest.values.astype(np.float32))
    latest_sequence = np.expand_dims(latest_scaled.astype(np.float32), axis=0)
    return latest_sequence, latest


def predict_next_close_from_latest(
    artifacts: Artifacts,
    df_features: pd.DataFrame,
    asset: str = "eth",
) -> Dict:
    asset_key = asset.lower()
    prefix = get_asset_prefix(asset_key)
    feature_columns = artifacts.metadata.get("feature_columns", get_feature_columns(asset_key))
    window = int(artifacts.metadata.get("window", WINDOW))

    latest_sequence, latest_window_df = build_latest_sequence(
        df_features=df_features,
        scaler=artifacts.scaler,
        feature_columns=feature_columns,
        window=window,
    )
    pred_log = float(artifacts.model.predict(latest_sequence, verbose=0).reshape(-1)[0])
    pred_price = float(np.exp(pred_log))

    close_col = f"{prefix}_Close"
    last_close = float(df_features[close_col].iloc[-1])
    pct_change = float((pred_price / last_close - 1.0) * 100.0)

    return {
        "asset": asset_key,
        "asset_prefix": prefix,
        "close_col": close_col,
        "last_feature_date": pd.Timestamp(df_features.index[-1]),
        "prediction_for_date": pd.Timestamp(df_features.index[-1]) + pd.Timedelta(days=1),
        "last_close": last_close,
        "pred_log_price": pred_log,
        "pred_close_price": pred_price,
        "pred_change_pct": pct_change,
        "latest_window": latest_window_df,
    }


def get_prediction_log_path(log_path_or_dir: str | Path, asset: str) -> Path:
    base_path = Path(log_path_or_dir)
    asset_key = asset.lower()

    if base_path.suffix.lower() == ".csv":
        return base_path

    asset_dir = base_path / asset_key
    if asset_dir.exists() or not (base_path / f"prediction_history_{asset_key}.csv").exists():
        return asset_dir / "prediction_history.csv"

    return base_path / f"prediction_history_{asset_key}.csv"


def load_prediction_history(log_path: str | Path) -> pd.DataFrame:
    log_path = Path(log_path)

    if log_path.exists():
        return pd.read_csv(
            log_path,
            parse_dates=["logged_at", "last_feature_date", "prediction_for_date"]
        )

    return pd.DataFrame(columns=PREDICTION_LOG_COLUMNS)


def append_prediction_history(
    log_path: str | Path,
    forecast: Dict,
    model_name: str,
) -> bool:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([{
        "logged_at": pd.Timestamp.now(),
        "model_name": model_name,
        "asset": forecast.get("asset", "eth"),
        "last_feature_date": pd.Timestamp(forecast["last_feature_date"]),
        "prediction_for_date": pd.Timestamp(forecast["prediction_for_date"]),
        "last_close": float(forecast["last_close"]),
        "pred_log_price": float(forecast["pred_log_price"]),
        "pred_close_price": float(forecast["pred_close_price"]),
        "pred_change_pct": float(forecast["pred_change_pct"]),
    }])

    history_df = load_prediction_history(log_path)

    if not history_df.empty:
        duplicate_mask = (
            (history_df["model_name"] == model_name) &
            (pd.to_datetime(history_df["last_feature_date"]) == pd.Timestamp(forecast["last_feature_date"])) &
            (pd.to_datetime(history_df["prediction_for_date"]) == pd.Timestamp(forecast["prediction_for_date"]))
        )

        if "asset" in history_df.columns:
            duplicate_mask = duplicate_mask & (history_df["asset"].astype(str).str.lower() == str(forecast.get("asset", "eth")).lower())

        if duplicate_mask.any():
            return False

    history_df = pd.concat([history_df, new_row], ignore_index=True)
    history_df.to_csv(log_path, index=False)
    return True
