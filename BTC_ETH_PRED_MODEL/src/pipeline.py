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

FEATURE_COLUMNS: List[str] = [
    "Eth_Close", "Eth_Volume", "Eth_Return",
    "Eth_MA_7", "Eth_RSI_14", "Eth_MACD", "Eth_Volatility_7", "Eth_Range_Pct",
    "Eth_Close_Lag1", "Eth_Close_Lag3", "Eth_Close_Lag7",
    "Eth_ret_lag1", "Eth_ret_lag3", "Eth_ret_lag7",
    "Eth_Range_Pct_Lag1", "Eth_Range_Pct_Lag3", "Eth_Range_Pct_Lag7",
    "Gold_ret_lag1", "Gold_ret_lag3", "Gold_ret_lag7",
    "USDIndex_ret_lag1", "USDIndex_ret_lag3", "USDIndex_ret_lag7",
    "SNP500_ret_lag1", "SNP500_ret_lag3", "SNP500_ret_lag7",
]

TARGET_COLUMN = "y_log_price"
ASSET_TICKERS = {
    "eth": "ETH-USD",
    "gold": "GC=F",
    "sp500": "^GSPC",
    "usd": "DX-Y.NYB",
}

PREDICTION_LOG_COLUMNS = [
    "logged_at",
    "model_name",
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
    """
    Download data harian Yahoo Finance.

    Jika data untuk end_date terbaru belum tersedia, fungsi akan mundurkan
    end_date satu per satu sampai data ditemukan atau limit lookback habis.
    end pada yfinance bersifat exclusive, jadi kita selalu kirim current_end + 1 hari.
    """
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
    start_date: str = START_DATE,
    end_date: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_eth = _download_one_ticker(
        ASSET_TICKERS["eth"],
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

    return df_eth, df_gold, df_sp500, df_usd


def build_feature_frame(
    df_eth: pd.DataFrame,
    df_gold: pd.DataFrame,
    df_sp500: pd.DataFrame,
    df_usd: pd.DataFrame,
    inference: bool = False,
) -> pd.DataFrame:
    import ta

    df_eth = df_eth.rename(columns={
        "Open": "Eth_Open",
        "High": "Eth_High",
        "Low": "Eth_Low",
        "Close": "Eth_Close",
        "Volume": "Eth_Volume",
    })
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

    cal = df_eth.index
    df_gold_ff = df_gold.reindex(cal).ffill()
    df_usd_ff = df_usd.reindex(cal).ffill()
    df_sp500_ff = df_sp500.reindex(cal).ffill()

    df = df_eth.join(df_gold_ff, how="left").join(df_usd_ff, how="left").join(df_sp500_ff, how="left")
    df = df.sort_index()

    df["Eth_Range_Pct"] = (df["Eth_High"] - df["Eth_Low"]) / df["Eth_Close"]
    df["Eth_Return"] = df["Eth_Close"].pct_change()
    df["Gold_Return"] = df["Gold_Close"].pct_change()
    df["USDIndex_Return"] = df["USDIndex_Close"].pct_change()
    df["SNP500_Return"] = df["SNP500_Close"].pct_change()

    df["Eth_MA_7"] = df["Eth_Close"].shift(1).rolling(7).mean()
    df["Eth_RSI_14"] = ta.momentum.RSIIndicator(df["Eth_Close"].shift(1), window=14).rsi()
    df["Eth_MACD"] = ta.trend.MACD(df["Eth_Close"].shift(1)).macd()
    df["Eth_Volatility_7"] = df["Eth_Return"].shift(1).rolling(7).std()

    for lag in [1, 3, 7]:
        df[f"Eth_ret_lag{lag}"] = df["Eth_Return"].shift(lag)
        df[f"Eth_Range_Pct_Lag{lag}"] = df["Eth_Range_Pct"].shift(lag)
        df[f"Eth_Close_Lag{lag}"] = df["Eth_Close"].shift(lag)
        df[f"SNP500_ret_lag{lag}"] = df["SNP500_Return"].shift(lag)
        df[f"USDIndex_ret_lag{lag}"] = df["USDIndex_Return"].shift(lag)
        df[f"Gold_ret_lag{lag}"] = df["Gold_Return"].shift(lag)

    df = df.replace([np.inf, -np.inf], np.nan)

    if inference:
        
        df = df.dropna(subset=FEATURE_COLUMNS).copy()
    else:
        
        df[TARGET_COLUMN] = np.log(df["Eth_Close"].shift(-1))
        df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

    return df


def load_artifacts(artifacts_dir: str | Path) -> Artifacts:
    artifacts_path = Path(artifacts_dir)
    model_path = artifacts_path / "gru_eth_next_log_price.keras"
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


def build_latest_sequence(df_features: pd.DataFrame, scaler: MinMaxScaler, feature_columns: List[str] = FEATURE_COLUMNS, window: int = WINDOW) -> Tuple[np.ndarray, pd.DataFrame]:
    missing_cols = [col for col in feature_columns if col not in df_features.columns]
    if missing_cols:
        raise ValueError(f"Kolom fitur hilang: {missing_cols}")

    if len(df_features) < window:
        raise ValueError(f"Data fitur kurang dari {window} hari, belum bisa membuat sequence.")

    latest = df_features[feature_columns].tail(window).copy()
    latest_scaled = scaler.transform(latest.values.astype(np.float32))
    latest_sequence = np.expand_dims(latest_scaled.astype(np.float32), axis=0)
    return latest_sequence, latest


def predict_next_close_from_latest(artifacts: Artifacts, df_features: pd.DataFrame) -> Dict:
    latest_sequence, latest_window_df = build_latest_sequence(df_features, artifacts.scaler)
    pred_log = float(artifacts.model.predict(latest_sequence, verbose=0).reshape(-1)[0])
    pred_price = float(np.exp(pred_log))
    last_close = float(df_features["Eth_Close"].iloc[-1])
    pct_change = float((pred_price / last_close - 1.0) * 100.0)

    return {
        "last_feature_date": pd.Timestamp(df_features.index[-1]),
        "prediction_for_date": pd.Timestamp(df_features.index[-1]) + pd.Timedelta(days=1),
        "last_close": last_close,
        "pred_log_price": pred_log,
        "pred_close_price": pred_price,
        "pred_change_pct": pct_change,
        "latest_window": latest_window_df,
    }


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

        if duplicate_mask.any():
            return False

    history_df = pd.concat([history_df, new_row], ignore_index=True)
    history_df.to_csv(log_path, index=False)
    return True
