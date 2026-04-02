from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import (
    FEATURE_COLUMNS,
    START_DATE,
    append_prediction_history,
    build_feature_frame,
    download_market_data,
    get_feature_columns,
    get_prediction_log_path,
    load_artifacts,
    load_prediction_history,
    predict_next_close_from_latest,
)

ROOT = Path(__file__).resolve().parent

def get_artifacts_dir(asset: str) -> Path:
    return ROOT / f"artifacts_{asset}"



ASSET_OPTIONS = {"ETH": "eth", "BTC": "btc"}

st.set_page_config(
    page_title="Crypto Forecast",
    page_icon="⛓️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_artifacts(asset: str):
    return load_artifacts(get_artifacts_dir(asset), asset=asset)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_latest_feature_frame(asset: str):
    df_target, df_gold, df_sp500, df_usd = download_market_data(asset=asset, start_date=START_DATE)
    return build_feature_frame(
        df_target,
        df_gold,
        df_sp500,
        df_usd,
        asset=asset,
        inference=True,
    )


def metric_fmt(value: float, prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}{value:,.2f}{suffix}"


with st.sidebar:
    st.subheader("Konfigurasi")
    asset_label = st.radio("Pilih asset", list(ASSET_OPTIONS.keys()), horizontal=True)
    asset = ASSET_OPTIONS[asset_label]
    auto_refresh = st.button(f"Refresh data {asset_label}")


ARTIFACTS_DIR = get_artifacts_dir(asset)

PRED_LOG_PATH = get_prediction_log_path(ARTIFACTS_DIR, asset)

st.title(f"{asset_label} Daily Forecast")
st.caption(f"Prediksi next-day close price {asset_label} berbasis model GRU.")

try:
    artifacts = get_artifacts(asset)
except Exception as exc:
    st.error(f"Artifacts model untuk {asset_label} belum ada atau belum cocok.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.write("**Model utama:** GRU")
    st.write(f"**Asset aktif:** {asset_label}")
    st.write(f"**Window:** {artifacts.metadata.get('window', 7)} hari")
    st.write(f"**Jumlah fitur:** {len(artifacts.metadata.get('feature_columns', get_feature_columns(asset)))}")
    st.divider()
    st.markdown("**Skenario inferensi**")
    st.write(
        "Model memakai window fitur terakhir yang sudah lengkap. Jika candle hari ini belum close, prediksi diarahkan ke close hari berikutnya dari bar terakhir yang lengkap."
    )

if auto_refresh:
    get_latest_feature_frame.clear()

forecast = None
forecast_error = None
df_features = None

try:
    df_features = get_latest_feature_frame(asset)
    forecast = predict_next_close_from_latest(artifacts, df_features, asset=asset)
except Exception as exc:
    forecast_error = exc

if forecast is not None:
    last_feature_date = pd.Timestamp(forecast["last_feature_date"]).date()
    prediction_for_date = pd.Timestamp(forecast["prediction_for_date"]).date()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Last complete {asset_label} close", metric_fmt(forecast["last_close"], prefix="$"))
    col2.metric("Predicted next close", metric_fmt(forecast["pred_close_price"], prefix="$"))
    col3.metric("Predicted change", metric_fmt(forecast["pred_change_pct"], suffix="%"))
    col4.metric("Latest feature date", str(last_feature_date))

    if forecast["pred_change_pct"] > 0:
        st.success(f"Sinyal model: **Bullish** untuk close {asset_label} pada {prediction_for_date}.")
    elif forecast["pred_change_pct"] < 0:
        st.warning(f"Sinyal model: **Bearish** untuk close {asset_label} pada {prediction_for_date}.")
    else:
        st.info(f"Sinyal model: **Netral** untuk close {asset_label} pada {prediction_for_date}.")
else:
    st.warning("Forecast terbaru belum bisa dibuat. Tab histori dan info model tetap tersedia.")


tab1, tab2, tab3 = st.tabs([f"Forecast {asset_label}", f"Log History {asset_label}", "Model Info"])

with tab1:
    st.subheader(f"Prediksi terbaru {asset_label}")

    if forecast_error is not None:
        st.error(f"Gagal menyiapkan prediksi {asset_label} dari data terbaru.")
        st.exception(forecast_error)

    elif forecast is not None and df_features is not None:
        st.write(
            f"Model membaca window fitur dari **{forecast['latest_window'].index.min().date()}** "
            f"sampai **{forecast['latest_window'].index.max().date()}**, "
            f"lalu memprediksi close **{asset_label}** untuk **{prediction_for_date}**."
        )

        close_col = forecast["close_col"]
        chart_df = df_features[[close_col]].tail(120).copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df[close_col],
            mode="lines",
            name=f"Actual {asset_label} Close",
        ))
        fig.add_trace(go.Scatter(
            x=[chart_df.index[-1], pd.Timestamp(prediction_for_date)],
            y=[chart_df[close_col].iloc[-1], forecast["pred_close_price"]],
            mode="lines+markers",
            name="Forecast Path",
        ))
        fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Window input terakhir")
        st.dataframe(
            forecast["latest_window"].tail(artifacts.metadata.get("window", 7)).style.format(precision=4),
            use_container_width=True,
        )

        if st.button(f"Simpan prediksi {asset_label} ke log historis"):
            saved = append_prediction_history(
                log_path=PRED_LOG_PATH,
                forecast=forecast,
                model_name=artifacts.metadata.get("model_name", f"GRU-{asset_label}"),
            )
            if saved:
                st.success(f"Prediksi {asset_label} berhasil disimpan ke log historis.")
            else:
                st.info(f"Prediksi {asset_label} ini sudah ada di log historis.")
    else:
        st.info("Forecast belum tersedia.")

with tab2:
    st.subheader(f"Histori log prediksi model {asset_label}")

    history_df = load_prediction_history(PRED_LOG_PATH)

    if not history_df.empty:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=history_df["prediction_for_date"],
            y=history_df["pred_close_price"],
            mode="lines+markers",
            name=f"Predicted {asset_label} Close",
        ))
        fig_hist.update_layout(height=500, xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.dataframe(
            history_df.sort_values("logged_at", ascending=False),
            use_container_width=True,
        )
    else:
        st.info(f"Belum ada histori prediksi {asset_label}.")

with tab3:
    st.subheader(f"Informasi model {asset_label}")
    metrics = artifacts.metadata.get("metrics", {})
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("**Metadata training**")
        st.json({
            "asset": artifacts.metadata.get("asset", asset),
            "model_name": artifacts.metadata.get("model_name"),
            "target": artifacts.metadata.get("target"),
            "train_start": artifacts.metadata.get("train_start"),
            "train_end": artifacts.metadata.get("train_end"),
            "last_training_feature_date": artifacts.metadata.get("last_training_feature_date"),
            "window": artifacts.metadata.get("window"),
        })

    with info_col2:
        st.markdown("**Metrik evaluasi**")
        st.json(metrics)

    st.markdown("**Fitur model**")
    st.code("\n".join(artifacts.metadata.get("feature_columns", get_feature_columns(asset))))

    st.caption(
        "Catatan: app ini diasumsikan menggunakan daily bar yang sudah close. Untuk deployment produksi, retrain model secara berkala dan simpan ulang artifacts asset terkait sebelum redeploy."
    )
