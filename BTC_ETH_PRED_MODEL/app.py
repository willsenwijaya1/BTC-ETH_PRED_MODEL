from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import (
    FEATURE_COLUMNS,
    START_DATE,
    build_feature_frame,
    download_market_data,
    load_artifacts,
    predict_next_close_from_latest,
    load_prediction_history,
    append_prediction_history,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
PRED_LOG_PATH = ARTIFACTS_DIR / "prediction_history.csv"

st.set_page_config(
    page_title="ETH Forecast with GRU",
    page_icon="⛓️",
    layout="wide",
)

st.title("ETH Daily Forecast - GRU")
st.caption("Prediksi next-day close price ETH berbasis model GRU dari eksperimen notebook Anda.")


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts(ARTIFACTS_DIR)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_latest_feature_frame():
    df_eth, df_gold, df_sp500, df_usd = download_market_data(start_date=START_DATE)
    return build_feature_frame(df_eth, df_gold, df_sp500, df_usd,inference=True)


def metric_fmt(value: float, prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}{value:,.2f}{suffix}"


try:
    artifacts = get_artifacts()
except Exception as exc:
    st.error(
        "Artifacts model belum ada."
    )
    st.exception(exc)
    st.stop()


with st.sidebar:
    st.subheader("Konfigurasi")
    st.write("**Model utama:** GRU")
    st.write(f"**Window:** {artifacts.metadata.get('window', 7)} hari")
    st.write(f"**Jumlah fitur:** {len(artifacts.metadata.get('feature_columns', FEATURE_COLUMNS))}")
    auto_refresh = st.button("Refresh data terbaru")
    st.divider()
    st.markdown("**Skenario inferensi**")
    st.write(
        "Model memakai 7 hari fitur terakhir yang sudah lengkap. Jika candle hari ini belum close, prediksi diarahkan ke close hari ini."
    )

if auto_refresh:
    get_latest_feature_frame.clear()

forecast = None
forecast_error = None
df_features = None
try:
    df_features = get_latest_feature_frame()
    forecast = predict_next_close_from_latest(artifacts, df_features)
except Exception as exc:
    forecast_error = exc

if forecast is not None:
    last_feature_date = pd.Timestamp(forecast["last_feature_date"]).date()
    prediction_for_date = pd.Timestamp(forecast["prediction_for_date"]).date()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last complete ETH close", metric_fmt(forecast["last_close"], prefix="$"))
    col2.metric("Predicted next close", metric_fmt(forecast["pred_close_price"], prefix="$"))
    col3.metric("Predicted change", metric_fmt(forecast["pred_change_pct"], suffix="%"))
    col4.metric("Latest feature date", str(last_feature_date))

    if forecast["pred_change_pct"] > 0:
        st.success(f"Sinyal model: **Bullish** untuk close {prediction_for_date}.")
    elif forecast["pred_change_pct"] < 0:
        st.warning(f"Sinyal model: **Bearish** untuk close {prediction_for_date}.")
    else:
        st.info(f"Sinyal model: **Netral** untuk close {prediction_for_date}.")
else:
    st.warning("Forecast terbaru belum bisa dibuat. Tab histori dan info model tetap tersedia.")

tab1, tab2, tab3 = st.tabs(["Forecast", "Backtest", "Model Info"])

with tab1:
    st.subheader("Prediksi terbaru")

    if forecast_error is not None:
        st.error("Gagal menyiapkan prediksi dari data terbaru.")
        st.exception(forecast_error)

    elif forecast is not None and df_features is not None:
        st.write(
            f"Model membaca window fitur dari **{forecast['latest_window'].index.min().date()}** "
            f"sampai **{forecast['latest_window'].index.max().date()}**, "
            f"lalu memprediksi close **{prediction_for_date}**."
        )

        chart_df = df_features[["Eth_Close"]].tail(120).copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df["Eth_Close"],
            mode="lines",
            name="Actual ETH Close",
        ))
        fig.add_trace(go.Scatter(
            x=[chart_df.index[-1], pd.Timestamp(prediction_for_date)],
            y=[chart_df["Eth_Close"].iloc[-1], forecast["pred_close_price"]],
            mode="lines+markers",
            name="Forecast Path",
        ))
        fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Window input terakhir")
        st.dataframe(
            forecast["latest_window"].tail(7).style.format(precision=4),
            use_container_width=True,
        )

        if st.button("Simpan prediksi ke log historis"):
            saved = append_prediction_history(
                log_path=PRED_LOG_PATH,
                forecast=forecast,
                model_name=artifacts.metadata.get("model_name", "GRU"),
            )
            if saved:
                st.success("Prediksi berhasil disimpan ke log historis.")
            else:
                st.info("Prediksi ini sudah ada di log historis.")
    else:
        st.info("Forecast belum tersedia.")

with tab2:
    st.subheader("Histori log prediksi model")

    history_df = load_prediction_history(PRED_LOG_PATH)

    if not history_df.empty:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=history_df["prediction_for_date"],
            y=history_df["pred_close_price"],
            mode="lines+markers",
            name="Predicted Close",
        ))
        fig_hist.update_layout(height=500, xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.dataframe(
            history_df.sort_values("logged_at", ascending=False),
            use_container_width=True,
        )
    else:
        st.info("Belum ada histori prediksi.")
        
with tab3:
    st.subheader("Informasi model")
    metrics = artifacts.metadata.get("metrics", {})
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("**Metadata training**")
        st.json({
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
    st.code("\n".join(artifacts.metadata.get("feature_columns", FEATURE_COLUMNS)))

    st.caption(
        "Catatan: app ini diasumsikan menggunakan daily bar yang sudah close. Untuk deployment produksi, retrain model secara berkala dan simpan ulang artifacts sebelum redeploy."
    )
