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
    get_feature_columns,
    load_artifacts,
    save_prediction_to_db,
    load_prediction_history_db,
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

st.title(f"{asset_label} Daily Forecast")

try:
    artifacts = get_artifacts(asset)
except Exception as exc:
    st.error(f"Artifacts model untuk {asset_label} belum ada atau belum cocok.")
    st.exception(exc)
    st.stop()
    
model_display_name = artifacts.metadata.get("model_name", "Model")
model_type = artifacts.metadata.get("model_type", "").lower()

st.caption(
    f"Prediksi next-day close price {asset_label} berbasis model {model_display_name}."
)
with st.sidebar:
    st.write(f"**Model utama:** {model_display_name}")
    st.write(f"**Asset aktif:** {asset_label}")
    st.write(f"**Jumlah fitur:** {len(artifacts.metadata.get('feature_columns', get_feature_columns(asset)))}")
    

if auto_refresh:
    get_latest_feature_frame.clear()

forecast = None
forecast_error = None
df_features = None

try:
    df_features = get_latest_feature_frame(asset)

    forecast = predict_next_close_from_latest(
        artifacts,
        df_features,
        asset=asset,
        save_prediction_to_db(
        forecast,
        artifacts.metadata.get(
            "model_name",
            f"Model-{asset.upper()}"
    )
)
    )

    save_prediction_to_db(
        forecast,
        artifacts.metadata.get(
            "model_name",
            f"Model-{asset.upper()}"
        )
    )

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
        if model_type == "xgboost":
            st.write(
                f"Model membaca fitur terbaru pada **{forecast['latest_window'].index.max().date()}**, "
                f"lalu memprediksi close **{asset_label}** untuk **{prediction_for_date}**."
            )
        else:
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

        if model_type == "xgboost":
            st.subheader("Fitur input terbaru")
            display_input = forecast["latest_window"].tail(1)
        else:
            st.subheader("Window input terakhir")
            display_input = forecast["latest_window"].tail(artifacts.metadata.get("window", 7))

        st.dataframe(
            display_input.style.format(precision=4),
            use_container_width=True,
        )



with tab2:
    st.subheader(f"Histori log prediksi model {asset_label}")

    history_df = load_prediction_history_db(asset)

if not history_df.empty:

    history_chart = history_df.sort_values(
        "prediction_for_date"
    )

    fig_history = go.Figure()

    fig_history.add_trace(
        go.Scatter(
            x=history_chart["prediction_for_date"],
            y=history_chart["pred_close_price"],
            mode="lines+markers",
            name=f"{asset_label} Historical Forecast"
        )
    )

    fig_history.update_layout(
        height=400,
        xaxis_title="Prediction Date",
        yaxis_title="Predicted Close Price (USD)"
    )

    st.plotly_chart(
        fig_history,
        use_container_width=True
    )

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
