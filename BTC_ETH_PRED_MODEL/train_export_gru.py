from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from src.pipeline import (
    FEATURE_COLUMNS,
    START_DATE,
    WINDOW,
    build_feature_frame,
    build_gru_model,
    download_market_data,
    evaluate_predictions,
    prepare_dataset,
    save_artifacts,
    split_sequences,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"


def main() -> None:
    print("Download market data...")
    df_eth, df_gold, df_sp500, df_usd = download_market_data(start_date=START_DATE)

    print("Build features...")
    df_features = build_feature_frame(df_eth, df_gold, df_sp500, df_usd)
    dataset = prepare_dataset(df_features, feature_columns=FEATURE_COLUMNS, window=WINDOW)
    split = split_sequences(dataset)

    print("Train GRU model...")
    model = build_gru_model(window=WINDOW, n_features=split["X_train"].shape[-1])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    history = model.fit(
        split["X_train"],
        split["y_train"],
        validation_data=(split["X_val"], split["y_val"]),
        epochs=200,
        batch_size=16,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluate on test split...")
    y_pred_test = model.predict(split["X_test"], verbose=0).reshape(-1)
    y_true_test = split["y_test"].reshape(-1)
    metrics = evaluate_predictions(y_true_test, y_pred_test)

    backtest_df = pd.DataFrame({
        "date": pd.to_datetime(split["id_test"]),
        "actual_log_price": y_true_test,
        "pred_log_price": y_pred_test,
        "actual_price": np.exp(y_true_test),
        "pred_price": np.exp(y_pred_test),
    })

    metadata = {
        "model_name": "GRU",
        "target": "next_day_log_close_price",
        "window": WINDOW,
        "feature_columns": FEATURE_COLUMNS,
        "train_start": str(df_features.index.min().date()),
        "train_end": str(df_features.index.max().date()),
        "last_training_feature_date": str(df_features.index[-1].date()),
        "n_rows_after_dropna": int(len(df_features)),
        "split_info": dataset.split_info,
        "metrics": metrics,
        "history": {
            "final_train_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "epochs_ran": int(len(history.history["loss"])),
        },
        "notes": {
            "inference_rule": "Gunakan hanya daily bar terakhir yang sudah close. Bila hari ini belum close, model memprediksi close hari ini dari 7 hari sebelumnya.",
            "scenario": "Input adalah 7 hari terakhir fitur ETH + makro lagged, output adalah log harga close hari berikutnya.",
        },
    }

    save_artifacts(
        output_dir=ARTIFACTS_DIR,
        model=model,
        scaler=dataset.scaler,
        metadata=metadata,
        backtest_df=backtest_df,
    )

    print("Artifacts saved to:", ARTIFACTS_DIR)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
