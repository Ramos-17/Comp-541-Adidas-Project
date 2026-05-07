"""
Uma Desai COMP 541
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress TensorFlow info logs (keep warnings/errors)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def aggregate_daily(df: pd.DataFrame, target: str = "Profit") -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not in DataFrame")
    if "Order_Date" not in df.columns:
        raise ValueError("'Order_Date' column required for daily aggregation")

    s = df.copy()
    s["Order_Date"] = pd.to_datetime(s["Order_Date"])

    daily = s.groupby("Order_Date")[target].sum().to_frame(target)

    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0.0)
    daily.index.name = "Order_Date"
    return daily

# ---------------------------------------------------------------------------
# Sequence construction (sliding window)
# ---------------------------------------------------------------------------

def make_sequences(series: np.ndarray, window: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series).reshape(-1)
    if len(series) <= window:
        raise ValueError(f"series length {len(series)} <= window {window}")

    X = np.lib.stride_tricks.sliding_window_view(series, window)[:-1]
    y = series[window:]
    X = X.reshape(-1, window, 1).astype(np.float32)
    y = y.astype(np.float32)
    return X, y

# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chronological_split(daily: pd.DataFrame,
                        train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(daily)
    cut = int(n * train_frac)
    return daily.iloc[:cut].copy(), daily.iloc[cut:].copy()

# ---------------------------------------------------------------------------
# Architecture + Regularization
# ---------------------------------------------------------------------------

def build_lstm(window: int = 10,
               hidden_units: int = 50,
               dropout: float = 0.20,
               l2_coeff: float = 0.01,
               lr: float = 0.001) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(
            hidden_units,
            activation="relu",
            kernel_regularizer=l2(l2_coeff),
        ),
        Dropout(dropout),
        Dense(1, kernel_regularizer=l2(l2_coeff)),
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="mse",
                  metrics=["mae"])
    return model

# ---------------------------------------------------------------------------
# 1D Convolutional Network
# ---------------------------------------------------------------------------

def build_cnn(window: int = 10,
              filters: Tuple[int, int] = (32, 64),
              kernel_size: int = 3,
              dropout: float = 0.20,
              l2_coeff: float = 0.01,
              lr: float = 0.001) -> tf.keras.Model:
    f1, f2 = filters
    model = Sequential([
        Input(shape=(window, 1)),
        Conv1D(f1, kernel_size=kernel_size, padding="causal",
               activation="relu", kernel_regularizer=l2(l2_coeff)),
        Dropout(dropout),
        Conv1D(f2, kernel_size=kernel_size, padding="causal",
               activation="relu", kernel_regularizer=l2(l2_coeff)),
        GlobalAveragePooling1D(),
        Dropout(dropout),
        Dense(1, kernel_regularizer=l2(l2_coeff)),
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="mse",
                  metrics=["mae"])
    return model

# ---------------------------------------------------------------------------
# Training - shared between LSTM and CNN
# ---------------------------------------------------------------------------

def _fit_sequence_model(model: tf.keras.Model,
                        daily_train: pd.DataFrame,
                        target: str,
                        window: int,
                        epochs: int,
                        batch_size: int,
                        val_frac: float,
                        early_stop_patience: int,
                        seed: int,
                        verbose: int):
    # Reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 1. Scale on training only
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(daily_train[[target]].values).reshape(-1)

    # 2. Sequences
    X, y = make_sequences(series_scaled, window=window)

    # 3. Forward-chain inner-validation split for EarlyStopping
    n = len(X)
    cut = int(n * (1 - val_frac))
    X_tr, y_tr = X[:cut], y[:cut]
    X_val, y_val = X[cut:], y[cut:]

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=early_stop_patience,
        restore_best_weights=True,
        verbose=1 if verbose else 0,
    )

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        shuffle=False,            # preserve temporal order
        verbose=verbose,
    )
    return model, history, scaler

def train_lstm(daily_train: pd.DataFrame,
               target: str = "Profit",
               window: int = 10,
               hidden_units: int = 50,
               dropout: float = 0.20,
               l2_coeff: float = 0.01,
               lr: float = 0.001,
               epochs: int = 50,
               batch_size: int = 16,
               val_frac: float = 0.20,
               early_stop_patience: int = 10,
               seed: int = 42,
               verbose: int = 1):
    model = build_lstm(window=window, hidden_units=hidden_units,
                       dropout=dropout, l2_coeff=l2_coeff, lr=lr)
    return _fit_sequence_model(
        model, daily_train, target, window, epochs, batch_size,
        val_frac, early_stop_patience, seed, verbose,
    )

def train_cnn(daily_train: pd.DataFrame,
              target: str = "Profit",
              window: int = 10,
              filters: Tuple[int, int] = (32, 64),
              kernel_size: int = 3,
              dropout: float = 0.20,
              l2_coeff: float = 0.01,
              lr: float = 0.001,
              epochs: int = 50,
              batch_size: int = 16,
              val_frac: float = 0.20,
              early_stop_patience: int = 10,
              seed: int = 42,
              verbose: int = 1):
    model = build_cnn(window=window, filters=filters, kernel_size=kernel_size,
                      dropout=dropout, l2_coeff=l2_coeff, lr=lr)
    return _fit_sequence_model(
        model, daily_train, target, window, epochs, batch_size,
        val_frac, early_stop_patience, seed, verbose,
    )

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(model: tf.keras.Model,
                     scaler: MinMaxScaler,
                     daily_train: pd.DataFrame,
                     daily_test: pd.DataFrame,
                     target: str = "Profit",
                     window: int = 10) -> Tuple[Dict[str, float], pd.DataFrame]:
    # Apply the train-fit scaler to the full series (no re-fitting!)
    full = pd.concat([daily_train, daily_test])
    full_scaled = scaler.transform(full[[target]].values).reshape(-1)

    # Build sequences over the full series and slice to test indices
    X_all, y_all = make_sequences(full_scaled, window=window)

    # The first test-period y-index sits at (n_train - window) inside X_all
    first_test_idx = len(daily_train) - window
    X_test = X_all[first_test_idx:]
    y_test_scaled = y_all[first_test_idx:]
    test_dates = full.index[len(daily_train):]

    # Predict + invert scaling
    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_true = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    nonzero = y_true > 0
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = float("nan")

    preds = pd.DataFrame(
        {"actual": y_true, "predicted": y_pred},
        index=test_dates,
    )

    metrics = {
        "rmse_dollars": rmse,
        "mae_dollars": mae,
        "mape_pct": mape,
        "r2": r2,
        "n_test_points": int(len(y_true)),
    }
    return metrics, preds

# ---------------------------------------------------------------------------
# Plots (used by the demo notebook)
# ---------------------------------------------------------------------------

def plot_loss_curves(history, save_path: Optional[str] = None,
                     model_name: str = "LSTM"):
    """Reproduces the 'Model Loss Function over Epochs' plot in Section 4.4."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="Training Loss (MSE)")
    ax.plot(history.history["val_loss"], label="Validation Loss (MSE)")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (Mean Squared Error)")
    ax.set_title(f"{model_name} Model Loss Function over Epochs")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_predictions(preds: pd.DataFrame,
                     target: str = "Profit",
                     save_path: Optional[str] = None,
                     model_name: str = "LSTM"):
    """Actual vs predicted on the held-out test set (Section 5.3 visualization)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(preds.index, preds["actual"], label="Actual", alpha=0.75)
    ax.plot(preds.index, preds["predicted"], label="Predicted", alpha=0.95)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Daily {target} ($)")
    ax.set_title(f"{model_name} Forecast vs Actual on Held-Out Test Set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig

# ---------------------------------------------------------------------------
# End-to-end runner (loads team's preprocessing, runs full pipeline)
# ---------------------------------------------------------------------------

def run_experiment(target: str = "Profit",
                   window: int = 10,
                   epochs: int = 50,
                   architecture: str = "lstm",
                   verbose: int = 0,
                   seed: int = 42) -> Dict:
    if architecture not in {"lstm", "cnn"}:
        raise ValueError(f"architecture must be 'lstm' or 'cnn', got {architecture!r}")

    # Make sibling import work whether run from src/ or a notebook
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from data_preprocessing import load_data, data_cleaning  # noqa: E402

    df = load_data()
    df = data_cleaning(df)  # parses dates, drops dupes, fills NA

    daily = aggregate_daily(df, target=target)
    daily_train, daily_test = chronological_split(daily, train_frac=0.8)

    trainer = train_lstm if architecture == "lstm" else train_cnn
    model, history, scaler = trainer(
        daily_train,
        target=target,
        window=window,
        epochs=epochs,
        verbose=verbose,
        seed=seed,
    )

    metrics, preds = evaluate_on_test(
        model, scaler, daily_train, daily_test,
        target=target, window=window,
    )

    return {
        "model": model,
        "history": history,
        "scaler": scaler,
        "metrics": metrics,
        "predictions": preds,
        "daily_train": daily_train,
        "daily_test": daily_test,
        "config": {
            "architecture": architecture,
            "target": target, "window": window, "epochs": epochs, "seed": seed,
        },
    }

# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------

def run_multi_seed(architecture: str = "lstm",
                   target: str = "Profit",
                   window: int = 10,
                   epochs: int = 50,
                   seeds: Tuple[int, ...] = (42, 123, 7),
                   verbose: int = 0) -> Dict:
    per_run = []
    for s in seeds:
        if verbose:
            print(f"  [{architecture.upper()}] seed={s} ...")
        results = run_experiment(architecture=architecture, target=target,
                                 window=window, epochs=epochs,
                                 verbose=0, seed=s)
        per_run.append({"seed": s, **results["metrics"]})

    df = pd.DataFrame(per_run).set_index("seed")

    # Aggregate numeric metric columns (skip the count column)
    metric_cols = [c for c in df.columns if c != "n_test_points"]
    summary = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=1))}
        for col in metric_cols
    }

    # Build a "mean ± std" string column for direct report use
    formatted = {col: f"{summary[col]['mean']:.4f} ± {summary[col]['std']:.4f}"
                 for col in metric_cols}
    summary_df = df.copy()
    summary_df.loc["mean ± std"] = [formatted[c] if c in formatted
                                     else df[c].iloc[0] for c in df.columns]

    return {"per_run": per_run, "summary": summary, "summary_df": summary_df}


def run_multi_seed_comparison(target: str = "Profit",
                              window: int = 10,
                              epochs: int = 50,
                              architectures: Tuple[str, ...] = ("lstm", "cnn"),
                              seeds: Tuple[int, ...] = (42, 123, 7),
                              verbose: int = 0) -> Dict:
    raw = {}
    rows = []
    for arch in architectures:
        if verbose:
            print(f"\nMulti-seed run for {arch.upper()} (seeds={list(seeds)})")
        out = run_multi_seed(architecture=arch, target=target, window=window,
                             epochs=epochs, seeds=seeds, verbose=verbose)
        raw[arch] = out
        for run in out["per_run"]:
            rows.append({"architecture": arch.upper(), **run})

    per_run_df = pd.DataFrame(rows)

    # Build summary indexed by architecture: mean ± std for each metric
    metric_cols = [c for c in per_run_df.columns
                   if c not in {"architecture", "seed", "n_test_points"}]
    summary_rows = []
    for arch in architectures:
        sub = per_run_df[per_run_df["architecture"] == arch.upper()]
        row = {"architecture": arch.upper(),
               "n_seeds": len(sub),
               "n_test_points": int(sub["n_test_points"].iloc[0])}
        for col in metric_cols:
            mean = sub[col].mean()
            std = sub[col].std(ddof=1)
            row[col] = f"{mean:.4f} ± {std:.4f}"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("architecture")

    return {"per_run_df": per_run_df, "summary_df": summary_df, "raw": raw}

if __name__ == "__main__":
    print("=" * 70)
    print("Multi-seed head-to-head: LSTM vs 1D CNN on daily Profit (window=10)")
    print("Seeds: 42, 123, 7   |   Epochs: 50   |   Target: Profit")
    print("=" * 70)

    out = run_multi_seed_comparison(
        target="Profit",
        window=10,
        epochs=50,
        architectures=("lstm", "cnn"),
        seeds=(42, 123, 7),
        verbose=1,
    )

    print("\n" + "=" * 70)
    print("Per-run results (one row per architecture x seed)")
    print("=" * 70)
    print(out["per_run_df"].to_string(index=False))

    print("\n" + "=" * 70)
    print("Summary table (mean ± std across seeds)")
    print("=" * 70)
    print(out["summary_df"].to_string())
