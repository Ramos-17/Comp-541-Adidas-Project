"""
Uma Desai COMP 541
LSTM time-series forecasting module for Adidas sales project.

Using the team's plan from the Intermediate Progress Report:
    Architecture Design
    Training Setup
    Regularization & Generalization

Architecture (per Section 4.1):
    LSTM(50, activation='relu')  ->  Dropout(0.20)  ->  Dense(1)
    Optimizer: Adam (lr=0.001), Loss: MSE
    Batch size: 16, Epochs: up to 50

Regularizatio:
    Dropout 0.20 on LSTM output
    kernel_regularizer on LSTM and Dense layers
    EarlyStopping on val_loss (patience=10, restore_best_weights=True)

Training setup
    Chronological 80/20 split (no shuffling)
    A forward-chaining inner-validation split is used for EarlyStopping monitoring
    so the held-out test set is never touched during training.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress TensorFlow logs (keep warnings/errors)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def aggregate_daily(df: pd.DataFrame, target: str = "Profit") -> pd.DataFrame:
    """
    Aggregate transaction-level rows into a complete daily time series. Days with no transactions in source data are inserted with a value of 0
    so the resulting series has no gaps (is required
    for the LSTM's fixed-window input)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data containing 'Order_Date' and the target column.
    target : str
        Column to aggregate. Defaults to 'Profit' per Section 5.1
        (Revenue and Profit are 0.97 correlated; the team chose Profit).

    Returns
    -------
    pd.DataFrame indexed by date with a single column [target].
    """

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
    """
    Build supervised (X, y) pairs from a 1D series via a sliding window.

    For each i, X[i] = series[i : i+window] and y[i] = series[i+window], so the
    model learns to predict day i+window from the previous `window` days.

    Returns
    -------
    X : np.ndarray of shape (n_samples, window, 1)
    y : np.ndarray of shape (n_samples,)
    """
    series = np.asarray(series).reshape(-1)
    if len(series) <= window:
        raise ValueError(f"series length {len(series)} <= window {window}")

    X = np.lib.stride_tricks.sliding_window_view(series, window)[:-1]
    y = series[window:]
    X = X.reshape(-1, window, 1).astype(np.float32)
    y = y.astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Chronological split (Section 4.2)
# ---------------------------------------------------------------------------

def chronological_split(daily: pd.DataFrame,
                        train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered 80/20 split. No shuffling - prevents future leakage.
    The test portion is held out and not touched until final evaluation.
    """
    n = len(daily)
    cut = int(n * train_frac)
    return daily.iloc[:cut].copy(), daily.iloc[cut:].copy()


# ---------------------------------------------------------------------------
# Architecture (Section 4.1) + Regularization (Section 4.3)
# ---------------------------------------------------------------------------

def build_lstm(window: int = 10,
               hidden_units: int = 50,
               dropout: float = 0.20,
               l2_coeff: float = 0.01,
               lr: float = 0.001) -> tf.keras.Model:
    """
    Build the LSTM model.

    Layers
    ------
    1. Input  : (window, 1)   - `window` days of the daily target series
    2. LSTM   : `hidden_units` units, ReLU activation, L2 kernel reg.
    3. Dropout: `dropout` fraction (Section 4.3)
    4. Dense  : 1 unit  - the predicted value for the next day

    Compile
    -------
    Optimizer : Adam(lr)
    Loss      : MSE  (Section 4.2)
    Metrics   : MAE  (reported alongside loss for interpretability)

    Defaults match Section 4.1 / 4.3 exactly. Window can be overridden to run
    the 7 / 14 / 30 experiments mentioned in Section 5.1.
    """
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
    """
    Build a 1D Convolutional Network for daily forecasting.

    Layers
    ------
    1. Input                : (window, 1)
    2. Conv1D(filters[0])   : kernel_size, ReLU, causal padding, L2 kernel reg.
    3. Dropout(dropout)
    4. Conv1D(filters[1])   : kernel_size, ReLU, causal padding, L2 kernel reg.
    5. GlobalAveragePooling1D
    6. Dropout(dropout)
    7. Dense(1)             : L2 kernel reg.

    Compile
    -------
    Optimizer : Adam(lr)
    Loss      : MSE  (Section 4.2)
    Metrics   : MAE  (reported alongside loss for interpretability)
    """
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
# Training (Section 4.2 + 4.3) - shared between LSTM and CNN
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
    """
    Shared training routine for any sequence-based univariate forecaster
    (LSTM, CNN, GRU, etc.). Implements Section 4.2 and the EarlyStopping piece
    of Section 4.3 in one place so different architectures fit identically.

    Pipeline
    --------
    1. MinMax-scale the target using ONLY training stats (no test leakage).
    2. Build sliding-window (X, y) supervised pairs.
    3. Forward-chain inner split: last `val_frac` of training sequences become
       a validation set monitored by EarlyStopping. The held-out test set
       (passed separately to evaluate_on_test) is NOT touched here.
    4. Fit with shuffle=False to preserve temporal order within each epoch.
    """
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
    """
    Fit the LSTM on the training portion of the daily series.
    Thin wrapper around `_fit_sequence_model` with LSTM-specific defaults.

    Returns
    -------
    model    : trained tf.keras.Model
    history  : Keras History (loss curves)
    scaler   : fitted MinMaxScaler (needed to invert predictions on test)
    """
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
    """
    Fit the 1D CNN on the training portion of the daily series.
    Mirrors `train_lstm`'s training setup so the two models are directly
    comparable - identical scaling, splits, optimizer, batch size, epochs,
    EarlyStopping, and L2/Dropout hyperparameters.

    Returns
    -------
    model    : trained tf.keras.Model
    history  : Keras History (loss curves)
    scaler   : fitted MinMaxScaler (needed to invert predictions on test)
    """
    model = build_cnn(window=window, filters=filters, kernel_size=kernel_size,
                      dropout=dropout, l2_coeff=l2_coeff, lr=lr)
    return _fit_sequence_model(
        model, daily_train, target, window, epochs, batch_size,
        val_frac, early_stop_patience, seed, verbose,
    )


# ---------------------------------------------------------------------------
# Evaluation (Section 5.2)
# ---------------------------------------------------------------------------

def evaluate_on_test(model: tf.keras.Model,
                     scaler: MinMaxScaler,
                     daily_train: pd.DataFrame,
                     daily_test: pd.DataFrame,
                     target: str = "Profit",
                     window: int = 10) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Metrics
    -------
    RMSE (dollars), MAE (dollars), MAPE (%) per Section 5.2.
    MAPE is computed only over days where actual > 0 to avoid divide-by-zero.

    Returns
    -------
    metrics : dict
    preds   : DataFrame with columns ['actual', 'predicted'] indexed by date
    """
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
        "n_test_points": int(len(y_true)),
    }
    return metrics, preds


# ---------------------------------------------------------------------------
# Plots (used by the demo notebook + Section
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
    """Actual vs predicted on the held-out test set (visualization)."""
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
# Running End-to-end (loads team's preprocessing, runs full pipeline)
# ---------------------------------------------------------------------------

def run_experiment(target: str = "Profit",
                   window: int = 10,
                   epochs: int = 50,
                   architecture: str = "lstm",
                   verbose: int = 0,
                   seed: int = 42) -> Dict:
    """
    Run full pipeline end-to-end using the team's data_preprocessing module.

    Parameters
    ----------
    architecture : {"lstm", "cnn"}
        Decide which architecture to run for the experiment. Both are implemented with same training.
    """
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


if __name__ == "__main__":
    print("=" * 60)
    print("Head-to-head: LSTM vs 1D CNN on daily Profit (window=10)")
    print("=" * 60)

    summary = []
    for arch in ["lstm", "cnn"]:
        print(f"\n--- Training {arch.upper()} ---")
        results = run_experiment(architecture=arch, target="Profit",
                                 window=10, epochs=50, verbose=0)
        m = results["metrics"]
        summary.append({"architecture": arch.upper(), **m})
        print(f"  RMSE: ${m['rmse_dollars']:.2f}  "
              f"MAE: ${m['mae_dollars']:.2f}  "
              f"MAPE: {m['mape_pct']:.1f}%")

    print("\n" + "=" * 60)
    print("Summary table")
    print("=" * 60)
    print(pd.DataFrame(summary).to_string(index=False))
