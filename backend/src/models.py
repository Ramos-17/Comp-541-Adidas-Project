"""
Uma Desai COMP 541 - Adidas daily forecasting.

Pipeline:
  1. Aggregate transactions to daily transaction count (the busyness signal).
  2. Add calendar features (day-of-week, month, weekend) and optional lag/rolling features.
  3. Scale features and target with separate MinMax scalers fit on train only.
  4. Build sliding-window sequences and feed them to a 1D CNN or LSTM.
  5. Evaluate with MAE, RMSE, MAPE, SMAPE, R^2 and directional accuracy.
  6. Walk-forward cross-validation + grid-search hyperparameter tuning.
"""

from __future__ import annotations

import os
import sys
import itertools
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Suppress TensorFlow info logs (keep warnings/errors)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv1D, GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber

# Canonical target column inside engineered feature frame
TARGET_COL = "y"

CALENDAR_FEATURES: List[str] = [
    "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "day_of_month",
]

DEFAULT_LAGS: Tuple[int, ...] = (7,)
DEFAULT_ROLLING: Tuple[int, ...] = (7, 28)

# ---------------------------------------------------------------------------
# Daily aggregation - 'busyness' = number of transactions per day
# ---------------------------------------------------------------------------

def aggregate_daily(df: pd.DataFrame, target: str = "transactions") -> pd.DataFrame:
    """
    target='transactions' -> daily count of rows (true 'busyness').
    target=<col>          -> daily sum of that numeric column (e.g. 'Units_Sold', 'Profit').
    Returns DataFrame indexed by date with a single column named TARGET_COL ('y').
    """
    if "Order_Date" not in df.columns:
        raise ValueError("'Order_Date' column required for daily aggregation")

    s = df.copy()
    s["Order_Date"] = pd.to_datetime(s["Order_Date"])

    if target == "transactions":
        daily = s.groupby("Order_Date").size().to_frame(TARGET_COL).astype(float)
    elif target in s.columns:
        daily = s.groupby("Order_Date")[target].sum().to_frame(TARGET_COL).astype(float)
    else:
        raise ValueError(
            f"target must be 'transactions' or an existing column; got {target!r}"
        )

    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0.0)
    daily.index.name = "Order_Date"
    return daily


# ---------------------------------------------------------------------------
# Calendar + lag/rolling feature engineering
# ---------------------------------------------------------------------------

def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    idx = daily.index
    # Cyclical encoding so Monday-Sunday wrap-around is continuous to the model
    daily["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    daily["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    daily["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    daily["is_weekend"] = (idx.dayofweek >= 5).astype(float)
    daily["day_of_month"] = (idx.day - 1) / 30.0
    return daily


def add_lag_features(daily: pd.DataFrame,
                     target_col: str = TARGET_COL,
                     lags: Sequence[int] = DEFAULT_LAGS,
                     rolling_windows: Sequence[int] = DEFAULT_ROLLING) -> pd.DataFrame:
    daily = daily.copy()
    for lag in lags:
        daily[f"lag_{lag}"] = daily[target_col].shift(lag)
    for w in rolling_windows:
        # shift(1) ensures the rolling window uses only past values (no leakage)
        daily[f"roll{w}_mean"] = daily[target_col].shift(1).rolling(w).mean()
    return daily


def build_feature_frame(df: pd.DataFrame,
                        target: str = "transactions",
                        use_calendar: bool = True,
                        lags: Sequence[int] = DEFAULT_LAGS,
                        rolling_windows: Sequence[int] = DEFAULT_ROLLING) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns (engineered_frame, feature_cols). The target column is always TARGET_COL ('y')
    and is included in feature_cols so the sequence model gets the past target values too.
    """
    daily = aggregate_daily(df, target=target)
    if use_calendar:
        daily = add_calendar_features(daily)
    if lags or rolling_windows:
        daily = add_lag_features(daily, target_col=TARGET_COL,
                                 lags=lags, rolling_windows=rolling_windows)
    daily = daily.dropna()

    feature_cols = [TARGET_COL]
    if use_calendar:
        feature_cols += CALENDAR_FEATURES
    feature_cols += [f"lag_{l}" for l in lags]
    feature_cols += [f"roll{w}_mean" for w in rolling_windows]
    return daily, feature_cols


# ---------------------------------------------------------------------------
# Sequence construction - multivariate
# ---------------------------------------------------------------------------

def make_sequences_mv(X_scaled: np.ndarray,
                      y_scaled: np.ndarray,
                      window: int) -> Tuple[np.ndarray, np.ndarray]:
    T = len(y_scaled)
    if T <= window:
        raise ValueError(f"length {T} <= window {window}")
    X_seq = np.stack([X_scaled[i:i + window] for i in range(T - window)], axis=0)
    y_seq = y_scaled[window:]
    return X_seq.astype(np.float32), y_seq.astype(np.float32)


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def chronological_split(daily: pd.DataFrame,
                        train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(daily)
    cut = int(n * train_frac)
    return daily.iloc[:cut].copy(), daily.iloc[cut:].copy()


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

def _resolve_loss(loss: str):
    return Huber(delta=1.0) if loss == "huber" else loss


def build_lstm(window: int = 14,
               n_features: int = 7,
               hidden_units: int = 64,
               dropout: float = 0.20,
               l2_coeff: float = 0.001,
               lr: float = 1e-3,
               loss: str = "huber") -> tf.keras.Model:
    model = Sequential([
        Input(shape=(window, n_features)),
        LSTM(hidden_units, activation="tanh",
             kernel_regularizer=l2(l2_coeff),
             recurrent_regularizer=l2(l2_coeff)),
        Dropout(dropout),
        Dense(1, kernel_regularizer=l2(l2_coeff)),
    ])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=_resolve_loss(loss),
                  metrics=["mae"])
    return model


def build_cnn(window: int = 14,
              n_features: int = 7,
              filters: Tuple[int, int] = (32, 64),
              kernel_size: int = 3,
              dropout: float = 0.20,
              l2_coeff: float = 0.001,
              lr: float = 1e-3,
              loss: str = "huber") -> tf.keras.Model:
    f1, f2 = filters
    model = Sequential([
        Input(shape=(window, n_features)),
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
                  loss=_resolve_loss(loss),
                  metrics=["mae"])
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _scale_and_sequence(daily_train: pd.DataFrame,
                        target_col: str,
                        feature_cols: List[str],
                        window: int):
    X_raw = daily_train[feature_cols].values
    y_raw = daily_train[target_col].values.reshape(-1, 1)
    x_scaler = MinMaxScaler().fit(X_raw)
    y_scaler = MinMaxScaler().fit(y_raw)
    X_scaled = x_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(y_raw).reshape(-1)
    X_seq, y_seq = make_sequences_mv(X_scaled, y_scaled, window)
    return X_seq, y_seq, x_scaler, y_scaler


def _fit_seq_model(model: tf.keras.Model,
                   X_seq: np.ndarray, y_seq: np.ndarray,
                   epochs: int, batch_size: int, val_frac: float,
                   patience: int, seed: int, verbose: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    n = len(X_seq)
    cut = int(n * (1 - val_frac))
    X_tr, y_tr = X_seq[:cut], y_seq[:cut]
    X_val, y_val = X_seq[cut:], y_seq[cut:]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True,
                      verbose=1 if verbose else 0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=max(patience // 2, 3),
                          min_lr=1e-5,
                          verbose=1 if verbose else 0),
    ]
    history = model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, shuffle=False, verbose=verbose,
    )
    return model, history


def train_lstm(daily_train: pd.DataFrame,
               target_col: str = TARGET_COL,
               feature_cols: Optional[List[str]] = None,
               window: int = 14,
               hidden_units: int = 64,
               dropout: float = 0.20,
               l2_coeff: float = 0.001,
               lr: float = 1e-3,
               loss: str = "huber",
               epochs: int = 100,
               batch_size: int = 32,
               val_frac: float = 0.20,
               early_stop_patience: int = 15,
               seed: int = 42,
               verbose: int = 1):
    if feature_cols is None:
        feature_cols = [target_col] + CALENDAR_FEATURES
    X_seq, y_seq, x_scaler, y_scaler = _scale_and_sequence(
        daily_train, target_col, feature_cols, window)
    model = build_lstm(window=window, n_features=len(feature_cols),
                       hidden_units=hidden_units, dropout=dropout,
                       l2_coeff=l2_coeff, lr=lr, loss=loss)
    model, history = _fit_seq_model(
        model, X_seq, y_seq, epochs, batch_size, val_frac,
        early_stop_patience, seed, verbose)
    return model, history, x_scaler, y_scaler, feature_cols


def train_cnn(daily_train: pd.DataFrame,
              target_col: str = TARGET_COL,
              feature_cols: Optional[List[str]] = None,
              window: int = 14,
              filters: Tuple[int, int] = (32, 64),
              kernel_size: int = 3,
              dropout: float = 0.20,
              l2_coeff: float = 0.001,
              lr: float = 1e-3,
              loss: str = "huber",
              epochs: int = 100,
              batch_size: int = 32,
              val_frac: float = 0.20,
              early_stop_patience: int = 15,
              seed: int = 42,
              verbose: int = 1):
    if feature_cols is None:
        feature_cols = [target_col] + CALENDAR_FEATURES
    X_seq, y_seq, x_scaler, y_scaler = _scale_and_sequence(
        daily_train, target_col, feature_cols, window)
    model = build_cnn(window=window, n_features=len(feature_cols),
                      filters=filters, kernel_size=kernel_size,
                      dropout=dropout, l2_coeff=l2_coeff, lr=lr, loss=loss)
    model, history = _fit_seq_model(
        model, X_seq, y_seq, epochs, batch_size, val_frac,
        early_stop_patience, seed, verbose)
    return model, history, x_scaler, y_scaler, feature_cols


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% of test days where the predicted direction (up/down vs prior day) matches actual."""
    if len(y_true) < 2:
        return float("nan")
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    valid = actual_dir != 0
    if not valid.any():
        return float("nan")
    return float(np.mean(actual_dir[valid] == pred_dir[valid]) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE - stable when actuals are near zero (unlike plain MAPE)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def evaluate_on_test(model: tf.keras.Model,
                     x_scaler: MinMaxScaler,
                     y_scaler: MinMaxScaler,
                     feature_cols: List[str],
                     daily_train: pd.DataFrame,
                     daily_test: pd.DataFrame,
                     target_col: str = TARGET_COL,
                     window: int = 14) -> Tuple[Dict[str, float], pd.DataFrame]:
    full = pd.concat([daily_train, daily_test])
    X_full = x_scaler.transform(full[feature_cols].values)
    y_full = y_scaler.transform(full[[target_col]].values).reshape(-1)
    X_seq, y_seq = make_sequences_mv(X_full, y_full, window)

    # First test prediction sits at index (len(train) - window) inside the sequence array
    first_test_idx = len(daily_train) - window
    X_test = X_seq[first_test_idx:]
    y_test_scaled = y_seq[first_test_idx:]
    test_dates = full.index[len(daily_train):]

    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_true = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    nonzero = y_true > 0
    mape = (float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
            if nonzero.any() else float("nan"))

    test_mean = float(np.mean(y_true)) if len(y_true) else 0.0
    mae_pct_of_mean = float(mae / test_mean * 100) if test_mean > 0 else float("nan")

    preds = pd.DataFrame(
        {"actual": y_true, "predicted": y_pred},
        index=test_dates,
    )
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape_pct": mape,
        "smape_pct": smape(y_true, y_pred),
        "directional_acc_pct": directional_accuracy(y_true, y_pred),
        "mae_pct_of_mean": mae_pct_of_mean,
        "r2": r2,
        "n_test_points": int(len(y_true)),
    }
    return metrics, preds


# ---------------------------------------------------------------------------
# Baselines (for sanity-checking the deep models)
# ---------------------------------------------------------------------------

def baseline_metrics(daily_train: pd.DataFrame,
                     daily_test: pd.DataFrame,
                     target_col: str = TARGET_COL) -> pd.DataFrame:
    """Three trivial forecasters - any deep model must beat these to be worth it."""
    full = pd.concat([daily_train, daily_test])[target_col].values
    n_train = len(daily_train)
    y_true = full[n_train:]

    rows = []
    # Yesterday's value
    y_pred_yest = full[n_train - 1:-1]
    # Same day last week
    y_pred_seas = full[n_train - 7:-7]
    # Mean of training period
    y_pred_mean = np.full_like(y_true, fill_value=full[:n_train].mean())

    for name, y_pred in [("yesterday", y_pred_yest),
                          ("same_day_last_week", y_pred_seas),
                          ("train_mean", y_pred_mean)]:
        nonzero = y_true > 0
        mape = (float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
                if nonzero.any() else float("nan"))
        rows.append({
            "baseline": name,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mape_pct": mape,
            "smape_pct": smape(y_true, y_pred),
            "directional_acc_pct": directional_accuracy(y_true, y_pred),
        })
    return pd.DataFrame(rows).set_index("baseline")


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(daily: pd.DataFrame,
                    feature_cols: List[str],
                    architecture: str = "lstm",
                    n_splits: int = 5,
                    target_col: str = TARGET_COL,
                    window: int = 14,
                    seed: int = 42,
                    verbose: int = 0,
                    **train_kwargs) -> pd.DataFrame:
    """
    Expanding-window CV: at each fold the train set grows and the test set is the next
    block of days. Far more reliable than a single 80/20 split for time series.
    """
    if architecture not in {"lstm", "cnn"}:
        raise ValueError(f"architecture must be 'lstm' or 'cnn', got {architecture!r}")
    train_fn = train_lstm if architecture == "lstm" else train_cnn

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(daily), start=1):
        daily_train = daily.iloc[train_idx]
        daily_test = daily.iloc[test_idx]
        if verbose:
            print(f"  fold {fold_idx}/{n_splits}: train={len(daily_train)} "
                  f"-> test={len(daily_test)}")
        model, _, x_scaler, y_scaler, fc = train_fn(
            daily_train, target_col=target_col, feature_cols=feature_cols,
            window=window, seed=seed, verbose=0, **train_kwargs)
        m, _ = evaluate_on_test(model, x_scaler, y_scaler, fc,
                                daily_train, daily_test,
                                target_col=target_col, window=window)
        rows.append({"fold": fold_idx, **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hyperparameter tuning (grid search + walk-forward CV)
# ---------------------------------------------------------------------------

def tune_hyperparameters(daily: pd.DataFrame,
                         feature_cols: List[str],
                         architecture: str = "lstm",
                         grid: Optional[Dict[str, Sequence]] = None,
                         n_splits: int = 3,
                         target_col: str = TARGET_COL,
                         epochs: int = 80,
                         seed: int = 42,
                         verbose: int = 1) -> pd.DataFrame:
    """
    Grid search over hyperparameters with walk-forward CV. Returns a DataFrame sorted
    by mean SMAPE (lower = better). Selects on SMAPE rather than MAPE because MAPE
    is unstable when test set contains very low or zero-traffic days.
    """
    if grid is None:
        if architecture == "lstm":
            grid = {
                "window": [14, 28],
                "lr": [5e-4, 1e-3],
                "hidden_units": [32, 64],
            }
        else:
            grid = {
                "window": [14, 28],
                "lr": [5e-4, 1e-3],
                "filters": [(16, 32), (32, 64)],
            }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if verbose:
        print(f"Tuning {architecture.upper()} over {len(combos)} configs "
              f"with {n_splits}-fold walk-forward CV")

    rows = []
    for combo in combos:
        cfg = dict(zip(keys, combo))
        if verbose:
            print(f"\n>> config: {cfg}")
        window = cfg.pop("window")
        try:
            fold_df = walk_forward_cv(
                daily, feature_cols=feature_cols,
                architecture=architecture, n_splits=n_splits,
                target_col=target_col, window=window,
                seed=seed, verbose=verbose, epochs=epochs, **cfg,
            )
        except Exception as exc:
            if verbose:
                print(f"   config failed: {exc}")
            continue
        rows.append({
            "window": window,
            **cfg,
            "smape_mean": fold_df["smape_pct"].mean(),
            "smape_std": fold_df["smape_pct"].std(ddof=1),
            "mape_mean": fold_df["mape_pct"].mean(),
            "mae_mean": fold_df["mae"].mean(),
            "dir_acc_mean": fold_df["directional_acc_pct"].mean(),
        })
    return pd.DataFrame(rows).sort_values("smape_mean").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loss_curves(history, save_path: Optional[str] = None,
                     model_name: str = "LSTM"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} Loss over Epochs")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_predictions(preds: pd.DataFrame,
                     ylabel: str = "Daily Transactions",
                     save_path: Optional[str] = None,
                     model_name: str = "LSTM"):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(preds.index, preds["actual"], label="Actual", alpha=0.75)
    ax.plot(preds.index, preds["predicted"], label="Predicted", alpha=0.95)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model_name} - Forecast vs Actual on Held-Out Test Set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# End-to-end experiment runners
# ---------------------------------------------------------------------------

def _load_clean_df() -> pd.DataFrame:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from data_preprocessing import load_data, data_cleaning  # noqa: E402
    df = load_data()
    return data_cleaning(df)


def run_experiment(target: str = "transactions",
                   architecture: str = "lstm",
                   window: int = 14,
                   epochs: int = 100,
                   use_calendar: bool = True,
                   lags: Sequence[int] = DEFAULT_LAGS,
                   rolling_windows: Sequence[int] = DEFAULT_ROLLING,
                   train_frac: float = 0.8,
                   verbose: int = 0,
                   seed: int = 42,
                   **hparams) -> Dict:
    if architecture not in {"lstm", "cnn"}:
        raise ValueError(f"architecture must be 'lstm' or 'cnn', got {architecture!r}")

    df = _load_clean_df()
    daily, feature_cols = build_feature_frame(
        df, target=target, use_calendar=use_calendar,
        lags=lags, rolling_windows=rolling_windows,
    )
    daily_train, daily_test = chronological_split(daily, train_frac=train_frac)

    train_fn = train_lstm if architecture == "lstm" else train_cnn
    model, history, x_scaler, y_scaler, fc = train_fn(
        daily_train, feature_cols=feature_cols, window=window,
        epochs=epochs, verbose=verbose, seed=seed, **hparams,
    )
    metrics, preds = evaluate_on_test(
        model, x_scaler, y_scaler, fc,
        daily_train, daily_test, window=window,
    )
    return {
        "model": model, "history": history,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
        "feature_cols": fc,
        "metrics": metrics, "predictions": preds,
        "daily": daily, "daily_train": daily_train, "daily_test": daily_test,
        "baselines": baseline_metrics(daily_train, daily_test),
        "config": {"architecture": architecture, "target": target,
                   "window": window, "epochs": epochs, "seed": seed},
    }


def run_multi_seed(architecture: str = "lstm",
                   target: str = "transactions",
                   window: int = 14,
                   epochs: int = 100,
                   seeds: Tuple[int, ...] = (42, 123, 7),
                   verbose: int = 0,
                   **hparams) -> Dict:
    rows = []
    for s in seeds:
        if verbose:
            print(f"  [{architecture.upper()}] seed={s} ...")
        out = run_experiment(architecture=architecture, target=target,
                             window=window, epochs=epochs,
                             verbose=0, seed=s, **hparams)
        rows.append({"seed": s, **out["metrics"]})
    df = pd.DataFrame(rows).set_index("seed")

    metric_cols = [c for c in df.columns if c != "n_test_points"]
    summary = {col: {"mean": float(df[col].mean()),
                     "std": float(df[col].std(ddof=1))} for col in metric_cols}
    return {"per_run": rows, "summary": summary, "summary_df": df}


def run_multi_seed_comparison(target: str = "transactions",
                              window: int = 14,
                              epochs: int = 100,
                              architectures: Tuple[str, ...] = ("lstm", "cnn"),
                              seeds: Tuple[int, ...] = (42, 123, 7),
                              verbose: int = 0,
                              **hparams) -> Dict:
    raw, rows = {}, []
    for arch in architectures:
        if verbose:
            print(f"\nMulti-seed run for {arch.upper()} (seeds={list(seeds)})")
        out = run_multi_seed(architecture=arch, target=target, window=window,
                             epochs=epochs, seeds=seeds, verbose=verbose, **hparams)
        raw[arch] = out
        for run in out["per_run"]:
            rows.append({"architecture": arch.upper(), **run})

    per_run_df = pd.DataFrame(rows)
    metric_cols = [c for c in per_run_df.columns
                   if c not in {"architecture", "seed", "n_test_points"}]
    summary_rows = []
    for arch in architectures:
        sub = per_run_df[per_run_df["architecture"] == arch.upper()]
        row = {"architecture": arch.upper(),
               "n_seeds": len(sub),
               "n_test_points": int(sub["n_test_points"].iloc[0])}
        for col in metric_cols:
            mean, std = sub[col].mean(), sub[col].std(ddof=1)
            row[col] = f"{mean:.4f} ± {std:.4f}"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).set_index("architecture")
    return {"per_run_df": per_run_df, "summary_df": summary_df, "raw": raw}


if __name__ == "__main__":
    print("=" * 72)
    print("Multi-seed head-to-head: LSTM vs 1D CNN on daily TRANSACTIONS")
    print("Calendar features + Huber loss + lag/rolling features | window=14")
    print("=" * 72)

    out = run_multi_seed_comparison(
        target="transactions",
        window=14,
        epochs=80,
        architectures=("lstm", "cnn"),
        seeds=(42, 123, 7),
        verbose=1,
    )
    print("\n" + "=" * 72)
    print("Per-run results")
    print("=" * 72)
    print(out["per_run_df"].to_string(index=False))
    print("\n" + "=" * 72)
    print("Summary (mean ± std across seeds)")
    print("=" * 72)
    print(out["summary_df"].to_string())
