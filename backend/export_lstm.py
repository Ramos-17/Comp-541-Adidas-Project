from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblib import dump

from src.UD_models import train_full_lstm


def export_lstm_artifacts(output_dir: Path,
                          target: str = "Profit",
                          window: int = 10,
                          epochs: int = 50,
                          seed: int = 42,
                          verbose: int = 0) -> dict[str, Path]:
    """
    Train the chosen LSTM architecture on the full cleaned series and export:
    - `lstm_model.keras`
    - `scaler.joblib`
    - `meta.joblib`
    - `metrics.json` (export metadata only)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    result = train_full_lstm(
        target=target,
        window=window,
        epochs=epochs,
        seed=seed,
        verbose=verbose,
    )

    model_path = output_dir / "lstm_model.keras"
    scaler_path = output_dir / "scaler.joblib"
    meta_path = output_dir / "meta.joblib"
    metrics_path = output_dir / "metrics.json"

    result["model"].save(model_path)
    dump(result["scaler"], scaler_path)

    meta = {
        **result["config"],
        "feature_name": target,
        "series_start": str(result["daily"].index.min().date()),
        "series_end": str(result["daily"].index.max().date()),
        "n_days": int(len(result["daily"])),
    }
    dump(meta, meta_path)

    metrics_payload = {
        "export_type": "full-series-training",
        "model": "LSTM",
        "target": target,
        "window": window,
        "epochs": epochs,
        "seed": seed,
        "series_start": meta["series_start"],
        "series_end": meta["series_end"],
        "n_days": meta["n_days"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    return {
        "model": model_path,
        "scaler": scaler_path,
        "meta": meta_path,
        "metrics": metrics_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the Adidas LSTM artifacts.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory where artifacts will be written.")
    parser.add_argument("--target", default="Profit", help="Target column to forecast.")
    parser.add_argument("--window", type=int, default=10, help="Sliding window size.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", type=int, default=1, help="Keras fit verbosity.")
    args = parser.parse_args()

    output_paths = export_lstm_artifacts(
        output_dir=Path(args.output_dir),
        target=args.target,
        window=args.window,
        epochs=args.epochs,
        seed=args.seed,
        verbose=args.verbose,
    )

    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
