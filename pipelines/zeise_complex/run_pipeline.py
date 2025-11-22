#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_dataset(base_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = base_dir / "data" / "zeise_synthetic_observations.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    data = np.load(data_path)
    return data["features"], data["targets"], data["coeff"]


def fit_linear_model(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    X = np.column_stack([features, np.ones(features.shape[0])])
    theta, *_ = np.linalg.lstsq(X, targets, rcond=None)
    return theta


def evaluate(theta: np.ndarray, features: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    X = np.column_stack([features, np.ones(features.shape[0])])
    preds = X @ theta
    resid = preds - targets
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {"mae": mae, "rmse": rmse, "predictions": preds.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Zeise-like regression pipeline")
    parser.add_argument("--report", type=Path, default=Path("zeise_report.json"))
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    features, targets, true_coeff = load_dataset(base_dir)

    t0 = time.time()
    theta = fit_linear_model(features, targets)
    metrics = evaluate(theta, features, targets)
    metrics["elapsed_s"] = time.time() - t0
    metrics["samples"] = int(features.shape[0])
    metrics["n_features"] = int(features.shape[1])
    metrics["estimated_coeff"] = theta[:-1].tolist()
    metrics["estimated_bias"] = float(theta[-1])
    metrics["true_coeff"] = true_coeff.tolist()

    args.report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

