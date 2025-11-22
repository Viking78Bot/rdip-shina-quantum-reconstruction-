#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def load_dataset(base_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = base_dir / "data" / "linear_chain_samples.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    data = np.load(data_path)
    return data["spins"], data["energies"], data["couplings"]


def solve_couplings(spins: np.ndarray, energies: np.ndarray) -> np.ndarray:
    features = spins[:, :-1] * spins[:, 1:]
    coeffs, *_ = np.linalg.lstsq(features, energies, rcond=None)
    return coeffs


def build_energy_predictions(spins: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    interactions = spins[:, :-1] * spins[:, 1:]
    return interactions @ coeffs


def summarize(true_couplings: np.ndarray, est_couplings: np.ndarray, true_energies: np.ndarray, pred_energies: np.ndarray) -> dict[str, float]:
    mae_energy = float(np.mean(np.abs(pred_energies - true_energies)))
    rmse_energy = float(np.sqrt(np.mean((pred_energies - true_energies) ** 2)))
    mae_couplings = float(np.mean(np.abs(est_couplings - true_couplings)))
    rmse_couplings = float(np.sqrt(np.mean((est_couplings - true_couplings) ** 2)))
    return {
        "mae_energy": mae_energy,
        "rmse_energy": rmse_energy,
        "mae_couplings": mae_couplings,
        "rmse_couplings": rmse_couplings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear Ising chain pipeline")
    parser.add_argument("--report", type=Path, default=Path("linear_report.json"), help="Path to save summary JSON")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    spins, energies, couplings = load_dataset(base_dir)

    t0 = time.time()
    est_couplings = solve_couplings(spins, energies)
    pred_energies = build_energy_predictions(spins, est_couplings)
    metrics = summarize(couplings, est_couplings, energies, pred_energies)
    metrics["samples"] = spins.shape[0]
    metrics["chain_length"] = spins.shape[1]
    metrics["elapsed_s"] = time.time() - t0

    args.report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

