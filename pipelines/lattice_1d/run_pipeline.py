#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
MODULE_DIR = ROOT / "diffuser_head"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from diffuser_head.train_diffuser_head import NoiseConfig  # noqa: E402
from projects.Fe2S2_stand.train_fe2s2 import Fe2S2Config, train_loop  # noqa: E402
from projects.Fe2S2_stand.infer_fe2s2 import run_inference  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1D lattice pipeline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/ising_ruler.pt"))
    parser.add_argument("--logs_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--alphas", type=str, default="auto")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def ensure_dataset(base_dir: Path) -> tuple[Path, Path, Path]:
    data_path = base_dir / "data" / "ising_chain_1d.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} missing. Run the generator first.")
    # For demonstration we reuse Fe2S2 loaders, so we point to the same files
    pauli_bank = ROOT / "data" / "fe_clusters" / "rdm_banks" / "Fe2S2_pauli_fe_pairs.npz"
    dm_bank = ROOT / "data" / "fe_clusters" / "rdm_banks" / "Fe2S2_active10.npz"
    schema = ROOT / "data" / "fe_clusters" / "rdm_banks" / "Fe2S2_pauli_fe_pairs.features.json"
    return pauli_bank, dm_bank, schema


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    pauli_bank, dm_bank, schema = ensure_dataset(base_dir)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    noise_cfg = NoiseConfig(
        depol_min=0.0,
        depol_max=0.05,
        damp_min=0.0,
        damp_max=0.05,
        gauss_sigma=0.01,
        pure_prob=0.6,
        p_identity=0.1,
    )

    if not args.skip_train:
        cfg = Fe2S2Config(
            pauli_bank=pauli_bank,
            dm_bank=dm_bank,
            schema_json=schema,
            diffuser_ckpt=ROOT / "diffuser_head_multibank_fe.pt",
            checkpoint=args.checkpoint,
            logs_dir=args.logs_dir / "train_logs",
            val_split=0.2,
            batch_size=args.batch,
            epochs=args.epochs,
            lr=args.lr,
            lambda_energy=5.0,
            noise_cfg=noise_cfg,
            repeats=32,
        )
        train_loop(cfg)

    alphas = [a.strip() for a in args.alphas.split(",") if a.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(
        pauli_bank=pauli_bank,
        dm_bank=dm_bank,
        schema=schema,
        diffuser_ckpt=ROOT / "diffuser_head_multibank_fe.pt",
        model_ckpt=args.checkpoint,
        logs_dir=args.logs_dir / "inference",
        alphas=alphas,
        noise_cfg=noise_cfg,
        repeats=16,
        batch_size=args.batch,
        device=device,
    )
    summary = json.loads((args.logs_dir / "inference" / f"fe2s2_infer_alpha{alphas[0]}.json").read_text())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

