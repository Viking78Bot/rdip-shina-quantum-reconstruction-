#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
MODULE_DIR = ROOT / "diffuser_head"
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from diffuser_head.train_diffuser_head import NoiseConfig
import torch
from projects.Fe2S2_stand.train_fe2s2 import Fe2S2Config, train_loop  # noqa: E402
from projects.Fe2S2_stand.infer_fe2s2 import run_inference  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fe2S2 cluster pipeline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda_energy", type=float, default=5.0)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/fe2s2_ruler_local.pt"))
    parser.add_argument("--logs_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--alphas", type=str, default="auto")
    parser.add_argument("--repeats", type=int, default=96)
    parser.add_argument("--repeats_infer", type=int, default=32)
    return parser.parse_args()


def ensure_paths(args: argparse.Namespace, base_dir: Path) -> tuple[Path, Path, Path]:
    data_dir = base_dir / "data"
    pauli_bank = data_dir / "Fe2S2_pauli_fe_pairs.npz"
    dm_bank = data_dir / "Fe2S2_active10.npz"
    schema = data_dir / "Fe2S2_pauli_fe_pairs.features.json"
    for path in (pauli_bank, dm_bank, schema):
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return pauli_bank, dm_bank, schema


def maybe_train(args: argparse.Namespace, pauli_bank: Path, dm_bank: Path, schema: Path) -> None:
    if args.skip_train and args.checkpoint.exists():
        print(f"[skip] Reusing checkpoint {args.checkpoint}")
        return

    diffuser_ckpt = ROOT / "diffuser_head_multibank_fe.pt"
    if not diffuser_ckpt.exists():
        diffuser_ckpt = Path(__file__).resolve().parent / "data" / "diffuser_head_multibank_fe.pt"
    cfg = Fe2S2Config(
        pauli_bank=pauli_bank,
        dm_bank=dm_bank,
        schema_json=schema,
        diffuser_ckpt=diffuser_ckpt,
        checkpoint=args.checkpoint,
        logs_dir=args.logs_dir / "train_logs",
        val_split=0.2,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        lambda_energy=args.lambda_energy,
        noise_cfg=NoiseConfig(
            depol_min=0.0,
            depol_max=0.05,
            damp_min=0.0,
            damp_max=0.05,
            gauss_sigma=0.01,
            pure_prob=0.6,
            p_identity=0.1,
        ),
        repeats=args.repeats,
    )
    train_loop(cfg)


def run_eval(args: argparse.Namespace, pauli_bank: Path, dm_bank: Path, schema: Path) -> Path:
    infer_logs = args.logs_dir / "inference"
    noise_cfg = NoiseConfig(
        depol_min=0.0,
        depol_max=0.05,
        damp_min=0.0,
        damp_max=0.05,
        gauss_sigma=0.01,
        pure_prob=0.6,
        p_identity=0.1,
    )
    alphas = [a.strip() for a in args.alphas.split(",") if a.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffuser_ckpt = ROOT / "diffuser_head_multibank_fe.pt"
    if not diffuser_ckpt.exists():
        diffuser_ckpt = Path(__file__).resolve().parent / "data" / "diffuser_head_multibank_fe.pt"
    run_inference(
        pauli_bank=pauli_bank,
        dm_bank=dm_bank,
        schema=schema,
        diffuser_ckpt=diffuser_ckpt,
        model_ckpt=args.checkpoint,
        logs_dir=infer_logs,
        alphas=alphas,
        noise_cfg=noise_cfg,
        repeats=args.repeats_infer,
        batch_size=args.batch,
        device=device,
    )
    first_alpha = alphas[0]
    return infer_logs / f"fe2s2_infer_alpha{first_alpha}.json"


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    pauli_bank, dm_bank, schema = ensure_paths(args, base_dir)
    maybe_train(args, pauli_bank, dm_bank, schema)
    summary_path = run_eval(args, pauli_bank, dm_bank, schema)

    summary = json.loads(summary_path.read_text())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

