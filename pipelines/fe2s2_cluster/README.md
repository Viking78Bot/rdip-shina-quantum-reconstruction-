## Fe₂S₂ Cluster Pipeline

`data/` ships with everything needed for reproduction:

- `Fe2S2_pauli_fe_pairs.npz`, `Fe2S2_active10.npz`, feature schema.
- `diffuser_head_multibank_fe.pt` — DiffuserHead weights.
- `fe2s2_ruler_patch32.pt` — best GraphRulerNet checkpoint (val |ΔE| ≈ 0.61 Ha).

By default the pipeline performs a short fine-tuning run and then executes the
energy-aware inference script.

### Usage

```bash
cd pipelines/fe2s2_cluster
python run_pipeline.py --epochs 5 --checkpoint outputs/fe2s2_local.pt

# run inference only, reusing shipped checkpoint
python run_pipeline.py --skip-train --checkpoint data/fe2s2_ruler_patch32.pt
```

Key arguments:

- `--epochs`: number of fine-tuning epochs (default: 5 for quick demo).
- `--skip-train`: reuse an existing checkpoint instead of fine-tuning.
- `--alphas`: inference alphas list (default: `auto`).

Outputs (training logs, checkpoints, inference metrics) are stored under
`outputs/`. Runtime on the reference desktop: ~5 s for training 5 epochs +
~20 s for inference with energy metrics.

### Requirements

Install the following inside a virtualenv:

```
pip install numpy scipy tqdm torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

(Alternatively, use the `.../whl/cpu` index for CPU-only installations.)

