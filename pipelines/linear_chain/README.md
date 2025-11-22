## Linear Chain Pipeline

Synthetic ±1 spin configurations (length 10) are stored in
`data/linear_chain_samples.npz`. Each sample comes with the ground-truth
nearest-neighbour couplings `J`.

`run_pipeline.py` performs the following steps:

1. Loads spins and energies.
2. Solves a least-squares problem to recover the couplings.
3. Reports MAE/RMSE for both the recovered couplings and the predicted energies.

### Usage

```bash
cd pipelines/linear_chain
python run_pipeline.py --report linear_report.json
```

Expected runtime on the i3-13100 desktop is <1 s (pure NumPy).

