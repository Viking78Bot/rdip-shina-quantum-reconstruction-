## 1D Lattice Pipeline

This pipeline demonstrates end-to-end dm₂ reconstruction on a synthetic
1D Ising-like dataset. It reuses the GraphRulerNet architecture, but the
dataset is generated locally (`data/ising_chain_1d.npz`) and contains 200
length-10 spin configurations with periodic boundary energies.

### Usage

```bash
cd pipelines/lattice_1d
python run_pipeline.py --epochs 5 --checkpoint outputs/ising_ruler.pt
```

The script:

1. Generates (or loads) the 1D spin dataset.
2. Trains a lightweight GraphRulerNet (same codepath as Fe₂S₂).
3. Evaluates RMSE/MAE on held-out samples.

On the reference desktop, a 5-epoch run finishes in ~3 s.

