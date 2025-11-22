## Pipelines Overview

This folder collects three self-contained pipelines so that a user can run the
entire workflow for each target directly from the command line:

| Pipeline        | Dataset type | Entry point                           | Notes                                   |
|-----------------|--------------|---------------------------------------|-----------------------------------------|
| `linear_chain`  | Synthetic    | `python run_pipeline.py`              | Fits Ising-like couplings on ±1 spins   |
| `zeise_complex` | Synthetic    | `python run_pipeline.py`              | Regresses spectroscopy-like descriptors |
| `fe2s2_cluster` | Real         | `python run_pipeline.py --epochs 5`   | Fine-tunes + evaluates GraphRulerNet    |
| `lattice_1d`    | Synthetic    | `python run_pipeline.py`              | End-to-end dm₂ diffusion demo for 1D    |

All datasets required to rerun the examples are stored under each pipeline's
`data/` directory, so the folder can be zipped and shared without extra steps.
See the individual `README.md` files for details on runtime, outputs and
hardware requirements.

