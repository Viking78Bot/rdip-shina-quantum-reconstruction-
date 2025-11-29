# RDIP + Shina Quantum Reconstruction

Reproducible pipelines and documentation for the RD-IP + Shina Fe₂S₂ study.

- `pipelines/` bundles four ready-to-run projects (linear chain, Zeise complex,
  Fe₂S₂ dm₂ training/inference, and a 1D lattice demo) with data, checkpoints,
  and per-folder README instructions.
- `docs/rdip_shina_paper.pdf` captures the RD-IP + Shina system overview.
- `docs/diffusion_math.pdf` documents the diffusion-based math experiments that
  feed the third part of the article series.

Each pipeline is executed via `python run_pipeline.py` inside its directory;
required dependencies and runtime notes live in the local README files.
