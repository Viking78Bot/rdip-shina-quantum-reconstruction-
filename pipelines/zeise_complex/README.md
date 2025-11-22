## Zeise-like Complex Pipeline

`data/zeise_synthetic_observations.npz` holds a synthetic spectroscopy-style
dataset with 6 descriptors per sample. The pipeline fits a linear model to
predict the target observable.

### Usage

```bash
cd pipelines/zeise_complex
python run_pipeline.py --report zeise_report.json
```

The script trains a closed-form linear regressor, prints MAE/RMSE and writes the
full report to the JSON file (<1 s runtime on the reference desktop).

