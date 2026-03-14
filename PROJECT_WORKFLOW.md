# Project Workflow

This file documents the practical workflow for this repository.

## 1. Prepare data

- Put files in `data/processed/`:
  - `train.csv`
  - `val.csv`
- Required columns: `text`, `label`

## 2. Choose experiment branch

Recommended branch naming:

- `experiment/<model>-<mode>`
- Examples:
  - `experiment/xlm-roberta-lora`
  - `experiment/mbert-finetuning`
  - `experiment/mbert-lao-cross-validation`

## 3. Train

Use one of:

- `./experiments/xlm-roberta-finetuning.sh`
- `./experiments/xlm-roberta-lora.sh`
- `./experiments/xlm-roberta-cross-validation.sh`

Or run `train.py` directly for full control.

## 4. Validate artifacts

Check inside output folder:

- `best_model/`
- `best_model_info.json`
- `validation_data.csv`
- `predictions.csv` (or `cross_validation_predictions.csv`)
- `timing_metrics.json`
- `hardware_metrics.json`

## 5. Commit and push

```bash
git checkout -b experiment/<model>-<mode>
git add .
git commit -m "experiment: <model> <mode> baseline"
git push -u origin experiment/<model>-<mode>
```

## Notes

- Best model selection is based on validation `eval_loss`.
- Training does not compute F1/Accuracy during epochs.
- Cross-validation default pool is `train.csv`; add `--cv_include_val` to include both `train + val`.
