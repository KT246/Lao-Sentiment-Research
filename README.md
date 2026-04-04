# Lao Sentiment Research

This repository contains the training pipeline for Lao sentiment classification with transformer models, sklearn baselines, and one custom neural baseline trained from scratch.

## What is implemented

- Models:
  - `xlm-roberta` -> `xlm-roberta-base`
  - `mbert` -> `bert-base-multilingual-cased`
  - `mbert-lao` -> `w11wo/lao-roberta-base` (current placeholder for Lao-specific setup)
  - `textcnn` -> custom TextCNN with random initialization
- Training modes:
  - `full-finetuning`
  - `from-scratch`
  - `lora`
  - `cross-validation` (`K=3`)
- During training:
  - Best model is selected by `eval_loss` on validation split
  - No F1/Accuracy metrics are computed during training
  - Runtime artifacts are saved (epoch timing, hardware info, predictions)

## Project structure

```text
Lao-Sentiment-Research/
├── data/
│   └── processed/
│       ├── train.csv
│       └── val.csv
├── experiments/
│   ├── run_experiment.sh
│   ├── xlm-roberta-finetuning.sh
│   ├── xlm-roberta-lora.sh
│   ├── xlm-roberta-cross-validation.sh
│   └── ... (same for mbert, mbert-lao)
├── src/sentiment_classification/
│   ├── data/dataset.py
│   ├── models/factory.py
│   ├── models/trainer.py
│   ├── scripts/train.py
│   └── utils/config.py
├── requirements.txt
├── run_training.sh
└── setup.py
```

## Data format

`data/processed/train.csv` and `data/processed/val.csv` must contain:

| column | type |
| --- | --- |
| `text` | string |
| `label` | integer (`0` negative, `1` positive) |

## Installation

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Run training

Default quick run:

```bash
./run_training.sh
```

Direct run:

```bash
python src/sentiment_classification/scripts/train.py \
  --model_key xlm-roberta \
  --training_mode full-finetuning \
  --output_dir outputs/experiment/xlm-roberta-finetuning
```

Run LoRA:

```bash
python src/sentiment_classification/scripts/train.py \
  --model_key xlm-roberta \
  --training_mode lora \
  --output_dir outputs/experiment/xlm-roberta-lora
```

Run from scratch:

```bash
python src/sentiment_classification/scripts/train.py \
  --model_key textcnn \
  --training_mode from-scratch \
  --output_dir outputs/experiment/textcnn-training
```

Run K-Fold=3:

```bash
python src/sentiment_classification/scripts/train.py \
  --model_key xlm-roberta \
  --training_mode cross-validation \
  --num_folds 3 \
  --output_dir outputs/experiment/xlm-roberta-cross-validation
```

If you want cross-validation to use `train + val` together, add:

```bash
--cv_include_val
```

TextCNN baseline note:

- `textcnn` uses the `xlm-roberta-base` tokenizer for subword IDs, but its embedding/CNN/classifier weights are randomly initialized.
- `textcnn` supports `from-scratch` and `cross-validation`; `LoRA` is intentionally disabled.

## Outputs

Single split (`full-finetuning`, `from-scratch`, `lora`) output folder contains:

- `best_model/`
- `best_model_info.json`
- `validation_data.csv`
- `predictions.csv`
- `timing_metrics.json`
- `hardware_metrics.json`
- `experiment_config.json`
- `trainable_params.json`

Cross-validation output folder contains:

- `fold_1/`, `fold_2/`, `fold_3/` each with:
  - `best_model/`
  - `best_model_info.json`
  - `validation_data.csv`
  - `predictions.csv`
  - `timing_metrics.json`
- `cross_validation_predictions.csv`
- `timing_metrics.json` (aggregated)
- `hardware_metrics.json`
- `experiment_config.json`

## Suggested branch strategy

Create one branch per experiment family, for example:

- `experiment/xlm-roberta-finetuning`
- `experiment/xlm-roberta-lora`
- `experiment/xlm-roberta-cross-validation`
- `experiment/textcnn-training`
- `experiment/textcnn-cross-validation`

The same naming pattern can be used for `mbert` and `mbert-lao`.
