#!/bin/bash
set -e
python src/sentiment_classification/scripts/train_baseline.py \
  --model_key "logistic-regression" \
  --output_dir "outputs/experiment/logistic-regression"
