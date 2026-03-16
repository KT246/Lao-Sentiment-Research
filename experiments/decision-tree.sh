#!/bin/bash
set -e
python src/sentiment_classification/scripts/train_baseline.py \
  --model_key "decision-tree" \
  --output_dir "outputs/experiment/decision-tree"
