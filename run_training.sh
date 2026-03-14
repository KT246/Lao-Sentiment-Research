#!/bin/bash

set -e

echo "[INFO] Starting training pipeline for Lao Sentiment Analysis..."

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

MODEL_KEY="xlm-roberta"
TRAINING_MODE="full-finetuning"
OUTPUT_DIR="outputs/experiment/xlm-roberta-finetuning"

python src/sentiment_classification/scripts/train.py \
  --model_key "$MODEL_KEY" \
  --training_mode "$TRAINING_MODE" \
  --output_dir "$OUTPUT_DIR"

echo "[SUCCESS] Training process completed! Artifacts saved to $OUTPUT_DIR"
