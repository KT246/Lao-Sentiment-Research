#!/bin/bash

set -e

MODEL_KEY="$1"
TRAINING_MODE="$2"
OUTPUT_DIR="$3"

if [ -z "$MODEL_KEY" ] || [ -z "$TRAINING_MODE" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: ./experiments/run_experiment.sh <model_key> <training_mode> <output_dir>"
  exit 1
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python src/sentiment_classification/scripts/train.py \
  --model_key "$MODEL_KEY" \
  --training_mode "$TRAINING_MODE" \
  --output_dir "$OUTPUT_DIR"
