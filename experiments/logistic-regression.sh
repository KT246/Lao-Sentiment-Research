#!/bin/bash
set -e
bash ./experiments/run_baseline.sh "logistic-regression" "baseline" "outputs/experiment/logistic-regression" \
  --epochs 25 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --save_strategy "epoch"
