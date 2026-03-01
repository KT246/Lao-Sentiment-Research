#!/bin/bash

# ==============================================================================
# Script: run_training.sh
# MLOps Pipeline: Lao Sentiment Analysis (E-commerce)
# 
# Cách sử dụng: 
#   1. Cấp quyền thực thi trước: chmod +x run_training.sh
#   2. Chạy file: ./run_training.sh
# ==============================================================================

# Dừng hệ thống ngay lập tức nếu có bất kỳ lệnh nào bị lỗi
set -e

echo "[INFO] Starting training pipeline for Lao Sentiment Analysis..."

# Biến môi trường giúp Python nhận diện package trong thư mục src/
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Cấu hình siêu tham số (Hyperparameters)
MODEL_NAME="w11wo/lao-roberta-base" # Đổi thành model tiếng Lào cụ thể nếu có
EPOCHS=25
BATCH_SIZE=16
OUTPUT_DIR="outputs/v1-lao-roberta"

# Chạy training script
python src/sentiment_classification/scripts/train.py \
    --model_name "$MODEL_NAME" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "$OUTPUT_DIR"

echo "[SUCCESS] Training process completed! Artifacts saved to $OUTPUT_DIR"
