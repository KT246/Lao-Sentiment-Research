import argparse
import os
import logging
import torch
import numpy as np
import time
from sklearn.utils.class_weight import compute_class_weight
from sentiment_classification.data.dataset import load_and_prepare_data
from sentiment_classification.models.trainer import setup_trainer
from sentiment_classification.utils.utils import get_hardware_info, save_json

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Lao Sentiment Analysis Model")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Pre-trained model name/path")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing train.csv and val.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/v1-xlm-roberta", help="Where to save the artifacts")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of sentiment classes (e.g., 2 for Binary Classification)")
    
    args = parser.parse_args()

    # Track Start Time
    start_time = time.time()

    logger.info(f"Model Name: {args.model_name}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Number of Labels (Classes): {args.num_labels}")
    
    # 0. Save Hardware Information
    logger.info("Gathering and saving hardware information...")
    hardware_info = get_hardware_info()
    os.makedirs(args.output_dir, exist_ok=True)
    save_json(hardware_info, os.path.join(args.output_dir, "hardware_metrics.json"))

    # 1. Load and Tokenize Data
    logger.info(f"Loading data from {args.data_dir}...")
    try:
        tokenized_datasets, tokenizer = load_and_prepare_data(
            data_dir=args.data_dir, 
            tokenizer_name=args.model_name
        )
        logger.info("Data loaded and tokenized successfully.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Compute class weights for imbalanced data
    logger.info("Computing class weights for imbalanced dataset...")
    train_labels = np.array(tokenized_datasets["train"]["label"])
    classes = np.unique(train_labels)
    # Tự động tính toán tỷ trọng dựa trên số lượng mẫu của class 0 và class 1
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class weights computed: {weights}")

    # 2. Setup Trainer
    logger.info("Initializing Custom HuggingFace WeightedTrainer...")
    trainer, model = setup_trainer(
        model_name=args.model_name,
        num_labels=args.num_labels,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weights=class_weights_tensor
    )

    # 3. Train Model
    logger.info("Starting Training...")
    trainer.train()

    # 4. Evaluate on Validation Set
    logger.info("Evaluating Best Model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation Results: {eval_results}")

    # 4.5 Save Detailed Validation Report
    logger.info("Generating detailed validation report (validation_results_detailed.csv)...")
    preds_output = trainer.predict(tokenized_datasets["validation"])
    y_pred = np.argmax(preds_output.predictions, axis=1)
    probs = torch.nn.functional.softmax(torch.tensor(preds_output.predictions), dim=-1).numpy()
    confidence = np.max(probs, axis=1)

    import pandas as pd
    val_path = os.path.join(args.data_dir, "val.csv")
    val_df = pd.read_csv(val_path)
    val_df['predicted_label'] = y_pred
    val_df['confidence_score'] = confidence
    val_df['is_correct'] = val_df['label'] == val_df['predicted_label']
    csv_path = os.path.join(args.output_dir, 'validation_results_detailed.csv')
    val_df.to_csv(csv_path, index=False)

    # 5. Save Final Model locally
    logger.info(f"Saving final model to {args.output_dir}/final_model")
    trainer.save_model(f"{args.output_dir}/final_model")
    
    # 6. Save Timing Metrics
    end_time = time.time()
    total_time_seconds = end_time - start_time
    timing_metrics = {
        "total_time_seconds": round(total_time_seconds, 2),
        "total_time_minutes": round(total_time_seconds / 60, 2),
        "total_time_hours": round(total_time_seconds / 3600, 2)
    }
    logger.info(f"Training completed in {timing_metrics['total_time_minutes']} minutes.")
    save_json(timing_metrics, os.path.join(args.output_dir, "timing_metrics.json"))
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()