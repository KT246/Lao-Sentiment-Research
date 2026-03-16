import argparse
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CURRENT_SRC_DIR = Path(__file__).resolve().parents[2]
if str(CURRENT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_SRC_DIR))

from sentiment_classification.models.sklearn_baselines import (
    build_feature_summary,
    build_sklearn_text_pipeline,
    predict_with_confidence,
)
from sentiment_classification.utils.utils import get_hardware_info, save_json


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


BASELINE_MODEL_REGISTRY = {
    "logistic-regression": "Logistic Regression",
    "decision-tree": "Decision Tree",
    "svm": "SVM",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train sklearn baselines for Lao Sentiment Analysis")
    parser.add_argument("--model_key", type=str, required=True, choices=sorted(BASELINE_MODEL_REGISTRY.keys()))
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dropna", action="store_true", default=True)
    parser.add_argument("--no-dropna", dest="dropna", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tfidf_ngram_min", type=int, default=3)
    parser.add_argument("--tfidf_ngram_max", type=int, default=5)
    parser.add_argument("--tfidf_min_df", type=int, default=2)
    parser.add_argument("--tfidf_max_features", type=int, default=50000)
    parser.add_argument("--logreg_c", type=float, default=1.0)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--decision_tree_max_depth", type=int, default=40)
    parser.add_argument("--decision_tree_min_samples_leaf", type=int, default=2)
    return parser.parse_args()


def load_train_val_dataframes(data_dir: str, dropna: bool = True):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing train.csv or val.csv in {data_dir}. Please add your data.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    required_cols = {"text", "label"}
    if not required_cols.issubset(train_df.columns) or not required_cols.issubset(val_df.columns):
        raise ValueError("train.csv and val.csv must contain 'text' and 'label' columns.")

    train_df = train_df[["text", "label"]].copy()
    val_df = val_df[["text", "label"]].copy()

    if dropna:
        train_df = train_df.dropna(subset=["text", "label"]).reset_index(drop=True)
        val_df = val_df.dropna(subset=["text", "label"]).reset_index(drop=True)

    return train_df, val_df


def prepare_runtime_metadata(args):
    return {
        "model_key": args.model_key,
        "model_name": BASELINE_MODEL_REGISTRY[args.model_key],
        "model_backend": "sklearn",
        "training_mode": "baseline",
        "seed": args.seed,
        "feature_extractor": {
            "type": "tfidf",
            "analyzer": "char",
            "ngram_range": [args.tfidf_ngram_min, args.tfidf_ngram_max],
            "min_df": args.tfidf_min_df,
            "max_features": args.tfidf_max_features,
            "sublinear_tf": True,
            "lowercase": False,
        },
        "classifier_hyperparameters": {
            "logreg_c": args.logreg_c,
            "svm_c": args.svm_c,
            "decision_tree_max_depth": args.decision_tree_max_depth,
            "decision_tree_min_samples_leaf": args.decision_tree_min_samples_leaf,
        },
    }


def main():
    args = parse_args()
    if not args.output_dir:
        args.output_dir = os.path.join("outputs", "experiment", args.model_key)

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(get_hardware_info(), os.path.join(args.output_dir, "hardware_metrics.json"))
    save_json(prepare_runtime_metadata(args), os.path.join(args.output_dir, "experiment_config.json"))

    train_df, val_df = load_train_val_dataframes(data_dir=args.data_dir, dropna=args.dropna)
    train_texts = train_df["text"].fillna("").astype(str)
    val_texts = val_df["text"].fillna("").astype(str)
    train_labels = train_df["label"].astype(int)

    logger.info(
        "Running baseline model=%s mode=baseline train_rows=%s val_rows=%s",
        args.model_key,
        len(train_df),
        len(val_df),
    )

    pipeline = build_sklearn_text_pipeline(
        model_key=args.model_key,
        seed=args.seed,
        tfidf_config={
            "ngram_range": (args.tfidf_ngram_min, args.tfidf_ngram_max),
            "min_df": args.tfidf_min_df,
            "max_features": args.tfidf_max_features,
        },
        logreg_c=args.logreg_c,
        svm_c=args.svm_c,
        decision_tree_max_depth=args.decision_tree_max_depth,
        decision_tree_min_samples_leaf=args.decision_tree_min_samples_leaf,
    )

    fit_start = time.time()
    pipeline.fit(train_texts, train_labels)
    fit_time_seconds = round(time.time() - fit_start, 2)

    predict_start = time.time()
    predicted_labels, confidence_scores, confidence_method = predict_with_confidence(pipeline, val_texts)
    prediction_time_seconds = round(time.time() - predict_start, 2)

    val_df.to_csv(os.path.join(args.output_dir, "validation_data.csv"), index=False)
    prediction_df = val_df.reset_index(drop=True).copy()
    prediction_df["true_label"] = prediction_df["label"]
    prediction_df["predicted_label"] = predicted_labels.astype(int)
    prediction_df["confidence_score"] = np.round(confidence_scores, 6)
    prediction_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    feature_summary = build_feature_summary(pipeline)
    model_artifact_path = os.path.join(args.output_dir, "best_model.joblib")
    joblib.dump(pipeline, model_artifact_path)

    total_training_time_seconds = round(fit_time_seconds + prediction_time_seconds, 2)
    save_json(
        {
            "model_artifact_path": model_artifact_path,
            "confidence_method": confidence_method,
            "feature_summary": feature_summary,
        },
        os.path.join(args.output_dir, "best_model_info.json"),
    )
    save_json(
        {
            "epoch_times_seconds": [fit_time_seconds],
            "average_epoch_time_seconds": fit_time_seconds,
            "epochs_completed": 1,
            "fit_time_seconds": fit_time_seconds,
            "prediction_time_seconds": prediction_time_seconds,
            "total_training_time_seconds": total_training_time_seconds,
        },
        os.path.join(args.output_dir, "timing_metrics.json"),
    )
    save_json(
        {
            "framework": "scikit-learn",
            "trainable_params": None,
            "total_params": None,
            "trainable_ratio": None,
        },
        os.path.join(args.output_dir, "trainable_params.json"),
    )
    save_json(feature_summary, os.path.join(args.output_dir, "feature_stats.json"))

    logger.info("Baseline experiment completed successfully.")


if __name__ == "__main__":
    main()
