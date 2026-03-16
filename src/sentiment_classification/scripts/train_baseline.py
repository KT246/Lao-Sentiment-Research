import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

CURRENT_SRC_DIR = Path(__file__).resolve().parents[2]
if str(CURRENT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_SRC_DIR))

from sentiment_classification.models.sklearn_baselines import (
    build_baseline_estimator,
    build_sklearn_text_pipeline,
    build_text_vectorizer,
    compute_eval_loss,
    predict_with_confidence,
)
from sentiment_classification.data.dataset import load_train_val_dataframes
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
    parser.add_argument("--training_mode", type=str, default="baseline", choices=["baseline"])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--dropna", action="store_true", default=True)
    parser.add_argument("--no-dropna", dest="dropna", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--tfidf_ngram_min", type=int, default=3)
    parser.add_argument("--tfidf_ngram_max", type=int, default=5)
    parser.add_argument("--tfidf_min_df", type=int, default=2)
    parser.add_argument("--tfidf_max_features", type=int, default=50000)
    parser.add_argument("--logreg_c", type=float, default=1.0)
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--decision_tree_max_depth", type=int, default=40)
    parser.add_argument("--decision_tree_min_samples_leaf", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--padding", type=str, default="max_length")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    return parser.parse_args()


def prepare_runtime_metadata(args):
    return {
        "model_key": args.model_key,
        "model_name": BASELINE_MODEL_REGISTRY[args.model_key],
        "model_backend": "sklearn",
        "training_mode": args.training_mode,
        "use_lora": False,
        "use_cross_validation": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_labels": args.num_labels,
        "max_length": args.max_length,
        "padding": args.padding,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "num_folds": 1,
        "cv_include_val": False,
        "save_strategy": args.save_strategy,
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


def _is_better_epoch(candidate_eval_loss, best_eval_loss):
    if best_eval_loss is None:
        return True

    return candidate_eval_loss < best_eval_loss


def save_best_model_artifacts(best_pipeline, output_dir, best_eval_loss):
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    model_artifact_path = os.path.join(best_model_dir, "model.joblib")
    joblib.dump(best_pipeline, model_artifact_path)
    save_json(
        {
            "best_model_checkpoint": best_model_dir,
            "best_metric_eval_loss": best_eval_loss,
        },
        os.path.join(output_dir, "best_model_info.json"),
    )


def save_epoch_runtime(output_dir, epoch, global_step):
    save_json(
        {
            "epoch": epoch,
            "global_step": global_step,
        },
        os.path.join(output_dir, f"epoch_{epoch}_runtime.json"),
    )


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
    train_labels = train_df["label"].astype(int).to_numpy()
    val_labels = val_df["label"].astype(int).to_numpy()

    logger.info(
        "Running baseline model=%s mode=%s epochs=%s train_rows=%s val_rows=%s",
        args.model_key,
        args.training_mode,
        args.epochs,
        len(train_df),
        len(val_df),
    )

    vectorizer = build_text_vectorizer(
        tfidf_config={
            "ngram_range": (args.tfidf_ngram_min, args.tfidf_ngram_max),
            "min_df": args.tfidf_min_df,
            "max_features": args.tfidf_max_features,
        }
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    classes = np.array(sorted(np.unique(train_labels)))

    best_classifier = None
    best_epoch = None
    best_eval_loss = None
    epoch_times = []
    rng = np.random.default_rng(args.seed)
    global_step = 0

    if args.model_key == "decision-tree":
        for epoch in range(1, args.epochs + 1):
            epoch_classifier = build_baseline_estimator(
                model_key=args.model_key,
                seed=args.seed + epoch - 1,
                logreg_c=args.logreg_c,
                svm_c=args.svm_c,
                decision_tree_max_depth=args.decision_tree_max_depth,
                decision_tree_min_samples_leaf=args.decision_tree_min_samples_leaf,
            )
            epoch_start = time.time()
            epoch_classifier.fit(x_train, train_labels)
            epoch_duration = round(time.time() - epoch_start, 2)
            epoch_times.append(epoch_duration)
            global_step += 1

            eval_loss = compute_eval_loss(args.model_key, epoch_classifier, x_val, val_labels)
            save_epoch_runtime(args.output_dir, epoch, global_step)

            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp} | epoch={epoch} | train_time={epoch_duration:.2f}s")

            if _is_better_epoch(eval_loss, best_eval_loss):
                best_classifier = copy.deepcopy(epoch_classifier)
                best_epoch = epoch
                best_eval_loss = eval_loss
    else:
        classifier = build_baseline_estimator(
            model_key=args.model_key,
            seed=args.seed,
            logreg_c=args.logreg_c,
            svm_c=args.svm_c,
            decision_tree_max_depth=args.decision_tree_max_depth,
            decision_tree_min_samples_leaf=args.decision_tree_min_samples_leaf,
        )
        for epoch in range(1, args.epochs + 1):
            permutation = rng.permutation(len(train_labels))
            epoch_start = time.time()
            for batch_start in range(0, len(train_labels), args.batch_size):
                batch_indices = permutation[batch_start: batch_start + args.batch_size]
                x_batch = x_train[batch_indices]
                y_batch = train_labels[batch_indices]
                if epoch == 1 and batch_start == 0:
                    classifier.partial_fit(x_batch, y_batch, classes=classes)
                else:
                    classifier.partial_fit(x_batch, y_batch)
                global_step += 1
            epoch_duration = round(time.time() - epoch_start, 2)
            epoch_times.append(epoch_duration)

            eval_loss = compute_eval_loss(args.model_key, classifier, x_val, val_labels)
            save_epoch_runtime(args.output_dir, epoch, global_step)

            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp} | epoch={epoch} | train_time={epoch_duration:.2f}s")

            if _is_better_epoch(eval_loss, best_eval_loss):
                best_classifier = copy.deepcopy(classifier)
                best_epoch = epoch
                best_eval_loss = eval_loss

    if best_classifier is None:
        raise RuntimeError("No baseline model was trained successfully.")

    best_pipeline = build_sklearn_text_pipeline(vectorizer, best_classifier)
    predicted_labels, confidence_scores, _ = predict_with_confidence(best_pipeline, val_texts)

    val_df.to_csv(os.path.join(args.output_dir, "validation_data.csv"), index=False)
    prediction_df = val_df.reset_index(drop=True).copy()
    prediction_df["true_label"] = prediction_df["label"]
    prediction_df["predicted_label"] = predicted_labels.astype(int)
    prediction_df["confidence_score"] = np.round(confidence_scores, 6)
    prediction_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    total_training_time_seconds = round(sum(epoch_times), 2)
    save_best_model_artifacts(best_pipeline, args.output_dir, best_eval_loss)
    save_json(
        {
            "epoch_times_seconds": epoch_times,
            "average_epoch_time_seconds": round(sum(epoch_times) / len(epoch_times), 2),
            "epochs_completed": len(epoch_times),
            "total_training_time_seconds": total_training_time_seconds,
        },
        os.path.join(args.output_dir, "timing_metrics.json"),
    )
    save_json(
        {
            "trainable_params": None,
            "total_params": None,
            "trainable_ratio": None,
        },
        os.path.join(args.output_dir, "trainable_params.json"),
    )

    logger.info(
        "Baseline experiment completed successfully. best_epoch=%s best_eval_loss=%.6f",
        best_epoch,
        best_eval_loss,
    )


if __name__ == "__main__":
    main()
