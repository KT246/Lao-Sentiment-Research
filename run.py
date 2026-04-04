import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


DEFAULT_MODEL_ORDER = {
    "xlm-roberta": 0,
    "mbert": 1,
    "mbert-lao": 2,
    "textcnn": 3,
    "logistic-regression": 4,
    "svm": 5,
    "decision-tree": 6,
}

DEFAULT_MODE_ORDER = {
    "baseline": 0,
    "from-scratch": 1,
    "full-finetuning": 2,
    "lora": 3,
    "cross-validation": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics into one CSV file")
    parser.add_argument("--results_dir", type=str, default="outputs/experiment", help="Directory containing experiment folders")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/experiment/cv_3_models_3_versions_metrics.csv",
        help="Path to the aggregated CSV file"
    )
    parser.add_argument(
        "--output_mode",
        type=str,
        default="combined",
        choices=["per-model", "combined", "both"],
        help="Export mode: one file per model, one combined file, or both"
    )
    return parser.parse_args()


def detect_prediction_file(experiment_dir: Path):
    cross_val_file = experiment_dir / "cross_validation_predictions.csv"
    single_run_file = experiment_dir / "predictions.csv"

    if cross_val_file.exists():
        return cross_val_file
    if single_run_file.exists():
        return single_run_file
    return None


def discover_experiment_dirs(results_dir: Path):
    experiment_dirs = []
    for item in results_dir.iterdir():
        if not item.is_dir():
            continue
        if detect_prediction_file(item) is not None:
            experiment_dirs.append(item)
    return experiment_dirs


def load_experiment_metadata(experiment_dir: Path):
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_metadata_from_name(experiment_name: str):
    if experiment_name.endswith("-cross-validation"):
        training_mode = "cross-validation"
        model_key = experiment_name[: -len("-cross-validation")]
    elif experiment_name.endswith("-finetuning"):
        training_mode = "full-finetuning"
        model_key = experiment_name[: -len("-finetuning")]
    elif experiment_name.endswith("-lora"):
        training_mode = "lora"
        model_key = experiment_name[: -len("-lora")]
    else:
        training_mode = "unknown"
        model_key = experiment_name

    return {
        "model_key": model_key,
        "training_mode": training_mode,
    }


def build_metrics_row(experiment_dir: Path):
    prediction_file = detect_prediction_file(experiment_dir)
    if prediction_file is None:
        return None

    df = pd.read_csv(prediction_file)
    true_label_col = "true_label" if "true_label" in df.columns else "label"
    if true_label_col not in df.columns or "predicted_label" not in df.columns:
        return None

    y_true = df[true_label_col]
    y_pred = df["predicted_label"]

    metadata = load_experiment_metadata(experiment_dir)
    if not metadata:
        metadata = infer_metadata_from_name(experiment_dir.name)

    return {
        "experiment_name": experiment_dir.name,
        "model_key": metadata.get("model_key", experiment_dir.name),
        "training_mode": metadata.get("training_mode", "unknown"),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 6),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 6),
    }


def save_csv_with_fallback(df: pd.DataFrame, target_path: Path):
    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        fallback_path = target_path.with_stem(f"{target_path.stem}_updated")
        df.to_csv(fallback_path, index=False)
        return fallback_path


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for experiment_dir in discover_experiment_dirs(results_dir):
        metrics_row = build_metrics_row(experiment_dir)
        if metrics_row is not None:
            rows.append(metrics_row)

    if not rows:
        raise FileNotFoundError(f"No valid prediction files found in {results_dir}")

    summary_df = pd.DataFrame(rows)
    summary_df["model_order"] = summary_df["model_key"].map(DEFAULT_MODEL_ORDER).fillna(999)
    summary_df["mode_order"] = summary_df["training_mode"].map(DEFAULT_MODE_ORDER).fillna(999)
    summary_df = summary_df.sort_values(
        ["model_order", "mode_order", "experiment_name"]
    ).drop(columns=["model_order", "mode_order"])
    if args.output_mode in ["combined", "both"]:
        saved_path = save_csv_with_fallback(summary_df, output_csv)
        print(f"Saved aggregated metrics to: {saved_path}")

    if args.output_mode in ["per-model", "both"]:
        output_dir = output_csv.parent
        for model_key, model_df in summary_df.groupby("model_key", sort=False):
            model_df = model_df.copy()
            if model_df.empty:
                continue
            model_csv = output_dir / f"cv_{model_key}_metrics.csv"
            saved_path = save_csv_with_fallback(model_df, model_csv)
            print(f"Saved model metrics to: {saved_path}")


if __name__ == "__main__":
    main()
