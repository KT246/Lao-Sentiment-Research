import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect false positive and false negative samples from all experiments into one CSV file."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=r"C:\Users\LENOVO\Downloads\res",
        help="Directory containing experiment folders.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=r"C:\Users\LENOVO\Downloads\res\latex\fp_fn\all_models_fp_fn.csv",
        help="Path to the combined false-positive/false-negative CSV file.",
    )
    parser.add_argument(
        "--max_per_error_type",
        type=int,
        default=0,
        help="Maximum number of false-positive and false-negative samples to keep for each experiment. Use 0 for full output.",
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
        if item.is_dir() and detect_prediction_file(item) is not None:
            experiment_dirs.append(item)
    return sorted(experiment_dirs, key=lambda p: p.name.lower())


def load_experiment_metadata(experiment_dir: Path):
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_metadata_from_name(experiment_name: str):
    if experiment_name.endswith("-cross-validation"):
        return {
            "model_key": experiment_name[: -len("-cross-validation")],
            "training_mode": "cross-validation",
        }
    if experiment_name.endswith("-finetuning"):
        return {
            "model_key": experiment_name[: -len("-finetuning")],
            "training_mode": "full-finetuning",
        }
    if experiment_name.endswith("-lora"):
        return {
            "model_key": experiment_name[: -len("-lora")],
            "training_mode": "lora",
        }
    return {
        "model_key": experiment_name,
        "training_mode": "baseline",
    }


def build_error_rows(experiment_dir: Path):
    prediction_file = detect_prediction_file(experiment_dir)
    if prediction_file is None:
        return []

    df = pd.read_csv(prediction_file)
    true_label_col = "true_label" if "true_label" in df.columns else "label"
    if true_label_col not in df.columns or "predicted_label" not in df.columns:
        return []

    metadata = load_experiment_metadata(experiment_dir) or infer_metadata_from_name(experiment_dir.name)

    working_df = df.copy()
    working_df["true_label"] = working_df[true_label_col]

    false_positive_df = working_df[
        (working_df["true_label"] == 0) & (working_df["predicted_label"] == 1)
    ].copy()
    false_positive_df["error_type"] = "false_positive"

    false_negative_df = working_df[
        (working_df["true_label"] == 1) & (working_df["predicted_label"] == 0)
    ].copy()
    false_negative_df["error_type"] = "false_negative"

    error_df = pd.concat([false_positive_df, false_negative_df], ignore_index=True)
    if error_df.empty:
        return []

    error_df.insert(0, "experiment_name", experiment_dir.name)
    error_df.insert(1, "model_key", metadata.get("model_key", experiment_dir.name))
    error_df.insert(2, "training_mode", metadata.get("training_mode", "unknown"))

    preferred_columns = [
        "experiment_name",
        "model_key",
        "training_mode",
        "error_type",
        "text",
        "true_label",
        "predicted_label",
        "confidence_score",
        "fold",
        "source_index",
    ]
    existing_columns = [col for col in preferred_columns if col in error_df.columns]
    remaining_columns = [col for col in error_df.columns if col not in existing_columns]
    error_df = error_df[existing_columns + remaining_columns]

    return error_df.to_dict(orient="records")


def save_csv_with_fallback(df: pd.DataFrame, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(target_path, index=False, encoding="utf-8-sig")
        return target_path
    except PermissionError:
        fallback_path = target_path.with_stem(f"{target_path.stem}_updated")
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        return fallback_path


def limit_unique_texts_per_group(df: pd.DataFrame, max_per_error_type: int):
    if max_per_error_type <= 0:
        return df.reset_index(drop=True)

    selected_groups = []
    for (_, _), group_df in df.groupby(["experiment_name", "error_type"], sort=False):
        unique_group_df = group_df.drop_duplicates(subset=["text"], keep="first")
        selected_groups.append(unique_group_df.head(max_per_error_type))

    return pd.concat(selected_groups, ignore_index=True)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)

    all_rows = []
    for experiment_dir in discover_experiment_dirs(results_dir):
        all_rows.extend(build_error_rows(experiment_dir))

    if not all_rows:
        raise FileNotFoundError(f"No false-positive or false-negative samples found in {results_dir}")

    combined_df = pd.DataFrame(all_rows)
    combined_df = combined_df.sort_values(
        by=["model_key", "training_mode", "experiment_name", "error_type"],
        kind="stable",
    ).reset_index(drop=True)

    combined_df = limit_unique_texts_per_group(combined_df, args.max_per_error_type)

    saved_path = save_csv_with_fallback(combined_df, output_csv)
    print(f"Saved FP/FN samples to: {saved_path}")
    print(f"Total rows: {len(combined_df)}")


if __name__ == "__main__":
    main()
