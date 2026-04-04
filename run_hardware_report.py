import argparse
import csv
import json
from pathlib import Path


EXCLUDED_DIRS = {".venv", "latex", "reports", "__pycache__"}


MODEL_DISPLAY_NAMES = {
    "xlm-roberta": "XLM-RoBERTa",
    "mbert": "mBERT",
    "mbert-lao": "mBERT-Lao",
    "textcnn": "TextCNN",
    "logistic-regression": "Logistic Regression",
    "svm": "SVM",
    "decision-tree": "Decision Tree",
}


TRAINING_MODE_DISPLAY = {
    "from-scratch": "From Scratch",
    "full-finetuning": "Full Fine-tuning",
    "lora": "LoRA",
    "cross-validation": "3-Fold Cross-Validation",
    "baseline": "Baseline",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate hardware information from Lao sentiment experiment folders."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path(r"C:\Users\LENOVO\Downloads\res"),
        help="Root directory that contains experiment folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(r"C:\Users\LENOVO\Downloads\res\latex\hardware"),
        help="Directory where CSV reports will be written.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_experiment_dirs(results_dir: Path):
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in EXCLUDED_DIRS:
            continue
        if (child / "hardware_metrics.json").exists():
            yield child


def flatten_gpu_names(gpu_list):
    if not gpu_list:
        return ""
    return "; ".join(str(item.get("name", "")) for item in gpu_list if item.get("name"))


def flatten_gpu_memory(gpu_list):
    if not gpu_list:
        return ""
    return "; ".join(str(item.get("total_memory_gb", "")) for item in gpu_list)


def build_row(experiment_dir: Path):
    hardware = load_json(experiment_dir / "hardware_metrics.json")
    config_path = experiment_dir / "experiment_config.json"
    config = load_json(config_path) if config_path.exists() else {}

    model_key = config.get("model_key", experiment_dir.name)
    training_mode = config.get("training_mode", "")
    gpu_list = hardware.get("gpu", [])
    cpu = hardware.get("cpu", {})
    ram = hardware.get("ram", {})

    return {
        "experiment_name": experiment_dir.name,
        "model_key": model_key,
        "model_name_display": MODEL_DISPLAY_NAMES.get(model_key, model_key),
        "training_mode": training_mode,
        "training_mode_display": TRAINING_MODE_DISPLAY.get(training_mode, training_mode),
        "cpu_physical_cores": cpu.get("physical_cores"),
        "cpu_total_cores": cpu.get("total_cores"),
        "cpu_max_frequency_mhz": cpu.get("max_frequency_mhz"),
        "cpu_current_frequency_mhz": cpu.get("current_frequency_mhz"),
        "cpu_usage_percent": cpu.get("cpu_usage_percent"),
        "ram_total_gb": ram.get("total_gb"),
        "ram_available_gb": ram.get("available_gb"),
        "ram_used_gb": ram.get("used_gb"),
        "ram_usage_percent": ram.get("usage_percent"),
        "gpu_count": len(gpu_list),
        "gpu_names": flatten_gpu_names(gpu_list),
        "gpu_total_memory_gb": flatten_gpu_memory(gpu_list),
    }


def write_csv(rows, output_path: Path):
    if not rows:
        raise ValueError("No hardware rows found to write.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(rows):
    grouped = {}
    for row in rows:
        key = row["model_key"]
        grouped.setdefault(key, row)

    summary_rows = []
    for key in sorted(grouped.keys()):
        row = grouped[key]
        summary_rows.append(
            {
                "model_key": row["model_key"],
                "model_name_display": row["model_name_display"],
                "cpu_physical_cores": row["cpu_physical_cores"],
                "cpu_total_cores": row["cpu_total_cores"],
                "cpu_current_frequency_mhz": row["cpu_current_frequency_mhz"],
                "ram_total_gb": row["ram_total_gb"],
                "gpu_count": row["gpu_count"],
                "gpu_names": row["gpu_names"],
                "gpu_total_memory_gb": row["gpu_total_memory_gb"],
            }
        )
    return summary_rows


def main():
    args = parse_args()
    rows = [build_row(exp_dir) for exp_dir in iter_experiment_dirs(args.results_dir)]
    if not rows:
        raise SystemExit("No experiment folders with hardware_metrics.json were found.")

    hardware_all_path = args.output_dir / "hardware_all_experiments.csv"
    hardware_summary_path = args.output_dir / "hardware_model_summary.csv"

    write_csv(rows, hardware_all_path)
    write_csv(build_summary_rows(rows), hardware_summary_path)

    print(f"Saved: {hardware_all_path}")
    print(f"Saved: {hardware_summary_path}")
    print(f"Experiments aggregated: {len(rows)}")


if __name__ == "__main__":
    main()
