import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


MODEL_ORDER = {
    "xlm-roberta": 0,
    "mbert": 1,
    "mbert-lao": 2,
    "textcnn": 3,
    "logistic-regression": 4,
    "svm": 5,
    "decision-tree": 6,
}

MODE_ORDER = {
    "baseline": 0,
    "from-scratch": 1,
    "full-finetuning": 2,
    "lora": 3,
    "cross-validation": 4,
}

MODEL_DISPLAY = {
    "xlm-roberta": "XLM-RoBERTa",
    "mbert": "mBERT",
    "mbert-lao": "mBERT-Lao",
    "textcnn": "TextCNN",
    "logistic-regression": "Logistic Regression",
    "svm": "SVM",
    "decision-tree": "Decision Tree",
}

MODE_DISPLAY = {
    "baseline": "Baseline",
    "from-scratch": "From Scratch",
    "full-finetuning": "Full Fine-tuning",
    "lora": "LoRA",
    "cross-validation": "Cross-Validation",
}

GROUP_COLORS = [
    "#f4f8ff",
    "#fff8f1",
    "#f3fff3",
    "#fff4fa",
    "#f7f7ff",
    "#f6fff4",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment outputs and generate updated report plots."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="C:/Users/LENOVO/Downloads/res",
        help="Directory containing experiment folders",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="C:/Users/LENOVO/Downloads/res/reports",
        help="Directory to save report outputs",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="cv_all_experiments_metrics.csv",
        help="Combined metrics CSV filename",
    )
    parser.add_argument(
        "--timing_csv",
        type=str,
        default="timing_all_experiments.csv",
        help="Timing summary CSV filename",
    )
    parser.add_argument(
        "--line_plot_name",
        type=str,
        default="lineplot_metrics_9_experiments.png",
        help="Overview metrics plot filename",
    )
    parser.add_argument(
        "--light_sorted_plot_name",
        type=str,
        default="lineplot_metrics_sorted_light.png",
        help="Sorted metrics plot filename",
    )
    parser.add_argument(
        "--timing_plot_name",
        type=str,
        default="lineplot_timing_sorted_light.png",
        help="Sorted timing plot filename",
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
    return sorted(
        experiment_dirs,
        key=lambda path: (
            MODEL_ORDER.get(resolve_metadata(path)["model_key"], 999),
            MODE_ORDER.get(resolve_metadata(path)["training_mode"], 999),
            path.name,
        ),
    )


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment_metadata(experiment_dir: Path):
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        return {}
    return load_json(config_path)


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
        "training_mode": "unknown",
    }


def get_model_display(model_key: str, metadata=None):
    if model_key in MODEL_DISPLAY:
        return MODEL_DISPLAY[model_key]
    if metadata and metadata.get("model_name"):
        return metadata["model_name"]
    return model_key


def get_mode_display(training_mode: str):
    return MODE_DISPLAY.get(training_mode, training_mode)


def resolve_metadata(experiment_dir: Path):
    inferred = infer_metadata_from_name(experiment_dir.name)
    config = load_experiment_metadata(experiment_dir)

    metadata = {
        "model_key": config.get("model_key", inferred["model_key"]),
        "training_mode": config.get("training_mode", inferred["training_mode"]),
    }
    metadata["model_name"] = get_model_display(metadata["model_key"], config)
    metadata["training_mode_display"] = get_mode_display(metadata["training_mode"])
    return metadata


def blend_with_white(color: str, blend_ratio: float = 0.5):
    base_rgb = mcolors.to_rgb(color)
    return tuple((1 - blend_ratio) * channel + blend_ratio for channel in base_rgb)


def save_dataframe_with_fallback(df: pd.DataFrame, target_path: Path):
    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        fallback_path = target_path.with_stem(f"{target_path.stem}_updated")
        df.to_csv(fallback_path, index=False)
        return fallback_path


def save_figure_variants(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    return output_path, svg_path


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
    metadata = resolve_metadata(experiment_dir)

    return {
        "experiment_name": experiment_dir.name,
        "experiment_display": f"{metadata['model_name']} ({metadata['training_mode_display']})",
        "model_key": metadata["model_key"],
        "model_name": metadata["model_name"],
        "training_mode": metadata["training_mode"],
        "training_mode_display": metadata["training_mode_display"],
        "num_samples": len(df),
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 6),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 6),
    }


def build_summary_dataframe(experiment_dirs):
    rows = []
    for experiment_dir in experiment_dirs:
        row = build_metrics_row(experiment_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        raise FileNotFoundError("No valid prediction files found.")

    summary_df = pd.DataFrame(rows)
    summary_df["model_order"] = summary_df["model_key"].map(MODEL_ORDER).fillna(999)
    summary_df["mode_order"] = summary_df["training_mode"].map(MODE_ORDER).fillna(999)
    summary_df = summary_df.sort_values(
        ["model_order", "mode_order", "experiment_name"]
    ).drop(columns=["model_order", "mode_order"])
    return summary_df


def build_timing_dataframe(experiment_dirs):
    rows = []
    for experiment_dir in experiment_dirs:
        timing_path = experiment_dir / "timing_metrics.json"
        if not timing_path.exists():
            continue

        timing_data = load_json(timing_path)
        metadata = resolve_metadata(experiment_dir)
        total_training_time_seconds = timing_data.get("total_training_time_seconds", 0.0)
        rows.append(
            {
                "experiment_name": experiment_dir.name,
                "model_key": metadata["model_key"],
                "model_name": metadata["model_name"],
                "training_mode": metadata["training_mode"],
                "training_mode_display": metadata["training_mode_display"],
                "avg_epoch_time_seconds": timing_data.get("average_epoch_time_seconds"),
                "total_training_time_seconds": total_training_time_seconds,
                "total_training_time_minutes": round(total_training_time_seconds / 60.0, 2),
                "epochs_completed": timing_data.get("epochs_completed"),
                "num_folds": timing_data.get("num_folds", 1),
                "folds_present": len(timing_data.get("folds", []))
                if isinstance(timing_data.get("folds", []), list)
                else 0,
            }
        )

    if not rows:
        return pd.DataFrame()

    timing_df = pd.DataFrame(rows)
    timing_df["model_order"] = timing_df["model_key"].map(MODEL_ORDER).fillna(999)
    timing_df["mode_order"] = timing_df["training_mode"].map(MODE_ORDER).fillna(999)
    timing_df = timing_df.sort_values(
        ["model_order", "mode_order", "experiment_name"]
    ).drop(columns=["model_order", "mode_order"])
    return timing_df


def add_model_group_bands(ax, plot_df, ymin):
    grouped = plot_df.groupby("model_name", sort=False)
    for idx, (model_name, group_df) in enumerate(grouped):
        start = group_df.index.min() - 0.5
        end = group_df.index.max() + 0.5
        ax.axvspan(start, end, color=GROUP_COLORS[idx % len(GROUP_COLORS)], alpha=0.75, zorder=0)
        if idx > 0:
            ax.axvline(start, linestyle="--", color="#9e9e9e", alpha=0.45, linewidth=1.0)
        ax.text(
            (start + end) / 2,
            ymin + 0.015,
            model_name,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="bottom",
        )


def save_line_plot(summary_df: pd.DataFrame, output_path: Path):
    sns.set_theme(style="whitegrid", context="talk")
    mode_short = {
        "Baseline": "Base",
        "Full Fine-tuning": "FT",
        "LoRA": "LoRA",
        "Cross-Validation": "CV",
    }
    metric_palette = {
        "Accuracy": "#1f77b4",
        "F1-macro": "#ff7f0e",
        "Precision-macro": "#2ca02c",
        "Recall-macro": "#d62728",
    }
    metric_map = {
        "accuracy": "Accuracy",
        "f1_macro": "F1-macro",
        "precision_macro": "Precision-macro",
        "recall_macro": "Recall-macro",
    }

    plot_df = summary_df.reset_index(drop=True).copy()
    plot_df["experiment_label"] = plot_df.apply(
        lambda row: f"{row['model_name']}\n{mode_short.get(row['training_mode_display'], row['training_mode_display'])}",
        axis=1,
    )
    x_positions = list(range(len(plot_df)))
    ymin = max(0.0, plot_df[["accuracy", "f1_macro", "precision_macro", "recall_macro"]].min().min() - 0.06)

    fig, ax = plt.subplots(figsize=(18, 8.5))
    add_model_group_bands(ax, plot_df, ymin)

    for metric_col, metric_name in metric_map.items():
        scores = plot_df[metric_col].tolist()
        fill_color = blend_with_white(metric_palette[metric_name], 0.58)
        ax.fill_between(
            x_positions,
            scores,
            [ymin] * len(scores),
            color=fill_color,
            alpha=0.18,
            zorder=1,
        )
        ax.plot(
            x_positions,
            scores,
            marker="o",
            linewidth=2.8,
            markersize=7,
            label=metric_name,
            color=metric_palette[metric_name],
            alpha=0.95,
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["experiment_label"], rotation=24, ha="right")
    ax.set_ylim(ymin, 1.0)
    ax.set_title("Metrics Comparison Across All Experiments", fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel("Experiment Group (Model + Strategy)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", frameon=True, fancybox=True, framealpha=0.92, title="Metric")
    plt.tight_layout()
    png_path, svg_path = save_figure_variants(fig, output_path)
    plt.close(fig)
    return png_path, svg_path


def save_light_sorted_plot(summary_df: pd.DataFrame, output_path: Path):
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        }
    )
    mode_short = {
        "Baseline": "Base",
        "Full Fine-tuning": "FT",
        "LoRA": "LoRA",
        "Cross-Validation": "CV",
    }
    light_palette = {
        "Accuracy": "#1f4e79",
        "F1-macro": "#b36b00",
        "Precision-macro": "#2f6b3d",
        "Recall-macro": "#8c2f39",
    }
    metric_markers = {
        "Accuracy": "o",
        "F1-macro": "s",
        "Precision-macro": "^",
        "Recall-macro": "D",
    }
    metric_map = {
        "accuracy": "Accuracy",
        "f1_macro": "F1-macro",
        "precision_macro": "Precision-macro",
        "recall_macro": "Recall-macro",
    }

    sorted_df = summary_df.sort_values(["f1_macro", "accuracy"], ascending=[False, False]).reset_index(drop=True).copy()
    sorted_df["experiment_label"] = sorted_df.apply(
        lambda row: f"{row['model_name']}\n{mode_short.get(row['training_mode_display'], row['training_mode_display'])}",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(18, 8.5))
    x_positions = list(range(len(sorted_df)))
    ymin = max(0.0, sorted_df[["accuracy", "f1_macro", "precision_macro", "recall_macro"]].min().min() - 0.06)

    best_family_rows = sorted_df[sorted_df["model_name"] == "XLM-RoBERTa"]
    if not best_family_rows.empty:
        ax.axvspan(
            best_family_rows.index.min() - 0.45,
            best_family_rows.index.max() + 0.45,
            color="#eef4fb",
            alpha=0.65,
            zorder=0,
        )
        ax.text(
            (best_family_rows.index.min() + best_family_rows.index.max()) / 2,
            0.992,
            "Best-performing family",
            fontsize=10,
            fontweight="bold",
            color="#1f4e79",
            ha="center",
            va="top",
        )

    for metric_col, metric_name in metric_map.items():
        y_vals = sorted_df[metric_col].tolist()
        ax.plot(
            x_positions,
            y_vals,
            marker=metric_markers[metric_name],
            linewidth=2.6,
            markersize=7,
            label=metric_name,
            color=light_palette[metric_name],
            alpha=0.95,
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_df["experiment_label"], rotation=45, ha="right")
    ax.set_ylim(ymin, 1.0)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.18)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        framealpha=0.95,
        title="Metric",
        fontsize=10,
        title_fontsize=11,
    )
    fig.subplots_adjust(left=0.07, right=0.82, bottom=0.24, top=0.95)
    png_path, svg_path = save_figure_variants(fig, output_path)
    plt.close(fig)
    return png_path, svg_path


def save_timing_sorted_plot(timing_df: pd.DataFrame, output_path: Path):
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        }
    )
    mode_short = {
        "Baseline": "Base",
        "Full Fine-tuning": "FT",
        "LoRA": "LoRA",
        "Cross-Validation": "CV",
    }
    timing_palette = {
        "Avg. Epoch Time (s)": "#8dbdff",
        "Total Training Time (min)": "#f4a261",
    }

    sorted_df = timing_df.sort_values(
        ["avg_epoch_time_seconds", "total_training_time_minutes"],
        ascending=[True, True],
    ).reset_index(drop=True)
    sorted_df["experiment_label"] = sorted_df.apply(
        lambda row: f"{row['model_name']}\n{mode_short.get(row['training_mode_display'], row['training_mode_display'])}",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(12, 5.6))
    x_positions = list(range(len(sorted_df)))
    epoch_vals = sorted_df["avg_epoch_time_seconds"].tolist()
    total_vals = sorted_df["total_training_time_minutes"].tolist()
    ax.plot(
        x_positions,
        epoch_vals,
        marker="o",
        linewidth=2.2,
        markersize=5.8,
        color=timing_palette["Avg. Epoch Time (s)"],
        alpha=0.95,
        label="Avg. Epoch Time (s)",
        zorder=3,
    )
    ax.fill_between(
        x_positions,
        epoch_vals,
        [0] * len(epoch_vals),
        color=timing_palette["Avg. Epoch Time (s)"],
        alpha=0.15,
        zorder=1,
    )
    ax.plot(
        x_positions,
        total_vals,
        marker="o",
        linewidth=2.2,
        markersize=5.8,
        color=timing_palette["Total Training Time (min)"],
        alpha=0.95,
        label="Total Training Time (min)",
        zorder=3,
    )
    ax.fill_between(
        x_positions,
        total_vals,
        [0] * len(total_vals),
        color=timing_palette["Total Training Time (min)"],
        alpha=0.15,
        zorder=1,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_df["experiment_label"], rotation=45, ha="right", fontsize=14)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.text(
        0.03,
        0.52,
        "Time",
        fontsize=12,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="center",
        color="#262626",
    )
    ax.set_ylim(-12, max(epoch_vals) * 1.05)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.30)
    ax.grid(axis="x", linestyle="-", alpha=0.22)
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        title="Timing Metric",
        fancybox=True,
        fontsize=10,
        title_fontsize=11,
    )

    if total_vals:
        ax.annotate(
            f"{total_vals[0]:.02f}",
            (x_positions[0], total_vals[0]),
            xytext=(-18, 8),
            textcoords="offset points",
            color=timing_palette["Total Training Time (min)"],
            fontsize=7.5,
        )
        ax.annotate(
            f"{epoch_vals[-1]:.2f}",
            (x_positions[-1], epoch_vals[-1]),
            xytext=(8, 6),
            textcoords="offset points",
            color=timing_palette["Avg. Epoch Time (s)"],
            fontsize=7.5,
        )
        ax.annotate(
            f"{total_vals[-1]:.2f}",
            (x_positions[-1], total_vals[-1]),
            xytext=(8, 6),
            textcoords="offset points",
            color=timing_palette["Total Training Time (min)"],
            fontsize=7.5,
        )

    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.22, top=0.92)
    png_path, svg_path = save_figure_variants(fig, output_path)
    plt.close(fig)
    return png_path, svg_path


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    experiment_dirs = discover_experiment_dirs(results_dir)
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment folders with predictions found in {results_dir}")

    summary_df = build_summary_dataframe(experiment_dirs)
    timing_df = build_timing_dataframe(experiment_dirs)

    metrics_path = save_dataframe_with_fallback(summary_df, report_dir / args.metrics_csv)
    timing_path = None
    if not timing_df.empty:
        timing_path = save_dataframe_with_fallback(timing_df, report_dir / args.timing_csv)

    overview_png, overview_svg = save_line_plot(summary_df, report_dir / args.line_plot_name)
    light_png, light_svg = save_light_sorted_plot(summary_df, report_dir / args.light_sorted_plot_name)

    timing_png = None
    timing_svg = None
    if not timing_df.empty:
        timing_png, timing_svg = save_timing_sorted_plot(timing_df, report_dir / args.timing_plot_name)

    print(f"Saved combined metrics CSV: {metrics_path}")
    if timing_path is not None:
        print(f"Saved timing CSV: {timing_path}")
    print(f"Saved overview plot PNG: {overview_png}")
    print(f"Saved overview plot SVG: {overview_svg}")
    print(f"Saved sorted metrics PNG: {light_png}")
    print(f"Saved sorted metrics SVG: {light_svg}")
    if timing_png is not None and timing_svg is not None:
        print(f"Saved sorted timing PNG: {timing_png}")
        print(f"Saved sorted timing SVG: {timing_svg}")


if __name__ == "__main__":
    main()
