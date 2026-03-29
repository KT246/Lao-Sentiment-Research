import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns


MODEL_DISPLAY = {
    "xlm-roberta": "XLM-RoBERTa",
    "mbert": "mBERT",
    "mbert-lao": "mBERT-Lao",
    "logistic-regression": "Logistic Regression",
    "svm": "SVM",
    "decision-tree": "Decision Tree",
}

SHORT_MODEL_DISPLAY = {
    "xlm-roberta": "XLM-R",
    "mbert": "mBERT",
    "mbert-lao": "mBERT-Lao",
    "logistic-regression": "LogReg",
    "svm": "SVM",
    "decision-tree": "DecTree",
}

MODE_DISPLAY = {
    "baseline": "Baseline",
    "full-finetuning": "Full FT",
    "lora": "LoRA",
    "cross-validation": "3-Fold CV",
}

SHORT_MODE_DISPLAY = {
    "baseline": "Base",
    "full-finetuning": "FT",
    "lora": "LoRA",
    "cross-validation": "CV",
}

MODEL_ORDER = {
    "xlm-roberta": 0,
    "mbert": 1,
    "mbert-lao": 2,
    "logistic-regression": 3,
    "svm": 4,
    "decision-tree": 5,
}

MODE_ORDER = {
    "baseline": 0,
    "full-finetuning": 1,
    "lora": 2,
    "cross-validation": 3,
}

ERROR_COLORS = {
    "false_negative": "#4472C4",
    "false_positive": "#E74C3C",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SVG plots for error analysis.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=r"C:\Users\LENOVO\Downloads\res\latex\fp_fn\all_models_fp_fn.csv",
        help="Path to the combined false-positive/false-negative CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\LENOVO\Downloads\res\latex\figures",
        help="Directory to save SVG plots.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="error_analysis_overview.svg",
        help="Output SVG filename.",
    )
    return parser.parse_args()


def format_experiment_label(row):
    model_name = MODEL_DISPLAY.get(row["model_key"], row["model_key"])
    mode_name = MODE_DISPLAY.get(row["training_mode"], row["training_mode"])
    return f"{model_name} ({mode_name})"


def format_short_experiment_label(row):
    model_name = SHORT_MODEL_DISPLAY.get(row["model_key"], row["model_key"])
    mode_name = SHORT_MODE_DISPLAY.get(row["training_mode"], row["training_mode"])
    return f"{model_name} ({mode_name})"


def format_shared_experiment_label(row):
    model_name = MODEL_DISPLAY.get(row["model_key"], row["model_key"])
    mode_name = SHORT_MODE_DISPLAY.get(row["training_mode"], row["training_mode"])
    return f"{model_name}\n{mode_name}"


def build_summary(df: pd.DataFrame):
    summary = (
        df.groupby(["experiment_name", "model_key", "training_mode", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary["false_negative"] = summary.get("false_negative", 0)
    summary["false_positive"] = summary.get("false_positive", 0)
    summary["total_errors"] = summary["false_negative"] + summary["false_positive"]
    summary["fp_share"] = summary["false_positive"] / summary["total_errors"]
    summary["fn_share"] = summary["false_negative"] / summary["total_errors"]
    summary["model_order"] = summary["model_key"].map(MODEL_ORDER).fillna(999)
    summary["mode_order"] = summary["training_mode"].map(MODE_ORDER).fillna(999)
    summary = summary.sort_values(
        ["total_errors", "model_order", "mode_order", "experiment_name"],
        ascending=[True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    summary["experiment_label"] = summary.apply(format_experiment_label, axis=1)
    summary["short_label"] = summary.apply(format_short_experiment_label, axis=1)
    summary["shared_label"] = summary.apply(format_shared_experiment_label, axis=1)
    return summary


def configure_plot_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 17,
            "axes.titleweight": "normal",
            "axes.labelsize": 14,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
        }
    )


def thousands_formatter(value, _):
    if value >= 1000:
        if value % 1000 == 0:
            return f"{int(value / 1000)}k"
        return f"{value / 1000:.1f}k"
    return f"{int(value)}"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D6DCE5")
    ax.spines["bottom"].set_color("#D6DCE5")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(axis="x", linestyle="-", alpha=0.22, color="#DDE3EC", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#000000")
    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")
    ax.title.set_color("#000000")


def add_count_labels(ax, plot_df: pd.DataFrame):
    max_total = float(plot_df["total_errors"].max())
    for idx, total_errors in enumerate(plot_df["total_errors"]):
        ax.text(
            total_errors + max_total * 0.014,
            idx,
            f"{int(total_errors)}",
            ha="left",
            va="center",
            fontsize=12.25,
            color="#333333",
        )


def add_composition_labels(ax, plot_df: pd.DataFrame, x_positions):
    for x_pos, (_, row) in zip(x_positions, plot_df.iterrows()):
        fp_share = row["fp_share"] * 100
        ax.text(
            x_pos,
            102.4,
            f"FP {fp_share:.1f}%",
            va="bottom",
            ha="center",
            fontsize=11.4,
            color="#000000",
            clip_on=False,
        )


def save_error_analysis_figure(summary: pd.DataFrame, output_path: Path):
    configure_plot_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13.2, 9.2),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.08, right=0.93, top=0.90, bottom=0.12, hspace=0.62)

    ax_counts, ax_comp = axes
    plot_df = summary.copy()
    x_positions = list(range(len(plot_df)))
    bar_width = 0.58

    ax_counts.bar(
        x_positions,
        plot_df["false_negative"],
        width=bar_width,
        color=ERROR_COLORS["false_negative"],
        alpha=0.9,
        label="False Negative",
    )
    ax_counts.bar(
        x_positions,
        plot_df["false_positive"],
        width=bar_width,
        bottom=plot_df["false_negative"],
        color=ERROR_COLORS["false_positive"],
        alpha=0.9,
        label="False Positive",
    )
    ax_counts.set_ylabel("Error count")
    ax_counts.set_title("(a) Error Counts", pad=8)
    ax_counts.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax_counts.set_ylim(0, float(plot_df["total_errors"].max()) * 1.14)

    labels = plot_df["shared_label"].tolist()
    ax_counts.set_xticks(x_positions)
    ax_counts.set_xticklabels(labels, rotation=45, ha="right")
    ax_counts.tick_params(axis="x", labelbottom=True)

    ax_comp.bar(
        x_positions,
        plot_df["fn_share"] * 100,
        width=bar_width,
        color=ERROR_COLORS["false_negative"],
        alpha=0.9,
        label="False Negative",
    )
    ax_comp.bar(
        x_positions,
        plot_df["fp_share"] * 100,
        width=bar_width,
        bottom=plot_df["fn_share"] * 100,
        color=ERROR_COLORS["false_positive"],
        alpha=0.9,
        label="False Positive",
    )
    ax_comp.set_ylabel("Share (%)")
    ax_comp.set_ylim(0, 108)
    ax_comp.set_title("(b) Error Composition", pad=16)
    ax_comp.axhline(50, color="#8F8F8F", linestyle="--", linewidth=0.85, alpha=0.7)
    ax_comp.set_xticks(x_positions)
    ax_comp.set_xticklabels(labels, rotation=45, ha="right")

    style_axis(ax_counts)
    style_axis(ax_comp)
    ax_counts.grid(axis="x", visible=False)
    ax_counts.grid(axis="y", linestyle="-", alpha=0.22, color="#DDE3EC", linewidth=0.9)
    ax_comp.grid(axis="x", visible=False)
    ax_comp.grid(axis="y", linestyle="-", alpha=0.22, color="#DDE3EC", linewidth=0.9)
    add_composition_labels(ax_comp, plot_df, x_positions)

    legend_handles = [
        Patch(facecolor=ERROR_COLORS["false_negative"], edgecolor="none", label="False Negative"),
        Patch(facecolor=ERROR_COLORS["false_positive"], edgecolor="none", label="False Positive"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor="#F7F9FC",
        edgecolor="#DCE3EC",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_error_count_figure(summary: pd.DataFrame, output_path: Path):
    configure_plot_style()
    plot_df = summary.copy()
    x_positions = list(range(len(plot_df)))
    bar_width = 0.58
    fig, ax = plt.subplots(figsize=(8.6, 6.4), constrained_layout=True)

    ax.bar(
        x_positions,
        plot_df["false_negative"],
        width=bar_width,
        color=ERROR_COLORS["false_negative"],
        alpha=0.9,
        label="False Negative",
    )
    ax.bar(
        x_positions,
        plot_df["false_positive"],
        width=bar_width,
        bottom=plot_df["false_negative"],
        color=ERROR_COLORS["false_positive"],
        alpha=0.9,
        label="False Positive",
    )
    ax.set_ylabel("Error count")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["shared_label"], rotation=35, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.set_ylim(0, float(plot_df["total_errors"].max()) * 1.14)
    style_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="-", alpha=0.22, color="#DDE3EC", linewidth=0.9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=True, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_error_composition_figure(summary: pd.DataFrame, output_path: Path):
    configure_plot_style()
    plot_df = summary.copy()
    x_positions = list(range(len(plot_df)))
    bar_width = 0.58
    fig, ax = plt.subplots(figsize=(7.1, 6.4), constrained_layout=True)

    ax.bar(
        x_positions,
        plot_df["fn_share"] * 100,
        width=bar_width,
        color=ERROR_COLORS["false_negative"],
        alpha=0.9,
        label="False Negative",
    )
    ax.bar(
        x_positions,
        plot_df["fp_share"] * 100,
        width=bar_width,
        bottom=plot_df["fn_share"] * 100,
        color=ERROR_COLORS["false_positive"],
        alpha=0.9,
        label="False Positive",
    )
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["shared_label"], rotation=35, ha="right")
    ax.axhline(50, color="#8F8F8F", linestyle="--", linewidth=0.85, alpha=0.7)
    style_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="-", alpha=0.22, color="#DDE3EC", linewidth=0.9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=True, ncol=2)
    add_composition_labels(ax, plot_df, x_positions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name

    df = pd.read_csv(input_csv)
    summary = build_summary(df)

    save_error_analysis_figure(summary, output_path)
    counts_path = output_path.with_name(f"{output_path.stem}_counts.svg")
    composition_path = output_path.with_name(f"{output_path.stem}_composition.svg")
    save_error_count_figure(summary, counts_path)
    save_error_composition_figure(summary, composition_path)

    print(f"Saved SVG figure to: {output_path}")
    print(f"Saved SVG figure to: {counts_path}")
    print(f"Saved SVG figure to: {composition_path}")


if __name__ == "__main__":
    main()
