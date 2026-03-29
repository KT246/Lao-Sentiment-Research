import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm
from wordcloud import WordCloud


FIXED_MODES = {"baseline", "full-finetuning", "lora"}
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z\u0E80-\u0EFF]+", re.UNICODE)
PROJECT_FONT_CANDIDATES = [
    Path(r"C:\Users\LENOVO\Downloads\res\fonts\saysettha_ot.ttf"),
]
WINDOWS_FONT_DIR = Path(r"C:\Windows\Fonts")
PREFERRED_FONT_PATTERNS = [
    "Saysettha*.ttf",
    "Saysettha*.otf",
    "Phetsarath*.ttf",
    "Phetsarath*.otf",
    "LaoUIb.ttf",
    "LaoUI.ttf",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract overlap-fail FP/FN samples and generate top-word statistics + word clouds."
    )
    parser.add_argument(
        "--input_csv",
        default=r"C:\Users\LENOVO\Downloads\res\latex\fp_fn\all_models_fp_fn.csv",
        help="Combined FP/FN CSV exported from trained experiments.",
    )
    parser.add_argument(
        "--output_csv_dir",
        default=r"C:\Users\LENOVO\Downloads\res\latex\fp_fn",
        help="Directory for extracted overlap CSV files.",
    )
    parser.add_argument(
        "--output_fig_dir",
        default=r"C:\Users\LENOVO\Downloads\res\latex\figures",
        help="Directory for SVG/PDF figures.",
    )
    return parser.parse_args()


def load_overlap_fail(df: pd.DataFrame):
    fixed_df = df[df["training_mode"].isin(FIXED_MODES)].copy()
    total_experiments = fixed_df["experiment_name"].nunique()

    grouped = (
        fixed_df.groupby(["error_type", "text"], dropna=False)
        .agg(
            n_experiments=("experiment_name", "nunique"),
            occurrences=("experiment_name", "size"),
            model_list=("experiment_name", lambda x: ", ".join(sorted(set(x)))),
            true_label=("true_label", "first"),
            predicted_label=("predicted_label", "first"),
        )
        .reset_index()
    )

    overlap = grouped[grouped["n_experiments"] == total_experiments].copy()
    overlap = overlap.sort_values(["error_type", "occurrences", "text"], ascending=[True, False, True]).reset_index(drop=True)

    overlap_rows = fixed_df.merge(
        overlap[["error_type", "text"]],
        on=["error_type", "text"],
        how="inner",
    ).copy()

    return fixed_df, overlap, overlap_rows, total_experiments


def build_ranked_error_texts(fixed_df: pd.DataFrame):
    ranked = (
        fixed_df.assign(char_len=fixed_df["text"].fillna("").astype(str).str.len())
        .groupby(["error_type", "text"], dropna=False)
        .agg(
            n_experiments=("experiment_name", "nunique"),
            occurrences=("experiment_name", "size"),
            char_len=("char_len", "first"),
            true_label=("true_label", "first"),
            predicted_label=("predicted_label", "first"),
        )
        .reset_index()
        .sort_values(
            ["error_type", "n_experiments", "occurrences", "char_len", "text"],
            ascending=[True, False, False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return ranked


def select_top_short_texts(ranked_df: pd.DataFrame, error_type: str, top_n: int = 30):
    subset = ranked_df[ranked_df["error_type"] == error_type].copy()
    char_limits = [18, 24, 30, 36, 42, 48, 60, 80]
    selected = subset
    used_limit = None
    for limit in char_limits:
        candidate = subset[subset["char_len"] <= limit].copy()
        if len(candidate) >= top_n:
            selected = candidate.head(top_n).copy()
            used_limit = limit
            break
    else:
        selected = subset.head(top_n).copy()
        if not selected.empty:
            used_limit = int(selected["char_len"].max())
        else:
            used_limit = None

    selected["rank"] = range(1, len(selected) + 1)
    selected["length_limit_used"] = used_limit
    return selected


def tokenize(text: str):
    if pd.isna(text):
        return []
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text)) if token.strip()]


def build_top_words(overlap_rows: pd.DataFrame, top_n: int = 30):
    records = []
    counters = {}
    for error_type in ["false_positive", "false_negative"]:
        subset = overlap_rows[overlap_rows["error_type"] == error_type]
        counter = Counter()
        for text in subset["text"].dropna():
            counter.update(tokenize(text))
        counters[error_type] = counter
        for rank, (token, count) in enumerate(counter.most_common(top_n), start=1):
            records.append(
                {
                    "error_type": error_type,
                    "rank": rank,
                    "token": token,
                    "count": count,
                }
            )
    return pd.DataFrame(records), counters


def apply_scientific_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def get_lao_font():
    for font_path in PROJECT_FONT_CANDIDATES:
        if font_path.exists():
            return fm.FontProperties(fname=str(font_path))
    for pattern in PREFERRED_FONT_PATTERNS:
        for font_path in WINDOWS_FONT_DIR.glob(pattern):
            if font_path.exists():
                return fm.FontProperties(fname=str(font_path))
    return None


def get_lao_font_path():
    for font_path in PROJECT_FONT_CANDIDATES:
        if font_path.exists():
            return str(font_path)
    for pattern in PREFERRED_FONT_PATTERNS:
        for font_path in WINDOWS_FONT_DIR.glob(pattern):
            if font_path.exists():
                return str(font_path)
    return None


def draw_top_words(top_words: pd.DataFrame, output_path: Path):
    apply_scientific_style()
    lao_font = get_lao_font()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.5), constrained_layout=True)
    colors = {
        "false_positive": "#D95F5F",
        "false_negative": "#4C78A8",
    }
    titles = {
        "false_positive": "(a) Top 30 words in overlap false positives",
        "false_negative": "(b) Top 30 words in overlap false negatives",
    }

    for ax, error_type in zip(axes, ["false_positive", "false_negative"]):
        subset = top_words[top_words["error_type"] == error_type].sort_values("count", ascending=True)
        ax.barh(subset["token"], subset["count"], color=colors[error_type], alpha=0.88)
        ax.set_title(titles[error_type], fontweight="bold")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.grid(axis="y", visible=False)
        if lao_font is not None:
            for label in ax.get_yticklabels():
                label.set_fontproperties(lao_font)
        for value, y in zip(subset["count"], range(len(subset))):
            ax.text(value + 0.05, y, str(value), va="center", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Most Frequent Tokens in Overlap-Fail Samples",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def _cloud_positions(n_items: int):
    base_positions = [
        (0.50, 0.58), (0.34, 0.70), (0.66, 0.70), (0.28, 0.48), (0.72, 0.48),
        (0.50, 0.34), (0.18, 0.62), (0.82, 0.62), (0.20, 0.32), (0.80, 0.32),
        (0.10, 0.78), (0.90, 0.78), (0.12, 0.18), (0.88, 0.18), (0.50, 0.82),
        (0.36, 0.18), (0.64, 0.18), (0.06, 0.50), (0.94, 0.50), (0.32, 0.86),
        (0.68, 0.86), (0.24, 0.08), (0.76, 0.08), (0.42, 0.50), (0.58, 0.50),
        (0.38, 0.62), (0.62, 0.62), (0.38, 0.26), (0.62, 0.26), (0.50, 0.10),
    ]
    return base_positions[:n_items]


def draw_word_cloud(counters: dict[str, Counter], output_path: Path):
    apply_scientific_style()
    lao_font = get_lao_font()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), constrained_layout=True)
    palettes = {
        "false_positive": ["#7F1D1D", "#991B1B", "#B91C1C", "#C2410C", "#9A3412"],
        "false_negative": ["#1E3A8A", "#1D4ED8", "#1D4E89", "#1E40AF", "#1F4E79"],
    }
    titles = {
        "false_positive": "(a) Overlap false-positive word cloud",
        "false_negative": "(b) Overlap false-negative word cloud",
    }

    for ax, error_type in zip(axes, ["false_positive", "false_negative"]):
        most_common = counters[error_type].most_common(30)
        if not most_common:
            ax.axis("off")
            ax.set_title(titles[error_type], fontweight="bold")
            continue

        counts = [count for _, count in most_common]
        min_count = min(counts)
        max_count = max(counts)
        positions = _cloud_positions(len(most_common))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(titles[error_type], fontweight="bold")

        for idx, ((token, count), (x, y)) in enumerate(zip(most_common, positions)):
            if max_count == min_count:
                size = 16
            else:
                size = 12 + (count - min_count) * (24 / (max_count - min_count))
            color = palettes[error_type][idx % len(palettes[error_type])]
            ax.text(
                x,
                y,
                token,
                fontsize=size,
                color=color,
                ha="center",
                va="center",
                alpha=0.95,
                fontproperties=lao_font,
                transform=ax.transAxes,
            )

    fig.suptitle(
        "Word Clouds from Overlap-Fail Samples",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_phrase_frequencies(selected_df: pd.DataFrame):
    if selected_df.empty:
        return {}
    max_rank = int(selected_df["rank"].max())
    freqs = {}
    for row in selected_df.itertuples(index=False):
        score = int(row.n_experiments * 100 + row.occurrences * 10 + (max_rank - row.rank + 1))
        freqs[str(row.text)] = max(score, 1)
    return freqs


def make_palette(colors):
    def _color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        idx = abs(hash(word)) % len(colors)
        return colors[idx]

    return _color_func


def draw_phrase_wordcloud(fp_selected: pd.DataFrame, fn_selected: pd.DataFrame, output_path: Path):
    apply_scientific_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.4), constrained_layout=True)
    font_path = get_lao_font_path()

    configs = [
        (
            axes[0],
            fp_selected,
            ["#9A3412", "#C2410C", "#EA580C", "#B45309", "#7C2D12"],
            "(a) False Positives",
        ),
        (
            axes[1],
            fn_selected,
            ["#9D174D", "#BE185D", "#DB2777", "#C026D3", "#A21CAF"],
            "(b) False Negatives",
        ),
    ]

    for ax, selected_df, palette, title in configs:
        frequencies = build_phrase_frequencies(selected_df)
        ax.axis("off")
        if not frequencies:
            ax.set_title(title, fontsize=12, fontweight="bold")
            continue

        wc = WordCloud(
            width=1200,
            height=700,
            background_color="white",
            font_path=font_path,
            prefer_horizontal=1.0,
            collocations=False,
            max_words=30,
            margin=4,
            relative_scaling=0.45,
            random_state=42,
        ).generate_from_frequencies(frequencies)

        wc.recolor(color_func=make_palette(palette), random_state=42)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def draw_single_phrase_wordcloud(selected_df: pd.DataFrame, output_path: Path, palette: list[str]):
    apply_scientific_style()
    font_path = get_lao_font_path()
    frequencies = build_phrase_frequencies(selected_df)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2), constrained_layout=True)
    ax.axis("off")

    if frequencies:
        wc = WordCloud(
            width=1100,
            height=700,
            background_color="white",
            font_path=font_path,
            prefer_horizontal=1.0,
            collocations=False,
            max_words=30,
            margin=4,
            relative_scaling=0.45,
            random_state=42,
        ).generate_from_frequencies(frequencies)
        wc.recolor(color_func=make_palette(palette), random_state=42)
        ax.imshow(wc, interpolation="bilinear")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv_dir = Path(args.output_csv_dir)
    output_fig_dir = Path(args.output_fig_dir)

    df = pd.read_csv(input_csv)
    _, overlap, overlap_rows, total_experiments = load_overlap_fail(df)
    ranked = build_ranked_error_texts(df[df["training_mode"].isin(FIXED_MODES)].copy())
    fp_selected = select_top_short_texts(ranked, "false_positive", top_n=30)
    fn_selected = select_top_short_texts(ranked, "false_negative", top_n=30)
    top_words, counters = build_top_words(overlap_rows, top_n=30)

    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    overlap.to_csv(output_csv_dir / "overlap_fail_all_fixed_split.csv", index=False, encoding="utf-8-sig")
    overlap_rows.to_csv(output_csv_dir / "overlap_fail_all_fixed_split_rows.csv", index=False, encoding="utf-8-sig")
    overlap[overlap["error_type"] == "false_positive"].to_csv(
        output_csv_dir / "overlap_fail_false_positive.csv", index=False, encoding="utf-8-sig"
    )
    overlap[overlap["error_type"] == "false_negative"].to_csv(
        output_csv_dir / "overlap_fail_false_negative.csv", index=False, encoding="utf-8-sig"
    )
    top_words.to_csv(output_csv_dir / "overlap_fail_top30_words.csv", index=False, encoding="utf-8-sig")
    fp_selected.to_csv(output_csv_dir / "short_fail_false_positive_top30.csv", index=False, encoding="utf-8-sig")
    fn_selected.to_csv(output_csv_dir / "short_fail_false_negative_top30.csv", index=False, encoding="utf-8-sig")

    draw_top_words(top_words, output_fig_dir / "overlap_fail_top30_words.svg")
    draw_word_cloud(counters, output_fig_dir / "overlap_fail_wordcloud.svg")
    draw_phrase_wordcloud(fp_selected, fn_selected, output_fig_dir / "short_fail_wordcloud.svg")
    draw_single_phrase_wordcloud(
        fp_selected,
        output_fig_dir / "short_fail_wordcloud_fp.svg",
        ["#9A3412", "#C2410C", "#EA580C", "#B45309", "#7C2D12"],
    )
    draw_single_phrase_wordcloud(
        fn_selected,
        output_fig_dir / "short_fail_wordcloud_fn.svg",
        ["#9D174D", "#BE185D", "#DB2777", "#C026D3", "#A21CAF"],
    )

    print(f"Total fixed-split experiments considered: {total_experiments}")
    print(f"Saved overlap fail CSVs to: {output_csv_dir}")
    print(f"Saved figures to: {output_fig_dir}")


if __name__ == "__main__":
    main()
