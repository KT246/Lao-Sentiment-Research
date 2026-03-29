from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd


INPUT_CSV = Path(r"C:\Users\LENOVO\Downloads\res\latex\fp_fn\paper_error_examples.csv")
OUTPUT_SVG = Path(r"C:\Users\LENOVO\Downloads\res\latex\figures\error_examples_table.svg")
LAO_FONT = Path(r"C:\Windows\Fonts\LaoUI.ttf")


def get_font(size: int, weight: str = "normal"):
    if LAO_FONT.exists():
        return FontProperties(fname=str(LAO_FONT), size=size, weight=weight)
    return FontProperties(size=size, weight=weight)


def wrap_text(text: str, width: int = 26):
    return fill(str(text), width=width, break_long_words=False, break_on_hyphens=False)


def main():
    df = pd.read_csv(INPUT_CSV)
    fp_rows = df[df["error_type"] == "false_positive"].reset_index(drop=True)
    fn_rows = df[df["error_type"] == "false_negative"].reset_index(drop=True)

    num_rows = max(len(fp_rows), len(fn_rows))
    fig_height = 2.6 + num_rows * 1.1
    fig, ax = plt.subplots(figsize=(12.6, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title_font = FontProperties(family="serif", size=16, weight="bold")
    serif_header = FontProperties(family="serif", size=12, weight="bold")
    lao_font = get_font(11)
    lao_header = get_font(12, "bold")
    label_font = FontProperties(family="serif", size=11, weight="bold")
    note_font = FontProperties(family="serif", size=10)

    left_x = 0.04
    right_x = 0.53
    col_example_width = 0.37
    col_label_x_left = 0.45
    col_label_x_right = 0.94

    ax.text(0.5, 0.95, "Representative Examples of Fixed-Split Error Patterns", ha="center", va="center", fontproperties=title_font)
    ax.text(0.5, 0.905, "Examples are drawn directly from the aggregated false-positive and false-negative file across the nine fixed-split experiments.", ha="center", va="center", fontproperties=note_font)

    top_y = 0.84
    ax.hlines(top_y, 0.04, 0.96, colors="black", linewidth=1.2)
    ax.text(0.24, top_y - 0.035, "False Positive", ha="center", va="center", fontproperties=serif_header)
    ax.text(0.73, top_y - 0.035, "False Negative", ha="center", va="center", fontproperties=serif_header)

    header_y = top_y - 0.08
    ax.hlines(header_y, 0.04, 0.96, colors="black", linewidth=0.8)
    ax.text(left_x, header_y - 0.03, "Example", ha="left", va="center", fontproperties=lao_header)
    ax.text(col_label_x_left, header_y - 0.03, "Label", ha="center", va="center", fontproperties=label_font)
    ax.text(right_x, header_y - 0.03, "Example", ha="left", va="center", fontproperties=lao_header)
    ax.text(col_label_x_right, header_y - 0.03, "Label", ha="center", va="center", fontproperties=label_font)
    ax.hlines(header_y - 0.06, 0.04, 0.96, colors="black", linewidth=0.8)

    start_y = header_y - 0.12
    row_gap = 0.19

    for idx in range(num_rows):
        y = start_y - idx * row_gap

        if idx < len(fp_rows):
            fp_text = wrap_text(fp_rows.loc[idx, "example_text"], width=24)
            ax.text(left_x, y, fp_text, ha="left", va="top", fontproperties=lao_font)
            ax.text(col_label_x_left, y - 0.01, str(int(fp_rows.loc[idx, "true_label"])), ha="center", va="top", fontproperties=label_font)

        if idx < len(fn_rows):
            fn_text = wrap_text(fn_rows.loc[idx, "example_text"], width=28)
            ax.text(right_x, y, fn_text, ha="left", va="top", fontproperties=lao_font)
            ax.text(col_label_x_right, y - 0.01, str(int(fn_rows.loc[idx, "true_label"])), ha="center", va="top", fontproperties=label_font)

    bottom_y = start_y - (num_rows - 1) * row_gap - 0.16
    ax.hlines(bottom_y, 0.04, 0.96, colors="black", linewidth=1.0)

    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT_SVG.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SVG table to: {OUTPUT_SVG}")
    print(f"Saved PDF table to: {OUTPUT_SVG.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
