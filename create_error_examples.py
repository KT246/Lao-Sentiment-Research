from pathlib import Path

import pandas as pd


INPUT_CSV = Path(r"C:\Users\LENOVO\Downloads\res\latex\fp_fn\all_models_fp_fn.csv")
OUTPUT_CSV = Path(r"C:\Users\LENOVO\Downloads\res\latex\fp_fn\paper_error_examples.csv")


SELECTED_TEXTS = [
    ("false_positive", "facebookຊ້າຫລາຍ"),
    ("false_positive", "ສ່ວນຫລຸດໄມ່ມີຈີງຫວະເອົາໄປ1ດາວພໍ"),
    ("false_negative", "ບັງຄັບໃຫ້ໃຊ້ ແຕ່ເຮັດຈັງໃດສີລົງທະບຽນ? ຕອນນີ້ລົງທະບຽນໄດ້ແລ້ວ ຂອບໃຈ"),
    ("false_negative", "ຂອບເຂດສົ່ງ ກ້ວາງໄກລ ສົ່ງໄວ ບໍ່ມີບັນຫາ"),
]


def main():
    df = pd.read_csv(INPUT_CSV)
    fixed_modes = {"baseline", "full-finetuning", "lora"}
    df = df[df["training_mode"].isin(fixed_modes)].copy()

    rows = []
    for error_type, text in SELECTED_TEXTS:
        subset = df[(df["error_type"] == error_type) & (df["text"] == text)].copy()
        if subset.empty:
            continue

        rows.append(
            {
                "error_type": error_type,
                "example_text": text,
                "true_label": int(subset["true_label"].iloc[0]),
                "predicted_label": int(subset["predicted_label"].iloc[0]),
                "n_fixed_split_experiments": int(subset["experiment_name"].nunique()),
                "fixed_split_experiments": ", ".join(sorted(subset["experiment_name"].unique())),
            }
        )

    output_df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved paper-ready error examples to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
