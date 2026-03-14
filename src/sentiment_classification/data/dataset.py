import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


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


def build_tokenized_dataset_dict(
    train_df,
    val_df,
    tokenizer_name: str,
    max_length: int = 128,
    padding: str = "max_length"
) -> DatasetDict:
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True))
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=max_length
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets, tokenizer


def load_and_prepare_data(
    data_dir: str,
    tokenizer_name: str,
    max_length: int = 128,
    padding: str = "max_length",
    dropna: bool = True
) -> DatasetDict:
    train_df, val_df = load_train_val_dataframes(data_dir=data_dir, dropna=dropna)
    return build_tokenized_dataset_dict(
        train_df=train_df,
        val_df=val_df,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        padding=padding
    )
