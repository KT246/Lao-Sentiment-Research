import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def load_and_prepare_data(data_dir: str, tokenizer_name: str, max_length: int = 128) -> DatasetDict:
    """
    Loads train and val CSV files from data_dir and tokenizes them.
    Expects CSVs to have 'text' and 'label' columns.
    """
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing train.csv or val.csv in {data_dir}. Please add your data.")

    # Load pandas dataframes
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Convert to HuggingFace Dataset
    hf_train = Dataset.from_pandas(df_train)
    hf_val = Dataset.from_pandas(df_val)
    
    dataset = DatasetDict({
        "train": hf_train,
        "validation": hf_val
    })

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_length
        )

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Format for PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets, tokenizer
