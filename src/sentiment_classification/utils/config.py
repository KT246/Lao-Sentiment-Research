MODEL_REGISTRY = {
    "xlm-roberta": {
        "hf_name": "xlm-roberta-base",
        "display_name": "XLM-R",
        "lora_target_modules": ["query", "key", "value"]
    },
    "mbert": {
        "hf_name": "bert-base-multilingual-cased",
        "display_name": "mBERT",
        "lora_target_modules": ["query", "key", "value"]
    },
    "mbert-lao": {
        "hf_name": "w11wo/lao-roberta-base",
        "display_name": "Lao-specific model",
        "lora_target_modules": ["query", "key", "value"]
    }
}


TRAINING_MODE_REGISTRY = {
    "full-finetuning": {
        "use_lora": False,
        "use_cross_validation": False
    },
    "lora": {
        "use_lora": True,
        "use_cross_validation": False
    },
    "cross-validation": {
        "use_lora": False,
        "use_cross_validation": True,
        "num_folds": 3
    }
}


DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "bias": "none"
}
