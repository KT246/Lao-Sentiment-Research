MODEL_REGISTRY = {
    "xlm-roberta": {
        "hf_name": "xlm-roberta-base",
        "display_name": "XLM-R",
        "lora_target_modules": ["query", "key", "value"],
        "tokenizer_name": "xlm-roberta-base",
        "config_name": "xlm-roberta-base",
        "architecture_type": "transformer",
        "init_from_pretrained": True,
        "supports_lora": True,
        "supported_training_modes": ["full-finetuning", "lora", "cross-validation"],
    },
    "mbert": {
        "hf_name": "bert-base-multilingual-cased",
        "display_name": "mBERT",
        "lora_target_modules": ["query", "key", "value"],
        "tokenizer_name": "bert-base-multilingual-cased",
        "config_name": "bert-base-multilingual-cased",
        "architecture_type": "transformer",
        "init_from_pretrained": True,
        "supports_lora": True,
        "supported_training_modes": ["full-finetuning", "lora", "cross-validation"],
    },
    "mbert-lao": {
        "hf_name": "w11wo/lao-roberta-base",
        "display_name": "Lao-specific model",
        "lora_target_modules": ["query", "key", "value"],
        "tokenizer_name": "w11wo/lao-roberta-base",
        "config_name": "w11wo/lao-roberta-base",
        "architecture_type": "transformer",
        "init_from_pretrained": True,
        "supports_lora": True,
        "supported_training_modes": ["full-finetuning", "lora", "cross-validation"],
    },
    "textcnn": {
        "hf_name": "textcnn",
        "display_name": "TextCNN (from scratch)",
        "lora_target_modules": [],
        "tokenizer_name": "xlm-roberta-base",
        "config_name": None,
        "architecture_type": "textcnn",
        "init_from_pretrained": False,
        "supports_lora": False,
        "supported_training_modes": ["from-scratch", "cross-validation"],
        "model_kwargs": {
            "embedding_dim": 128,
            "num_filters": 128,
            "kernel_sizes": [3, 4, 5],
            "dropout": 0.3,
        },
    },
}


TRAINING_MODE_REGISTRY = {
    "full-finetuning": {
        "use_lora": False,
        "use_cross_validation": False,
        "requires_random_init": False,
    },
    "from-scratch": {
        "use_lora": False,
        "use_cross_validation": False,
        "requires_random_init": True,
    },
    "lora": {
        "use_lora": True,
        "use_cross_validation": False,
        "requires_random_init": False,
    },
    "cross-validation": {
        "use_lora": False,
        "use_cross_validation": True,
        "num_folds": 3,
        "requires_random_init": False,
    }
}


DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "bias": "none"
}
