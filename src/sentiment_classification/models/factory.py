from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

from sentiment_classification.utils.config import DEFAULT_LORA_CONFIG


def build_sequence_classification_model(
    model_name: str,
    num_labels: int,
    use_lora: bool = False,
    lora_target_modules=None,
    lora_config=None
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    if not use_lora:
        return model

    merged_lora_config = dict(DEFAULT_LORA_CONFIG)
    if lora_config:
        merged_lora_config.update(lora_config)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=merged_lora_config["r"],
        lora_alpha=merged_lora_config["lora_alpha"],
        lora_dropout=merged_lora_config["lora_dropout"],
        bias=merged_lora_config["bias"],
        target_modules=lora_target_modules or ["query", "key", "value"]
    )

    return get_peft_model(model, peft_config)


def get_trainable_parameter_stats(model):
    trainable_params = 0
    total_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_ratio = 0.0 if total_params == 0 else trainable_params / total_params
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": round(trainable_ratio, 6)
    }
