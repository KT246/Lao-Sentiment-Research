from transformers import AutoConfig, AutoModelForSequenceClassification

from sentiment_classification.models.textcnn import TextCNNConfig, TextCNNForSequenceClassification
from sentiment_classification.utils.config import DEFAULT_LORA_CONFIG


def build_sequence_classification_model(
    model_name: str,
    num_labels: int,
    use_lora: bool = False,
    lora_target_modules=None,
    lora_config=None,
    init_from_pretrained: bool = True,
    config_name: str = None,
    architecture_type: str = "transformer",
    model_kwargs=None,
    vocab_size: int = None,
    pad_token_id: int = 0,
):
    merged_model_kwargs = dict(model_kwargs or {})

    if architecture_type == "textcnn":
        model_config = TextCNNConfig(
            vocab_size=vocab_size or merged_model_kwargs.pop("vocab_size", 32000),
            pad_token_id=pad_token_id,
            num_labels=num_labels,
            **merged_model_kwargs,
        )
        return TextCNNForSequenceClassification(model_config)

    if init_from_pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    else:
        resolved_config_name = config_name or model_name
        model_config = AutoConfig.from_pretrained(
            resolved_config_name,
            num_labels=num_labels
        )
        model = AutoModelForSequenceClassification.from_config(model_config)

    if not use_lora:
        return model

    merged_lora_config = dict(DEFAULT_LORA_CONFIG)
    if lora_config:
        merged_lora_config.update(lora_config)

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:
        raise RuntimeError(
            "LoRA mode requires a healthy PEFT/Transformers environment. "
            "Please reinstall compatible torch/torchvision/transformers/peft in Colab."
        ) from exc

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
