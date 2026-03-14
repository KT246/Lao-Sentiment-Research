import os
import json
import time
import datetime
import torch
from torch import nn
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorWithPadding
)
from transformers.trainer_callback import PrinterCallback

from sentiment_classification.models.factory import build_sequence_classification_model


class EpochTimingCallback(TrainerCallback):
    """Track epoch durations and keep a compact runtime summary."""
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time is None:
            return

        duration = round(time.time() - self.epoch_start_time, 2)
        self.epoch_times.append(duration)
        epoch = int(round(state.epoch)) if state.epoch is not None else len(self.epoch_times)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"{timestamp} | epoch={epoch} | train_time={duration:.2f}s")

    def build_summary(self):
        total_time_seconds = round(sum(self.epoch_times), 2)
        average_epoch_time_seconds = round(total_time_seconds / len(self.epoch_times), 2) if self.epoch_times else 0.0
        return {
            "epoch_times_seconds": self.epoch_times,
            "average_epoch_time_seconds": average_epoch_time_seconds,
            "epochs_completed": len(self.epoch_times),
            "total_training_time_seconds": total_time_seconds
        }


class EpochLoggingCallback(TrainerCallback):
    """Save runtime-only logs after each epoch."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        payload = {
            "epoch": int(round(state.epoch)) if state.epoch is not None else None,
            "global_step": state.global_step
        }
        filepath = os.path.join(self.output_dir, f"epoch_{payload['epoch']}_runtime.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)


class WeightedTrainer(Trainer):
    """Trainer that supports class weights for imbalanced data."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is not None:
            if self.class_weights is not None:
                self.class_weights = self.class_weights.to(model.device)
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.get("loss")

        return (loss, outputs) if return_outputs else loss


def setup_trainer(
    model_name: str,
    num_labels: int,
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    epochs: int = 25,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 1e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    save_strategy: str = "no",
    seed: int = 42,
    class_weights=None,
    use_lora: bool = False,
    lora_target_modules=None,
    lora_config=None
):
    timing_callback = EpochTimingCallback()
    has_eval_dataset = eval_dataset is not None
    effective_save_strategy = save_strategy if has_eval_dataset and save_strategy != "no" else ("epoch" if has_eval_dataset else "no")

    model = build_sequence_classification_model(
        model_name=model_name,
        num_labels=num_labels,
        use_lora=use_lora,
        lora_target_modules=lora_target_modules,
        lora_config=lora_config
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch" if has_eval_dataset else "no",
        save_strategy=effective_save_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        load_best_model_at_end=has_eval_dataset and effective_save_strategy != "no",
        metric_for_best_model="eval_loss" if has_eval_dataset else None,
        greater_is_better=False if has_eval_dataset else None,
        save_total_limit=1,
        seed=seed,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",
        report_to="none",
        disable_tqdm=True,
        push_to_hub=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[timing_callback, EpochLoggingCallback(output_dir=output_dir)],
        class_weights=class_weights
    )

    try:
        trainer.remove_callback(PrinterCallback)
    except Exception:
        pass

    return trainer, model, timing_callback
