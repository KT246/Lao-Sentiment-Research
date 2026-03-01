import os
import torch
from torch import nn
import numpy as np
import evaluate
import json
import time
import datetime
from transformers import (
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback, 
    TrainerCallback,
    DataCollatorWithPadding
)

class TimeLogCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        print(f"\n⏱️  Epoch {int(state.epoch) + 1} Start: {datetime.datetime.now().strftime('%H:%M:%S')}")
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"✅ Epoch {int(state.epoch)} End - Duration: {time.time() - self.start:.2f}s")

class EpochLoggingCallback(TrainerCallback):
    """Callback lưu metrics ra file JSON sau mỗi epoch."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            epoch = int(round(state.epoch))
            filepath = os.path.join(self.output_dir, f"epoch_{epoch}_metrics.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)

class WeightedTrainer(Trainer):
    """Custom Trainer xử lý mất cân bằng dữ liệu."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
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

def compute_metrics(eval_pred):
    """Tính F1-Macro và Accuracy."""
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1_macro = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    
    return {"accuracy": accuracy, "f1_macro": f1_macro}

def setup_trainer(
    model_name: str, 
    num_labels: int, 
    train_dataset, 
    eval_dataset, 
    tokenizer, 
    output_dir: str,
    epochs: int = 25,
    batch_size: int = 16,
    class_weights = None
):
    """Khởi tạo Model và Trainer hoàn chỉnh."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )

    # CẤU HÌNH ĐÃ SỬA CHUẨN: eval_strategy thay cho evaluation_strategy
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        fp16=torch.cuda.is_available(), # Tăng tốc Train với GPU
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        push_to_hub=False,
    )

    # Tối ưu RAM cho GPU
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            TimeLogCallback(),
            EpochLoggingCallback(output_dir=output_dir)
        ],
        class_weights=class_weights
    )

    return trainer, model
