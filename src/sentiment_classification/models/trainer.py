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
from transformers.trainer_callback import PrinterCallback

class TerminalTableCallback(TrainerCallback):
    """Callback tạo bảng Log đẹp mắt trên Terminal."""
    def __init__(self):
        self.start_time = None
        self.start_str = ""
        self.metrics_cache = {}
        self.header_printed = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.start_str = datetime.datetime.now().strftime('%H:%M:%S')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        self.metrics_cache.update(logs)
        
        # Đợi đến khi có eval_loss (khi evaluate ở cuối epoch) mới in table row
        if 'eval_loss' in logs:
            if not self.header_printed:
                print(f"\n{'Epoch':>5} | {'Training Loss':>13} | {'Validation Loss':>15} | {'Accuracy':>8} | {'F1 Macro':>8} | {'Start':>8} | {'End - Duration':>14}")
                print("-" * 95)
                self.header_printed = True
            
            epoch = int(round(state.epoch))
            t_loss = self.metrics_cache.get('loss', 0.0)
            v_loss = logs.get('eval_loss', 0.0)
            acc = logs.get('eval_accuracy', 0.0)
            f1 = logs.get('eval_f1_macro', 0.0)
            
            duration = time.time() - self.start_time if self.start_time else 0.0
            
            print(f"{epoch:5d} | {t_loss:13.6f} | {v_loss:15.6f} | {acc:8.6f} | {f1:8.6f} | {self.start_str:>8} | {duration:13.2f}s")
            
            self.metrics_cache = {}

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

# Tải metrics 1 lần duy nhất ở ngoài (global) để không in script download liên tục
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Tính F1-Macro và Accuracy."""
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

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,               # 🟢 ĐÃ GIẢM: Tốc độ học chậm lại để chống vẹt
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.05,                # 🟢 ĐÃ TĂNG: Phạt nặng hơn nếu model cố tình học vẹt
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        fp16=torch.cuda.is_available(), 
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch",  
        report_to="none",          
        disable_tqdm=True,         
        push_to_hub=False,
    )

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
            TerminalTableCallback(),
            EarlyStoppingCallback(early_stopping_patience=7), # 🟢 ĐÃ THÊM: Dừng sớm nếu 7 vòng không tốt lên
            EpochLoggingCallback(output_dir=output_dir)
        ],
        class_weights=class_weights
    )

    # Ẩn luồng log JSON mặc định của Hugging Face
    try:
        trainer.remove_callback(PrinterCallback)
    except Exception:
        pass

    return trainer, model