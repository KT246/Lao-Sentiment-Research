import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from sentiment_classification.data.dataset import (
    build_tokenized_dataset_dict,
    load_train_val_dataframes,
)
from sentiment_classification.models.factory import get_trainable_parameter_stats
from sentiment_classification.models.trainer import setup_trainer
from sentiment_classification.utils.config import MODEL_REGISTRY, TRAINING_MODE_REGISTRY
from sentiment_classification.utils.utils import get_hardware_info, save_json


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Lao Sentiment Analysis experiments")
    parser.add_argument("--model_key", type=str, default="xlm-roberta", choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--training_mode", type=str, default="full-finetuning", choices=sorted(TRAINING_MODE_REGISTRY.keys()))
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs/experiment")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--padding", type=str, default="max_length")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--dropna", action="store_true", default=True)
    parser.add_argument("--no-dropna", dest="dropna", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_folds", type=int, default=3)
    parser.add_argument("--cv_include_val", action="store_true", help="Use train+val for cross-validation pool")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    return parser.parse_args()


def compute_class_weights_tensor(labels, num_labels: int):
    classes_present = np.unique(labels)
    weights_present = compute_class_weight(class_weight="balanced", classes=classes_present, y=labels)
    weight_map = {int(cls): float(weight) for cls, weight in zip(classes_present, weights_present)}
    full_weights = [weight_map.get(label_id, 1.0) for label_id in range(num_labels)]
    return torch.tensor(full_weights, dtype=torch.float32)


def validate_experiment_configuration(args, model_config, mode_config):
    init_from_pretrained = model_config.get("init_from_pretrained", True)
    supports_lora = model_config.get("supports_lora", True)

    if mode_config.get("requires_random_init", False) and init_from_pretrained:
        raise ValueError(
            f"Training mode '{args.training_mode}' requires a randomly initialized model, "
            f"but model '{args.model_key}' is configured to load pretrained weights."
        )

    if mode_config["use_lora"] and not supports_lora:
        raise ValueError(
            f"Model '{args.model_key}' does not support LoRA mode because it is configured "
            "for from-scratch training. Use 'from-scratch' or 'cross-validation' instead."
        )


def build_prediction_dataframe(source_df, predictions_output, extra_columns=None):
    logits = predictions_output.predictions
    predicted_labels = np.argmax(logits, axis=1)
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
    confidence_scores = np.max(probabilities, axis=1)

    result_df = source_df.reset_index(drop=True).copy()
    result_df["true_label"] = result_df["label"]
    result_df["predicted_label"] = predicted_labels
    result_df["confidence_score"] = confidence_scores

    if extra_columns:
        for key, value in extra_columns.items():
            result_df[key] = value

    return result_df


def save_best_model_artifacts(trainer, output_dir):
    best_model_dir = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_dir)
    save_json(
        {
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric_eval_loss": trainer.state.best_metric
        },
        os.path.join(output_dir, "best_model_info.json")
    )


def prepare_runtime_metadata(args, model_config, mode_config):
    init_from_pretrained = model_config.get("init_from_pretrained", True)
    tokenizer_name = model_config.get("tokenizer_name", model_config["hf_name"])
    config_name = model_config.get("config_name", model_config["hf_name"])

    metadata = {
        "model_key": args.model_key,
        "model_name": model_config["hf_name"],
        "display_name": model_config.get("display_name"),
        "training_mode": args.training_mode,
        "use_lora": mode_config["use_lora"],
        "use_cross_validation": mode_config["use_cross_validation"],
        "initialization_strategy": "pretrained" if init_from_pretrained else "scratch",
        "tokenizer_name": tokenizer_name,
        "config_name": config_name,
        "pretrained_checkpoint": model_config["hf_name"] if init_from_pretrained else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_length": args.max_length,
        "padding": args.padding,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "num_folds": args.num_folds if mode_config["use_cross_validation"] else 1,
        "cv_include_val": args.cv_include_val if mode_config["use_cross_validation"] else False,
        "seed": args.seed
    }

    if mode_config["use_lora"]:
        metadata["lora"] = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout
        }

    return metadata


def run_single_training(train_df, val_df, args, model_config, mode_config, output_dir):
    tokenized_datasets, tokenizer = build_tokenized_dataset_dict(
        train_df=train_df,
        val_df=val_df,
        tokenizer_name=model_config.get("tokenizer_name", model_config["hf_name"]),
        max_length=args.max_length,
        padding=args.padding
    )

    class_weights_tensor = compute_class_weights_tensor(train_df["label"].to_numpy(), args.num_labels)
    trainer, model, timing_callback = setup_trainer(
        model_name=model_config["hf_name"],
        num_labels=args.num_labels,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_strategy=args.save_strategy,
        seed=args.seed,
        class_weights=class_weights_tensor,
        use_lora=mode_config["use_lora"],
        lora_target_modules=model_config["lora_target_modules"],
        lora_config={
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout
        },
        init_from_pretrained=model_config.get("init_from_pretrained", True),
        config_name=model_config.get("config_name", model_config["hf_name"])
    )

    trainer.train()
    val_df.to_csv(os.path.join(output_dir, "validation_data.csv"), index=False)
    prediction_output = trainer.predict(tokenized_datasets["validation"])
    prediction_df = build_prediction_dataframe(val_df, prediction_output)
    prediction_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    save_best_model_artifacts(trainer, output_dir)

    save_json(timing_callback.build_summary(), os.path.join(output_dir, "timing_metrics.json"))
    save_json(get_trainable_parameter_stats(model), os.path.join(output_dir, "trainable_params.json"))


def run_cross_validation(full_df, args, model_config, mode_config, output_dir):
    splitter = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_prediction_frames = []
    fold_timing_summaries = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(full_df["text"], full_df["label"]), start=1):
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_index}")
        os.makedirs(fold_output_dir, exist_ok=True)

        fold_train_df = full_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = full_df.iloc[val_idx].reset_index(drop=True)

        tokenized_datasets, tokenizer = build_tokenized_dataset_dict(
            train_df=fold_train_df,
            val_df=fold_val_df,
            tokenizer_name=model_config.get("tokenizer_name", model_config["hf_name"]),
            max_length=args.max_length,
            padding=args.padding
        )

        class_weights_tensor = compute_class_weights_tensor(fold_train_df["label"].to_numpy(), args.num_labels)
        trainer, model, timing_callback = setup_trainer(
            model_name=model_config["hf_name"],
            num_labels=args.num_labels,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            output_dir=fold_output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            save_strategy=args.save_strategy,
            seed=args.seed + fold_index,
            class_weights=class_weights_tensor,
            use_lora=mode_config["use_lora"],
            lora_target_modules=model_config["lora_target_modules"],
            lora_config={
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout
            },
            init_from_pretrained=model_config.get("init_from_pretrained", True),
            config_name=model_config.get("config_name", model_config["hf_name"])
        )

        trainer.train()
        fold_val_df.to_csv(os.path.join(fold_output_dir, "validation_data.csv"), index=False)
        prediction_output = trainer.predict(tokenized_datasets["validation"])
        fold_prediction_df = build_prediction_dataframe(
            fold_val_df,
            prediction_output,
            extra_columns={
                "fold": fold_index,
                "source_index": list(val_idx)
            }
        )
        fold_prediction_df.to_csv(os.path.join(fold_output_dir, "predictions.csv"), index=False)
        fold_prediction_frames.append(fold_prediction_df)

        timing_summary = timing_callback.build_summary()
        fold_timing_summaries.append({"fold": fold_index, **timing_summary})
        save_json(timing_summary, os.path.join(fold_output_dir, "timing_metrics.json"))
        save_json(get_trainable_parameter_stats(model), os.path.join(fold_output_dir, "trainable_params.json"))
        save_best_model_artifacts(trainer, fold_output_dir)

    combined_predictions = pd.concat(fold_prediction_frames, ignore_index=True)
    combined_predictions.to_csv(os.path.join(output_dir, "cross_validation_predictions.csv"), index=False)

    all_epoch_times = []
    for summary in fold_timing_summaries:
        all_epoch_times.extend(summary["epoch_times_seconds"])

    cross_validation_timing = {
        "num_folds": args.num_folds,
        "folds": fold_timing_summaries,
        "average_epoch_time_seconds": round(sum(all_epoch_times) / len(all_epoch_times), 2) if all_epoch_times else 0.0,
        "total_training_time_seconds": round(sum(all_epoch_times), 2)
    }
    save_json(cross_validation_timing, os.path.join(output_dir, "timing_metrics.json"))


def main():
    args = parse_args()
    model_config = MODEL_REGISTRY[args.model_key]
    mode_config = TRAINING_MODE_REGISTRY[args.training_mode]
    validate_experiment_configuration(args, model_config, mode_config)

    initialization_strategy = "pretrained" if model_config.get("init_from_pretrained", True) else "scratch"

    os.makedirs(args.output_dir, exist_ok=True)
    save_json(get_hardware_info(), os.path.join(args.output_dir, "hardware_metrics.json"))
    save_json(
        prepare_runtime_metadata(args, model_config, mode_config),
        os.path.join(args.output_dir, "experiment_config.json")
    )

    train_df, val_df = load_train_val_dataframes(data_dir=args.data_dir, dropna=args.dropna)
    logger.info(
        "Running experiment model=%s mode=%s train_rows=%s val_rows=%s",
        args.model_key,
        args.training_mode,
        len(train_df),
        len(val_df)
    )
    logger.info("Model initialization strategy=%s", initialization_strategy)

    if mode_config["use_cross_validation"]:
        full_df = pd.concat([train_df, val_df], ignore_index=True) if args.cv_include_val else train_df.copy()
        run_cross_validation(
            full_df=full_df,
            args=args,
            model_config=model_config,
            mode_config=mode_config,
            output_dir=args.output_dir
        )
    else:
        run_single_training(
            train_df=train_df,
            val_df=val_df,
            args=args,
            model_config=model_config,
            mode_config=mode_config,
            output_dir=args.output_dir
        )

    logger.info("Experiment completed successfully.")


if __name__ == "__main__":
    main()
