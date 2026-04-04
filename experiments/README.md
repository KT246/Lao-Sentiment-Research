# Experiment Branch Plan

This folder maps the experiment strategy into branch-ready run scripts.

Suggested branches:
- `experiment/xlm-roberta-finetuning`
- `experiment/xlm-roberta-lora`
- `experiment/xlm-roberta-cross-validation`
- `experiment/textcnn-training`
- `experiment/textcnn-cross-validation`
- `experiment/mbert-finetuning`
- `experiment/mbert-lora`
- `experiment/mbert-cross-validation`
- `experiment/mbert-lao-finetuning`
- `experiment/mbert-lao-lora`
- `experiment/mbert-lao-cross-validation`

Common behavior:
- No F1/Accuracy is computed during training.
- Each run saves hardware info.
- Each run saves prediction CSV outputs.
- Each run saves timing metrics with average epoch time.
- `textcnn` is trained from scratch with randomly initialized embeddings/CNN layers and does not support LoRA.
