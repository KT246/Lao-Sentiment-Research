# LaoSA: A Benchmark for Lao Sentiment Analysis

[![Dataset](https://img.shields.io/badge/Dataset-LaoSA-blue.svg)](https://github.com/KT246/LaoSA)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)

## Abstract

Effective sentiment analysis for the Lao language is hindered by a scarcity of annotated resources and the inherent noise of real-world user feedback, which often features code-mixing and informal script patterns. This study addresses these challenges by introducing a high-quality, filtered benchmark of 25,139 Lao-language reviews collected from popular digital applications. We develop a comprehensive framework based on fine-tuning pre-trained language models for Lao sentiment analysis. Our experimental results demonstrate that XLM-RoBERTa achieves superior performance with 96.64% accuracy and an F1-macro score of 0.9617. We also find that while Lao-oriented pre-training provides significant benefits, low-rank adaptation offers a highly efficient performance-cost trade-off, maintaining competitive accuracy with minimal trainable parameters. Our research highlights the robustness of multilingual transfer for low-resource sentiment classification and provides a publicly available dataset and code at [https://github.com/KT246/LaoSA](https://github.com/KT246/LaoSA).

## Dataset

The released LaoSA benchmark contains 25,139 cleaned Lao-language application reviews for binary sentiment classification:

| Split | File | Rows |
| --- | --- | ---: |
| Train | [`data/train.csv`](data/train.csv) | 20,111 |
| Validation | [`data/val.csv`](data/val.csv) | 5,028 |

No separate test split is used in the paper. The reported fixed-split results are evaluated on `data/val.csv`.

## Experimental Protocol

The paper reports two distinct uses of cross-validation:

| Stage | Fold Setting | Purpose |
| --- | --- | --- |
| Dataset auditing | 5-fold cross-validation | Identify unstable or suspicious samples during data cleaning. |
| Model evaluation | 3-fold cross-validation | Report cross-validation performance for selected experiments. |

This distinction is important: the 5-fold procedure belongs to dataset refinement, while the paper's cross-validation experiment setting is 3-fold.

## Trained Model Artifacts

Large trained model files are stored outside GitHub. The table below lists the artifact links recorded in each experiment branch under `outputs/<experiment>/model_link.txt`.

| Experiment | Model | Artifact Link |
| --- | --- | --- |
| `xlm-roberta-finetuning` | XLM-RoBERTa (Full Fine-tuning) | [Google Drive](https://drive.google.com/drive/folders/1MsJ_TaeQawzFlN5_4NdGXdFLcYg3ugxp?usp=sharing) |
| `xlm-roberta-lora` | XLM-RoBERTa + LoRA | [Google Drive](https://drive.google.com/drive/folders/1-OgEXnzl2NMld-_M5CN9DibwiRtI2pDa?usp=sharing) |
| `xlm-roberta-cross-validation` | XLM-RoBERTa (Cross-Validation) | [Google Drive](https://drive.google.com/drive/folders/1FaDloG0f4Bj_V8W5qQHhP20YgdFsUjco?usp=sharing) |
| `mbert-finetuning` | mBERT (Full Fine-tuning) | [Google Drive](https://drive.google.com/drive/folders/17WmRK5ijuT3PqCI7BXUvPP19p7IbrsYf?usp=sharing) |
| `mbert-lora` | mBERT + LoRA | [Google Drive](https://drive.google.com/drive/folders/1c3yeq8J7zqVKq2dk7Yl66yecwzAboFZB?usp=sharing) |
| `mbert-cross-validation` | mBERT (Cross-Validation) | [Google Drive](https://drive.google.com/drive/folders/1bOyrU9ND_CKTJD6OnwRtw6SoxUsb3rF8?usp=sharing) |
| `mbert-lao-finetuning` | mBERT-Lao (Full Fine-tuning) | [Google Drive](https://drive.google.com/drive/folders/1VZqKRHPcBzjsASLTk4uMhMoccx90cr14?usp=sharing) |
| `mbert-lao-lora` | mBERT-Lao + LoRA | [Google Drive](https://drive.google.com/drive/folders/1qpM99zH8uj4sVFydRsCxqoVDTMMS6chT?usp=sharing) |
| `mbert-lao-cross-validation` | mBERT-Lao (Cross-Validation) | [Google Drive](https://drive.google.com/drive/folders/1SSVH42_m9aENmsrK9cj6hxqEdYe5JP_E?usp=sharing) |
| `logistic-regression` | Logistic Regression | [Google Drive](https://drive.google.com/drive/folders/1O5oOxhBIpL08csS9I6GXxtcaiCpVL0J2?usp=sharing) |
| `svm` | Support Vector Machine (SVM) | [Google Drive](https://drive.google.com/drive/folders/1O5oOxhBIpL08csS9I6GXxtcaiCpVL0J2?usp=sharing) |
| `decision-tree` | Decision Tree | [Google Drive](https://drive.google.com/drive/folders/1O5oOxhBIpL08csS9I6GXxtcaiCpVL0J2?usp=sharing) |

## Repository Structure

```text
Lao-Sentiment-Research/
+-- data/
|   +-- README.md
|   +-- train.csv
|   +-- val.csv
+-- experiments/
+-- src/
+-- requirements.txt
+-- run_training.sh
+-- setup.py
```

## Reference

If you find this work useful in your research, please consider citing:

```bibtex
@misc{laosa2026,
  title={LaoSA: A Benchmark for Lao Sentiment Analysis},
  author={Khamtay Kongmanh, Quang-Vinh Pham, Quang-Hung Le},
  year={2026}
}
```
