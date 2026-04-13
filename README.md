# LaoSA: A Benchmark for Lao Sentiment Analysis

[![Dataset](https://img.shields.io/badge/Dataset-LaoSA-blue.svg)](https://github.com/KT246/LaoSA)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)

## Abstract

Effective sentiment analysis for the Lao language is hindered by a scarcity of annotated resources and the inherent noise of real-world user feedback, which often features code-mixing and informal script patterns. This study addresses these challenges by introducing a high-quality, filtered benchmark of 25,139 Lao-language reviews collected from popular digital applications. We develop a comprehensive framework based on fine-tuning pre-trained language models for Lao sentiment analysis. Our experimental results demonstrate that XLM-RoBERTa achieves superior performance with 96.64% accuracy and an F1-macro score of 0.9617. We also find that while Lao-oriented pre-training provides significant benefits, low-rank adaptation offers a highly efficient performance-cost trade-off, maintaining competitive accuracy with minimal trainable parameters. Our research highlights the robustness of multilingual transfer for low-resource sentiment classification and provides a publicly available dataset and code at [https://github.com/KT246/LaoSA](https://github.com/KT246/LaoSA).

## Reference

If you find this work useful in your research, please consider citing:

```bibtex
@misc{laosa2026,
  title={LaoSA: A Benchmark for Lao Sentiment Analysis},
  author={Khamtay Kongmanh, Quang-Vinh Pham, Quang-Hung Le},
  year={2026}
}
```

---
