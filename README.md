# Paper Report Branch

This branch, `paper-report`, is a standalone branch for the manuscript, experiment artifacts, analysis outputs, and report-generation materials for the Lao sentiment analysis project.

It is intentionally separated from the main training/code branch. In other words, this branch is for the paper package and report assets, not for the original training pipeline source tree from `main`.

## What This Branch Contains

- `latex/`
  - Main LaTeX manuscript source: `samplepaper.tex`
  - Compiled paper PDF: `samplepaper.pdf`
  - Bib file, generated figure assets, hardware/error-analysis tables, and SVG-to-PDF exports
- `reports/`
  - Consolidated metrics/timing CSV files
  - Plot exports in `.png` and `.svg`
- Experiment result folders
  - `xlm-roberta-*`
  - `mbert-*`
  - `mbert-lao-*`
  - `logistic-regression/`
  - `svm/`
  - `decision-tree/`
- Report/analysis scripts
  - `run_report.py`
  - `run_error_analysis.py`
  - `run_fp_fn.py`
  - `run_hardware_report.py`
  - `run_overlap_fail_analysis.py`
  - `create_error_examples.py`
  - `create_error_example_table.py`
- Supporting assets
  - `fonts/saysettha_ot.ttf`
  - root-level `cv_*.csv` summary files
  - notes/audit files such as `reference_audit_2021plus.md`

## Recommended Starting Points

- Read the manuscript source:
  - [`latex/samplepaper.tex`](./latex/samplepaper.tex)
- Open the compiled paper:
  - [`latex/samplepaper.pdf`](./latex/samplepaper.pdf)
- Review consolidated experiment summaries:
  - [`reports/`](./reports)
- Review raw paper analysis tables:
  - [`latex/fp_fn/`](./latex/fp_fn)
  - [`latex/hardware/`](./latex/hardware)

## Build The Paper

This branch includes a simple build script:

```bash
cd latex
bash build.sh
```

Requirements:

- `xelatex` available in `PATH`
- shell escape enabled by the script

The script builds:

- `latex/samplepaper.pdf`

## Branch Notes

- This branch was created as an orphan branch so it can exist independently from `main`.
- The training pipeline source code from the original project branch is intentionally not included here.
- The purpose of this branch is to keep the paper package smaller, easier to review, and easier to share.

## Ignored Files

The root `.gitignore` is tailored for this branch and currently excludes:

- local virtual environments such as `.venv/`
- Python cache directories
- `backup/` content
- temporary LaTeX outputs such as `*.aux`, `*.log`, `*.out`
- temporary preview images such as `latex/__*.png` and `latex/figures/__*.png`

## Suggested Usage

Use this branch when you want to:

- edit the paper
- inspect the final experiment outputs used by the paper
- regenerate report tables/figures
- share a clean paper-focused branch without the full training codebase
