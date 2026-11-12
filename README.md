# Prior-data Fitted Networks (PFN) Tutorial

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barhanc/pfn-tutorial/blob/main/pfn-tutorial.ipynb)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An intuitive, hands-on tutorial and PyTorch implementation of **Prior-data Fitted Networks (PFNs)** — the breakthrough meta-learning framework powering foundational tabular models like **TabPFN**.

## Getting Started

Ensure you have [uv](https://docs.astral.sh/uv/) installed.

### 1. Clone & Sync Environment

```bash
git clone https://github.com/barhanc/pfn-tutorial.git
cd pfn-tutorial

# Synchronize the virtual environment with uv.lock in seconds
uv sync
```

### 2. Launch the Tutorial

Run JupyterLab or Jupyter Notebook directly via `uv`:

```bash
uv run jupyter lab
# or
uv run jupyter notebook
```

Alternatively, open the notebook in one click using [Google Colab](https://colab.research.google.com/github/barhanc/pfn-tutorial/blob/main/pfn-tutorial.ipynb).

## Key References

- **Prior-data Fitted Networks**: Müller et al., *"Transformers Can Do Bayesian Inference"*, ICLR 2022. [[arXiv:2112.10510](https://arxiv.org/abs/2112.10510)]
- **TabPFN**: Hollmann et al., *"TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"*, Nature 2025 / ICLR 2023. [[arXiv:2207.01848](https://arxiv.org/abs/2207.01848)]
