# Prior-data Fitted Networks (PFN) Tutorial ⚡🧠

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barhanc/pfn-tutorial/blob/main/pfn-tutorial.ipynb)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An intuitive, hands-on tutorial and PyTorch implementation of **Prior-data Fitted Networks (PFNs)** — the breakthrough meta-learning framework powering foundational tabular models like **TabPFN**.

---

## 📑 Overview

| Notebook | Description | Target Audience |
| :--- | :--- | :--- |
| [**`pfn-tutorial.ipynb`**](./pfn-tutorial.ipynb) | Complete tutorial with theoretical background, full PyTorch implementation, training loop, and evaluation. | Self-study, reference, and experimentation. |
| [**`pfn-tutorial-empty.ipynb`**](./pfn-tutorial-empty.ipynb) | Exercise version containing empty implementation blocks (`Zadanie 1–3`) and guided instructions. | Classroom labs and hands-on workshops. |

---

## 🚀 Getting Started

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

---

## 📚 Key References

- **Prior-data Fitted Networks**: Müller et al., *"Transformers Can Do Bayesian Inference"*, ICLR 2022. [[arXiv:2112.10510](https://arxiv.org/abs/2112.10510)]
- **TabPFN**: Hollmann et al., *"TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"*, Nature 2025 / ICLR 2023. [[arXiv:2207.01848](https://arxiv.org/abs/2207.01848)]
