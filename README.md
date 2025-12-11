# CNN-Based Poem Learning & Interpretation
## Inspired by Human Rote Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A novel neural architecture that mimics human rote learning for poetry memorization, understanding, and generation.

## 🎯 Key Features

- **Rote Learning Simulation**: LSTM memory cells that simulate human repetition-based memorization
- **CNN Feature Extraction**: 1D convolutions to capture rhythm, rhyme, and local patterns
- **Hierarchical Understanding**: Character-level and line-level RNNs for multi-granular analysis
- **Knowledge-Grounded Feedback**: Iterative refinement using poetic rules and grammar

## 🏗️ Architecture

```
Input Text → Preprocessing → CNN Features → Memory & Attention → Decoder → Output
                               ↓                    ↓
                    Hierarchical RNN ←→ Knowledge Base Feedback
```

## 📁 Project Structure

```
├── config/              # Configuration files
├── data/                # Datasets and knowledge base
├── src/                 # Source code
│   ├── preprocessing/   # Text cleaning, tokenization, embeddings
│   ├── models/          # Neural network modules
│   ├── training/        # Training pipeline
│   └── evaluation/      # Metrics and visualizations
├── notebooks/           # Jupyter notebooks for experiments
├── scripts/             # Training and generation scripts
└── tests/               # Unit tests
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python scripts/train.py --config config/config.yaml

# Generate poem
python scripts/generate.py --prompt "roses are red"
```

## 📊 Evaluation Metrics

- BLEU Score, Perplexity
- Rhyme Accuracy, Meter Consistency
- Novel: Memorization Curve, Retention Score

## 📝 Citation

```bibtex
@article{poemlearner2024,
  title={CNN-Based Poem Learning & Interpretation Inspired by Human Rote Learning},
  author={Your Name},
  year={2024}
}
```

## 📄 License

MIT License
