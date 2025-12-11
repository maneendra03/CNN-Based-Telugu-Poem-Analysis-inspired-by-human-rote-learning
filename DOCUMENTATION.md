# CNN-Based Poem Learning & Interpretation
## Inspired by Human Rote Learning

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [What Makes This Unique](#what-makes-this-unique)
4. [Architecture](#architecture)
5. [Technology Stack](#technology-stack)
6. [Module Descriptions](#module-descriptions)
7. [Training Process](#training-process)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Missing Elements & Future Work](#missing-elements--future-work)
10. [How to Run](#how-to-run)
11. [Publication Readiness](#publication-readiness)

---

## 🎯 Project Overview

This project develops a novel neural architecture for **poem learning and generation** that mimics how humans memorize poetry through repetition (rote learning). Unlike standard language models that simply predict the next word, our system:

- **Remembers** patterns through explicit memory cells
- **Refines** output through iterative feedback loops
- **Understands** poetry at multiple levels (character, word, line, stanza)
- **Applies** knowledge of poetic rules (rhyme, meter, style)

---

## 📝 Problem Statement

**"CNN-Based Poem Learning & Interpretation Inspired by Human Rote Learning"**

### Key Challenges Addressed:

| Challenge | How We Solve It |
|-----------|-----------------|
| Poems require pattern memorization | **Rote Learning Memory Module** |
| Poetry has multi-level structure | **Hierarchical RNN (char + line)** |
| Poems must follow rules (rhyme, meter) | **Knowledge Base Integration** |
| Generation needs refinement | **Feedback Loop with Comparator** |
| Local patterns (rhythm, alliteration) | **1D CNN Feature Extraction** |

---

## ⭐ What Makes This Unique

### Comparison with Existing Systems

| Existing Approach | Our Novel Approach | Why It's Better |
|------------------|-------------------|-----------------|
| GPT-2/GPT-3 alone | GPT-2 + **Rote Learning Memory** | Simulates how humans memorize through repetition |
| Standard attention | **Repetition Attention** | Weights frequently-seen patterns higher |
| Single-pass generation | **Iterative Feedback Loop** | Refines output like human practice |
| Only word-level | **Hierarchical (char + line + poem)** | Captures syllables, rhythm, and flow |
| Implicit knowledge | **Explicit Knowledge Base** | Applies poetic rules directly |
| BLEU evaluation only | **Memorization Curve Metric** | Measures learning efficiency |

### Novel Contributions (For Paper)

1. **Rote Learning Memory Module** - First to explicitly model human memorization for poetry
2. **Repetition Attention Mechanism** - Cognitively-inspired attention design
3. **Knowledge-Grounded Feedback Loop** - Bridges neural and symbolic approaches
4. **Memorization Curve Metric** - New evaluation paradigm for creative text

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT: Raw Poem Text                      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   1. PREPROCESSING LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Text Cleaner │→ │  Tokenizer   │→ │ Embedding (GPT-2)    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. CNN FEATURE MODULE                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1D CNNs with multiple kernel sizes (3, 5, 7)             │   │
│  │ Captures: rhythm, rhyme patterns, alliteration           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   3. HIERARCHICAL RNN                            │
│  ┌────────────────────┐    ┌────────────────────┐               │
│  │ Character-Level    │    │ Line-Level RNN     │               │
│  │ RNN (syllables)    │ →  │ (semantics, flow)  │               │
│  └────────────────────┘    └────────────────────┘               │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│               4. MEMORY & ATTENTION MODULE [NOVEL]               │
│  ┌────────────────────┐    ┌────────────────────┐               │
│  │ Rote Learning      │    │ Repetition         │               │
│  │ Memory Cells       │ ←→ │ Attention          │               │
│  │ (stores patterns)  │    │ (weights familiar) │               │
│  └────────────────────┘    └────────────────────┘               │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│               5. KNOWLEDGE INTEGRATION [NOVEL]                   │
│  ┌────────────────────┐    ┌────────────────────┐               │
│  │ Knowledge Base     │    │ Feedback           │               │
│  │ (rhyme rules,      │ →  │ Comparator         │               │
│  │  meter patterns)   │    │ (iterative refine) │               │
│  └────────────────────┘    └────────────────────┘               │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   6. DECODER & OUTPUT                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Autoregressive Decoder with Attention + Beam Search      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                OUTPUT: Generated/Interpreted Poem                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Core Framework** | PyTorch 2.0+ | Deep learning |
| **Pre-trained Model** | GPT-2 (Hugging Face) | Language understanding |
| **Text Processing** | NLTK, spaCy | Tokenization, NLP |
| **Embeddings** | BERT / Word2Vec | Word representations |
| **UI** | Gradio | Web interface |
| **Training** | MPS (Mac M2) | GPU acceleration |
| **Experiment Tracking** | Weights & Biases | Logging |
| **Visualization** | Matplotlib, Seaborn | Plots |

---

## 📁 Module Descriptions

### `src/models/cnn_module.py`
**CNN Feature Extractor** - Captures local poetic patterns
- Multiple kernel sizes (3, 5, 7) for different n-gram patterns
- Batch normalization and residual connections
- Outputs: rhythm patterns, rhyme sounds, alliteration

### `src/models/hierarchical_rnn.py`
**Hierarchical RNN** - Multi-level poem understanding
- Character-Level RNN: syllables, phonetics
- Line-Level RNN: line semantics, inter-line flow
- Poem-Level: overall theme and structure

### `src/models/memory_attention.py` ⭐ NOVEL
**Rote Learning Memory** - Simulates human memorization
- Memory cells that store and reinforce patterns
- Memory strength increases with repetition
- Decay mechanism for unused patterns
- Repetition Attention weights familiar patterns higher

### `src/models/knowledge_base.py`
**Knowledge Base** - Poetic rules and constraints
- Rhyme scheme definitions (ABAB, AABB, etc.)
- Meter patterns (iambic, trochaic, etc.)
- Style embeddings for different poetry types

### `src/models/feedback_loop.py` ⭐ NOVEL
**Feedback Loop** - Iterative refinement
- Compares generated output with poetic rules
- Adjusts weights based on rule violations
- Multiple refinement iterations
- Simulates "practice makes perfect"

### `src/models/decoder.py`
**Poem Decoder** - Text generation
- Autoregressive generation with attention
- Beam search for quality
- Style-conditioned generation

### `src/models/poem_learner.py`
**Main Model** - Integrates all modules
- 125M total parameters (46M trainable)
- Uses GPT-2 as backbone (frozen)
- Our novel modules on top (trainable)

---

## 🏋️ Training Process

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 (recommended: 15-20) |
| Batch Size | 4 |
| Learning Rate | 3e-5 |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |
| Max Sequence Length | 128 |
| Dataset Size | 12,000 poems |

### Training Results

| Metric | Start | End | Improvement |
|--------|-------|-----|-------------|
| Loss | 5.40 | 5.08 | 6% |
| Rhyme Accuracy | ~5% | 17% | 240% |
| Meter Consistency | ~20% | 34% | 70% |

---

## 📊 Evaluation Metrics

### Standard Metrics
- **BLEU Score**: N-gram overlap with references
- **Perplexity**: Language model confidence

### Poetry-Specific Metrics
- **Rhyme Accuracy**: % of lines that rhyme correctly
- **Meter Consistency**: Syllable pattern variance
- **Distinct-1/2**: Vocabulary diversity

### Novel Metrics ⭐
- **Memorization Curve**: How quickly model learns patterns
- **Retention Score**: Memory cell utilization
- **Learning Efficiency**: Steps to achieve target performance

---

## ❗ Missing Elements & Future Work

### Currently Missing

| Element | Status | Impact |
|---------|--------|--------|
| Larger dataset (50K+ poems) | Needed | Higher accuracy |
| More training epochs (15-20) | Needed | Better generation |
| Rhyme-specific loss function | TODO | Better rhyming |
| Human evaluation study | Needed for paper | Subjective quality |
| Baseline comparison | Needed for paper | Prove improvement |

### Future Improvements

1. **Add Rhyme Loss**: Penalize non-rhyming lines during training
2. **Meter Loss**: Enforce syllable count consistency
3. **Larger GPT-2**: Use gpt2-medium (345M) or gpt2-large (774M)
4. **More Data**: Combine multiple poetry datasets
5. **Style Control**: Better conditioning on poetry styles

---

## 🚀 How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Download Dataset
```bash
python scripts/download_datasets.py
```

### Training
```bash
python scripts/train_lightweight.py
```

### Generation
```bash
python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "Roses are red"
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Web UI
```bash
python app/ui.py
# Open http://localhost:7860
```

---

## 📄 Publication Readiness

### ✅ Ready For Paper

| Requirement | Status |
|-------------|--------|
| Novel architecture | ✅ Memory + Feedback Loop |
| Working implementation | ✅ 42 files, tested |
| Training pipeline | ✅ Complete |
| Evaluation metrics | ✅ BLEU, Rhyme, Meter |
| Code on GitHub | ✅ Pushed |

### ⚠️ Needed Before Submission

| Requirement | Status |
|-------------|--------|
| Baseline comparison | ❌ Need vanilla GPT-2 results |
| Ablation study | ❌ Test each module's contribution |
| Human evaluation | ❌ Get human ratings |
| More training | ❌ 15-20 epochs recommended |

### Suggested Venues

- **ACL** (Association for Computational Linguistics)
- **EMNLP** (Empirical Methods in NLP)
- **AAAI** (AI Conference)
- **IEEE Access** (Journal)
- **COLING** (Computational Linguistics)

---

## 📬 Contact

**Author**: Maneendra
**GitHub**: [maneendra03](https://github.com/maneendra03)
**Repository**: [CNN-Based-Poem-Learning-Interpretation-Inspired-by-Human-Rote-Learning-](https://github.com/maneendra03/CNN-Based-Poem-Learning-Interpretation-Inspired-by-Human-Rote-Learning-)

---

*This project was developed as a research contribution to computational creativity and cognitive-inspired NLP.*
