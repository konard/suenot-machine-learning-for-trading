# Chapter 165: Hard Negative Mining for Contrastive Learning

## Overview

In contrastive learning, the model learns by pulling similar examples (positives) together and pushing dissimilar examples (negatives) apart. In standard implementations like Chapter 164 (Cross-Modal Contrastive), negative examples are typically chosen randomly from the current batch.

However, not all negatives are created equal. "Easy" negatives (samples that look nothing like the anchor) provide very little gradient information because the model can distinguish them with zero effort. **Hard Negative Mining** is the process of identifying samples that are very similar to the anchor but belong to a different class. These "hard" examples force the model to learn much finer boundaries and more robust features.

## Why it Matters for Trading

Financial time series often exhibit similar local patterns (e.g., a small "dip") while having completely different long-term trends or underlying fundamentals. 
- **Random Negatives**: Might compare a "Sine Wave" price chart with a "Trending Up" one. This is too easy.
- **Hard Negatives**: Might compare two different "Sine Waves" that belong to different regimes or assets. This forces the model to ignore superficial similarities and find deeper structural features.

## Architecture

We implement two main strategies for selecting negatives:
1. **Top-K Batch Mining**: Within each training batch, we compute a full similarity matrix and select the $K$ most similar (but incorrect) samples as negatives.
2. **Semi-Hard Mining**: Selecting negatives whose similarity is high but strictly less than the anchor's similarity to the positive.

## Project Structure

```
165_hard_negative_mining/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py       # Encoder architecture
│   ├── miner.py       # Mining logic (Top-K)
│   └── train.py       # Training loop
└── rust/src/
    └── lib.rs         # Optimized Rust miner
```
