# Chapter 168: NT-Xent Loss and Temperature Scaling

## Overview

The **Normalized Temperature-scaled Cross Entropy (NT-Xent)** loss is a foundational component of modern contrastive learning (e.g., SimCLR). In trading, it is used to learn robust representations by maximizing the agreement between different "views" of the same market situation through a contrastive objective.

The "Magic" of NT-Xent lies in the **Temperature parameter ($\tau$)**, which controls how much the model penalizes "hard" negatives compared to "easy" ones.

## The Loss Formula

Given a pair of positive samples $(z_i, z_j)$ in a batch of size $N$, the loss for that pair is:

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

Where $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ is the cosine similarity.

## Why Temperature Scaling Matters

1. **Gradient Sharpening**: A small $\tau$ (e.g., 0.07) makes the softmax distribution "sharper," focusing the gradient on the most similar negative samples (the hardest ones).
2. **Feature Uniformity**: NT-Xent encourages embeddings to be uniformly distributed on the unit hypersphere, preventing "feature collapse" where all samples map to the same vector.
3. **Robustness in Finance**: Financial data is extremely noisy. If $\tau$ is too small, the model may overfit to "noise-induced similarity." If $\tau$ is too large, it learns too slowly.

## Project Structure

```
168_nt_xent_trading/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py       # Base CNN Encoder
│   ├── nt_xent_loss.py# NT-Xent implementation
│   └── train.py       # Temperature sweep experiments
└── rust/src/
    └── lib.rs         # High-speed NT-Xent for production
```
