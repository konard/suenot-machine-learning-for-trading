# Chapter 167: Supervised Contrastive Learning (SupCon)

## Overview

In previous chapters, we used **Self-Supervised Learning** (InfoNCE, CPC) to learn features by comparing different views of the same data. While powerful, these methods don't know which samples are truly similar in terms of outcome (e.g., "Bullish" vs "Bearish").

**Supervised Contrastive Learning (SupCon)** extends the contrastive framework by using class labels. Instead of just pulling a sample toward its augmented version, SupCon pulls it toward **all samples that share the same label** while pushing it away from all samples with different labels.

## How it Works

1. **Labels**: Each time-series window is assigned a label (e.g., 0 for Neutral, 1 for Up, 2 for Down).
2. **Contrastive Objective**: For a given "anchor" sample, all other samples in the batch with the *same label* act as positives. All samples with *different labels* act as negatives.
3. **Loss Function**: The model minimizes a generalized InfoNCE loss that handles multiple positive pairs per batch.

## Benefits for Trading

- **Better Discrimination**: Standard Cross-Entropy loss can be sensitive to noise. SupCon focuses on the relative structure of the latent space, making embeddings more robust.
- **Improved Clusters**: The resulting embeddings form tighter clusters for specific market regimes, which is ideal for k-Nearest Neighbor (k-NN) classification or anomaly detection.
- **Transfer Learning**: SupCon encoders are often better "pre-trained" models for downstream tasks like Reinforcement Learning.

## Project Structure

```
167_supervised_contrastive_learning/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py       # Encoder with Projection Head
│   ├── supcon_loss.py # Supervised Contrastive Loss
│   └── train.py       # Labeled training loop
└── rust/src/
    └── lib.rs         # Optimized embedding extraction
```
