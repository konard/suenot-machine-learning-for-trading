# Chapter 156: SimCLR for Stocks (Self-Supervised Contrastive Learning)

## Overview

Labeling financial data is notoriously difficult and prone to noise. **SimCLR (Simple Contrastive Learning of Visual Representations)**, originally a computer vision breakthrough, provides a way to learn powerful feature representations from **unlabeled** stock price data.

In this chapter, we adapt SimCLR for 1D financial time series. The model learns to map similar (augmented) versions of the same price pattern to nearby points in a latent space, while pushing different patterns far apart. This "pre-training" allows downstream models (like buy/sell classifiers) to work much better with very small amounts of labeled data.

## The SimCLR Workflow for Time Series

1. **Stochastic Data Augmentation**: For a given price window $x$, we generate two different "views" $x_i$ and $x_j$ using random transformations (scaling, jittering, masking).
2. **Base Encoder**: A neural network $h = f(x)$ (e.g., ResNet-1D or TCN) extracts feature vectors.
3. **Projection Head**: A small MLP $z = g(h)$ maps features to a space where contrastive loss is applied.
4. **Contrastive Loss (NT-Xent)**: We maximize the agreement between $z_i$ and $z_j$ (the positive pair) compared to all other patterns in the batch (negative pairs).

## Why SimCLR for Stocks?

- **Feature Discovery**: The model discovers "what a stock pattern looks like" without being told what a "Head and Shoulders" is.
- **Robustness**: By training to be invariant to noise (jittering) and scale, the learned features are more stable across different market regimes.
- **Data Efficiency**: You can use years of unlabeled tick data to pre-train the encoder, then fine-tune it on just a few hundred labeled events.

---

## Contents

- **`python/augmentations.py`**: Temporal augmentations (Jitter, Scale, Permute, Mask).
- **`python/model.py`**: 1D-CNN Encoder and Projection Head.
- **`python/train.py`**: Self-supervised pre-training loop using NT-Xent loss.
- **`python/evaluate.py`**: Linear probing / Downstream task on labeled data.
- **`rust/src/`**: High-performance feature extraction for live streaming data.

---

## References

1. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations.* [arXiv:2002.05709](https://arxiv.org/abs/2002.05709).
2. Eldele, E., et al. (2021). *Time-Series Representation Learning via Temporal and Contextual Contrasting.*
3. *SimCLR for Time Series* - Various adaptations for financial anomaly detection.
