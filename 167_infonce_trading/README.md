# Chapter 167: InfoNCE for Trading

## Overview

**InfoNCE (Information Noise-Contrastive Estimation)** is a contrastive loss function introduced in the Contrastive Predictive Coding (CPC) paper (van den Oord et al., 2018). It maximizes the mutual information between a learned representation and the data it encodes by distinguishing a "positive" sample from a set of "negative" distractors.

While Chapter 166 focused on the full CPC architecture (encoder + autoregressive model), this chapter isolates the **InfoNCE loss itself** as a standalone, general-purpose tool for learning trading representations. InfoNCE can be applied to any scenario where you need to learn embeddings that capture similarity — between time windows, between assets, or between market conditions.

## Mathematical Foundation

### Mutual Information Maximization

The goal of InfoNCE is to maximize a lower bound on the mutual information $I(X; C)$ between data $X$ and its context $C$:

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{f_k(x_{t+k}, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)}\right]$$

where:
- $f_k(x, c) = \exp(z_x^\top W_k c)$ is the scoring function (log-bilinear model)
- $x_{t+k}$ is the **positive** sample (the true future)
- $\{x_j\}$ contains one positive and $N-1$ **negative** samples (random distractors)
- $c_t$ is the context vector at time $t$

### Connection to Cross-Entropy

In practice, InfoNCE reduces to a **softmax cross-entropy** over the positive sample index:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\sum_{i=0}^{N} \exp(\text{sim}(q, k_i) / \tau)}$$

where $\tau$ is a temperature hyperparameter controlling the sharpness of the distribution.

### Why Temperature Matters

- **Low $\tau$** (e.g., 0.05): The model must make very confident distinctions. Good for well-separated classes.
- **High $\tau$** (e.g., 1.0): The model spreads probability more evenly. Useful when categories overlap.
- In trading, $\tau \in [0.07, 0.5]$ works well. Markets have fuzzy boundaries between regimes, so moderate temperatures are preferred.

## Trading Applications

### 1. Market Regime Similarity

Train an encoder so that windows from the same market regime (e.g., "high-volatility uptrend") are closer in embedding space than windows from different regimes.

### 2. Cross-Asset Pair Discovery

Use InfoNCE to learn which assets move similarly during specific conditions, enabling dynamic pair trading or sector rotation.

### 3. Temporal Pattern Matching

Embed historical windows and use InfoNCE to find past periods most similar to the current market state, supporting analogy-based forecasting.

### 4. Order Flow Representation

Learn compact representations of order book snapshots where similar liquidity states are clustered together.

## Architecture

```
         ┌──────────────┐
   x_i → │   Encoder    │ → z_i (anchor)
         │  (1D-CNN)    │
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │  InfoNCE     │ ← z_+ (positive), z_1...z_N (negatives)
         │  Loss        │
         └──────────────┘
                │
                ▼
         Maximize similarity(z_i, z_+)
         Minimize similarity(z_i, z_j) for j ≠ +
```

## Project Structure

```
167_infonce_trading/
├── README.md               # English overview (this file)
├── README.ru.md            # Russian overview
├── readme.simple.md        # Simplified explanation (English)
├── readme.simple.ru.md     # Simplified explanation (Russian)
├── python/
│   ├── requirements.txt    # Python dependencies
│   ├── model.py            # Encoder + InfoNCE model
│   ├── infonce_loss.py     # Standalone InfoNCE loss
│   └── train.py            # Training loop with synthetic + Bybit data
└── rust/
    ├── Cargo.toml
    └── src/
        └── lib.rs          # InfoNCE scoring in Rust
```

## Key Differences from Standard CPC (Chapter 166)

| Aspect | CPC (Ch. 166) | InfoNCE (Ch. 167) |
|--------|---------------|-------------------|
| Scope | Full architecture | Loss function focus |
| Negatives | Other batch samples | Configurable strategy |
| Context | Autoregressive GRU | Any anchor embedding |
| Temperature | Implicit | Explicit $\tau$ parameter |
| Use case | Temporal prediction | General similarity learning |

## References

1. **Representation Learning with Contrastive Predictive Coding** — van den Oord et al., 2018. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
2. **Momentum Contrast for Unsupervised Visual Representation Learning** — He et al., 2020. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
3. **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)** — Chen et al., 2020. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
