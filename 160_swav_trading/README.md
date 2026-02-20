# Chapter 160: SwAV for Algorithmic Trading

## Overview

**SwAV (Swapping Assignments between Views)** revolutionizes Self-Supervised Learning by introducing **Online Clustering**. Instead of dragging individual continuous feature vectors directly towards each other (as in SimCLR or MoCo), SwAV trains the network to map features to a set of learnable **Prototypes** (Cluster Centers).

This approach scales incredibly well and introduces immediate interpretability: the model natively learns "Market Regimes" (e.g., bull trends, high-volatility sideways chops, bear sweeps) via the discrete prototypes.

## Core Mechanisms

### 1. Prototypes (The Codebook)
The model maintains a matrix of weight vectors called Prototypes. Representing structural states of the market, each produced continuous embedding vector is compared against these Prototypes via cosine similarity to find the best match.

### 2. Sinkhorn-Knopp Algorithm (The Balancer)
If the model was left to its own devices, it would succumb to "representation collapse"—assigning every single stock chart to Prototype #1 (e.g., just guessing "Flat Market" constantly). 

To prevent this, SwAV uses the **Sinkhorn-Knopp algorithm** over the batch. This is an Optimal Transport mathematical technique that enforces an **equipartition constraint**: it strictly guarantees that charts in a batch are distributed evenly across all available prototypes. 

### 3. Swapped Prediction (The Loss)
For a given financial chart, we create View A and View B using data augmentations (like adding noise or jitter).
1. We compute continuous features for both: $z_A$ and $z_B$.
2. We use Sinkhorn-Knopp to assign View A to a hard cluster: $q_A$.
3. We assign View B to a hard cluster: $q_B$.
4. **The Swap**: We train the network such that the continuous features of A ($z_A$) can accurately predict the discrete cluster of B ($q_B$), and $z_B$ predicts $q_A$.

## Trading Advantages

- **Native Regime Classification**: Your trading bot immediately gets discrete "States" ($1-K$) classifying the current market, rather than raw floating-point gradients that require an external K-Means pass.
- **Batched Efficiency**: The Sinkhorn algorithm stabilizes gradients significantly better than point-to-point contrastive losses.
- **Interpretability**: You can visually inspect what specific market patterns trigger Prototype 1 versus Prototype 7.

---

## Contents

- **`python/model.py`**: Implementation of the Prototypes, Sinkhorn-Knopp online optimal transport, and the Swapped Cross-Entropy loss.
- **`python/train.py`**: Training loop optimizing the continuous CNN alongside the discrete prototypes.
- **`python/evaluate.py`**: Verification of clustering health and regime assignment overlap.
- **`rust/src/`**: High-performance Rust inference pipeline assigning real-time LOB/Price ticks to the nearest SwAV Prototype instantly.

---

## References

1. Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., & Joulin, A. (2020). *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments.* [arXiv:2006.09882](https://arxiv.org/abs/2006.09882).
