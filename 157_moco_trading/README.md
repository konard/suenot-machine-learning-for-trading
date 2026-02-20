# Chapter 157: MoCo Trading (Momentum Contrast for Finance)

## Overview

Self-supervised learning on financial time series often suffers from "representation collapse" or the need for massive batch sizes to get enough negative samples. **MoCo (Momentum Contrast)** solves this by maintaining a large, dynamic **Queue** of sample features and using a **Momentum Encoder** to ensure consistency.

In this chapter, we adapt MoCo for 1D stock price patterns. By decoupling the dictionary size from the batch size, MoCo allows us to contrast current price movements against thousands of historical patterns without requiring massive GPU memory.

## Key Mechanisms

1. **Dictionary as a Queue**: We maintain a FIFO queue of feature vectors from previous batches. This provides a "long memory" of market states to contrast against.
2. **Momentum Update**: Instead of backpropagating through both encoders, the "Key Encoder" (Momentum Encoder) is updated as a moving average of the "Query Encoder":
   $$\theta_k \leftarrow m\theta_k + (1-m)\theta_q$$
   where $m$ is typically $0.999$. This ensures that the features in the queue stay consistent even as the query encoder trains.
3. **Contrastive Loss**: We use InfoNCE loss to make the current price pattern (query) similar to its augmented version (positive key) and dissimilar to all patterns stored in the queue (negative keys).

## Why MoCo for Trading?

- **Temporal Stability**: The momentum update prevents the representation from changing too rapidly, which is crucial for the volatile and non-stationary nature of financial markets.
- **Rich Negative Samples**: Trading patterns are diverse. Having a large queue allows the model to differentiate a "bullish breakout" from thousands of different types of "sideways noise" or "distribution phases."
- **Scalability**: MoCo can be trained on standard hardware while still utilizing a vast "dictionary" of historical market behaviors.

---

## Contents

- **`python/model.py`**: Implementation of Query/Momentum encoders and the Queue logic.
- **`python/train.py`**: Training loop with momentum updates and queue management.
- **`python/evaluate.py`**: Feature stability verification.
- **`rust/src/`**: High-performance feature extraction using the momentum-tuned weights.

---

## References

1. He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning.* [arXiv:1911.05722](https://arxiv.org/abs/1911.05722).
2. *Contrastive Learning for Time Series* - Various financial adaptations.
