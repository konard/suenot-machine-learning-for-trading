# Chapter 158: BYOL Trading (Bootstrap Your Own Latent)

## Overview

Self-supervised learning on financial data usually requires contrasting a "positive" sample against many "negative" samples (SimCLR) or using a large memory queue (MoCo). **BYOL (Bootstrap Your Own Latent)** fundamentally changes this: it learns to represent data **without using negative samples at all**.

In this chapter, we adapt BYOL for 1D stock price patterns. The model learns by ensuring that one neural network (the Online Network) can predict the output of another slowly moving neural network (the Target Network) for a different augmented view of the same price window.

## Key Mechanisms

1. **Two Networks**: 
   - **Online Network**: Needs to be fast and learns rapidly. It consists of an Encoder, a Projector, and a unique **Predictor**.
   - **Target Network**: A slower, more stable network. It consists only of an Encoder and a Projector. Its weights are an exponential moving average (EMA) of the Online Network's weights.
2. **The Asymmetry**: The crucial part that prevents "representation collapse" (where the network just outputs constant zeros for everything) is the **Predictor** in the online network, combined with a **Stop-Gradient** operation on the target network.
3. **The Objective**: For two augmented views of a stock chart ($v$ and $v'$), the Online Network uses $v$ to predict the Target Network's representation of $v'$. 

## Why BYOL for Trading?

- **No Negatives Needed**: Financial markets can be tricky; sometimes a "negative" sample (a random other chart) might actually represent the exact same macroeconomic state. BYOL doesn't care about negatives, eliminating the risk of accidental "false negatives."
- **Batch Size Efficiency**: Because it doesn't rely on negative pairs within the batch, BYOL can be trained effectively with significantly smaller batch sizes compared to SimCLR.
- **Robustness**: The reliance on predicting a moving average target makes the learned features highly robust to the extreme noise often found in tick-level or 1-minute bar data.

---

## Contents

- **`python/model.py`**: Implementation of the Asymmetric Online and Target Networks.
- **`python/train.py`**: BYOL training loop with Stop-Gradient and EMA updates.
- **`python/evaluate.py`**: Verification that "collapse" did not occur.
- **`rust/src/`**: Feature extraction using the stable Target Encoder.

---

## References

1. Grill, J. B., et al. (2020). *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.* [arXiv:2006.07733](https://arxiv.org/abs/2006.07733).
2. Extensions to Time Series: TSBYOL paradigms for financial anomaly detection.
