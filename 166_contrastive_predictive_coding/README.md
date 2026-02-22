# Chapter 166: Contrastive Predictive Coding (CPC)

## Overview

**Contrastive Predictive Coding (CPC)** is a powerful self-supervised learning framework designed to extract compact latent representations from high-dimensional data, especially sequences like audio, text, or financial time-series.

The core idea of CPC is to learn features that "predict the future" in a latent space. Instead of predicting the raw data (which is extremely noisy in finance), the model predicts the *representations* of future time steps.

## Architecture

1. **Encoder ($g_{enc}$)**: Maps a window of raw data $x_t$ to a latent representation $z_t$.
2. **Autoregressive Model ($g_{ar}$)**: Summarizes all $z_{\leq t}$ into a context vector $c_t$ (e.g., using a GRU or LSTM).
3. **Prediction**: A linear transformation maps $c_t$ to predicted future latents $\hat{z}_{t+k}$.
4. **InfoNCE Loss**: Measures how well $\hat{z}_{t+k}$ matches the actual $z_{t+k}$ compared to "negative" samples drawn from other time steps.

## Why it Matters for Trading

Financial markets are non-stationary and noisy. Predicting the next price (regression) is often an exercise in futility. CPC allows the model to:
- Ignore local noise and focus on slow-moving structural features.
- Learn temporal dependencies without needing explicit labels.
- Create "Context Vectors" that capture the current market regime (Bull/Bear/Sideways).

## Project Structure

```
166_contrastive_predictive_coding/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py       # CNN Encoder + GRU context
│   ├── cpc_loss.py    # InfoNCE loss for sequences
│   └── train.py       # Self-supervised training
└── rust/src/
    └── lib.rs         # Inference for the context vector
```
