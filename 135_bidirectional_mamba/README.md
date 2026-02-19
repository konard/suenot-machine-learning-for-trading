# Chapter 135: Bidirectional Mamba

## Overview

Mamba, introduced by Gu and Dao, revolutionized sequence modeling by providing an $O(N)$ linear-time alternative to Transformers using Selective State Space Models (SSMs). Standard Mamba relies on a causal architecture—processing sequences strictly left-to-right. While perfect for language generation, this unidirectional scan misses critical global dependencies when dealing with structured data formats that do not need to be causally masked during the representation extraction phase.

Enter **Bidirectional Mamba**. Drawing inspiration from **Vision Mamba (Vim)** (2024), Bidirectional Mamba processes the input sequence both forward and backward. 

In algorithmic trading, we cannot peek into the future (data leakage is highly forbidden). Therefore, **how is Bidirectional Mamba applied to trading?**
When making a prediction for $t+1$, we use an observed historical window of $[t-H, t]$. Because the entire historical window observed up to $t$ is fully known, we can run a Bidirectional Mamba block over this fixed context. This allows the model to deeply understand the structural relationships across the whole lookback period without computational explosion, enabling it to synthesize a much denser context vector than a unidirectional model could achieve. It’s highly effective for extracting correlations from multi-dimensional Limit Order Books (LOB), long-term market regimes, and complex feature clusters.

## Table of Contents

1. [Theoretical Foundations: Vision Mamba to Finance](#theoretical-foundations-vision-mamba-to-finance)
2. [Why Bidirectionality for Time-Series?](#why-bidirectionality-for-time-series)
3. [The Trade-Offs: Causal Inference vs Context Representation](#the-trade-offs-causal-inference-vs-context-representation)
4. [Python Implementation Details](#python-implementation-details)
5. [Rust Implementation Details](#rust-implementation-details)
6. [Backtesting Methodology](#backtesting-methodology)
7. [References](#references)

---

## Theoretical Foundations: Vision Mamba to Finance

In **Vision Mamba (Vim)**, the model flattens an image into patches and processes them bidirectionally. Unlike 1D text, an image patch's context relies on patches both before and after it.
The core mathematical operation of a discrete Mamba step is the Selective Scan:
$$ h_t = \bar{A} h_{t-1} + \bar{B} x_t $$
$$ y_t = C h_t $$

In **Bidirectional Mamba**, we run two selective scans:
1. **Forward Scan:** $h^{fwd}_t = \bar{A}^{fwd} h^{fwd}_{t-1} + \bar{B}^{fwd} x_t$
2. **Backward Scan:** $h^{bwd}_t = \bar{A}^{bwd} h^{bwd}_{t+1} + \bar{B}^{bwd} x_t$

The outputs $y^{fwd}_t$ and $y^{bwd}_t$ are then aggregated (often concatenated or summed) and passed through a projection layer.

## Why Bidirectionality for Time-Series?

While time is strictly directional, the relationships *within a historical lookback window* are complex and interconnected. For instance, understanding the context of a sudden volume spike at $t-50$ might require analyzing the price consolidation between $t-20$ and $t$. 
By scanning backward from $t$ to $t-H$, the model's hidden state directly carries the most recent (and often most relevant) events $t, t-1, t-2...$ deep into the sequence representations instead of forgetting them.

Architectures like **CMDMamba** and **SAMBA** validate that utilizing a bidirectional SSM block dramatically increases downstream forecasting accuracy while retaining the $O(N)$ efficiency advantage over Transformer models.

## The Trade-Offs: Causal Inference vs Context Representation

- **Standard Mamba (Unidirectional):** Computes $O(1)$ updates per tick. Ideal for streaming high-frequency trading where you add exactly one tick at a time.
- **Bidirectional Mamba:** Computes an $O(N)$ sweep over the entire context window $H$ at each bar. It cannot run in true $O(1)$ streaming mode because the backward pass requires seeing from the end of the window to the beginning. However, for Bar-based trading (e.g., 5-minute bars or Daily bars), $O(N)$ is trivially cheap compared to the Transformer's $O(N^2)$, making it the ultimate tool for heavy representation extraction.

---

## Python Implementation Details

The Python implementations leverage PyTorch to build a custom `BidirectionalMambaBlock`.
- **File:** [`python/model.py`](python/model.py): Implements the core Bidirectional SSD mechanisms showing the forward and backward unrolling.
- **File:** [`python/train.py`](python/train.py): A multi-epoch training script that simulates processing historical context windows to predict the future market return.
- **File:** [`python/notebooks/example.ipynb`](python/notebooks/example.ipynb): Interactive exploration of the architecture.

**Usage Snippet:**
```bash
python python/model.py
python python/train.py
```

## Rust Implementation Details

The Rust engine demonstrates the implementation of a bidirectional sweep in high-performance environments using `ndarray`.
- **Core Library:** [`rust/src/lib.rs`](rust/src/lib.rs)
- **Execution Script:** [`rust/src/main.rs`](rust/src/main.rs)

The code mimics how a system would pull a historical buffer of $N$ floats and quickly extract the bidirectional fused representation.
```bash
cd rust
cargo run
```

## Backtesting Methodology

When backtesting Bidirectional Mamba, strict Walk-Forward validation is required. Since the model sees "forward" *within its context window*, any data leakage from the test set into the window will result in catastrophic overfitting.
- **File:** [`python/backtest.py`](python/backtest.py): Implements a strictly partitioned rolling-window prediction engine, ensuring the model's $t+1$ target is never included in the bidirectional pool.

**Usage Snippet:**
```bash
python python/backtest.py
```

---

## References

1. **Zhu, L., et al.** (2024). Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model. *ICML 2024*. [arXiv:2401.09417](https://arxiv.org/abs/2401.09417)
2. **Gu, A., & Dao, T.** (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
3. Advanced applications like CMDMamba and HIGSTM for financial time series prediction.
