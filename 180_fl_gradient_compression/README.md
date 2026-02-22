# Chapter 180: Ternary Gradient Compression for Trading

## Overview

Sending thousands of model updates across a global trading network is a bandwidth nightmare. In the previous chapter, we explored basic quantization. Now, we implement **Ternary Gradient Compression** (TGC).

TGC reduces each weight update to just one of three values: $\{-1, 0, +1\}$. This not only compresses the data but also acts as a powerful regularizer, filtering out market noise.

## Adaptive Thresholding

Markets are not static. During a high-volatility event, small gradient updates might be crucial. During a sideways market, most updates are just noise.
Our implementation uses an **Adaptive Threshold**:
- **High Volatility**: The threshold drops to capture more details.
- **Low Volatility**: The threshold rises to maximize sparsity and save bandwidth.

## Project Structure

```
180_fl_gradient_compression/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py            # Base Neural Network
│   ├── ternary_core.py     # TGC & Adaptive Logic
│   └── train.py            # Adaptive vs. Fixed compression simulation
└── rust/src/
    └── lib.rs              # Optimized 2-bit packing engine
```
