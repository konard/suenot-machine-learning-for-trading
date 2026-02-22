# Chapter 179: Communication-Efficient Federated Learning

## Overview

In global trading networks, bandwidth is a precious resource. Sending full 32-bit model updates from thousands of clients across international borders is slow and expensive.

**Communication-Efficient Federated Learning** aims to reduce the synchronization payload without sacrificing the predictive power of the model.

## Core Techniques

### 1. Gradient Quantization
Instead of sending precise 32-bit floats, we map values into a smaller set of discrete levels (e.g., 8-bit, 4-bit, or even 1-bit).
- **SignSGD**: Only the sign of the gradient (+1 or -1) is sent. This reduces each parameter update to a single bit.

### 2. Sparsification (Top-K Updates)
Trading models often have sparse updates. Instead of sending the entire model, we only transmit the most significant changes (Top-K gradients) and keep the rest as zero.

### 3. Error Compensation
To prevent information loss from compression, clients accumulate the "difference" between compressed and original gradients and add it to the next update.

## Project Structure

```
179_communication_efficient_fl/
├── README.md           # English Overview
├── README.ru.md        # Russian Overview
├── docs/ru/theory.md   # Mathematical deep-dive
├── python/
│   ├── model.py            # Base Neural Network
│   ├── compression_core.py # Quantization & Sparsification logic
│   └── train.py            # Dense vs. Compressed simulation
└── rust/src/
    └── lib.rs              # Optimized bit-packing engine
```
