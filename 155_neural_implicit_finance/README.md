# Chapter 155: Neural Implicit Finance (INRs & SIREN)

## Overview

Traditional financial modeling represents data on discrete grids (e.g., historical prices at specific timestamps, volatility surfaces at specific strike/maturity grid points). **Implicit Neural Representations (INRs)** change this paradigm by representing financial data as a **continuous function** parameterized by a neural network.

In this chapter, we explore **SIREN (Sinusoidal Representation Networks)** to model the **Implied Volatility (IV) Surface**. Unlike standard ReLU-based networks, SIREN uses periodic activation functions, allowing it to capture high-frequency details and, more importantly, its derivatives (Greeks) with extreme accuracy.

## Why Implicit Representations for Finance?

1. **Resolution Independence**: Once trained, you can query the model at *any* coordinate (moneyness, time-to-expiry), not just the grid points used for training.
2. **Analytical Greeks via Autograd**: Since the model is a continuous, differentiable function, Greeks like Delta ($\Delta$), Gamma ($\Gamma$), and Vega ($\nu$) can be calculated directly via backpropagation through the network coordinates.
3. **Arbitrage-Free Constraints**: Real-world volatility surfaces must satisfy no-arbitrage conditions (butterfly, calendar). INRs allow us to incorporate these constraints directly into the loss function or architecture.
4. **Memory Efficiency**: A complex volatility surface can be compressed into the weights of a small MLP.

## SIREN: The Periodic Powerhouse

SIREN uses the $\text{sin}(\omega_0 \cdot \phi)$ activation function. This is critical because:
- The derivative of a sine is a cosine (another periodic function).
- This allows the network to maintain its "expressiveness" through multiple layers of differentiation.
- It is uniquely suited for modeling surfaces where smooth second-order derivatives (like Gamma) are required.

---

## Contents

- **`python/model.py`**: Implementation of SIREN architecture in PyTorch.
- **`python/train.py`**: Fitting sparse market quotes to a continuous IV surface.
- **`python/backtest.py`**: Calculating Greeks and verifying arbitrage constraints via autograd.
- **`rust/src/`**: Optimized Rust implementation for real-time inference and calibration.

---

## References

1. Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020). *Implicit Neural Representations with Periodic Activation Functions.* [arXiv:2006.09661](https://arxiv.org/abs/2006.09661).
2. *HyperIV: Real-time implied volatility smoothing.* University of Edinburgh research.
