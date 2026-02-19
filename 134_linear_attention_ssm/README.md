# Linear Attention and State Space Models (SSM)

This chapter explores the connection between linear attention mechanisms and state space models (SSMs) for financial time series applications. It bridges the gap between efficient attention computations and continuous-time dynamical systems.

## Content

1. [Theoretical Foundations](#theoretical-foundations)
    * [From Attention to Linear Attention](#from-attention-to-linear-attention)
    * [State Space Models (SSM) in Finance](#state-space-models-ssm-in-finance)
    * [The Connection: Transformers are SSMs](#the-connection-transformers-are-ssms)
2. [Key Architectures](#key-architectures)
    * [Generalized Linear Attention](#generalized-linear-attention)
    * [Efficient Algorithms for Trading Applications](#efficient-algorithms-for-trading-applications)
3. [Code Examples](#code-examples)
    * [01: PyTorch Model Implementation](#01-pytorch-model-implementation)
    * [02: Training and Testing on Financial Data](#02-training-and-testing-on-financial-data)
    * [03: Backtesting Framework](#03-backtesting-framework)
4. [Rust Implementation](#rust-implementation)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Resources](#resources)

## Theoretical Foundations

Linear attention reformulates the standard $O(N^2)$ dot-product attention to an $O(N)$ complexity algorithm by decomposing the softmax operation and utilizing associative properties of matrix multiplication. 
Simultaneously, Sequence-to-Sequence State Space Models have shown tremendous capacity in capturing long-term dependencies efficiently.

### The Connection: Transformers are SSMs

Recent research demonstrates that linear attention and SSMs are structurally connected. This equivalence provides new insights for developing fast, sequential models specifically designed for trading tasks such as prediction on Limit Order Book (LOB) data or long-horizon asset forecasting.

## Code Examples

### 01: PyTorch Model Implementation
The fundamental block of the architecture is developed in Python. It includes building custom layers where linear attention is mapped efficiently onto state spaces.
- See: [`python/model.py`](python/model.py)

### 02: Training and Testing on Financial Data
Implementation using datasets like Yahoo Finance, Binance API, and LOBSTER. This stage measures:
- Accuracy / F1-score for market direction classification
- MSE / MAE for price prediction regression
- See: [`python/train.py`](python/train.py) and [`python/notebooks/example.ipynb`](python/notebooks/example.ipynb)

### 03: Backtesting Framework
Demonstrating profitability and risk metrics using Backtrader or Zipline. Key metrics include Sharpe Ratio, Sortino Ratio, and Maximum Drawdown to ensure real-world viability compared to baseline sequence models.
- See: [`python/backtest.py`](python/backtest.py)

## Rust Implementation

For production-ready trading systems requiring minimal latency, a Rust execution engine is built utilizing libraries such as `ndarray`, `polars`, and `burn` (or `candle`). This implementation achieves optimal real-time inference handling.
- See: [`rust/src/lib.rs`](rust/src/lib.rs)

## Resources

### Papers
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through the Lens of Information Retrieval](https://arxiv.org/abs/2405.21060), 2024

### Related Chapters
- [Chapter 133: HIPPO Framework](../133_hippo_framework)
- [Chapter 135: Bidirectional Mamba](../135_bidirectional_mamba)
