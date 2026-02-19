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

### From Attention to Linear Attention
Linear attention replaces the softmax kernel with feature maps, enabling the attention mechanism to be expressed as an RNN, meaning state space updates happen in $O(1)$ recurrent time.

### State Space Models (SSM) in Finance
SSMs can elegantly manage the continuous, high-frequency nature of financial data, making them ideal for Limit Order Books (LOBs) and tick-level forecasting.

### The Connection: Transformers are SSMs
Recent research demonstrates that linear attention and SSMs are structurally connected. This equivalence provides new insights for developing fast, sequential models specifically designed for trading tasks such as prediction on Limit Order Book (LOB) data or long-horizon asset forecasting.

## Key Architectures

### Generalized Linear Attention
Allows different kernel functions to replace the exponential kernel found in standard dot-product attention, balancing efficiency and approximation accuracy.

### Efficient Algorithms for Trading Applications
By mapping Transformers to SSMs, we can perform inference without recomputing attention scores across entire histories.

## Code Examples

### 01: PyTorch Model Implementation
The fundamental block of the architecture is developed in Python. It includes building custom layers where linear attention is mapped efficiently onto state spaces.
- See: [`python/model.py`](python/model.py)

**Usage Snippet:**
```bash
python python/model.py
```

### 02: Training and Testing on Financial Data
Implementation using datasets like Yahoo Finance, Bybit API, and LOBSTER. This stage measures performance using MSE/MAE and accuracy scores. Note that cryptocurrency data is pulled from Bybit's API rather than Binance.
- See: [`python/train.py`](python/train.py) and [`python/notebooks/example.ipynb`](python/notebooks/example.ipynb)

**Usage Snippet:**
```bash
python python/train.py
```

### 03: Backtesting Framework
Demonstrating profitability and risk metrics using Backtrader or Zipline. Key metrics include Sharpe Ratio, Sortino Ratio, and Maximum Drawdown to ensure real-world viability compared to baseline sequence models.
- See: [`python/backtest.py`](python/backtest.py)

**Usage Snippet:**
```bash
python python/backtest.py
```

## Rust Implementation

For production-ready trading systems requiring minimal latency, a Rust execution engine is built utilizing libraries such as `ndarray`, `polars`, and `burn` (or `candle`). This implementation achieves optimal real-time inference handling.
- Library: [`rust/src/lib.rs`](rust/src/lib.rs)
- Inference Binary: [`rust/src/main.rs`](rust/src/main.rs)

**Usage Snippet:**
```bash
cd rust
cargo run
```

## Evaluation Metrics

This section details how SSM-based models are measured against standard baselines:
- Accuracy / F1-score for market direction classification
- MSE / MAE for price prediction regression
- Sharpe Ratio, Sortino Ratio, Maximum Drawdown for strategy risk

## Resources

### Papers
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through the Lens of Information Retrieval](https://arxiv.org/abs/2405.21060), 2024

### Related Chapters
- [Chapter 133: HIPPO Framework](../133_hippo_framework)
- [Chapter 135: Bidirectional Mamba](../135_bidirectional_mamba)
