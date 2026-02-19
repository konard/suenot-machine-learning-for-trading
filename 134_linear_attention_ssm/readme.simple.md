# Linear Attention and SSM (Simple Guide)

This is a simplified guide for Chapter 134, focusing on practical usage of Linear Attention State Space Models (SSMs).

## Content
1. Theoretical Foundations
2. Key Architectures
3. Code Examples
4. Rust Implementation
5. Evaluation Metrics
6. Resources

## Code Examples
- PyTorch implementation: [`python/model.py`](python/model.py)
```bash
python python/model.py
```

- Training (Bybit Crypto & Stocks): [`python/train.py`](python/train.py), [`python/notebooks/example.ipynb`](python/notebooks/example.ipynb)
```bash
python python/train.py
```

- Backtest: [`python/backtest.py`](python/backtest.py)
```bash
python python/backtest.py
```

## Rust Implementation
We provide high-performance code in Rust for processing Bybit cryptocurrency data and stock market data continuously using state space models.
- Core: [`rust/src/lib.rs`](rust/src/lib.rs)
- Runner: [`rust/src/main.rs`](rust/src/main.rs)

**How to run:**
```bash
cd rust
cargo run
```
