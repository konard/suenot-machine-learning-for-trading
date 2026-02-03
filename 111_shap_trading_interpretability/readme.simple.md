# SHAP for Trading - Simple Guide

## What is SHAP?

SHAP (SHapley Additive exPlanations) helps you understand *why* a trading model makes predictions. Instead of just getting "buy" or "sell" signals, SHAP tells you which features contributed to that decision.

## Simple Example

Imagine your model says "BUY Bitcoin" with 70% confidence. SHAP explains:

```
Signal: BUY (70%)
Why?
  +15% from RSI (oversold condition)
  +12% from MACD (bullish crossover)
  +8% from Volume (high buying pressure)
  -5% from Volatility (some uncertainty)
```

Now you know the signal is driven by technical indicators, not just a black box.

## Quick Start

### Python

```bash
cd 111_shap_trading_interpretability/python
pip install -r requirements.txt
python shap_model.py
```

### Rust

```bash
cd 111_shap_trading_interpretability
cargo run --example basic_shap
```

## Key Concepts

### 1. Base Value
The average prediction across all data. Think of it as the "default" before looking at features.

### 2. SHAP Values
How much each feature pushes the prediction up or down from the base.

### 3. Feature Importance
Which features matter most (average of absolute SHAP values).

## Trading Use Cases

| Use Case | How SHAP Helps |
|----------|----------------|
| Trust signals | See which features drive the prediction |
| Debug models | Find if model relies on spurious patterns |
| Risk management | Understand what causes risky predictions |
| Feature selection | Keep only features that matter |

## Files in This Chapter

```
111_shap_trading_interpretability/
├── README.md           # Full documentation
├── python/
│   ├── shap_model.py   # SHAP implementation
│   ├── data_loader.py  # Bybit/yfinance data
│   └── backtest.py     # Strategy testing
├── src/                # Rust implementation
└── examples/           # Rust examples
```

## Example Output

```
Latest Signal Explanation:
  Prediction: 0.682 (bullish)
  Base value: 0.500

  Top contributing features:
    returns_1: +0.0523 (recent momentum)
    rsi_14: +0.0412 (oversold bounce)
    volume_ma_ratio: +0.0287 (high volume)
    volatility_10: -0.0156 (some caution)
```

## Learn More

- Full documentation: [README.md](README.md)
- Original paper: [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
- SHAP library: [github.com/slundberg/shap](https://github.com/slundberg/shap)
