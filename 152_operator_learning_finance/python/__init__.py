"""
Chapter 152: Operator Learning for Finance

This package implements neural operator learning architectures (DeepONet, FNO)
for financial applications including option pricing, yield curve evolution,
volatility surface dynamics, and crypto trading with Bybit data.

Key modules:
    - model: DeepONet and FNO architectures
    - train: Training loop with multi-resolution strategy
    - data_loader: Market data fetching (stocks + Bybit crypto)
    - visualize: Operator learning visualization
    - backtest: Trading strategy backtesting
"""

__version__ = "0.1.0"
__author__ = "ML Trading Examples"

from .model import (
    DeepONet,
    DeepONetConfig,
    FourierNeuralOperator,
    FNOConfig,
    OperatorType,
)

__all__ = [
    "DeepONet",
    "DeepONetConfig",
    "FourierNeuralOperator",
    "FNOConfig",
    "OperatorType",
]
