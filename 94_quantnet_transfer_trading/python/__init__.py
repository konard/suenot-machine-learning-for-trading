"""
QuantNet Transfer Trading.

This package provides:
- QuantNetEncoder: Shared encoder with BatchNorm, GELU, Dropout
- QuantNetDecoder: Mirror decoder for reconstruction pretraining
- TradingHead: Asset-specific signal generator (tanh output in [-1, 1])
- QuantNet: Complete model with encoder, decoder, and ModuleDict of heads
- SharpeRatioLoss: Differentiable Sharpe ratio loss
- QuantNetTrainer: Two-phase training (pretrain + finetune)
- Data loading utilities for crypto (Bybit) and stock (yfinance) markets
- Backtesting framework for QuantNet strategies
"""

from .quantnet_model import (
    QuantNetEncoder,
    QuantNetDecoder,
    TradingHead,
    QuantNet,
    SharpeRatioLoss,
    QuantNetTrainer,
)
from .data_loader import (
    BybitDataLoader,
    StockDataLoader,
    create_features,
)
from .backtest import QuantNetBacktester, calculate_metrics

__all__ = [
    "QuantNetEncoder",
    "QuantNetDecoder",
    "TradingHead",
    "QuantNet",
    "SharpeRatioLoss",
    "QuantNetTrainer",
    "BybitDataLoader",
    "StockDataLoader",
    "create_features",
    "QuantNetBacktester",
    "calculate_metrics",
]
