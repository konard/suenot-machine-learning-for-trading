"""
C-Mamba: Channel Correlation Enhanced State Space Models for Trading

This module provides implementations for multivariate time series forecasting
using C-Mamba architecture, specifically designed for financial trading applications.

Modules:
    - cmamba_model: Core C-Mamba implementation
    - data_loader: Data loading from Bybit and Yahoo Finance
    - backtest: Backtesting framework for trading strategies
    - train: Training utilities and helpers
"""

from .cmamba_model import CMamba, CMambaConfig
from .data_loader import MultiAssetDataLoader
from .backtest import CMambaBacktester

__all__ = [
    "CMamba",
    "CMambaConfig",
    "MultiAssetDataLoader",
    "CMambaBacktester",
]

__version__ = "0.1.0"
