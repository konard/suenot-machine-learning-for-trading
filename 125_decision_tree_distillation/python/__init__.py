"""
Decision Tree Distillation for Trading

This module provides tools for distilling complex machine learning models
into interpretable decision trees for algorithmic trading applications.

Modules:
    model: Core distillation implementation
    backtest: Backtesting framework for distilled models
    data_loader: Data fetching utilities for stocks and crypto
"""

from .model import DistillationModel, DistilledRule, DistillationResult
from .backtest import DistillationBacktester, BacktestResult
from .data_loader import load_stock_data, load_crypto_data, compute_features

__all__ = [
    'DistillationModel',
    'DistilledRule',
    'DistillationResult',
    'DistillationBacktester',
    'BacktestResult',
    'load_stock_data',
    'load_crypto_data',
    'compute_features',
]

__version__ = '0.1.0'
