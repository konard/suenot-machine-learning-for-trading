"""
Attention Rollout Trading Module

This module provides attention rollout computation for interpretable
trading models using transformer architectures.
"""

from .attention_rollout import AttentionRollout, TradingAttentionRollout
from .model import TradingTransformer
from .data_loader import load_stock_data, load_bybit_data, prepare_features
from .backtest import AttentionBacktester, BacktestResult, print_backtest_report

__all__ = [
    "AttentionRollout",
    "TradingAttentionRollout",
    "TradingTransformer",
    "load_stock_data",
    "load_bybit_data",
    "prepare_features",
    "AttentionBacktester",
    "BacktestResult",
    "print_backtest_report",
]

__version__ = "0.1.0"
