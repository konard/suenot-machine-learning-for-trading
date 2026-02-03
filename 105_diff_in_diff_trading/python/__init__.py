"""Difference-in-Differences (DiD) for Trading.

This module implements DiD estimation methods for financial markets,
including event studies, causal effect estimation, and trading strategies.
"""

from .model import DifferenceInDifferences, DiDResults
from .data_loader import DataLoader, BybitDataLoader, YFinanceDataLoader
from .backtest import Backtester, TradingStrategy

__all__ = [
    "DifferenceInDifferences",
    "DiDResults",
    "DataLoader",
    "BybitDataLoader",
    "YFinanceDataLoader",
    "Backtester",
    "TradingStrategy",
]
