"""
Synthetic Control Method for Trading.

This package provides tools for applying the Synthetic Control Method
to financial event studies and trading strategies.

Modules:
- synthetic_control: Core SCM model with weight optimization
- data_loader: Data fetching from Bybit and Yahoo Finance
- backtest: Backtesting framework for event-driven strategies
"""

from python.synthetic_control import SyntheticControlModel
from python.data_loader import SCMDataLoader, fetch_bybit_klines, fetch_yfinance_data
from python.backtest import SCMEventBacktester

__all__ = [
    "SyntheticControlModel",
    "SCMDataLoader",
    "SCMEventBacktester",
    "fetch_bybit_klines",
    "fetch_yfinance_data",
]
