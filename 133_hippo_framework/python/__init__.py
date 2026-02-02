"""
HiPPO Framework for Trading.

This package implements the HiPPO (High-order Polynomial Projection Operators)
framework for algorithmic trading, providing optimal polynomial projections
for long-range memory in time series modeling.

Modules:
    hippo_model: Core HiPPO operators and trading model
    data_loader: Data fetching from Bybit and yfinance
    backtest: Backtesting framework with performance metrics
"""

from .hippo_model import HiPPOLegS, HiPPOLagT, HiPPOTradingModel
from .data_loader import BybitDataLoader
from .backtest import BacktestEngine
