"""
S4 Trading module.

This module provides S4 (Structured State Space) models for algorithmic trading.
S4 efficiently handles long-range dependencies in financial time series.
"""

from .s4_model import S4Layer, S4DLayer, S4TradingModel, S4Config
from .data_loader import BybitDataLoader, MarketData, load_stock_data, generate_synthetic_data
from .backtest import BacktestEngine, BacktestResult, BacktestConfig

__all__ = [
    "S4Layer",
    "S4DLayer",
    "S4TradingModel",
    "S4Config",
    "BybitDataLoader",
    "MarketData",
    "load_stock_data",
    "generate_synthetic_data",
    "BacktestEngine",
    "BacktestResult",
    "BacktestConfig",
]
