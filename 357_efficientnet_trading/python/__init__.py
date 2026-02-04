"""
EfficientNet Trading Module

This module provides implementations for EfficientNet-based trading strategies
using time series to image transformations (GAF, MTF, Spectrograms).

Modules:
    - data_loader: Data fetching from Bybit via CCXT
    - image_transform: Time series to image transformations
    - model: EfficientNet trading model
    - train: Training utilities
    - backtest: Backtesting framework
    - visualize: Attention visualization (Grad-CAM)
"""

from .data_loader import BybitDataLoader, CCXTDataLoader, DataManager
from .image_transform import GAFTransformer, MTFTransformer, ImageStackGenerator
from .model import EfficientNetTrader, TradingHead
from .backtest import EfficientNetBacktest, BacktestResult

__version__ = "0.1.0"
__all__ = [
    "BybitDataLoader",
    "CCXTDataLoader",
    "DataManager",
    "GAFTransformer",
    "MTFTransformer",
    "ImageStackGenerator",
    "EfficientNetTrader",
    "TradingHead",
    "EfficientNetBacktest",
    "BacktestResult",
]
