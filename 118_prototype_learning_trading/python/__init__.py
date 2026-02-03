"""
Prototype Learning for Trading

This module provides implementations of prototype-based learning models
for market state classification and trading signal generation.
"""

from .model import ProtoPNet, MarketState
from .data_loader import BybitDataLoader, StockDataLoader, FeatureEngineering
from .train import ProtoPNetTrainer
from .backtest import PrototypeBacktester

__all__ = [
    'ProtoPNet',
    'MarketState',
    'BybitDataLoader',
    'StockDataLoader',
    'FeatureEngineering',
    'ProtoPNetTrainer',
    'PrototypeBacktester',
]

__version__ = '0.1.0'
