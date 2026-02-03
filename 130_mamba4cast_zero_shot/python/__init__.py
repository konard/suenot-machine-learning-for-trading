"""
Mamba4Cast Zero-Shot Forecasting

A zero-shot time series forecasting implementation based on the Mamba
state space model architecture.

Modules:
    mamba4cast_model: Core Mamba4Cast architecture
    synthetic_data: Synthetic data generation for training
    data_loader: Data loading utilities for Yahoo Finance and Bybit
    features: Feature preprocessing utilities
    forecast: Zero-shot forecasting utilities
    backtest: Backtesting framework
"""

from .mamba4cast_model import (
    Mamba4CastBlock,
    Mamba4CastForecaster,
    create_mamba4cast_model,
)
from .data_loader import StockDataLoader, BybitDataLoader
from .synthetic_data import SyntheticDataGenerator
from .forecast import ZeroShotForecaster
from .backtest import Mamba4CastBacktest

__version__ = "1.0.0"
__all__ = [
    "Mamba4CastBlock",
    "Mamba4CastForecaster",
    "create_mamba4cast_model",
    "StockDataLoader",
    "BybitDataLoader",
    "SyntheticDataGenerator",
    "ZeroShotForecaster",
    "Mamba4CastBacktest",
]
