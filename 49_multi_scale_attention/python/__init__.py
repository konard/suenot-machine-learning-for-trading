"""
Multi-Scale Attention for Financial Time Series.

This module provides a PyTorch implementation of multi-scale attention
mechanisms for financial time series prediction. It captures patterns
across different temporal resolutions simultaneously.
"""

from .model import (
    MultiScaleAttention,
    ScaleEncoder,
    MultiResolutionAttention,
    CrossScaleFusion,
)
from .data import (
    fetch_bybit_klines,
    create_multi_scale_features,
    create_sequences,
    MultiScaleDataset,
)
from .strategy import (
    BacktestResult,
    backtest_multi_scale_strategy,
    compare_scale_strategies,
)

__version__ = "0.1.0"
__all__ = [
    "MultiScaleAttention",
    "ScaleEncoder",
    "MultiResolutionAttention",
    "CrossScaleFusion",
    "fetch_bybit_klines",
    "create_multi_scale_features",
    "create_sequences",
    "MultiScaleDataset",
    "BacktestResult",
    "backtest_multi_scale_strategy",
    "compare_scale_strategies",
]
