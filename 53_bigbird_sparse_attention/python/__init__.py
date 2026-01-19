"""
BigBird Sparse Attention for Trading

This module provides:
- BigBirdConfig: Model configuration
- BigBirdForTrading: Main model for financial time series
- BigBirdSparseAttention: Sparse attention mechanism
- Data utilities for Bybit and stock market data
"""

from .model import (
    BigBirdConfig,
    BigBirdForTrading,
    BigBirdSparseAttention,
    BigBirdEncoderLayer,
)
from .data import (
    fetch_bybit_data,
    fetch_stock_data,
    prepare_features,
    create_sequences,
    TradingDataset,
)
from .strategy import (
    backtest_strategy,
    calculate_metrics,
    generate_signals,
)

__version__ = "1.0.0"
__all__ = [
    "BigBirdConfig",
    "BigBirdForTrading",
    "BigBirdSparseAttention",
    "BigBirdEncoderLayer",
    "fetch_bybit_data",
    "fetch_stock_data",
    "prepare_features",
    "create_sequences",
    "TradingDataset",
    "backtest_strategy",
    "calculate_metrics",
    "generate_signals",
]
