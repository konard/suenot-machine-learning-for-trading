"""
Linformer: Self-Attention with Linear Complexity

This module provides a PyTorch implementation of Linformer for
efficient processing of long financial time series sequences.

Key components:
- LinformerAttention: Linear complexity attention mechanism
- Linformer: Complete model for financial forecasting
- Data utilities for Bybit integration
- Backtesting framework
"""

from .model import (
    LinformerAttention,
    LinformerEncoderLayer,
    Linformer
)
from .data import (
    load_bybit_data,
    prepare_long_sequence_data,
    calculate_features
)
from .strategy import (
    backtest_linformer_strategy,
    calculate_sharpe_ratio,
    calculate_max_drawdown
)

__version__ = "1.0.0"
__all__ = [
    "LinformerAttention",
    "LinformerEncoderLayer",
    "Linformer",
    "load_bybit_data",
    "prepare_long_sequence_data",
    "calculate_features",
    "backtest_linformer_strategy",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown"
]
