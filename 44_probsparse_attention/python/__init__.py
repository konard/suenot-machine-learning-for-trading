"""
ProbSparse Attention Implementation for Trading

This module provides:
- InformerConfig: Model configuration
- InformerModel: Main Informer model with ProbSparse attention
- ProbSparseAttention: Efficient sparse attention mechanism
- AttentionDistilling: Sequence length reduction layer
- Data loading utilities for Bybit and stock data
- Backtesting framework for trading strategies
"""

from .model import (
    InformerConfig,
    InformerModel,
    ProbSparseAttention,
    AttentionDistilling,
    TokenEmbedding,
    PositionalEncoding,
    EncoderLayer,
)

from .data import (
    BybitDataLoader,
    StockDataLoader,
    TimeSeriesDataset,
    prepare_informer_data,
)

from .strategy import (
    InformerStrategy,
    BacktestEngine,
    BacktestResult,
    SignalGenerator,
)

__version__ = "0.1.0"
__all__ = [
    # Model
    "InformerConfig",
    "InformerModel",
    "ProbSparseAttention",
    "AttentionDistilling",
    "TokenEmbedding",
    "PositionalEncoding",
    "EncoderLayer",
    # Data
    "BybitDataLoader",
    "StockDataLoader",
    "TimeSeriesDataset",
    "prepare_informer_data",
    # Strategy
    "InformerStrategy",
    "BacktestEngine",
    "BacktestResult",
    "SignalGenerator",
]
