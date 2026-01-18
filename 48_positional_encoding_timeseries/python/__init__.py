"""
Chapter 48: Positional Encoding for Time Series

This module provides various positional encoding implementations
specifically designed for time series and financial data.

Available encodings:
- SinusoidalPositionalEncoding: Fixed sinusoidal encoding
- LearnedPositionalEncoding: Trainable position embeddings
- RelativePositionalEncoding: Shaw-style relative encoding
- RotaryPositionalEncoding: RoPE for efficient long sequences
- CalendarEncoding: Calendar-aware features for trading
- MarketSessionEncoding: Trading session information
- MultiScaleTemporalEncoding: Multi-frequency temporal features
"""

from .positional_encoding import (
    SinusoidalPositionalEncoding,
    TimeSeriesSinusoidalEncoding,
    LearnedPositionalEncoding,
    FinancialLearnedEncoding,
    RelativePositionalEncoding,
    RotaryPositionalEncoding,
    RoPETimeSeriesAttention,
    CalendarEncoding,
    MarketSessionEncoding,
    MultiScaleTemporalEncoding,
)

from .model import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformer,
)

from .data import (
    BybitDataLoader,
    prepare_features,
    create_sequences,
)

from .strategy import (
    BacktestResult,
    TradingStrategy,
    run_backtest,
)

__version__ = "1.0.0"
__all__ = [
    # Encodings
    "SinusoidalPositionalEncoding",
    "TimeSeriesSinusoidalEncoding",
    "LearnedPositionalEncoding",
    "FinancialLearnedEncoding",
    "RelativePositionalEncoding",
    "RotaryPositionalEncoding",
    "RoPETimeSeriesAttention",
    "CalendarEncoding",
    "MarketSessionEncoding",
    "MultiScaleTemporalEncoding",
    # Model
    "TimeSeriesTransformerConfig",
    "TimeSeriesTransformer",
    # Data
    "BybitDataLoader",
    "prepare_features",
    "create_sequences",
    # Strategy
    "BacktestResult",
    "TradingStrategy",
    "run_backtest",
]
