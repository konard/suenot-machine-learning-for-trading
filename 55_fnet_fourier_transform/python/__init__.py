"""
FNet: Fourier Transform for Efficient Token Mixing

This module provides an implementation of FNet architecture adapted for
financial time series prediction.

Key Components:
- FNet: Main model architecture
- FourierLayer: Replaces attention with FFT
- BybitDataLoader: Data fetching from Bybit
- FNetTradingStrategy: Trading signal generation
- Backtester: Strategy backtesting

Example:
    >>> from fnet import FNet, BybitDataLoader, Backtester
    >>> model = FNet(n_features=7, d_model=256)
    >>> loader = BybitDataLoader()
    >>> df = loader.fetch_klines("BTCUSDT")
"""

from .model import (
    FourierLayer,
    FNetEncoderBlock,
    FNet,
    MultiFNet,
)

from .data import (
    BybitDataLoader,
    YahooDataLoader,
    create_sequences,
    normalize_features,
)

from .strategy import (
    FNetTradingStrategy,
    Backtester,
    calculate_metrics,
)

__version__ = "1.0.0"
__author__ = "Machine Learning for Trading"
__all__ = [
    "FourierLayer",
    "FNetEncoderBlock",
    "FNet",
    "MultiFNet",
    "BybitDataLoader",
    "YahooDataLoader",
    "create_sequences",
    "normalize_features",
    "FNetTradingStrategy",
    "Backtester",
    "calculate_metrics",
]
