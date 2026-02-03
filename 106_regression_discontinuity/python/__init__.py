"""
Regression Discontinuity Design (RDD) for Trading

This module provides tools for causal inference using regression discontinuity
design in financial markets. It supports analysis of threshold-based effects
such as index inclusion, technical indicator signals, and regulatory thresholds.

Components:
- rdd_model: Core RDD estimation and validation
- data_loader: Data fetching from Yahoo Finance and Bybit
- backtest: Backtesting framework for RDD-based strategies
"""

from .rdd_model import (
    RegressionDiscontinuity,
    RDDResults,
    RDDValidator,
    Kernel,
    compute_rsi,
)
from .data_loader import (
    BybitClient,
    MarketData,
    Candle,
    FundingRate,
    fetch_stock_data,
    generate_synthetic_data,
    generate_synthetic_funding_rates,
)
from .backtest import (
    RDDBacktester,
    BacktestResults,
    Trade,
    SignalType,
)

__all__ = [
    # RDD Model
    "RegressionDiscontinuity",
    "RDDResults",
    "RDDValidator",
    "Kernel",
    "compute_rsi",
    # Data Loading
    "BybitClient",
    "MarketData",
    "Candle",
    "FundingRate",
    "fetch_stock_data",
    "generate_synthetic_data",
    "generate_synthetic_funding_rates",
    # Backtesting
    "RDDBacktester",
    "BacktestResults",
    "Trade",
    "SignalType",
]

__version__ = "0.1.0"
