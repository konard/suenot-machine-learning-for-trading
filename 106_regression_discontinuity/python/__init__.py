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
)
from .data_loader import RDDDataLoader
from .backtest import RDDBacktester, RDDStrategy

__all__ = [
    "RegressionDiscontinuity",
    "RDDResults",
    "RDDValidator",
    "Kernel",
    "RDDDataLoader",
    "RDDBacktester",
    "RDDStrategy",
]

__version__ = "0.1.0"
