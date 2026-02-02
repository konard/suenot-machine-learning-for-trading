"""
PCMCI Causal Discovery for Trading

This module provides implementations of:
- PCMCICausalDiscovery: PCMCI algorithm for time-lagged causal discovery
- Data loaders for yfinance and Bybit
- Causal trading strategy and backtesting utilities

The PCMCI algorithm (Runge et al., 2019) combines the PC algorithm's
condition-selection phase with the MCI test to robustly discover
causal links in multivariate time series.
"""

from .model import (
    PCMCIConfig,
    PCMCICausalDiscovery,
    IndependenceTest,
)
from .data import (
    load_stock_data,
    load_bybit_data,
    create_sequences,
    prepare_causal_data,
)
from .strategy import (
    BacktestConfig,
    CausalTradingStrategy,
    Backtester,
)

__all__ = [
    # Model
    "PCMCIConfig",
    "PCMCICausalDiscovery",
    "IndependenceTest",
    # Data
    "load_stock_data",
    "load_bybit_data",
    "create_sequences",
    "prepare_causal_data",
    # Strategy
    "BacktestConfig",
    "CausalTradingStrategy",
    "Backtester",
]
