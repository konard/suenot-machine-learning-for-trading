"""
MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting

This module implements the MambaTS architecture for financial time series prediction,
combining the efficiency of Mamba's selective state spaces with time series-specific
innovations like variable-aware scanning and temporal patching.
"""

from .model import MambaTS
from .mamba_block import MambaBlock
from .data_loader import FinancialDataLoader
from .backtest import Backtester, Strategy

__version__ = "0.1.0"
__all__ = ["MambaTS", "MambaBlock", "FinancialDataLoader", "Backtester", "Strategy"]
