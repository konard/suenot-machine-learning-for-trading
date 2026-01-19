"""
BloombergGPT Trading - Financial LLM for Trading Applications

This module provides tools for financial sentiment analysis, trading signal
generation, and backtesting using LLM-based approaches inspired by BloombergGPT.

Since BloombergGPT is proprietary, this implementation uses open-source alternatives
(FinBERT, FinGPT) with a similar interface for demonstration purposes.
"""

from .sentiment import FinancialSentimentAnalyzer
from .signals import TradingSignal, LLMSignalGenerator
from .impact import NewsImpactPredictor
from .backtest import LLMSignalBacktester, BacktestConfig, BacktestResult

__all__ = [
    "FinancialSentimentAnalyzer",
    "TradingSignal",
    "LLMSignalGenerator",
    "NewsImpactPredictor",
    "LLMSignalBacktester",
    "BacktestConfig",
    "BacktestResult",
]

__version__ = "0.1.0"
