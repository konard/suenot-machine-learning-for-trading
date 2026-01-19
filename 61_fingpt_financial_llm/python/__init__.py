"""
FinGPT Trading - Open-Source Financial LLM Toolkit

This package provides tools for financial sentiment analysis, trading signal
generation, and backtesting using FinGPT-style approaches.

Modules:
    fingpt_sentiment: Financial sentiment analysis
    signals: Trading signal generation
    backtest: Backtesting engine
    data_loader: Market data loading utilities
    bybit_client: Bybit exchange API client
"""

from .fingpt_sentiment import (
    FinGPTSentimentAnalyzer,
    SentimentResult,
    analyze_sentiment,
)
from .signals import (
    TradingSignal,
    FinGPTTradingEngine,
)
from .backtest import (
    BacktestConfig,
    BacktestResult,
    FinGPTBacktester,
)
from .data_loader import (
    MarketDataLoader,
    OHLCVBar,
)

__version__ = "0.1.0"
__all__ = [
    "FinGPTSentimentAnalyzer",
    "SentimentResult",
    "analyze_sentiment",
    "TradingSignal",
    "FinGPTTradingEngine",
    "BacktestConfig",
    "BacktestResult",
    "FinGPTBacktester",
    "MarketDataLoader",
    "OHLCVBar",
]
