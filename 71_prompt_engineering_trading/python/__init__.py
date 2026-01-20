"""
Prompt Engineering for Trading - Python Implementation

This module provides tools for using prompt engineering techniques
to generate trading signals from LLM analysis.
"""

from .sentiment_analysis import FinancialSentimentAnalyzer, Sentiment, SentimentResult
from .signal_generator import PromptBasedSignalGenerator, TradingSignal, SignalDirection
from .regime_detection import MarketRegimeDetector, MarketRegime, RegimeAnalysis
from .backtest import LLMSignalBacktester, BacktestConfig, BacktestResult

__all__ = [
    "FinancialSentimentAnalyzer",
    "Sentiment",
    "SentimentResult",
    "PromptBasedSignalGenerator",
    "TradingSignal",
    "SignalDirection",
    "MarketRegimeDetector",
    "MarketRegime",
    "RegimeAnalysis",
    "LLMSignalBacktester",
    "BacktestConfig",
    "BacktestResult",
]

__version__ = "0.1.0"
