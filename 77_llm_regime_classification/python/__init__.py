"""
Chapter 77: LLM Regime Classification

This module provides tools for classifying market regimes using
Large Language Models (LLMs) and traditional statistical methods.

Components:
- data_loader: Load financial data from Yahoo Finance and Bybit
- classifier: Regime classification algorithms
- embeddings: Generate LLM embeddings for text and numerical data
- signals: Generate trading signals from regime detection
- backtest: Backtesting framework for regime-based strategies
- evaluate: Evaluation metrics and visualization
"""

from .data_loader import YahooFinanceLoader, BybitDataLoader
from .classifier import (
    MarketRegime,
    RegimeResult,
    HMMRegimeDetector,
    StatisticalRegimeClassifier,
    TextRegimeClassifier,
    HybridRegimeClassifier,
)
from .signals import RegimeSignalGenerator, TradingSignal, SignalType
from .backtest import RegimeBacktester, BacktestResult
from .evaluate import RegimeEvaluator, generate_report

__version__ = "0.1.0"
__all__ = [
    "MarketRegime",
    "RegimeResult",
    "YahooFinanceLoader",
    "BybitDataLoader",
    "HMMRegimeDetector",
    "StatisticalRegimeClassifier",
    "TextRegimeClassifier",
    "HybridRegimeClassifier",
    "RegimeSignalGenerator",
    "TradingSignal",
    "SignalType",
    "RegimeBacktester",
    "BacktestResult",
    "RegimeEvaluator",
    "generate_report",
]
