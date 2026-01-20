"""
Chapter 76: LLM Anomaly Detection

This module provides tools for detecting anomalies in financial data using
Large Language Models (LLMs) and traditional statistical methods.

Components:
- data_loader: Load financial data from Yahoo Finance and Bybit
- embeddings: Generate LLM embeddings for text and numerical data
- detector: Anomaly detection algorithms
- signals: Generate trading signals from anomalies
- backtest: Backtesting framework for anomaly-based strategies
- evaluate: Evaluation metrics and visualization
"""

from .data_loader import YahooFinanceLoader, BybitDataLoader
from .detector import (
    StatisticalAnomalyDetector,
    ZeroShotAnomalyDetector,
    EmbeddingAnomalyDetector,
    TimeSeriesAnomalyDetector,
)
from .signals import AnomalySignalGenerator
from .backtest import AnomalyBacktester

__version__ = "0.1.0"
__all__ = [
    "YahooFinanceLoader",
    "BybitDataLoader",
    "StatisticalAnomalyDetector",
    "ZeroShotAnomalyDetector",
    "EmbeddingAnomalyDetector",
    "TimeSeriesAnomalyDetector",
    "AnomalySignalGenerator",
    "AnomalyBacktester",
]
