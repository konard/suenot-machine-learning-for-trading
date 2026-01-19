"""
RAG for Trading - Python Implementation

This module provides a complete Retrieval-Augmented Generation (RAG) system
for trading applications. It includes document retrieval, signal generation,
and backtesting capabilities.

Components:
    - retriever: Document retrieval with semantic search
    - signals: Trading signal generation from RAG analysis
    - backtest: Backtesting framework for RAG-based strategies
    - data_loader: Market data loading from Yahoo Finance and Bybit
"""

from .retriever import (
    Document,
    SearchResult,
    FinancialDocumentRetriever,
)
from .signals import (
    SignalDirection,
    TradingSignal,
    RAGTradingSignalGenerator,
)
from .backtest import (
    BacktestResult,
    RAGBacktester,
)
from .data_loader import (
    MarketData,
    YahooFinanceLoader,
    BybitLoader,
    DataLoader,
    combine_prices,
    calculate_features,
)

__all__ = [
    # Retriever
    "Document",
    "SearchResult",
    "FinancialDocumentRetriever",
    # Signals
    "SignalDirection",
    "TradingSignal",
    "RAGTradingSignalGenerator",
    # Backtest
    "BacktestResult",
    "RAGBacktester",
    # Data Loader
    "MarketData",
    "YahooFinanceLoader",
    "BybitLoader",
    "DataLoader",
    "combine_prices",
    "calculate_features",
]

__version__ = "0.1.0"
