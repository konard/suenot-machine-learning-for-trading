"""
Chain-of-Thought Trading Module

This module provides tools for explainable trading decisions using
Chain-of-Thought (CoT) prompting techniques with Large Language Models.

Main components:
- cot_analyzer: Chain-of-Thought analysis for trading
- signal_generator: Multi-step signal generation
- position_sizer: Risk-aware position sizing with reasoning
- backtest: Backtesting engine with full audit trails
- data_loader: Data loading from Yahoo Finance and Bybit
"""

from .cot_analyzer import (
    ChainOfThoughtAnalyzer,
    MockChainOfThoughtAnalyzer,
    CoTAnalysis,
    ReasoningStep,
)
from .signal_generator import (
    MultiStepSignalGenerator,
    Signal,
    CoTSignal,
)
from .position_sizer import (
    CoTPositionSizer,
    PositionSizeResult,
)
from .backtest import (
    CoTBacktester,
    BacktestResult,
    Trade,
    BacktestConfig,
)
from .data_loader import (
    DataLoader,
    YahooFinanceLoader,
    BybitLoader,
    MockDataLoader,
    OHLCV,
    create_loader,
    add_technical_indicators,
    prepare_for_analysis,
)

__version__ = "0.1.0"
__all__ = [
    # Analyzer
    "ChainOfThoughtAnalyzer",
    "MockChainOfThoughtAnalyzer",
    "CoTAnalysis",
    "ReasoningStep",
    # Signals
    "MultiStepSignalGenerator",
    "Signal",
    "CoTSignal",
    # Position sizing
    "CoTPositionSizer",
    "PositionSizeResult",
    # Backtesting
    "CoTBacktester",
    "BacktestResult",
    "Trade",
    "BacktestConfig",
    # Data
    "DataLoader",
    "YahooFinanceLoader",
    "BybitLoader",
    "MockDataLoader",
    "OHLCV",
    "create_loader",
    "add_technical_indicators",
    "prepare_for_analysis",
]
