"""
LLM Alpha Mining - Python Implementation

A toolkit for using Large Language Models to generate, evaluate, and
trade alpha factors in financial markets.

Key Components:
    - alpha_generator: Generate alpha factors using LLMs
    - data_loader: Load market data from Yahoo Finance and Bybit
    - factor_evaluator: Evaluate alpha factor performance (IC, Sharpe, etc.)
    - quantagent: Self-improving alpha mining agent
    - backtest: Backtesting framework for LLM-generated strategies

Example:
    >>> from llm_alpha_mining import AlphaGenerator, DataLoader
    >>> loader = DataLoader()
    >>> data = loader.load("BTCUSDT", source="bybit")
    >>> generator = AlphaGenerator()
    >>> factors = generator.generate(data.ohlcv)
"""

from .data_loader import (
    DataLoader,
    YahooFinanceLoader,
    BybitLoader,
    MarketData,
    combine_prices,
    calculate_features,
)
from .alpha_generator import AlphaGenerator, AlphaFactor
from .factor_evaluator import FactorEvaluator, FactorMetrics
from .quantagent import QuantAgent, KnowledgeBase
from .backtest import Backtester, BacktestResult

__version__ = "0.1.0"
__all__ = [
    "DataLoader",
    "YahooFinanceLoader",
    "BybitLoader",
    "MarketData",
    "combine_prices",
    "calculate_features",
    "AlphaGenerator",
    "AlphaFactor",
    "FactorEvaluator",
    "FactorMetrics",
    "QuantAgent",
    "KnowledgeBase",
    "Backtester",
    "BacktestResult",
]
