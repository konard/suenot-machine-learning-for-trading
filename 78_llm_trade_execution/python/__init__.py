"""
LLM Trade Execution - Python Implementation

This package provides tools for optimizing trade execution using Large Language Models (LLMs)
to minimize market impact in both stock and cryptocurrency markets.

Modules:
    data: Market data structures and exchange connectivity (Bybit)
    execution: Execution engine, strategies, and LLM integration
    impact: Market impact models (Almgren-Chriss)
    strategy: Execution strategies (TWAP, VWAP, IS, Adaptive)
"""

from .execution import (
    ExecutionEngine,
    ExecutionConfig,
    ParentOrder,
    ChildOrder,
    Side,
    ExecutionResult,
)
from .data import (
    BybitClient,
    BybitConfig,
    OrderBook,
    Ticker,
    OhlcvBar,
)
from .impact import (
    MarketImpactEstimator,
    AlmgrenChrissModel,
    AlmgrenChrissParams,
)
from .strategy import (
    TwapStrategy,
    VwapStrategy,
    AdaptiveStrategy,
    ImplementationShortfallStrategy,
)
from .llm import (
    LlmAdapter,
    LlmConfig,
    LlmDecision,
)

__version__ = "0.1.0"
__all__ = [
    # Execution
    "ExecutionEngine",
    "ExecutionConfig",
    "ParentOrder",
    "ChildOrder",
    "Side",
    "ExecutionResult",
    # Data
    "BybitClient",
    "BybitConfig",
    "OrderBook",
    "Ticker",
    "OhlcvBar",
    # Impact
    "MarketImpactEstimator",
    "AlmgrenChrissModel",
    "AlmgrenChrissParams",
    # Strategy
    "TwapStrategy",
    "VwapStrategy",
    "AdaptiveStrategy",
    "ImplementationShortfallStrategy",
    # LLM
    "LlmAdapter",
    "LlmConfig",
    "LlmDecision",
]
