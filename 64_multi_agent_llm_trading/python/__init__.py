"""
Multi-Agent LLM Trading System

A framework for building multi-agent trading systems using Large Language Models.
"""

from .agents import (
    BaseAgent,
    FundamentalsAgent,
    TechnicalAgent,
    SentimentAgent,
    NewsAgent,
    BullAgent,
    BearAgent,
    RiskManagerAgent,
    TraderAgent,
)
from .communication import (
    Message,
    AgentMessage,
    Debate,
    DebateModerator,
    RoundTable,
)
from .data_loader import (
    MarketData,
    DataLoader,
    YahooFinanceLoader,
    BybitLoader,
)
from .backtest import (
    BacktestResult,
    MultiAgentBacktester,
)

__version__ = "0.1.0"

__all__ = [
    # Agents
    "BaseAgent",
    "FundamentalsAgent",
    "TechnicalAgent",
    "SentimentAgent",
    "NewsAgent",
    "BullAgent",
    "BearAgent",
    "RiskManagerAgent",
    "TraderAgent",
    # Communication
    "Message",
    "AgentMessage",
    "Debate",
    "DebateModerator",
    "RoundTable",
    # Data
    "MarketData",
    "DataLoader",
    "YahooFinanceLoader",
    "BybitLoader",
    # Backtest
    "BacktestResult",
    "MultiAgentBacktester",
]
