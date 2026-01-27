"""
Meta-RL Trading - Meta-Reinforcement Learning for Trading

This package provides a complete implementation of Meta-RL (RL^2)
for adaptive trading strategies.

Modules:
- data_loader: Data fetching and feature engineering
- meta_rl_trader: Meta-RL agent, environment, and PPO trainer
- backtest: Backtesting framework
"""

from .data_loader import (
    Kline,
    BybitClient,
    SimulatedDataGenerator,
    FeatureGenerator,
    klines_to_dataframe
)

from .meta_rl_trader import (
    MetaRLAgent,
    TradingEnvironment,
    PPOMetaTrainer,
    RolloutBuffer,
)

from .backtest import (
    Trade,
    BacktestConfig,
    BacktestResults,
    BacktestEngine
)

__version__ = "0.1.0"
__all__ = [
    # Data
    "Kline",
    "BybitClient",
    "SimulatedDataGenerator",
    "FeatureGenerator",
    "klines_to_dataframe",
    # Meta-RL
    "MetaRLAgent",
    "TradingEnvironment",
    "PPOMetaTrainer",
    "RolloutBuffer",
    # Backtest
    "Trade",
    "BacktestConfig",
    "BacktestResults",
    "BacktestEngine",
]
