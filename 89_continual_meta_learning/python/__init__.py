"""
Continual Meta-Learning Trading - Continual MAML for Adaptive Trading

This package provides a complete implementation of continual meta-learning
for adaptive trading strategies that can learn new market regimes without
forgetting previously learned ones.

Modules:
- data_loader: Data fetching and feature engineering
- continual_meta_learner: Continual MAML algorithm with EWC and replay
- backtest: Backtesting framework
"""

from .data_loader import (
    Kline,
    BybitClient,
    SimulatedDataGenerator,
    FeatureGenerator,
    klines_to_dataframe
)

from .continual_meta_learner import (
    TradingModel,
    TaskData,
    ReplayBuffer,
    EWC,
    ContinualMAMLTrainer,
    TradingSignal,
    TradingStrategy,
    create_tasks_from_data
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
    # Continual Meta-Learning
    "TradingModel",
    "TaskData",
    "ReplayBuffer",
    "EWC",
    "ContinualMAMLTrainer",
    "TradingSignal",
    "TradingStrategy",
    "create_tasks_from_data",
    # Backtest
    "Trade",
    "BacktestConfig",
    "BacktestResults",
    "BacktestEngine",
]
