"""
MAML Trading - Model-Agnostic Meta-Learning for Trading

This package provides a complete implementation of MAML for
adaptive trading strategies.

Modules:
- data_loader: Data fetching and feature engineering
- maml_trader: MAML algorithm and trading model
- backtest: Backtesting framework
"""

from .data_loader import (
    Kline,
    BybitClient,
    SimulatedDataGenerator,
    FeatureGenerator,
    klines_to_dataframe
)

from .maml_trader import (
    TradingModel,
    TaskData,
    MAMLTrainer,
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
    # MAML
    "TradingModel",
    "TaskData",
    "MAMLTrainer",
    "TradingSignal",
    "TradingStrategy",
    "create_tasks_from_data",
    # Backtest
    "Trade",
    "BacktestConfig",
    "BacktestResults",
    "BacktestEngine",
]
