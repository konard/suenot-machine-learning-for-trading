"""
Multi-Task Learning (MTL) for Trading.

This package provides:
- MultiTaskTradingModel: Shared-encoder multi-task neural network
- MTLTrainer: Trainer with uncertainty-based task weighting
- Data loading utilities for stock and crypto markets (Bybit)
- Backtesting framework for MTL strategies
"""

from .mtl_model import (
    SharedEncoder,
    TaskHead,
    MultiTaskTradingModel,
    UncertaintyWeighting,
    MTLTrainer,
)
from .data_loader import (
    create_mtl_features,
    create_mtl_targets,
    fetch_bybit_klines,
    prepare_mtl_dataset,
)
from .backtest import MTLBacktester, calculate_metrics

__all__ = [
    "SharedEncoder",
    "TaskHead",
    "MultiTaskTradingModel",
    "UncertaintyWeighting",
    "MTLTrainer",
    "create_mtl_features",
    "create_mtl_targets",
    "fetch_bybit_klines",
    "prepare_mtl_dataset",
    "MTLBacktester",
    "calculate_metrics",
]
