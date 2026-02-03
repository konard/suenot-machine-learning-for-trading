"""
Layer-wise Relevance Propagation (LRP) for Trading

This package provides LRP-enabled neural networks for explainable
trading predictions with support for multiple propagation rules.

Modules:
    model: LRP-enabled neural network implementation
    data: Data loading and preprocessing utilities
    strategy: Trading strategy and backtesting with LRP insights

Example:
    >>> from model import LRPNetwork, LRPConfig
    >>> from data import prepare_trading_data, TradingDataset
    >>> from strategy import backtest_with_lrp
    >>>
    >>> # Load data
    >>> data = prepare_trading_data(['BTCUSDT'], lookback=60)
    >>> dataset = TradingDataset(data['X'], data['y'])
    >>>
    >>> # Create model
    >>> config = LRPConfig(epsilon=0.01, gamma=0.25)
    >>> model = LRPNetwork(input_dim=420, hidden_dims=[128, 64], output_dim=2, config=config)
    >>>
    >>> # Get explanations
    >>> relevance = model.explain(sample)
"""

from .model import (
    LRPRule,
    LRPConfig,
    LRPLinear,
    LRPNetwork,
)

from .data import (
    prepare_trading_data,
    TradingDataset,
    compute_rsi,
    compute_macd,
    compute_bollinger_position,
    compute_atr,
)

from .strategy import (
    BacktestConfig,
    backtest_with_lrp,
    analyze_trading_signal,
    compute_feature_importance,
)

__version__ = "0.1.0"
__author__ = "ML Trading Team"

__all__ = [
    # Model
    "LRPRule",
    "LRPConfig",
    "LRPLinear",
    "LRPNetwork",
    # Data
    "prepare_trading_data",
    "TradingDataset",
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_position",
    "compute_atr",
    # Strategy
    "BacktestConfig",
    "backtest_with_lrp",
    "analyze_trading_signal",
    "compute_feature_importance",
]
