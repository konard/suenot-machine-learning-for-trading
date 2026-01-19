"""
Grouped Query Attention (GQA) for Trading

This module implements Grouped Query Attention mechanism optimized for
financial time series prediction, including cryptocurrency and stock trading.

Key Components:
- GroupedQueryAttention: Core GQA mechanism
- GQATrader: Complete trading model using GQA
- Data loading utilities for Bybit and Yahoo Finance
- Backtesting framework

Example Usage:
    from gqa_trading import GQATrader, load_bybit_data

    # Load data
    data = load_bybit_data("BTCUSDT", "1h", limit=1000)

    # Create model
    model = GQATrader(
        input_dim=5,
        d_model=64,
        num_heads=8,
        num_kv_heads=2,
        num_layers=4
    )

    # Train and predict
    predictions = model(data)
"""

from .model import GroupedQueryAttention, GQATrader, GQABlock
from .data import load_bybit_data, load_yahoo_data, prepare_sequences
from .train import train_model, evaluate_model
from .predict import predict_next, predict_batch
from .strategy import backtest_strategy, calculate_metrics

__version__ = "1.0.0"
__author__ = "Machine Learning for Trading"

__all__ = [
    # Model components
    "GroupedQueryAttention",
    "GQATrader",
    "GQABlock",
    # Data utilities
    "load_bybit_data",
    "load_yahoo_data",
    "prepare_sequences",
    # Training
    "train_model",
    "evaluate_model",
    # Prediction
    "predict_next",
    "predict_batch",
    # Strategy
    "backtest_strategy",
    "calculate_metrics",
]
