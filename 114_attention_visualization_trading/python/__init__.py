"""Attention Visualization for Trading.

This module provides tools for visualizing and interpreting attention weights
in Transformer models applied to financial time series prediction.
"""

from .model import AttentionTransformer, AttentionTrainer
from .visualization import AttentionVisualizer, plot_attention_heatmap
from .data_loader import StockDataLoader, BybitDataLoader, TradingDataset
from .backtest import Backtester, TradingMetrics

__all__ = [
    "AttentionTransformer",
    "AttentionTrainer",
    "AttentionVisualizer",
    "plot_attention_heatmap",
    "StockDataLoader",
    "BybitDataLoader",
    "TradingDataset",
    "Backtester",
    "TradingMetrics",
]
