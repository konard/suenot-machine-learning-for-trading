"""
Saliency Maps for Trading

This package provides tools for computing and using saliency maps
in algorithmic trading applications.

Modules:
    model: Neural network models for price prediction
    saliency: Saliency map computation methods
    train: Training and data preparation utilities
    backtest: Backtesting framework for saliency-based strategies
"""

from .model import TradingLSTM, TradingCNN, TradingTransformer, create_model
from .saliency import (
    SaliencyComputer,
    aggregate_temporal_importance,
    aggregate_feature_importance,
    compute_saliency_concentration,
    plot_saliency_heatmap,
    plot_temporal_importance,
    plot_feature_importance
)
from .train import (
    TradingDataset,
    fetch_stock_data,
    compute_technical_indicators,
    prepare_sequences,
    create_dataloaders,
    train_model,
    evaluate_model
)
from .backtest import (
    Trade,
    BacktestResult,
    SaliencyTradingStrategy,
    run_backtest,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compare_strategies
)

__all__ = [
    # Models
    'TradingLSTM',
    'TradingCNN',
    'TradingTransformer',
    'create_model',
    # Saliency
    'SaliencyComputer',
    'aggregate_temporal_importance',
    'aggregate_feature_importance',
    'compute_saliency_concentration',
    'plot_saliency_heatmap',
    'plot_temporal_importance',
    'plot_feature_importance',
    # Training
    'TradingDataset',
    'fetch_stock_data',
    'compute_technical_indicators',
    'prepare_sequences',
    'create_dataloaders',
    'train_model',
    'evaluate_model',
    # Backtesting
    'Trade',
    'BacktestResult',
    'SaliencyTradingStrategy',
    'run_backtest',
    'compute_sharpe_ratio',
    'compute_sortino_ratio',
    'compute_max_drawdown',
    'compare_strategies'
]
