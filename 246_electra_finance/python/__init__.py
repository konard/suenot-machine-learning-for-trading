"""
ELECTRA for Finance

This package provides tools for using ELECTRA-based models
in financial sentiment analysis and algorithmic trading applications.

Modules:
    model: ELECTRA-based models for financial text classification
    train: Training and data preparation utilities
    backtest: Backtesting framework for sentiment-based strategies
"""

from .model import (
    FinancialElectra,
    ElectraGenerator,
    ElectraDiscriminator,
    create_model
)
from .train import (
    FinancialTextDataset,
    fetch_stock_data,
    generate_synthetic_news,
    prepare_text_data,
    create_dataloaders,
    train_model,
    evaluate_model
)
from .backtest import (
    Trade,
    BacktestResult,
    ElectraTradingStrategy,
    run_backtest,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compare_strategies
)

__all__ = [
    # Models
    'FinancialElectra',
    'ElectraGenerator',
    'ElectraDiscriminator',
    'create_model',
    # Training
    'FinancialTextDataset',
    'fetch_stock_data',
    'generate_synthetic_news',
    'prepare_text_data',
    'create_dataloaders',
    'train_model',
    'evaluate_model',
    # Backtesting
    'Trade',
    'BacktestResult',
    'ElectraTradingStrategy',
    'run_backtest',
    'compute_sharpe_ratio',
    'compute_sortino_ratio',
    'compute_max_drawdown',
    'compare_strategies'
]
