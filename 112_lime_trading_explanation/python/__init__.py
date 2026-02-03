"""
LIME for Trading Explanation

This module provides implementations of LIME (Local Interpretable Model-agnostic
Explanations) for explaining trading model predictions.

Modules:
    - data_loader: Data fetching from Yahoo Finance and Bybit
    - model: Example trading models (Random Forest, XGBoost)
    - lime_explainer: Core LIME implementation for trading
    - backtest: Backtesting framework with explainability
"""

from .data_loader import DataLoader
from .model import TradingModel
from .lime_explainer import LimeExplainer
from .backtest import Backtester

__all__ = ['DataLoader', 'TradingModel', 'LimeExplainer', 'Backtester']
