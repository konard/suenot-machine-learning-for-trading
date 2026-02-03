"""
Counterfactual Explanations for Trading

This module provides tools for generating counterfactual explanations
for trading models, helping understand what minimal changes would flip
a model's prediction.
"""

from .model import TradingClassifier
from .counterfactual import CounterfactualGenerator, CounterfactualOptimizer
from .data import BybitClient, prepare_features
from .backtest import Backtester, BacktestResult

__all__ = [
    'TradingClassifier',
    'CounterfactualGenerator',
    'CounterfactualOptimizer',
    'BybitClient',
    'prepare_features',
    'Backtester',
    'BacktestResult',
]

__version__ = '0.1.0'
