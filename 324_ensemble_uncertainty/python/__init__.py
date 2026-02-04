"""
Ensemble Uncertainty for Trading

This module provides implementations for ensemble-based uncertainty quantification
in trading strategies using data from Bybit exchange via CCXT.

Modules:
- data_fetcher: CCXT-based data fetching from Bybit
- features: Feature engineering for trading
- ensemble: Ensemble model implementations
- uncertainty: Uncertainty quantification methods
- calibration: Model calibration techniques
- strategy: Trading strategy with uncertainty
- backtest: Backtesting framework
"""

from .data_fetcher import BybitDataFetcher
from .features import FeatureEngineer
from .ensemble import EnsembleModel, RandomForestEnsemble, GradientBoostingEnsemble
from .uncertainty import UncertaintyQuantifier
from .calibration import Calibrator
from .strategy import UncertaintyStrategy
from .backtest import Backtester

__all__ = [
    'BybitDataFetcher',
    'FeatureEngineer',
    'EnsembleModel',
    'RandomForestEnsemble',
    'GradientBoostingEnsemble',
    'UncertaintyQuantifier',
    'Calibrator',
    'UncertaintyStrategy',
    'Backtester',
]

__version__ = '0.1.0'
