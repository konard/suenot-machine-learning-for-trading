"""
Mediation Analysis Module for Financial Trading

This module provides comprehensive tools for mediation analysis
in financial applications, including:
- Classical Baron-Kenny mediation
- Causal mediation analysis with sensitivity
- Bootstrap confidence intervals
- Trading signal generation
"""

from .model import (
    MediationAnalyzer,
    CausalMediationAnalysis,
    baron_kenny_mediation,
    bootstrap_mediation,
    MediationResults
)

from .data import (
    prepare_stock_data,
    prepare_crypto_data_bybit
)

from .strategy import (
    MediationTradingStrategy,
    MediationSignal,
    MediationBacktester
)

__version__ = "0.1.0"
__all__ = [
    "MediationAnalyzer",
    "CausalMediationAnalysis",
    "baron_kenny_mediation",
    "bootstrap_mediation",
    "MediationResults",
    "prepare_stock_data",
    "prepare_crypto_data_bybit",
    "MediationTradingStrategy",
    "MediationSignal",
    "MediationBacktester"
]
