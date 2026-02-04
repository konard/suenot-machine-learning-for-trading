"""
Normalizing Flows for Finance

This package provides implementations of normalizing flow models
for density estimation and risk management in financial applications.

Modules:
    - flows: Normalizing flow model implementations
    - layers: Coupling layers and transformations
    - risk_metrics: VaR, CVaR, and other risk calculations
    - data_fetcher: Cryptocurrency data fetching via CCXT
    - training: Training utilities and loops
"""

from .flows import NormalizingFlow, RealNVP, MAF
from .layers import AffineCouplingLayer, MaskedLinear
from .risk_metrics import compute_var, compute_cvar, compute_tail_probability
from .data_fetcher import BybitDataFetcher

__version__ = "0.1.0"
__all__ = [
    "NormalizingFlow",
    "RealNVP",
    "MAF",
    "AffineCouplingLayer",
    "MaskedLinear",
    "compute_var",
    "compute_cvar",
    "compute_tail_probability",
    "BybitDataFetcher",
]
