"""
TimeParticle SSM - Particle-like Multiscale State Space Models for Trading

This module provides implementations of the TimeParticle architecture
for financial time series forecasting and trading signal generation.
"""

from .timeparticle_model import (
    Particle,
    CrossScaleAttention,
    TimeParticleSSM,
    TimeParticleTradingModel,
)
from .data_loader import BybitDataLoader, fetch_stock_data
from .features import prepare_multiscale_features, compute_technical_indicators
from .backtest import TimeParticleBacktest

__version__ = "0.1.0"
__all__ = [
    "Particle",
    "CrossScaleAttention",
    "TimeParticleSSM",
    "TimeParticleTradingModel",
    "BybitDataLoader",
    "fetch_stock_data",
    "prepare_multiscale_features",
    "compute_technical_indicators",
    "TimeParticleBacktest",
]
