"""
VAE Volatility Surface — Variational Autoencoder for implied volatility surface modeling.

This package provides tools for:
- Representing and manipulating implied volatility surfaces
- Training a VAE to learn surface distributions
- Checking no-arbitrage conditions (butterfly and calendar spread)
- Generating synthetic surfaces via the SABR model
- Fetching live BTC options data from Bybit V5 API
"""

from .model import VAEModel, VAEConfig
from .vol_surface import VolSurface, ArbitrageChecker, compute_skew, compute_butterfly
from .surface_generator import SurfaceGenerator, default_moneyness_grid, default_maturity_days
from .bybit_client import BybitClient, ParsedOption

__all__ = [
    "VAEModel",
    "VAEConfig",
    "VolSurface",
    "ArbitrageChecker",
    "SurfaceGenerator",
    "BybitClient",
    "ParsedOption",
    "compute_skew",
    "compute_butterfly",
    "default_moneyness_grid",
    "default_maturity_days",
]
