"""Disentangled VAE Trading - Chapter 238

This module implements disentangled Variational Autoencoder (VAE) models
for financial market analysis and trading signal generation. The core idea
is to learn independent latent factors that capture distinct market dynamics
such as trend, volatility, momentum, and regime, then use these factors
to generate trading signals.

Supported disentanglement methods:
  - Beta-VAE: increased KL penalty for disentanglement
  - Factor-VAE: adversarial Total Correlation minimization
  - DIP-VAE: Disentangled Inferred Prior matching
  - Beta-TCVAE: Total Correlation decomposition of the KL term
"""

from .model import DisentangledVAEConfig, BetaVAE, FactorVAE, DisentangledVAEForecaster
from .data import load_bybit_data, prepare_features, DisentangledVAEDataset
from .strategy import DisentangledVAEStrategy, backtest_disentangled_strategy
