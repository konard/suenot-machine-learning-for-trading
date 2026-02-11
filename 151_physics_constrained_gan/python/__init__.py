"""
Physics-Constrained GAN for Trading

This package implements a Wasserstein GAN with gradient penalty (WGAN-GP)
that incorporates financial physics constraints including martingale property,
volatility clustering, fat tails, leverage effect, and autocorrelation structure.

Modules:
    model       - Generator, Critic, PhysicsLoss, and PhysicsConstrainedGAN
    data_loader - Data fetching from Bybit API and Yahoo Finance
    train       - WGAN-GP training loop with physics constraints
    visualize   - Visualization of real vs synthetic data comparison
    backtest    - Trading strategy backtest using GAN-generated scenarios
"""

__version__ = "0.1.0"
__author__ = "ML Trading Examples"

from .model import (
    GANConfig,
    Generator,
    Critic,
    PhysicsLoss,
    PhysicsConstrainedGAN,
)

__all__ = [
    "GANConfig",
    "Generator",
    "Critic",
    "PhysicsLoss",
    "PhysicsConstrainedGAN",
]
