"""
Bayesian Neural Network module for trading.

This module provides:
- BayesianLinear: Bayesian linear layer with weight uncertainty
- BayesianNN: Full Bayesian Neural Network model
- ELBO loss function
- Monte Carlo inference utilities
"""

from .layers import BayesianLinear
from .model import BayesianNN, BayesianNNConfig
from .loss import elbo_loss, get_kl_weight
from .inference import MCInference, UncertaintyDecomposition

__all__ = [
    'BayesianLinear',
    'BayesianNN',
    'BayesianNNConfig',
    'elbo_loss',
    'get_kl_weight',
    'MCInference',
    'UncertaintyDecomposition',
]
