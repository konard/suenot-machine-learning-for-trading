"""
Chapter 146: Physics-Informed Neural Networks for the CIR Model
================================================================

This package implements a PINN-based solver for the Cox-Ingersoll-Ross (CIR)
bond pricing PDE, with support for analytical validation, MLE calibration,
credit risk modeling, and application to crypto funding rates (Bybit).

The CIR SDE: dr = kappa * (theta - r) * dt + sigma * sqrt(r) * dW

Modules:
    - cir_pinn: PINN model architecture for the CIR bond pricing PDE
    - train: Training loop with sqrt(r)-aware collocation
    - data_loader: Rate data fetching (treasury + Bybit funding rates)
    - analytical: CIR bond pricing (Bessel functions), transition density
    - calibration: MLE calibration of CIR parameters
    - credit_risk: CIR for default intensity modeling
    - visualize: Plotting rate paths, bond surfaces, yield curves
    - backtest: Funding rate trading strategy backtest

Usage:
    from python.cir_pinn import CIRPINN
    from python.train import train_pinn
    from python.analytical import cir_bond_price, cir_yield
"""

__version__ = "0.1.0"
__author__ = "ML Trading Examples"

from .cir_pinn import CIRPINN
from .analytical import (
    cir_bond_price,
    cir_yield,
    cir_forward_rate,
    cir_transition_density,
)
from .data_loader import (
    generate_synthetic_cir_data,
    fetch_bybit_funding_rates,
    fetch_treasury_rates,
)
