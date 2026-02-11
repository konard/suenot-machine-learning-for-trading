"""
Chapter 145: Physics-Informed Neural Networks for Hull-White Interest Rate Model

This package implements a PINN-based solver for the Hull-White (extended Vasicek)
short-rate model. The PINN embeds the bond pricing PDE directly into the neural
network loss function, enabling mesh-free, differentiable bond pricing.

Modules:
    hull_white_pinn  - PINN architecture for Hull-White PDE
    train            - Training loop with term structure fitting
    data_loader      - Treasury yield data + Bybit funding rates
    analytical       - Analytical Hull-White pricing formulas
    calibration      - Calibrate model to yield curve
    derivatives      - Price caps, floors, swaptions
    visualize        - Yield curves, rate distributions, surfaces
    backtest         - Interest rate strategy backtesting
"""

__version__ = "0.1.0"
__author__ = "ML Trading Examples"
