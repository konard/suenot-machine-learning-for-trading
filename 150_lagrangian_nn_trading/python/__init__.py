"""
Chapter 150: Lagrangian Neural Networks for Trading

Lagrangian Neural Networks (LNNs) learn the Lagrangian function L(q, q-dot)
from market data and derive dynamics via the Euler-Lagrange equations.

Unlike Hamiltonian Neural Networks (Chapter 149), LNNs:
  - Work in (q, q-dot) space instead of (q, p)
  - Do not require a Legendre transform
  - Handle non-separable systems naturally
  - Use second-order derivatives (Euler-Lagrange equations)

Modules:
  - model: LagrangianNN, DissipativeLNN, ForcedLNN, MultiScaleLNN
  - data_loader: Bybit/Yahoo data fetching + config space construction
  - train: Training pipeline
  - backtest: Trading strategy and backtesting
  - visualize: Plotting utilities
"""
