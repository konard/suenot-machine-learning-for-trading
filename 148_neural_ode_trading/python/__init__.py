"""
Chapter 148: Neural ODE Trading

Neural Ordinary Differential Equations applied to financial time series
modeling and trading. Supports irregularly-sampled data, continuous-time
dynamics, and both stock and crypto (Bybit) markets.

Modules:
    neural_ode: Core Neural ODE model with RK4/Dopri5 solvers
    latent_ode: Latent ODE for irregularly-sampled time series
    ode_rnn: ODE-RNN hybrid architecture
    train: Training pipeline with adjoint sensitivity method
    data_loader: Stock + Bybit crypto data loading (irregular timestamps)
    visualize: Trajectory plots, latent space, phase portraits
    backtest: Trading strategy backtesting
"""

from .neural_ode import NeuralODE, ODEFunc, ODENet
from .latent_ode import LatentODE, RecognitionRNN, Decoder
from .ode_rnn import ODERNN, ODERNNCell
from .data_loader import BybitDataLoader, StockDataLoader, IrregularTimeSeriesDataset
from .visualize import (
    plot_trajectories,
    plot_latent_space,
    plot_phase_portrait,
    plot_predictions,
)
from .backtest import NeuralODEBacktester

__all__ = [
    "NeuralODE",
    "ODEFunc",
    "ODENet",
    "LatentODE",
    "RecognitionRNN",
    "Decoder",
    "ODERNN",
    "ODERNNCell",
    "BybitDataLoader",
    "StockDataLoader",
    "IrregularTimeSeriesDataset",
    "plot_trajectories",
    "plot_latent_space",
    "plot_phase_portrait",
    "plot_predictions",
    "NeuralODEBacktester",
]
