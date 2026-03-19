"""Trading strategy built on disentangled VAE latent factors.

The core idea is that a well-disentangled VAE learns latent dimensions
that correspond to independent market factors (trend, volatility, momentum,
etc.).  By monitoring how these factors evolve, we can generate trading
signals that are more interpretable and robust than those derived from
raw price data.

Classes:
    DisentangledVAEStrategy -- live / walk-forward signal generator

Functions:
    backtest_disentangled_strategy -- vectorised backtest on historical data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .model import BetaVAE, DisentangledVAEConfig, DisentangledVAEForecaster


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class DisentangledVAEStrategy:
    """Generate trading signals from a trained disentangled VAE.

    The strategy encodes each new market window into latent space, then
    combines per-dimension signals into an aggregate position.

    Attributes:
        model: Trained disentangled VAE (or forecaster wrapper).
        config: Model configuration.
        signal_threshold: Minimum absolute signal to trigger a trade.
        factor_weights: Per-dimension importance weights (learned or fixed).
    """

    def __init__(
        self,
        model: BetaVAE | DisentangledVAEForecaster,
        config: DisentangledVAEConfig,
        signal_threshold: float = 0.3,
        factor_weights: Optional[np.ndarray] = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.signal_threshold = signal_threshold

        # Unwrap forecaster to get the raw VAE if needed
        if isinstance(model, DisentangledVAEForecaster):
            self._forecaster = model.to(self.device)
            self._vae = model.vae
        else:
            self._forecaster = None
            self._vae = model.to(self.device)

        self._vae.eval()

        if factor_weights is not None:
            self.factor_weights = factor_weights
        else:
            self.factor_weights = np.ones(config.latent_dim) / config.latent_dim

        # Running statistics for z-scoring latent activations
        self._z_history: List[np.ndarray] = []

    # ---- encoding ----

    @torch.no_grad()
    def encode_market_state(
        self, window: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a market data window into latent space.

        Args:
            window: Array of shape ``(n_features, seq_len)`` or
                ``(1, n_features, seq_len)``.

        Returns:
            Tuple of ``(mu, logvar)`` arrays each of shape ``(latent_dim,)``.
        """
        if window.ndim == 2:
            window = window[np.newaxis, ...]  # add batch dim
        x = torch.tensor(window, dtype=torch.float32, device=self.device)
        mu, logvar = self._vae.encode(x)
        mu_np = mu.cpu().numpy().squeeze()
        logvar_np = logvar.cpu().numpy().squeeze()
        self._z_history.append(mu_np)
        return mu_np, logvar_np

    # ---- signal generation ----

    def generate_signals(
        self, window: np.ndarray
    ) -> Dict[str, float | np.ndarray]:
        """Produce trading signals from a single market window.

        Returns a dictionary containing:
            - ``position``: target position in [-1, 1]
            - ``confidence``: signal confidence in [0, 1]
            - ``factor_signals``: per-dimension raw signals
            - ``mu``: latent mean vector
        """
        mu, logvar = self.encode_market_state(window)

        # Per-dimension signal: use z-scored mu if we have history
        if len(self._z_history) > 20:
            history = np.array(self._z_history[-100:])
            z_mean = history.mean(axis=0)
            z_std = history.std(axis=0)
            z_std[z_std < 1e-8] = 1.0
            factor_signals = (mu - z_mean) / z_std
        else:
            factor_signals = mu

        # Weighted aggregate signal
        raw_signal = float(np.dot(self.factor_weights, factor_signals))

        # Confidence based on inverse uncertainty
        uncertainty = float(np.mean(np.exp(0.5 * logvar)))
        confidence = float(np.clip(1.0 / (1.0 + uncertainty), 0.0, 1.0))

        # Threshold and clip
        if abs(raw_signal) < self.signal_threshold:
            position = 0.0
        else:
            position = float(np.clip(np.tanh(raw_signal), -1.0, 1.0))

        return {
            "position": position,
            "confidence": confidence,
            "factor_signals": factor_signals,
            "mu": mu,
        }

    # ---- factor exposures ----

    def get_factor_exposures(self) -> pd.DataFrame:
        """Return a DataFrame summarising latent factor statistics.

        Requires at least one call to ``encode_market_state`` or
        ``generate_signals`` to have populated the history buffer.
        """
        if not self._z_history:
            return pd.DataFrame()

        history = np.array(self._z_history)
        records = []
        for dim in range(history.shape[1]):
            vals = history[:, dim]
            records.append({
                "factor": f"z_{dim}",
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "last": float(vals[-1]),
                "weight": float(self.factor_weights[dim]),
            })
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def backtest_disentangled_strategy(
    model: BetaVAE | DisentangledVAEForecaster,
    config: DisentangledVAEConfig,
    test_data: np.ndarray,
    returns: np.ndarray,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
    signal_threshold: float = 0.3,
) -> Dict[str, float | list]:
    """Run a vectorised backtest of the disentangled VAE strategy.

    Args:
        model: Trained VAE or forecaster model.
        config: Model configuration (must match the model).
        test_data: Normalised feature array of shape ``(T, n_features)``.
        returns: Simple returns array of shape ``(T,)`` aligned with
            ``test_data``.
        initial_capital: Starting capital in currency units.
        transaction_cost: Proportional cost per trade (e.g. 0.001 = 10 bps).
        signal_threshold: Minimum absolute signal to open a position.

    Returns:
        Dictionary with:
            - ``total_return``: cumulative return as a fraction
            - ``sharpe_ratio``: annualised Sharpe ratio (252 trading days)
            - ``sortino_ratio``: annualised Sortino ratio
            - ``max_drawdown``: maximum peak-to-trough drawdown
            - ``win_rate``: fraction of profitable trades
            - ``profit_factor``: gross profits / gross losses
            - ``capital_history``: list of capital at each step
            - ``positions``: list of positions at each step
            - ``n_trades``: total number of position changes
    """
    strategy = DisentangledVAEStrategy(
        model=model,
        config=config,
        signal_threshold=signal_threshold,
    )

    seq_len = config.seq_len
    n_steps = len(test_data) - seq_len
    if n_steps <= 0:
        raise ValueError("test_data must be longer than seq_len")

    capital = initial_capital
    capital_history: List[float] = [capital]
    positions: List[float] = []
    trade_returns: List[float] = []
    prev_position = 0.0
    n_trades = 0

    for t in range(n_steps):
        # Build window: (n_features, seq_len)
        window = test_data[t : t + seq_len].T

        signals = strategy.generate_signals(window)
        position = signals["position"]

        # Track trades
        if position != prev_position:
            n_trades += 1
        positions.append(position)

        # PnL for this step
        step_idx = t + seq_len
        if step_idx < len(returns):
            step_return = position * returns[step_idx]

            # Transaction cost on position change
            cost = transaction_cost * abs(position - prev_position) * capital
            capital = capital * (1.0 + step_return) - cost

            trade_returns.append(step_return)
        else:
            trade_returns.append(0.0)

        capital_history.append(capital)
        prev_position = position

    # --- Metrics ---
    trade_returns_arr = np.array(trade_returns)
    capital_arr = np.array(capital_history)

    total_return = (capital_arr[-1] / capital_arr[0]) - 1.0

    # Sharpe ratio (annualised)
    if len(trade_returns_arr) > 1 and trade_returns_arr.std() > 1e-10:
        sharpe_ratio = float(
            (trade_returns_arr.mean() / trade_returns_arr.std()) * np.sqrt(252)
        )
    else:
        sharpe_ratio = 0.0

    # Sortino ratio
    downside = trade_returns_arr[trade_returns_arr < 0]
    if len(downside) > 0 and downside.std() > 1e-10:
        sortino_ratio = float(
            (trade_returns_arr.mean() / downside.std()) * np.sqrt(252)
        )
    else:
        sortino_ratio = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(capital_arr)
    drawdowns = (capital_arr - peak) / peak
    max_drawdown = float(drawdowns.min())

    # Win rate
    winning = trade_returns_arr[trade_returns_arr > 0]
    losing = trade_returns_arr[trade_returns_arr < 0]
    nonzero_trades = trade_returns_arr[trade_returns_arr != 0]
    win_rate = float(len(winning) / max(len(nonzero_trades), 1))

    # Profit factor
    gross_profit = float(winning.sum()) if len(winning) > 0 else 0.0
    gross_loss = float(abs(losing.sum())) if len(losing) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    return {
        "total_return": float(total_return),
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": float(max_drawdown),
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "capital_history": capital_history,
        "positions": positions,
        "n_trades": n_trades,
    }
