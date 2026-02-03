"""
Backtesting framework for HiPPO-based trading strategies.

Provides:
- BacktestEngine: Run strategy simulations on historical data
- PerformanceMetrics: Compute Sharpe, Sortino, max drawdown, etc.
- HiPPOStrategy: Example strategy using HiPPO coefficient dynamics
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Trading strategy performance metrics."""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0

    def __str__(self) -> str:
        return (
            f"Performance Metrics:\n"
            f"  Total Return:   {self.total_return:>8.2%}\n"
            f"  Annual Return:  {self.annual_return:>8.2%}\n"
            f"  Sharpe Ratio:   {self.sharpe_ratio:>8.2f}\n"
            f"  Sortino Ratio:  {self.sortino_ratio:>8.2f}\n"
            f"  Max Drawdown:   {self.max_drawdown:>8.2%}\n"
            f"  Calmar Ratio:   {self.calmar_ratio:>8.2f}\n"
            f"  Win Rate:       {self.win_rate:>8.2%}\n"
            f"  Profit Factor:  {self.profit_factor:>8.2f}\n"
            f"  Num Trades:     {self.num_trades:>8d}"
        )


def compute_metrics(
    returns: np.ndarray,
    positions: np.ndarray,
    annual_factor: float = 252.0,
) -> PerformanceMetrics:
    """
    Compute performance metrics from strategy returns and positions.

    Args:
        returns: Array of asset returns per period.
        positions: Array of positions (-1, 0, or 1) per period.
        annual_factor: Annualization factor (252 for daily, 365*24 for hourly).

    Returns:
        PerformanceMetrics dataclass.
    """
    strategy_returns = returns * positions

    # Total and annual return
    cumulative = (1 + strategy_returns).cumprod()
    total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0.0
    n_periods = len(strategy_returns)
    annual_return = (1 + total_return) ** (annual_factor / max(n_periods, 1)) - 1

    # Sharpe ratio
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    sharpe = (mean_ret / std_ret * np.sqrt(annual_factor)) if std_ret > 1e-10 else 0.0

    # Sortino ratio (only downside deviation)
    downside = strategy_returns[strategy_returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-10
    sortino = (mean_ret / downside_std * np.sqrt(annual_factor)) if downside_std > 1e-10 else 0.0

    # Maximum drawdown
    cumulative_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - cumulative_max) / cumulative_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0.0

    # Win rate and profit factor
    trades = strategy_returns[strategy_returns != 0]
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0.0
    gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    # Count position changes as trades
    position_changes = np.diff(positions)
    num_trades = int(np.sum(position_changes != 0))

    return PerformanceMetrics(
        total_return=total_return,
        annual_return=annual_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=num_trades,
    )


class HiPPOStrategy:
    """
    Trading strategy based on HiPPO polynomial coefficient dynamics.

    Generates signals from the trend coefficient (c1) with optional
    acceleration (c2) and volatility filters.

    Args:
        n_hippo: Number of HiPPO polynomial coefficients.
        trend_threshold: Minimum c1 magnitude for trend signal.
        vol_filter: Whether to filter signals by volatility.
    """

    def __init__(self, n_hippo: int = 64, trend_threshold: float = 0.0,
                 vol_filter: bool = True):
        self.n_hippo = n_hippo
        self.trend_threshold = trend_threshold
        self.vol_filter = vol_filter

    def _build_legs_matrices(self, N: int):
        """Build HiPPO-LegS A and B matrices (numpy version)."""
        A = np.zeros((N, N))
        B = np.zeros(N)
        for n in range(N):
            B[n] = (2 * n + 1) ** 0.5
            for k in range(n + 1):
                if n > k:
                    A[n, k] = -(2 * n + 1) ** 0.5 * (2 * k + 1) ** 0.5
                elif n == k:
                    A[n, k] = -(n + 1)
        return A, B

    def generate_signals(self, prices: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Generate trading signals from price series.

        Args:
            prices: 1D array of prices.
            dt: Time step.

        Returns:
            1D array of positions (-1, 0, or 1).
        """
        N = self.n_hippo
        A, B = self._build_legs_matrices(N)

        # Forward Euler discretization with reduced dt for stability
        # For large N, the eigenvalues of A grow as O(N), requiring dt < 1/N
        effective_dt = min(dt, 1.0 / (N + 1))
        A_d = np.eye(N) + effective_dt * A
        B_d = effective_dt * B

        # Normalize prices
        p_mean = np.mean(prices)
        p_std = np.std(prices) if np.std(prices) > 1e-10 else 1.0
        p_norm = (prices - p_mean) / p_std

        # Run HiPPO
        state = np.zeros(N)
        c1_history = []
        c2_history = []
        high_coeff_var = []

        for t in range(len(p_norm)):
            state = A_d @ state + B_d * p_norm[t]
            c1_history.append(state[1] if N > 1 else 0.0)
            c2_history.append(state[2] if N > 2 else 0.0)
            if N > 4:
                high_coeff_var.append(np.var(state[N // 2:]))
            else:
                high_coeff_var.append(0.0)

        c1 = np.array(c1_history)
        c2 = np.array(c2_history)
        hcv = np.array(high_coeff_var)

        # Signal: based on trend coefficient
        positions = np.zeros(len(prices))
        for t in range(1, len(prices)):
            if c1[t] > self.trend_threshold:
                positions[t] = 1.0  # Long
            elif c1[t] < -self.trend_threshold:
                positions[t] = -1.0  # Short
            else:
                positions[t] = 0.0  # Flat

        # Volatility filter: reduce exposure in high-vol regimes
        if self.vol_filter and len(hcv) > 0:
            vol_threshold = np.median(hcv) * 2
            for t in range(len(positions)):
                if hcv[t] > vol_threshold:
                    positions[t] *= 0.5

        return positions


class BacktestEngine:
    """
    Backtesting engine for HiPPO strategies.

    Runs a strategy simulation and computes performance metrics.

    Args:
        annual_factor: Annualization factor (252 for daily, 8760 for hourly).
    """

    def __init__(self, annual_factor: float = 8760.0):
        self.annual_factor = annual_factor

    def run(
        self,
        prices: np.ndarray,
        strategy: Optional[HiPPOStrategy] = None,
        positions: Optional[np.ndarray] = None,
    ) -> PerformanceMetrics:
        """
        Run backtest.

        Either provide a strategy (which generates positions from prices)
        or provide positions directly.

        Args:
            prices: 1D array of prices.
            strategy: HiPPOStrategy instance (optional).
            positions: Pre-computed positions (optional).

        Returns:
            PerformanceMetrics for the strategy.
        """
        if positions is None and strategy is None:
            strategy = HiPPOStrategy()

        if positions is None:
            positions = strategy.generate_signals(prices)

        # Compute returns
        returns = np.diff(prices) / prices[:-1]
        # Align positions (use previous position to avoid lookahead)
        pos = positions[:-1]

        metrics = compute_metrics(returns, pos, self.annual_factor)
        return metrics


def demo():
    """Run a backtesting demo with synthetic data."""
    from data_loader import generate_synthetic_data

    logger.info("=== HiPPO Backtesting Demo ===")

    # Generate synthetic data
    data = generate_synthetic_data(
        symbol="BTCUSDT", n_points=2000, base_price=50000.0, volatility=0.02
    )
    prices = data.close

    # Run strategy
    strategy = HiPPOStrategy(n_hippo=32, trend_threshold=0.0, vol_filter=True)
    engine = BacktestEngine(annual_factor=8760.0)  # Hourly data

    metrics = engine.run(prices, strategy=strategy)
    logger.info(f"\n{metrics}")

    # Compare with buy-and-hold
    bh_positions = np.ones(len(prices))
    bh_metrics = engine.run(prices, positions=bh_positions)
    logger.info(f"\nBuy-and-Hold Benchmark:")
    logger.info(f"  Sharpe: {bh_metrics.sharpe_ratio:.2f}")
    logger.info(f"  Max DD: {bh_metrics.max_drawdown:.2%}")

    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo()
