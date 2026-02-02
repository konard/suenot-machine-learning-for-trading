"""
Backtesting framework for meta-volatility prediction strategies.

Implements a volatility-adaptive position sizing strategy where
the predicted volatility drives position sizes and risk management.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    target_daily_risk: float = 0.02  # 2% daily risk budget
    max_position_pct: float = 0.5    # Max 50% of capital in one position
    min_position_pct: float = 0.01   # Min 1% position
    transaction_cost_bps: float = 10.0  # 10 bps round-trip
    slippage_bps: float = 5.0        # 5 bps slippage


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: pd.Timestamp
    direction: int  # 1 for long, -1 for short
    position_size: float
    entry_price: float
    exit_price: float = 0.0
    pnl: float = 0.0
    predicted_vol: float = 0.0
    realized_vol: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    trades: List[Trade] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    total_return: float = 0.0
    vol_forecast_mse: float = 0.0
    vol_forecast_mae: float = 0.0
    hit_rate: float = 0.0


def compute_position_size(
    predicted_vol: float,
    config: BacktestConfig,
) -> float:
    """Compute position size based on predicted volatility.

    position_size = target_risk / predicted_volatility
    Clipped to [min_position_pct, max_position_pct].
    """
    if predicted_vol <= 1e-8:
        return config.max_position_pct

    raw_size = config.target_daily_risk / predicted_vol
    return np.clip(raw_size, config.min_position_pct, config.max_position_pct)


def run_backtest(
    prices: np.ndarray,
    predicted_vols: np.ndarray,
    realized_vols: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """Run volatility-adaptive position sizing backtest.

    Strategy:
        - Size positions inversely proportional to predicted volatility
        - Higher predicted vol -> smaller position (risk reduction)
        - Lower predicted vol -> larger position (risk taking)

    Args:
        prices: Array of prices.
        predicted_vols: Array of predicted volatilities.
        realized_vols: Array of realized volatilities.
        timestamps: Optional timestamps for trades.
        config: Backtest configuration.

    Returns:
        BacktestResult with performance metrics.
    """
    if config is None:
        config = BacktestConfig()

    n = len(prices)
    if n < 2:
        return BacktestResult()

    equity = config.initial_capital
    equity_curve = [equity]
    daily_returns = []
    trades = []
    cost_bps = (config.transaction_cost_bps + config.slippage_bps) / 10_000.0

    for i in range(1, n):
        pos_pct = compute_position_size(predicted_vols[i - 1], config)
        position_value = equity * pos_pct

        # Simple momentum signal: go long if price trending up
        price_return = (prices[i] - prices[i - 1]) / prices[i - 1]
        direction = 1  # Always long for simplicity

        gross_pnl = position_value * price_return * direction
        cost = position_value * cost_bps * 2  # Round-trip cost
        net_pnl = gross_pnl - cost

        equity += net_pnl
        equity_curve.append(equity)
        daily_returns.append(net_pnl / (equity - net_pnl))

        ts = timestamps[i] if timestamps is not None else pd.Timestamp.now()
        trades.append(Trade(
            timestamp=ts,
            direction=direction,
            position_size=pos_pct,
            entry_price=prices[i - 1],
            exit_price=prices[i],
            pnl=net_pnl,
            predicted_vol=predicted_vols[i - 1],
            realized_vol=realized_vols[i - 1] if i - 1 < len(realized_vols) else 0.0,
        ))

    returns = np.array(daily_returns)
    equity_arr = np.array(equity_curve)

    # Performance metrics
    result = BacktestResult(
        returns=returns,
        equity_curve=equity_arr,
        trades=trades,
        total_return=(equity_arr[-1] / equity_arr[0]) - 1.0,
    )

    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(returns) > 1 and returns.std() > 1e-10:
        result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

    # Sortino ratio (downside deviation)
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 1e-10:
        result.sortino_ratio = (
            returns.mean() / downside.std()
        ) * np.sqrt(252)

    # Maximum drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    result.max_drawdown = drawdown.min()

    # Calmar ratio
    if abs(result.max_drawdown) > 1e-10:
        annual_return = result.total_return * (252 / max(len(returns), 1))
        result.calmar_ratio = annual_return / abs(result.max_drawdown)

    # Volatility forecast accuracy
    min_len = min(len(predicted_vols), len(realized_vols))
    if min_len > 0:
        pv = predicted_vols[:min_len]
        rv = realized_vols[:min_len]
        result.vol_forecast_mse = float(np.mean((pv - rv) ** 2))
        result.vol_forecast_mae = float(np.mean(np.abs(pv - rv)))

        # Hit rate: did predicted vol direction match realized?
        if min_len > 1:
            pred_diff = np.diff(pv)
            real_diff = np.diff(rv)
            hits = np.sign(pred_diff) == np.sign(real_diff)
            result.hit_rate = float(hits.mean())

    return result


def print_backtest_summary(result: BacktestResult) -> None:
    """Print a summary of backtest results."""
    print("=" * 50)
    print("Backtest Results")
    print("=" * 50)
    print(f"Total Return:      {result.total_return:>10.2%}")
    print(f"Sharpe Ratio:      {result.sharpe_ratio:>10.3f}")
    print(f"Sortino Ratio:     {result.sortino_ratio:>10.3f}")
    print(f"Max Drawdown:      {result.max_drawdown:>10.2%}")
    print(f"Calmar Ratio:      {result.calmar_ratio:>10.3f}")
    print(f"Vol Forecast MSE:  {result.vol_forecast_mse:>10.6f}")
    print(f"Vol Forecast MAE:  {result.vol_forecast_mae:>10.6f}")
    print(f"Hit Rate:          {result.hit_rate:>10.2%}")
    print(f"Number of Trades:  {len(result.trades):>10d}")
    print("=" * 50)
