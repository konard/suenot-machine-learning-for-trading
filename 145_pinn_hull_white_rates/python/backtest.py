"""
Interest Rate Strategy Backtester

Backtest strategies based on Hull-White model predictions:
1. Yield curve mean-reversion strategy (traditional)
2. Funding rate mean-reversion strategy (crypto/Bybit)
3. Bond duration management strategy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from analytical import HullWhiteAnalytical, FlatCurveHullWhite
from calibration import calibrate_hull_white
from data_loader import (
    generate_synthetic_funding_rates,
    load_sample_treasury_curve,
    compute_funding_rate_statistics,
)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: int
    exit_time: int = 0
    direction: int = 0  # 1 = long, -1 = short
    entry_rate: float = 0.0
    exit_rate: float = 0.0
    pnl: float = 0.0
    holding_periods: int = 0


@dataclass
class BacktestResult:
    """Holds backtest performance metrics."""
    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_holding_period: float
    profit_factor: float
    equity_curve: np.ndarray
    trades: List[Trade] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class FundingRateMeanReversionStrategy:
    """
    Mean-reversion strategy on crypto funding rates.

    The Hull-White model implies that funding rates revert to a long-run mean.
    When the rate deviates significantly from this mean, we trade in the
    direction of mean reversion.

    Strategy logic:
    - If rate > mean + threshold: short (expect rate to decrease, long the asset)
    - If rate < mean - threshold: long (expect rate to increase, short the asset)
    - Close when rate returns to within band of mean

    In practice, this translates to:
    - High positive funding -> short perpetual (collect funding, expect rate to drop)
    - High negative funding -> long perpetual (collect funding, expect rate to rise)
    """

    def __init__(
        self,
        lookback: int = 100,
        entry_z_score: float = 1.5,
        exit_z_score: float = 0.5,
        max_holding: int = 50,
        position_size: float = 1.0,
    ):
        """
        Args:
            lookback: Window for estimating mean and std
            entry_z_score: Z-score threshold for entry
            exit_z_score: Z-score threshold for exit
            max_holding: Maximum holding period (in 8h intervals)
            position_size: Fraction of capital to deploy
        """
        self.lookback = lookback
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.max_holding = max_holding
        self.position_size = position_size

    def run(self, funding_rates: np.ndarray) -> BacktestResult:
        """
        Run the backtest.

        Args:
            funding_rates: Array of funding rates

        Returns:
            BacktestResult
        """
        n = len(funding_rates)
        if n < self.lookback + 10:
            raise ValueError(f"Need at least {self.lookback + 10} observations")

        equity = np.ones(n)
        position = 0  # 0 = flat, 1 = long, -1 = short
        trades: List[Trade] = []
        current_trade: Optional[Trade] = None
        holding_count = 0

        for i in range(self.lookback, n):
            window = funding_rates[i - self.lookback:i]
            mean = np.mean(window)
            std = np.std(window)

            if std < 1e-10:
                std = 1e-10

            z_score = (funding_rates[i] - mean) / std

            # Position management
            if position == 0:
                # Entry signals
                if z_score > self.entry_z_score:
                    # Rate too high -> short perpetual (we earn funding)
                    position = -1
                    current_trade = Trade(
                        entry_time=i,
                        direction=-1,
                        entry_rate=funding_rates[i],
                    )
                    holding_count = 0
                elif z_score < -self.entry_z_score:
                    # Rate too low -> long perpetual (we earn funding)
                    position = 1
                    current_trade = Trade(
                        entry_time=i,
                        direction=1,
                        entry_rate=funding_rates[i],
                    )
                    holding_count = 0
            else:
                holding_count += 1

                # Exit signals
                should_exit = False
                if position == -1 and z_score < self.exit_z_score:
                    should_exit = True
                elif position == 1 and z_score > -self.exit_z_score:
                    should_exit = True
                elif holding_count >= self.max_holding:
                    should_exit = True

                if should_exit and current_trade is not None:
                    current_trade.exit_time = i
                    current_trade.exit_rate = funding_rates[i]
                    current_trade.holding_periods = holding_count

                    # PnL from funding payments collected
                    # When short with positive funding: we earn
                    # When long with negative funding: we earn
                    funding_collected = 0.0
                    for j in range(current_trade.entry_time, i):
                        funding_collected += -position * funding_rates[j]

                    current_trade.pnl = funding_collected * self.position_size
                    trades.append(current_trade)

                    position = 0
                    current_trade = None

            # Update equity (simplified: funding PnL accrual)
            if position != 0:
                period_pnl = -position * funding_rates[i] * self.position_size
                equity[i] = equity[i - 1] * (1 + period_pnl)
            else:
                equity[i] = equity[i - 1]

        return self._compute_metrics(equity, trades, "Funding Rate Mean Reversion")

    def _compute_metrics(
        self, equity: np.ndarray, trades: List[Trade], name: str
    ) -> BacktestResult:
        """Compute backtest performance metrics."""
        returns = np.diff(equity) / equity[:-1]
        returns = returns[returns != 0]  # Remove flat periods

        total_return = equity[-1] / equity[0] - 1
        n_periods = len(equity)
        periods_per_year = 3 * 365  # 8h intervals

        if n_periods > 1:
            ann_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        else:
            ann_return = 0.0

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0

        # Max drawdown
        cummax = np.maximum.accumulate(equity)
        drawdowns = (cummax - equity) / cummax
        max_dd = np.max(drawdowns)

        # Trade statistics
        n_trades = len(trades)
        if n_trades > 0:
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate = wins / n_trades
            avg_holding = np.mean([t.holding_periods for t in trades])
            gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            win_rate = 0.0
            avg_holding = 0.0
            profit_factor = 0.0

        return BacktestResult(
            strategy_name=name,
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=n_trades,
            avg_holding_period=avg_holding,
            profit_factor=profit_factor,
            equity_curve=equity,
            trades=trades,
            metrics={
                "mean_return_per_period": float(np.mean(returns)) if len(returns) > 0 else 0,
                "std_return_per_period": float(np.std(returns)) if len(returns) > 0 else 0,
                "skewness": float(pd.Series(returns).skew()) if len(returns) > 1 else 0,
                "kurtosis": float(pd.Series(returns).kurtosis()) if len(returns) > 1 else 0,
            }
        )


class BondDurationStrategy:
    """
    Duration management strategy based on Hull-White rate predictions.

    Uses the Hull-White model to predict the direction of rate changes
    and adjusts portfolio duration accordingly.

    When rates are expected to decrease -> increase duration (buy long bonds)
    When rates are expected to increase -> decrease duration (buy short bonds)
    """

    def __init__(
        self,
        a: float = 0.1,
        sigma: float = 0.01,
        short_maturity: float = 1.0,
        long_maturity: float = 10.0,
        rebalance_frequency: int = 21,  # monthly in trading days
    ):
        self.a = a
        self.sigma = sigma
        self.short_maturity = short_maturity
        self.long_maturity = long_maturity
        self.rebalance_frequency = rebalance_frequency

    def run(
        self, rate_path: np.ndarray, dt: float = 1 / 252
    ) -> BacktestResult:
        """
        Run duration management backtest on a simulated rate path.

        Args:
            rate_path: Array of daily short rates
            dt: Time step in years

        Returns:
            BacktestResult
        """
        n = len(rate_path)
        hw = FlatCurveHullWhite(a=self.a, sigma=self.sigma, flat_rate=np.mean(rate_path))

        equity = np.ones(n)
        position_weight_long = 0.5  # initial 50/50 allocation
        trades: List[Trade] = []

        for i in range(1, n):
            current_rate = rate_path[i]
            prev_rate = rate_path[i - 1]

            # Rebalance at frequency
            if i % self.rebalance_frequency == 0:
                # Hull-White mean reversion prediction
                long_run = np.mean(rate_path[:i])
                expected_dr = self.a * (long_run - current_rate)

                if expected_dr < 0:
                    # Rates expected to fall -> increase duration
                    position_weight_long = min(0.8, position_weight_long + 0.1)
                else:
                    # Rates expected to rise -> decrease duration
                    position_weight_long = max(0.2, position_weight_long - 0.1)

            # Portfolio PnL approximation using duration
            dr = current_rate - prev_rate
            duration_short = self.short_maturity
            duration_long = self.long_maturity

            # Bond return ≈ -duration * dr + carry
            carry_short = prev_rate * dt
            carry_long = prev_rate * dt  # simplified

            return_short = -duration_short * dr + carry_short
            return_long = -duration_long * dr + carry_long

            # Portfolio return
            portfolio_return = (
                (1 - position_weight_long) * return_short
                + position_weight_long * return_long
            )

            equity[i] = equity[i - 1] * (1 + portfolio_return)

        return self._compute_metrics(equity, trades, "Bond Duration Management")

    def _compute_metrics(self, equity, trades, name):
        """Compute backtest metrics."""
        returns = np.diff(equity) / equity[:-1]

        total_return = equity[-1] / equity[0] - 1
        ann_return = (1 + total_return) ** (252 / len(equity)) - 1 if len(equity) > 1 else 0

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        cummax = np.maximum.accumulate(equity)
        drawdowns = (cummax - equity) / cummax
        max_dd = np.max(drawdowns)

        return BacktestResult(
            strategy_name=name,
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=0.0,
            n_trades=0,
            avg_holding_period=0,
            profit_factor=0,
            equity_curve=equity,
            trades=trades,
        )


def print_backtest_report(result: BacktestResult):
    """Print a formatted backtest report."""
    print(f"\n{'=' * 60}")
    print(f"  Backtest Report: {result.strategy_name}")
    print(f"{'=' * 60}")
    print(f"  Total Return:       {result.total_return:>10.2%}")
    print(f"  Annualized Return:  {result.annualized_return:>10.2%}")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:>10.2f}")
    print(f"  Max Drawdown:       {result.max_drawdown:>10.2%}")
    print(f"  Win Rate:           {result.win_rate:>10.2%}")
    print(f"  Number of Trades:   {result.n_trades:>10d}")
    print(f"  Avg Holding Period: {result.avg_holding_period:>10.1f}")
    print(f"  Profit Factor:      {result.profit_factor:>10.2f}")
    if result.metrics:
        print(f"\n  Additional Metrics:")
        for key, val in result.metrics.items():
            if isinstance(val, float):
                print(f"    {key}: {val:.6f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Hull-White Interest Rate Strategy Backtest")
    print("=" * 60)

    # Strategy 1: Funding Rate Mean Reversion (Crypto)
    print("\n--- Strategy 1: Funding Rate Mean Reversion ---")
    funding_df = generate_synthetic_funding_rates(
        n_periods=2000,
        mean_rate=0.0001,
        mean_reversion=0.3,
        volatility=0.0003,
        seed=42,
    )

    strategy1 = FundingRateMeanReversionStrategy(
        lookback=100,
        entry_z_score=1.5,
        exit_z_score=0.5,
        max_holding=50,
    )
    result1 = strategy1.run(funding_df["funding_rate"].values)
    print_backtest_report(result1)

    # Strategy 2: Bond Duration Management (Traditional)
    print("\n--- Strategy 2: Bond Duration Management ---")
    hw = FlatCurveHullWhite(a=0.1, sigma=0.01, flat_rate=0.03)
    times, paths = hw.simulate_rate_path(0.03, T=5.0, n_steps=1260, n_paths=1, seed=42)

    strategy2 = BondDurationStrategy(a=0.1, sigma=0.01)
    result2 = strategy2.run(paths[0])
    print_backtest_report(result2)

    # Compare strategies
    print("\n--- Strategy Comparison ---")
    print(f"{'Metric':<25} {'Funding MR':>15} {'Duration Mgmt':>15}")
    print("-" * 55)
    print(f"{'Total Return':<25} {result1.total_return:>14.2%} {result2.total_return:>14.2%}")
    print(f"{'Ann. Return':<25} {result1.annualized_return:>14.2%} {result2.annualized_return:>14.2%}")
    print(f"{'Sharpe Ratio':<25} {result1.sharpe_ratio:>14.2f} {result2.sharpe_ratio:>14.2f}")
    print(f"{'Max Drawdown':<25} {result1.max_drawdown:>14.2%} {result2.max_drawdown:>14.2%}")
