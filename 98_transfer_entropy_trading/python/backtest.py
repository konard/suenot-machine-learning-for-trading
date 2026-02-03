"""Backtesting framework for Transfer Entropy-based trading strategies."""

import numpy as np
import pandas as pd
from typing import Optional

from .te_model import TransferEntropyEstimator, TENetworkBuilder


class TEBacktester:
    """Backtester for Transfer Entropy leader-follower strategies.

    The strategy monitors leader assets and generates signals for follower
    assets based on information flow detected by Transfer Entropy.

    Args:
        prices_df: DataFrame of asset prices (columns = asset names).
        returns_df: DataFrame of asset returns (columns = asset names).
        te_lookback: Rolling window for TE computation.
        signal_threshold: Minimum leader return magnitude to trigger signal.
        te_threshold: Minimum TE to consider a pair as leader-follower.
        initial_capital: Starting capital.
        transaction_cost: Cost per trade as a fraction.
        history_length: TE history parameter.
        n_bins: Number of bins for TE estimation.
    """

    def __init__(
        self,
        prices_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        te_lookback: int = 60,
        signal_threshold: float = 0.01,
        te_threshold: float = 0.01,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        history_length: int = 3,
        n_bins: int = 8,
    ):
        self.prices = prices_df
        self.returns = returns_df
        self.te_lookback = te_lookback
        self.signal_threshold = signal_threshold
        self.te_threshold = te_threshold
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.history_length = history_length
        self.n_bins = n_bins

        self.symbols = list(prices_df.columns)
        self.estimator = TransferEntropyEstimator(
            method="binning", n_bins=n_bins, history_length=history_length
        )
        self.network_builder = TENetworkBuilder(self.symbols, self.estimator)

        self.results: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        """Run the backtest.

        Returns:
            DataFrame with columns: capital, position, signal, returns.
        """
        n = len(self.prices)
        capitals = np.full(n, np.nan)
        positions = np.zeros(n)
        signals = np.zeros(n)

        capitals[0] = self.initial_capital
        capital = self.initial_capital
        position = 0.0  # -1, 0, or 1

        # We trade the last asset (strongest follower) based on first asset (leader)
        leader_idx = 0
        follower_idx = len(self.symbols) - 1

        leader_returns = self.returns.iloc[:, leader_idx].values
        follower_returns = self.returns.iloc[:, follower_idx].values

        for t in range(1, n):
            # Update TE periodically to find leader-follower pairs
            if t >= self.te_lookback and t % self.te_lookback == 0:
                window_returns = {
                    sym: self.returns[sym].values[t - self.te_lookback : t]
                    for sym in self.symbols
                }
                try:
                    te_matrix = self.network_builder.compute_te_matrix(window_returns)
                    net_te = self.network_builder.get_net_te(te_matrix)
                    sorted_symbols = sorted(
                        net_te.items(), key=lambda x: x[1], reverse=True
                    )
                    leader_idx = self.symbols.index(sorted_symbols[0][0])
                    follower_idx = self.symbols.index(sorted_symbols[-1][0])
                    leader_returns = self.returns.iloc[:, leader_idx].values
                    follower_returns = self.returns.iloc[:, follower_idx].values
                except (ValueError, ZeroDivisionError):
                    pass

            # Generate signal based on leader's return
            if t >= self.te_lookback:
                leader_ret = leader_returns[t - 1]
                if leader_ret > self.signal_threshold:
                    new_signal = 1.0
                elif leader_ret < -self.signal_threshold:
                    new_signal = -1.0
                else:
                    new_signal = 0.0
            else:
                new_signal = 0.0

            signals[t] = new_signal

            # Apply transaction cost on position change
            if new_signal != position:
                capital *= 1.0 - self.transaction_cost
                position = new_signal

            positions[t] = position

            # Compute PnL based on follower's return
            ret = follower_returns[t]
            capital *= 1.0 + position * ret
            capitals[t] = capital

        # Fill NaN for warm-up period
        capitals[0] = self.initial_capital
        for i in range(1, min(self.te_lookback, n)):
            if np.isnan(capitals[i]):
                capitals[i] = capitals[i - 1]

        self.results = pd.DataFrame(
            {
                "capital": capitals,
                "position": positions,
                "signal": signals,
            }
        )
        return self.results

    def calculate_metrics(self) -> dict:
        """Calculate performance metrics from backtest results.

        Returns:
            Dictionary of performance metrics.
        """
        if self.results is None:
            raise RuntimeError("Must call run() before calculate_metrics().")

        capital = self.results["capital"].values
        capital = capital[~np.isnan(capital)]

        if len(capital) < 2:
            return {"error": "Not enough data for metrics computation."}

        # Returns
        returns = np.diff(capital) / capital[:-1]
        returns = returns[np.isfinite(returns)]

        total_return = capital[-1] / capital[0] - 1.0
        n_periods = len(returns)
        ann_factor = 252  # assume daily data

        # Annualized return
        if total_return > -1.0:
            ann_return = (1 + total_return) ** (ann_factor / max(n_periods, 1)) - 1
        else:
            ann_return = -1.0

        # Volatility
        ann_vol = np.std(returns) * np.sqrt(ann_factor) if len(returns) > 1 else 0.0

        # Sharpe ratio
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = (
            np.sqrt(ann_factor) * mean_ret / std_ret if std_ret > 0 else 0.0
        )

        # Sortino ratio
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 1 else 0.0
        sortino = (
            np.sqrt(ann_factor) * mean_ret / downside_std
            if downside_std > 0
            else 0.0
        )

        # Max drawdown
        cummax = np.maximum.accumulate(capital)
        drawdowns = (capital - cummax) / cummax
        max_drawdown = np.min(drawdowns)

        # Win rate
        positions = self.results["position"].values
        trade_returns = []
        for i in range(1, len(positions)):
            if positions[i] != 0:
                trade_returns.append(returns[i - 1] if i - 1 < len(returns) else 0)

        wins = sum(1 for r in trade_returns if r > 0)
        win_rate = wins / len(trade_returns) if trade_returns else 0.0

        # Profit factor
        gross_profit = sum(r for r in trade_returns if r > 0)
        gross_loss = abs(sum(r for r in trade_returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Number of trades
        n_trades = sum(
            1 for i in range(1, len(positions)) if positions[i] != positions[i - 1]
        )

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": n_trades,
            "n_periods": n_periods,
        }
