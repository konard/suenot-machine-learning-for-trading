"""
Backtesting framework for trading strategies with counterfactual insights.

This module provides tools for backtesting trading strategies and
evaluating the usefulness of counterfactual explanations for risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import torch
from .counterfactual import CounterfactualOptimizer, CounterfactualResult


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    direction: str  # 'long' or 'short'
    size: float
    pnl: float
    pnl_pct: float
    counterfactual_distance: Optional[float] = None


@dataclass
class BacktestResult:
    """Results of a backtest."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    trades: List[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))

    def __str__(self) -> str:
        return (
            f"Backtest Results\n"
            f"================\n"
            f"Total Return:    {self.total_return:.2%}\n"
            f"Sharpe Ratio:    {self.sharpe_ratio:.2f}\n"
            f"Sortino Ratio:   {self.sortino_ratio:.2f}\n"
            f"Max Drawdown:    {self.max_drawdown:.2%}\n"
            f"Win Rate:        {self.win_rate:.2%}\n"
            f"Number of Trades: {self.num_trades}"
        )


class Backtester:
    """
    Backtester for trading strategies with counterfactual analysis.

    Supports simple trading strategies based on model predictions,
    with optional counterfactual-based risk management.

    Args:
        initial_capital: Starting capital
        commission: Trading commission rate
        slippage: Slippage rate
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        cf_distances: Optional[np.ndarray] = None,
        cf_threshold: Optional[float] = None
    ) -> BacktestResult:
        """
        Run backtest on price data with trading signals.

        Args:
            prices: Array of prices
            signals: Array of signals (0=SELL, 1=HOLD, 2=BUY)
            cf_distances: Optional counterfactual distances for risk management
            cf_threshold: Minimum CF distance to take a trade

        Returns:
            BacktestResult with performance metrics
        """
        n = len(prices)
        equity = np.zeros(n)
        equity[0] = self.initial_capital

        position = 0  # 1=long, -1=short, 0=flat
        entry_price = 0.0
        entry_time = 0
        trades = []

        for i in range(1, n):
            # Apply CF threshold filter if provided
            if cf_distances is not None and cf_threshold is not None:
                if cf_distances[i] < cf_threshold:
                    signals[i] = 1  # Force HOLD if CF distance too small

            # Generate trading actions
            signal = signals[i]

            # Entry logic
            if position == 0:
                if signal == 2:  # BUY
                    position = 1
                    entry_price = prices[i] * (1 + self.slippage)
                    entry_time = i
                elif signal == 0:  # SELL
                    position = -1
                    entry_price = prices[i] * (1 - self.slippage)
                    entry_time = i

            # Exit logic
            elif position == 1:  # Long position
                if signal == 0:  # SELL signal = close long
                    exit_price = prices[i] * (1 - self.slippage)
                    pnl_pct = (exit_price / entry_price - 1) - 2 * self.commission
                    pnl = equity[i-1] * pnl_pct

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=i,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction='long',
                        size=1.0,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        counterfactual_distance=cf_distances[entry_time] if cf_distances is not None else None
                    ))

                    equity[i] = equity[i-1] + pnl
                    position = 0
                else:
                    # Mark to market
                    pnl_pct = (prices[i] / entry_price - 1)
                    equity[i] = equity[i-1] * (1 + pnl_pct * 0.01)  # Simplified MTM

            elif position == -1:  # Short position
                if signal == 2:  # BUY signal = close short
                    exit_price = prices[i] * (1 + self.slippage)
                    pnl_pct = (entry_price / exit_price - 1) - 2 * self.commission
                    pnl = equity[i-1] * pnl_pct

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=i,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        direction='short',
                        size=1.0,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        counterfactual_distance=cf_distances[entry_time] if cf_distances is not None else None
                    ))

                    equity[i] = equity[i-1] + pnl
                    position = 0
                else:
                    # Mark to market
                    pnl_pct = (entry_price / prices[i] - 1)
                    equity[i] = equity[i-1] * (1 + pnl_pct * 0.01)

            # No position change
            if equity[i] == 0:
                equity[i] = equity[i-1]

        # Calculate metrics
        daily_returns = np.diff(equity) / equity[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]

        total_return = (equity[-1] / self.initial_capital) - 1
        sharpe = self._calculate_sharpe(daily_returns)
        sortino = self._calculate_sortino(daily_returns)
        max_dd = self._calculate_max_drawdown(equity)

        winning_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = winning_trades / len(trades) if trades else 0.0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=len(trades),
            trades=trades,
            equity_curve=equity,
            daily_returns=daily_returns
        )

    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)

    def _calculate_sortino(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns)
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std

    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return np.max(drawdown)


class CounterfactualRiskManager:
    """
    Risk manager using counterfactual explanations.

    Uses counterfactual distance as a measure of prediction stability
    to adjust position sizing and filter trades.

    Args:
        model: Trading classifier model
        cf_optimizer: Counterfactual optimizer
        min_distance: Minimum CF distance to take a trade
        scale_by_distance: Whether to scale position size by CF distance
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cf_optimizer: CounterfactualOptimizer,
        min_distance: float = 0.5,
        scale_by_distance: bool = True
    ):
        self.model = model
        self.cf_optimizer = cf_optimizer
        self.min_distance = min_distance
        self.scale_by_distance = scale_by_distance

    def evaluate_prediction(
        self,
        x: torch.Tensor,
        prediction: int
    ) -> Tuple[bool, float, Optional[CounterfactualResult]]:
        """
        Evaluate a prediction using counterfactual analysis.

        Args:
            x: Input features
            prediction: Model prediction (0, 1, or 2)

        Returns:
            Tuple of (should_trade, position_size, counterfactual_result)
        """
        # Only generate CF for non-HOLD predictions
        if prediction == 1:
            return False, 0.0, None

        # Find counterfactual to opposite class
        target_class = 0 if prediction == 2 else 2
        cf_result = self.cf_optimizer.generate(x, target_class)

        # Calculate distance
        distance = cf_result.l1_distance

        # Filter by minimum distance
        if distance < self.min_distance:
            return False, 0.0, cf_result

        # Calculate position size
        if self.scale_by_distance:
            # Scale between 0.5 and 1.0 based on distance
            position_size = min(1.0, 0.5 + distance * 0.5)
        else:
            position_size = 1.0

        return True, position_size, cf_result

    def analyze_trades(
        self,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """
        Analyze trade performance by counterfactual distance.

        Args:
            trades: List of trades with CF distances

        Returns:
            Analysis dictionary
        """
        trades_with_cf = [t for t in trades if t.counterfactual_distance is not None]

        if not trades_with_cf:
            return {}

        distances = np.array([t.counterfactual_distance for t in trades_with_cf])
        pnls = np.array([t.pnl_pct for t in trades_with_cf])

        # Split by median distance
        median_dist = np.median(distances)
        high_conf_mask = distances >= median_dist
        low_conf_mask = distances < median_dist

        return {
            'mean_distance': float(np.mean(distances)),
            'median_distance': float(median_dist),
            'high_conf_win_rate': float(np.mean(pnls[high_conf_mask] > 0)) if high_conf_mask.any() else 0,
            'low_conf_win_rate': float(np.mean(pnls[low_conf_mask] > 0)) if low_conf_mask.any() else 0,
            'high_conf_avg_pnl': float(np.mean(pnls[high_conf_mask])) if high_conf_mask.any() else 0,
            'low_conf_avg_pnl': float(np.mean(pnls[low_conf_mask])) if low_conf_mask.any() else 0,
            'correlation': float(np.corrcoef(distances, pnls)[0, 1]) if len(distances) > 1 else 0
        }


def evaluate_counterfactual_quality(
    results: List[CounterfactualResult]
) -> Dict[str, float]:
    """
    Evaluate the quality of counterfactual explanations.

    Args:
        results: List of CounterfactualResult objects

    Returns:
        Quality metrics dictionary
    """
    if not results:
        return {}

    validity_rate = np.mean([r.is_valid for r in results])
    avg_sparsity = np.mean([r.num_features_changed for r in results])
    avg_l1 = np.mean([r.l1_distance for r in results])
    avg_l2 = np.mean([r.l2_distance for r in results])

    return {
        'validity_rate': float(validity_rate),
        'avg_features_changed': float(avg_sparsity),
        'avg_l1_distance': float(avg_l1),
        'avg_l2_distance': float(avg_l2)
    }
