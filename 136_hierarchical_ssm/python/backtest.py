"""Backtesting engine for HiSS-based trading strategies.

Evaluates trading strategy performance with standard financial metrics:
Sharpe ratio, Sortino ratio, maximum drawdown, win rate, and profit factor.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Trade:
    """Record of a single trade."""
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    entry_time: int
    exit_time: int
    pnl: float = 0.0

    def __post_init__(self):
        self.pnl = self.direction * (self.exit_price - self.entry_price)


@dataclass
class BacktestResult:
    """Complete backtest result with performance metrics."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class TradingStrategy:
    """HiSS-based trading strategy with risk management.

    Uses model predictions (return, direction probability, volatility)
    to generate trading signals and manage positions.

    Args:
        direction_threshold_long: Min direction probability for long entry
        direction_threshold_short: Max direction probability for short entry
        return_threshold: Min absolute predicted return for entry
        max_position_pct: Maximum position as fraction of portfolio
        stop_loss_multiplier: Stop-loss distance as multiple of predicted volatility
        take_profit_multiplier: Take-profit distance as multiple of predicted volatility
        max_drawdown_limit: Maximum allowed portfolio drawdown
        transaction_cost: Cost per trade as fraction of trade value
    """

    def __init__(
        self,
        direction_threshold_long: float = 0.6,
        direction_threshold_short: float = 0.4,
        return_threshold: float = 0.001,
        max_position_pct: float = 0.2,
        stop_loss_multiplier: float = 2.0,
        take_profit_multiplier: float = 3.0,
        max_drawdown_limit: float = 0.15,
        transaction_cost: float = 0.001,
    ):
        self.direction_threshold_long = direction_threshold_long
        self.direction_threshold_short = direction_threshold_short
        self.return_threshold = return_threshold
        self.max_position_pct = max_position_pct
        self.stop_loss_multiplier = stop_loss_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self.max_drawdown_limit = max_drawdown_limit
        self.transaction_cost = transaction_cost

    def generate_signal(
        self,
        pred_return: float,
        direction_prob: float,
        pred_volatility: float,
    ) -> int:
        """Generate trading signal from model predictions.

        Returns:
            1 for long, -1 for short, 0 for no position
        """
        if (
            direction_prob > self.direction_threshold_long
            and pred_return > self.return_threshold
        ):
            return 1
        elif (
            direction_prob < self.direction_threshold_short
            and pred_return < -self.return_threshold
        ):
            return -1
        return 0

    def compute_position_size(
        self,
        pred_return: float,
        pred_volatility: float,
        portfolio_value: float,
    ) -> float:
        """Compute position size using Kelly-inspired sizing.

        Args:
            pred_return: Predicted return
            pred_volatility: Predicted volatility
            portfolio_value: Current portfolio value

        Returns:
            Position size in currency units
        """
        if pred_volatility <= 0:
            return 0.0

        # Kelly fraction: f = mu / sigma^2
        kelly = abs(pred_return) / (pred_volatility ** 2 + 1e-8)
        # Scale down and cap
        fraction = min(kelly * 0.25, self.max_position_pct)
        return fraction * portfolio_value


class BacktestEngine:
    """Backtesting engine that simulates trading with HiSS predictions.

    Args:
        initial_capital: Starting portfolio value
        strategy: Trading strategy configuration
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        strategy: Optional[TradingStrategy] = None,
    ):
        self.initial_capital = initial_capital
        self.strategy = strategy or TradingStrategy()

    @torch.no_grad()
    def run(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        prices: np.ndarray,
        device: str = "cpu",
    ) -> BacktestResult:
        """Run backtest using model predictions.

        Args:
            model: Trained HiSS model
            dataloader: DataLoader for test data
            prices: Array of close prices aligned with test data
            device: Computation device

        Returns:
            BacktestResult with performance metrics
        """
        model.eval()

        # Collect all predictions
        all_pred_ret = []
        all_pred_dir = []
        all_pred_vol = []

        for batch in dataloader:
            features = batch[0].to(device)
            pred_ret, pred_dir, pred_vol = model(features)
            all_pred_ret.extend(pred_ret.squeeze().cpu().numpy().tolist())
            all_pred_dir.extend(pred_dir.squeeze().cpu().numpy().tolist())
            all_pred_vol.extend(pred_vol.squeeze().cpu().numpy().tolist())

        # Simulate trading
        portfolio_value = self.initial_capital
        peak_value = portfolio_value
        position = 0  # 1, -1, or 0
        entry_price = 0.0
        trades: List[Trade] = []
        equity_curve = [portfolio_value]
        max_drawdown = 0.0

        n_steps = min(len(all_pred_ret), len(prices) - 1)

        for t in range(n_steps):
            current_price = prices[t]
            next_price = prices[t + 1] if t + 1 < len(prices) else current_price

            pred_ret = all_pred_ret[t]
            pred_dir = all_pred_dir[t]
            pred_vol = all_pred_vol[t]

            signal = self.strategy.generate_signal(pred_ret, pred_dir, pred_vol)

            # Check stop-loss / take-profit for existing position
            if position != 0 and pred_vol > 0:
                price_change = (current_price - entry_price) * position
                sl_distance = self.strategy.stop_loss_multiplier * pred_vol * entry_price
                tp_distance = self.strategy.take_profit_multiplier * pred_vol * entry_price

                if price_change <= -sl_distance or price_change >= tp_distance:
                    # Close position
                    trade = Trade(
                        entry_price=entry_price,
                        exit_price=current_price,
                        direction=position,
                        entry_time=t - 1,
                        exit_time=t,
                    )
                    trades.append(trade)
                    cost = abs(current_price) * self.strategy.transaction_cost
                    portfolio_value += trade.pnl - cost
                    position = 0

            # Open new position if not in one
            if position == 0 and signal != 0:
                # Check max drawdown limit
                drawdown = (peak_value - portfolio_value) / peak_value
                if drawdown < self.strategy.max_drawdown_limit:
                    position = signal
                    entry_price = current_price
                    cost = abs(current_price) * self.strategy.transaction_cost
                    portfolio_value -= cost

            # Close position if signal reverses
            elif position != 0 and signal == -position:
                trade = Trade(
                    entry_price=entry_price,
                    exit_price=current_price,
                    direction=position,
                    entry_time=t - 1,
                    exit_time=t,
                )
                trades.append(trade)
                cost = abs(current_price) * self.strategy.transaction_cost
                portfolio_value += trade.pnl - cost
                position = 0

            # Track equity
            if position != 0:
                unrealized = position * (next_price - entry_price)
                equity_curve.append(portfolio_value + unrealized)
            else:
                equity_curve.append(portfolio_value)

            peak_value = max(peak_value, equity_curve[-1])
            current_dd = (peak_value - equity_curve[-1]) / peak_value
            max_drawdown = max(max_drawdown, current_dd)

        # Close any remaining position
        if position != 0 and n_steps > 0:
            trade = Trade(
                entry_price=entry_price,
                exit_price=prices[n_steps],
                direction=position,
                entry_time=n_steps - 1,
                exit_time=n_steps,
            )
            trades.append(trade)
            portfolio_value += trade.pnl

        return self._compute_metrics(trades, equity_curve, max_drawdown)

    def _compute_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        max_drawdown: float,
    ) -> BacktestResult:
        """Compute performance metrics from trade history."""
        if not trades:
            return BacktestResult(
                equity_curve=equity_curve,
                max_drawdown=max_drawdown,
            )

        pnls = [t.pnl for t in trades]
        returns = np.array(pnls) / self.initial_capital

        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside)
            sortino = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
        else:
            sortino = float("inf") if np.mean(returns) > 0 else 0.0

        # Win rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(trades),
            trades=trades,
            equity_curve=equity_curve,
        )
