"""
Backtesting framework for saliency-based trading strategies.

This module provides tools for backtesting trading strategies that use
saliency maps for signal filtering and position sizing.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass
from .saliency import SaliencyComputer, compute_saliency_concentration, aggregate_feature_importance


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    """Results from a backtest."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series


class SaliencyTradingStrategy:
    """
    Trading strategy that uses saliency maps for filtering and sizing.

    Args:
        model: Trained PyTorch model
        saliency_method: Method for computing saliency ('vanilla', 'gradient_x_input', etc.)
        min_confidence: Minimum prediction confidence to trade
        min_concentration: Minimum saliency concentration to trade
        interpretable_features: Indices of features that should be important for valid signals
        position_sizing: 'fixed', 'concentration', or 'confidence'
    """

    def __init__(
        self,
        model: nn.Module,
        saliency_method: str = 'gradient_x_input',
        min_confidence: float = 0.6,
        min_concentration: float = 0.3,
        interpretable_features: Optional[List[int]] = None,
        position_sizing: str = 'fixed'
    ):
        self.model = model
        self.model.eval()
        self.saliency_computer = SaliencyComputer(model)
        self.saliency_method = saliency_method
        self.min_confidence = min_confidence
        self.min_concentration = min_concentration
        self.interpretable_features = interpretable_features
        self.position_sizing = position_sizing

    def compute_saliency(self, x: torch.Tensor) -> torch.Tensor:
        """Compute saliency using the configured method."""
        if self.saliency_method == 'vanilla':
            return self.saliency_computer.vanilla_gradient(x)
        elif self.saliency_method == 'gradient_x_input':
            return self.saliency_computer.gradient_x_input(x)
        elif self.saliency_method == 'integrated':
            return self.saliency_computer.integrated_gradients(x)
        elif self.saliency_method == 'smoothgrad':
            return self.saliency_computer.smoothgrad(x)
        else:
            raise ValueError(f"Unknown saliency method: {self.saliency_method}")

    def check_interpretability(
        self,
        saliency: torch.Tensor,
        top_k: int = 3
    ) -> bool:
        """
        Check if the top salient features are interpretable.

        Args:
            saliency: Saliency map
            top_k: Number of top features to check

        Returns:
            True if top features are in the interpretable set
        """
        if self.interpretable_features is None:
            return True

        feature_importance = aggregate_feature_importance(saliency)
        if feature_importance.dim() == 2:
            feature_importance = feature_importance[0]

        top_features = feature_importance.topk(top_k).indices.tolist()

        overlap = len(set(top_features) & set(self.interpretable_features))
        return overlap >= (top_k // 2 + 1)

    def generate_signal(
        self,
        x: torch.Tensor
    ) -> Tuple[int, float]:
        """
        Generate a trading signal.

        Args:
            x: Input tensor of shape (1, sequence_length, features)

        Returns:
            Tuple of (signal, position_size) where signal is -1, 0, or 1
        """
        with torch.no_grad():
            prediction = self.model(x).item()

        saliency = self.compute_saliency(x)
        concentration = compute_saliency_concentration(saliency).item()

        if not self.check_interpretability(saliency):
            return 0, 0.0

        if concentration < self.min_concentration:
            return 0, 0.0

        if prediction > self.min_confidence:
            signal = 1
        elif prediction < (1 - self.min_confidence):
            signal = -1
        else:
            return 0, 0.0

        if self.position_sizing == 'fixed':
            position_size = 1.0
        elif self.position_sizing == 'concentration':
            position_size = concentration
        elif self.position_sizing == 'confidence':
            position_size = abs(prediction - 0.5) * 2
        else:
            position_size = 1.0

        return signal, position_size


def run_backtest(
    strategy: SaliencyTradingStrategy,
    features: np.ndarray,
    prices: pd.Series,
    dates: pd.DatetimeIndex,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    max_position: float = 1.0
) -> BacktestResult:
    """
    Run a backtest on the given data.

    Args:
        strategy: Trading strategy to test
        features: Feature array of shape (num_samples, sequence_length, num_features)
        prices: Price series
        dates: Date index
        initial_capital: Starting capital
        transaction_cost: Transaction cost as fraction of trade value
        max_position: Maximum position size as fraction of capital

    Returns:
        BacktestResult object
    """
    capital = initial_capital
    position = 0
    position_size = 0.0
    entry_price = 0.0
    entry_date = None

    trades: List[Trade] = []
    equity = [capital]
    equity_dates = [dates[0]]

    for i in range(len(features)):
        x = torch.FloatTensor(features[i:i+1])
        current_price = prices.iloc[i]
        current_date = dates[i]

        signal, size = strategy.generate_signal(x)

        if position != 0 and signal != position:
            exit_price = current_price
            trade_cost = abs(position_size * exit_price * transaction_cost)

            if position == 1:
                pnl = position_size * (exit_price - entry_price) - trade_cost
            else:
                pnl = position_size * (entry_price - exit_price) - trade_cost

            pnl_pct = pnl / (position_size * entry_price)

            capital += pnl

            trades.append(Trade(
                entry_date=entry_date,
                exit_date=current_date,
                direction=position,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                pnl=pnl,
                pnl_pct=pnl_pct
            ))

            position = 0
            position_size = 0.0

        if signal != 0 and position == 0:
            position = signal
            entry_price = current_price
            entry_date = current_date
            position_size = size * max_position * capital / current_price
            trade_cost = position_size * current_price * transaction_cost
            capital -= trade_cost

        if position != 0:
            if position == 1:
                current_equity = capital + position_size * (current_price - entry_price)
            else:
                current_equity = capital + position_size * (entry_price - current_price)
            equity.append(current_equity)
        else:
            equity.append(capital)

        equity_dates.append(current_date)

    equity_curve = pd.Series(equity, index=equity_dates)
    daily_returns = equity_curve.pct_change().dropna()

    total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    sharpe_ratio = compute_sharpe_ratio(daily_returns)
    sortino_ratio = compute_sortino_ratio(daily_returns)
    max_drawdown = compute_max_drawdown(equity_curve)

    if len(trades) > 0:
        win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / (gross_loss + 1e-10)
    else:
        win_rate = 0.0
        profit_factor = 0.0

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=len(trades),
        trades=trades,
        equity_curve=equity_curve,
        daily_returns=daily_returns
    )


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252
) -> float:
    """Compute annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / annualization_factor
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(annualization_factor) * excess_returns.mean() / excess_returns.std()


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252
) -> float:
    """Compute annualized Sortino ratio."""
    excess_returns = returns - risk_free_rate / annualization_factor
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    downside_std = downside_returns.std()
    return np.sqrt(annualization_factor) * excess_returns.mean() / downside_std


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown."""
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return abs(drawdowns.min())


def compare_strategies(
    strategies: Dict[str, SaliencyTradingStrategy],
    features: np.ndarray,
    prices: pd.Series,
    dates: pd.DatetimeIndex,
    **kwargs
) -> pd.DataFrame:
    """
    Compare multiple strategies.

    Args:
        strategies: Dictionary of strategy name to strategy object
        features: Feature array
        prices: Price series
        dates: Date index
        **kwargs: Additional arguments for run_backtest

    Returns:
        DataFrame with comparison metrics
    """
    results = {}

    for name, strategy in strategies.items():
        result = run_backtest(strategy, features, prices, dates, **kwargs)
        results[name] = {
            'Total Return': f'{result.total_return:.2%}',
            'Sharpe Ratio': f'{result.sharpe_ratio:.2f}',
            'Sortino Ratio': f'{result.sortino_ratio:.2f}',
            'Max Drawdown': f'{result.max_drawdown:.2%}',
            'Win Rate': f'{result.win_rate:.2%}',
            'Profit Factor': f'{result.profit_factor:.2f}',
            'Num Trades': result.num_trades
        }

    return pd.DataFrame(results).T
