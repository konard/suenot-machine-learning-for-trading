"""
Backtesting framework for ELECTRA-based trading strategies.

This module provides tools for backtesting trading strategies that use
ELECTRA sentiment analysis for signal generation and position sizing.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from .model import FinancialElectra, ElectraTradingSignal


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


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    if returns.std() == 0:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std())


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sortino ratio.

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    return float(np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std())


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute maximum drawdown from an equity curve.

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Maximum drawdown as a negative fraction
    """
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


class ElectraTradingStrategy:
    """
    Trading strategy that uses ELECTRA sentiment for signal generation.

    The strategy:
    1. Analyzes news/text sentiment using ELECTRA
    2. Generates trading signals based on sentiment scores
    3. Sizes positions based on confidence levels
    4. Applies risk management rules

    Args:
        model: Trained FinancialElectra model
        vocab: Vocabulary dictionary for tokenization
        signal_threshold: Minimum absolute signal to trade
        confidence_threshold: Minimum confidence to trade
        max_position_size: Maximum position size as fraction of portfolio
        stop_loss_pct: Stop-loss as percentage of entry price
        holding_period: Maximum holding period in days
        device: Device for model inference
    """

    def __init__(
        self,
        model: FinancialElectra,
        vocab: Dict[str, int],
        signal_threshold: float = 0.3,
        confidence_threshold: float = 0.6,
        max_position_size: float = 1.0,
        stop_loss_pct: float = 0.05,
        holding_period: int = 5,
        device: str = 'cpu'
    ):
        self.signal_gen = ElectraTradingSignal(model, vocab, device)
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.holding_period = holding_period

    def generate_signals(
        self,
        news_by_date: Dict[pd.Timestamp, List[str]]
    ) -> pd.DataFrame:
        """
        Generate trading signals from news data.

        Args:
            news_by_date: Dictionary mapping dates to lists of news texts

        Returns:
            DataFrame with columns: date, signal, confidence, direction, position_size
        """
        signals = []
        for date, news_items in sorted(news_by_date.items()):
            signal, confidence = self.signal_gen.generate_signal(
                news_items, threshold=self.confidence_threshold
            )

            if abs(signal) > self.signal_threshold:
                direction = 1 if signal > 0 else -1
                position_size = min(
                    abs(signal) * confidence,
                    self.max_position_size
                )

                signals.append({
                    'date': date,
                    'signal': signal,
                    'confidence': confidence,
                    'direction': direction,
                    'position_size': position_size
                })

        return pd.DataFrame(signals)


def run_backtest(
    strategy: ElectraTradingStrategy,
    price_data: pd.DataFrame,
    news_by_date: Dict[pd.Timestamp, List[str]],
    initial_capital: float = 100000.0
) -> BacktestResult:
    """
    Run a backtest of the ELECTRA trading strategy.

    Args:
        strategy: Configured ElectraTradingStrategy
        price_data: DataFrame with OHLCV price data
        news_by_date: Dictionary mapping dates to lists of news texts
        initial_capital: Starting portfolio value

    Returns:
        BacktestResult with performance metrics
    """
    signals = strategy.generate_signals(news_by_date)

    if signals.empty:
        equity = pd.Series(
            [initial_capital] * len(price_data),
            index=price_data.index
        )
        return BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            num_trades=0,
            trades=[],
            equity_curve=equity,
            daily_returns=pd.Series(0.0, index=price_data.index)
        )

    trades = []
    capital = initial_capital
    equity_values = []
    daily_returns_list = []

    position = None
    position_days = 0

    for i, (date, row) in enumerate(price_data.iterrows()):
        # Check for exit conditions
        if position is not None:
            position_days += 1
            current_price = row['Close']

            # Check stop-loss
            if position['direction'] == 1:
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

            should_exit = (
                pnl_pct <= -strategy.stop_loss_pct or
                position_days >= strategy.holding_period
            )

            if should_exit:
                pnl = pnl_pct * position['position_size'] * capital
                capital += pnl

                trade = Trade(
                    entry_date=position['entry_date'],
                    exit_date=date,
                    direction=position['direction'],
                    entry_price=position['entry_price'],
                    exit_price=current_price,
                    position_size=position['position_size'],
                    pnl=pnl,
                    pnl_pct=pnl_pct
                )
                trades.append(trade)
                position = None
                position_days = 0

        # Check for entry signals
        if position is None:
            matching_signals = signals[signals['date'] == date]
            if not matching_signals.empty:
                sig = matching_signals.iloc[0]
                position = {
                    'entry_date': date,
                    'entry_price': row['Close'],
                    'direction': int(sig['direction']),
                    'position_size': float(sig['position_size']),
                }
                position_days = 0

        equity_values.append(capital)
        if len(equity_values) > 1:
            daily_ret = (equity_values[-1] - equity_values[-2]) / equity_values[-2]
        else:
            daily_ret = 0.0
        daily_returns_list.append(daily_ret)

    equity_curve = pd.Series(equity_values, index=price_data.index)
    daily_returns = pd.Series(daily_returns_list, index=price_data.index)

    # Compute metrics
    total_return = (capital - initial_capital) / initial_capital
    sharpe = compute_sharpe_ratio(daily_returns)
    sortino = compute_sortino_ratio(daily_returns)
    max_dd = compute_max_drawdown(equity_curve)

    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0.0

    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = abs(sum(t.pnl for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=len(trades),
        trades=trades,
        equity_curve=equity_curve,
        daily_returns=daily_returns
    )


def compare_strategies(
    results: Dict[str, BacktestResult]
) -> pd.DataFrame:
    """
    Compare performance of multiple strategy variants.

    Args:
        results: Dictionary mapping strategy names to BacktestResult objects

    Returns:
        DataFrame with comparison metrics
    """
    comparison = []
    for name, result in results.items():
        comparison.append({
            'Strategy': name,
            'Total Return': f"{result.total_return:.2%}",
            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{result.sortino_ratio:.2f}",
            'Max Drawdown': f"{result.max_drawdown:.2%}",
            'Win Rate': f"{result.win_rate:.2%}",
            'Profit Factor': f"{result.profit_factor:.2f}",
            'Num Trades': result.num_trades,
        })

    return pd.DataFrame(comparison)
