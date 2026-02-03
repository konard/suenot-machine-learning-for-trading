"""
Backtesting framework for evaluating Concept Bottleneck Trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .model import ConceptBottleneckTrader, TradingSignal
from .concepts import TradingConcepts


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    direction: str = "long"  # "long" or "short"
    size: float = 1.0
    pnl: float = 0.0
    concepts_at_entry: Optional[Dict] = None


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    trades: List[Trade]
    equity_curve: pd.Series
    concept_accuracy: Dict[str, float] = field(default_factory=dict)


class Backtester:
    """
    Backtesting engine for CBM trading strategies.

    Supports:
    - Long and short positions
    - Transaction costs
    - Position sizing
    - Performance metrics calculation
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size: float = 0.1,  # Fraction of capital per trade
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def run(
        self,
        model: ConceptBottleneckTrader,
        df: pd.DataFrame,
        warmup_period: int = 50,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            model: Trained ConceptBottleneckTrader model
            df: DataFrame with OHLCV data
            warmup_period: Number of periods to skip for model initialization

        Returns:
            BacktestResult with performance metrics
        """
        capital = self.initial_capital
        position = 0  # 1 for long, -1 for short, 0 for flat
        entry_price = 0.0
        entry_time = None
        entry_concepts = None

        trades: List[Trade] = []
        equity_curve = [capital]
        timestamps = [df['timestamp'].iloc[warmup_period]]

        for i in range(warmup_period, len(df) - 1):
            window_df = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i + 1]

            try:
                prediction = model.predict_with_explanation(window_df)
                signal = prediction.signal
                concepts = prediction.concepts
            except (ValueError, IndexError):
                signal = TradingSignal.HOLD
                concepts = None

            # Process signal
            if position == 0:  # No position
                if signal == TradingSignal.BUY:
                    position = 1
                    entry_price = current_price * (1 + self.slippage)
                    entry_time = df['timestamp'].iloc[i]
                    entry_concepts = concepts.to_dict() if concepts else None
                    cost = capital * self.position_size * self.transaction_cost
                    capital -= cost

                elif signal == TradingSignal.SELL:
                    position = -1
                    entry_price = current_price * (1 - self.slippage)
                    entry_time = df['timestamp'].iloc[i]
                    entry_concepts = concepts.to_dict() if concepts else None
                    cost = capital * self.position_size * self.transaction_cost
                    capital -= cost

            elif position == 1:  # Long position
                if signal == TradingSignal.SELL or signal == TradingSignal.HOLD:
                    exit_price = current_price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) / entry_price
                    trade_pnl = capital * self.position_size * pnl
                    capital += trade_pnl
                    cost = capital * self.position_size * self.transaction_cost
                    capital -= cost

                    trades.append(Trade(
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=df['timestamp'].iloc[i],
                        exit_price=exit_price,
                        direction="long",
                        size=self.position_size,
                        pnl=pnl,
                        concepts_at_entry=entry_concepts,
                    ))
                    position = 0

            elif position == -1:  # Short position
                if signal == TradingSignal.BUY or signal == TradingSignal.HOLD:
                    exit_price = current_price * (1 + self.slippage)
                    pnl = (entry_price - exit_price) / entry_price
                    trade_pnl = capital * self.position_size * pnl
                    capital += trade_pnl
                    cost = capital * self.position_size * self.transaction_cost
                    capital -= cost

                    trades.append(Trade(
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=df['timestamp'].iloc[i],
                        exit_price=exit_price,
                        direction="short",
                        size=self.position_size,
                        pnl=pnl,
                        concepts_at_entry=entry_concepts,
                    ))
                    position = 0

            # Update equity curve
            if position == 1:
                unrealized_pnl = (next_price - entry_price) / entry_price
                equity = capital + capital * self.position_size * unrealized_pnl
            elif position == -1:
                unrealized_pnl = (entry_price - next_price) / entry_price
                equity = capital + capital * self.position_size * unrealized_pnl
            else:
                equity = capital

            equity_curve.append(equity)
            timestamps.append(df['timestamp'].iloc[i + 1])

        # Close any remaining position
        if position != 0:
            final_price = df['close'].iloc[-1]
            if position == 1:
                pnl = (final_price - entry_price) / entry_price
            else:
                pnl = (entry_price - final_price) / entry_price

            trade_pnl = capital * self.position_size * pnl
            capital += trade_pnl

            trades.append(Trade(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=df['timestamp'].iloc[-1],
                exit_price=final_price,
                direction="long" if position == 1 else "short",
                size=self.position_size,
                pnl=pnl,
                concepts_at_entry=entry_concepts,
            ))

        equity_series = pd.Series(equity_curve, index=timestamps)

        return BacktestResult(
            total_return=self._calculate_total_return(equity_curve),
            sharpe_ratio=self._calculate_sharpe_ratio(equity_curve),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            win_rate=self._calculate_win_rate(trades),
            profit_factor=self._calculate_profit_factor(trades),
            num_trades=len(trades),
            trades=trades,
            equity_curve=equity_series,
        )

    def _calculate_total_return(self, equity_curve: List[float]) -> float:
        """Calculate total return percentage."""
        if not equity_curve:
            return 0.0
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    def _calculate_sharpe_ratio(
        self,
        equity_curve: List[float],
        risk_free_rate: float = 0.02,
        periods_per_year: float = 252 * 24,  # Hourly data
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(equity_curve) < 2:
            return 0.0

        returns = np.diff(equity_curve) / equity_curve[:-1]
        excess_returns = returns - risk_free_rate / periods_per_year

        if np.std(returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(returns)
        return sharpe * np.sqrt(periods_per_year)

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        """Calculate win rate."""
        if not trades:
            return 0.0
        winning = sum(1 for t in trades if t.pnl > 0)
        return winning / len(trades)

    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not trades:
            return 0.0

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss


def analyze_concept_performance(result: BacktestResult) -> Dict[str, Dict]:
    """
    Analyze how different concepts correlated with trade outcomes.

    Args:
        result: BacktestResult from backtesting

    Returns:
        Dictionary with concept analysis
    """
    analysis = {}

    concept_names = [
        'trend_direction', 'trend_strength', 'volatility_regime',
        'momentum_signal', 'volume_profile', 'rsi'
    ]

    for concept in concept_names:
        winning_values = []
        losing_values = []

        for trade in result.trades:
            if trade.concepts_at_entry and concept in trade.concepts_at_entry:
                value = trade.concepts_at_entry[concept]
                if trade.pnl > 0:
                    winning_values.append(value)
                else:
                    losing_values.append(value)

        analysis[concept] = {
            'winning_trades': winning_values,
            'losing_trades': losing_values,
        }

    return analysis


def print_backtest_report(result: BacktestResult):
    """Print formatted backtest report."""
    print("=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Return:   {result.total_return * 100:.2f}%")
    print(f"Sharpe Ratio:   {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:   {result.max_drawdown * 100:.2f}%")
    print(f"Win Rate:       {result.win_rate * 100:.2f}%")
    print(f"Profit Factor:  {result.profit_factor:.2f}")
    print(f"Number of Trades: {result.num_trades}")
    print("=" * 50)

    if result.trades:
        print("\nRecent Trades:")
        for trade in result.trades[-5:]:
            direction = "LONG" if trade.direction == "long" else "SHORT"
            pnl_pct = trade.pnl * 100
            print(f"  {direction}: {trade.entry_time} -> {trade.exit_time} | PnL: {pnl_pct:.2f}%")
