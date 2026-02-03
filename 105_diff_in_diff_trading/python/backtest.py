"""Backtesting framework for DiD-based trading strategies.

Provides tools for:
- Event-driven trading strategy backtesting
- Performance metrics computation
- Risk analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Represents a single trade."""

    symbol: str
    entry_date: datetime
    entry_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    reason: str = ""


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    avg_holding_period: float

    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(\n"
            f"  total_return      = {self.total_return:.2%}\n"
            f"  annualized_return = {self.annualized_return:.2%}\n"
            f"  volatility        = {self.volatility:.2%}\n"
            f"  sharpe_ratio      = {self.sharpe_ratio:.2f}\n"
            f"  sortino_ratio     = {self.sortino_ratio:.2f}\n"
            f"  max_drawdown      = {self.max_drawdown:.2%}\n"
            f"  win_rate          = {self.win_rate:.2%}\n"
            f"  profit_factor     = {self.profit_factor:.2f}\n"
            f"  num_trades        = {self.num_trades}\n"
            f"  avg_trade_return  = {self.avg_trade_return:.2%}\n"
            f"  avg_holding_period= {self.avg_holding_period:.1f} days\n"
            f")"
        )


class TradingStrategy:
    """Base class for DiD-based trading strategies.

    Subclass this to implement custom entry/exit logic.
    """

    def __init__(
        self,
        max_position_size: float = 0.15,
        stop_loss: float = 0.10,
        take_profit: float = 0.15,
        holding_period: int = 10,
    ):
        """Initialize trading strategy.

        Args:
            max_position_size: Maximum position as fraction of portfolio
            stop_loss: Stop loss as fraction of entry price
            take_profit: Take profit as fraction of entry price
            holding_period: Maximum holding period in days
        """
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.holding_period = holding_period

    def generate_signal(
        self,
        did_estimate: float,
        p_value: float,
        threshold: float = 0.02,
        significance_level: float = 0.05,
    ) -> int:
        """Generate trading signal from DiD results.

        Args:
            did_estimate: DiD point estimate
            p_value: Statistical significance
            threshold: Minimum effect size
            significance_level: Maximum p-value for significance

        Returns:
            Signal: 1 (long), -1 (short), or 0 (no trade)
        """
        if p_value > significance_level:
            return 0

        if did_estimate > threshold:
            return 1
        elif did_estimate < -threshold:
            return -1

        return 0

    def size_position(
        self,
        did_estimate: float,
        std_error: float,
        portfolio_value: float,
    ) -> float:
        """Determine position size based on signal strength.

        Uses a Kelly-criterion inspired approach.

        Args:
            did_estimate: DiD point estimate
            std_error: Standard error of estimate
            portfolio_value: Current portfolio value

        Returns:
            Position size in currency units
        """
        if std_error <= 0:
            return 0

        # Confidence-based sizing
        t_stat = abs(did_estimate / std_error)
        confidence = min(1.0, t_stat / 2.0)  # Normalize to [0, 1]

        # Kelly-inspired sizing
        size_fraction = confidence * self.max_position_size

        return portfolio_value * size_fraction


class Backtester:
    """Backtesting engine for event-driven trading strategies.

    Args:
        initial_capital: Starting portfolio value
        transaction_cost: Cost per trade as fraction of trade value
        slippage: Slippage as fraction of price
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Trade] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

    def reset(self) -> None:
        """Reset backtester state."""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def run(
        self,
        events: List[dict],
        price_data: pd.DataFrame,
        strategy: TradingStrategy,
    ) -> PerformanceMetrics:
        """Run backtest on a list of events.

        Args:
            events: List of event dictionaries with keys:
                - date: Event date
                - treated_symbol: Symbol affected by event
                - control_symbol: Control symbol
                - did_estimate: DiD estimate
                - std_error: Standard error
                - p_value: P-value
            price_data: DataFrame with price data for all symbols
            strategy: Trading strategy to use

        Returns:
            PerformanceMetrics for the backtest
        """
        self.reset()

        # Sort events by date
        events = sorted(events, key=lambda x: x["date"])

        # Get all unique dates from price data
        all_dates = sorted(price_data["timestamp"].unique())

        for date in all_dates:
            # Update portfolio value
            self._update_portfolio_value(date, price_data)

            # Check for exits on existing positions
            self._check_exits(date, price_data, strategy)

            # Check for new events
            for event in events:
                if event["date"] == date:
                    signal = strategy.generate_signal(
                        event["did_estimate"],
                        event["p_value"],
                    )

                    if signal != 0 and event["treated_symbol"] not in self.positions:
                        size = strategy.size_position(
                            event["did_estimate"],
                            event["std_error"],
                            self.portfolio_value,
                        )

                        self._enter_position(
                            date,
                            event["treated_symbol"],
                            signal,
                            size,
                            price_data,
                        )

            # Record equity
            self.equity_curve.append((date, self.portfolio_value))

        # Close any remaining positions
        if all_dates:
            for symbol in list(self.positions.keys()):
                self._exit_position(all_dates[-1], symbol, price_data, "end_of_backtest")

        return self._compute_metrics()

    def _enter_position(
        self,
        date: datetime,
        symbol: str,
        direction: int,
        size: float,
        price_data: pd.DataFrame,
    ) -> None:
        """Enter a new position."""
        price_row = price_data[
            (price_data["symbol"] == symbol) &
            (price_data["timestamp"] == date)
        ]

        if price_row.empty:
            return

        entry_price = price_row["close"].iloc[0]

        # Apply slippage
        if direction == 1:
            entry_price *= (1 + self.slippage)
        else:
            entry_price *= (1 - self.slippage)

        # Calculate position size in units
        units = size / entry_price

        # Deduct transaction cost
        cost = size * self.transaction_cost
        self.cash -= cost

        trade = Trade(
            symbol=symbol,
            entry_date=date,
            entry_price=entry_price,
            direction=direction,
            size=units,
        )

        self.positions[symbol] = trade

    def _exit_position(
        self,
        date: datetime,
        symbol: str,
        price_data: pd.DataFrame,
        reason: str,
    ) -> None:
        """Exit an existing position."""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]

        price_row = price_data[
            (price_data["symbol"] == symbol) &
            (price_data["timestamp"] == date)
        ]

        if price_row.empty:
            return

        exit_price = price_row["close"].iloc[0]

        # Apply slippage
        if trade.direction == 1:
            exit_price *= (1 - self.slippage)
        else:
            exit_price *= (1 + self.slippage)

        # Calculate PnL
        price_change = (exit_price - trade.entry_price) / trade.entry_price
        pnl = trade.direction * price_change * trade.size * trade.entry_price

        # Deduct transaction cost
        trade_value = trade.size * exit_price
        cost = trade_value * self.transaction_cost
        pnl -= cost

        # Update trade record
        trade.exit_date = date
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.reason = reason

        # Update cash
        self.cash += trade.size * exit_price + pnl

        # Record and remove from positions
        self.trades.append(trade)
        del self.positions[symbol]

    def _check_exits(
        self,
        date: datetime,
        price_data: pd.DataFrame,
        strategy: TradingStrategy,
    ) -> None:
        """Check exit conditions for all open positions."""
        for symbol in list(self.positions.keys()):
            trade = self.positions[symbol]

            price_row = price_data[
                (price_data["symbol"] == symbol) &
                (price_data["timestamp"] == date)
            ]

            if price_row.empty:
                continue

            current_price = price_row["close"].iloc[0]
            price_change = (current_price - trade.entry_price) / trade.entry_price

            # Directional price change
            directional_change = trade.direction * price_change

            # Check stop loss
            if directional_change < -strategy.stop_loss:
                self._exit_position(date, symbol, price_data, "stop_loss")
                continue

            # Check take profit
            if directional_change > strategy.take_profit:
                self._exit_position(date, symbol, price_data, "take_profit")
                continue

            # Check holding period
            holding_days = (date - trade.entry_date).days
            if holding_days >= strategy.holding_period:
                self._exit_position(date, symbol, price_data, "holding_period")

    def _update_portfolio_value(
        self,
        date: datetime,
        price_data: pd.DataFrame,
    ) -> None:
        """Update portfolio value based on current prices."""
        position_value = 0

        for symbol, trade in self.positions.items():
            price_row = price_data[
                (price_data["symbol"] == symbol) &
                (price_data["timestamp"] == date)
            ]

            if not price_row.empty:
                current_price = price_row["close"].iloc[0]
                position_value += trade.size * current_price

        self.portfolio_value = self.cash + position_value

    def _compute_metrics(self) -> PerformanceMetrics:
        """Compute performance metrics from backtest results."""
        if not self.equity_curve:
            return self._empty_metrics()

        # Extract equity values
        dates, values = zip(*self.equity_curve)
        equity = np.array(values)
        returns = np.diff(equity) / equity[:-1]

        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0]

        # Annualized return (assume 252 trading days)
        n_days = (dates[-1] - dates[0]).days
        if n_days > 0:
            annualized_return = (1 + total_return) ** (365 / n_days) - 1
        else:
            annualized_return = 0

        # Volatility
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # Sortino ratio (downside volatility)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino = annualized_return / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        # Trade statistics
        if self.trades:
            pnls = [t.pnl for t in self.trades if t.pnl is not None]
            wins = sum(1 for p in pnls if p > 0)
            win_rate = wins / len(pnls) if pnls else 0

            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            avg_return = np.mean(pnls) / self.initial_capital if pnls else 0

            holding_periods = [
                (t.exit_date - t.entry_date).days
                for t in self.trades
                if t.exit_date is not None
            ]
            avg_holding = np.mean(holding_periods) if holding_periods else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_return = 0
            avg_holding = 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(self.trades),
            avg_trade_return=avg_return,
            avg_holding_period=avg_holding,
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no data."""
        return PerformanceMetrics(
            total_return=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            num_trades=0,
            avg_trade_return=0,
            avg_holding_period=0,
        )

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for trade in self.trades:
            records.append({
                "symbol": trade.symbol,
                "entry_date": trade.entry_date,
                "entry_price": trade.entry_price,
                "direction": "long" if trade.direction == 1 else "short",
                "size": trade.size,
                "exit_date": trade.exit_date,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "reason": trade.reason,
            })

        return pd.DataFrame(records)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()

        dates, values = zip(*self.equity_curve)
        return pd.DataFrame({
            "date": dates,
            "equity": values,
        })
