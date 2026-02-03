"""
Backtesting Framework for MambaTS Trading Strategies

This module provides a comprehensive backtesting framework for evaluating
trading strategies based on MambaTS predictions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade."""

    entry_time: datetime
    entry_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None

    def close(self, exit_time: datetime, exit_price: float):
        """Close the trade and calculate PnL."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.pnl = (exit_price - self.entry_price) * self.direction * self.size
        self.pnl_percent = (exit_price / self.entry_price - 1) * self.direction * 100


@dataclass
class BacktestResult:
    """Results from backtesting a strategy."""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    cumulative_returns: np.ndarray = field(default_factory=lambda: np.array([]))

    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Trade statistics
    n_trades: int = 0
    n_winning_trades: int = 0
    n_losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0

    # Other
    trades: List[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        """Generate a summary string of the results."""
        return f"""
Backtest Results Summary
========================
Total Return: {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Volatility: {self.volatility:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}
Max Drawdown Duration: {self.max_drawdown_duration} periods

Trade Statistics
----------------
Total Trades: {self.n_trades}
Winning Trades: {self.n_winning_trades} ({self.win_rate:.1%})
Losing Trades: {self.n_losing_trades}
Profit Factor: {self.profit_factor:.2f}
Average Win: {self.avg_win:.2%}
Average Loss: {self.avg_loss:.2%}
"""


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def generate_signals(self, data: np.ndarray) -> np.ndarray:
        """
        Generate trading signals from data.

        Args:
            data: Input data array

        Returns:
            Array of signals: 1 for long, -1 for short, 0 for no position
        """
        pass


class MambaTSStrategy(Strategy):
    """
    Trading strategy based on MambaTS predictions.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.02,
        lookback: int = 168,
        device: str = "cpu",
    ):
        """
        Args:
            model: Trained MambaTS model
            threshold: Prediction threshold for generating signals
            lookback: Lookback window size
            device: Device for inference
        """
        self.model = model
        self.threshold = threshold
        self.lookback = lookback
        self.device = device

    def generate_signals(self, data: np.ndarray) -> np.ndarray:
        """
        Generate signals based on model predictions.
        """
        import torch

        self.model.eval()
        signals = np.zeros(len(data))

        with torch.no_grad():
            for i in range(self.lookback, len(data)):
                # Prepare input
                x = data[i - self.lookback : i]
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Get prediction
                pred = self.model(x)

                # For regression: interpret as expected return
                if len(pred.shape) == 3:
                    # Multi-step: use first step prediction
                    expected_return = pred[0, 0, 0].item()
                else:
                    expected_return = pred[0, 0].item()

                # Generate signal based on threshold
                if expected_return > self.threshold:
                    signals[i] = 1  # Long
                elif expected_return < -self.threshold:
                    signals[i] = -1  # Short
                else:
                    signals[i] = 0  # No position

        return signals


class SimpleThresholdStrategy(Strategy):
    """
    Simple threshold-based strategy using direct predictions.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        long_threshold: float = 0.02,
        short_threshold: float = -0.02,
    ):
        """
        Args:
            predictions: Array of model predictions
            long_threshold: Threshold for long signals
            short_threshold: Threshold for short signals
        """
        self.predictions = predictions
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def generate_signals(self, data: np.ndarray) -> np.ndarray:
        """Generate signals from pre-computed predictions."""
        signals = np.zeros(len(self.predictions))
        signals[self.predictions > self.long_threshold] = 1
        signals[self.predictions < self.short_threshold] = -1
        return signals


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 1.0,
        risk_free_rate: float = 0.02,
    ):
        """
        Args:
            strategy: Trading strategy to backtest
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
            slippage: Slippage per trade (as fraction)
            position_size: Fraction of capital to use per trade
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        prices: np.ndarray,
        data: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            prices: Array of prices (for PnL calculation)
            data: Feature data for strategy (if different from prices)
            timestamps: Optional timestamps for trades

        Returns:
            BacktestResult with all metrics
        """
        if data is None:
            data = prices.reshape(-1, 1)

        if timestamps is None:
            timestamps = np.arange(len(prices))

        # Generate signals
        signals = self.strategy.generate_signals(data)

        # Initialize tracking variables
        capital = self.initial_capital
        position = 0
        position_price = 0
        trades: List[Trade] = []
        equity_curve = [capital]
        current_trade: Optional[Trade] = None

        # Process each time step
        for i in range(1, len(signals)):
            price = prices[i]
            signal = signals[i]
            prev_signal = signals[i - 1]

            # Check for position changes
            if signal != prev_signal:
                # Close existing position
                if position != 0 and current_trade is not None:
                    # Apply slippage
                    exit_price = price * (1 - self.slippage * position)
                    # Apply commission
                    commission_cost = abs(position) * exit_price * self.commission

                    current_trade.close(timestamps[i], exit_price)
                    capital += current_trade.pnl - commission_cost
                    trades.append(current_trade)
                    current_trade = None

                # Open new position
                if signal != 0:
                    # Calculate position size
                    trade_capital = capital * self.position_size
                    # Apply slippage
                    entry_price = price * (1 + self.slippage * signal)
                    # Apply commission
                    commission_cost = trade_capital * self.commission

                    position = signal * trade_capital / entry_price
                    position_price = entry_price
                    capital -= commission_cost

                    current_trade = Trade(
                        entry_time=timestamps[i],
                        entry_price=entry_price,
                        direction=signal,
                        size=abs(position),
                    )
                else:
                    position = 0

            # Update equity
            if position != 0:
                unrealized_pnl = (price - position_price) * position
                equity_curve.append(capital + unrealized_pnl)
            else:
                equity_curve.append(capital)

        # Close any remaining position
        if position != 0 and current_trade is not None:
            exit_price = prices[-1] * (1 - self.slippage * np.sign(position))
            current_trade.close(timestamps[-1], exit_price)
            capital += current_trade.pnl - abs(position) * exit_price * self.commission
            trades.append(current_trade)

        equity_curve = np.array(equity_curve)

        # Calculate metrics
        result = self._calculate_metrics(equity_curve, trades)
        result.equity_curve = equity_curve
        result.trades = trades

        return result

    def _calculate_metrics(
        self, equity_curve: np.ndarray, trades: List[Trade]
    ) -> BacktestResult:
        """Calculate all performance metrics."""
        result = BacktestResult()

        # Returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        result.cumulative_returns = (1 + returns).cumprod() - 1
        result.total_return = (equity_curve[-1] / equity_curve[0]) - 1

        # Annualized metrics (assuming daily data)
        n_periods = len(returns)
        periods_per_year = 252  # Adjust based on data frequency
        result.annualized_return = (1 + result.total_return) ** (
            periods_per_year / n_periods
        ) - 1
        result.volatility = returns.std() * np.sqrt(periods_per_year)

        # Sharpe ratio
        if result.volatility > 0:
            excess_return = result.annualized_return - self.risk_free_rate
            result.sharpe_ratio = excess_return / result.volatility
        else:
            result.sharpe_ratio = 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(periods_per_year)
            if downside_std > 0:
                excess_return = result.annualized_return - self.risk_free_rate
                result.sortino_ratio = excess_return / downside_std
            else:
                result.sortino_ratio = 0
        else:
            result.sortino_ratio = float("inf")

        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        result.max_drawdown = abs(drawdown.min())

        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_starts = np.diff(in_drawdown.astype(int)) == 1
        drawdown_ends = np.diff(in_drawdown.astype(int)) == -1
        if drawdown_starts.sum() > 0:
            durations = []
            start_idx = 0
            for i, (start, end) in enumerate(
                zip(drawdown_starts, drawdown_ends)
            ):
                if start:
                    start_idx = i
                if end:
                    durations.append(i - start_idx)
            result.max_drawdown_duration = max(durations) if durations else 0
        else:
            result.max_drawdown_duration = 0

        # Trade statistics
        result.n_trades = len(trades)
        if result.n_trades > 0:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]

            result.n_winning_trades = len(winning_trades)
            result.n_losing_trades = len(losing_trades)
            result.win_rate = result.n_winning_trades / result.n_trades

            total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
            total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0

            if total_losses > 0:
                result.profit_factor = total_wins / total_losses
            else:
                result.profit_factor = float("inf") if total_wins > 0 else 0

            if winning_trades:
                result.avg_win = np.mean([t.pnl_percent for t in winning_trades])
            if losing_trades:
                result.avg_loss = np.mean([t.pnl_percent for t in losing_trades])

        return result


class WalkForwardOptimizer:
    """
    Walk-forward optimization for strategy parameters.
    """

    def __init__(
        self,
        strategy_factory: Callable,
        param_grid: Dict[str, List],
        train_size: int,
        test_size: int,
        step_size: int,
    ):
        """
        Args:
            strategy_factory: Function that creates a strategy given params
            param_grid: Dictionary of parameter names to lists of values
            train_size: Size of training window
            test_size: Size of testing window
            step_size: Step size for walk-forward
        """
        self.strategy_factory = strategy_factory
        self.param_grid = param_grid
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def optimize(
        self,
        prices: np.ndarray,
        data: np.ndarray,
        metric: str = "sharpe_ratio",
    ) -> List[BacktestResult]:
        """
        Run walk-forward optimization.

        Returns:
            List of test period results with optimized parameters
        """
        from itertools import product

        results = []
        n = len(prices)

        for start in range(0, n - self.train_size - self.test_size, self.step_size):
            train_end = start + self.train_size
            test_end = train_end + self.test_size

            train_prices = prices[start:train_end]
            train_data = data[start:train_end]
            test_prices = prices[train_end:test_end]
            test_data = data[train_end:test_end]

            # Find best parameters on training data
            best_metric = -float("inf")
            best_params = None

            param_names = list(self.param_grid.keys())
            param_values = list(self.param_grid.values())

            for param_combo in product(*param_values):
                params = dict(zip(param_names, param_combo))
                strategy = self.strategy_factory(**params)
                backtester = Backtester(strategy)
                result = backtester.run(train_prices, train_data)

                metric_value = getattr(result, metric)
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params

            # Evaluate on test data with best parameters
            best_strategy = self.strategy_factory(**best_params)
            backtester = Backtester(best_strategy)
            test_result = backtester.run(test_prices, test_data)
            results.append(test_result)

        return results


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)

    # Generate synthetic price data
    n_samples = 1000
    returns = np.random.normal(0.0001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate synthetic predictions (with some skill)
    noise = np.random.normal(0, 0.01, n_samples)
    predictions = returns + noise

    # Create strategy
    strategy = SimpleThresholdStrategy(predictions, long_threshold=0.005, short_threshold=-0.005)

    # Run backtest
    backtester = Backtester(
        strategy=strategy,
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
    )

    result = backtester.run(prices)
    print(result.summary())
