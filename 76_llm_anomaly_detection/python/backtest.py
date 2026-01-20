"""
Backtesting framework for anomaly-based trading strategies.

This module provides tools to evaluate the performance of trading
strategies based on anomaly detection signals.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import logging

import numpy as np
import pandas as pd

try:
    from .detector import BaseAnomalyDetector, AnomalyResult
    from .signals import AnomalySignalGenerator, TradingSignal, SignalType
except ImportError:
    from detector import BaseAnomalyDetector, AnomalyResult
    from signals import AnomalySignalGenerator, TradingSignal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    entry_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def close(self, exit_time: pd.Timestamp, exit_price: float) -> None:
        """Close the trade."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.pnl = (exit_price - self.entry_price) * self.direction * self.size
        self.pnl_pct = (exit_price / self.entry_price - 1) * self.direction * 100


@dataclass
class BacktestResult:
    """Results of a backtest."""
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series
    anomaly_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "anomaly_stats": self.anomaly_stats,
        }

    def print_summary(self) -> None:
        """Print a summary of the backtest results."""
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Total Return: ${self.total_return:,.2f} ({self.total_return_pct:.2f}%)")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {self.sortino_ratio:.2f}")
        print(f"Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)")
        print(f"\nNumber of Trades: {self.num_trades}")
        print(f"Win Rate: {self.win_rate:.1f}%")
        print(f"Average Win: ${self.avg_win:,.2f}")
        print(f"Average Loss: ${self.avg_loss:,.2f}")
        print(f"Profit Factor: {self.profit_factor:.2f}")
        print("\nAnomaly Statistics:")
        for key, value in self.anomaly_stats.items():
            print(f"  {key}: {value}")
        print("=" * 50 + "\n")


class AnomalyBacktester:
    """
    Backtest anomaly-based trading strategies.

    Supports:
    - Walk-forward testing
    - Position sizing
    - Transaction costs
    - Multiple exit strategies
    """

    def __init__(
        self,
        detector: BaseAnomalyDetector,
        signal_generator: Optional[AnomalySignalGenerator] = None,
        initial_capital: float = 100000.0,
        position_size: float = 0.1,  # Fraction of capital per trade
        max_positions: int = 5,
        transaction_cost: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
    ):
        """
        Initialize backtester.

        Args:
            detector: Anomaly detector to use
            signal_generator: Signal generator (default: contrarian)
            initial_capital: Starting capital
            position_size: Fraction of capital per trade
            max_positions: Maximum concurrent positions
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
        """
        self.detector = detector
        self.signal_generator = signal_generator or AnomalySignalGenerator(
            strategy="contrarian"
        )
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        # State
        self._trades: List[Trade] = []
        self._open_trades: List[Trade] = []
        self._capital = initial_capital
        self._equity_history: List[float] = []

    def _apply_transaction_costs(self, price: float, direction: int) -> float:
        """Apply transaction costs and slippage."""
        cost_factor = 1 + self.transaction_cost + self.slippage
        if direction > 0:  # Buying
            return price * cost_factor
        else:  # Selling
            return price / cost_factor

    def _open_position(
        self,
        timestamp: pd.Timestamp,
        price: float,
        signal: TradingSignal,
    ) -> Optional[Trade]:
        """Open a new position based on signal."""
        if len(self._open_trades) >= self.max_positions:
            return None

        # Determine direction
        if signal.signal_type == SignalType.BUY:
            direction = 1
        elif signal.signal_type == SignalType.SELL:
            direction = -1
        else:
            return None

        # Calculate position size
        size_fraction = signal.suggested_position_size or self.position_size
        position_value = self._capital * size_fraction

        # Apply transaction costs
        adjusted_price = self._apply_transaction_costs(price, direction)
        size = position_value / adjusted_price

        trade = Trade(
            entry_time=timestamp,
            entry_price=adjusted_price,
            direction=direction,
            size=size,
            reason=signal.reason,
        )

        self._open_trades.append(trade)
        self._capital -= position_value

        logger.debug(f"Opened {'long' if direction > 0 else 'short'} at {adjusted_price:.2f}")
        return trade

    def _close_position(
        self,
        trade: Trade,
        timestamp: pd.Timestamp,
        price: float,
    ) -> None:
        """Close an open position."""
        adjusted_price = self._apply_transaction_costs(price, -trade.direction)
        trade.close(timestamp, adjusted_price)

        # Return capital + PnL
        position_value = trade.entry_price * trade.size
        self._capital += position_value + trade.pnl

        self._trades.append(trade)
        self._open_trades.remove(trade)

        logger.debug(f"Closed trade, PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")

    def _check_exit_conditions(
        self,
        trade: Trade,
        current_price: float,
        signal: TradingSignal,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
    ) -> bool:
        """Check if trade should be closed."""
        # Calculate unrealized PnL
        unrealized_pnl_pct = (current_price / trade.entry_price - 1) * trade.direction

        # Stop loss
        if unrealized_pnl_pct <= -stop_loss:
            return True

        # Take profit
        if unrealized_pnl_pct >= take_profit:
            return True

        # Exit signal
        if signal.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]:
            if trade.direction > 0 and signal.signal_type == SignalType.EXIT_LONG:
                return True
            if trade.direction < 0 and signal.signal_type == SignalType.EXIT_SHORT:
                return True

        # Opposite signal
        if signal.signal_type == SignalType.BUY and trade.direction < 0:
            return True
        if signal.signal_type == SignalType.SELL and trade.direction > 0:
            return True

        return False

    def _calculate_equity(self, current_prices: Dict[int, float]) -> float:
        """Calculate current equity (capital + open positions)."""
        equity = self._capital

        for i, trade in enumerate(self._open_trades):
            current_price = current_prices.get(i, trade.entry_price)
            unrealized_pnl = (current_price - trade.entry_price) * trade.direction * trade.size
            equity += trade.entry_price * trade.size + unrealized_pnl

        return equity

    def run(
        self,
        data: pd.DataFrame,
        train_period: int = 100,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
        walk_forward: bool = True,
        retrain_period: int = 50,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: OHLCV DataFrame with features
            train_period: Initial training period
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            walk_forward: Use walk-forward testing
            retrain_period: Periods between retraining

        Returns:
            BacktestResult with performance metrics
        """
        if len(data) < train_period + 10:
            raise ValueError("Not enough data for backtest")

        # Reset state
        self._trades = []
        self._open_trades = []
        self._capital = self.initial_capital
        self._equity_history = []

        # Initial training
        train_data = data.iloc[:train_period]
        self.detector.fit(train_data)

        # Track anomalies
        anomaly_count = 0
        anomaly_types: Dict[str, int] = {}

        # Run through data
        for i in range(train_period, len(data)):
            row = data.iloc[i]
            timestamp = row.get("timestamp", pd.Timestamp.now())
            current_price = row.get("close", 0)

            # Retrain if walk-forward
            if walk_forward and i > train_period and (i - train_period) % retrain_period == 0:
                retrain_data = data.iloc[max(0, i-train_period):i]
                self.detector.fit(retrain_data)
                logger.debug(f"Retrained detector at index {i}")

            # Detect anomaly
            anomaly_result = self.detector.detect_single(row)

            if anomaly_result.is_anomaly:
                anomaly_count += 1
                at = anomaly_result.anomaly_type.value
                anomaly_types[at] = anomaly_types.get(at, 0) + 1

            # Generate signal
            signal = self.signal_generator.generate_signal(anomaly_result, row)

            # Check exit conditions for open trades
            for trade in self._open_trades.copy():
                if self._check_exit_conditions(trade, current_price, signal, stop_loss, take_profit):
                    self._close_position(trade, timestamp, current_price)

            # Open new positions
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                self._open_position(timestamp, current_price, signal)

            # Record equity
            self._equity_history.append(self._calculate_equity({0: current_price}))

        # Close remaining positions
        if self._open_trades:
            last_row = data.iloc[-1]
            last_timestamp = last_row.get("timestamp", pd.Timestamp.now())
            last_price = last_row.get("close", 0)
            for trade in self._open_trades.copy():
                self._close_position(trade, last_timestamp, last_price)

        # Calculate metrics
        return self._calculate_metrics(anomaly_count, anomaly_types)

    def _calculate_metrics(
        self,
        anomaly_count: int,
        anomaly_types: Dict[str, int],
    ) -> BacktestResult:
        """Calculate backtest performance metrics."""
        # Equity curve
        equity_curve = pd.Series(self._equity_history)

        # Returns
        if len(equity_curve) > 1:
            daily_returns = equity_curve.pct_change().dropna()
        else:
            daily_returns = pd.Series([0.0])

        # Total return
        total_return = equity_curve.iloc[-1] - self.initial_capital if len(equity_curve) > 0 else 0
        total_return_pct = (total_return / self.initial_capital) * 100

        # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 1 and negative_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * daily_returns.mean() / negative_returns.std()
        else:
            sortino_ratio = sharpe_ratio

        # Maximum drawdown
        if len(equity_curve) > 0:
            peak = equity_curve.expanding().max()
            drawdown = equity_curve - peak
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / peak.max()) * 100 if peak.max() > 0 else 0
        else:
            max_drawdown = 0
            max_drawdown_pct = 0

        # Trade statistics
        num_trades = len(self._trades)
        winning_trades = [t for t in self._trades if t.pnl > 0]
        losing_trades = [t for t in self._trades if t.pnl <= 0]

        win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BacktestResult(
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self._trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            anomaly_stats={
                "total_anomalies": anomaly_count,
                "anomaly_types": anomaly_types,
            },
        )


class WalkForwardOptimizer:
    """
    Walk-forward optimization for anomaly detection parameters.

    Finds optimal parameters while avoiding overfitting through
    out-of-sample validation.
    """

    def __init__(
        self,
        detector_class: type,
        signal_generator_class: type = AnomalySignalGenerator,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ):
        """
        Initialize optimizer.

        Args:
            detector_class: Anomaly detector class to optimize
            signal_generator_class: Signal generator class
            param_grid: Parameter grid to search
        """
        self.detector_class = detector_class
        self.signal_generator_class = signal_generator_class
        self.param_grid = param_grid or {
            "z_threshold": [2.0, 2.5, 3.0, 3.5],
            "contamination": [0.01, 0.05, 0.10],
        }

    def optimize(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.6,
        validation_ratio: float = 0.2,
        metric: str = "sharpe_ratio",
    ) -> Dict[str, Any]:
        """
        Find optimal parameters.

        Args:
            data: Full dataset
            train_ratio: Fraction for training
            validation_ratio: Fraction for validation
            metric: Metric to optimize

        Returns:
            Dictionary with best parameters and results
        """
        train_end = int(len(data) * train_ratio)
        val_end = int(len(data) * (train_ratio + validation_ratio))

        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]

        best_params = None
        best_score = float("-inf")
        results = []

        # Grid search
        from itertools import product

        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            try:
                # Create detector with parameters
                detector = self.detector_class(**params)
                signal_gen = self.signal_generator_class()

                # Run backtest on validation set
                backtester = AnomalyBacktester(detector, signal_gen)
                result = backtester.run(val_data, train_period=min(50, len(val_data)//3))

                score = getattr(result, metric, 0)
                results.append({"params": params, "score": score})

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                continue

        # Test with best parameters
        if best_params:
            detector = self.detector_class(**best_params)
            signal_gen = self.signal_generator_class()
            backtester = AnomalyBacktester(detector, signal_gen)
            test_result = backtester.run(test_data, train_period=min(50, len(test_data)//3))
        else:
            test_result = None

        return {
            "best_params": best_params,
            "best_validation_score": best_score,
            "test_result": test_result.to_dict() if test_result else None,
            "all_results": results,
        }


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data
    from detector import StatisticalAnomalyDetector

    print("Loading sample data...")
    data = load_sample_data(source="bybit")

    if not data.empty and len(data) > 150:
        print(f"Loaded {len(data)} rows")

        # Create detector and backtester
        detector = StatisticalAnomalyDetector(z_threshold=2.5)
        signal_gen = AnomalySignalGenerator(strategy="contrarian")
        backtester = AnomalyBacktester(
            detector,
            signal_gen,
            initial_capital=100000,
            position_size=0.1,
        )

        print("\nRunning backtest...")
        result = backtester.run(
            data,
            train_period=100,
            stop_loss=0.05,
            take_profit=0.10,
        )

        result.print_summary()

    else:
        print("Not enough data for backtest")
