"""
Backtesting Engine for MC Dropout Trading Strategy
===================================================

This module implements a backtesting framework for evaluating
trading strategies with uncertainty estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from trading_strategy import (
    MCDropoutTradingStrategy,
    TradingSignal,
    SignalType,
    StrategyConfig
)
from mc_dropout_model import MCDropoutModel


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    symbol: str
    signal_type: SignalType
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    predicted_return: float
    uncertainty: float
    actual_return: Optional[float] = None
    pnl: Optional[float] = None

    def close(self, exit_time: pd.Timestamp, exit_price: float):
        """Close the trade."""
        self.exit_time = exit_time
        self.exit_price = exit_price

        if self.signal_type == SignalType.LONG:
            self.actual_return = (exit_price - self.entry_price) / self.entry_price
        else:
            self.actual_return = (self.entry_price - exit_price) / self.entry_price

        self.pnl = self.actual_return * self.position_size


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Uncertainty metrics
    mean_uncertainty: float = 0.0
    uncertainty_coverage_95: float = 0.0
    uncertainty_correlation: float = 0.0

    # Trade lists
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'mean_uncertainty': self.mean_uncertainty,
            'uncertainty_coverage_95': self.uncertainty_coverage_95,
            'uncertainty_correlation': self.uncertainty_correlation,
        }

    def print_summary(self):
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print("\n--- Trade Statistics ---")
        print(f"Total Trades:    {self.total_trades}")
        print(f"Winning Trades:  {self.winning_trades}")
        print(f"Losing Trades:   {self.losing_trades}")
        print(f"Win Rate:        {self.win_rate:.2%}")

        print("\n--- Performance Metrics ---")
        print(f"Total Return:      {self.total_return:.2%}")
        print(f"Annualized Return: {self.annualized_return:.2%}")
        print(f"Sharpe Ratio:      {self.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:     {self.sortino_ratio:.2f}")
        print(f"Max Drawdown:      {self.max_drawdown:.2%}")

        print("\n--- Uncertainty Metrics ---")
        print(f"Mean Uncertainty:       {self.mean_uncertainty:.4f}")
        print(f"95% Coverage:           {self.uncertainty_coverage_95:.2%}")
        print(f"Uncertainty-Error Corr: {self.uncertainty_correlation:.4f}")

        print("=" * 60)


class BacktestEngine:
    """
    Backtesting engine for MC Dropout trading strategies.
    """

    def __init__(self,
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,  # 0.1% commission
                 slippage: float = 0.0005,   # 0.05% slippage
                 holding_period: int = 1):    # Bars to hold position
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            commission: Trading commission (fraction)
            slippage: Slippage (fraction)
            holding_period: Number of bars to hold each position
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.holding_period = holding_period

    def run(self,
            strategy: MCDropoutTradingStrategy,
            features: np.ndarray,
            prices: np.ndarray,
            timestamps: Optional[pd.DatetimeIndex] = None,
            symbol: str = "BTCUSDT") -> BacktestResult:
        """
        Run backtest.

        Args:
            strategy: Trading strategy instance
            features: Feature array [n_samples, n_features]
            prices: Price array [n_samples]
            timestamps: Optional timestamps
            symbol: Trading symbol

        Returns:
            BacktestResult with performance metrics
        """
        n_samples = len(features)

        if timestamps is None:
            timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='h')

        # Initialize tracking
        capital = self.initial_capital
        equity_history = [capital]
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None
        bars_held = 0

        # Track predictions for uncertainty analysis
        predictions = []
        uncertainties = []
        actual_returns = []

        for i in range(n_samples - self.holding_period):
            current_price = prices[i]
            current_time = timestamps[i]

            # Check if we should close existing position
            if open_trade is not None:
                bars_held += 1

                if bars_held >= self.holding_period:
                    # Close position
                    exit_price = prices[i] * (1 - self.slippage if open_trade.signal_type == SignalType.LONG else 1 + self.slippage)
                    open_trade.close(current_time, exit_price)

                    # Apply commission
                    trade_pnl = open_trade.pnl - self.commission * 2  # Entry + exit
                    capital += capital * trade_pnl

                    # Track for analysis
                    actual_returns.append(open_trade.actual_return)

                    trades.append(open_trade)
                    open_trade = None
                    bars_held = 0

            # Generate signal if no open position
            if open_trade is None:
                signal = strategy.generate_signal(features[i], current_time)

                if signal.signal_type != SignalType.HOLD:
                    # Open position
                    entry_price = current_price * (1 + self.slippage if signal.signal_type == SignalType.LONG else 1 - self.slippage)

                    open_trade = Trade(
                        entry_time=current_time,
                        exit_time=None,
                        symbol=symbol,
                        signal_type=signal.signal_type,
                        entry_price=entry_price,
                        exit_price=None,
                        position_size=signal.position_size,
                        predicted_return=signal.predicted_return,
                        uncertainty=signal.uncertainty,
                    )

                    # Track predictions
                    predictions.append(signal.predicted_return)
                    uncertainties.append(signal.uncertainty)

            equity_history.append(capital)

        # Close any remaining position
        if open_trade is not None:
            exit_price = prices[-1]
            open_trade.close(timestamps[-1], exit_price)
            actual_returns.append(open_trade.actual_return)
            trades.append(open_trade)

        # Calculate metrics
        result = self._calculate_metrics(
            trades=trades,
            equity_history=equity_history,
            predictions=predictions,
            uncertainties=uncertainties,
            actual_returns=actual_returns,
            timestamps=timestamps[:len(equity_history)]
        )

        return result

    def _calculate_metrics(self,
                          trades: List[Trade],
                          equity_history: List[float],
                          predictions: List[float],
                          uncertainties: List[float],
                          actual_returns: List[float],
                          timestamps: pd.DatetimeIndex) -> BacktestResult:
        """Calculate backtest metrics."""
        result = BacktestResult()
        result.trades = trades

        # Trade statistics
        result.total_trades = len(trades)
        if result.total_trades > 0:
            result.winning_trades = sum(1 for t in trades if t.pnl and t.pnl > 0)
            result.losing_trades = sum(1 for t in trades if t.pnl and t.pnl <= 0)
            result.win_rate = result.winning_trades / result.total_trades

        # Equity curve
        equity = pd.Series(equity_history, index=timestamps)
        result.equity_curve = equity

        # Returns
        result.total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]

        # Annualized return (assuming hourly data)
        n_years = len(equity) / (24 * 365)
        if n_years > 0:
            result.annualized_return = (1 + result.total_return) ** (1 / n_years) - 1

        # Calculate returns series
        returns = equity.pct_change().dropna()

        # Sharpe ratio (annualized, assuming hourly)
        if len(returns) > 0 and returns.std() > 0:
            result.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(24 * 365)

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            result.sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(24 * 365)

        # Maximum drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        result.max_drawdown = abs(drawdown.min())

        # Uncertainty metrics
        if len(uncertainties) > 0:
            result.mean_uncertainty = np.mean(uncertainties)

            if len(actual_returns) == len(predictions):
                predictions = np.array(predictions)
                uncertainties = np.array(uncertainties)
                actual_returns = np.array(actual_returns)

                # 95% coverage: how often is actual within 1.96 * uncertainty of prediction
                lower = predictions - 1.96 * uncertainties
                upper = predictions + 1.96 * uncertainties
                within_ci = (actual_returns >= lower) & (actual_returns <= upper)
                result.uncertainty_coverage_95 = np.mean(within_ci)

                # Correlation between uncertainty and absolute error
                errors = np.abs(actual_returns - predictions)
                if np.std(uncertainties) > 0 and np.std(errors) > 0:
                    result.uncertainty_correlation = np.corrcoef(uncertainties, errors)[0, 1]

        return result


def run_backtest_example():
    """Run a simple backtest example."""
    from mc_dropout_model import create_mc_dropout_model

    print("=== MC Dropout Backtest Example ===\n")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Features (20 dimensions)
    features = np.random.randn(n_samples, 20).astype(np.float32)

    # Simulate price series with some trend
    returns = features[:, 0] * 0.001 + np.random.randn(n_samples) * 0.002
    prices = 100 * np.cumprod(1 + returns)

    # Create target (next period return)
    y = returns[1:]
    features = features[:-1]
    prices = prices[:-1]

    # Split data
    train_size = int(0.7 * len(features))
    X_train, X_test = features[:train_size], features[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    prices_test = prices[train_size:]

    # Train model
    print("Training MC Dropout model...")
    model = create_mc_dropout_model(
        input_dim=20,
        hidden_dims=[64, 32],
        dropout_rate=0.1,
        n_mc_samples=50,
        task='regression'
    )
    model.fit(X_train, y_train, epochs=30, patience=5)

    # Create strategy
    config = StrategyConfig(
        confidence_threshold=1.0,
        uncertainty_threshold=0.05,
        max_position_size=0.1,
        use_kelly=True,
        n_mc_samples=50
    )
    strategy = MCDropoutTradingStrategy(model, config)

    # Run backtest
    print("\nRunning backtest...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        holding_period=1
    )

    result = engine.run(
        strategy=strategy,
        features=X_test,
        prices=prices_test,
        symbol="SYNTH"
    )

    # Print results
    result.print_summary()

    return result


if __name__ == '__main__':
    result = run_backtest_example()
