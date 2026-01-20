"""
Backtesting framework for regime-based trading strategies.

This module provides tools to evaluate regime classification
strategies on historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .classifier import MarketRegime, RegimeResult
from .signals import TradingSignal, SignalType, RegimeSignalGenerator


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float  # Positive = long, negative = short
    regime_at_entry: MarketRegime
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    trades: List[Trade]
    equity_curve: pd.Series
    regime_history: pd.Series
    daily_returns: pd.Series

    # Additional metrics
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0


class RegimeBacktester:
    """
    Backtest regime-based trading strategies.

    Simulates trading based on regime classifications and
    generates performance metrics.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,   # 0.05% slippage
        risk_free_rate: float = 0.02  # 2% annual
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        prices: pd.Series,
        regime_results: List[RegimeResult],
        signal_generator: RegimeSignalGenerator
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            prices: Price series (close prices)
            regime_results: List of regime classifications
            signal_generator: Signal generator to use

        Returns:
            BacktestResult with performance metrics
        """
        if len(prices) != len(regime_results):
            raise ValueError("Prices and regime_results must have same length")

        # Generate signals
        signals = signal_generator.generate_signals_series(regime_results)

        # Initialize tracking
        capital = self.initial_capital
        position = 0.0  # Current position in units
        position_value = 0.0
        entry_price = 0.0
        entry_date = None
        current_regime = None

        trades: List[Trade] = []
        equity_curve = []
        regime_history = []
        daily_returns = []

        prev_equity = capital

        for i, (date, price) in enumerate(prices.items()):
            signal = signals[i]
            regime = regime_results[i].regime
            regime_history.append(regime.value)

            # Calculate current equity
            if position != 0:
                current_equity = capital + position * price
            else:
                current_equity = capital

            equity_curve.append(current_equity)

            # Calculate daily return
            if prev_equity > 0:
                daily_ret = (current_equity - prev_equity) / prev_equity
            else:
                daily_ret = 0.0
            daily_returns.append(daily_ret)
            prev_equity = current_equity

            # Check for position change
            target_position = signal.position_size

            # Determine if we need to trade
            should_trade = False

            if position == 0 and abs(target_position) > 0.1:
                # Open new position
                should_trade = True
            elif position != 0 and abs(target_position) < 0.1:
                # Close position
                should_trade = True
            elif position > 0 and target_position < -0.1:
                # Flip from long to short
                should_trade = True
            elif position < 0 and target_position > 0.1:
                # Flip from short to long
                should_trade = True

            if should_trade:
                # Close existing position if any
                if position != 0:
                    exit_price = price * (1 - self.slippage if position > 0 else 1 + self.slippage)
                    pnl = position * (exit_price - entry_price)
                    pnl -= abs(position * exit_price) * self.commission

                    trade = Trade(
                        entry_date=entry_date,
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=position,
                        regime_at_entry=current_regime,
                        pnl=pnl,
                        pnl_pct=pnl / (abs(position) * entry_price) if entry_price > 0 else 0
                    )
                    trades.append(trade)

                    capital += position * exit_price
                    capital -= abs(position * exit_price) * self.commission
                    position = 0.0

                # Open new position if target is non-zero
                if abs(target_position) > 0.1:
                    # Calculate position size
                    available_capital = capital * abs(target_position)
                    entry_price = price * (1 + self.slippage if target_position > 0 else 1 - self.slippage)

                    position = available_capital / entry_price
                    if target_position < 0:
                        position = -position

                    # Deduct commission
                    capital -= abs(position * entry_price) * self.commission
                    entry_date = date
                    current_regime = regime

        # Close any remaining position at the end
        if position != 0:
            final_price = prices.iloc[-1]
            exit_price = final_price * (1 - self.slippage if position > 0 else 1 + self.slippage)
            pnl = position * (exit_price - entry_price)

            trade = Trade(
                entry_date=entry_date,
                exit_date=prices.index[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position,
                regime_at_entry=current_regime,
                pnl=pnl,
                pnl_pct=pnl / (abs(position) * entry_price) if entry_price > 0 else 0
            )
            trades.append(trade)
            capital += position * exit_price

        # Convert to pandas series
        equity_series = pd.Series(equity_curve, index=prices.index)
        regime_series = pd.Series(regime_history, index=prices.index)
        returns_series = pd.Series(daily_returns, index=prices.index)

        # Calculate metrics
        result = self._calculate_metrics(
            equity_series, returns_series, trades, regime_series
        )

        return result

    def _calculate_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        trades: List[Trade],
        regimes: pd.Series
    ) -> BacktestResult:
        """Calculate performance metrics."""
        # Total and annualized return
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital

        num_days = len(equity)
        annualized_return = (1 + total_return) ** (252 / num_days) - 1 if num_days > 0 else 0

        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # Max drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # Win rate
        winning_trades = [t for t in trades if t.pnl is not None and t.pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0

        # Calmar ratio
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = np.sqrt(252) * returns.mean() / downside_std if downside_std > 0 else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl is not None and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl is not None and t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade metrics
        avg_trade_pnl = np.mean([t.pnl for t in trades if t.pnl is not None]) if trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in trades if t.pnl is not None and t.pnl < 0]
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_trades=len(trades),
            trades=trades,
            equity_curve=equity,
            regime_history=regimes,
            daily_returns=returns,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss
        )


class WalkForwardOptimizer:
    """
    Walk-forward optimization for regime classification parameters.

    Tests parameter robustness by optimizing on training windows
    and validating on out-of-sample periods.
    """

    def __init__(
        self,
        train_periods: int = 252,  # 1 year training
        test_periods: int = 63,     # 3 months testing
        step_size: int = 21         # 1 month step
    ):
        """
        Initialize optimizer.

        Args:
            train_periods: Number of periods for training
            test_periods: Number of periods for testing
            step_size: Step size between windows
        """
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.step_size = step_size

    def run(
        self,
        prices: pd.Series,
        classifier_factory: Callable,
        param_grid: Dict[str, List],
        backtester: RegimeBacktester
    ) -> Dict:
        """
        Run walk-forward optimization.

        Args:
            prices: Full price series
            classifier_factory: Function to create classifier with params
            param_grid: Parameter grid to search
            backtester: Backtester to use

        Returns:
            Dictionary with optimization results
        """
        results = []

        # Generate parameter combinations
        param_combinations = self._generate_combinations(param_grid)

        # Walk through data
        total_periods = len(prices)
        start_idx = 0

        while start_idx + self.train_periods + self.test_periods <= total_periods:
            train_end = start_idx + self.train_periods
            test_end = train_end + self.test_periods

            train_prices = prices.iloc[start_idx:train_end]
            test_prices = prices.iloc[train_end:test_end]

            # Find best params on training data
            best_params = None
            best_sharpe = -np.inf

            for params in param_combinations:
                try:
                    classifier = classifier_factory(**params)
                    classifier.fit(train_prices.to_frame(name='close'))

                    # Classify training data
                    train_results = []
                    for i in range(len(train_prices)):
                        window = train_prices.iloc[max(0, i-20):i+1]
                        if len(window) > 5:
                            result = classifier.classify(window.to_frame(name='close'))
                            train_results.append(result)

                    if train_results:
                        signal_gen = RegimeSignalGenerator()
                        backtest_result = backtester.run(
                            train_prices.iloc[-len(train_results):],
                            train_results,
                            signal_gen
                        )

                        if backtest_result.sharpe_ratio > best_sharpe:
                            best_sharpe = backtest_result.sharpe_ratio
                            best_params = params
                except Exception:
                    continue

            # Test best params on out-of-sample data
            if best_params is not None:
                try:
                    classifier = classifier_factory(**best_params)
                    classifier.fit(train_prices.to_frame(name='close'))

                    test_results = []
                    for i in range(len(test_prices)):
                        window = test_prices.iloc[max(0, i-20):i+1]
                        if len(window) > 5:
                            result = classifier.classify(window.to_frame(name='close'))
                            test_results.append(result)

                    if test_results:
                        signal_gen = RegimeSignalGenerator()
                        oos_result = backtester.run(
                            test_prices.iloc[-len(test_results):],
                            test_results,
                            signal_gen
                        )

                        results.append({
                            'train_start': prices.index[start_idx],
                            'train_end': prices.index[train_end - 1],
                            'test_start': prices.index[train_end],
                            'test_end': prices.index[test_end - 1],
                            'best_params': best_params,
                            'train_sharpe': best_sharpe,
                            'test_sharpe': oos_result.sharpe_ratio,
                            'test_return': oos_result.total_return
                        })
                except Exception:
                    pass

            start_idx += self.step_size

        return {
            'windows': results,
            'avg_test_sharpe': np.mean([r['test_sharpe'] for r in results]) if results else 0,
            'avg_test_return': np.mean([r['test_return'] for r in results]) if results else 0
        }

    def _generate_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations."""
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations
