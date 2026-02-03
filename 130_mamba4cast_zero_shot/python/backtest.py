"""
Backtesting Framework for Mamba4Cast

This module provides a comprehensive backtesting framework for evaluating
Mamba4Cast zero-shot forecasting strategies on historical data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    context_length: int = 100
    forecast_horizon: int = 24
    signal_threshold: float = 0.01
    rebalance_frequency: int = 1
    max_position_size: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class Mamba4CastBacktest:
    """
    Backtesting framework for Mamba4Cast zero-shot forecasting strategies.

    Example:
        model = Mamba4CastForecaster(...)
        backtest = Mamba4CastBacktest(model, initial_capital=100000)
        results = backtest.run(data, context_length=100, horizon=24)

        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    """

    def __init__(
        self,
        model,
        config: Optional[BacktestConfig] = None,
        **kwargs,
    ):
        """
        Initialize the backtester.

        Args:
            model: Mamba4CastForecaster model
            config: Backtest configuration
            **kwargs: Override config parameters
        """
        if config is None:
            config = BacktestConfig(**kwargs)
        else:
            # Override with kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.model = model
        self.config = config

        # Initialize state
        self._reset()

    def _reset(self):
        """Reset backtest state."""
        self.capital = self.config.initial_capital
        self.position = 0
        self.position_price = 0
        self.trades = []
        self.equity_curve = []
        self.signals = []

    def run(
        self,
        data: pd.DataFrame,
        price_col: str = 'close',
        verbose: bool = False,
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with OHLCV data
            price_col: Column to use for prices
            verbose: Print progress

        Returns:
            Dictionary with backtest results
        """
        self._reset()

        prices = data[price_col].values
        n_samples = len(prices)

        start_idx = self.config.context_length
        end_idx = n_samples - self.config.forecast_horizon

        for i in range(start_idx, end_idx, self.config.rebalance_frequency):
            if verbose and i % 100 == 0:
                print(f"Processing sample {i}/{end_idx}")

            # Get context
            context = prices[i - self.config.context_length:i]
            context = context.reshape(-1, 1)

            # Generate forecast
            forecast = self._get_forecast(context)

            # Current price
            current_price = prices[i]

            # Generate signal
            signal = self._generate_signal(forecast, current_price)
            self.signals.append({
                'index': i,
                'signal': signal['signal'],
                'expected_return': signal['expected_return'],
            })

            # Execute trade
            self._execute_trade(signal, current_price, i)

            # Check stop-loss / take-profit
            if self.position != 0:
                self._check_exit_conditions(current_price, i)

            # Update equity
            equity = self.capital + self.position * current_price
            self.equity_curve.append(equity)

        # Final exit if still in position
        if self.position != 0:
            final_price = prices[end_idx - 1]
            self._close_position(final_price, end_idx - 1, 'END')

        # Calculate metrics
        return self._calculate_metrics()

    def _get_forecast(self, context: np.ndarray) -> np.ndarray:
        """Get model forecast for context."""
        if HAS_TORCH:
            self.model.eval()
            with torch.no_grad():
                forecast = self.model.zero_shot_forecast(
                    context,
                    context_length=self.config.context_length,
                    horizon=self.config.forecast_horizon,
                )
        else:
            # Fallback: simple naive forecast
            forecast = np.tile(context[-1], (self.config.forecast_horizon, 1))

        return forecast

    def _generate_signal(
        self,
        forecast: np.ndarray,
        current_price: float,
    ) -> Dict:
        """Generate trading signal from forecast."""
        # Use forecast at horizon end
        predicted_price = forecast[-1, 0]
        expected_return = (predicted_price - current_price) / current_price

        if expected_return > self.config.signal_threshold:
            signal = 'BUY'
        elif expected_return < -self.config.signal_threshold:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return {
            'signal': signal,
            'expected_return': expected_return,
            'predicted_price': predicted_price,
        }

    def _execute_trade(
        self,
        signal: Dict,
        price: float,
        index: int,
    ):
        """Execute trade based on signal."""
        effective_price = price * (1 + self.config.slippage)
        cost_rate = self.config.transaction_cost

        if signal['signal'] == 'BUY' and self.position == 0:
            # Calculate position size
            position_value = self.capital * self.config.max_position_size
            cost = position_value * cost_rate

            shares = (position_value - cost) / effective_price
            self.position = shares
            self.position_price = effective_price
            self.capital -= position_value

            self.trades.append({
                'type': 'BUY',
                'index': index,
                'price': effective_price,
                'shares': shares,
                'cost': cost,
                'expected_return': signal['expected_return'],
            })

        elif signal['signal'] == 'SELL' and self.position > 0:
            self._close_position(price, index, 'SIGNAL')

    def _close_position(
        self,
        price: float,
        index: int,
        reason: str,
    ):
        """Close current position."""
        effective_price = price * (1 - self.config.slippage)
        proceeds = self.position * effective_price
        cost = proceeds * self.config.transaction_cost

        self.capital += proceeds - cost

        pnl = proceeds - cost - (self.position * self.position_price)
        pnl_pct = pnl / (self.position * self.position_price) * 100

        self.trades.append({
            'type': 'SELL',
            'index': index,
            'price': effective_price,
            'shares': self.position,
            'proceeds': proceeds,
            'cost': cost,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
        })

        self.position = 0
        self.position_price = 0

    def _check_exit_conditions(self, price: float, index: int):
        """Check stop-loss and take-profit conditions."""
        if self.position == 0:
            return

        current_pnl_pct = (price - self.position_price) / self.position_price

        # Check stop-loss
        if self.config.stop_loss is not None:
            if current_pnl_pct < -self.config.stop_loss:
                self._close_position(price, index, 'STOP_LOSS')
                return

        # Check take-profit
        if self.config.take_profit is not None:
            if current_pnl_pct > self.config.take_profit:
                self._close_position(price, index, 'TAKE_PROFIT')

    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics."""
        equity = np.array(self.equity_curve)

        if len(equity) == 0:
            return {'error': 'No equity data'}

        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]

        # Total return
        total_return = (equity[-1] / self.config.initial_capital - 1) * 100

        # Sharpe ratio (annualized)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0

        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino = np.sqrt(252) * returns.mean() / negative_returns.std()
        else:
            sortino = 0

        # Max drawdown
        peak = equity[0]
        max_dd = 0
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)

        # Trade statistics
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] == 'SELL']

        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]

        n_trades = len(buy_trades)
        win_rate = len(winning_trades) / len(sell_trades) * 100 if sell_trades else 0

        # Average win/loss
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd * 100,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_equity': equity[-1],
            'equity_curve': equity,
            'trades': self.trades,
            'signals': self.signals,
        }

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        train_pct: float = 0.7,
        n_splits: int = 5,
        price_col: str = 'close',
    ) -> List[Dict]:
        """
        Run walk-forward analysis.

        Args:
            data: Full dataset
            train_pct: Percentage for training window
            n_splits: Number of walk-forward splits
            price_col: Price column

        Returns:
            List of results for each split
        """
        n = len(data)
        split_size = n // n_splits
        results = []

        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = min((i + 2) * split_size, n)

            split_data = data.iloc[start_idx:end_idx].copy()
            split_data = split_data.reset_index(drop=True)

            # Run backtest on this split
            result = self.run(split_data, price_col=price_col)
            result['split'] = i
            result['start_idx'] = start_idx
            result['end_idx'] = end_idx
            results.append(result)

        return results

    def generate_report(self, results: Dict) -> str:
        """
        Generate a text report from backtest results.

        Args:
            results: Backtest results dictionary

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("MAMBA4CAST BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")

        # Performance metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Total Return:      {results['total_return']:>12.2f}%")
        report.append(f"Sharpe Ratio:      {results['sharpe_ratio']:>12.2f}")
        report.append(f"Sortino Ratio:     {results['sortino_ratio']:>12.2f}")
        report.append(f"Max Drawdown:      {results['max_drawdown']:>12.2f}%")
        report.append(f"Final Equity:      ${results['final_equity']:>11,.2f}")
        report.append("")

        # Trade statistics
        report.append("TRADE STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Trades:      {results['n_trades']:>12d}")
        report.append(f"Win Rate:          {results['win_rate']:>12.2f}%")
        report.append(f"Avg Win:           {results['avg_win']:>12.2f}%")
        report.append(f"Avg Loss:          {results['avg_loss']:>12.2f}%")
        report.append(f"Profit Factor:     {results['profit_factor']:>12.2f}")
        report.append("")

        # Configuration
        report.append("CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Initial Capital:   ${self.config.initial_capital:>11,.2f}")
        report.append(f"Transaction Cost:  {self.config.transaction_cost*100:>12.2f}%")
        report.append(f"Context Length:    {self.config.context_length:>12d}")
        report.append(f"Forecast Horizon:  {self.config.forecast_horizon:>12d}")
        report.append(f"Signal Threshold:  {self.config.signal_threshold*100:>12.2f}%")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)


def run_simple_backtest(
    model,
    data: pd.DataFrame,
    initial_capital: float = 100000,
    context_length: int = 100,
    horizon: int = 24,
    threshold: float = 0.01,
) -> Dict:
    """
    Convenience function for quick backtesting.

    Args:
        model: Mamba4CastForecaster model
        data: DataFrame with OHLCV data
        initial_capital: Starting capital
        context_length: Historical context
        horizon: Forecast horizon
        threshold: Signal threshold

    Returns:
        Backtest results
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        context_length=context_length,
        forecast_horizon=horizon,
        signal_threshold=threshold,
    )

    backtest = Mamba4CastBacktest(model, config)
    return backtest.run(data)
