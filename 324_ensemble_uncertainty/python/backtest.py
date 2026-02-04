"""
Backtesting module for uncertainty-aware trading strategies.

This module provides a backtesting framework to evaluate trading
strategies that incorporate ensemble uncertainty.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Support both module import and direct script execution
try:
    from .strategy import UncertaintyStrategy, TradingSignal, SignalType
except ImportError:
    from strategy import UncertaintyStrategy, TradingSignal, SignalType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    direction: str  # 'LONG' or 'SHORT'
    prediction: float
    uncertainty: float
    confidence: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None  # 'take_profit', 'stop_loss', 'signal', 'end'


@dataclass
class BacktestResult:
    """Results of a backtest."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_trade_duration: float
    equity_curve: pd.Series
    trades: List[Trade]
    daily_returns: pd.Series
    metrics_by_confidence: Optional[pd.DataFrame] = None


class Backtester:
    """
    Backtesting engine for uncertainty-aware strategies.

    Features:
    - Position sizing based on uncertainty
    - Transaction costs
    - Slippage modeling
    - Performance attribution by confidence level
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1%
        slippage: float = 0.0005,  # 0.05%
        risk_free_rate: float = 0.02  # 2% annual
    ):
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate

    def run_backtest(
        self,
        prices: pd.DataFrame,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        strategy: UncertaintyStrategy,
        hold_periods: int = 1
    ) -> BacktestResult:
        """
        Run a backtest on historical data.

        Args:
            prices: DataFrame with 'timestamp' and 'close' columns
            predictions: Array of predictions aligned with prices
            uncertainties: Array of uncertainties aligned with prices
            strategy: UncertaintyStrategy instance
            hold_periods: Number of periods to hold each position

        Returns:
            BacktestResult with performance metrics
        """
        n_periods = len(prices)
        capital = self.initial_capital
        equity = [capital]
        trades: List[Trade] = []
        current_position: Optional[Trade] = None

        logger.info(f"Running backtest on {n_periods} periods...")

        for i in range(n_periods - hold_periods):
            timestamp = prices['timestamp'].iloc[i]
            current_price = prices['close'].iloc[i]
            future_price = prices['close'].iloc[i + hold_periods]

            # Generate signal
            signal = strategy.generate_signal(
                symbol=prices['symbol'].iloc[i] if 'symbol' in prices.columns else 'UNKNOWN',
                prediction=predictions[i],
                uncertainty=uncertainties[i],
                current_price=current_price,
                timestamp=timestamp
            )

            # Handle existing position
            if current_position is not None:
                # Check for exit conditions or hold period end
                periods_held = i - current_position.entry_time.value
                if periods_held >= hold_periods:
                    # Close position
                    exit_price = current_price * (1 - self.slippage if current_position.direction == 'LONG' else 1 + self.slippage)
                    current_position.exit_time = timestamp
                    current_position.exit_price = exit_price
                    current_position.exit_reason = 'hold_period'

                    # Calculate PnL
                    if current_position.direction == 'LONG':
                        pnl_pct = (exit_price - current_position.entry_price) / current_position.entry_price
                    else:
                        pnl_pct = (current_position.entry_price - exit_price) / current_position.entry_price

                    pnl_pct -= self.transaction_cost  # Exit cost
                    current_position.pnl_pct = pnl_pct
                    current_position.pnl = capital * current_position.position_size * pnl_pct
                    capital += current_position.pnl

                    trades.append(current_position)
                    current_position = None

            # Open new position if no current position
            if current_position is None and signal.signal_type != SignalType.HOLD:
                entry_price = current_price * (1 + self.slippage if signal.signal_type == SignalType.LONG else 1 - self.slippage)
                entry_price *= (1 + self.transaction_cost)  # Entry cost

                current_position = Trade(
                    symbol=signal.symbol,
                    entry_time=timestamp,
                    exit_time=None,
                    entry_price=entry_price,
                    exit_price=None,
                    position_size=signal.position_size,
                    direction=signal.signal_type.value,
                    prediction=signal.prediction,
                    uncertainty=signal.uncertainty,
                    confidence=signal.confidence
                )

            equity.append(capital)

        # Close any remaining position
        if current_position is not None:
            exit_price = prices['close'].iloc[-1]
            current_position.exit_time = prices['timestamp'].iloc[-1]
            current_position.exit_price = exit_price
            current_position.exit_reason = 'end'

            if current_position.direction == 'LONG':
                pnl_pct = (exit_price - current_position.entry_price) / current_position.entry_price
            else:
                pnl_pct = (current_position.entry_price - exit_price) / current_position.entry_price

            pnl_pct -= self.transaction_cost
            current_position.pnl_pct = pnl_pct
            current_position.pnl = capital * current_position.position_size * pnl_pct
            capital += current_position.pnl
            trades.append(current_position)

        # Create equity curve
        equity_curve = pd.Series(equity, index=prices['timestamp'].iloc[:len(equity)])

        # Calculate metrics
        result = self._calculate_metrics(equity_curve, trades)

        return result

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Trade]
    ) -> BacktestResult:
        """Calculate performance metrics from equity curve and trades."""

        # Returns
        returns = equity_curve.pct_change().dropna()
        daily_returns = returns.resample('D').sum() if len(returns) > 0 else pd.Series()

        # Total return
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

        # Sharpe ratio (annualized)
        if len(returns) > 0 and returns.std() > 0:
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0

        # Sortino ratio
        if len(returns) > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std()
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = 0.0

        # Maximum drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]

            win_rate = len(winning_trades) / len(trades) if trades else 0.0

            total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0.0
            total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1e-10
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

            avg_trade_pnl = np.mean([t.pnl for t in trades if t.pnl]) if trades else 0.0
            avg_trade_duration = hold_periods = 1  # Simplified

            # Metrics by confidence level
            metrics_by_confidence = self._metrics_by_confidence(trades)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_pnl = 0.0
            avg_trade_duration = 0.0
            metrics_by_confidence = None

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_pnl=avg_trade_pnl,
            avg_trade_duration=avg_trade_duration,
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=daily_returns,
            metrics_by_confidence=metrics_by_confidence
        )

    def _metrics_by_confidence(
        self,
        trades: List[Trade],
        n_bins: int = 5
    ) -> pd.DataFrame:
        """Calculate metrics stratified by confidence level."""
        if not trades:
            return pd.DataFrame()

        confidences = np.array([t.confidence for t in trades])
        pnls = np.array([t.pnl_pct if t.pnl_pct else 0 for t in trades])

        # Create confidence bins
        bins = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(confidences, bins[1:-1])

        results = []
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                bin_pnls = pnls[mask]
                results.append({
                    'Confidence Bin': bin_idx + 1,
                    'Min Confidence': bins[bin_idx],
                    'Max Confidence': bins[bin_idx + 1],
                    'Num Trades': mask.sum(),
                    'Win Rate': np.mean(bin_pnls > 0),
                    'Avg Return': np.mean(bin_pnls),
                    'Sharpe': np.mean(bin_pnls) / (np.std(bin_pnls) + 1e-8) if mask.sum() > 1 else 0,
                })

        return pd.DataFrame(results)

    def compare_with_without_uncertainty(
        self,
        prices: pd.DataFrame,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        hold_periods: int = 1
    ) -> Dict[str, BacktestResult]:
        """
        Compare strategy with and without uncertainty weighting.

        Args:
            prices: Price data
            predictions: Predictions
            uncertainties: Uncertainties
            hold_periods: Holding period

        Returns:
            Dictionary with 'with_uncertainty' and 'without_uncertainty' results
        """
        # Strategy with uncertainty
        strategy_with = UncertaintyStrategy(
            min_prediction_threshold=0.001,
            max_uncertainty_threshold=0.05,
            min_confidence_threshold=0.5
        )

        # Strategy without uncertainty (always full position if signal)
        strategy_without = UncertaintyStrategy(
            min_prediction_threshold=0.001,
            max_uncertainty_threshold=1.0,  # Effectively no limit
            min_confidence_threshold=0.0,   # Always confident
            base_position_size=1.0
        )

        result_with = self.run_backtest(
            prices, predictions, uncertainties, strategy_with, hold_periods
        )
        result_without = self.run_backtest(
            prices, predictions, uncertainties, strategy_without, hold_periods
        )

        return {
            'with_uncertainty': result_with,
            'without_uncertainty': result_without
        }


def print_backtest_report(result: BacktestResult, title: str = "Backtest Results"):
    """Print a formatted backtest report."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"\nPerformance Metrics:")
    print(f"  Total Return:    {result.total_return:>10.2%}")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:   {result.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:    {result.max_drawdown:>10.2%}")

    print(f"\nTrade Statistics:")
    print(f"  Total Trades:    {result.total_trades:>10d}")
    print(f"  Win Rate:        {result.win_rate:>10.2%}")
    print(f"  Profit Factor:   {result.profit_factor:>10.2f}")
    print(f"  Avg Trade PnL:   {result.avg_trade_pnl:>10.2f}")

    if result.metrics_by_confidence is not None and len(result.metrics_by_confidence) > 0:
        print(f"\nPerformance by Confidence Level:")
        print(result.metrics_by_confidence.to_string(index=False))


def main():
    """Example backtest."""
    np.random.seed(42)

    print("=" * 60)
    print("Backtesting Uncertainty-Aware Strategy")
    print("=" * 60)

    # Generate sample price data
    n_periods = 1000
    dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='1h')
    price = 50000 + np.cumsum(np.random.randn(n_periods) * 100)

    prices = pd.DataFrame({
        'timestamp': dates,
        'close': price,
        'symbol': 'BTC/USDT'
    })

    # Generate predictions and uncertainties
    # Predictions based on some "features" with noise
    true_signal = np.sin(np.arange(n_periods) * 0.1) * 0.02
    predictions = true_signal + np.random.randn(n_periods) * 0.01

    # Uncertainty varies - higher when signal is weak
    base_uncertainty = 0.015
    uncertainties = base_uncertainty + np.abs(np.random.randn(n_periods)) * 0.01

    # Initialize backtester and strategy
    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )

    # Compare with and without uncertainty
    results = backtester.compare_with_without_uncertainty(
        prices, predictions, uncertainties, hold_periods=5
    )

    print_backtest_report(results['with_uncertainty'], "With Uncertainty Weighting")
    print_backtest_report(results['without_uncertainty'], "Without Uncertainty Weighting")

    # Summary comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    r_with = results['with_uncertainty']
    r_without = results['without_uncertainty']

    print(f"\n{'Metric':<20} {'With Unc.':<15} {'Without Unc.':<15} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Total Return':<20} {r_with.total_return:>14.2%} {r_without.total_return:>14.2%} {(r_with.total_return - r_without.total_return):>14.2%}")
    print(f"{'Sharpe Ratio':<20} {r_with.sharpe_ratio:>14.2f} {r_without.sharpe_ratio:>14.2f} {(r_with.sharpe_ratio - r_without.sharpe_ratio):>14.2f}")
    print(f"{'Max Drawdown':<20} {r_with.max_drawdown:>14.2%} {r_without.max_drawdown:>14.2%} {(r_with.max_drawdown - r_without.max_drawdown):>14.2%}")
    print(f"{'Win Rate':<20} {r_with.win_rate:>14.2%} {r_without.win_rate:>14.2%} {(r_with.win_rate - r_without.win_rate):>14.2%}")


if __name__ == '__main__':
    main()
