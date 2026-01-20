"""
Backtesting Module for LLM Alpha Mining

This module provides a backtesting framework for evaluating
LLM-generated alpha factors in realistic trading scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    quantity: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_price is not None

    def close(self, exit_price: float, exit_timestamp: datetime, fees: float = 0.0):
        """Close the trade."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.fees = fees

        if self.side == "long":
            self.pnl = (exit_price - self.entry_price) * self.quantity - fees
            self.pnl_pct = (exit_price / self.entry_price - 1) * 100
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity - fees
            self.pnl_pct = (1 - exit_price / self.entry_price) * 100


@dataclass
class BacktestResult:
    """
    Results from a backtest run.

    Contains comprehensive performance metrics and trade history.
    """
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_trade_duration: float
    max_consecutive_losses: int
    calmar_ratio: float
    sortino_ratio: float
    equity_curve: pd.Series
    trades: List[Trade]
    daily_returns: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = [
            "=" * 50,
            "BACKTEST RESULTS",
            "=" * 50,
            f"Total Return:         {self.total_return:.2%}",
            f"Annualized Return:    {self.annualized_return:.2%}",
            f"Sharpe Ratio:         {self.sharpe_ratio:.2f}",
            f"Sortino Ratio:        {self.sortino_ratio:.2f}",
            f"Max Drawdown:         {self.max_drawdown:.2%}",
            f"Calmar Ratio:         {self.calmar_ratio:.2f}",
            "-" * 50,
            f"Total Trades:         {self.total_trades}",
            f"Win Rate:             {self.win_rate:.2%}",
            f"Profit Factor:        {self.profit_factor:.2f}",
            f"Avg Trade PnL:        {self.avg_trade_pnl:.4f}",
            f"Avg Trade Duration:   {self.avg_trade_duration:.1f} periods",
            f"Max Consec. Losses:   {self.max_consecutive_losses}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_trade_pnl": self.avg_trade_pnl,
            "avg_trade_duration": self.avg_trade_duration,
            "max_consecutive_losses": self.max_consecutive_losses,
            "calmar_ratio": self.calmar_ratio,
            "sortino_ratio": self.sortino_ratio,
        }


class Backtester:
    """
    Backtesting engine for alpha factors.

    Supports:
    - Long-only and long-short strategies
    - Transaction costs and slippage
    - Position sizing
    - Walk-forward analysis

    Examples:
        >>> backtester = Backtester(initial_capital=100000)
        >>> result = backtester.run(
        ...     signals=factor_signals,
        ...     prices=price_data,
        ...     long_threshold=0.5,
        ...     short_threshold=-0.5
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size: float = 1.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        periods_per_year: int = 252
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            position_size: Fraction of capital per trade (0-1)
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            periods_per_year: Trading periods per year
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.slippage = slippage
        self.periods_per_year = periods_per_year

    def run(
        self,
        signals: pd.Series,
        prices: pd.Series,
        long_threshold: float = 0.0,
        short_threshold: Optional[float] = None,
        max_holding_periods: int = 10,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> BacktestResult:
        """
        Run backtest on alpha signals.

        Args:
            signals: Alpha factor signals (positive = bullish)
            prices: Price series (close prices)
            long_threshold: Signal threshold for going long
            short_threshold: Signal threshold for going short (None = long-only)
            max_holding_periods: Maximum holding period before forced exit
            stop_loss: Stop loss percentage (e.g., 0.05 for 5%)
            take_profit: Take profit percentage

        Returns:
            BacktestResult with performance metrics
        """
        # Align data
        aligned = pd.DataFrame({
            "signal": signals,
            "price": prices
        }).dropna()

        if len(aligned) < 10:
            raise ValueError("Insufficient data for backtest")

        trades = []
        equity = [self.initial_capital]
        positions = []
        current_position: Optional[Trade] = None

        for i in range(len(aligned)):
            row = aligned.iloc[i]
            timestamp = aligned.index[i]
            signal = row["signal"]
            price = row["price"]

            # Check for exit conditions on existing position
            if current_position is not None:
                holding_periods = i - positions[-1] if positions else 0

                should_exit = False
                exit_reason = ""

                # Max holding period
                if holding_periods >= max_holding_periods:
                    should_exit = True
                    exit_reason = "max_holding"

                # Stop loss
                if stop_loss is not None:
                    if current_position.side == "long":
                        loss_pct = (price / current_position.entry_price - 1)
                        if loss_pct <= -stop_loss:
                            should_exit = True
                            exit_reason = "stop_loss"
                    else:
                        loss_pct = (1 - price / current_position.entry_price)
                        if loss_pct <= -stop_loss:
                            should_exit = True
                            exit_reason = "stop_loss"

                # Take profit
                if take_profit is not None:
                    if current_position.side == "long":
                        profit_pct = (price / current_position.entry_price - 1)
                        if profit_pct >= take_profit:
                            should_exit = True
                            exit_reason = "take_profit"
                    else:
                        profit_pct = (1 - price / current_position.entry_price)
                        if profit_pct >= take_profit:
                            should_exit = True
                            exit_reason = "take_profit"

                # Signal reversal
                if current_position.side == "long" and short_threshold is not None and signal <= short_threshold:
                    should_exit = True
                    exit_reason = "signal_reversal"
                elif current_position.side == "short" and signal >= long_threshold:
                    should_exit = True
                    exit_reason = "signal_reversal"

                if should_exit:
                    # Apply slippage to exit
                    exit_price = price * (1 - self.slippage if current_position.side == "long" else 1 + self.slippage)
                    fees = exit_price * current_position.quantity * self.commission

                    current_position.close(exit_price, timestamp, fees)
                    current_position.metadata["exit_reason"] = exit_reason
                    trades.append(current_position)
                    current_position = None

            # Check for entry signal
            if current_position is None:
                if signal >= long_threshold:
                    # Enter long
                    entry_price = price * (1 + self.slippage)
                    quantity = (equity[-1] * self.position_size) / entry_price
                    fees = entry_price * quantity * self.commission

                    current_position = Trade(
                        timestamp=timestamp,
                        symbol="ASSET",
                        side="long",
                        entry_price=entry_price,
                        quantity=quantity,
                        fees=fees
                    )
                    positions.append(i)

                elif short_threshold is not None and signal <= short_threshold:
                    # Enter short
                    entry_price = price * (1 - self.slippage)
                    quantity = (equity[-1] * self.position_size) / entry_price
                    fees = entry_price * quantity * self.commission

                    current_position = Trade(
                        timestamp=timestamp,
                        symbol="ASSET",
                        side="short",
                        entry_price=entry_price,
                        quantity=quantity,
                        fees=fees
                    )
                    positions.append(i)

            # Update equity
            if current_position is not None and not current_position.is_closed:
                unrealized_pnl = 0
                if current_position.side == "long":
                    unrealized_pnl = (price - current_position.entry_price) * current_position.quantity
                else:
                    unrealized_pnl = (current_position.entry_price - price) * current_position.quantity
                equity.append(equity[0] + sum(t.pnl for t in trades) + unrealized_pnl)
            else:
                equity.append(equity[0] + sum(t.pnl for t in trades))

        # Close any remaining position at end
        if current_position is not None:
            final_price = aligned["price"].iloc[-1]
            exit_price = final_price * (1 - self.slippage if current_position.side == "long" else 1 + self.slippage)
            fees = exit_price * current_position.quantity * self.commission
            current_position.close(exit_price, aligned.index[-1], fees)
            current_position.metadata["exit_reason"] = "end_of_data"
            trades.append(current_position)

        # Calculate metrics
        return self._calculate_metrics(trades, equity, aligned.index)

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity: List[float],
        index: pd.DatetimeIndex
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        equity_curve = pd.Series(equity[1:], index=index)

        # Returns
        total_return = (equity[-1] / self.initial_capital) - 1
        n_periods = len(equity) - 1
        annualized_return = (1 + total_return) ** (self.periods_per_year / n_periods) - 1

        # Daily returns
        daily_returns = equity_curve.pct_change().dropna()

        # Sharpe ratio
        if daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(self.periods_per_year)
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = daily_returns.mean() / downside_returns.std() * np.sqrt(self.periods_per_year)
        else:
            sortino = sharpe  # Fallback

        # Max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        total_trades = len(trades)

        if total_trades > 0:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]

            win_rate = len(winning_trades) / total_trades

            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            avg_trade_pnl = sum(t.pnl for t in trades) / total_trades

            # Average trade duration
            durations = []
            for t in trades:
                if t.exit_timestamp:
                    duration = (t.exit_timestamp - t.timestamp).total_seconds() / 86400
                    durations.append(duration)
            avg_duration = np.mean(durations) if durations else 0

            # Max consecutive losses
            max_consec_losses = 0
            current_consec = 0
            for t in trades:
                if t.pnl <= 0:
                    current_consec += 1
                    max_consec_losses = max(max_consec_losses, current_consec)
                else:
                    current_consec = 0
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_pnl = 0.0
            avg_duration = 0.0
            max_consec_losses = 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_pnl=avg_trade_pnl,
            avg_trade_duration=avg_duration,
            max_consecutive_losses=max_consec_losses,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=daily_returns
        )

    def walk_forward(
        self,
        signals: pd.Series,
        prices: pd.Series,
        train_periods: int = 252,
        test_periods: int = 63,
        **kwargs
    ) -> List[BacktestResult]:
        """
        Walk-forward analysis.

        Splits data into rolling train/test windows and runs
        backtests on each test period.

        Args:
            signals: Alpha factor signals
            prices: Price series
            train_periods: Training window size
            test_periods: Test window size
            **kwargs: Additional arguments for run()

        Returns:
            List of BacktestResults for each test window
        """
        results = []
        total_periods = len(signals)

        start = train_periods
        while start + test_periods <= total_periods:
            test_signals = signals.iloc[start:start + test_periods]
            test_prices = prices.iloc[start:start + test_periods]

            try:
                result = self.run(test_signals, test_prices, **kwargs)
                result.metadata["window_start"] = start
                result.metadata["window_end"] = start + test_periods
                results.append(result)
            except Exception as e:
                print(f"Walk-forward window {start} failed: {e}")

            start += test_periods

        return results

    def monte_carlo(
        self,
        signals: pd.Series,
        prices: pd.Series,
        n_simulations: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Monte Carlo analysis of backtest results.

        Shuffles trade order to estimate distribution of outcomes.

        Args:
            signals: Alpha factor signals
            prices: Price series
            n_simulations: Number of simulations
            **kwargs: Additional arguments for run()

        Returns:
            Dictionary with Monte Carlo statistics
        """
        # Run initial backtest
        result = self.run(signals, prices, **kwargs)

        if result.total_trades < 10:
            return {
                "error": "Insufficient trades for Monte Carlo analysis",
                "original_result": result
            }

        # Get trade PnLs
        trade_pnls = [t.pnl for t in result.trades]

        # Run simulations
        simulated_returns = []
        for _ in range(n_simulations):
            shuffled_pnls = np.random.permutation(trade_pnls)
            equity = self.initial_capital + np.cumsum(shuffled_pnls)
            total_return = (equity[-1] / self.initial_capital) - 1
            simulated_returns.append(total_return)

        return {
            "original_return": result.total_return,
            "mean_return": np.mean(simulated_returns),
            "std_return": np.std(simulated_returns),
            "percentile_5": np.percentile(simulated_returns, 5),
            "percentile_25": np.percentile(simulated_returns, 25),
            "percentile_50": np.percentile(simulated_returns, 50),
            "percentile_75": np.percentile(simulated_returns, 75),
            "percentile_95": np.percentile(simulated_returns, 95),
            "probability_profit": np.mean([r > 0 for r in simulated_returns]),
            "original_result": result
        }


def backtest_factor(
    factor_expression: str,
    data: pd.DataFrame,
    long_threshold: float = 0.5,
    short_threshold: Optional[float] = -0.5,
    initial_capital: float = 100000.0,
    **kwargs
) -> BacktestResult:
    """
    Convenience function to backtest an alpha factor expression.

    Args:
        factor_expression: The alpha expression to test
        data: OHLCV DataFrame
        long_threshold: Signal threshold for long entry
        short_threshold: Signal threshold for short entry
        initial_capital: Starting capital
        **kwargs: Additional backtester arguments

    Returns:
        BacktestResult
    """
    from .alpha_generator import AlphaExpressionParser

    parser = AlphaExpressionParser()
    backtester = Backtester(initial_capital=initial_capital, **kwargs)

    # Calculate factor values
    factor_values = parser.evaluate(factor_expression, data)

    # Normalize to z-score for consistent thresholds
    factor_zscore = (factor_values - factor_values.mean()) / factor_values.std()

    return backtester.run(
        signals=factor_zscore,
        prices=data["close"],
        long_threshold=long_threshold,
        short_threshold=short_threshold
    )


if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    from alpha_generator import AlphaExpressionParser, PREDEFINED_FACTORS

    print("LLM Alpha Mining - Backtester Demo")
    print("=" * 60)

    # Generate data
    print("\n1. Loading Data")
    print("-" * 40)
    data = generate_synthetic_data(["BTCUSDT"], days=500)
    btc_data = data["BTCUSDT"].ohlcv
    print(f"Data shape: {btc_data.shape}")
    print(f"Date range: {btc_data.index[0].date()} to {btc_data.index[-1].date()}")

    # Initialize components
    parser = AlphaExpressionParser()
    backtester = Backtester(
        initial_capital=100000,
        position_size=0.5,
        commission=0.001,
        slippage=0.0005
    )

    # Backtest predefined factors
    print("\n2. Backtesting Predefined Factors")
    print("-" * 40)

    results = {}
    for factor in PREDEFINED_FACTORS[:4]:
        try:
            # Calculate factor
            factor_values = parser.evaluate(factor.expression, btc_data)

            # Normalize to z-score
            factor_zscore = (factor_values - factor_values.mean()) / factor_values.std()

            # Run backtest
            result = backtester.run(
                signals=factor_zscore,
                prices=btc_data["close"],
                long_threshold=0.5,
                short_threshold=-0.5,
                max_holding_periods=10,
                stop_loss=0.05,
                take_profit=0.10
            )

            results[factor.name] = result

            print(f"\n{factor.name}:")
            print(f"  Total Return: {result.total_return:.2%}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {result.max_drawdown:.2%}")
            print(f"  Win Rate: {result.win_rate:.2%}")
            print(f"  Total Trades: {result.total_trades}")

        except Exception as e:
            print(f"\n{factor.name}: Error - {e}")

    # Compare results
    print("\n3. Factor Comparison")
    print("-" * 40)

    comparison_df = pd.DataFrame({
        name: {
            "Return": f"{r.total_return:.2%}",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "MaxDD": f"{r.max_drawdown:.2%}",
            "WinRate": f"{r.win_rate:.2%}",
            "Trades": r.total_trades
        }
        for name, r in results.items()
    }).T

    print(comparison_df)

    # Walk-forward analysis
    print("\n4. Walk-Forward Analysis (momentum_5d)")
    print("-" * 40)

    if "momentum_5d" in results:
        factor_values = parser.evaluate(
            PREDEFINED_FACTORS[0].expression,
            btc_data
        )
        factor_zscore = (factor_values - factor_values.mean()) / factor_values.std()

        wf_results = backtester.walk_forward(
            signals=factor_zscore,
            prices=btc_data["close"],
            train_periods=100,
            test_periods=50,
            long_threshold=0.5,
            short_threshold=-0.5
        )

        print(f"\nWalk-forward windows: {len(wf_results)}")
        for i, r in enumerate(wf_results):
            print(f"  Window {i + 1}: Return={r.total_return:.2%}, Sharpe={r.sharpe_ratio:.2f}")

        # Aggregate walk-forward stats
        wf_returns = [r.total_return for r in wf_results]
        print(f"\nAggregate Walk-Forward Stats:")
        print(f"  Mean Return: {np.mean(wf_returns):.2%}")
        print(f"  Std Return: {np.std(wf_returns):.2%}")
        print(f"  Win Rate (windows): {np.mean([r > 0 for r in wf_returns]):.2%}")

    # Monte Carlo analysis
    print("\n5. Monte Carlo Analysis (1000 simulations)")
    print("-" * 40)

    if "momentum_5d" in results:
        mc_result = backtester.monte_carlo(
            signals=factor_zscore,
            prices=btc_data["close"],
            n_simulations=1000,
            long_threshold=0.5,
            short_threshold=-0.5
        )

        print(f"Original Return: {mc_result['original_return']:.2%}")
        print(f"Mean Simulated Return: {mc_result['mean_return']:.2%}")
        print(f"5th Percentile: {mc_result['percentile_5']:.2%}")
        print(f"95th Percentile: {mc_result['percentile_95']:.2%}")
        print(f"Probability of Profit: {mc_result['probability_profit']:.2%}")

    # Detailed result for best factor
    print("\n6. Best Factor Details")
    print("-" * 40)

    if results:
        best_name = max(results, key=lambda x: results[x].sharpe_ratio)
        best_result = results[best_name]

        print(best_result.summary())

        print("\nRecent Trades:")
        for trade in best_result.trades[-5:]:
            print(f"  {trade.timestamp.date()}: {trade.side.upper()} @ {trade.entry_price:.2f} -> "
                  f"{trade.exit_price:.2f if trade.exit_price else 'OPEN'}, PnL: {trade.pnl:.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
