"""
Backtesting framework for RDD trading strategies.

Provides:
- Trade simulation with transaction costs
- Performance metrics (Sharpe, Sortino, drawdown)
- Strategy evaluation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum


class SignalType(Enum):
    """Trading signal type."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class Trade:
    """Trade record."""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    size: float  # Positive = long, negative = short
    gross_return: float
    net_return: float
    signal_strength: float

    @property
    def pnl(self) -> float:
        """Calculate PnL."""
        if self.size > 0:
            return (self.exit_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.exit_price) * abs(self.size)

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.net_return > 0

    @property
    def holding_period(self) -> int:
        """Get holding period in bars."""
        return self.exit_idx - self.entry_idx


@dataclass
class BacktestResults:
    """Backtest results and statistics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    avg_trade_return: float = 0.0
    avg_holding_period: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """Return summary string."""
        return f"""=== Backtest Results ===
Total Return: {self.total_return * 100:.2f}%
Annualized Return: {self.annualized_return * 100:.2f}%
Sharpe Ratio: {self.sharpe_ratio:.3f}
Sortino Ratio: {self.sortino_ratio:.3f}
Max Drawdown: {self.max_drawdown * 100:.2f}%
Win Rate: {self.win_rate * 100:.2f}%
Profit Factor: {self.profit_factor:.2f}
Number of Trades: {self.num_trades}
Avg Trade Return: {self.avg_trade_return * 100:.4f}%
Avg Holding Period: {self.avg_holding_period:.1f} bars"""


class RDDBacktester:
    """
    Backtester for RDD-based trading strategies.

    Parameters
    ----------
    initial_capital : float
        Starting capital
    transaction_cost : float
        Transaction cost as fraction (e.g., 0.001 for 0.1%)
    slippage : float
        Slippage as fraction
    position_size : float
        Maximum position size as fraction of capital
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0001,
        position_size: float = 0.1,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.position_size = position_size
        self.bars_per_year = 252 * 24  # Hourly bars

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        signal_strengths: Optional[np.ndarray] = None,
        holding_period: int = 24,
    ) -> BacktestResults:
        """
        Run backtest on price data with signals.

        Parameters
        ----------
        prices : np.ndarray
            Price series
        signals : np.ndarray
            Signal series (1 = long, -1 = short, 0 = neutral)
        signal_strengths : np.ndarray, optional
            Signal strength (0 to 1) for position sizing
        holding_period : int
            Fixed holding period in bars

        Returns
        -------
        BacktestResults
            Backtest results and statistics.
        """
        n = len(prices)
        if signal_strengths is None:
            signal_strengths = np.ones(n)

        trades = []
        equity = self.initial_capital
        equity_curve = [equity]

        in_position = False
        position_entry_idx = 0
        position_size = 0.0
        position_entry_price = 0.0
        position_signal_strength = 0.0

        for i in range(n):
            # Check for exit
            if in_position and i >= position_entry_idx + holding_period:
                # Exit position
                exit_slippage = self.slippage * np.sign(position_size)
                exit_price = prices[i] * (1 - exit_slippage)

                if position_size > 0:
                    gross_return = (exit_price - position_entry_price) / position_entry_price
                else:
                    gross_return = (position_entry_price - exit_price) / position_entry_price

                net_return = gross_return - 2 * self.transaction_cost

                trade = Trade(
                    entry_idx=position_entry_idx,
                    exit_idx=i,
                    entry_price=position_entry_price,
                    exit_price=exit_price,
                    size=position_size,
                    gross_return=gross_return,
                    net_return=net_return,
                    signal_strength=position_signal_strength,
                )

                equity *= 1 + net_return * abs(position_size)
                trades.append(trade)
                in_position = False

            # Check for entry
            if not in_position and i + holding_period < n:
                if signals[i] != 0 and signal_strengths[i] > 0.01:
                    size = min(signal_strengths[i] * self.position_size, self.position_size)
                    if size > 0.001:
                        in_position = True
                        position_entry_idx = i
                        position_size = size if signals[i] > 0 else -size
                        entry_slippage = self.slippage * np.sign(position_size)
                        position_entry_price = prices[i] * (1 + entry_slippage)
                        position_signal_strength = signal_strengths[i]

            equity_curve.append(equity)

        return self._calculate_statistics(trades, equity_curve)

    def run_rdd_strategy(
        self,
        prices: np.ndarray,
        running_var: np.ndarray,
        rdd_results: 'RDDResults',
        cutoff: float,
        bandwidth: float,
        holding_period: int = 24,
    ) -> BacktestResults:
        """
        Run backtest using RDD model results.

        Parameters
        ----------
        prices : np.ndarray
            Price series
        running_var : np.ndarray
            Running variable values (e.g., RSI)
        rdd_results : RDDResults
            Fitted RDD model results
        cutoff : float
            RDD cutoff/threshold
        bandwidth : float
            Bandwidth around cutoff
        holding_period : int
            Holding period in bars

        Returns
        -------
        BacktestResults
            Backtest results.
        """
        from python.rdd_model import RDDResults

        n = len(prices)
        signals = np.zeros(n)
        strengths = np.zeros(n)

        for i in range(n):
            distance = running_var[i] - cutoff
            if abs(distance) <= bandwidth:
                # Within bandwidth - generate signal
                if rdd_results.p_value < 0.1:
                    # Signal direction based on treatment effect sign
                    signals[i] = np.sign(rdd_results.tau)
                    # Strength based on distance from cutoff
                    strengths[i] = 1 - abs(distance) / bandwidth

        return self.run(prices, signals, strengths, holding_period)

    def _calculate_statistics(
        self,
        trades: List[Trade],
        equity_curve: List[float],
    ) -> BacktestResults:
        """Calculate backtest statistics."""
        results = BacktestResults(trades=trades, equity_curve=equity_curve)
        num_trades = len(trades)

        if num_trades == 0:
            return results

        results.num_trades = num_trades

        # Total return
        initial = equity_curve[0]
        final = equity_curve[-1]
        results.total_return = (final - initial) / initial

        # Annualized return
        n_bars = len(equity_curve)
        years = n_bars / self.bars_per_year
        if years > 0:
            results.annualized_return = (1 + results.total_return) ** (1 / years) - 1

        # Returns statistics
        returns = np.array([t.net_return for t in trades])
        mean_return = returns.mean()
        results.avg_trade_return = mean_return

        # Sharpe ratio
        std_return = returns.std()
        if std_return > 0:
            trades_per_year = num_trades / years if years > 0 else num_trades
            results.sharpe_ratio = mean_return / std_return * np.sqrt(trades_per_year)

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.sqrt(np.mean(downside_returns ** 2))
            if downside_std > 0:
                trades_per_year = num_trades / years if years > 0 else num_trades
                results.sortino_ratio = mean_return / downside_std * np.sqrt(trades_per_year)

        # Max drawdown
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        results.max_drawdown = drawdowns.max()

        # Win rate
        winners = sum(1 for t in trades if t.is_winner)
        results.win_rate = winners / num_trades

        # Profit factor
        gross_profit = sum(t.net_return for t in trades if t.net_return > 0)
        gross_loss = abs(sum(t.net_return for t in trades if t.net_return < 0))
        if gross_loss > 0:
            results.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            results.profit_factor = float('inf')

        # Average holding period
        results.avg_holding_period = np.mean([t.holding_period for t in trades])

        return results

    def create_rsi_strategy(
        self,
        threshold_low: float = 30,
        threshold_high: float = 70,
        bandwidth: float = 5,
        holding_period: int = 24,
    ) -> Callable:
        """
        Create an RSI threshold strategy function.

        Parameters
        ----------
        threshold_low : float
            RSI oversold threshold
        threshold_high : float
            RSI overbought threshold
        bandwidth : float
            Bandwidth around thresholds
        holding_period : int
            Holding period in bars

        Returns
        -------
        Callable
            Strategy function that takes (prices, rsi) and returns BacktestResults.
        """
        def strategy(prices: np.ndarray, rsi: np.ndarray) -> BacktestResults:
            n = len(prices)
            signals = np.zeros(n)
            strengths = np.zeros(n)

            for i in range(n):
                # Oversold - go long
                low_dist = rsi[i] - threshold_low
                if abs(low_dist) <= bandwidth:
                    signals[i] = 1
                    strengths[i] = 1 - abs(low_dist) / bandwidth

                # Overbought - go short
                high_dist = rsi[i] - threshold_high
                if abs(high_dist) <= bandwidth:
                    signals[i] = -1
                    strengths[i] = 1 - abs(high_dist) / bandwidth

            return self.run(prices, signals, strengths, holding_period)

        return strategy


def walk_forward_validation(
    prices: np.ndarray,
    running_var: np.ndarray,
    rdd_class,
    cutoff: float,
    train_size: int = 500,
    test_size: int = 100,
    step_size: int = 50,
    holding_period: int = 24,
) -> List[BacktestResults]:
    """
    Perform walk-forward validation of RDD strategy.

    Parameters
    ----------
    prices : np.ndarray
        Price series
    running_var : np.ndarray
        Running variable values
    rdd_class : class
        RegressionDiscontinuity class
    cutoff : float
        RDD cutoff value
    train_size : int
        Training window size
    test_size : int
        Test window size
    step_size : int
        Step size between windows
    holding_period : int
        Holding period for trades

    Returns
    -------
    list of BacktestResults
        Results for each test window.
    """
    n = len(prices)
    results = []
    backtester = RDDBacktester()

    # Calculate forward returns for fitting
    forward_returns = np.zeros(n)
    for i in range(n - holding_period):
        if prices[i] > 0:
            forward_returns[i] = (prices[i + holding_period] - prices[i]) / prices[i]

    start = 0
    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size

        # Fit RDD on training data
        train_x = running_var[start:train_end]
        train_y = forward_returns[start:train_end]

        try:
            rdd = rdd_class(cutoff=cutoff, bandwidth='optimal')
            rdd_results = rdd.fit(train_x, train_y)

            # Test on out-of-sample data
            test_prices = prices[train_end:test_end]
            test_x = running_var[train_end:test_end]

            test_result = backtester.run_rdd_strategy(
                test_prices,
                test_x,
                rdd_results,
                cutoff,
                rdd_results.bandwidth,
                holding_period,
            )
            results.append(test_result)

        except Exception as e:
            print(f"Error in window {start}-{train_end}: {e}")

        start += step_size

    return results


if __name__ == "__main__":
    # Example usage
    from python.rdd_model import compute_rsi

    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n)))
    rsi = compute_rsi(prices, 14)

    # Run RSI strategy backtest
    backtester = RDDBacktester(initial_capital=100_000)
    rsi_strategy = backtester.create_rsi_strategy(
        threshold_low=30,
        threshold_high=70,
        bandwidth=5,
        holding_period=24,
    )

    results = rsi_strategy(prices, rsi)
    print(results.summary())
