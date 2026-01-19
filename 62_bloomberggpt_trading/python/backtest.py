"""
Backtesting Module for LLM Trading Signals

This module provides tools for backtesting trading strategies
based on LLM-generated signals.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class BacktestConfig:
    """Configuration for LLM signal backtesting."""
    initial_capital: float = 100000
    max_position_size: float = 0.1       # Max 10% per position
    max_total_exposure: float = 1.0      # Max 100% invested
    transaction_cost_bps: float = 10     # 10 basis points
    slippage_bps: float = 5              # 5 basis points
    signal_decay_hours: float = 24       # Signal relevance window
    rebalance_frequency: str = "daily"   # "hourly", "daily"
    min_signal_confidence: float = 0.5   # Minimum confidence to act
    short_selling: bool = True           # Allow short positions


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: datetime
    symbol: str
    shares: float
    price: float
    value: float
    type: str  # "BUY" or "SELL"
    signal_strength: float
    transaction_cost: float


@dataclass
class BacktestResult:
    """Results from backtesting LLM signals."""
    returns: pd.Series
    positions: pd.DataFrame
    trades: List[Trade]
    metrics: Dict[str, float]
    equity_curve: pd.Series
    config: BacktestConfig


class LLMSignalBacktester:
    """
    Backtest trading signals generated from LLM analysis.

    This backtester handles the unique characteristics of LLM-derived signals:
    - Irregular signal timing (news-driven)
    - Signal decay over time
    - Varying confidence levels
    - Multi-asset portfolios

    Examples:
        >>> config = BacktestConfig(initial_capital=100000)
        >>> backtester = LLMSignalBacktester(config)
        >>> result = backtester.run_backtest(signals_df, prices_df)
        >>> print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest on LLM signals.

        Args:
            signals: DataFrame with columns [timestamp, symbol, signal, confidence]
            prices: DataFrame with price data, columns are symbols, index is timestamp
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResult with performance metrics
        """
        # Ensure timestamps are datetime
        signals = signals.copy()
        if not isinstance(signals['timestamp'].iloc[0], datetime):
            signals['timestamp'] = pd.to_datetime(signals['timestamp'])

        prices = prices.copy()
        if not isinstance(prices.index[0], datetime):
            prices.index = pd.to_datetime(prices.index)

        # Filter date range
        if start_date:
            signals = signals[signals['timestamp'] >= start_date]
            prices = prices[prices.index >= start_date]
        if end_date:
            signals = signals[signals['timestamp'] <= end_date]
            prices = prices[prices.index <= end_date]

        if prices.empty:
            return self._empty_result()

        # Initialize tracking
        capital = self.config.initial_capital
        positions: Dict[str, float] = {}  # symbol -> shares
        position_history = []
        trades: List[Trade] = []
        portfolio_values = []

        # Get rebalance points
        rebalance_points = self._get_rebalance_points(prices)

        for ts in rebalance_points:
            # Get current prices
            if ts not in prices.index:
                continue
            current_prices = prices.loc[ts]

            # Get active signals with decay
            active_signals = self._get_active_signals(signals, ts)

            # Calculate target positions
            target_positions = self._calculate_positions(
                active_signals,
                current_prices,
                capital,
                positions
            )

            # Execute rebalance
            new_trades, capital = self._execute_rebalance(
                positions,
                target_positions,
                current_prices,
                capital,
                ts,
                active_signals
            )
            trades.extend(new_trades)

            # Update positions
            positions = target_positions.copy()

            # Calculate portfolio value
            position_value = sum(
                shares * current_prices.get(symbol, 0)
                for symbol, shares in positions.items()
                if symbol in current_prices.index
            )
            portfolio_value = capital + position_value
            portfolio_values.append({
                "timestamp": ts,
                "value": portfolio_value,
                "capital": capital,
                "position_value": position_value
            })

            # Record positions
            position_history.append({
                "timestamp": ts,
                **{f"pos_{s}": shares for s, shares in positions.items()}
            })

        # Build results DataFrames
        pv_df = pd.DataFrame(portfolio_values).set_index('timestamp')
        if pv_df.empty:
            return self._empty_result()

        returns = pv_df['value'].pct_change().dropna()
        equity_curve = pv_df['value']

        # Calculate metrics
        metrics = self._calculate_metrics(returns, trades, equity_curve)

        return BacktestResult(
            returns=returns,
            positions=pd.DataFrame(position_history),
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
            config=self.config
        )

    def _get_rebalance_points(self, prices: pd.DataFrame) -> pd.DatetimeIndex:
        """Get rebalance timestamps based on frequency."""
        if self.config.rebalance_frequency == "daily":
            # Daily at market close (use last timestamp per day)
            return prices.groupby(prices.index.date).last().index
        elif self.config.rebalance_frequency == "hourly":
            return prices.index
        else:
            return prices.index

    def _get_active_signals(
        self,
        signals: pd.DataFrame,
        current_time: datetime
    ) -> pd.DataFrame:
        """Get signals that are still active with decay applied."""
        decay_hours = self.config.signal_decay_hours

        # Filter to signals within decay window
        if isinstance(current_time, pd.Timestamp):
            current_time = current_time.to_pydatetime()

        cutoff = current_time - timedelta(hours=decay_hours)
        active = signals[
            (signals['timestamp'] >= cutoff) &
            (signals['timestamp'] <= current_time)
        ].copy()

        if active.empty:
            return active

        # Apply time decay
        active['hours_ago'] = active['timestamp'].apply(
            lambda x: (current_time - x.to_pydatetime()).total_seconds() / 3600
            if hasattr(x, 'to_pydatetime') else (current_time - x).total_seconds() / 3600
        )
        active['decay_factor'] = np.exp(-active['hours_ago'] / decay_hours)
        active['adjusted_signal'] = (
            active['signal'] * active['confidence'] * active['decay_factor']
        )

        # Aggregate by symbol (sum adjusted signals)
        aggregated = active.groupby('symbol').agg({
            'adjusted_signal': 'sum',
            'confidence': 'mean',
            'signal': 'mean'
        }).reset_index()

        # Clip signals to [-1, 1]
        aggregated['adjusted_signal'] = aggregated['adjusted_signal'].clip(-1, 1)

        return aggregated

    def _calculate_positions(
        self,
        signals: pd.DataFrame,
        prices: pd.Series,
        capital: float,
        current_positions: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate target positions from signals."""
        if signals.empty:
            return {}

        positions = {}
        max_position_value = capital * self.config.max_position_size

        for _, row in signals.iterrows():
            symbol = row['symbol']
            if symbol not in prices.index:
                continue

            # Skip low confidence signals
            if row['confidence'] < self.config.min_signal_confidence:
                continue

            price = prices[symbol]
            if price <= 0:
                continue

            # Signal determines position direction and size
            signal_strength = row['adjusted_signal']

            # Skip if short selling disabled and signal is negative
            if not self.config.short_selling and signal_strength < 0:
                continue

            # Position size based on signal strength
            position_value = signal_strength * max_position_value
            shares = position_value / price

            positions[symbol] = shares

        # Apply total exposure limit
        total_exposure = sum(
            abs(shares * prices.get(symbol, 0))
            for symbol, shares in positions.items()
        )
        max_exposure = capital * self.config.max_total_exposure

        if total_exposure > max_exposure and total_exposure > 0:
            scale_factor = max_exposure / total_exposure
            positions = {s: shares * scale_factor for s, shares in positions.items()}

        return positions

    def _execute_rebalance(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        prices: pd.Series,
        capital: float,
        timestamp: datetime,
        signals: pd.DataFrame
    ) -> Tuple[List[Trade], float]:
        """Execute rebalance trades."""
        trades = []

        all_symbols = set(current.keys()) | set(target.keys())

        for symbol in all_symbols:
            current_shares = current.get(symbol, 0)
            target_shares = target.get(symbol, 0)

            if symbol not in prices.index:
                continue

            delta = target_shares - current_shares
            if abs(delta) < 0.0001:  # Skip tiny trades
                continue

            price = prices[symbol]

            # Get signal strength for this trade
            signal_row = signals[signals['symbol'] == symbol]
            signal_strength = (
                signal_row['adjusted_signal'].iloc[0]
                if not signal_row.empty else 0
            )

            # Calculate costs
            cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
            cost_factor = cost_bps / 10000

            if delta > 0:  # Buy
                trade_value = delta * price * (1 + cost_factor)
                transaction_cost = delta * price * cost_factor
            else:  # Sell
                trade_value = delta * price * (1 - cost_factor)
                transaction_cost = abs(delta * price * cost_factor)

            capital -= trade_value

            trades.append(Trade(
                timestamp=timestamp,
                symbol=symbol,
                shares=delta,
                price=price,
                value=trade_value,
                type="BUY" if delta > 0 else "SELL",
                signal_strength=signal_strength,
                transaction_cost=transaction_cost
            ))

        return trades, capital

    def _calculate_metrics(
        self,
        returns: pd.Series,
        trades: List[Trade],
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """Calculate backtest performance metrics."""
        if returns.empty or len(returns) < 2:
            return {}

        # Annualization factor (assuming daily returns)
        ann_factor = 252

        # Returns metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ann_factor)

        # Risk-adjusted returns
        rf_rate = 0.02  # 2% risk-free rate
        sharpe = (ann_return - rf_rate) / volatility if volatility > 0 else 0

        # Downside deviation for Sortino
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(ann_factor) if len(downside_returns) > 0 else volatility
        sortino = (ann_return - rf_rate) / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        n_trades = len(trades)
        if trades:
            buy_trades = [t for t in trades if t.type == "BUY"]
            sell_trades = [t for t in trades if t.type == "SELL"]
            total_costs = sum(t.transaction_cost for t in trades)

            # Win rate (based on signal direction vs actual return)
            # Simplified: count trades with positive value
            winning_trades = sum(1 for t in trades if t.value > 0)
            win_rate = winning_trades / n_trades if n_trades > 0 else 0
        else:
            total_costs = 0
            win_rate = 0

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "num_trades": n_trades,
            "total_transaction_costs": total_costs,
            "win_rate": win_rate,
            "start_value": equity_curve.iloc[0] if len(equity_curve) > 0 else 0,
            "end_value": equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
        }

    def _empty_result(self) -> BacktestResult:
        """Return empty result for invalid inputs."""
        return BacktestResult(
            returns=pd.Series(dtype=float),
            positions=pd.DataFrame(),
            trades=[],
            metrics={},
            equity_curve=pd.Series(dtype=float),
            config=self.config
        )


def generate_synthetic_data(
    symbols: List[str],
    n_days: int = 252,
    n_signals_per_day: float = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic price and signal data for testing.

    Args:
        symbols: List of symbols
        n_days: Number of trading days
        n_signals_per_day: Average signals per day

    Returns:
        Tuple of (signals_df, prices_df)
    """
    np.random.seed(42)

    # Generate dates
    dates = pd.date_range(
        start="2024-01-01",
        periods=n_days,
        freq="B"  # Business days
    )

    # Generate prices with random walk
    prices_data = {}
    for symbol in symbols:
        initial_price = np.random.uniform(50, 500)
        returns = np.random.randn(n_days) * 0.02  # 2% daily vol
        prices_data[symbol] = initial_price * (1 + returns).cumprod()

    prices_df = pd.DataFrame(prices_data, index=dates)

    # Generate signals
    total_signals = int(n_days * n_signals_per_day)
    signal_dates = np.random.choice(dates, size=total_signals, replace=True)
    signal_symbols = np.random.choice(symbols, size=total_signals)
    signal_values = np.random.uniform(-1, 1, size=total_signals)
    confidences = np.random.uniform(0.5, 1.0, size=total_signals)

    signals_df = pd.DataFrame({
        "timestamp": signal_dates,
        "symbol": signal_symbols,
        "signal": signal_values,
        "confidence": confidences
    })

    return signals_df, prices_df


def print_backtest_report(result: BacktestResult):
    """Print formatted backtest report."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    if not result.metrics:
        print("No metrics available (empty backtest)")
        return

    m = result.metrics

    print(f"\nPERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"Total Return:        {m.get('total_return', 0):>10.2%}")
    print(f"Annualized Return:   {m.get('annualized_return', 0):>10.2%}")
    print(f"Volatility:          {m.get('volatility', 0):>10.2%}")

    print(f"\nRISK-ADJUSTED METRICS")
    print("-" * 40)
    print(f"Sharpe Ratio:        {m.get('sharpe_ratio', 0):>10.2f}")
    print(f"Sortino Ratio:       {m.get('sortino_ratio', 0):>10.2f}")
    print(f"Calmar Ratio:        {m.get('calmar_ratio', 0):>10.2f}")

    print(f"\nRISK METRICS")
    print("-" * 40)
    print(f"Max Drawdown:        {m.get('max_drawdown', 0):>10.2%}")

    print(f"\nTRADING STATISTICS")
    print("-" * 40)
    print(f"Number of Trades:    {m.get('num_trades', 0):>10}")
    print(f"Win Rate:            {m.get('win_rate', 0):>10.2%}")
    print(f"Transaction Costs:   ${m.get('total_transaction_costs', 0):>9,.2f}")

    print(f"\nPORTFOLIO VALUE")
    print("-" * 40)
    print(f"Starting Value:      ${m.get('start_value', 0):>9,.2f}")
    print(f"Ending Value:        ${m.get('end_value', 0):>9,.2f}")

    print("=" * 60)


if __name__ == "__main__":
    # Demo backtest
    print("LLM Signal Backtesting Demo")

    # Generate synthetic data
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    signals_df, prices_df = generate_synthetic_data(symbols, n_days=252)

    print(f"\nGenerated {len(signals_df)} signals over {len(prices_df)} days")
    print(f"Symbols: {symbols}")

    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        max_position_size=0.15,
        transaction_cost_bps=10,
        signal_decay_hours=48
    )

    backtester = LLMSignalBacktester(config)
    result = backtester.run_backtest(signals_df, prices_df)

    # Print report
    print_backtest_report(result)
