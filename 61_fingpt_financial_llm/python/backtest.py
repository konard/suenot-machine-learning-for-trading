"""
FinGPT Backtesting Module

Backtest trading strategies based on FinGPT sentiment analysis signals.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    max_position_pct: float = 0.1  # Max 10% per position
    transaction_cost_bps: float = 10  # 10 basis points
    slippage_bps: float = 5  # 5 basis points
    signal_threshold: float = 0.3
    signal_decay_hours: float = 24.0
    rebalance_frequency: str = "daily"


@dataclass
class BacktestResult:
    """Results of backtesting."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    profit_factor: float
    portfolio_values: pd.Series
    returns: pd.Series
    trades: List[Dict]


class FinGPTBacktester:
    """
    Backtest FinGPT-based trading strategies.

    This backtester is designed specifically for news-driven
    trading signals with irregular timing.

    Examples:
        >>> config = BacktestConfig(initial_capital=100000)
        >>> backtester = FinGPTBacktester(config)
        >>> result = backtester.run_backtest(signals, prices)
        >>> print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtester.

        Args:
            config: BacktestConfig with backtest parameters
        """
        self.config = config

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest on FinGPT signals.

        Args:
            signals: DataFrame with [timestamp, symbol, signal, confidence]
            prices: DataFrame with price data indexed by timestamp
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            BacktestResult with performance metrics
        """
        # Filter date range
        if start_date:
            signals = signals[signals['timestamp'] >= start_date]
            prices = prices[prices.index >= start_date]
        if end_date:
            signals = signals[signals['timestamp'] <= end_date]
            prices = prices[prices.index <= end_date]

        # Initialize tracking
        capital = self.config.initial_capital
        positions: Dict[str, float] = {}  # symbol -> shares
        portfolio_values = []
        trades = []

        # Get all unique dates
        dates = prices.index.unique().sort_values()

        for date in dates:
            # Get active signals for this date
            day_signals = self._get_active_signals(signals, date)

            # Calculate target positions
            date_prices = prices.loc[date]
            if isinstance(date_prices, pd.DataFrame):
                date_prices = date_prices.iloc[0]

            target = self._calculate_targets(day_signals, date_prices, capital)

            # Execute trades
            new_trades, capital = self._execute_trades(
                positions, target, date_prices, capital, date
            )
            trades.extend(new_trades)
            positions = target.copy()

            # Calculate portfolio value
            position_value = sum(
                shares * date_prices[symbol]
                for symbol, shares in positions.items()
                if symbol in date_prices.index
            )
            portfolio_values.append({
                "date": date,
                "value": capital + position_value
            })

        # Calculate metrics
        pv = pd.DataFrame(portfolio_values).set_index("date")["value"]
        returns = pv.pct_change().dropna()

        # Calculate trade statistics
        trade_pnls = [t.get("pnl", 0) for t in trades if "pnl" in t]
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p < 0]

        win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0.0
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return BacktestResult(
            total_return=(pv.iloc[-1] / pv.iloc[0]) - 1 if len(pv) > 0 else 0.0,
            annualized_return=self._annualize_return(returns),
            volatility=self._calc_volatility(returns),
            sharpe_ratio=self._calc_sharpe(returns),
            sortino_ratio=self._calc_sortino(returns),
            max_drawdown=self._calc_max_drawdown(pv),
            num_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            portfolio_values=pv,
            returns=returns,
            trades=trades
        )

    def _get_active_signals(
        self,
        signals: pd.DataFrame,
        date: datetime
    ) -> pd.DataFrame:
        """Get signals active on a given date with decay applied."""
        lookback = timedelta(hours=self.config.signal_decay_hours)
        mask = (
            (signals["timestamp"] >= date - lookback) &
            (signals["timestamp"] <= date)
        )
        active = signals[mask].copy()

        if active.empty:
            return active

        # Apply time decay
        now = date
        active["hours_ago"] = (now - active["timestamp"]).dt.total_seconds() / 3600
        active["decay"] = np.exp(-active["hours_ago"] / self.config.signal_decay_hours)
        active["adjusted_signal"] = active["signal"] * active["confidence"] * active["decay"]

        # Aggregate by symbol
        aggregated = active.groupby("symbol").agg({
            "adjusted_signal": "sum",
            "confidence": "mean"
        }).reset_index()

        return aggregated

    def _calculate_targets(
        self,
        signals: pd.DataFrame,
        prices: pd.Series,
        capital: float
    ) -> Dict[str, float]:
        """Calculate target positions from signals."""
        if signals.empty:
            return {}

        targets = {}
        max_position = capital * self.config.max_position_pct

        for _, row in signals.iterrows():
            symbol = row["symbol"]
            if symbol not in prices.index:
                continue

            signal = row["adjusted_signal"]
            if abs(signal) < self.config.signal_threshold:
                continue

            price = prices[symbol]
            if price <= 0:
                continue

            position_value = signal * max_position
            shares = position_value / price
            targets[symbol] = shares

        return targets

    def _execute_trades(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        prices: pd.Series,
        capital: float,
        date: datetime
    ) -> Tuple[List[Dict], float]:
        """Execute rebalancing trades."""
        trades = []
        all_symbols = set(current.keys()) | set(target.keys())
        cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps

        for symbol in all_symbols:
            curr_shares = current.get(symbol, 0)
            tgt_shares = target.get(symbol, 0)
            delta = tgt_shares - curr_shares

            if abs(delta) < 0.01 or symbol not in prices.index:
                continue

            price = prices[symbol]
            cost_mult = 1 + cost_bps / 10000
            trade_value = abs(delta) * price

            if delta > 0:
                # Buying
                capital -= trade_value * cost_mult
                pnl = 0  # PnL calculated on close
            else:
                # Selling
                capital += trade_value / cost_mult
                # Simple PnL estimation
                pnl = -delta * price * 0.01  # Placeholder

            trades.append({
                "date": date,
                "symbol": symbol,
                "shares": delta,
                "price": price,
                "value": trade_value,
                "type": "BUY" if delta > 0 else "SELL",
                "pnl": pnl
            })

        return trades, capital

    def _annualize_return(self, returns: pd.Series) -> float:
        """Annualize returns assuming daily frequency."""
        if returns.empty:
            return 0.0
        total = (1 + returns).prod()
        n_years = len(returns) / 252
        return total ** (1 / n_years) - 1 if n_years > 0 else 0.0

    def _calc_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if returns.empty:
            return 0.0
        return returns.std() * np.sqrt(252)

    def _calc_sharpe(self, returns: pd.Series, rf: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if returns.empty or returns.std() == 0:
            return 0.0
        excess = returns - rf / 252
        return np.sqrt(252) * excess.mean() / excess.std()

    def _calc_sortino(self, returns: pd.Series, rf: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if returns.empty:
            return 0.0
        excess = returns - rf / 252
        downside = returns[returns < 0]
        if downside.empty or downside.std() == 0:
            return float('inf') if excess.mean() > 0 else 0.0
        return np.sqrt(252) * excess.mean() / downside.std()

    def _calc_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if values.empty:
            return 0.0
        peak = values.expanding().max()
        dd = (values - peak) / peak
        return dd.min()


def generate_mock_data(
    symbols: List[str] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    n_signals: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate mock signals and prices for demonstration.

    Args:
        symbols: List of symbols to generate data for
        start_date: Start date string
        end_date: End date string
        n_signals: Number of signals to generate

    Returns:
        Tuple of (signals DataFrame, prices DataFrame)
    """
    np.random.seed(42)

    symbols = symbols or ["AAPL", "NVDA", "TSLA", "MSFT", "BTC"]
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate price data
    prices = pd.DataFrame(index=dates)
    base_prices = {"AAPL": 180, "NVDA": 500, "TSLA": 250, "MSFT": 380, "BTC": 45000}

    for symbol in symbols:
        base = base_prices.get(symbol, 100)
        returns = np.random.randn(len(dates)) * 0.02
        prices[symbol] = base * (1 + returns).cumprod()

    # Generate signals
    signal_dates = np.random.choice(dates, size=n_signals, replace=False)
    signals = pd.DataFrame({
        "timestamp": pd.to_datetime(signal_dates),
        "symbol": np.random.choice(symbols, size=n_signals),
        "signal": np.random.uniform(-1, 1, size=n_signals),
        "confidence": np.random.uniform(0.6, 0.95, size=n_signals)
    })

    return signals, prices


if __name__ == "__main__":
    print("=" * 70)
    print("FinGPT Trading Strategy Backtest")
    print("=" * 70)

    # Generate test data
    signals, prices = generate_mock_data()

    # Configure and run backtest
    config = BacktestConfig(
        initial_capital=100000,
        max_position_pct=0.1,
        transaction_cost_bps=10,
        signal_threshold=0.3
    )

    backtester = FinGPTBacktester(config)
    result = backtester.run_backtest(signals, prices)

    print(f"\nBacktest Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Initial Capital: ${config.initial_capital:,.0f}")
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Number of Trades: {result.num_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")
    print(f"Final Portfolio Value: ${result.portfolio_values.iloc[-1]:,.0f}")
