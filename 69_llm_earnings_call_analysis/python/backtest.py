"""
Backtesting Framework for LLM Earnings Call Analysis Strategy

This module provides tools for backtesting trading strategies
based on earnings call analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum

from earnings_analyzer import TradingSignal, SignalDirection


@dataclass
class Trade:
    """A single trade in the backtest"""
    date: datetime
    symbol: str
    direction: str
    signal_strength: float
    confidence: float
    entry_price: float
    exit_price: float
    trade_return: float
    portfolio_return: float
    portfolio_value: float


@dataclass
class BacktestResult:
    """Results from backtesting earnings strategy"""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return_per_trade: float
    num_trades: int
    trades: List[Trade]


class EarningsBacktest:
    """
    Backtest LLM earnings call analysis strategy

    Simulates trading based on earnings call signals and
    calculates performance metrics.
    """

    def __init__(self,
                 hold_period: int = 5,
                 position_size: float = 0.1,
                 signal_threshold: float = 0.5,
                 trading_cost: float = 0.001):
        """
        Initialize backtest

        Args:
            hold_period: Number of days to hold position after earnings
            position_size: Base position size as fraction of portfolio
            signal_threshold: Minimum signal strength to trade
            trading_cost: Transaction cost as fraction (e.g., 0.001 = 0.1%)
        """
        self.hold_period = hold_period
        self.position_size = position_size
        self.signal_threshold = signal_threshold
        self.trading_cost = trading_cost

    def run(self,
            signals: List[TradingSignal],
            earnings_dates: List[datetime],
            price_data: pd.DataFrame,
            symbols: List[str]) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            signals: List of trading signals from earnings analysis
            earnings_dates: Dates of earnings calls
            price_data: DataFrame with price columns for each symbol
            symbols: List of stock/crypto symbols

        Returns:
            BacktestResult with performance metrics
        """
        trades = []
        portfolio_value = 1.0
        peak_value = 1.0
        max_drawdown = 0.0

        for i, (signal, date, symbol) in enumerate(zip(signals, earnings_dates, symbols)):
            # Skip weak signals
            if signal.strength < self.signal_threshold:
                continue

            # Skip if signal direction is neutral
            if signal.direction == SignalDirection.NEUTRAL:
                continue

            # Get entry and exit prices
            entry_date = date + timedelta(days=1)  # Enter day after earnings
            exit_date = entry_date + timedelta(days=self.hold_period)

            try:
                # Handle both DataFrame column access patterns
                if symbol in price_data.columns:
                    entry_price = price_data.loc[entry_date, symbol]
                    exit_price = price_data.loc[exit_date, symbol]
                else:
                    # Try to find nearest available date
                    entry_idx = price_data.index.get_indexer([entry_date], method='nearest')[0]
                    exit_idx = price_data.index.get_indexer([exit_date], method='nearest')[0]

                    if symbol in price_data.columns:
                        entry_price = price_data.iloc[entry_idx][symbol]
                        exit_price = price_data.iloc[exit_idx][symbol]
                    else:
                        continue

            except (KeyError, IndexError):
                continue  # Skip if dates not in data

            # Calculate return based on direction
            if signal.direction == SignalDirection.BULLISH:
                trade_return = (exit_price - entry_price) / entry_price
            else:  # BEARISH
                trade_return = (entry_price - exit_price) / entry_price

            # Subtract trading costs
            trade_return -= 2 * self.trading_cost  # Entry + exit

            # Scale by position size and signal strength
            position = self.position_size * signal.strength * signal.confidence
            portfolio_return = trade_return * position

            portfolio_value *= (1 + portfolio_return)

            # Track drawdown
            peak_value = max(peak_value, portfolio_value)
            drawdown = (peak_value - portfolio_value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)

            trades.append(Trade(
                date=date,
                symbol=symbol,
                direction=signal.direction.value,
                signal_strength=signal.strength,
                confidence=signal.confidence,
                entry_price=entry_price,
                exit_price=exit_price,
                trade_return=trade_return,
                portfolio_return=portfolio_return,
                portfolio_value=portfolio_value
            ))

        # Calculate metrics
        if len(trades) > 0:
            returns = [t.portfolio_return for t in trades]
            total_return = portfolio_value - 1

            # Annualized Sharpe (assuming ~4 earnings per year per stock)
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / self.hold_period)
            else:
                sharpe = 0.0

            # Sortino (downside deviation)
            downside = [r for r in returns if r < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                sortino = np.mean(returns) / np.std(downside) * np.sqrt(252 / self.hold_period)
            else:
                sortino = 0.0

            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            avg_return = np.mean(returns)
        else:
            total_return = 0
            sharpe = 0
            sortino = 0
            win_rate = 0
            avg_return = 0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_return_per_trade=avg_return,
            num_trades=len(trades),
            trades=trades
        )

    def calculate_metrics(self, results: BacktestResult) -> Dict:
        """
        Calculate detailed performance metrics

        Args:
            results: BacktestResult from run()

        Returns:
            Dictionary of performance metrics
        """
        trades = results.trades

        if len(trades) == 0:
            return {'error': 'No trades executed'}

        # Direction analysis
        bullish_trades = [t for t in trades if t.direction == 'bullish']
        bearish_trades = [t for t in trades if t.direction == 'bearish']

        metrics = {
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate,
            'num_trades': results.num_trades,
            'avg_return_per_trade': results.avg_return_per_trade,

            # Directional breakdown
            'bullish_trades': len(bullish_trades),
            'bearish_trades': len(bearish_trades),
            'bullish_win_rate': (
                sum(1 for t in bullish_trades if t.trade_return > 0) / len(bullish_trades)
                if len(bullish_trades) > 0 else 0
            ),
            'bearish_win_rate': (
                sum(1 for t in bearish_trades if t.trade_return > 0) / len(bearish_trades)
                if len(bearish_trades) > 0 else 0
            ),

            # Signal quality
            'avg_signal_strength': np.mean([t.signal_strength for t in trades]),
            'avg_confidence': np.mean([t.confidence for t in trades]),

            # Return distribution
            'best_trade': max(t.trade_return for t in trades),
            'worst_trade': min(t.trade_return for t in trades),
            'return_std': np.std([t.trade_return for t in trades]),

            # Correlation between signal and return
            'signal_return_correlation': np.corrcoef(
                [t.signal_strength for t in trades],
                [t.trade_return for t in trades]
            )[0, 1] if len(trades) > 1 else 0
        }

        return metrics


def generate_sample_data(num_earnings: int = 50,
                        start_date: datetime = None) -> tuple:
    """
    Generate sample data for backtesting demonstration

    Args:
        num_earnings: Number of earnings events to generate
        start_date: Starting date for the data

    Returns:
        Tuple of (signals, dates, price_data, symbols)
    """
    if start_date is None:
        start_date = datetime(2023, 1, 1)

    np.random.seed(42)

    # Generate random signals
    signals = []
    dates = []
    symbols = []

    symbol_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']

    for i in range(num_earnings):
        # Random signal
        sentiment = np.random.uniform(-1, 1)

        if sentiment > 0.3:
            direction = SignalDirection.BULLISH
        elif sentiment < -0.3:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        signal = TradingSignal(
            direction=direction,
            strength=abs(sentiment),
            confidence=np.random.uniform(0.5, 1.0),
            sentiment_score=sentiment,
            confidence_level=np.random.uniform(0.4, 0.9),
            guidance_direction=np.random.choice(['raised', 'maintained', 'lowered']),
            qa_quality=np.random.uniform(0.4, 0.9),
            reasoning="Sample signal"
        )
        signals.append(signal)

        # Random date (spread across quarters)
        date = start_date + timedelta(days=i * 7)
        dates.append(date)

        # Random symbol
        symbols.append(np.random.choice(symbol_list))

    # Generate price data
    date_range = pd.date_range(
        start=start_date - timedelta(days=5),
        end=start_date + timedelta(days=num_earnings * 7 + 30),
        freq='D'
    )

    price_data = pd.DataFrame(index=date_range)

    for symbol in symbol_list:
        # Generate random walk prices
        initial_price = np.random.uniform(100, 500)
        returns = np.random.normal(0.0005, 0.02, len(date_range))

        prices = [initial_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))

        price_data[symbol] = prices

    return signals, dates, price_data, symbols


def run_sample_backtest():
    """Run a sample backtest with generated data"""

    print("=== LLM Earnings Call Analysis Backtest ===\n")

    # Generate sample data
    print("Generating sample data...")
    signals, dates, price_data, symbols = generate_sample_data(num_earnings=100)

    # Initialize backtest
    backtest = EarningsBacktest(
        hold_period=5,
        position_size=0.1,
        signal_threshold=0.4
    )

    # Run backtest
    print("Running backtest...\n")
    results = backtest.run(signals, dates, price_data, symbols)

    # Calculate metrics
    metrics = backtest.calculate_metrics(results)

    # Print results
    print("=== Backtest Results ===")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Number of Trades: {metrics['num_trades']}")

    print("\n=== Directional Analysis ===")
    print(f"Bullish Trades: {metrics['bullish_trades']}")
    print(f"Bearish Trades: {metrics['bearish_trades']}")
    print(f"Bullish Win Rate: {metrics['bullish_win_rate']*100:.1f}%")
    print(f"Bearish Win Rate: {metrics['bearish_win_rate']*100:.1f}%")

    print("\n=== Signal Quality ===")
    print(f"Avg Signal Strength: {metrics['avg_signal_strength']:.2f}")
    print(f"Avg Confidence: {metrics['avg_confidence']:.2f}")
    print(f"Signal-Return Correlation: {metrics['signal_return_correlation']:.3f}")

    print("\n=== Trade Distribution ===")
    print(f"Best Trade: {metrics['best_trade']*100:.2f}%")
    print(f"Worst Trade: {metrics['worst_trade']*100:.2f}%")
    print(f"Return Std Dev: {metrics['return_std']*100:.2f}%")

    return results, metrics


if __name__ == "__main__":
    run_sample_backtest()
