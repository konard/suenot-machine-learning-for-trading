"""
Example 03: Backtesting LLM Trading Signals

This script demonstrates how to backtest trading strategies
based on LLM-generated signals.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_realistic_signals(
    symbols: list,
    dates: pd.DatetimeIndex,
    n_signals: int = 100
) -> pd.DataFrame:
    """
    Generate realistic-looking LLM signals for demonstration.

    The signals have some predictive power (not random) to show
    meaningful backtest results.
    """
    np.random.seed(42)

    # Generate signal timestamps
    signal_times = np.random.choice(dates, size=n_signals, replace=True)

    # Generate signals with some bias based on time patterns
    signals = []
    for ts in signal_times:
        symbol = np.random.choice(symbols)

        # Add some patterns to make signals somewhat predictive
        hour = ts.hour
        day = ts.dayofweek

        # Morning signals tend to be slightly positive
        base_signal = 0.1 if 9 <= hour <= 11 else -0.05 if 14 <= hour <= 16 else 0

        # Add randomness
        signal = base_signal + np.random.randn() * 0.4
        signal = np.clip(signal, -1, 1)

        # Confidence varies
        confidence = np.random.uniform(0.5, 0.95)

        signals.append({
            "timestamp": ts,
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "source": np.random.choice(["news", "earnings", "social"])
        })

    return pd.DataFrame(signals)


def generate_correlated_prices(
    symbols: list,
    dates: pd.DatetimeIndex,
    signals: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate price data that's partially correlated with signals.

    This creates a scenario where good signals should produce profits.
    """
    np.random.seed(42)

    n_days = len(dates)
    prices = {}

    for symbol in symbols:
        initial_price = np.random.uniform(50, 500)

        # Base random walk
        base_returns = np.random.randn(n_days) * 0.015

        # Add signal-based returns (partial correlation)
        symbol_signals = signals[signals["symbol"] == symbol]

        for _, sig in symbol_signals.iterrows():
            try:
                idx = dates.get_loc(sig["timestamp"])
                # Signal affects next day's return
                if idx + 1 < n_days:
                    signal_impact = sig["signal"] * sig["confidence"] * 0.02
                    base_returns[idx + 1] += signal_impact * 0.3  # 30% predictive power
            except KeyError:
                continue

        # Cumulative prices
        prices[symbol] = initial_price * (1 + base_returns).cumprod()

    return pd.DataFrame(prices, index=dates)


def main():
    """Run backtesting demonstration."""
    print("=" * 60)
    print("LLM Signal Backtesting Demo")
    print("=" * 60)

    from backtest import (
        LLMSignalBacktester,
        BacktestConfig,
        print_backtest_report
    )

    # Setup
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")

    print(f"\nBacktest Configuration:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period: {dates[0].date()} to {dates[-1].date()}")
    print(f"  Trading Days: {len(dates)}")

    # Generate data
    print("\nGenerating synthetic data...")
    signals_df = generate_realistic_signals(symbols, dates, n_signals=200)
    prices_df = generate_correlated_prices(symbols, dates, signals_df)

    print(f"  Generated {len(signals_df)} signals")
    print(f"  Signal distribution:")
    for symbol in symbols:
        n = len(signals_df[signals_df["symbol"] == symbol])
        print(f"    {symbol}: {n} signals")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        max_position_size=0.15,
        max_total_exposure=0.8,
        transaction_cost_bps=10,
        slippage_bps=5,
        signal_decay_hours=48,
        rebalance_frequency="daily",
        min_signal_confidence=0.6,
        short_selling=True
    )

    print(f"\nBacktest Settings:")
    print(f"  Initial Capital: ${config.initial_capital:,.2f}")
    print(f"  Max Position Size: {config.max_position_size:.0%}")
    print(f"  Transaction Costs: {config.transaction_cost_bps}bps")
    print(f"  Signal Decay: {config.signal_decay_hours}h")
    print(f"  Short Selling: {'Enabled' if config.short_selling else 'Disabled'}")

    # Run backtest
    print("\nRunning backtest...")
    backtester = LLMSignalBacktester(config)
    result = backtester.run_backtest(signals_df, prices_df)

    # Print results
    print_backtest_report(result)

    # Additional analysis
    if result.trades:
        print("\nTRADE ANALYSIS")
        print("-" * 40)

        # Trades by symbol
        trade_df = pd.DataFrame([
            {"symbol": t.symbol, "type": t.type, "value": t.value}
            for t in result.trades
        ])

        trades_by_symbol = trade_df.groupby("symbol").agg({
            "value": ["count", "sum"]
        })
        trades_by_symbol.columns = ["num_trades", "total_value"]
        print("\nTrades by Symbol:")
        print(trades_by_symbol.to_string())

        # Buy vs Sell
        buys = len([t for t in result.trades if t.type == "BUY"])
        sells = len([t for t in result.trades if t.type == "SELL"])
        print(f"\nTrade Direction: {buys} buys, {sells} sells")

    # Compare to buy and hold
    print("\n" + "-" * 40)
    print("BENCHMARK COMPARISON")
    print("-" * 40)

    # Equal weight buy and hold
    equal_weight_returns = prices_df.pct_change().mean(axis=1)
    bh_total_return = (1 + equal_weight_returns).prod() - 1
    bh_vol = equal_weight_returns.std() * np.sqrt(252)
    bh_sharpe = (bh_total_return * 252 / len(equal_weight_returns) - 0.02) / bh_vol if bh_vol > 0 else 0

    print(f"\nEqual Weight Buy & Hold:")
    print(f"  Total Return: {bh_total_return:.2%}")
    print(f"  Volatility: {bh_vol:.2%}")
    print(f"  Sharpe Ratio: {bh_sharpe:.2f}")

    if result.metrics:
        strategy_return = result.metrics.get("total_return", 0)
        strategy_sharpe = result.metrics.get("sharpe_ratio", 0)

        alpha = strategy_return - bh_total_return
        print(f"\nLLM Strategy vs Benchmark:")
        print(f"  Alpha: {alpha:+.2%}")
        print(f"  Sharpe Improvement: {strategy_sharpe - bh_sharpe:+.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
