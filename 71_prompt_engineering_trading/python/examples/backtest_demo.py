#!/usr/bin/env python3
"""
Backtesting Demo

Demonstrates backtesting LLM-generated trading signals
with comprehensive performance metrics.
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import LLMSignalBacktester, BacktestConfig, WalkForwardOptimizer
from signal_generator import TradingSignal, SignalDirection
from data_loader import MockDataLoader


async def basic_backtest_demo():
    """Basic backtesting demonstration."""
    print("=" * 60)
    print("Basic Backtest Demo")
    print("=" * 60)

    # Create historical signals
    base_time = datetime(2024, 1, 1, 9, 30)
    signals = [
        TradingSignal(
            symbol="AAPL",
            direction=SignalDirection.BUY,
            confidence=85,
            entry_price=185.00,
            stop_loss=178.00,
            take_profit=198.00,
            timeframe="1d",
            reasoning="Strong earnings momentum",
            timestamp=base_time
        ),
        TradingSignal(
            symbol="AAPL",
            direction=SignalDirection.SELL,
            confidence=72,
            entry_price=195.00,
            stop_loss=202.00,
            take_profit=182.00,
            timeframe="1d",
            reasoning="Resistance reached, taking profits",
            timestamp=base_time + timedelta(days=7)
        ),
        TradingSignal(
            symbol="MSFT",
            direction=SignalDirection.BUY,
            confidence=78,
            entry_price=380.00,
            stop_loss=365.00,
            take_profit=405.00,
            timeframe="1d",
            reasoning="AI momentum play",
            timestamp=base_time + timedelta(days=3)
        ),
        TradingSignal(
            symbol="GOOGL",
            direction=SignalDirection.BUY,
            confidence=65,
            entry_price=140.00,
            stop_loss=135.00,
            take_profit=150.00,
            timeframe="1d",
            reasoning="Undervalued relative to peers",
            timestamp=base_time + timedelta(days=5)
        ),
        TradingSignal(
            symbol="MSFT",
            direction=SignalDirection.HOLD,
            confidence=55,
            entry_price=390.00,
            timeframe="1d",
            reasoning="Wait for clearer direction",
            timestamp=base_time + timedelta(days=10)
        )
    ]

    # Load mock price data
    loader = MockDataLoader({
        "AAPL": 185.0,
        "MSFT": 380.0,
        "GOOGL": 140.0
    })

    start = base_time - timedelta(days=1)
    end = base_time + timedelta(days=30)

    price_data = {}
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        bars = await loader.get_ohlcv(symbol, "1d", start, end)
        price_data[symbol] = [bar.to_dict() for bar in bars]

    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.1,
        max_positions=3,
        commission_pct=0.001,
        slippage_pct=0.0005,
        use_stop_loss=True,
        use_take_profit=True
    )

    # Run backtest
    backtester = LLMSignalBacktester(config)
    result = await backtester.run(signals, price_data, verbose=True)

    # Display results
    print(f"\n--- Backtest Results ---")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Equity: ${result.equity_curve[-1]:,.2f}")
    print(f"Total Return: ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")
    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.1f}%")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Avg Win: ${result.avg_win:.2f}")
    print(f"  Avg Loss: ${result.avg_loss:.2f}")
    print(f"  Avg Holding Period: {result.avg_holding_period:.1f} hours")
    print(f"\nLLM Performance:")
    print(f"  Confidence-Return Correlation: {result.confidence_correlation:.3f}")

    return result


async def trade_analysis_demo(result):
    """Analyze individual trades."""
    print("\n" + "=" * 60)
    print("Trade-by-Trade Analysis")
    print("=" * 60)

    if not result.trades:
        print("No trades to analyze.")
        return

    print(f"\nAnalyzing {len(result.trades)} trades:\n")

    for i, trade in enumerate(result.trades, 1):
        pnl_emoji = "+" if trade.pnl > 0 else "-"
        direction = "LONG" if trade.direction == SignalDirection.BUY else "SHORT"

        print(f"Trade #{i}: {trade.symbol} ({direction})")
        print(f"  Entry: ${trade.entry_price:.2f} @ {trade.entry_time}")
        print(f"  Exit:  ${trade.exit_price:.2f} @ {trade.exit_time}")
        print(f"  Exit Reason: {trade.exit_reason}")
        print(f"  P&L: {pnl_emoji}${abs(trade.pnl):.2f} ({trade.pnl_pct:+.2f}%)")
        print(f"  LLM Confidence: {trade.llm_confidence}%")

        # Check if high confidence led to better results
        if trade.llm_confidence >= 75:
            confidence_label = "HIGH"
        elif trade.llm_confidence >= 50:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"

        outcome = "WIN" if trade.pnl > 0 else "LOSS"
        print(f"  Confidence Level: {confidence_label} -> {outcome}")
        print()

    # Confidence analysis
    print("\n--- Confidence Analysis ---")
    high_conf_trades = [t for t in result.trades if t.llm_confidence >= 75]
    med_conf_trades = [t for t in result.trades if 50 <= t.llm_confidence < 75]
    low_conf_trades = [t for t in result.trades if t.llm_confidence < 50]

    for label, trades in [("High (>=75%)", high_conf_trades),
                          ("Medium (50-74%)", med_conf_trades),
                          ("Low (<50%)", low_conf_trades)]:
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate = wins / len(trades) * 100
            avg_return = sum(t.pnl_pct for t in trades) / len(trades)
            print(f"{label}: {len(trades)} trades, {win_rate:.1f}% win rate, {avg_return:+.2f}% avg return")


async def crypto_backtest_demo():
    """Cryptocurrency backtesting demo."""
    print("\n" + "=" * 60)
    print("Cryptocurrency Backtest Demo (Bybit-style)")
    print("=" * 60)

    base_time = datetime(2024, 1, 1, 0, 0)

    # Crypto signals
    signals = [
        TradingSignal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            confidence=80,
            entry_price=42000.00,
            stop_loss=40000.00,
            take_profit=46000.00,
            timeframe="4h",
            reasoning="Breakout above resistance with volume",
            timestamp=base_time
        ),
        TradingSignal(
            symbol="ETH/USDT",
            direction=SignalDirection.BUY,
            confidence=75,
            entry_price=2300.00,
            stop_loss=2200.00,
            take_profit=2500.00,
            timeframe="4h",
            reasoning="Following BTC momentum",
            timestamp=base_time + timedelta(hours=4)
        ),
        TradingSignal(
            symbol="BTC/USDT",
            direction=SignalDirection.SELL,
            confidence=70,
            entry_price=45000.00,
            stop_loss=47000.00,
            take_profit=42000.00,
            timeframe="4h",
            reasoning="Overbought on multiple timeframes",
            timestamp=base_time + timedelta(days=3)
        )
    ]

    # Mock crypto price data
    loader = MockDataLoader({
        "BTC/USDT": 42000.0,
        "ETH/USDT": 2300.0
    })

    start = base_time - timedelta(hours=4)
    end = base_time + timedelta(days=7)

    price_data = {}
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        bars = await loader.get_ohlcv(symbol, "4h", start, end)
        price_data[symbol] = [bar.to_dict() for bar in bars]

    # Crypto-specific config (smaller position sizes, tighter risk)
    config = BacktestConfig(
        initial_capital=10000,  # Smaller account
        position_size_pct=0.05,  # 5% per trade
        max_positions=2,
        commission_pct=0.0006,  # Bybit taker fee
        slippage_pct=0.001,  # Higher slippage in crypto
        use_stop_loss=True,
        use_take_profit=True,
        max_drawdown_pct=0.15  # Tighter drawdown limit
    )

    backtester = LLMSignalBacktester(config)
    result = await backtester.run(signals, price_data)

    print(f"\n--- Crypto Backtest Results ---")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Total Return: {result.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Total Trades: {result.total_trades}")


async def equity_curve_demo(result):
    """Display equity curve (text-based)."""
    print("\n" + "=" * 60)
    print("Equity Curve (Text Visualization)")
    print("=" * 60)

    if len(result.equity_curve) < 2:
        print("Not enough data for equity curve.")
        return

    # Sample equity curve for display
    curve = result.equity_curve
    min_eq = min(curve)
    max_eq = max(curve)
    range_eq = max_eq - min_eq if max_eq != min_eq else 1

    # Normalize to 40 character width
    width = 40
    samples = min(20, len(curve))
    step = len(curve) // samples

    print(f"\nEquity Range: ${min_eq:,.0f} - ${max_eq:,.0f}")
    print()

    for i in range(0, len(curve), step):
        equity = curve[i]
        normalized = int((equity - min_eq) / range_eq * width)
        bar = "*" * max(1, normalized)
        print(f"  {i:3d} | {bar} ${equity:,.0f}")


async def main():
    """Run all backtest demos."""
    print("\n" + "#" * 60)
    print("  PROMPT ENGINEERING FOR TRADING - BACKTEST DEMO")
    print("#" * 60)

    # Run basic backtest and get result
    result = await basic_backtest_demo()

    # Analyze trades
    await trade_analysis_demo(result)

    # Crypto backtest
    await crypto_backtest_demo()

    # Equity curve
    await equity_curve_demo(result)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("1. LLM confidence can correlate with trade outcomes")
    print("2. Higher confidence signals may warrant larger position sizes")
    print("3. Self-consistency validation improves signal quality")
    print("4. Always backtest before deploying any strategy")
    print("\nNote: Past performance does not guarantee future results.")


if __name__ == "__main__":
    asyncio.run(main())
