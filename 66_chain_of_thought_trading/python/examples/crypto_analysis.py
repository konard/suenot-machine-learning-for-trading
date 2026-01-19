#!/usr/bin/env python3
"""
Cryptocurrency Analysis with Chain-of-Thought Trading

This example demonstrates how to analyze cryptocurrency data
from Bybit using the CoT trading system.
"""

import sys
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, "..")

from cot_analyzer import MockChainOfThoughtAnalyzer
from signal_generator import MultiStepSignalGenerator
from position_sizer import CoTPositionSizer
from data_loader import MockDataLoader, add_technical_indicators, prepare_for_analysis


def analyze_crypto(symbol: str, portfolio_value: float = 100000.0):
    """
    Analyze a cryptocurrency and generate a trading plan.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        portfolio_value: Current portfolio value in USD
    """
    print(f"\n{'=' * 60}")
    print(f"Analyzing {symbol}")
    print("=" * 60)

    # Load mock data (in production, use BybitLoader)
    loader = MockDataLoader(seed=hash(symbol) % 1000)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    df = loader.load(symbol, start_date, end_date, "1d")
    df = add_technical_indicators(df)
    data = prepare_for_analysis(df, lookback=50)

    print(f"\nMarket Data:")
    print(f"  Current Price:   ${data['current_price']:,.2f}")
    print(f"  24h Change:      {data['price_change_1d']:.2f}%")
    print(f"  5d Change:       {data['price_change_5d']:.2f}%")
    print(f"  20d Change:      {data['price_change_20d']:.2f}%")
    print(f"  RSI (14):        {data['rsi']:.1f}")
    print(f"  Volume Ratio:    {data['volume_ratio']:.2f}x")

    # Initialize components
    signal_gen = MultiStepSignalGenerator()
    position_sizer = CoTPositionSizer(
        account_size=portfolio_value,
        max_risk_per_trade=0.02,   # Max 2% risk per trade
        max_position_size=0.1,     # Max 10% per trade
    )

    # Prepare price data dict
    price_data = {
        'open': data.get('open', data['current_price'] * 0.99),
        'high': data.get('high', data['current_price'] * 1.01),
        'low': data.get('low', data['current_price'] * 0.98),
        'close': data['current_price'],
        'prev_close': data['current_price'] * (1 - data['price_change_1d'] / 100),
        'volume': data.get('volume', 1000000),
        'avg_volume': data.get('avg_volume', 1000000),
    }

    # Prepare indicators dict
    indicators = {
        'rsi': data['rsi'],
        'macd': data['macd'],
        'macd_signal': data['macd_signal'],
        'sma_20': data['sma_20'],
        'sma_50': data['sma_50'],
        'sma_200': data.get('sma_200', data['sma_50'] * 0.98),
        'atr': data['atr'],
    }

    # Generate trading signal
    print("\nGenerating Trading Signal...")
    signal = signal_gen.generate_signal(symbol, price_data, indicators)

    print(f"\n  Signal:          {signal.signal.name}")
    print(f"  Confidence:      {signal.confidence:.0%}")

    # Calculate position size
    print("\nCalculating Position Size...")
    position = position_sizer.calculate_position_size(
        symbol=symbol,
        entry_price=data["current_price"],
        stop_loss=signal.stop_loss,
        signal_confidence=signal.confidence,
        volatility=data["atr"] / data["current_price"],
    )

    print(f"\n  Position Size:   ${position.size_in_units * data['current_price']:,.2f}")
    print(f"  Units:           {position.size_in_units:.6f}")
    print(f"  Risk Amount:     ${position.risk_amount:,.2f}")
    print(f"  Portfolio %:     {position.recommended_size:.1%}")

    # Display reasoning
    print("\nReasoning Chain:")
    for i, step in enumerate(signal.reasoning_chain[:5], 1):
        print(f"  {i}. {step}")

    print("\nPosition Sizing Reasoning:")
    for i, step in enumerate(position.reasoning_chain[:5], 1):
        print(f"  {i}. {step}")

    return {
        "symbol": symbol,
        "signal": signal,
        "position": position,
        "data": data,
    }


def main():
    """Run multi-crypto analysis example."""
    print("=" * 60)
    print("Chain-of-Thought Cryptocurrency Analysis")
    print("=" * 60)

    portfolio_value = 100000.0
    print(f"\nPortfolio Value: ${portfolio_value:,.2f}")

    # Analyze multiple cryptocurrencies
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    results = []

    for symbol in symbols:
        result = analyze_crypto(symbol, portfolio_value)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("Portfolio Summary")
    print("=" * 60)
    print(f"\n{'Symbol':<12} {'Signal':<12} {'Confidence':<12} {'Position':<15} {'Risk':<12}")
    print("-" * 63)

    total_position = 0
    total_risk = 0

    for r in results:
        signal_name = r["signal"].signal.name
        confidence = r["signal"].confidence
        position_val = r["position"].size_in_units * r["data"]["current_price"]
        risk_val = r["position"].risk_amount

        print(f"{r['symbol']:<12} {signal_name:<12} {confidence:>10.0%} ${position_val:>12,.2f} ${risk_val:>10,.2f}")

        total_position += position_val
        total_risk += risk_val

    print("-" * 63)
    print(f"{'TOTAL':<12} {'':<12} {'':<12} ${total_position:>12,.2f} ${total_risk:>10,.2f}")
    print(f"\nTotal Allocation: {total_position / portfolio_value:.1%} of portfolio")
    print(f"Total Risk:       {total_risk / portfolio_value:.2%} of portfolio")

    print("\n" + "=" * 60)
    print("Note: This uses mock data. For real trading:")
    print("  1. Use BybitLoader for real-time data")
    print("  2. Configure with your API keys")
    print("  3. Use a real LLM (OpenAI/Anthropic) for analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()
