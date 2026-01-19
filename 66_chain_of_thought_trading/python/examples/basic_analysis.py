#!/usr/bin/env python3
"""
Basic Chain-of-Thought Trading Analysis Example

This example demonstrates how to use the CoT trading system
to analyze a stock and generate an explainable trading signal.
"""

import sys
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, "..")

from cot_analyzer import MockChainOfThoughtAnalyzer
from signal_generator import MultiStepSignalGenerator, Signal
from data_loader import MockDataLoader, add_technical_indicators, prepare_for_analysis


def main():
    """Run basic CoT analysis example."""
    print("=" * 60)
    print("Chain-of-Thought Trading Analysis Example")
    print("=" * 60)

    # Step 1: Load data
    print("\n1. Loading market data...")
    loader = MockDataLoader(seed=42)

    # Load 6 months of daily data for AAPL
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    df = loader.load("AAPL", start_date, end_date, "1d")
    print(f"   Loaded {len(df)} data points")

    # Step 2: Add technical indicators
    print("\n2. Calculating technical indicators...")
    df = add_technical_indicators(df)
    print("   Indicators calculated: RSI, MACD, SMA, Bollinger Bands, ATR")

    # Step 3: Prepare data for analysis
    print("\n3. Preparing data for CoT analysis...")
    market_data = prepare_for_analysis(df, lookback=100)
    print(f"   Current price: ${market_data['current_price']:.2f}")
    print(f"   RSI: {market_data['rsi']:.1f}")
    print(f"   MACD: {market_data['macd']:.4f}")

    # Step 4: Run CoT analysis
    print("\n4. Running Chain-of-Thought analysis...")
    analyzer = MockChainOfThoughtAnalyzer()

    # Prepare analysis data dict
    analysis_data = {
        "price": market_data['current_price'],
        "change_24h": market_data['price_change_1d'],
        "rsi": market_data['rsi'],
        "macd": market_data['macd'],
        "macd_signal": market_data['macd_signal'],
        "sma_50": market_data['sma_50'],
        "sma_200": market_data.get('sma_200', market_data['sma_50'] * 0.98),
        "volume": market_data.get('volume', 1000000),
        "atr": market_data['atr'],
    }

    analysis = analyzer.analyze("AAPL", analysis_data)

    print("\n   Reasoning Chain:")
    for i, step in enumerate(analysis.steps, 1):
        print(f"   Step {i}: {step.title}")
        reasoning_preview = step.reasoning[:80] + "..." if len(step.reasoning) > 80 else step.reasoning
        print(f"           {reasoning_preview}")
        print(f"           Conclusion: {step.conclusion}")
        print()

    print(f"   Final Recommendation: {analysis.final_signal}")
    print(f"   Confidence: {analysis.confidence:.0%}")

    # Step 5: Generate signal using multi-step generator
    print("\n5. Generating trading signal...")
    generator = MultiStepSignalGenerator()

    # Prepare price data dict
    price_data = {
        'open': market_data.get('open', market_data['current_price'] * 0.99),
        'high': market_data.get('high', market_data['current_price'] * 1.01),
        'low': market_data.get('low', market_data['current_price'] * 0.98),
        'close': market_data['current_price'],
        'prev_close': market_data['current_price'] * (1 - market_data['price_change_1d'] / 100),
        'volume': market_data.get('volume', 1000000),
        'avg_volume': market_data.get('avg_volume', 1000000),
    }

    # Prepare indicators dict
    indicators = {
        'rsi': market_data['rsi'],
        'macd': market_data['macd'],
        'macd_signal': market_data['macd_signal'],
        'sma_20': market_data['sma_20'],
        'sma_50': market_data['sma_50'],
        'sma_200': market_data.get('sma_200', market_data['sma_50'] * 0.98),
        'atr': market_data['atr'],
    }

    signal = generator.generate_signal("AAPL", price_data, indicators)

    print(f"\n   Signal: {signal.signal.name}")
    print(f"   Confidence: {signal.confidence:.0%}")
    print(f"   Stop Loss: ${signal.stop_loss:.2f}")
    print(f"   Take Profit: ${signal.take_profit:.2f}")

    print("\n   Detailed Reasoning:")
    for i, step in enumerate(signal.reasoning_chain, 1):
        print(f"   {i}. {step}")

    # Step 6: Display final summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"""
    Symbol:         AAPL
    Current Price:  ${market_data['current_price']:.2f}
    Signal:         {signal.signal.name}
    Confidence:     {signal.confidence:.0%}

    Risk Management:
    - Stop Loss:    ${signal.stop_loss:.2f} ({(signal.stop_loss / market_data['current_price'] - 1) * 100:.1f}%)
    - Take Profit:  ${signal.take_profit:.2f} ({(signal.take_profit / market_data['current_price'] - 1) * 100:.1f}%)
    - Risk/Reward:  1:{abs((signal.take_profit - market_data['current_price']) / (market_data['current_price'] - signal.stop_loss)):.1f}
    """)

    print("Note: This is a demonstration using mock data and mock AI analysis.")
    print("For real trading, use actual data sources and LLM APIs.")


if __name__ == "__main__":
    main()
