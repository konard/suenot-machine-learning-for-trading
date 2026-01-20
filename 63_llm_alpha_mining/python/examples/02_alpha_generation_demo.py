#!/usr/bin/env python3
"""
Example 02: Alpha Factor Generation Demo

This example demonstrates how to use LLMs to generate alpha factors
for cryptocurrency trading.

Run with: python 02_alpha_generation_demo.py
"""

import sys
sys.path.insert(0, '..')

from data_loader import generate_synthetic_data
from alpha_generator import (
    AlphaGenerator,
    AlphaExpressionParser,
    AlphaFactor,
    PREDEFINED_FACTORS
)


def main():
    print("=" * 60)
    print("LLM Alpha Mining - Alpha Generation Demo")
    print("=" * 60)

    # Load data
    print("\n1. LOADING DATA")
    print("-" * 40)

    data = generate_synthetic_data(["BTCUSDT"], days=200, seed=42)
    btc_data = data["BTCUSDT"].ohlcv
    print(f"Loaded {len(btc_data)} records for BTCUSDT")

    # Initialize generator (mock mode - no API key needed)
    print("\n2. INITIALIZING GENERATOR")
    print("-" * 40)

    generator = AlphaGenerator(model="mock")
    parser = AlphaExpressionParser()

    print("Generator initialized in mock mode")
    print("(Use model='openai' or 'anthropic' with API key for real LLM)")

    # Generate factors using different prompts
    print("\n3. GENERATING FACTORS")
    print("-" * 40)

    prompt_types = ["basic", "momentum", "volatility", "crypto_specific"]

    all_factors = []
    for prompt_type in prompt_types:
        print(f"\n{prompt_type.upper()} factors:")
        factors = generator.generate(
            btc_data,
            prompt_type=prompt_type,
            symbol="BTCUSDT"
        )

        for factor in factors:
            print(f"  - {factor.name}")
            print(f"    Expression: {factor.expression}")
            print(f"    Description: {factor.description}")
            all_factors.append(factor)

    # Evaluate generated factors
    print("\n4. EVALUATING FACTORS")
    print("-" * 40)

    for factor in all_factors[:5]:  # First 5 factors
        try:
            values = parser.evaluate(factor.expression, btc_data)
            print(f"\n{factor.name}:")
            print(f"  Valid values: {values.notna().sum()}/{len(values)}")
            print(f"  Mean: {values.mean():.6f}")
            print(f"  Std: {values.std():.6f}")
            print(f"  Min: {values.min():.6f}")
            print(f"  Max: {values.max():.6f}")
        except Exception as e:
            print(f"\n{factor.name}: Error - {e}")

    # Use predefined factors
    print("\n5. PREDEFINED FACTORS")
    print("-" * 40)

    for factor in PREDEFINED_FACTORS:
        print(f"\n{factor.name}:")
        print(f"  Expression: {factor.expression}")
        print(f"  Description: {factor.description}")
        print(f"  Confidence: {factor.confidence}")

        # Evaluate
        try:
            values = parser.evaluate(factor.expression, btc_data)
            print(f"  Sample values (last 5): {values.dropna().tail().values.round(4)}")
        except Exception as e:
            print(f"  Error: {e}")

    # Generate from natural language
    print("\n6. NATURAL LANGUAGE GENERATION")
    print("-" * 40)

    descriptions = [
        "A trend following strategy that buys when momentum is strong",
        "A mean reversion strategy for oversold conditions",
        "A volume-based breakout detector",
    ]

    for desc in descriptions:
        print(f"\nDescription: '{desc}'")
        factors = generator.generate_from_description(desc, btc_data, "BTCUSDT")

        for factor in factors:
            print(f"  Generated: {factor.name}")
            print(f"  Expression: {factor.expression}")

    # Expression validation
    print("\n7. EXPRESSION VALIDATION")
    print("-" * 40)

    test_expressions = [
        "ts_mean(close, 20)",  # Valid
        "ts_delta(close, 5) / ts_delay(close, 5)",  # Valid
        "rank(close) * volume",  # Valid
        "import os; os.system('ls')",  # Invalid - dangerous
        "eval('print(1)')",  # Invalid - dangerous
        "unknown_function(close)",  # Invalid - unknown function
    ]

    for expr in test_expressions:
        is_valid = parser.validate(expr)
        status = "VALID" if is_valid else "INVALID"
        print(f"  [{status}] {expr[:50]}...")

    # Combine multiple factors
    print("\n8. FACTOR COMBINATION")
    print("-" * 40)

    # Create a composite factor
    factor1 = parser.evaluate("ts_delta(close, 5) / ts_delay(close, 5)", btc_data)
    factor2 = parser.evaluate("-1 * (close - ts_mean(close, 20)) / ts_std(close, 20)", btc_data)
    factor3 = parser.evaluate("volume / ts_mean(volume, 20) - 1", btc_data)

    # Equal-weighted combination
    composite = (factor1.rank(pct=True) + factor2.rank(pct=True) + factor3.rank(pct=True)) / 3

    print("\nComposite factor (equal-weighted ranks):")
    print(f"  Components: momentum_5d, mean_reversion, volume_breakout")
    print(f"  Valid values: {composite.notna().sum()}/{len(composite)}")
    print(f"  Distribution: min={composite.min():.2f}, median={composite.median():.2f}, max={composite.max():.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
