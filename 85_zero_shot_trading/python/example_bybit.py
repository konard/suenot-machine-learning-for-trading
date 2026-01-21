#!/usr/bin/env python3
"""
Bybit example: Live data analysis with the zero-shot model.

This example demonstrates:
1. Fetching live kline data from Bybit
2. Preparing features from market data
3. Predicting regimes for multiple assets
4. Generating trading signals

Run: python example_bybit.py

Note: Requires internet connection and Bybit API access.
"""

import asyncio
import numpy as np
import torch
from datetime import datetime
from zero_shot_trading import (
    MarketEncoder,
    AttributeEncoder,
    ZeroShotTradingModel,
    ZeroShotTradingStrategy,
    BybitZeroShotClient,
    prepare_features,
    MarketRegime,
)


async def main():
    print("Zero-Shot Trading: Bybit Live Data Example")
    print("=" * 60)
    print()

    # Create model
    input_features = 11
    embedding_dim = 64
    num_regimes = 5

    market_encoder = MarketEncoder(
        input_dim=input_features,
        hidden_dim=128,
        embedding_dim=embedding_dim
    )

    attribute_encoder = AttributeEncoder(
        categorical_dims=[5, 4, 4, 15],
        numerical_dim=4,
        embedding_dim=embedding_dim
    )

    model = ZeroShotTradingModel(
        market_encoder=market_encoder,
        attribute_encoder=attribute_encoder,
        embedding_dim=embedding_dim,
        num_regimes=num_regimes
    )

    model.eval()
    print("✓ Model initialized")

    # Create Bybit client
    client = BybitZeroShotClient()
    print("✓ Bybit client initialized")

    # Create trading strategy
    strategy = ZeroShotTradingStrategy(
        model=model,
        confidence_threshold=0.5,
        position_sizing="regime_based"
    )
    print("✓ Trading strategy initialized")
    print()

    # Symbols to analyze
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    interval = "15"  # 15-minute candles
    limit = 200

    print("═" * 60)
    print("  FETCHING & ANALYZING MARKET DATA")
    print("═" * 60)
    print()

    results = []

    for symbol in symbols:
        print(f"{'─' * 60}")
        print(f"  {symbol}")
        print(f"{'─' * 60}")

        try:
            # Fetch kline data
            df = await client.fetch_klines(symbol, interval, limit)
            print(f"  Fetched {len(df)} klines")

            # Current price
            current_price = df["close"].iloc[-1]
            price_change_24h = (df["close"].iloc[-1] / df["close"].iloc[-96] - 1) * 100
            direction = "↑" if price_change_24h >= 0 else "↓"
            print(f"  Price: ${current_price:,.2f} ({direction} {abs(price_change_24h):.2f}%)")

            # Prepare features
            features = prepare_features(df)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            print(f"  Features: {features_tensor.shape[1]} timesteps × {features_tensor.shape[2]} features")

            # Get prediction
            with torch.no_grad():
                market_embedding = model.market_encoder(features_tensor)
                regime_logits = model.regime_head(market_embedding)
                probs = torch.softmax(regime_logits, dim=-1).squeeze()
                predicted_idx = torch.argmax(probs).item()
                regime = MarketRegime(predicted_idx)
                confidence = probs[predicted_idx].item()

            print()
            print(f"  📊 PREDICTION:")
            print(f"     Regime: {regime.name}")
            print(f"     Confidence: {confidence:.1%}")

            # Generate trading signal
            signal = strategy.generate_signal(regime, confidence, current_price)

            print()
            print(f"  📈 TRADING SIGNAL:")
            print(f"     Action: {signal['action']}")
            print(f"     Position Size: {signal['position_size']:.0%}")

            # Store result
            results.append({
                "symbol": symbol,
                "price": current_price,
                "change_24h": price_change_24h,
                "regime": regime.name,
                "confidence": confidence,
                "signal": signal["action"],
            })

            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()

    # Summary table
    print("═" * 60)
    print("  ANALYSIS SUMMARY")
    print("═" * 60)
    print()
    print(f"{'Symbol':<10} {'Price':>12} {'24h':>8} {'Regime':<18} {'Signal':<12}")
    print("─" * 60)

    for r in results:
        direction = "↑" if r["change_24h"] >= 0 else "↓"
        print(f"{r['symbol']:<10} ${r['price']:>10,.2f} "
              f"{direction} {abs(r['change_24h']):>5.1f}% "
              f"{r['regime']:<18} {r['signal']:<12}")

    print()

    # Calculate portfolio allocation
    print("═" * 60)
    print("  SUGGESTED PORTFOLIO ALLOCATION")
    print("═" * 60)
    print()

    total_long = sum(1 for r in results if "BUY" in r["signal"])
    total_short = sum(1 for r in results if "SELL" in r["signal"])
    total_neutral = len(results) - total_long - total_short

    print(f"  Long positions: {total_long}")
    print(f"  Short positions: {total_short}")
    print(f"  Neutral/Cash: {total_neutral}")
    print()

    if results:
        # Weighted allocation based on confidence
        total_confidence = sum(r["confidence"] for r in results if "BUY" in r["signal"] or "SELL" in r["signal"])

        print("  Allocation (by confidence):")
        for r in results:
            if "BUY" in r["signal"] or "SELL" in r["signal"]:
                weight = r["confidence"] / total_confidence if total_confidence > 0 else 0
                direction = "LONG" if "BUY" in r["signal"] else "SHORT"
                print(f"    {r['symbol']}: {weight:.1%} ({direction})")

    print()
    print("═" * 60)
    print(f"  Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)


if __name__ == "__main__":
    asyncio.run(main())
