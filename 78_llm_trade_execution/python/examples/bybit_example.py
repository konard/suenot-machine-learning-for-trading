#!/usr/bin/env python3
"""
Bybit exchange integration example.

This example demonstrates how to:
1. Connect to Bybit API
2. Fetch market data
3. Analyze order book for execution
"""

import asyncio
from llm_trade_execution import (
    BybitClient,
    BybitConfig,
    TimeFrame,
    MarketImpactEstimator,
)


async def main():
    print("=== Bybit Integration Example ===\n")

    # Create a Bybit client for public endpoints (no auth needed)
    config = BybitConfig()  # Use mainnet
    # For testnet: config = BybitConfig.testnet_config()

    symbol = "BTCUSDT"

    async with BybitClient(config) as client:
        # Example 1: Fetch current ticker
        print("--- Fetching Ticker ---")
        try:
            ticker = await client.get_ticker(symbol)
            print(f"Symbol: {ticker.symbol}")
            print(f"Last Price: {ticker.last_price:.2f}")
            print(f"Bid: {ticker.bid_price:.2f} (qty: {ticker.bid_qty:.4f})")
            print(f"Ask: {ticker.ask_price:.2f} (qty: {ticker.ask_qty:.4f})")
            print(f"Spread: {ticker.spread():.2f} ({ticker.spread_bps():.2f} bps)")
            print(f"24h High: {ticker.high_24h:.2f}")
            print(f"24h Low: {ticker.low_24h:.2f}")
            print(f"24h Volume: {ticker.volume_24h:.2f}")

            if ticker.open_interest:
                print(f"Open Interest: {ticker.open_interest:.2f}")
            if ticker.funding_rate:
                print(f"Funding Rate: {ticker.funding_rate * 100:.4f}%")
        except Exception as e:
            print(f"Failed to fetch ticker: {e}")
            print("(This is expected if running without network access)")
        print()

        # Example 2: Fetch order book and analyze
        print("--- Fetching Order Book ---")
        try:
            book = await client.get_orderbook(symbol, limit=25)
            print(f"Order Book for {symbol}")
            print(f"Best Bid: {book.best_bid():.2f}")
            print(f"Best Ask: {book.best_ask():.2f}")
            print(f"Mid Price: {book.mid_price():.2f}")
            print(f"Spread: {book.spread_bps():.2f} bps")
            print(f"Bid Depth (10 levels): {book.bid_depth(10):.4f}")
            print(f"Ask Depth (10 levels): {book.ask_depth(10):.4f}")
            print(f"Imbalance: {book.imbalance(10):.2f}")

            # Estimate impact for different order sizes
            print("\n--- Impact Estimation ---")
            estimator = MarketImpactEstimator.crypto()

            for qty in [0.1, 0.5, 1.0, 5.0, 10.0]:
                estimate = estimator.estimate(qty, 1.0, book)
                print(
                    f"  {qty:.1f} BTC: {estimate.impact.as_bps():.2f} bps total "
                    f"({estimate.impact.permanent * 10000:.2f} permanent + "
                    f"{estimate.impact.temporary * 10000:.2f} temporary)"
                )

            # Direct order book impact
            print("\n--- Direct Book Impact ---")
            for qty in [0.1, 0.5, 1.0]:
                result = book.buy_impact(qty)
                if result:
                    avg_price, impact = result
                    print(
                        f"  Buy {qty:.1f} BTC: avg price {avg_price:.2f}, "
                        f"impact {impact * 100:.4f}%"
                    )
        except Exception as e:
            print(f"Failed to fetch order book: {e}")
        print()

        # Example 3: Fetch historical klines
        print("--- Fetching Historical Data ---")
        try:
            bars = await client.get_klines(symbol, TimeFrame.H1, limit=24)
            print(f"Got {len(bars)} hourly bars")

            if bars:
                volumes = [b.volume for b in bars]
                avg_volume = sum(volumes) / len(volumes)

                ranges = [b.range() for b in bars]
                avg_range = sum(ranges) / len(ranges)

                print(f"Average hourly volume: {avg_volume:.2f}")
                print(f"Average hourly range: {avg_range:.2f}")

                # Show latest bar
                latest = bars[-1]
                print(f"\nLatest bar:")
                print(f"  Time: {latest.timestamp}")
                print(f"  Open: {latest.open:.2f}")
                print(f"  High: {latest.high:.2f}")
                print(f"  Low: {latest.low:.2f}")
                print(f"  Close: {latest.close:.2f}")
                print(f"  Volume: {latest.volume:.2f}")
        except Exception as e:
            print(f"Failed to fetch klines: {e}")
        print()

        # Example 4: Fetch recent trades
        print("--- Fetching Recent Trades ---")
        try:
            trades = await client.get_trades(symbol, limit=20)
            print(f"Got {len(trades)} recent trades")

            if trades:
                total_volume = sum(t.quantity for t in trades)
                total_value = sum(t.value() for t in trades)
                vwap = total_value / total_volume if total_volume > 0 else 0

                from llm_trade_execution.data import TradeDirection
                buy_volume = sum(
                    t.quantity for t in trades if t.direction == TradeDirection.BUY
                )

                print(f"Total Volume: {total_volume:.4f}")
                print(
                    f"Buy Volume: {buy_volume:.4f} "
                    f"({buy_volume / total_volume * 100:.1f}%)"
                )
                print(f"VWAP: {vwap:.2f}")

                print("\nRecent trades:")
                for trade in trades[:5]:
                    direction = "BUY " if trade.direction == TradeDirection.BUY else "SELL"
                    print(
                        f"  {direction} {trade.quantity:.4f} @ {trade.price:.2f} "
                        f"({trade.timestamp.strftime('%H:%M:%S')})"
                    )
        except Exception as e:
            print(f"Failed to fetch trades: {e}")

    print("\n=== Example Complete ===")
    print("Note: This example uses public endpoints only.")
    print("For actual trading, configure API credentials:")
    print("  - Set BYBIT_API_KEY and BYBIT_API_SECRET environment variables")
    print("  - Set BYBIT_TESTNET=true for testnet")


if __name__ == "__main__":
    asyncio.run(main())
