"""
Example: Fetch cryptocurrency data from Bybit.

Usage:
    python -m examples.fetch_data --symbol BTC/USDT --days 30
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from data import BybitFetcher


def main():
    parser = argparse.ArgumentParser(description='Fetch Bybit Data')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    parser.add_argument('--days', type=int, default=30, help='Days of data')
    parser.add_argument('--output', type=str, default='data.csv', help='Output file')

    args = parser.parse_args()

    logger.info(f"Fetching {args.symbol} {args.timeframe} data for {args.days} days")

    # Initialize fetcher
    fetcher = BybitFetcher()

    # Fetch OHLCV data
    ohlcv_data = fetcher.fetch_ohlcv(
        args.symbol,
        timeframe=args.timeframe,
        days=args.days
    )

    logger.info(f"Fetched {len(ohlcv_data.df)} candles")

    # Show summary
    df = ohlcv_data.df
    print("\nData Summary:")
    print(f"  Symbol: {ohlcv_data.symbol}")
    print(f"  Timeframe: {ohlcv_data.timeframe}")
    print(f"  Start: {df.index[0]}")
    print(f"  End: {df.index[-1]}")
    print(f"  Candles: {len(df)}")

    print("\nPrice Statistics:")
    print(f"  Open:  min={df['open'].min():.2f}, max={df['open'].max():.2f}")
    print(f"  High:  min={df['high'].min():.2f}, max={df['high'].max():.2f}")
    print(f"  Low:   min={df['low'].min():.2f}, max={df['low'].max():.2f}")
    print(f"  Close: min={df['close'].min():.2f}, max={df['close'].max():.2f}")

    print("\nVolume Statistics:")
    print(f"  Total: {df['volume'].sum():.2f}")
    print(f"  Mean:  {df['volume'].mean():.2f}")

    # Save to CSV
    df.to_csv(args.output)
    logger.info(f"Saved data to {args.output}")

    # Also fetch current price
    current_price = fetcher.get_current_price(args.symbol)
    print(f"\nCurrent {args.symbol} price: {current_price:.2f}")


if __name__ == '__main__':
    main()
