"""
Data Fetcher for Conformal Prediction Trading

Fetches OHLCV data from Bybit via CCXT library.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import time


class BybitDataFetcher:
    """
    Fetches market data from Bybit exchange using CCXT.
    """

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit data fetcher.

        Args:
            testnet: Whether to use testnet (default: False)
        """
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # Use perpetual futures
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT:USDT',
        timeframe: str = '1h',
        limit: int = 1000,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT')
            timeframe: Candlestick timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (max 1000)
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except ccxt.BaseError as e:
            print(f"Error fetching OHLCV data: {e}")
            raise

    def fetch_historical_data(
        self,
        symbol: str = 'BTC/USDT:USDT',
        timeframe: str = '1h',
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple days.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            days: Number of days of historical data

        Returns:
            DataFrame with OHLCV data
        """
        all_data = []
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Convert timeframe to milliseconds
        timeframe_ms = self._timeframe_to_ms(timeframe)

        current_time = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)

        print(f"Fetching {days} days of {timeframe} data for {symbol}...")

        while current_time < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=1000,
                    since=current_time
                )

                if not ohlcv:
                    break

                all_data.extend(ohlcv)

                # Move to next batch
                current_time = ohlcv[-1][0] + timeframe_ms

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

            except ccxt.BaseError as e:
                print(f"Error fetching data: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        return df

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            timeframe: Candlestick timeframe
            days: Number of days of historical data

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}

        for symbol in symbols:
            print(f"\nFetching {symbol}...")
            try:
                df = self.fetch_historical_data(symbol, timeframe, days)
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
                continue

        return data

    def fetch_ticker(self, symbol: str = 'BTC/USDT:USDT') -> Dict[str, Any]:
        """
        Fetch current ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with ticker information
        """
        try:
            return self.exchange.fetch_ticker(symbol)
        except ccxt.BaseError as e:
            print(f"Error fetching ticker: {e}")
            raise

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000,
        }

        unit = timeframe[-1]
        value = int(timeframe[:-1])

        return value * multipliers.get(unit, 60 * 1000)


# Default symbols to fetch
DEFAULT_SYMBOLS = [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'SOL/USDT:USDT',
    'BNB/USDT:USDT',
    'XRP/USDT:USDT',
    'ADA/USDT:USDT',
    'AVAX/USDT:USDT',
    'DOGE/USDT:USDT',
    'MATIC/USDT:USDT',
    'DOT/USDT:USDT',
]


if __name__ == "__main__":
    # Example usage
    fetcher = BybitDataFetcher()

    # Fetch single symbol
    df = fetcher.fetch_ohlcv('BTC/USDT:USDT', timeframe='1h', limit=100)
    print(f"\nBTC/USDT:USDT data shape: {df.shape}")
    print(df.tail())

    # Fetch historical data
    df_hist = fetcher.fetch_historical_data('BTC/USDT:USDT', timeframe='1h', days=7)
    print(f"\nHistorical data shape: {df_hist.shape}")
