"""
Data Fetcher module for fetching market data from Bybit via CCXT.

This module provides functionality to fetch OHLCV data, order books,
and trade history from Bybit exchange.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BybitDataFetcher:
    """
    Fetches market data from Bybit exchange using CCXT library.

    Supports:
    - OHLCV (candlestick) data
    - Order book snapshots
    - Recent trades
    - Multiple timeframes
    """

    # Default trading symbols
    DEFAULT_SYMBOLS = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
        'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT',
    ]

    # Supported timeframes
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

    def __init__(self, api_key: Optional[str] = None, secret: Optional[str] = None):
        """
        Initialize the Bybit data fetcher.

        Args:
            api_key: Optional API key for authenticated endpoints
            secret: Optional API secret for authenticated endpoints
        """
        config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # Linear perpetual contracts
            }
        }

        if api_key and secret:
            config['apiKey'] = api_key
            config['secret'] = secret

        self.exchange = ccxt.bybit(config)
        logger.info("Initialized Bybit data fetcher")

    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        limit: int = 500,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (max 1000)
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since
            )

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol

            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise

    def fetch_multiple_symbols(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = '1h',
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pairs (uses DEFAULT_SYMBOLS if None)
            timeframe: Candle timeframe
            limit: Number of candles per symbol

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.DEFAULT_SYMBOLS

        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch_ohlcv(symbol, timeframe, limit)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue

        logger.info(f"Fetched data for {len(data)} symbols")
        return data

    def fetch_order_book(
        self,
        symbol: str = 'BTC/USDT',
        limit: int = 20
    ) -> Dict:
        """
        Fetch current order book for a symbol.

        Args:
            symbol: Trading pair
            limit: Number of levels to fetch

        Returns:
            Dictionary with 'bids', 'asks', 'timestamp', and derived metrics
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)

            # Calculate derived metrics
            bids = np.array(order_book['bids'])
            asks = np.array(order_book['asks'])

            if len(bids) > 0 and len(asks) > 0:
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                spread = (best_ask - best_bid) / mid_price * 10000  # in basis points

                bid_volume = np.sum(bids[:, 1])
                ask_volume = np.sum(asks[:, 1])
                volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

                order_book['metrics'] = {
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'mid_price': mid_price,
                    'spread_bps': spread,
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'volume_imbalance': volume_imbalance,
                }

            logger.info(f"Fetched order book for {symbol}")
            return order_book

        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise

    def fetch_recent_trades(
        self,
        symbol: str = 'BTC/USDT',
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch recent trades for a symbol.

        Args:
            symbol: Trading pair
            limit: Number of trades to fetch

        Returns:
            DataFrame with columns: timestamp, price, amount, side
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)

            df = pd.DataFrame(trades)
            df = df[['timestamp', 'price', 'amount', 'side']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol

            logger.info(f"Fetched {len(df)} recent trades for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise

    def fetch_ticker(self, symbol: str = 'BTC/USDT') -> Dict:
        """
        Fetch current ticker information for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Dictionary with ticker information
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.info(f"Fetched ticker for {symbol}")
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise

    def fetch_all_tickers(self) -> Dict[str, Dict]:
        """
        Fetch tickers for all trading pairs.

        Returns:
            Dictionary mapping symbol to ticker data
        """
        try:
            tickers = self.exchange.fetch_tickers()
            logger.info(f"Fetched {len(tickers)} tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching all tickers: {e}")
            raise

    def fetch_historical_data(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a specified number of days.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days of history to fetch

        Returns:
            DataFrame with historical OHLCV data
        """
        # Calculate start timestamp
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        since = int(start_time.timestamp() * 1000)

        # Calculate number of candles based on timeframe
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
        }
        minutes = timeframe_minutes.get(timeframe, 60)
        total_candles = (days * 24 * 60) // minutes

        all_data = []
        current_since = since

        while len(all_data) < total_candles:
            try:
                batch = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000
                )

                if not batch:
                    break

                all_data.extend(batch)
                current_since = batch[-1][0] + 1
                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                break

        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        logger.info(f"Fetched {len(df)} historical candles for {symbol}")
        return df


def main():
    """Example usage of the BybitDataFetcher."""
    fetcher = BybitDataFetcher()

    # Fetch OHLCV data
    print("\n=== Fetching OHLCV Data ===")
    df = fetcher.fetch_ohlcv('BTC/USDT', '1h', limit=100)
    print(df.head())
    print(f"\nShape: {df.shape}")

    # Fetch order book
    print("\n=== Fetching Order Book ===")
    order_book = fetcher.fetch_order_book('BTC/USDT')
    print(f"Best Bid: {order_book['metrics']['best_bid']}")
    print(f"Best Ask: {order_book['metrics']['best_ask']}")
    print(f"Spread (bps): {order_book['metrics']['spread_bps']:.2f}")
    print(f"Volume Imbalance: {order_book['metrics']['volume_imbalance']:.4f}")

    # Fetch recent trades
    print("\n=== Fetching Recent Trades ===")
    trades = fetcher.fetch_recent_trades('BTC/USDT', limit=10)
    print(trades)

    # Fetch ticker
    print("\n=== Fetching Ticker ===")
    ticker = fetcher.fetch_ticker('BTC/USDT')
    print(f"Last Price: {ticker['last']}")
    print(f"24h Volume: {ticker['quoteVolume']}")
    print(f"24h Change: {ticker['percentage']}%")


if __name__ == '__main__':
    main()
