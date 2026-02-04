"""
Bybit API Client using CCXT
===========================

This module provides a simple interface to fetch market data from Bybit
using the CCXT library.
"""

import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class BybitClient:
    """
    Client for fetching data from Bybit exchange using CCXT.
    """

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit client.

        Args:
            testnet: If True, use Bybit testnet
        """
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # Use perpetual futures
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

        logger.info(f"Initialized Bybit client (testnet={testnet})")

    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')

        Returns:
            Ticker dictionary with price, volume, etc.
        """
        ticker = self.exchange.fetch_ticker(symbol)
        return {
            'symbol': symbol,
            'last_price': ticker['last'],
            'bid': ticker['bid'],
            'ask': ticker['ask'],
            'volume_24h': ticker['baseVolume'],
            'change_24h': ticker['percentage'] / 100 if ticker['percentage'] else 0,
            'high_24h': ticker['high'],
            'low_24h': ticker['low'],
            'timestamp': ticker['timestamp'],
        }

    def get_tickers(self, symbols: List[str]) -> List[Dict]:
        """
        Get tickers for multiple symbols.

        Args:
            symbols: List of trading pairs

        Returns:
            List of ticker dictionaries
        """
        tickers = []
        for symbol in symbols:
            try:
                ticker = self.get_ticker(symbol)
                tickers.append(ticker)
            except Exception as e:
                logger.warning(f"Failed to fetch ticker for {symbol}: {e}")
        return tickers

    def get_ohlcv(self,
                  symbol: str,
                  timeframe: str = '1h',
                  limit: int = 100,
                  since: Optional[int] = None) -> pd.DataFrame:
        """
        Get OHLCV (candlestick) data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            timeframe: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """
        Get order book data.

        Args:
            symbol: Trading pair
            limit: Number of levels to fetch

        Returns:
            Order book dictionary with bids and asks
        """
        orderbook = self.exchange.fetch_order_book(symbol, limit)

        bids = np.array(orderbook['bids'])
        asks = np.array(orderbook['asks'])

        mid_price = (bids[0, 0] + asks[0, 0]) / 2 if len(bids) > 0 and len(asks) > 0 else 0
        spread = (asks[0, 0] - bids[0, 0]) / mid_price * 10000 if mid_price > 0 else 0

        # Calculate depth imbalance
        bid_volume = bids[:, 1].sum() if len(bids) > 0 else 0
        ask_volume = asks[:, 1].sum() if len(asks) > 0 else 0
        depth_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0

        return {
            'symbol': symbol,
            'bids': bids.tolist(),
            'asks': asks.tolist(),
            'mid_price': mid_price,
            'spread_bps': spread,
            'depth_imbalance': depth_imbalance,
            'timestamp': orderbook['timestamp'],
        }

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades.

        Args:
            symbol: Trading pair
            limit: Number of trades to fetch

        Returns:
            List of trade dictionaries
        """
        trades = self.exchange.fetch_trades(symbol, limit=limit)

        return [{
            'timestamp': trade['timestamp'],
            'price': trade['price'],
            'amount': trade['amount'],
            'side': trade['side'],
            'cost': trade['cost'],
        } for trade in trades]


# Default symbols for examples
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


def fetch_training_data(symbol: str = 'BTC/USDT:USDT',
                        timeframe: str = '1h',
                        days: int = 30) -> pd.DataFrame:
    """
    Fetch historical data for model training.

    Args:
        symbol: Trading pair
        timeframe: Candlestick timeframe
        days: Number of days of history to fetch

    Returns:
        DataFrame with OHLCV data
    """
    client = BybitClient()

    # Calculate how many candles we need
    timeframe_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }

    minutes = timeframe_minutes.get(timeframe, 60)
    candles_needed = (days * 24 * 60) // minutes

    # CCXT has limits on how many candles can be fetched at once
    # We fetch in batches if needed
    all_data = []
    limit_per_request = 200

    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    while candles_needed > 0:
        limit = min(candles_needed, limit_per_request)
        df = client.get_ohlcv(symbol, timeframe, limit=limit, since=since)

        if len(df) == 0:
            break

        all_data.append(df)
        candles_needed -= len(df)
        since = int(df.index[-1].timestamp() * 1000) + 1

    if all_data:
        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep='first')]
        return result.sort_index()

    return pd.DataFrame()


if __name__ == '__main__':
    # Example usage
    client = BybitClient()

    print("=== Fetching BTC/USDT Ticker ===")
    ticker = client.get_ticker('BTC/USDT:USDT')
    print(f"Price: ${ticker['last_price']:,.2f}")
    print(f"24h Change: {ticker['change_24h']*100:.2f}%")
    print(f"24h Volume: {ticker['volume_24h']:,.0f} BTC")

    print("\n=== Fetching OHLCV Data ===")
    df = client.get_ohlcv('BTC/USDT:USDT', '1h', limit=10)
    print(df)

    print("\n=== Fetching Order Book ===")
    orderbook = client.get_orderbook('BTC/USDT:USDT', limit=5)
    print(f"Mid Price: ${orderbook['mid_price']:,.2f}")
    print(f"Spread: {orderbook['spread_bps']:.2f} bps")
    print(f"Depth Imbalance: {orderbook['depth_imbalance']:.4f}")
