"""
Data loading module for fetching market data from Bybit and other sources.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta


class BybitDataLoader:
    """
    Fetches cryptocurrency market data from Bybit exchange API.

    Supports both perpetual futures and spot markets.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 200,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data from Bybit.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Candle interval ('1', '5', '15', '60', '240', 'D')
            limit: Number of candles to fetch (max 200)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        interval_map = {
            '1m': '1', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '4h': '240', '1d': 'D', '1D': 'D'
        }

        bybit_interval = interval_map.get(interval, interval)

        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': bybit_interval,
            'limit': min(limit, 200),
        }

        if start_time:
            params['start'] = start_time
        if end_time:
            params['end'] = end_time

        response = self.session.get(
            f"{self.BASE_URL}/v5/market/kline",
            params=params
        )
        response.raise_for_status()
        data = response.json()

        if data['retCode'] != 0:
            raise ValueError(f"Bybit API error: {data['retMsg']}")

        records = []
        for item in data['result']['list']:
            records.append({
                'timestamp': pd.to_datetime(int(item[0]), unit='ms'),
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'volume': float(item[5]),
            })

        df = pd.DataFrame(records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def fetch_orderbook(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 50
    ) -> dict:
        """
        Fetch current orderbook snapshot.

        Args:
            symbol: Trading pair symbol
            limit: Depth of orderbook (max 200)

        Returns:
            Dictionary with 'bids' and 'asks' lists
        """
        params = {
            'category': 'linear',
            'symbol': symbol,
            'limit': min(limit, 200),
        }

        response = self.session.get(
            f"{self.BASE_URL}/v5/market/orderbook",
            params=params
        )
        response.raise_for_status()
        data = response.json()

        if data['retCode'] != 0:
            raise ValueError(f"Bybit API error: {data['retMsg']}")

        result = data['result']
        return {
            'bids': [(float(p), float(q)) for p, q in result['b']],
            'asks': [(float(p), float(q)) for p, q in result['a']],
            'timestamp': pd.to_datetime(int(result['ts']), unit='ms'),
        }


class YahooFinanceLoader:
    """
    Fetches stock market data from Yahoo Finance.
    """

    def fetch_history(
        self,
        symbol: str = "SPY",
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data.

        Args:
            symbol: Stock ticker symbol
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d', '1wk')

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            elif 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except ImportError:
            raise ImportError("yfinance is required for Yahoo Finance data. Install with: pip install yfinance")


def generate_sample_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.

    Args:
        n_samples: Number of candles to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic OHLCV data
    """
    np.random.seed(seed)

    timestamps = pd.date_range(
        end=datetime.now(),
        periods=n_samples,
        freq='1h'
    )

    price = 50000
    prices = [price]

    for _ in range(n_samples - 1):
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        prices.append(price)

    prices = np.array(prices)

    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
    opens = lows + (highs - lows) * np.random.random(n_samples)
    closes = lows + (highs - lows) * np.random.random(n_samples)
    volumes = np.random.exponential(1000, n_samples)

    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    })
