"""
Data Loader Module for EfficientNet Trading

This module provides data loading utilities for cryptocurrency markets
using both direct Bybit API and CCXT library.

All data fetching and processing is contained within this chapter's folder.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests


class BybitDataLoader:
    """
    Load cryptocurrency data directly from Bybit exchange.

    Supports both spot and linear perpetual contracts.

    Example:
        >>> loader = BybitDataLoader()
        >>> df = loader.fetch_klines("BTCUSDT", interval="60", limit=1000)
        >>> print(df.head())
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 1000,
        category: str = "linear",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Kline interval in minutes ('1', '5', '15', '60', '240', 'D', 'W')
            limit: Number of klines to fetch (max 1000)
            category: Market category ('linear', 'inverse', 'spot')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, turnover
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data["retCode"] != 0:
                raise ValueError(f"Bybit API error: {data['retMsg']}")

            records = data["result"]["list"]

            df = pd.DataFrame(records, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)

            # Sort by time (Bybit returns descending)
            df = df.sort_values("timestamp").reset_index(drop=True)
            df.set_index("timestamp", inplace=True)

            return df

        except Exception as e:
            raise ValueError(f"Failed to fetch Bybit data for {symbol}: {e}")

    def fetch_extended(
        self,
        symbol: str,
        interval: str = "60",
        days: int = 30,
        category: str = "linear",
    ) -> pd.DataFrame:
        """
        Fetch extended historical data by making multiple API calls.

        Args:
            symbol: Trading pair
            interval: Kline interval
            days: Number of days of data to fetch
            category: Market category

        Returns:
            DataFrame with extended historical data
        """
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)

        # Calculate interval in milliseconds
        interval_ms = self._interval_to_ms(interval)
        records_per_call = 1000
        ms_per_call = interval_ms * records_per_call

        # Calculate number of calls needed
        total_ms = days * 24 * 60 * 60 * 1000
        n_calls = int(np.ceil(total_ms / ms_per_call))

        for i in range(n_calls):
            try:
                df = self.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=1000,
                    category=category,
                    end_time=end_time,
                )

                if df.empty:
                    break

                all_data.append(df)
                end_time = int(df.index.min().timestamp() * 1000) - 1
                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                print(f"Warning: Error fetching batch {i}: {e}")
                break

        if not all_data:
            raise ValueError(f"No data fetched for {symbol}")

        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep="first")]
        result = result.sort_index()

        return result

    def fetch_multi_timeframe(
        self,
        symbol: str,
        intervals: List[str] = ["1", "5", "15"],
        lookback_candles: int = 120,
        category: str = "linear",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes.

        Args:
            symbol: Trading pair
            intervals: List of intervals to fetch
            lookback_candles: Number of candles per timeframe
            category: Market category

        Returns:
            Dictionary mapping interval to DataFrame
        """
        data = {}
        for interval in intervals:
            try:
                df = self.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=lookback_candles,
                    category=category,
                )
                data[interval] = df
                time.sleep(0.1)
            except Exception as e:
                print(f"Warning: Could not fetch {interval}m data: {e}")

        return data

    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds."""
        interval_map = {
            "1": 60 * 1000,
            "3": 3 * 60 * 1000,
            "5": 5 * 60 * 1000,
            "15": 15 * 60 * 1000,
            "30": 30 * 60 * 1000,
            "60": 60 * 60 * 1000,
            "120": 2 * 60 * 60 * 1000,
            "240": 4 * 60 * 60 * 1000,
            "360": 6 * 60 * 60 * 1000,
            "720": 12 * 60 * 60 * 1000,
            "D": 24 * 60 * 60 * 1000,
            "W": 7 * 24 * 60 * 60 * 1000,
            "M": 30 * 24 * 60 * 60 * 1000,
        }
        return interval_map.get(interval, 60 * 60 * 1000)

    def get_tickers(self, category: str = "linear") -> pd.DataFrame:
        """
        Get all available tickers.

        Args:
            category: Market category

        Returns:
            DataFrame with ticker information
        """
        endpoint = f"{self.BASE_URL}/v5/market/tickers"
        params = {"category": category}

        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] != 0:
            raise ValueError(f"Bybit API error: {data['retMsg']}")

        return pd.DataFrame(data["result"]["list"])


class CCXTDataLoader:
    """
    Load cryptocurrency data using CCXT library.

    Provides a unified interface across multiple exchanges.

    Example:
        >>> loader = CCXTDataLoader(exchange='bybit')
        >>> df = loader.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=1000)
        >>> print(df.head())
    """

    def __init__(self, exchange: str = "bybit"):
        """
        Initialize CCXT data loader.

        Args:
            exchange: Exchange name (e.g., 'bybit', 'binance', 'okx')
        """
        try:
            import ccxt
        except ImportError:
            raise ImportError("ccxt is required. Install with: pip install ccxt")

        exchange_class = getattr(ccxt, exchange)
        self.exchange = exchange_class({
            "enableRateLimit": True,
        })
        self.exchange_name = exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data using CCXT.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit,
                since=since,
            )

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            return df

        except Exception as e:
            raise ValueError(f"Failed to fetch CCXT data for {symbol}: {e}")

    def fetch_extended(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch extended historical data.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days of data

        Returns:
            DataFrame with extended data
        """
        all_data = []
        timeframe_ms = self._timeframe_to_ms(timeframe)

        # Calculate how many candles we need
        candles_per_day = 24 * 60 * 60 * 1000 // timeframe_ms
        total_candles = days * candles_per_day

        # Start from the past
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        while len(all_data) < total_candles:
            try:
                df = self.fetch_ohlcv(symbol, timeframe, limit=1000, since=since)

                if df.empty:
                    break

                all_data.append(df)
                since = int(df.index[-1].timestamp() * 1000) + timeframe_ms
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"Warning: Error fetching batch: {e}")
                break

        if not all_data:
            raise ValueError(f"No data fetched for {symbol}")

        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep="first")]
        result = result.sort_index()

        return result

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = ["1m", "5m", "15m"],
        limit: int = 120,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes.

        Args:
            symbol: Trading pair
            timeframes: List of timeframes
            limit: Candles per timeframe

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        data = {}
        for tf in timeframes:
            try:
                df = self.fetch_ohlcv(symbol, tf, limit)
                data[tf] = df
            except Exception as e:
                print(f"Warning: Could not fetch {tf} data: {e}")

        return data

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        mapping = {
            "1m": 60 * 1000,
            "3m": 3 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "6h": 6 * 60 * 60 * 1000,
            "12h": 12 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
        }
        return mapping.get(timeframe, 60 * 60 * 1000)


class DataManager:
    """
    Unified data manager for multiple data sources.

    Provides caching and a consistent interface.

    Example:
        >>> manager = DataManager()
        >>> df = manager.get_crypto_data("BTCUSDT", days=30)
        >>> multi_tf = manager.get_multi_timeframe("BTCUSDT")
    """

    def __init__(
        self,
        use_ccxt: bool = False,
        exchange: str = "bybit",
        cache_enabled: bool = True,
    ):
        """
        Initialize data manager.

        Args:
            use_ccxt: Use CCXT library instead of direct API
            exchange: Exchange name for CCXT
            cache_enabled: Enable data caching
        """
        if use_ccxt:
            self.loader = CCXTDataLoader(exchange)
        else:
            self.loader = BybitDataLoader()

        self.use_ccxt = use_ccxt
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_crypto_data(
        self,
        symbol: str,
        interval: str = "60",
        days: int = 30,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get cryptocurrency data with optional caching.

        Args:
            symbol: Trading pair
            interval: Kline interval (Bybit format: '1', '5', '60')
                     or timeframe (CCXT format: '1m', '5m', '1h')
            days: Number of days
            use_cache: Use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"crypto_{symbol}_{interval}_{days}"

        if use_cache and self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        if self.use_ccxt:
            # Convert Bybit interval to CCXT timeframe if needed
            timeframe = self._convert_to_ccxt_timeframe(interval)
            df = self.loader.fetch_extended(symbol, timeframe, days)
        else:
            df = self.loader.fetch_extended(symbol, interval, days)

        if use_cache and self.cache_enabled:
            self._cache[cache_key] = df

        return df

    def get_multi_timeframe(
        self,
        symbol: str,
        intervals: List[str] = ["1", "5", "15"],
        lookback: int = 120,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get multi-timeframe data.

        Args:
            symbol: Trading pair
            intervals: List of intervals
            lookback: Candles per timeframe

        Returns:
            Dictionary mapping interval to DataFrame
        """
        if self.use_ccxt:
            timeframes = [self._convert_to_ccxt_timeframe(i) for i in intervals]
            return self.loader.fetch_multi_timeframe(symbol, timeframes, lookback)
        else:
            return self.loader.fetch_multi_timeframe(symbol, intervals, lookback)

    def _convert_to_ccxt_timeframe(self, interval: str) -> str:
        """Convert Bybit interval to CCXT timeframe."""
        mapping = {
            "1": "1m",
            "3": "3m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "120": "2h",
            "240": "4h",
            "360": "6h",
            "720": "12h",
            "D": "1d",
            "W": "1w",
        }
        return mapping.get(interval, interval)

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()


def prepare_dataset_for_efficientnet(
    df: pd.DataFrame,
    lookback: int = 120,
    prediction_horizon: int = 5,
    threshold: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare dataset for EfficientNet training.

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of candles for each sample
        prediction_horizon: How many candles ahead to predict
        threshold: Return threshold for classification (default 0.5%)

    Returns:
        Tuple of (close_prices_windows, labels, returns)
        - close_prices_windows: Shape (N, lookback)
        - labels: 0=down, 1=neutral, 2=up
        - returns: Actual returns for each sample
    """
    close_prices = df["close"].values

    windows = []
    labels = []
    returns = []

    for i in range(lookback, len(close_prices) - prediction_horizon):
        # Extract window
        window = close_prices[i - lookback:i]
        windows.append(window)

        # Calculate future return
        current_price = close_prices[i - 1]
        future_price = close_prices[i + prediction_horizon - 1]
        ret = (future_price - current_price) / current_price
        returns.append(ret)

        # Classify return
        if ret > threshold:
            labels.append(2)  # UP
        elif ret < -threshold:
            labels.append(0)  # DOWN
        else:
            labels.append(1)  # NEUTRAL

    return np.array(windows), np.array(labels), np.array(returns)


if __name__ == "__main__":
    # Example usage
    print("Testing BybitDataLoader...")
    loader = BybitDataLoader()

    # Fetch single timeframe
    df = loader.fetch_klines("BTCUSDT", interval="60", limit=100)
    print(f"Fetched {len(df)} candles for BTCUSDT 1h")
    print(df.head())

    # Fetch multi-timeframe
    print("\nFetching multi-timeframe data...")
    multi_tf = loader.fetch_multi_timeframe("BTCUSDT", ["1", "5", "15"], 50)
    for tf, data in multi_tf.items():
        print(f"  {tf}m: {len(data)} candles")

    # Prepare dataset
    print("\nPreparing dataset for EfficientNet...")
    windows, labels, returns = prepare_dataset_for_efficientnet(df, lookback=60)
    print(f"Generated {len(windows)} samples")
    print(f"Label distribution: DOWN={sum(labels==0)}, NEUTRAL={sum(labels==1)}, UP={sum(labels==2)}")
