"""
Data Loading Utilities for Mamba4Cast

This module provides data loading functionality for stock market data
(via Yahoo Finance) and cryptocurrency data (via Bybit API).
"""

from typing import Optional, List
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class StockDataLoader:
    """
    Load and preprocess stock data from Yahoo Finance.

    Example:
        loader = StockDataLoader()
        df = loader.fetch("AAPL", period="1y")
        context = loader.prepare_for_forecast(df)
    """

    def __init__(self):
        if not HAS_YFINANCE:
            raise ImportError(
                "yfinance is required for StockDataLoader. "
                "Install with: pip install yfinance"
            )

    def fetch(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo',
                   '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m',
                     '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            DataFrame with OHLCV data
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        df.columns = df.columns.str.lower()
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.index.name = 'timestamp'

        return df.reset_index()

    def fetch_multiple(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> dict:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            period: Data period
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch(symbol, period, interval)
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol}: {e}")
        return data

    def prepare_for_forecast(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        context_length: int = 100,
    ) -> np.ndarray:
        """
        Prepare data for zero-shot forecasting.

        Args:
            df: DataFrame with OHLCV data
            target_col: Column to forecast
            context_length: Number of historical points

        Returns:
            Numpy array ready for model input
        """
        values = df[target_col].values[-context_length:]
        return values.reshape(-1, 1).astype(np.float32)

    def prepare_multivariate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        context_length: int = 100,
    ) -> np.ndarray:
        """
        Prepare multivariate data for forecasting.

        Args:
            df: DataFrame with data
            feature_cols: Columns to include (default: OHLCV)
            context_length: Number of historical points

        Returns:
            Numpy array of shape (context_length, n_features)
        """
        if feature_cols is None:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']

        values = df[feature_cols].values[-context_length:]
        return values.astype(np.float32)


class BybitDataLoader:
    """
    Load cryptocurrency data from Bybit exchange.

    Example:
        loader = BybitDataLoader()
        df = loader.fetch_klines("BTCUSDT", interval="60", limit=500)
        context = loader.prepare_for_forecast(df)
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        if not HAS_REQUESTS:
            raise ImportError(
                "requests is required for BybitDataLoader. "
                "Install with: pip install requests"
            )

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",
        limit: int = 1000,
        category: str = "linear",
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Kline interval in minutes
                     ('1', '3', '5', '15', '30', '60', '120', '240',
                      '360', '720', 'D', 'W', 'M')
            limit: Number of klines to fetch (max 1000)
            category: Product category ('linear', 'inverse', 'spot')

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if data.get("retCode") != 0:
            raise ValueError(f"API Error: {data.get('retMsg')}")

        result = data["result"]["list"]

        if not result:
            raise ValueError(f"No data found for symbol: {symbol}")

        df = pd.DataFrame(result, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        return df.sort_values('timestamp').reset_index(drop=True)

    def fetch_multiple(
        self,
        symbols: List[str],
        interval: str = "60",
        limit: int = 1000,
    ) -> dict:
        """
        Fetch klines for multiple symbols.

        Args:
            symbols: List of trading pairs
            interval: Kline interval
            limit: Number of klines

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch_klines(symbol, interval, limit)
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol}: {e}")
        return data

    def fetch_orderbook(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 50,
        category: str = "linear",
    ) -> dict:
        """
        Fetch current orderbook.

        Args:
            symbol: Trading pair
            limit: Orderbook depth
            category: Product category

        Returns:
            Dictionary with bids and asks
        """
        endpoint = f"{self.BASE_URL}/v5/market/orderbook"
        params = {
            "category": category,
            "symbol": symbol,
            "limit": limit,
        }

        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if data.get("retCode") != 0:
            raise ValueError(f"API Error: {data.get('retMsg')}")

        result = data["result"]

        bids = [[float(p), float(q)] for p, q in result.get("b", [])]
        asks = [[float(p), float(q)] for p, q in result.get("a", [])]

        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": result.get("ts"),
        }

    def prepare_for_forecast(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        context_length: int = 100,
    ) -> np.ndarray:
        """
        Prepare Bybit data for zero-shot forecasting.

        Args:
            df: DataFrame from fetch_klines
            target_col: Column to forecast
            context_length: Number of historical points

        Returns:
            Numpy array ready for model input
        """
        values = df[target_col].values[-context_length:]
        return values.reshape(-1, 1).astype(np.float32)

    def prepare_multivariate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None,
        context_length: int = 100,
    ) -> np.ndarray:
        """
        Prepare multivariate data for forecasting.

        Args:
            df: DataFrame from fetch_klines
            feature_cols: Columns to include
            context_length: Number of historical points

        Returns:
            Numpy array of shape (context_length, n_features)
        """
        if feature_cols is None:
            feature_cols = ['open', 'high', 'low', 'close', 'volume']

        values = df[feature_cols].values[-context_length:]
        return values.astype(np.float32)


class UnifiedDataLoader:
    """
    Unified data loader that supports both stocks and crypto.

    Example:
        loader = UnifiedDataLoader()

        # Load stock data
        aapl = loader.load("AAPL", source="stock")

        # Load crypto data
        btc = loader.load("BTCUSDT", source="crypto")
    """

    def __init__(self):
        self._stock_loader = None
        self._crypto_loader = None

    @property
    def stock_loader(self):
        if self._stock_loader is None:
            self._stock_loader = StockDataLoader()
        return self._stock_loader

    @property
    def crypto_loader(self):
        if self._crypto_loader is None:
            self._crypto_loader = BybitDataLoader()
        return self._crypto_loader

    def load(
        self,
        symbol: str,
        source: str = "auto",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from the appropriate source.

        Args:
            symbol: Asset symbol
            source: Data source ('stock', 'crypto', 'auto')
            **kwargs: Additional arguments for the loader

        Returns:
            DataFrame with OHLCV data
        """
        if source == "auto":
            source = self._detect_source(symbol)

        if source == "stock":
            return self.stock_loader.fetch(symbol, **kwargs)
        elif source == "crypto":
            return self.crypto_loader.fetch_klines(symbol, **kwargs)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _detect_source(self, symbol: str) -> str:
        """Attempt to detect the data source from symbol."""
        crypto_suffixes = ['USDT', 'USD', 'BTC', 'ETH', 'BUSD']

        for suffix in crypto_suffixes:
            if symbol.endswith(suffix) and len(symbol) > len(suffix):
                return "crypto"

        return "stock"

    def prepare_for_forecast(
        self,
        data: pd.DataFrame,
        target_col: str = 'close',
        context_length: int = 100,
    ) -> np.ndarray:
        """
        Prepare data for zero-shot forecasting.

        Args:
            data: DataFrame with OHLCV data
            target_col: Column to forecast
            context_length: Number of historical points

        Returns:
            Numpy array ready for model input
        """
        values = data[target_col].values[-context_length:]
        return values.reshape(-1, 1).astype(np.float32)
