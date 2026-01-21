"""
Data loaders for Yahoo Finance and Bybit cryptocurrency exchange.

This module provides unified interfaces for loading financial data
from multiple sources for regime classification.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class YahooFinanceLoader:
    """
    Load financial data from Yahoo Finance.

    Provides methods to fetch historical OHLCV data for stocks, ETFs,
    and indices with automatic data cleaning and feature engineering.
    """

    def __init__(self):
        """Initialize Yahoo Finance loader."""
        self._yf = None

    def _get_yfinance(self):
        """Lazy import of yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance is required for YahooFinanceLoader. "
                    "Install with: pip install yfinance"
                )
        return self._yf

    def get_daily(
        self,
        symbol: str,
        period: str = "1y",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a symbol.

        Args:
            symbol: Stock/ETF ticker symbol (e.g., "SPY", "AAPL")
            period: Time period (e.g., "1y", "6mo", "5y")
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Returns
        """
        yf = self._get_yfinance()

        ticker = yf.Ticker(symbol)

        if start and end:
            data = ticker.history(start=start, end=end)
        else:
            data = ticker.history(period=period)

        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")

        # Clean column names
        data.columns = [col.lower().replace(" ", "_") for col in data.columns]

        # Add returns
        data['returns'] = data['close'].pct_change()

        # Add volatility (20-day rolling)
        data['volatility'] = data['returns'].rolling(window=20).std() * np.sqrt(252)

        # Add simple moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()

        # Drop NaN rows from calculations
        data = data.dropna()

        logger.info(f"Loaded {len(data)} rows for {symbol}")
        return data

    def get_multiple(
        self,
        symbols: List[str],
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            period: Time period

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.get_daily(symbol, period=period)
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        return result

    def _generate_mock_data(self, symbol: str, days: int, interval: str = "1d") -> pd.DataFrame:
        """
        Generate mock stock data for demonstration purposes.

        Args:
            symbol: Stock symbol
            days: Number of days
            interval: Time interval (ignored for stocks, always daily)

        Returns:
            DataFrame with mock OHLCV data
        """
        np.random.seed(42)

        # Generate timestamps
        timestamps = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Starting prices for common symbols
        start_prices = {
            'SPY': 450.0,
            'QQQ': 380.0,
            'IWM': 200.0,
            'AAPL': 180.0,
            'MSFT': 400.0,
        }
        start_price = start_prices.get(symbol, 100.0)

        # Random walk with slight upward drift
        returns = np.random.randn(days) * 0.015 + 0.0003  # ~1.5% daily vol, small positive drift
        prices = start_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(days) * 0.002),
            'high': prices * (1 + np.abs(np.random.randn(days) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(days) * 0.01)),
            'close': prices,
            'volume': np.random.exponential(10000000, days)
        }, index=timestamps)

        # Add calculated fields
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        return df.dropna()


class BybitDataLoader:
    """
    Load cryptocurrency data from Bybit exchange.

    Provides methods to fetch historical klines (candlestick data)
    for crypto trading pairs with automatic data processing.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Initialize Bybit loader.

        Args:
            api_key: Optional Bybit API key (for authenticated endpoints)
            api_secret: Optional Bybit API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self._session = None

    def _get_session(self):
        """Get or create requests session."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
            except ImportError:
                raise ImportError(
                    "requests is required for BybitDataLoader. "
                    "Install with: pip install requests"
                )
        return self._session

    def get_historical_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        days: int = 30,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical klines (candlestick data) from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
            interval: Candlestick interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            days: Number of days of history to fetch
            limit: Maximum number of candles per request

        Returns:
            DataFrame with columns: open, high, low, close, volume, returns
        """
        session = self._get_session()

        # Map interval to Bybit format
        interval_map = {
            "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
            "1d": "D", "1w": "W", "1M": "M"
        }
        bybit_interval = interval_map.get(interval, interval)

        # Calculate start time
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_data = []

        # Fetch data in chunks
        current_end = end_time
        while current_end > start_time and len(all_data) < days * 24:  # Max iterations
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "end": current_end,
                "limit": min(limit, 200)
            }

            try:
                response = session.get(
                    f"{self.BASE_URL}/v5/market/kline",
                    params=params
                )
                response.raise_for_status()
                data = response.json()

                if data.get("retCode") != 0:
                    logger.warning(f"Bybit API error: {data.get('retMsg')}")
                    break

                klines = data.get("result", {}).get("list", [])
                if not klines:
                    break

                all_data.extend(klines)

                # Update end time for next iteration
                oldest_timestamp = int(klines[-1][0])
                if oldest_timestamp >= current_end:
                    break
                current_end = oldest_timestamp

            except Exception as e:
                logger.error(f"Error fetching Bybit data: {e}")
                break

        if not all_data:
            # Return mock data for demonstration
            logger.warning("Using mock data for demonstration")
            return self._generate_mock_data(symbol, days)

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.set_index('timestamp')

        # Add calculated fields
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=24).std() * np.sqrt(365 * 24)

        # Drop NaN
        df = df.dropna()

        logger.info(f"Loaded {len(df)} klines for {symbol}")
        return df

    def _generate_mock_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate mock data for demonstration purposes."""
        np.random.seed(42)

        periods = days * 24  # Hourly data

        # Generate price series with regime changes
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='H')

        # Random walk with drift
        returns = np.random.randn(periods) * 0.02  # 2% hourly volatility
        prices = 40000 * np.exp(np.cumsum(returns))  # BTC-like starting price

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.randn(periods) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(periods) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(periods) * 0.01)),
            'close': prices,
            'volume': np.random.exponential(1000, periods) * 100
        })

        df = df.set_index('timestamp')
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=24).std() * np.sqrt(365 * 24)
        df = df.dropna()

        return df

    def get_ticker(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Get current ticker information.

        Args:
            symbol: Trading pair

        Returns:
            Dictionary with ticker information
        """
        session = self._get_session()

        try:
            response = session.get(
                f"{self.BASE_URL}/v5/market/tickers",
                params={"category": "linear", "symbol": symbol}
            )
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") == 0:
                tickers = data.get("result", {}).get("list", [])
                if tickers:
                    return tickers[0]
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")

        return {}

    def get_funding_rate(self, symbol: str = "BTCUSDT") -> float:
        """
        Get current funding rate for perpetual contracts.

        Args:
            symbol: Trading pair

        Returns:
            Current funding rate as decimal
        """
        ticker = self.get_ticker(symbol)
        return float(ticker.get("fundingRate", 0))

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Alias for get_historical_klines with simpler interface.

        Args:
            symbol: Trading pair
            interval: Candlestick interval
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        # Convert limit to approximate days
        interval_hours = {
            "1m": 1/60, "3m": 3/60, "5m": 5/60, "15m": 15/60, "30m": 30/60,
            "1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12,
            "1d": 24, "1w": 168, "1M": 720
        }
        hours_per_candle = interval_hours.get(interval, 1)
        days = max(1, int(limit * hours_per_candle / 24))

        return self.get_historical_klines(symbol, interval, days, limit)

    def _generate_mock_crypto_data(self, symbol: str, periods: int, interval: str = "1h") -> pd.DataFrame:
        """
        Generate mock crypto data for demonstration purposes.

        Args:
            symbol: Crypto pair
            periods: Number of periods
            interval: Time interval

        Returns:
            DataFrame with mock OHLCV data
        """
        np.random.seed(42)

        # Map interval to frequency
        freq_map = {
            "1m": "min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h",
            "1d": "D", "1w": "W"
        }
        freq = freq_map.get(interval, "h")

        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq=freq)

        # Starting prices for common crypto pairs
        start_prices = {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 2500.0,
            'SOLUSDT': 100.0,
            'BNBUSDT': 300.0,
        }
        start_price = start_prices.get(symbol, 1000.0)

        # Crypto is more volatile
        vol = 0.02 if interval in ["1h", "1m", "5m", "15m", "30m"] else 0.03
        returns = np.random.randn(periods) * vol
        prices = start_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(periods) * 0.003),
            'high': prices * (1 + np.abs(np.random.randn(periods) * 0.015)),
            'low': prices * (1 - np.abs(np.random.randn(periods) * 0.015)),
            'close': prices,
            'volume': np.random.exponential(100, periods) * 10
        }, index=timestamps)

        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=24).std() * np.sqrt(365 * 24)

        return df.dropna()


# Utility functions
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional features for regime classification.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with additional features
    """
    result = df.copy()

    # Returns and volatility
    if 'returns' not in result.columns:
        result['returns'] = result['close'].pct_change()

    if 'volatility' not in result.columns:
        result['volatility'] = result['returns'].rolling(window=20).std()

    # Trend indicators
    result['trend'] = np.where(
        result['close'] > result['close'].shift(20), 1,
        np.where(result['close'] < result['close'].shift(20), -1, 0)
    )

    # Momentum
    result['momentum'] = result['close'] / result['close'].shift(20) - 1

    # Volume ratio
    result['volume_ratio'] = result['volume'] / result['volume'].rolling(window=20).mean()

    # Range (high-low as percentage of close)
    result['range'] = (result['high'] - result['low']) / result['close']

    return result.dropna()
