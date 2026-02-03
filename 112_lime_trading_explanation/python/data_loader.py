"""
Data Loader for Trading Models

Provides functionality to fetch market data from various sources:
- Yahoo Finance (stocks)
- Bybit API (cryptocurrency)

Also includes feature engineering for technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import requests


class DataLoader:
    """
    Data loader for fetching and preprocessing market data.

    Supports:
    - Yahoo Finance for stocks
    - Bybit API for cryptocurrency
    """

    def __init__(self, source: str = 'yahoo'):
        """
        Initialize the data loader.

        Args:
            source: Data source ('yahoo' or 'bybit')
        """
        self.source = source

    def fetch_yahoo(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance using yfinance.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            interval: Data interval ('1d', '1h', '5m', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Please install yfinance: pip install yfinance")

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        return df[['open', 'high', 'low', 'close', 'volume']]

    def fetch_bybit(
        self,
        symbol: str = 'BTCUSDT',
        interval: str = '60',
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch data from Bybit API.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Candle interval in minutes ('1', '5', '15', '60', '240', 'D')
            limit: Number of candles to fetch (max 1000)

        Returns:
            DataFrame with OHLCV data
        """
        url = 'https://api.bybit.com/v5/market/kline'

        params = {
            'category': 'spot',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data['retCode'] != 0:
            raise ValueError(f"Bybit API error: {data['retMsg']}")

        records = data['result']['list']

        df = pd.DataFrame(records, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df = df.set_index('timestamp')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df = df.sort_index()

        return df[['open', 'high', 'low', 'close', 'volume']]

    def fetch(
        self,
        symbol: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data from the configured source.

        Args:
            symbol: Asset symbol
            **kwargs: Additional arguments passed to the source-specific method

        Returns:
            DataFrame with OHLCV data
        """
        if self.source == 'yahoo':
            return self.fetch_yahoo(symbol, **kwargs)
        elif self.source == 'bybit':
            return self.fetch_bybit(symbol, **kwargs)
        else:
            raise ValueError(f"Unknown source: {self.source}")


class FeatureEngineering:
    """
    Feature engineering for trading models.

    Computes technical indicators commonly used in algorithmic trading.
    """

    @staticmethod
    def compute_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Compute returns over various periods.

        Args:
            df: DataFrame with 'close' column
            periods: List of periods for return calculation

        Returns:
            DataFrame with return columns added
        """
        result = df.copy()
        for period in periods:
            result[f'return_{period}d'] = df['close'].pct_change(period)
        return result

    @staticmethod
    def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Compute Relative Strength Index (RSI).

        Args:
            df: DataFrame with 'close' column
            period: RSI period

        Returns:
            DataFrame with RSI column added
        """
        result = df.copy()
        delta = df['close'].diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        result[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return result

    @staticmethod
    def compute_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Compute MACD indicator.

        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with MACD columns added
        """
        result = df.copy()

        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        result['macd'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd'].ewm(span=signal, adjust=False).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']

        return result

    @staticmethod
    def compute_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Compute Bollinger Bands.

        Args:
            df: DataFrame with 'close' column
            period: Moving average period
            std_dev: Number of standard deviations

        Returns:
            DataFrame with Bollinger Band columns added
        """
        result = df.copy()

        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        result['bb_middle'] = sma
        result['bb_upper'] = sma + (std * std_dev)
        result['bb_lower'] = sma - (std * std_dev)
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['bb_position'] = (df['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])

        return result

    @staticmethod
    def compute_sma(df: pd.DataFrame, periods: List[int] = [10, 20, 50, 200]) -> pd.DataFrame:
        """
        Compute Simple Moving Averages.

        Args:
            df: DataFrame with 'close' column
            periods: List of SMA periods

        Returns:
            DataFrame with SMA columns added
        """
        result = df.copy()
        for period in periods:
            result[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            result[f'price_sma_{period}_ratio'] = df['close'] / result[f'sma_{period}']
        return result

    @staticmethod
    def compute_volatility(df: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        """
        Compute volatility measures.

        Args:
            df: DataFrame with 'close' column
            periods: List of periods for volatility calculation

        Returns:
            DataFrame with volatility columns added
        """
        result = df.copy()
        returns = df['close'].pct_change()

        for period in periods:
            result[f'volatility_{period}d'] = returns.rolling(window=period).std() * np.sqrt(252)

        return result

    @staticmethod
    def compute_volume_features(df: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        """
        Compute volume-based features.

        Args:
            df: DataFrame with 'volume' column
            periods: List of periods for volume calculations

        Returns:
            DataFrame with volume feature columns added
        """
        result = df.copy()

        for period in periods:
            result[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            result[f'volume_ratio_{period}'] = df['volume'] / result[f'volume_sma_{period}']

        return result

    @classmethod
    def compute_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all feature columns added
        """
        result = df.copy()

        result = cls.compute_returns(result)
        result = cls.compute_rsi(result)
        result = cls.compute_macd(result)
        result = cls.compute_bollinger_bands(result)
        result = cls.compute_sma(result)
        result = cls.compute_volatility(result)
        result = cls.compute_volume_features(result)

        result = result.dropna()

        return result


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str = 'return_1d',
    feature_columns: Optional[List[str]] = None,
    classification: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for model training.

    Args:
        df: DataFrame with features
        target_column: Column to use as prediction target
        feature_columns: List of feature columns (default: all except target)
        classification: If True, convert target to binary (up/down)

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if feature_columns is None:
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', target_column]
        exclude_cols += [col for col in df.columns if col.startswith('return_')]
        feature_columns = [col for col in df.columns if col not in exclude_cols]

    target = df[target_column].shift(-1)

    if classification:
        target = (target > 0).astype(int)

    features = df[feature_columns]

    valid_idx = features.dropna().index.intersection(target.dropna().index)
    valid_idx = valid_idx[:-1]

    return features.loc[valid_idx], target.loc[valid_idx]
