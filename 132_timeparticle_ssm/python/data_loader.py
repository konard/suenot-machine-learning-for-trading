"""
Data Loader Module for TimeParticle SSM

Provides data loading utilities for stock and cryptocurrency markets,
including Yahoo Finance and Bybit API integration.
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import time


def fetch_stock_data(
    symbol: str,
    period: str = "2y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")
        period: Data period (e.g., "1y", "2y", "5y")
        interval: Data interval (e.g., "1d", "1h", "5m")

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        df.columns = df.columns.str.lower()
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except ImportError:
        print("yfinance not installed. Using simulated data.")
        return generate_simulated_data(500)


def generate_simulated_data(n_points: int = 1000) -> pd.DataFrame:
    """
    Generate simulated OHLCV data for testing.

    Args:
        n_points: Number of data points to generate

    Returns:
        DataFrame with simulated OHLCV data
    """
    np.random.seed(42)

    # Generate price with trend and noise
    trend = np.cumsum(np.random.randn(n_points) * 0.5) + 100
    noise = np.random.randn(n_points) * 2

    close = trend + noise
    close = np.maximum(close, 1)  # Ensure positive prices

    # Generate OHLC from close
    high = close + np.abs(np.random.randn(n_points) * 1.5)
    low = close - np.abs(np.random.randn(n_points) * 1.5)
    open_price = low + (high - low) * np.random.rand(n_points)

    # Generate volume
    volume = np.random.exponential(1000000, n_points)

    # Create date index
    dates = pd.date_range(end=datetime.now(), periods=n_points, freq='D')

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


class BybitDataLoader:
    """
    Data loader for Bybit cryptocurrency exchange.

    Provides methods to fetch historical kline data and real-time prices
    from the Bybit API.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit data loader.

        Args:
            testnet: Whether to use testnet API
        """
        if testnet:
            self.BASE_URL = "https://api-testnet.bybit.com"

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Bybit.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of klines to fetch (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["retCode"] != 0:
                raise ValueError(f"API Error: {data['retMsg']}")

            klines = data["result"]["list"]

            if not klines:
                print(f"No data returned for {symbol}")
                return generate_simulated_data(limit)

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)

            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.set_index('timestamp')

            return df[['open', 'high', 'low', 'close', 'volume']]

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Using simulated data.")
            return generate_simulated_data(limit)

    def fetch_multiscale_data(
        self,
        symbol: str,
        scales: Optional[Dict[str, str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data at multiple time scales.

        Args:
            symbol: Trading pair symbol
            scales: Dictionary mapping scale names to intervals

        Returns:
            Dictionary of DataFrames for each scale
        """
        if scales is None:
            scales = {
                'minute': '1',
                'hourly': '60',
                'daily': 'D',
                'weekly': 'W'
            }

        data = {}
        for scale_name, interval in scales.items():
            try:
                df = self.fetch_klines(symbol, interval)
                data[scale_name] = df
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Failed to fetch {scale_name} data: {e}")

        return data

    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with ticker information
        """
        endpoint = f"{self.BASE_URL}/v5/market/tickers"

        params = {
            "category": "linear",
            "symbol": symbol
        }

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["retCode"] != 0:
                raise ValueError(f"API Error: {data['retMsg']}")

            if data["result"]["list"]:
                return data["result"]["list"][0]
            return {}

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return {}


class DataPreprocessor:
    """
    Preprocessor for preparing trading data for TimeParticle models.
    """

    def __init__(self, lookback: int = 100, normalize: bool = True):
        """
        Initialize preprocessor.

        Args:
            lookback: Number of lookback periods for sequences
            normalize: Whether to normalize features
        """
        self.lookback = lookback
        self.normalize = normalize
        self.feature_means = None
        self.feature_stds = None

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'close',
        threshold: float = 0.005
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for model training.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            target_col: Column to use for target generation
            threshold: Threshold for buy/sell classification

        Returns:
            X: Feature sequences [n_samples, lookback, n_features]
            y: Labels [n_samples]
        """
        features = df[feature_cols].values

        if self.normalize:
            if self.feature_means is None:
                self.feature_means = np.nanmean(features, axis=0)
                self.feature_stds = np.nanstd(features, axis=0)
                self.feature_stds[self.feature_stds == 0] = 1  # Avoid division by zero

            features = (features - self.feature_means) / self.feature_stds

        # Replace NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        X, y = [], []

        for i in range(self.lookback, len(features)):
            X.append(features[i - self.lookback:i])

            # Calculate future return for target
            future_return = df[target_col].iloc[i] / df[target_col].iloc[i-1] - 1

            if future_return > threshold:
                y.append(2)  # Buy
            elif future_return < -threshold:
                y.append(0)  # Sell
            else:
                y.append(1)  # Hold

        return np.array(X), np.array(y)

    def prepare_inference_sequence(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        Prepare a single sequence for inference.

        Args:
            df: DataFrame with features (must have at least lookback rows)
            feature_cols: List of feature column names

        Returns:
            Feature sequence [1, lookback, n_features]
        """
        if len(df) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} rows for inference")

        features = df[feature_cols].values[-self.lookback:]

        if self.normalize and self.feature_means is not None:
            features = (features - self.feature_means) / self.feature_stds

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features[np.newaxis, :, :]


if __name__ == "__main__":
    print("Testing Data Loaders...")

    # Test simulated data
    print("\n1. Testing simulated data generation...")
    sim_data = generate_simulated_data(500)
    print(f"Generated {len(sim_data)} rows of simulated data")
    print(sim_data.head())

    # Test Bybit loader (will use simulated data if API fails)
    print("\n2. Testing Bybit data loader...")
    loader = BybitDataLoader()
    btc_data = loader.fetch_klines("BTCUSDT", interval="60", limit=100)
    print(f"Loaded {len(btc_data)} rows of BTC data")
    print(btc_data.head())

    # Test data preprocessor
    print("\n3. Testing data preprocessor...")
    from features import prepare_multiscale_features

    features_df = prepare_multiscale_features(sim_data)
    feature_cols = [c for c in features_df.columns
                    if c not in ['open', 'high', 'low', 'close', 'volume']]

    preprocessor = DataPreprocessor(lookback=50)
    X, y = preprocessor.prepare_sequences(features_df, feature_cols)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Label distribution: Sell={sum(y==0)}, Hold={sum(y==1)}, Buy={sum(y==2)}")

    print("\nAll tests passed!")
