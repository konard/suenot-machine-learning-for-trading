"""
Data loading utilities for prototype learning.

Supports loading data from:
- Stock market via yfinance
- Cryptocurrency via Bybit API
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
from datetime import datetime, timedelta
import requests
import time

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class BybitDataLoader:
    """
    Data loader for Bybit cryptocurrency exchange.

    Fetches OHLCV data via Bybit's public API.
    """

    BASE_URL = "https://api.bybit.com"

    INTERVALS = {
        '1m': '1',
        '3m': '3',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '1h': '60',
        '2h': '120',
        '4h': '240',
        '6h': '360',
        '12h': '720',
        '1d': 'D',
        '1w': 'W',
        '1M': 'M',
    }

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit data loader.

        Args:
            testnet: If True, use testnet API
        """
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = self.BASE_URL

    def get_klines(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 1000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Candlestick interval ('1m', '5m', '1h', '1d', etc.)
            limit: Number of candles (max 1000)
            start_time: Start time as string 'YYYY-MM-DD' or timestamp
            end_time: End time as string 'YYYY-MM-DD' or timestamp

        Returns:
            DataFrame with OHLCV data
        """
        if interval not in self.INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Valid: {list(self.INTERVALS.keys())}")

        params = {
            'category': 'spot',
            'symbol': symbol,
            'interval': self.INTERVALS[interval],
            'limit': min(limit, 1000),
        }

        if start_time:
            if isinstance(start_time, str):
                start_time = int(pd.Timestamp(start_time).timestamp() * 1000)
            params['start'] = start_time

        if end_time:
            if isinstance(end_time, str):
                end_time = int(pd.Timestamp(end_time).timestamp() * 1000)
            params['end'] = end_time

        url = f"{self.base_url}/v5/market/kline"
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data['retCode'] != 0:
            raise RuntimeError(f"API error: {data['retMsg']}")

        # Parse klines
        klines = data['result']['list']
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)

        # Sort by timestamp ascending
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def get_klines_history(
        self,
        symbol: str,
        interval: str,
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        """
        Fetch historical klines with pagination.

        Args:
            symbol: Trading pair symbol
            interval: Candlestick interval
            start_time: Start time as 'YYYY-MM-DD'
            end_time: End time as 'YYYY-MM-DD'

        Returns:
            DataFrame with OHLCV data
        """
        all_data = []
        current_end = pd.Timestamp(end_time)
        start_ts = pd.Timestamp(start_time)

        while current_end > start_ts:
            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                end_time=int(current_end.timestamp() * 1000),
            )

            if df.empty:
                break

            all_data.append(df)

            # Move window back
            oldest = df['timestamp'].min()
            if oldest <= start_ts:
                break
            current_end = oldest - timedelta(minutes=1)

            # Rate limiting
            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['timestamp'])
        result = result[result['timestamp'] >= start_ts]
        result = result[result['timestamp'] <= pd.Timestamp(end_time)]
        result = result.sort_values('timestamp').reset_index(drop=True)

        return result


class StockDataLoader:
    """
    Data loader for stock market data using yfinance.
    """

    def __init__(self):
        if not HAS_YFINANCE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

    def get_data(
        self,
        symbols: Union[str, List[str]],
        start: str,
        end: str,
        interval: str = '1d',
    ) -> pd.DataFrame:
        """
        Fetch stock data.

        Args:
            symbols: Single symbol or list of symbols
            start: Start date as 'YYYY-MM-DD'
            end: End date as 'YYYY-MM-DD'
            interval: Data interval ('1d', '1h', '5m', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        data = yf.download(
            symbols,
            start=start,
            end=end,
            interval=interval,
            progress=False,
        )

        if len(symbols) == 1:
            data.columns = data.columns.droplevel(1) if isinstance(data.columns, pd.MultiIndex) else data.columns

        return data


class FeatureEngineering:
    """
    Feature engineering for prototype learning.

    Computes technical indicators and normalizes features
    for input to the prototype network.
    """

    def __init__(self, lookback: int = 60):
        """
        Initialize feature engineering.

        Args:
            lookback: Lookback period for creating sequences
        """
        self.lookback = lookback

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical features from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with computed features
        """
        df = df.copy()

        # Returns
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_20'] = df['close'].pct_change(20)

        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # Price relative to MAs
        df['price_sma20_ratio'] = df['close'] / df['sma_20'] - 1
        df['price_sma50_ratio'] = df['close'] / df['sma_50'] - 1

        # Bollinger Bands
        df['bb_middle'] = df['sma_20']
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['close']

        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Volatility
        df['volatility'] = df['returns_1'].rolling(20).std() * np.sqrt(252)

        # Momentum
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)

        return df

    def label_market_states(
        self,
        df: pd.DataFrame,
        forward_returns_period: int = 5,
        threshold_trend: float = 0.02,
        threshold_breakout: float = 0.03,
    ) -> pd.DataFrame:
        """
        Label market states based on forward returns and volatility.

        Args:
            df: DataFrame with price data
            forward_returns_period: Period for calculating forward returns
            threshold_trend: Threshold for trend classification
            threshold_breakout: Threshold for breakout classification

        Returns:
            DataFrame with 'label' column
        """
        df = df.copy()

        # Forward returns
        df['forward_returns'] = df['close'].shift(-forward_returns_period) / df['close'] - 1

        # Volatility regime (rolling)
        df['vol_20'] = df['returns_1'].rolling(20).std()
        df['vol_regime'] = df['vol_20'] / df['vol_20'].rolling(60).mean()

        # Initialize labels
        df['label'] = 2  # Default: Consolidation

        # Bullish trend
        bullish_mask = (
            (df['forward_returns'] > threshold_trend) &
            (df['vol_regime'] < 1.5)
        )
        df.loc[bullish_mask, 'label'] = 0

        # Bearish trend
        bearish_mask = (
            (df['forward_returns'] < -threshold_trend) &
            (df['vol_regime'] < 1.5)
        )
        df.loc[bearish_mask, 'label'] = 1

        # Breakout up (high returns with high volatility)
        breakout_up_mask = (
            (df['forward_returns'] > threshold_breakout) &
            (df['vol_regime'] >= 1.5)
        )
        df.loc[breakout_up_mask, 'label'] = 3

        # Breakout down
        breakout_down_mask = (
            (df['forward_returns'] < -threshold_breakout) &
            (df['vol_regime'] >= 1.5)
        )
        df.loc[breakout_down_mask, 'label'] = 4

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str = 'label',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training.

        Args:
            df: DataFrame with features and labels
            feature_columns: List of feature column names
            label_column: Name of label column

        Returns:
            Tuple of (X, y) arrays
            X shape: (num_samples, num_features, lookback)
            y shape: (num_samples,)
        """
        df = df.dropna(subset=feature_columns + [label_column])

        X_list = []
        y_list = []

        for i in range(self.lookback, len(df)):
            # Get sequence
            sequence = df[feature_columns].iloc[i - self.lookback:i].values
            label = df[label_column].iloc[i]

            # Transpose to (features, time)
            X_list.append(sequence.T)
            y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        return X, y

    def normalize_features(
        self,
        X: np.ndarray,
        fit: bool = True,
    ) -> np.ndarray:
        """
        Normalize features using z-score normalization.

        Args:
            X: Input array of shape (num_samples, num_features, lookback)
            fit: If True, compute and store normalization parameters

        Returns:
            Normalized array
        """
        # Reshape to (samples * time, features) for normalization
        n_samples, n_features, lookback = X.shape
        X_flat = X.transpose(0, 2, 1).reshape(-1, n_features)

        if fit:
            self.feature_mean = X_flat.mean(axis=0)
            self.feature_std = X_flat.std(axis=0)
            # Avoid division by zero
            self.feature_std[self.feature_std < 1e-8] = 1.0

        X_normalized = (X_flat - self.feature_mean) / self.feature_std
        X_normalized = X_normalized.reshape(n_samples, lookback, n_features)
        X_normalized = X_normalized.transpose(0, 2, 1)

        return X_normalized.astype(np.float32)

    def get_default_features(self) -> List[str]:
        """Get default list of features for training."""
        return [
            'returns_1', 'returns_5', 'returns_20',
            'price_sma20_ratio', 'price_sma50_ratio',
            'bb_position', 'bb_width',
            'rsi', 'macd_hist',
            'atr_ratio', 'volume_ratio',
            'volatility', 'roc_10',
        ]


def prepare_data_for_training(
    df: pd.DataFrame,
    lookback: int = 60,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict:
    """
    Prepare data for training prototype network.

    Args:
        df: DataFrame with OHLCV data
        lookback: Sequence length
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation

    Returns:
        Dictionary with train/val/test data
    """
    fe = FeatureEngineering(lookback=lookback)

    # Compute features
    df = fe.compute_features(df)

    # Label market states
    df = fe.label_market_states(df)

    # Get feature columns
    feature_cols = fe.get_default_features()

    # Create sequences
    X, y = fe.create_sequences(df, feature_cols)

    # Split data
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Normalize (fit on training data only)
    X_train = fe.normalize_features(X_train, fit=True)
    X_val = fe.normalize_features(X_val, fit=False)
    X_test = fe.normalize_features(X_test, fit=False)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_engineering': fe,
    }
