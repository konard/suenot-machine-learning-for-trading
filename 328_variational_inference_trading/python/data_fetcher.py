"""
Data fetching from Bybit using CCXT
"""

import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
from loguru import logger
import ta

from config import DataConfig


class BybitDataFetcher:
    """Fetches market data from Bybit exchange using CCXT"""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        logger.info(f"Initialized Bybit data fetcher for {len(self.config.symbols)} symbols")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (default from config)
            limit: Number of candles to fetch (default from config)
            since: Timestamp to start from (in milliseconds)

        Returns:
            DataFrame with OHLCV data
        """
        timeframe = timeframe or self.config.timeframe
        limit = limit or self.config.limit

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
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol

            logger.debug(f"Fetched {len(df)} candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols

        Args:
            symbols: List of trading pairs (default from config)
            timeframe: Candle timeframe
            limit: Number of candles per symbol

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        symbols = symbols or self.config.symbols
        data = {}

        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                data[symbol] = df
            time.sleep(0.1)  # Rate limiting

        logger.info(f"Fetched data for {len(data)}/{len(symbols)} symbols")
        return data

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional indicator columns
        """
        if df.empty:
            return df

        df = df.copy()

        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).average_true_range()

        # Trend indicators
        df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()

        # MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # OBV
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'], volume=df['volume']
        ).on_balance_volume()

        # Normalized features
        df['close_norm'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
        df['volume_norm'] = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Prepare feature matrix from DataFrame

        Args:
            df: DataFrame with technical indicators
            feature_columns: List of column names to use as features

        Returns:
            NumPy array of features
        """
        if feature_columns is None:
            feature_columns = [
                'returns', 'log_returns', 'volatility', 'atr',
                'rsi', 'macd_diff', 'bb_position', 'bb_width',
                'stoch_k', 'stoch_d', 'volume_ratio',
                'close_norm', 'volume_norm'
            ]

        # Filter available columns
        available_cols = [c for c in feature_columns if c in df.columns]
        features = df[available_cols].values

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    def create_sequences(
        self,
        features: np.ndarray,
        sequence_length: Optional[int] = None,
        prediction_horizon: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training

        Args:
            features: Feature matrix
            sequence_length: Length of input sequences
            prediction_horizon: Steps ahead to predict

        Returns:
            Tuple of (X, y) arrays
        """
        sequence_length = sequence_length or self.config.sequence_length
        prediction_horizon = prediction_horizon or self.config.prediction_horizon

        X, y = [], []

        for i in range(len(features) - sequence_length - prediction_horizon + 1):
            X.append(features[i:i + sequence_length])
            # Target: returns over prediction horizon
            y.append(features[i + sequence_length + prediction_horizon - 1, 0])  # returns column

        return np.array(X), np.array(y)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/validation/test sets

        Args:
            X: Feature sequences
            y: Target values

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(X)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]

        X_test = X[val_end:]
        y_test = y[val_end:]

        logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test


def fetch_and_prepare_data(config: Optional[DataConfig] = None) -> Dict:
    """
    Main function to fetch and prepare all data

    Args:
        config: Data configuration

    Returns:
        Dictionary with prepared data
    """
    fetcher = BybitDataFetcher(config)

    # Fetch data for all symbols
    raw_data = fetcher.fetch_all_symbols()

    prepared_data = {}
    for symbol, df in raw_data.items():
        # Add indicators
        df_with_indicators = fetcher.add_technical_indicators(df)

        # Prepare features
        features = fetcher.prepare_features(df_with_indicators)

        # Create sequences
        X, y = fetcher.create_sequences(features)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = fetcher.split_data(X, y)

        prepared_data[symbol] = {
            'raw': df,
            'features': df_with_indicators,
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
        }

    return prepared_data


if __name__ == "__main__":
    # Test data fetching
    from loguru import logger
    import sys

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    config = DataConfig(symbols=["BTC/USDT", "ETH/USDT"], limit=100)
    fetcher = BybitDataFetcher(config)

    print("Fetching OHLCV data...")
    df = fetcher.fetch_ohlcv("BTC/USDT")
    print(f"\nRaw data shape: {df.shape}")
    print(df.head())

    print("\nAdding technical indicators...")
    df_with_indicators = fetcher.add_technical_indicators(df)
    print(f"With indicators shape: {df_with_indicators.shape}")
    print(df_with_indicators.columns.tolist())

    print("\nPreparing features...")
    features = fetcher.prepare_features(df_with_indicators)
    print(f"Features shape: {features.shape}")

    print("\nCreating sequences...")
    X, y = fetcher.create_sequences(features)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = fetcher.split_data(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
