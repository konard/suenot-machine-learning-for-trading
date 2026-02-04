"""
Feature Engineering module for trading.

This module provides functionality to compute technical indicators
and features from OHLCV data for use in ensemble models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Computes technical indicators and features from OHLCV data.

    Features include:
    - Returns at multiple timeframes
    - Volatility measures
    - Momentum indicators (RSI, MACD)
    - Trend indicators (Moving Averages, Bollinger Bands)
    - Volume features
    """

    def __init__(self, include_all: bool = True):
        """
        Initialize the feature engineer.

        Args:
            include_all: Whether to include all available features
        """
        self.include_all = include_all
        self.feature_names: List[str] = []

    def compute_returns(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Compute returns at multiple timeframes.

        Args:
            df: DataFrame with 'close' column
            periods: List of periods to compute returns for

        Returns:
            DataFrame with return columns added
        """
        if periods is None:
            periods = [1, 5, 10, 20, 60]

        df = df.copy()
        for period in periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        return df

    def compute_volatility(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Compute volatility measures.

        Args:
            df: DataFrame with return columns
            windows: Rolling window sizes

        Returns:
            DataFrame with volatility columns added
        """
        if windows is None:
            windows = [5, 10, 20, 60]

        df = df.copy()

        if 'return_1' not in df.columns:
            df['return_1'] = df['close'].pct_change()

        for window in windows:
            df[f'volatility_{window}'] = df['return_1'].rolling(window).std()
            df[f'volatility_{window}_annualized'] = df[f'volatility_{window}'] * np.sqrt(252 * 24)

        # Garman-Klass volatility
        for window in windows:
            log_hl = np.log(df['high'] / df['low'])
            log_co = np.log(df['close'] / df['open'])
            gk_var = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
            df[f'gk_volatility_{window}'] = np.sqrt(gk_var.rolling(window).mean())

        return df

    def compute_rsi(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Compute Relative Strength Index (RSI).

        Args:
            df: DataFrame with 'close' column
            periods: RSI periods

        Returns:
            DataFrame with RSI columns added
        """
        if periods is None:
            periods = [6, 14, 28]

        df = df.copy()
        delta = df['close'].diff()

        for period in periods:
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return df

    def compute_macd(
        self,
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
        df = df.copy()

        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()

        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_normalized'] = df['macd'] / df['close']

        return df

    def compute_bollinger_bands(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Compute Bollinger Bands.

        Args:
            df: DataFrame with 'close' column
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            DataFrame with Bollinger Band columns added
        """
        df = df.copy()

        df['bb_middle'] = df['close'].rolling(window).mean()
        rolling_std = df['close'].rolling(window).std()

        df['bb_upper'] = df['bb_middle'] + (rolling_std * num_std)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * num_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def compute_moving_averages(
        self,
        df: pd.DataFrame,
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Compute various moving averages.

        Args:
            df: DataFrame with 'close' column
            windows: MA periods

        Returns:
            DataFrame with MA columns added
        """
        if windows is None:
            windows = [5, 10, 20, 50, 100, 200]

        df = df.copy()

        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            df[f'sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']

        return df

    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume-based features.

        Args:
            df: DataFrame with 'volume' column

        Returns:
            DataFrame with volume feature columns added
        """
        df = df.copy()

        # Volume moving averages
        for window in [5, 10, 20]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']

        # Volume-weighted price
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_sma_10'] = df['obv'].rolling(10).mean()

        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change(5)

        return df

    def compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price-based features.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with price feature columns added
        """
        df = df.copy()

        # Candle features
        df['candle_body'] = (df['close'] - df['open']) / df['open']
        df['candle_range'] = (df['high'] - df['low']) / df['low']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

        # High-Low features
        for window in [5, 10, 20]:
            df[f'high_{window}'] = df['high'].rolling(window).max()
            df[f'low_{window}'] = df['low'].rolling(window).min()
            df[f'price_position_{window}'] = (df['close'] - df[f'low_{window}']) / \
                                              (df[f'high_{window}'] - df[f'low_{window}'] + 1e-8)

        # Distance from highs/lows
        df['dist_from_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['dist_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']

        return df

    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum-based features.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with momentum feature columns added
        """
        df = df.copy()

        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100

        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        # Average True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        for window in [7, 14, 21]:
            df[f'atr_{window}'] = true_range.rolling(window).mean()
            df[f'atr_{window}_ratio'] = df[f'atr_{window}'] / df['close']

        return df

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all available features.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all feature columns added
        """
        logger.info("Computing all features...")

        df = self.compute_returns(df)
        df = self.compute_volatility(df)
        df = self.compute_rsi(df)
        df = self.compute_macd(df)
        df = self.compute_bollinger_bands(df)
        df = self.compute_moving_averages(df)
        df = self.compute_volume_features(df)
        df = self.compute_price_features(df)
        df = self.compute_momentum_features(df)

        # Drop rows with NaN values
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values")

        # Store feature names
        self.feature_names = [col for col in df.columns
                             if col not in ['timestamp', 'open', 'high', 'low', 'close',
                                           'volume', 'symbol']]

        logger.info(f"Computed {len(self.feature_names)} features")
        return df

    def get_feature_names(self) -> List[str]:
        """Get list of computed feature names."""
        return self.feature_names

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'return_1',
        lookforward: int = 1,
        exclude_columns: Optional[List[str]] = None
    ) -> tuple:
        """
        Prepare features and target for training.

        Args:
            df: DataFrame with features
            target_column: Column to use as target (shifted)
            lookforward: Number of periods to look forward for target
            exclude_columns: Columns to exclude from features

        Returns:
            Tuple of (X, y, feature_names)
        """
        if exclude_columns is None:
            exclude_columns = ['timestamp', 'open', 'high', 'low', 'close',
                             'volume', 'symbol']

        df = df.copy()

        # Create target (future return)
        df['target'] = df['close'].pct_change(lookforward).shift(-lookforward)

        # Remove rows with NaN target
        df = df.dropna(subset=['target'])

        # Get feature columns
        feature_cols = [col for col in df.columns
                       if col not in exclude_columns + ['target', target_column]]

        X = df[feature_cols].values
        y = df['target'].values

        logger.info(f"Prepared training data: X shape = {X.shape}, y shape = {y.shape}")
        return X, y, feature_cols


def main():
    """Example usage of the FeatureEngineer."""
    # Create sample data
    np.random.seed(42)
    n = 1000

    dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    price = 50000 + np.cumsum(np.random.randn(n) * 100)
    volume = np.random.uniform(100, 1000, n)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(n) * 50,
        'high': price + np.abs(np.random.randn(n) * 100),
        'low': price - np.abs(np.random.randn(n) * 100),
        'close': price,
        'volume': volume
    })

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Compute all features
    df_features = fe.compute_all_features(df)

    print(f"\nDataFrame shape: {df_features.shape}")
    print(f"Number of features: {len(fe.get_feature_names())}")
    print(f"\nFeature names:\n{fe.get_feature_names()[:20]}...")
    print(f"\nSample data:\n{df_features.head()}")

    # Prepare training data
    X, y, feature_names = fe.prepare_training_data(df_features)
    print(f"\nTraining data shape: X={X.shape}, y={y.shape}")


if __name__ == '__main__':
    main()
