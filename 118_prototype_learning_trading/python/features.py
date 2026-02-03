"""
Feature engineering module for prototype learning trading.

Provides technical indicators, price patterns, and feature normalization
for preprocessing market data before input to prototype models.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    momentum_periods: List[int] = None
    volatility_periods: List[int] = None

    def __post_init__(self):
        if self.momentum_periods is None:
            self.momentum_periods = [5, 10, 20]
        if self.volatility_periods is None:
            self.volatility_periods = [10, 20, 50]


class TechnicalIndicators:
    """
    Technical indicator calculations.

    Provides methods for computing common technical indicators
    used in trading analysis.
    """

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Array of close prices
            period: RSI period

        Returns:
            RSI values (0-100 scale)
        """
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate rolling averages
        avg_gain = pd.Series(gains).rolling(window=period, min_periods=1).mean().values
        avg_loss = pd.Series(losses).rolling(window=period, min_periods=1).mean().values

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Prepend NaN for the first element (no delta available)
        rsi = np.concatenate([[np.nan], rsi])

        return rsi

    @staticmethod
    def macd(
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Array of close prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        prices_series = pd.Series(prices)

        ema_fast = prices_series.ewm(span=fast_period, adjust=False).mean().values
        ema_slow = prices_series.ewm(span=slow_period, adjust=False).mean().values

        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal_period, adjust=False).mean().values
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        prices: np.ndarray,
        period: int = 20,
        num_std: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Array of close prices
            period: Moving average period
            num_std: Number of standard deviations for bands

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        prices_series = pd.Series(prices)

        middle = prices_series.rolling(window=period).mean().values
        std = prices_series.rolling(window=period).std().values

        upper = middle + num_std * std
        lower = middle - num_std * std

        return upper, middle, lower

    @staticmethod
    def atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """
        Calculate Average True Range (ATR).

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            period: ATR period

        Returns:
            ATR values
        """
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value

        # ATR
        atr = pd.Series(tr).rolling(window=period).mean().values

        return atr

    @staticmethod
    def stochastic(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
            k_period: %K period
            d_period: %D period

        Returns:
            Tuple of (%K, %D)
        """
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)

        lowest_low = low_series.rolling(window=k_period).min()
        highest_high = high_series.rolling(window=k_period).max()

        k = 100 * (close_series - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(window=d_period).mean()

        return k.values, d.values

    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().values

    @staticmethod
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        return pd.Series(prices).rolling(window=period).mean().values


class PricePatterns:
    """
    Price pattern calculations.

    Computes returns, volatility, and momentum features.
    """

    @staticmethod
    def returns(prices: np.ndarray, period: int = 1) -> np.ndarray:
        """
        Calculate percentage returns.

        Args:
            prices: Array of prices
            period: Return period

        Returns:
            Returns array
        """
        returns = (prices[period:] - prices[:-period]) / prices[:-period]
        padding = np.full(period, np.nan)
        return np.concatenate([padding, returns])

    @staticmethod
    def log_returns(prices: np.ndarray, period: int = 1) -> np.ndarray:
        """
        Calculate log returns.

        Args:
            prices: Array of prices
            period: Return period

        Returns:
            Log returns array
        """
        log_returns = np.log(prices[period:] / prices[:-period])
        padding = np.full(period, np.nan)
        return np.concatenate([padding, log_returns])

    @staticmethod
    def volatility(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Calculate rolling volatility (standard deviation of returns).

        Args:
            prices: Array of prices
            period: Rolling window

        Returns:
            Volatility array
        """
        returns = PricePatterns.returns(prices, period=1)
        return pd.Series(returns).rolling(window=period).std().values

    @staticmethod
    def momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Calculate price momentum.

        Args:
            prices: Array of prices
            period: Momentum period

        Returns:
            Momentum array
        """
        momentum = prices - np.roll(prices, period)
        momentum[:period] = np.nan
        return momentum

    @staticmethod
    def rate_of_change(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """
        Calculate Rate of Change (ROC).

        Args:
            prices: Array of prices
            period: ROC period

        Returns:
            ROC array (percentage)
        """
        roc = ((prices - np.roll(prices, period)) / np.roll(prices, period)) * 100
        roc[:period] = np.nan
        return roc


class FeatureNormalizer:
    """
    Feature normalization utilities.

    Provides various normalization methods for preprocessing features.
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray) -> 'FeatureNormalizer':
        """
        Fit normalizer on features.

        Args:
            features: Array of shape (n_samples, n_features) or
                     (n_samples, n_features, sequence_length)

        Returns:
            self
        """
        if features.ndim == 3:
            # Reshape to 2D for statistics
            n_samples, n_features, seq_len = features.shape
            features_2d = features.transpose(0, 2, 1).reshape(-1, n_features)
        else:
            features_2d = features

        self.mean = features_2d.mean(axis=0)
        self.std = features_2d.std(axis=0)
        self.min = features_2d.min(axis=0)
        self.max = features_2d.max(axis=0)

        # Handle zero std
        self.std[self.std < 1e-8] = 1.0

        return self

    def transform(
        self,
        features: np.ndarray,
        method: str = 'zscore',
    ) -> np.ndarray:
        """
        Transform features using fitted parameters.

        Args:
            features: Features to normalize
            method: 'zscore', 'minmax', or 'robust'

        Returns:
            Normalized features
        """
        if self.mean is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        original_shape = features.shape
        if features.ndim == 3:
            n_samples, n_features, seq_len = features.shape
            features = features.transpose(0, 2, 1).reshape(-1, n_features)
            reshaped = True
        else:
            reshaped = False

        if method == 'zscore':
            normalized = (features - self.mean) / self.std
        elif method == 'minmax':
            normalized = (features - self.min) / (self.max - self.min + 1e-8)
        elif method == 'robust':
            # Clip outliers to 3 std
            normalized = np.clip(
                (features - self.mean) / self.std,
                -3, 3
            )
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if reshaped:
            n_samples = original_shape[0]
            n_features = original_shape[1]
            seq_len = original_shape[2]
            normalized = normalized.reshape(n_samples, seq_len, n_features)
            normalized = normalized.transpose(0, 2, 1)

        return normalized.astype(np.float32)

    def fit_transform(
        self,
        features: np.ndarray,
        method: str = 'zscore',
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features, method)

    def inverse_transform(
        self,
        features: np.ndarray,
        method: str = 'zscore',
    ) -> np.ndarray:
        """Inverse transform normalized features."""
        if method == 'zscore':
            return features * self.std + self.mean
        elif method == 'minmax':
            return features * (self.max - self.min) + self.min
        else:
            raise ValueError(f"Cannot inverse transform method: {method}")


class FeatureEngineer:
    """
    Complete feature engineering pipeline.

    Combines technical indicators, price patterns, and normalization
    into a unified preprocessing pipeline.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer.

        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        self.normalizer = FeatureNormalizer()
        self.feature_names: List[str] = []

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with all computed features
        """
        df = df.copy()

        # Ensure column names are lowercase
        df.columns = [c.lower() for c in df.columns]

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Returns
        for period in [1, 5, 10, 20]:
            df[f'returns_{period}'] = PricePatterns.returns(close, period)
            df[f'log_returns_{period}'] = PricePatterns.log_returns(close, period)

        # Volatility
        for period in self.config.volatility_periods:
            df[f'volatility_{period}'] = PricePatterns.volatility(close, period)

        # Momentum
        for period in self.config.momentum_periods:
            df[f'momentum_{period}'] = PricePatterns.momentum(close, period)
            df[f'roc_{period}'] = PricePatterns.rate_of_change(close, period)

        # RSI
        df['rsi'] = TechnicalIndicators.rsi(close, self.config.rsi_period)

        # MACD
        macd, signal, hist = TechnicalIndicators.macd(
            close,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal,
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            close,
            self.config.bb_period,
            self.config.bb_std,
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle

        # ATR
        df['atr'] = TechnicalIndicators.atr(high, low, close, self.config.atr_period)
        df['atr_ratio'] = df['atr'] / close

        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(high, low, close)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d

        # Moving averages
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = TechnicalIndicators.sma(close, period)
            df[f'ema_{period}'] = TechnicalIndicators.ema(close, period)
            df[f'price_sma_{period}_ratio'] = close / df[f'sma_{period}'] - 1

        # Volume features
        df['volume_sma'] = TechnicalIndicators.sma(volume, 20)
        df['volume_ratio'] = volume / df['volume_sma']
        df['volume_change'] = PricePatterns.returns(volume, 1)

        # Price range features
        df['high_low_range'] = (high - low) / close
        df['close_open_range'] = (close - df['open'].values) / df['open'].values

        # Fill NaN with 0 for first periods
        df = df.fillna(0)

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names for model input."""
        return [
            'returns_1', 'returns_5', 'returns_10', 'returns_20',
            'volatility_10', 'volatility_20',
            'momentum_5', 'momentum_10', 'roc_10',
            'rsi', 'macd_hist',
            'bb_position', 'bb_width',
            'atr_ratio',
            'stoch_k', 'stoch_d',
            'price_sma_20_ratio', 'price_sma_50_ratio',
            'volume_ratio',
            'high_low_range',
        ]

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        target_column: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for model input.

        Args:
            df: DataFrame with features
            sequence_length: Length of each sequence
            target_column: Column name for labels (optional)

        Returns:
            Tuple of (X, y) where X is (n_samples, sequence_length, n_features)
            and y is (n_samples,) or None if no target_column
        """
        feature_columns = self.get_feature_columns()
        self.feature_names = feature_columns

        # Get features
        features = df[feature_columns].values

        X_list = []
        y_list = []

        for i in range(sequence_length, len(features)):
            sequence = features[i - sequence_length:i]
            X_list.append(sequence)

            if target_column is not None:
                label = df[target_column].iloc[i]
                y_list.append(label)

        X = np.array(X_list, dtype=np.float32)

        if target_column is not None:
            y = np.array(y_list, dtype=np.int64)
            return X, y
        else:
            return X, None

    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Full data preparation pipeline.

        Args:
            df: Raw OHLCV DataFrame
            sequence_length: Sequence length
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            normalize: Whether to normalize features

        Returns:
            Dictionary with train/val/test splits
        """
        # Compute features
        df = self.compute_all_features(df)

        # Create labels (future returns direction)
        df['label'] = (df['returns_5'].shift(-5) > 0).astype(int)

        # Create sequences
        X, y = self.create_sequences(df, sequence_length, 'label')

        # Split data
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]

        # Normalize
        if normalize:
            X_train = self.normalizer.fit_transform(X_train)
            X_val = self.normalizer.transform(X_val)
            X_test = self.normalizer.transform(X_test)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
        }

    def extract_features_for_prediction(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract features for making predictions.

        Args:
            df: OHLCV DataFrame
            sequence_length: Sequence length
            normalize: Whether to normalize

        Returns:
            Feature array of shape (1, sequence_length, n_features)
        """
        # Compute features
        df = self.compute_all_features(df)

        # Get last sequence
        feature_columns = self.get_feature_columns()
        features = df[feature_columns].iloc[-sequence_length:].values

        X = features.reshape(1, sequence_length, -1).astype(np.float32)

        if normalize:
            if self.normalizer.mean is None:
                logger.warning("Normalizer not fitted, using raw features")
            else:
                X = self.normalizer.transform(X)

        return X


def label_market_regimes(
    df: pd.DataFrame,
    forward_period: int = 5,
    trend_threshold: float = 0.02,
    volatility_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Label market regimes based on forward returns and volatility.

    Regimes:
    - 0: BULLISH - Positive trend, normal volatility
    - 1: BEARISH - Negative trend, normal volatility
    - 2: SIDEWAYS - Small moves, normal volatility
    - 3: HIGH_VOLATILITY - High volatility regime
    - 4: LOW_VOLATILITY - Unusually low volatility

    Args:
        df: DataFrame with close prices
        forward_period: Period for forward returns
        trend_threshold: Threshold for trend classification
        volatility_threshold: Multiplier for high volatility

    Returns:
        DataFrame with 'regime' column
    """
    df = df.copy()

    # Calculate forward returns
    df['forward_return'] = df['close'].shift(-forward_period) / df['close'] - 1

    # Calculate volatility regime
    df['vol_20'] = df['close'].pct_change().rolling(20).std()
    df['vol_ratio'] = df['vol_20'] / df['vol_20'].rolling(60).mean()

    # Initialize labels
    df['regime'] = 2  # Default: SIDEWAYS

    # BULLISH
    df.loc[
        (df['forward_return'] > trend_threshold) &
        (df['vol_ratio'] < volatility_threshold) &
        (df['vol_ratio'] > 0.5),
        'regime'
    ] = 0

    # BEARISH
    df.loc[
        (df['forward_return'] < -trend_threshold) &
        (df['vol_ratio'] < volatility_threshold) &
        (df['vol_ratio'] > 0.5),
        'regime'
    ] = 1

    # HIGH_VOLATILITY
    df.loc[df['vol_ratio'] >= volatility_threshold, 'regime'] = 3

    # LOW_VOLATILITY
    df.loc[df['vol_ratio'] <= 0.5, 'regime'] = 4

    return df


if __name__ == "__main__":
    # Demo
    print("Feature Engineering Demo")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start='2023-01-01', periods=n, freq='1H')
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close * (1 + np.random.randn(n) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(n) * 0.01)),
        'low': close * (1 - np.abs(np.random.randn(n) * 0.01)),
        'close': close,
        'volume': np.random.lognormal(10, 1, n),
    })

    print(f"\nRaw data shape: {df.shape}")

    # Create feature engineer
    fe = FeatureEngineer()

    # Compute features
    df_features = fe.compute_all_features(df)
    print(f"Features computed: {len(df_features.columns)} columns")

    # Get feature columns
    feature_cols = fe.get_feature_columns()
    print(f"\nModel features ({len(feature_cols)}):")
    for col in feature_cols[:10]:
        print(f"  - {col}")
    print("  ...")

    # Create sequences
    X, _ = fe.create_sequences(df_features, sequence_length=60)
    print(f"\nSequence shape: {X.shape}")

    # Normalize
    X_norm = fe.normalizer.fit_transform(X)
    print(f"Normalized shape: {X_norm.shape}")
    print(f"Normalized mean: {X_norm.mean():.4f}, std: {X_norm.std():.4f}")

    # Label regimes
    df_labeled = label_market_regimes(df_features)
    print(f"\nRegime distribution:")
    print(df_labeled['regime'].value_counts())

    print("\n" + "=" * 50)
    print("Demo complete!")
