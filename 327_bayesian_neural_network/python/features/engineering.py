"""
Feature engineering for trading models.

Creates technical indicators and features from OHLCV data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


@dataclass
class TradingFeatures:
    """Container for trading features."""
    X: np.ndarray  # Feature matrix
    y_class: np.ndarray  # Classification labels (up/neutral/down)
    y_return: np.ndarray  # Continuous returns
    feature_names: List[str]
    timestamps: pd.DatetimeIndex


class FeatureEngineer:
    """
    Feature engineering for cryptocurrency trading.

    Creates a comprehensive set of features including:
    - Price returns at multiple horizons
    - Volume features
    - Volatility measures
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Price patterns

    Example:
        engineer = FeatureEngineer(lookback=20, prediction_horizon=5)
        features = engineer.create_features(ohlcv_df)
    """

    def __init__(
        self,
        lookback: int = 20,
        prediction_horizon: int = 5,
        return_threshold: float = 0.002,
        normalize: bool = True
    ):
        """
        Initialize feature engineer.

        Args:
            lookback: Number of periods for rolling calculations
            prediction_horizon: Periods ahead for prediction target
            return_threshold: Threshold for up/down classification
            normalize: Whether to normalize features
        """
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.return_threshold = return_threshold
        self.normalize = normalize
        self.feature_names: List[str] = []

    def create_features(self, df: pd.DataFrame) -> TradingFeatures:
        """
        Create comprehensive feature set from OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            TradingFeatures object
        """
        df = df.copy()

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # 1. Price Returns
        self._add_returns(df)

        # 2. Volume Features
        self._add_volume_features(df)

        # 3. Volatility Features
        self._add_volatility_features(df)

        # 4. Technical Indicators
        self._add_rsi(df)
        self._add_macd(df)
        self._add_bollinger_bands(df)
        self._add_atr(df)
        self._add_stochastic(df)

        # 5. Price Patterns
        self._add_price_patterns(df)

        # 6. Lagged Features
        self._add_lagged_features(df)

        # Create target
        df['future_return'] = df['close'].shift(-self.prediction_horizon) / df['close'] - 1

        # Drop NaN rows
        df = df.dropna()

        # Get feature columns (exclude non-feature columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                       'timestamp', 'future_return']
        self.feature_names = [c for c in df.columns if c not in exclude_cols]

        # Extract features and targets
        X = df[self.feature_names].values.astype(np.float32)
        y_return = df['future_return'].values.astype(np.float32)

        # Create classification labels
        y_class = np.where(
            y_return > self.return_threshold, 0,  # Up
            np.where(y_return < -self.return_threshold, 2, 1)  # Down, Neutral
        ).astype(np.int64)

        # Normalize features
        if self.normalize:
            X = self._normalize_features(X)

        timestamps = df.index

        return TradingFeatures(
            X=X,
            y_class=y_class,
            y_return=y_return,
            feature_names=self.feature_names,
            timestamps=timestamps
        )

    def _add_returns(self, df: pd.DataFrame):
        """Add return features at multiple horizons."""
        horizons = [1, 5, 10, 20]
        for h in horizons:
            df[f'return_{h}'] = df['close'].pct_change(h)
            df[f'log_return_{h}'] = np.log(df['close'] / df['close'].shift(h))

    def _add_volume_features(self, df: pd.DataFrame):
        """Add volume-based features."""
        # Volume change
        df['volume_change'] = df['volume'].pct_change()

        # Volume moving averages
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

        # On-Balance Volume (OBV) change
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_change'] = df['obv'].pct_change(5)

    def _add_volatility_features(self, df: pd.DataFrame):
        """Add volatility features."""
        # Rolling standard deviation of returns
        df['volatility_5'] = df['close'].pct_change().rolling(5).std()
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        df['volatility_20'] = df['close'].pct_change().rolling(20).std()

        # High-Low range
        df['range'] = (df['high'] - df['low']) / df['close']
        df['range_ma'] = df['range'].rolling(10).mean()

        # Parkinson volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2).rolling(20).mean()
        )

    def _add_rsi(self, df: pd.DataFrame, period: int = 14):
        """Add Relative Strength Index."""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]

    def _add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ):
        """Add MACD indicator."""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Normalize by price
        df['macd_normalized'] = df['macd'] / df['close']
        df['macd_signal_normalized'] = df['macd_signal'] / df['close']
        df['macd_histogram_normalized'] = df['macd_histogram'] / df['close']

    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0):
        """Add Bollinger Bands."""
        ma = df['close'].rolling(period).mean()
        std_dev = df['close'].rolling(period).std()

        df['bb_upper'] = ma + std * std_dev
        df['bb_lower'] = ma - std * std_dev
        df['bb_middle'] = ma

        # Position within bands (0 = at lower, 1 = at upper)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Band width (volatility measure)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    def _add_atr(self, df: pd.DataFrame, period: int = 14):
        """Add Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(period).mean()
        df['atr_normalized'] = df['atr'] / df['close']

    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """Add Stochastic Oscillator."""
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(d_period).mean()
        df['stoch_k_normalized'] = (df['stoch_k'] - 50) / 50
        df['stoch_d_normalized'] = (df['stoch_d'] - 50) / 50

    def _add_price_patterns(self, df: pd.DataFrame):
        """Add price pattern features."""
        # Candlestick features
        df['body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        # Normalize by range
        candle_range = df['high'] - df['low'] + 1e-10
        df['body_ratio'] = df['body'] / candle_range
        df['upper_shadow_ratio'] = df['upper_shadow'] / candle_range
        df['lower_shadow_ratio'] = df['lower_shadow'] / candle_range

        # Higher highs / lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
        df['higher_close'] = (df['close'] > df['close'].shift(1)).astype(float)

        # Consecutive moves
        df['consecutive_up'] = df['higher_close'].rolling(5).sum()
        df['consecutive_down'] = (1 - df['higher_close']).rolling(5).sum()

    def _add_lagged_features(self, df: pd.DataFrame):
        """Add lagged versions of key features."""
        lag_features = ['return_1', 'volume_ratio', 'rsi_normalized', 'bb_position']
        lags = [1, 2, 3, 5]

        for feature in lag_features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using robust scaling."""
        # Replace inf with nan
        X = np.where(np.isinf(X), np.nan, X)

        # Robust normalization (using median and IQR)
        for i in range(X.shape[1]):
            col = X[:, i]
            valid_mask = ~np.isnan(col)

            if valid_mask.sum() > 0:
                median = np.nanmedian(col)
                q75, q25 = np.nanpercentile(col[valid_mask], [75, 25])
                iqr = q75 - q25

                if iqr > 1e-10:
                    X[:, i] = (col - median) / iqr
                else:
                    X[:, i] = col - median

        # Replace remaining nan with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X


def create_sequences(
    X: np.ndarray,
    y_class: np.ndarray,
    y_return: np.ndarray,
    seq_length: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM/RNN models.

    Args:
        X: Feature matrix (n_samples, n_features)
        y_class: Class labels
        y_return: Return values
        seq_length: Sequence length

    Returns:
        X_seq: Sequences (n_sequences, seq_length, n_features)
        y_class_seq: Labels for sequences
        y_return_seq: Returns for sequences
    """
    n_samples = len(X)
    n_features = X.shape[1]
    n_sequences = n_samples - seq_length

    X_seq = np.zeros((n_sequences, seq_length, n_features), dtype=np.float32)
    y_class_seq = np.zeros(n_sequences, dtype=np.int64)
    y_return_seq = np.zeros(n_sequences, dtype=np.float32)

    for i in range(n_sequences):
        X_seq[i] = X[i:i + seq_length]
        y_class_seq[i] = y_class[i + seq_length - 1]
        y_return_seq[i] = y_return[i + seq_length - 1]

    return X_seq, y_class_seq, y_return_seq
