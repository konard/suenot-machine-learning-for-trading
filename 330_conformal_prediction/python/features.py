"""
Feature Engineering for Conformal Prediction Trading

Creates technical indicators and features from OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class FeatureEngineer:
    """
    Creates features from OHLCV data for prediction models.
    """

    def __init__(self, prediction_horizon: int = 5):
        """
        Initialize feature engineer.

        Args:
            prediction_horizon: Number of periods ahead to predict
        """
        self.prediction_horizon = prediction_horizon

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features added
        """
        df = df.copy()

        # Price returns
        df = self._add_returns(df)

        # Volatility features
        df = self._add_volatility(df)

        # Momentum indicators
        df = self._add_momentum(df)

        # Volume features
        df = self._add_volume_features(df)

        # Trend indicators
        df = self._add_trend_indicators(df)

        # Target variable
        df = self._add_target(df)

        # Drop NaN rows
        df.dropna(inplace=True)

        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return features at multiple timeframes."""
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)

        # Log returns
        df['log_returns_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_returns_5'] = np.log(df['close'] / df['close'].shift(5))

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        # Rolling standard deviation of returns
        df['volatility_10'] = df['returns_1'].rolling(10).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        df['volatility_50'] = df['returns_1'].rolling(50).std()

        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        df['atr_norm'] = df['atr_14'] / df['close']

        # Bollinger Band width
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_width'] = (4 * std_20) / sma_20

        # Realized volatility
        df['realized_vol_20'] = df['log_returns_1'].rolling(20).std() * np.sqrt(252)

        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_7'] = self._calculate_rsi(df['close'], 7)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_norm'] = df['macd'] / df['close']

        # Momentum
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

        # Rate of Change
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)

        # Stochastic oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

        # On-Balance Volume (simplified)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_norm'] = df['obv'] / df['obv'].rolling(20).mean()

        # Volume-weighted price
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # Simple Moving Averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        # Exponential Moving Averages
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

        # Price relative to moving averages
        df['price_sma_10_ratio'] = df['close'] / df['sma_10'] - 1
        df['price_sma_20_ratio'] = df['close'] / df['sma_20'] - 1
        df['price_sma_50_ratio'] = df['close'] / df['sma_50'] - 1

        # Moving average crossovers
        df['sma_cross_10_20'] = df['sma_10'] - df['sma_20']
        df['sma_cross_20_50'] = df['sma_20'] - df['sma_50']
        df['sma_cross_10_20_norm'] = df['sma_cross_10_20'] / df['close']

        # Bollinger Bands position
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + 2 * std_20
        df['bb_lower'] = sma_20 - 2 * std_20
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable (future returns)."""
        df['target'] = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        df['target_direction'] = np.sign(df['target'])

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return [
            # Returns
            'returns_1', 'returns_5', 'returns_10', 'returns_20',
            'log_returns_1', 'log_returns_5',

            # Volatility
            'volatility_10', 'volatility_20', 'volatility_50',
            'atr_norm', 'bb_width', 'realized_vol_20',

            # Momentum
            'rsi_14', 'rsi_7', 'macd_norm', 'macd_hist',
            'momentum_10', 'momentum_20', 'roc_10',
            'stoch_k', 'stoch_d',

            # Volume
            'volume_ratio', 'obv_norm', 'vwap_deviation',

            # Trend
            'price_sma_10_ratio', 'price_sma_20_ratio', 'price_sma_50_ratio',
            'sma_cross_10_20_norm', 'bb_position',
        ]

    def prepare_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        cal_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Prepare data for conformal prediction.

        Args:
            df: DataFrame with features and target
            train_ratio: Ratio of data for training
            cal_ratio: Ratio of data for calibration

        Returns:
            X_train, y_train, X_cal, y_cal, X_test, y_test, test_index
        """
        feature_cols = self.get_feature_columns()

        # Ensure all features exist
        available_features = [col for col in feature_cols if col in df.columns]

        X = df[available_features].values
        y = df['target'].values

        n = len(X)
        train_end = int(n * train_ratio)
        cal_end = int(n * (train_ratio + cal_ratio))

        X_train = X[:train_end]
        y_train = y[:train_end]

        X_cal = X[train_end:cal_end]
        y_cal = y[train_end:cal_end]

        X_test = X[cal_end:]
        y_test = y[cal_end:]

        test_index = df.index[cal_end:]

        return X_train, y_train, X_cal, y_cal, X_test, y_test, test_index


if __name__ == "__main__":
    # Example usage
    from data_fetcher import BybitDataFetcher

    fetcher = BybitDataFetcher()
    df = fetcher.fetch_historical_data('BTC/USDT:USDT', timeframe='1h', days=30)

    engineer = FeatureEngineer(prediction_horizon=5)
    df_features = engineer.create_features(df)

    print(f"Features shape: {df_features.shape}")
    print(f"\nFeature columns: {engineer.get_feature_columns()}")
    print(f"\nSample data:")
    print(df_features[engineer.get_feature_columns()[:5]].tail())
