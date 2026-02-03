"""
Feature Engineering Module for TimeParticle SSM

Provides functions to compute multiscale features and technical indicators
for financial time series data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def prepare_multiscale_features(
    df: pd.DataFrame,
    return_periods: Optional[List[int]] = None,
    sma_windows: Optional[List[int]] = None,
    volatility_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Prepare features at multiple time scales.

    Creates features that capture patterns at different temporal resolutions,
    which is essential for TimeParticle's multiscale processing.

    Args:
        df: DataFrame with OHLCV columns
        return_periods: Periods for return calculation
        sma_windows: Windows for simple moving averages
        volatility_windows: Windows for volatility calculation

    Returns:
        DataFrame with multiscale features
    """
    if return_periods is None:
        return_periods = [1, 5, 10, 20, 60]
    if sma_windows is None:
        sma_windows = [5, 10, 20, 50, 100]
    if volatility_windows is None:
        volatility_windows = [5, 10, 20, 50]

    features = df.copy()

    # Price returns at different scales
    for period in return_periods:
        if len(df) > period:
            features[f'return_{period}'] = df['close'].pct_change(period)

    # Simple moving averages and ratios at different scales
    for window in sma_windows:
        if len(df) > window:
            features[f'sma_{window}'] = df['close'].rolling(window).mean()
            features[f'sma_ratio_{window}'] = df['close'] / features[f'sma_{window}']

    # Volatility at different scales
    for window in volatility_windows:
        if len(df) > window:
            features[f'volatility_{window}'] = (
                df['close'].pct_change().rolling(window).std()
            )

    # Volume features at different scales
    for window in [5, 10, 20]:
        if len(df) > window:
            features[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            features[f'volume_ratio_{window}'] = df['volume'] / features[f'volume_sma_{window}']

    return features.dropna()


def compute_technical_indicators(
    df: pd.DataFrame,
    include_momentum: bool = True,
    include_volatility: bool = True,
    include_volume: bool = True,
    include_trend: bool = True
) -> pd.DataFrame:
    """
    Compute comprehensive technical indicators.

    Args:
        df: DataFrame with OHLCV columns
        include_momentum: Include momentum indicators
        include_volatility: Include volatility indicators
        include_volume: Include volume indicators
        include_trend: Include trend indicators

    Returns:
        DataFrame with technical indicators
    """
    features = df.copy()

    if include_momentum:
        features = _add_momentum_indicators(features)

    if include_volatility:
        features = _add_volatility_indicators(features)

    if include_volume:
        features = _add_volume_indicators(features)

    if include_trend:
        features = _add_trend_indicators(features)

    return features.dropna()


def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum-based indicators."""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Rate of Change
    for period in [5, 10, 20]:
        if len(df) > period:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100

    # Momentum
    for period in [10, 20]:
        if len(df) > period:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

    return df


def _add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility-based indicators."""
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    df['atr_ratio'] = df['atr_14'] / df['close']

    # Historical Volatility
    for window in [10, 20, 30]:
        if len(df) > window:
            df[f'hvol_{window}'] = df['close'].pct_change().rolling(window).std() * np.sqrt(252)

    return df


def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based indicators."""
    # On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv
    df['obv_sma'] = obv.rolling(window=20).mean()

    # Volume Rate of Change
    df['vroc'] = df['volume'].pct_change(10) * 100

    # Accumulation/Distribution Line
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    df['ad_line'] = (clv * df['volume']).cumsum()

    # Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()

    mfi_ratio = positive_flow / (negative_flow + 1e-10)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))

    return df


def _add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend-based indicators."""
    # Exponential Moving Averages
    for span in [9, 21, 50, 200]:
        if len(df) > span:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
            df[f'ema_ratio_{span}'] = df['close'] / df[f'ema_{span}']

    # EMA Crossovers
    if 'ema_9' in df.columns and 'ema_21' in df.columns:
        df['ema_9_21_cross'] = df['ema_9'] - df['ema_21']

    if 'ema_50' in df.columns and 'ema_200' in df.columns:
        df['ema_50_200_cross'] = df['ema_50'] - df['ema_200']

    # ADX (Average Directional Index)
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)

    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)

    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    return df


def compute_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price pattern features.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with pattern features
    """
    features = df.copy()

    # Candlestick patterns
    features['body'] = df['close'] - df['open']
    features['body_pct'] = features['body'] / df['open']
    features['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    features['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    features['range'] = df['high'] - df['low']

    # Body size relative to range
    features['body_to_range'] = features['body'].abs() / (features['range'] + 1e-10)

    # Gap features
    features['gap'] = df['open'] - df['close'].shift(1)
    features['gap_pct'] = features['gap'] / df['close'].shift(1)

    # High/Low position
    features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    # Inside/Outside bars
    features['inside_bar'] = (
        (df['high'] < df['high'].shift(1)) &
        (df['low'] > df['low'].shift(1))
    ).astype(int)

    features['outside_bar'] = (
        (df['high'] > df['high'].shift(1)) &
        (df['low'] < df['low'].shift(1))
    ).astype(int)

    return features


def create_target_variable(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.005,
    method: str = 'classification'
) -> pd.Series:
    """
    Create target variable for model training.

    Args:
        df: DataFrame with 'close' column
        horizon: Prediction horizon
        threshold: Threshold for buy/sell classification
        method: 'classification' or 'regression'

    Returns:
        Series with target values
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1

    if method == 'classification':
        target = pd.Series(1, index=df.index)  # Default: Hold
        target[future_return > threshold] = 2   # Buy
        target[future_return < -threshold] = 0  # Sell
        return target
    else:
        return future_return


def get_feature_columns(df: pd.DataFrame, exclude_base: bool = True) -> List[str]:
    """
    Get list of feature columns from DataFrame.

    Args:
        df: DataFrame with features
        exclude_base: Exclude base OHLCV columns

    Returns:
        List of feature column names
    """
    base_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']

    if exclude_base:
        return [c for c in df.columns if c not in base_cols]
    else:
        return list(df.columns)


if __name__ == "__main__":
    print("Testing Feature Engineering...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(end='2024-01-01', periods=500, freq='D')

    close = 100 + np.cumsum(np.random.randn(500) * 2)
    high = close + np.abs(np.random.randn(500) * 1.5)
    low = close - np.abs(np.random.randn(500) * 1.5)
    open_price = low + (high - low) * np.random.rand(500)
    volume = np.random.exponential(1000000, 500)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # Test multiscale features
    print("\n1. Testing multiscale features...")
    ms_features = prepare_multiscale_features(df)
    print(f"Original columns: {len(df.columns)}")
    print(f"With multiscale features: {len(ms_features.columns)}")
    print(f"New features: {[c for c in ms_features.columns if c not in df.columns][:10]}...")

    # Test technical indicators
    print("\n2. Testing technical indicators...")
    tech_features = compute_technical_indicators(df)
    print(f"With technical indicators: {len(tech_features.columns)}")
    print(f"Sample indicators: {[c for c in tech_features.columns if c not in df.columns][:10]}...")

    # Test price patterns
    print("\n3. Testing price patterns...")
    pattern_features = compute_price_patterns(df)
    print(f"With patterns: {len(pattern_features.columns)}")
    print(f"Pattern features: {[c for c in pattern_features.columns if c not in df.columns]}")

    # Test target creation
    print("\n4. Testing target creation...")
    target = create_target_variable(df, horizon=1, threshold=0.01)
    print(f"Target distribution: Sell={sum(target==0)}, Hold={sum(target==1)}, Buy={sum(target==2)}")

    # Get feature columns
    print("\n5. Testing feature column selection...")
    all_features = prepare_multiscale_features(compute_technical_indicators(df))
    feature_cols = get_feature_columns(all_features)
    print(f"Total feature columns: {len(feature_cols)}")

    print("\nAll tests passed!")
