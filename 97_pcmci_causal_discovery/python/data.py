"""
Data Loading Utilities for PCMCI Causal Discovery

Provides:
- load_stock_data: Load stock data from yfinance
- load_bybit_data: Load crypto data from Bybit REST API
- prepare_causal_data: Prepare data for PCMCI analysis
- create_sequences: Create time-lagged sequences
- generate_synthetic_causal_data: Generate synthetic data with known causal structure
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def load_stock_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1d'
) -> Dict[str, pd.DataFrame]:
    """
    Load stock data from yfinance.

    Args:
        symbols: List of stock symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data frequency ('1m', '5m', '15m', '30m', '1h', '1d')

    Returns:
        Dictionary mapping symbols to DataFrames with OHLCV + features

    Example:
        data = load_stock_data(['AAPL', 'MSFT'], '2023-01-01', '2024-01-01')
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    data = {}

    for symbol in symbols:
        logger.info(f"Loading {symbol}...")
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                continue

            df = df.reset_index()

            # Normalize column names
            df.columns = [
                c.lower() if isinstance(c, str) else c[0].lower()
                for c in df.columns
            ]

            # Rename datetime column
            if 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'timestamp'})
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})

            # Add features
            df = _add_features(df)

            data[symbol] = df.dropna()
            logger.info(f"Loaded {len(df)} rows for {symbol}")

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    return data


def load_bybit_data(
    symbols: List[str],
    interval: str = '60',
    limit: int = 1000
) -> Dict[str, pd.DataFrame]:
    """
    Load cryptocurrency data from Bybit REST API.

    Uses the public Bybit v5 API endpoint to fetch kline (candlestick) data
    without requiring authentication.

    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        interval: Candle interval in minutes ('1', '5', '15', '30', '60', '240', 'D', 'W')
        limit: Number of candles to fetch (max 1000)

    Returns:
        Dictionary mapping symbols to DataFrames with OHLCV + features

    Example:
        data = load_bybit_data(['BTCUSDT', 'ETHUSDT'], interval='60', limit=500)
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required. Install with: pip install requests")

    base_url = "https://api.bybit.com/v5/market/kline"
    data = {}

    for symbol in symbols:
        logger.info(f"Loading {symbol} from Bybit...")
        try:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)
            }

            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result.get('retCode') != 0:
                logger.error(
                    f"Bybit API error for {symbol}: {result.get('retMsg', 'Unknown')}"
                )
                continue

            rows = result.get('result', {}).get('list', [])
            if not rows:
                logger.warning(f"No data found for {symbol}")
                continue

            # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
            df = pd.DataFrame(rows, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])

            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)

            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')

            # Sort by timestamp (Bybit returns newest first)
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Add features
            df = _add_features(df)

            data[symbol] = df.dropna()
            logger.info(f"Loaded {len(data[symbol])} rows for {symbol}")

        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")

    return data


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical features to OHLCV data.

    Computes returns, volatility, volume indicators, moving averages,
    and RSI for use in causal analysis.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()

    # Log returns
    df['returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility (rolling standard deviation of returns)
    df['volatility'] = df['returns'].rolling(20).std()

    # Volume features
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_change'] = df['volume'] / (df['volume_ma'] + 1e-8)
    df['volume_zscore'] = (
        (df['volume'] - df['volume_ma']) /
        (df['volume'].rolling(20).std() + 1e-8)
    )

    # Price features
    df['price_range'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    df['price_position'] = (
        (df['close'] - df['low']) /
        (df['high'] - df['low'] + 1e-8)
    )

    # Moving averages
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_ratio'] = df['ma_5'] / (df['ma_20'] + 1e-8)

    # RSI
    df['rsi'] = _calculate_rsi(df['close'], 14)

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        prices: Price series
        period: RSI period (default 14)

    Returns:
        RSI values normalized to [0, 1]
    """
    delta = prices.diff()

    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()

    rs = avg_gains / (avg_losses + 1e-8)
    rsi = 100 - (100 / (1 + rs))

    return rsi / 100  # Normalize to [0, 1]


def prepare_causal_data(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Prepare data for PCMCI causal discovery analysis.

    Selects columns, handles missing values, and optionally normalizes
    the data to zero mean and unit variance.

    Args:
        df: Input DataFrame with time series data
        columns: List of column names to include. If None, uses a default set.
        normalize: Whether to standardize the data (z-score normalization)

    Returns:
        numpy array of shape (n_timesteps, n_variables) ready for PCMCI

    Example:
        data = prepare_causal_data(df, columns=['returns', 'volatility', 'volume_change'])
        graph = pcmci.fit(data)
    """
    if columns is None:
        # Default feature set for causal analysis
        default_columns = [
            'returns', 'volatility', 'volume_change',
            'price_range', 'rsi', 'macd_hist'
        ]
        columns = [c for c in default_columns if c in df.columns]

    if not columns:
        raise ValueError("No valid columns found in DataFrame")

    # Extract data
    data = df[columns].values.astype(np.float64)

    # Handle missing values
    if np.any(np.isnan(data)):
        logger.warning("Data contains NaN values, forward-filling then dropping remaining")
        df_subset = df[columns].ffill().bfill()
        data = df_subset.values.astype(np.float64)

    # Remove any remaining NaN rows
    valid_mask = ~np.any(np.isnan(data), axis=1)
    data = data[valid_mask]

    if len(data) == 0:
        raise ValueError("No valid data remaining after NaN removal")

    # Normalize
    if normalize:
        means = data.mean(axis=0)
        stds = data.std(axis=0)
        stds[stds < 1e-10] = 1.0
        data = (data - means) / stds
        logger.info(f"Normalized {data.shape[1]} variables, {data.shape[0]} timesteps")

    return data


def create_sequences(
    data: np.ndarray,
    seq_len: int = 50,
    horizon: int = 1,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create time-lagged sequences for supervised learning from causal data.

    Args:
        data: Input data, shape (n_timesteps, n_variables)
        seq_len: Length of input sequence (lookback window)
        horizon: Number of steps ahead to predict
        stride: Step size between consecutive sequences

    Returns:
        X: Input sequences, shape (n_samples, seq_len, n_variables)
        y: Target values, shape (n_samples, n_variables)

    Example:
        X, y = create_sequences(data, seq_len=50, horizon=1)
    """
    X, y = [], []

    for i in range(0, len(data) - seq_len - horizon + 1, stride):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len + horizon - 1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(f"Created {len(X)} sequences: X={X.shape}, y={y.shape}")
    return X, y


def generate_synthetic_causal_data(
    n_vars: int = 4,
    n_samples: int = 1000,
    true_links: Optional[List[Tuple[int, int, int, float]]] = None,
    noise_std: float = 0.3,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[Tuple[int, int, int, float]]]:
    """
    Generate synthetic time series data with known causal structure.

    Creates multivariate time series where specific lagged causal
    relationships are embedded, useful for testing and validation.

    Args:
        n_vars: Number of variables
        n_samples: Number of time steps to generate
        true_links: List of (source, target, lag, coefficient) tuples
                    defining the true causal structure. If None, a default
                    structure is used.
        noise_std: Standard deviation of the innovation noise
        seed: Random seed for reproducibility

    Returns:
        data: Generated time series, shape (n_samples, n_vars)
        true_links: The true causal links used for generation

    Example:
        data, links = generate_synthetic_causal_data(
            n_vars=4,
            n_samples=500,
            true_links=[(0, 1, 1, 0.6), (1, 2, 2, 0.4)]
        )
    """
    if seed is not None:
        np.random.seed(seed)

    if true_links is None:
        # Default causal structure
        true_links = [
            (0, 1, 1, 0.6),   # X0(t-1) -> X1(t)
            (1, 2, 2, 0.4),   # X1(t-2) -> X2(t)
            (0, 3, 1, 0.3),   # X0(t-1) -> X3(t)
            (2, 3, 1, 0.25),  # X2(t-1) -> X3(t)
        ]

    # Determine max lag
    max_lag = max(lag for _, _, lag, _ in true_links) if true_links else 1

    # Generate data
    data = np.random.randn(n_samples, n_vars) * noise_std

    for t in range(max_lag, n_samples):
        for source, target, lag, coeff in true_links:
            data[t, target] += coeff * data[t - lag, source]

    logger.info(
        f"Generated synthetic data: {n_samples} samples, {n_vars} variables, "
        f"{len(true_links)} true causal links"
    )

    return data, true_links


def test_stationarity(
    series: np.ndarray,
    significance: float = 0.05
) -> Dict[str, Any]:
    """
    Test stationarity using the Augmented Dickey-Fuller (ADF) test.

    Args:
        series: 1D time series data
        significance: Significance level for the test

    Returns:
        Dictionary with test statistic, p-value, and whether the series
        is stationary at the given significance level.

    Example:
        result = test_stationarity(data[:, 0])
        print(f"Stationary: {result['is_stationary']}")
    """
    from scipy import stats

    # Simple ADF-like test using OLS regression on differenced series
    # For a proper ADF test, use statsmodels if available
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series, autolag='AIC')
        return {
            'adf_statistic': float(result[0]),
            'p_value': float(result[1]),
            'n_lags_used': int(result[2]),
            'n_obs': int(result[3]),
            'critical_values': result[4],
            'is_stationary': result[1] < significance
        }
    except ImportError:
        pass

    # Fallback: simple unit root test using regression
    n = len(series)
    y = series[1:]
    y_lag = series[:-1]

    # Regress delta_y on y_lag
    delta_y = y - y_lag

    # OLS: delta_y = alpha + beta * y_lag
    X = np.column_stack([y_lag, np.ones(len(y_lag))])
    beta, _, _, _ = np.linalg.lstsq(X, delta_y, rcond=None)

    # Test statistic
    residuals = delta_y - X @ beta
    se = np.sqrt(np.sum(residuals ** 2) / (n - 3))
    se_beta = se / np.sqrt(np.sum((y_lag - y_lag.mean()) ** 2))
    t_stat = beta[0] / (se_beta + 1e-10)

    # Approximate critical values for ADF test
    # These are rough approximations
    critical_1 = -3.43
    critical_5 = -2.86
    critical_10 = -2.57

    is_stationary = t_stat < critical_5

    return {
        'adf_statistic': float(t_stat),
        'p_value': float(0.01 if t_stat < critical_1 else
                         0.05 if t_stat < critical_5 else
                         0.10 if t_stat < critical_10 else 0.50),
        'critical_values': {
            '1%': critical_1,
            '5%': critical_5,
            '10%': critical_10
        },
        'is_stationary': is_stationary
    }


def make_stationary(
    data: np.ndarray,
    method: str = 'diff'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Transform data to achieve stationarity.

    Args:
        data: Input data, shape (n_timesteps, n_variables)
        method: Method for stationarity transformation
                ('diff' for first differencing, 'log_diff' for log returns)

    Returns:
        transformed_data: Stationary data
        info: Dictionary with transformation details

    Example:
        stationary_data, info = make_stationary(data, method='diff')
    """
    info = {'method': method, 'original_shape': data.shape}

    if method == 'diff':
        transformed = np.diff(data, axis=0)
        info['rows_removed'] = 1
    elif method == 'log_diff':
        # Log returns (only for positive data)
        data_positive = np.maximum(data, 1e-10)
        transformed = np.diff(np.log(data_positive), axis=0)
        info['rows_removed'] = 1
    else:
        raise ValueError(f"Unknown method: {method}. Use 'diff' or 'log_diff'")

    info['transformed_shape'] = transformed.shape

    # Test stationarity of each variable
    stationarity_results = []
    for i in range(transformed.shape[1]):
        result = test_stationarity(transformed[:, i])
        stationarity_results.append(result['is_stationary'])

    info['stationarity_results'] = stationarity_results
    info['all_stationary'] = all(stationarity_results)

    if not info['all_stationary']:
        non_stationary = [
            i for i, s in enumerate(stationarity_results) if not s
        ]
        logger.warning(
            f"Variables {non_stationary} may still be non-stationary "
            f"after {method} transformation"
        )

    return transformed, info


def merge_multi_asset_data(
    data: Dict[str, pd.DataFrame],
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge multiple asset DataFrames into a single DataFrame for causal analysis.

    Each asset's columns are prefixed with the symbol name.

    Args:
        data: Dictionary mapping symbols to DataFrames
        columns: Columns to include from each asset

    Returns:
        merged_df: Merged DataFrame aligned on timestamps
        column_names: List of merged column names (symbol_feature format)

    Example:
        merged, names = merge_multi_asset_data(data, columns=['returns', 'volatility'])
    """
    if columns is None:
        columns = ['returns', 'volatility', 'volume_change']

    symbols = list(data.keys())
    merged_frames = []
    column_names = []

    for symbol in symbols:
        df = data[symbol]
        available = [c for c in columns if c in df.columns]

        subset = df[['timestamp'] + available].copy()
        rename_map = {c: f"{symbol}_{c}" for c in available}
        subset = subset.rename(columns=rename_map)
        column_names.extend(rename_map.values())

        merged_frames.append(subset)

    # Merge on timestamp
    if not merged_frames:
        raise ValueError("No data to merge")

    merged = merged_frames[0]
    for frame in merged_frames[1:]:
        merged = pd.merge(merged, frame, on='timestamp', how='inner')

    merged = merged.sort_values('timestamp').reset_index(drop=True)

    logger.info(
        f"Merged {len(symbols)} assets: "
        f"{len(merged)} common timestamps, {len(column_names)} features"
    )

    return merged, column_names


if __name__ == "__main__":
    # Test data loading utilities
    print("Testing data loading utilities...")

    # Test synthetic data generation
    print("\n1. Testing synthetic data generation...")
    data, true_links = generate_synthetic_causal_data(
        n_vars=4,
        n_samples=500,
        seed=42
    )
    print(f"   Data shape: {data.shape}")
    print(f"   True links:")
    for src, tgt, lag, coeff in true_links:
        print(f"     X{src}(t-{lag}) -> X{tgt}(t): {coeff:.2f}")

    # Test stationarity
    print("\n2. Testing stationarity test...")
    for i in range(data.shape[1]):
        result = test_stationarity(data[:, i])
        print(f"   Variable {i}: stationary={result['is_stationary']}, "
              f"p={result['p_value']:.4f}")

    # Test prepare_causal_data with synthetic OHLCV
    print("\n3. Testing data preparation...")
    n = 500
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(n) * 0.1,
        'high': price + np.abs(np.random.randn(n) * 0.5),
        'low': price - np.abs(np.random.randn(n) * 0.5),
        'close': price,
        'volume': np.random.exponential(1000, n)
    })
    df = _add_features(df)
    df = df.dropna()

    causal_data = prepare_causal_data(
        df,
        columns=['returns', 'volatility', 'volume_change', 'rsi'],
        normalize=True
    )
    print(f"   Prepared causal data shape: {causal_data.shape}")

    # Test create_sequences
    print("\n4. Testing sequence creation...")
    X, y = create_sequences(causal_data, seq_len=30, horizon=1)
    print(f"   X shape: {X.shape}, y shape: {y.shape}")

    # Test make_stationary
    print("\n5. Testing stationarity transformation...")
    prices = np.column_stack([price[:400], price[:400] * 1.1])
    stationary_data, info = make_stationary(prices, method='diff')
    print(f"   Original shape: {info['original_shape']}")
    print(f"   Transformed shape: {info['transformed_shape']}")
    print(f"   All stationary: {info['all_stationary']}")

    print("\nAll tests passed!")
