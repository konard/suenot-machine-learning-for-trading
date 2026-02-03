"""
Data loading utilities for Feature Attribution Trading.

Supports:
- Bybit API for cryptocurrency OHLCV data
- yfinance for stock market data
- Technical indicator calculation
- Feature engineering for trading models

All data is returned as pandas DataFrames with standardized column names.
"""

import numpy as np
import pandas as pd
import requests
import logging
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default feature names for the attribution model
DEFAULT_FEATURE_NAMES = [
    "returns", "log_returns", "volatility", "volume_change",
    "rsi", "macd", "macd_signal", "sma_ratio", "ema_ratio", "atr"
]


@dataclass
class MarketData:
    """Container for market OHLCV data with feature engineering."""
    df: pd.DataFrame
    symbol: str
    interval: str
    source: str

    @property
    def close(self) -> np.ndarray:
        """Get close prices."""
        return self.df["close"].values

    @property
    def returns(self) -> np.ndarray:
        """Calculate returns series."""
        return self.df["close"].pct_change().fillna(0).values

    @property
    def volume(self) -> np.ndarray:
        """Get volume series."""
        return self.df["volume"].values

    @property
    def volatility(self) -> np.ndarray:
        """Rolling volatility (20-period standard deviation of returns)."""
        returns = pd.Series(self.returns)
        return returns.rolling(20).std().fillna(returns.std()).values

    def to_ohlcv(self) -> np.ndarray:
        """
        Convert to OHLCV feature matrix.

        Returns:
            Feature matrix of shape (n_samples, 5)
        """
        return self.df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)

    def to_features(
        self,
        feature_names: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Convert to feature matrix for model input.

        Args:
            feature_names: List of feature names to include
            normalize: Whether to normalize features

        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        if feature_names is None:
            feature_names = DEFAULT_FEATURE_NAMES

        # Add technical features if not present
        if "returns" not in self.df.columns:
            self._add_technical_features()

        # Select features that exist
        available_features = [f for f in feature_names if f in self.df.columns]
        features = self.df[available_features].values.copy()

        if normalize:
            # Z-score normalization
            mean = np.nanmean(features, axis=0, keepdims=True)
            std = np.nanstd(features, axis=0, keepdims=True) + 1e-8
            features = (features - mean) / std

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features.astype(np.float32)

    def _add_technical_features(self) -> None:
        """Add technical indicator features to the dataframe."""
        df = self.df

        # Price returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Volatility
        df["volatility"] = df["returns"].rolling(20).std()

        # Volume features
        df["volume_change"] = df["volume"].pct_change()
        df["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # RSI (Relative Strength Index)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"] / 100  # Normalize to 0-1

        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = (ema_12 - ema_26) / df["close"]  # Normalized
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Moving average ratios
        df["sma_ratio"] = df["close"] / df["close"].rolling(20).mean() - 1
        df["ema_ratio"] = df["close"] / df["close"].ewm(span=20).mean() - 1

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df["atr"] = true_range.rolling(14).mean() / df["close"]  # Normalized

        # Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = (sma_20 + 2 * std_20 - df["close"]) / df["close"]
        df["bb_lower"] = (df["close"] - sma_20 + 2 * std_20) / df["close"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"])

        # Momentum
        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_10"] = df["close"].pct_change(10)
        df["momentum_20"] = df["close"].pct_change(20)

        # Stochastic Oscillator
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = (df["close"] - low_14) / (high_14 - low_14 + 1e-8)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # On-Balance Volume trend
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        df["obv_ma_ratio"] = df["obv"] / df["obv"].rolling(20).mean() - 1

        # Fill NaN values
        df.fillna(method="ffill", inplace=True)
        df.fillna(0, inplace=True)

    def add_labels(
        self,
        lookahead: int = 5,
        threshold: float = 0.01,
    ) -> "MarketData":
        """
        Add classification labels based on future returns.

        Args:
            lookahead: Number of periods to look ahead
            threshold: Threshold for BUY/SELL classification

        Returns:
            MarketData with labels column
        """
        df = self.df.copy()

        # Calculate future returns
        future_returns = df["close"].pct_change(lookahead).shift(-lookahead)

        # Create labels: 0=BUY, 1=HOLD, 2=SELL
        labels = np.where(
            future_returns > threshold, 0,
            np.where(future_returns < -threshold, 2, 1)
        )

        df["label"] = labels
        df["future_return"] = future_returns

        return MarketData(df=df, symbol=self.symbol, interval=self.interval, source=self.source)


class BybitDataLoader:
    """
    Data loader for Bybit exchange.

    Fetches OHLCV (kline) data from the Bybit public API v5.
    No API key required for public market data.
    """

    BASE_URL = "https://api.bybit.com"

    # Valid intervals for Bybit API
    VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]

    def __init__(self, timeout: int = 10):
        """
        Initialize Bybit data loader.

        Args:
            timeout: Request timeout in seconds
        """
        self.session = requests.Session()
        self.timeout = timeout

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",
        limit: int = 1000,
        category: str = "linear",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> MarketData:
        """
        Fetch kline (OHLCV) data from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of data points (max 1000)
            category: Market category ("linear", "inverse", "spot")
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            MarketData object with OHLCV DataFrame
        """
        if interval not in self.VALID_INTERVALS:
            logger.warning(f"Invalid interval {interval}, using 60")
            interval = "60"

        endpoint = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = self.session.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                logger.warning(f"Bybit API error: {data.get('retMsg')}")
                return self._generate_synthetic(symbol, interval, limit)

            records = data["result"]["list"]

            if not records:
                logger.warning(f"No data returned for {symbol}")
                return self._generate_synthetic(symbol, interval, limit)

            # Parse response
            df = pd.DataFrame(records, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df = df.drop(columns=["turnover"])

            # Convert types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)

            logger.info(f"Fetched {len(df)} klines for {symbol} from Bybit")
            return MarketData(df=df, symbol=symbol, interval=interval, source="bybit")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error fetching from Bybit: {e}")
            return self._generate_synthetic(symbol, interval, limit)
        except Exception as e:
            logger.warning(f"Failed to fetch from Bybit: {e}. Using synthetic data.")
            return self._generate_synthetic(symbol, interval, limit)

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "60",
        limit: int = 1000,
    ) -> Dict[str, MarketData]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of trading pairs
            interval: Kline interval
            limit: Number of data points per symbol

        Returns:
            Dictionary mapping symbols to MarketData objects
        """
        result = {}
        for symbol in symbols:
            data = self.fetch_klines(symbol, interval, limit)
            result[symbol] = data
        return result

    def _generate_synthetic(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> MarketData:
        """Generate synthetic data when API is unavailable."""
        return generate_synthetic_data(
            symbol=symbol,
            interval=interval,
            n_points=limit,
        )


def generate_synthetic_data(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    n_points: int = 1000,
    base_price: Optional[float] = None,
    volatility: float = 0.02,
    seed: Optional[int] = 42,
) -> MarketData:
    """
    Generate synthetic OHLCV data for testing.

    Produces realistic-looking price data with trends, mean-reversion,
    and volatility clustering (simplified GARCH-like process).

    Args:
        symbol: Symbol name for labeling
        interval: Interval label
        n_points: Number of data points
        base_price: Starting price (auto-detected from symbol if None)
        volatility: Base volatility per step
        seed: Random seed for reproducibility

    Returns:
        MarketData with synthetic DataFrame
    """
    if seed is not None:
        np.random.seed(seed)

    # Auto-detect base price from symbol
    if base_price is None:
        if "BTC" in symbol.upper():
            base_price = 50000.0
        elif "ETH" in symbol.upper():
            base_price = 3000.0
        elif "SOL" in symbol.upper():
            base_price = 100.0
        elif "AAPL" in symbol.upper():
            base_price = 150.0
        elif "GOOGL" in symbol.upper() or "GOOG" in symbol.upper():
            base_price = 140.0
        elif "MSFT" in symbol.upper():
            base_price = 350.0
        else:
            base_price = 100.0

    # Generate returns with volatility clustering
    vol = np.ones(n_points) * volatility
    returns = np.zeros(n_points)

    for i in range(1, n_points):
        # GARCH(1,1)-like volatility
        vol[i] = 0.85 * vol[i - 1] + 0.15 * volatility * (1 + 2 * abs(returns[i - 1]) / volatility)
        returns[i] = np.random.randn() * vol[i]

    # Add mean reversion and slight trend
    trend = np.linspace(0, 0.05, n_points) * np.random.choice([-1, 1])
    mean_reversion = np.zeros(n_points)

    for i in range(1, n_points):
        deviation = np.sum(returns[:i]) - trend[i]
        mean_reversion[i] = -0.01 * deviation

    returns = returns + trend / n_points + mean_reversion

    # Cumulative sum for log-prices
    log_prices = np.log(base_price) + np.cumsum(returns)
    close = np.exp(log_prices)

    # Generate OHLV from close
    intraday_vol = volatility * close * np.random.uniform(0.5, 1.5, n_points)
    high = close + np.abs(intraday_vol * np.random.rand(n_points))
    low = close - np.abs(intraday_vol * np.random.rand(n_points))

    # Ensure OHLC consistency
    low = np.minimum(low, close)
    high = np.maximum(high, close)

    open_price = np.roll(close, 1)
    open_price[0] = base_price

    # Generate volume with autocorrelation
    log_volume = np.zeros(n_points)
    base_log_vol = 10

    for i in range(n_points):
        if i == 0:
            log_volume[i] = base_log_vol + np.random.randn() * 0.5
        else:
            log_volume[i] = 0.7 * log_volume[i - 1] + 0.3 * base_log_vol + np.random.randn() * 0.5
            # Higher volume on big price moves
            log_volume[i] += abs(returns[i]) / volatility * 0.3

    volume = np.exp(log_volume)

    # Create timestamps
    if interval == "D":
        freq = "D"
    elif interval == "W":
        freq = "W"
    elif interval == "M":
        freq = "MS"
    else:
        freq = f"{interval}min" if interval.isdigit() else "h"

    try:
        timestamps = pd.date_range(start="2023-01-01", periods=n_points, freq=freq)
    except ValueError:
        timestamps = pd.date_range(start="2023-01-01", periods=n_points, freq="h")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    logger.info(f"Generated {n_points} synthetic data points for {symbol}")
    return MarketData(df=df, symbol=symbol, interval=interval, source="synthetic")


def load_stock_data(
    symbol: str = "AAPL",
    start: str = "2020-01-01",
    end: str = "2024-01-01",
    interval: str = "1d",
) -> MarketData:
    """
    Load stock data using yfinance.

    Falls back to synthetic data if yfinance is not installed or fails.

    Args:
        symbol: Stock ticker
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        interval: Data interval (1d, 1h, 5m, etc.)

    Returns:
        MarketData with stock OHLCV DataFrame
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return generate_synthetic_data(symbol, interval, 1000)

        df = df.reset_index()

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]

        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})

        # Keep only needed columns
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep_cols if c in df.columns]]

        logger.info(f"Loaded {len(df)} data points for {symbol} from yfinance")
        return MarketData(df=df, symbol=symbol, interval=interval, source="yfinance")

    except ImportError:
        logger.warning("yfinance not installed. Using synthetic data.")
        return generate_synthetic_data(
            symbol=symbol, interval="1d", n_points=1000,
            volatility=0.015
        )
    except Exception as e:
        logger.warning(f"Failed to load stock data: {e}. Using synthetic data.")
        return generate_synthetic_data(
            symbol=symbol, interval="1d", n_points=1000,
            volatility=0.015
        )


def create_sequences(
    data: MarketData,
    seq_len: int = 64,
    target_horizon: int = 5,
    threshold: float = 0.01,
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create sequences for training the attribution model.

    Args:
        data: MarketData object
        seq_len: Sequence length for input
        target_horizon: How many steps ahead to predict
        threshold: Threshold for classification labels
        feature_names: Features to include

    Returns:
        X: Input sequences (n_samples, seq_len, n_features)
        y: Target labels (n_samples,)
        feature_names: List of feature names used
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURE_NAMES

    # Get features
    features = data.to_features(feature_names)
    n_features = features.shape[1]

    # Add labels
    data_with_labels = data.add_labels(target_horizon, threshold)
    labels = data_with_labels.df["label"].values

    X, y = [], []

    for i in range(seq_len, len(features) - target_horizon):
        X.append(features[i - seq_len:i])
        y.append(labels[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Remove samples with NaN labels
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx].astype(np.int64)

    logger.info(f"Created {len(X)} sequences with shape ({seq_len}, {n_features})")
    logger.info(f"Label distribution: {np.bincount(y, minlength=3)}")

    return X, y, feature_names


def create_train_test_split(
    data: MarketData,
    seq_len: int = 64,
    target_horizon: int = 5,
    threshold: float = 0.01,
    train_ratio: float = 0.8,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Create train/test split for model training.

    Uses temporal split to avoid look-ahead bias.

    Args:
        data: MarketData object
        seq_len: Sequence length
        target_horizon: Prediction horizon
        threshold: Classification threshold
        train_ratio: Ratio of data for training
        feature_names: Features to include

    Returns:
        Dictionary with X_train, y_train, X_test, y_test, feature_names
    """
    X, y, used_features = create_sequences(
        data, seq_len, target_horizon, threshold, feature_names
    )

    # Temporal split
    split_idx = int(len(X) * train_ratio)

    return {
        "X_train": X[:split_idx],
        "y_train": y[:split_idx],
        "X_test": X[split_idx:],
        "y_test": y[split_idx:],
        "feature_names": used_features,
    }


class DataAugmenter:
    """
    Data augmentation for financial time series.

    Provides methods to augment training data while preserving
    temporal structure and realistic properties.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize augmenter."""
        self.rng = np.random.default_rng(seed)

    def add_noise(
        self,
        X: np.ndarray,
        noise_level: float = 0.01,
    ) -> np.ndarray:
        """Add Gaussian noise to features."""
        noise = self.rng.normal(0, noise_level, X.shape)
        return X + noise

    def time_warp(
        self,
        X: np.ndarray,
        sigma: float = 0.2,
    ) -> np.ndarray:
        """Apply random time warping."""
        batch, seq_len, features = X.shape
        warped = np.zeros_like(X)

        for b in range(batch):
            # Generate smooth warping path
            warp = self.rng.normal(1, sigma, seq_len)
            warp = np.cumsum(warp)
            warp = warp / warp[-1] * (seq_len - 1)
            warp = np.clip(warp, 0, seq_len - 1).astype(int)

            warped[b] = X[b, warp]

        return warped

    def magnitude_warp(
        self,
        X: np.ndarray,
        sigma: float = 0.1,
    ) -> np.ndarray:
        """Apply random magnitude scaling."""
        batch, seq_len, features = X.shape

        # Generate smooth scaling curve
        scale = self.rng.normal(1, sigma, (batch, 1, features))

        return X * scale

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_augmented: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate augmented samples.

        Args:
            X: Input sequences
            y: Labels
            n_augmented: Number of augmented copies per sample

        Returns:
            Augmented X and y arrays
        """
        all_X = [X]
        all_y = [y]

        for _ in range(n_augmented):
            # Randomly choose augmentation
            aug_type = self.rng.choice(["noise", "time_warp", "magnitude"])

            if aug_type == "noise":
                aug_X = self.add_noise(X, 0.01)
            elif aug_type == "time_warp":
                aug_X = self.time_warp(X, 0.1)
            else:
                aug_X = self.magnitude_warp(X, 0.05)

            all_X.append(aug_X)
            all_y.append(y)

        return np.concatenate(all_X), np.concatenate(all_y)


if __name__ == "__main__":
    # Demo: data loading and feature engineering
    print("Feature Attribution Data Loader Demo")
    print("=" * 60)

    # Test Bybit data
    print("\n1. Fetching Bybit cryptocurrency data...")
    loader = BybitDataLoader()
    btc_data = loader.fetch_klines("BTCUSDT", interval="60", limit=500)

    print(f"   Source: {btc_data.source}")
    print(f"   Symbol: {btc_data.symbol}")
    print(f"   Shape: {btc_data.df.shape}")
    print(f"   Date range: {btc_data.df['timestamp'].min()} to {btc_data.df['timestamp'].max()}")

    # Test stock data
    print("\n2. Loading stock market data...")
    stock_data = load_stock_data("AAPL", "2022-01-01", "2024-01-01")

    print(f"   Source: {stock_data.source}")
    print(f"   Symbol: {stock_data.symbol}")
    print(f"   Shape: {stock_data.df.shape}")

    # Test feature engineering
    print("\n3. Feature engineering...")
    features = btc_data.to_features()
    print(f"   Features shape: {features.shape}")
    print(f"   Feature names: {DEFAULT_FEATURE_NAMES}")
    print(f"   Sample features (first row):\n   {features[50]}")

    # Test sequence creation
    print("\n4. Creating training sequences...")
    X, y, feature_names = create_sequences(btc_data, seq_len=64, target_horizon=5)

    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Features used: {feature_names}")
    print(f"   Label distribution: BUY={np.sum(y==0)}, HOLD={np.sum(y==1)}, SELL={np.sum(y==2)}")

    # Test train/test split
    print("\n5. Train/test split...")
    split = create_train_test_split(btc_data, seq_len=64)

    print(f"   Train: X={split['X_train'].shape}, y={split['y_train'].shape}")
    print(f"   Test: X={split['X_test'].shape}, y={split['y_test'].shape}")

    # Test data augmentation
    print("\n6. Data augmentation...")
    augmenter = DataAugmenter(seed=42)
    X_aug, y_aug = augmenter.augment(X[:100], y[:100], n_augmented=2)

    print(f"   Original: X={X[:100].shape}, y={y[:100].shape}")
    print(f"   Augmented: X={X_aug.shape}, y={y_aug.shape}")

    # Test synthetic data
    print("\n7. Synthetic data generation...")
    synthetic = generate_synthetic_data("TESTUSDT", "60", n_points=1000, seed=123)

    print(f"   Symbol: {synthetic.symbol}")
    print(f"   Shape: {synthetic.df.shape}")
    print(f"   Price range: ${synthetic.close.min():.2f} - ${synthetic.close.max():.2f}")
    print(f"   Volatility: {np.std(synthetic.returns) * 100:.2f}%")

    print("\n" + "=" * 60)
    print("Demo complete!")
