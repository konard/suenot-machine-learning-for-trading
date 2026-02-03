"""
Data loading utilities for S4 trading experiments.

Supports:
- Bybit API for cryptocurrency OHLCV data
- yfinance for stock market data
- Synthetic data generation for testing

All data is returned as pandas DataFrames with columns:
    timestamp, open, high, low, close, volume
"""

import numpy as np
import pandas as pd
import requests
import logging
from typing import Optional, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market OHLCV data."""
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
        """Rolling squared returns as volatility proxy."""
        return self.returns ** 2

    def to_features(self, normalize: bool = True) -> np.ndarray:
        """
        Convert to feature matrix for model input.

        Args:
            normalize: Whether to normalize features

        Returns:
            Feature matrix of shape (n_samples, 5)
        """
        features = self.df[["open", "high", "low", "close", "volume"]].values.copy()

        if normalize:
            # Normalize OHLC by close price
            close = features[:, 3:4]
            features[:, :4] = features[:, :4] / close

            # Log-normalize volume
            features[:, 4] = np.log1p(features[:, 4])
            features[:, 4] = (features[:, 4] - features[:, 4].mean()) / (features[:, 4].std() + 1e-8)

        return features.astype(np.float32)

    def add_technical_features(self) -> "MarketData":
        """Add technical indicator features to the dataframe."""
        df = self.df.copy()

        # Returns
        df["returns"] = df["close"].pct_change()

        # Moving averages
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # Volatility
        df["volatility_10"] = df["returns"].rolling(10).std()
        df["volatility_20"] = df["returns"].rolling(20).std()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        # Volume features
        df["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        df = df.fillna(0)

        return MarketData(df=df, symbol=self.symbol, interval=self.interval, source=self.source)


class BybitDataLoader:
    """
    Data loader for Bybit exchange.

    Fetches OHLCV (kline) data from the Bybit public API v5.
    No API key required for public market data.
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",
        limit: int = 1000,
        category: str = "linear",
    ) -> MarketData:
        """
        Fetch kline (OHLCV) data from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT").
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M).
            limit: Number of data points (max 1000).
            category: Market category ("linear", "inverse", "spot").

        Returns:
            MarketData object with OHLCV DataFrame.
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                logger.warning(f"Bybit API error: {data.get('retMsg')}")
                return self._generate_synthetic(symbol, interval, limit)

            records = data["result"]["list"]
            df = pd.DataFrame(records, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df = df.drop(columns=["turnover"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)

            logger.info(f"Fetched {len(df)} klines for {symbol} from Bybit")
            return MarketData(df=df, symbol=symbol, interval=interval, source="bybit")

        except Exception as e:
            logger.warning(f"Failed to fetch from Bybit: {e}. Using synthetic data.")
            return self._generate_synthetic(symbol, interval, limit)

    def _generate_synthetic(
        self, symbol: str, interval: str, limit: int
    ) -> MarketData:
        """Generate synthetic OHLCV data when API is unavailable."""
        return generate_synthetic_data(symbol=symbol, interval=interval, n_points=limit)


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
    and volatility clustering.

    Args:
        symbol: Symbol name for labeling.
        interval: Interval label.
        n_points: Number of data points.
        base_price: Starting price (auto-detected from symbol if None).
        volatility: Base volatility per step.
        seed: Random seed for reproducibility.

    Returns:
        MarketData with synthetic DataFrame.
    """
    if seed is not None:
        np.random.seed(seed)

    # Auto-detect base price from symbol
    if base_price is None:
        if "BTC" in symbol:
            base_price = 50000.0
        elif "ETH" in symbol:
            base_price = 3000.0
        elif "SOL" in symbol:
            base_price = 100.0
        else:
            base_price = 100.0

    # Generate returns with volatility clustering (simplified GARCH-like)
    vol = np.ones(n_points) * volatility
    returns = np.zeros(n_points)

    for i in range(1, n_points):
        vol[i] = 0.9 * vol[i - 1] + 0.1 * volatility * (1 + abs(returns[i - 1]) / volatility)
        returns[i] = np.random.randn() * vol[i]

    # Add slight trend
    trend = np.linspace(0, 0.1, n_points) * np.random.choice([-1, 1])
    returns = returns + trend / n_points

    # Cumulative sum for log-prices
    log_prices = np.log(base_price) + np.cumsum(returns)
    close = np.exp(log_prices)

    # Generate OHLV from close
    noise = np.random.rand(n_points) * volatility * close
    high = close + abs(noise)
    low = close - abs(noise)
    open_price = np.roll(close, 1)
    open_price[0] = base_price
    volume = np.random.lognormal(mean=10, sigma=1, size=n_points)

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
) -> MarketData:
    """
    Load stock data using yfinance (if available).

    Falls back to synthetic data if yfinance is not installed.

    Args:
        symbol: Stock ticker.
        start: Start date string.
        end: End date string.

    Returns:
        MarketData with stock OHLCV DataFrame.
    """
    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, progress=False)
        df = df.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        if "adj close" in df.columns:
            df = df.drop(columns=["adj close"])
        logger.info(f"Loaded {len(df)} data points for {symbol} from yfinance")
        return MarketData(df=df, symbol=symbol, interval="1d", source="yfinance")
    except ImportError:
        logger.warning("yfinance not installed. Using synthetic data.")
        return generate_synthetic_data(
            symbol=symbol, interval="1d", n_points=1000,
            base_price=150.0, volatility=0.015
        )
    except Exception as e:
        logger.warning(f"Failed to load stock data: {e}. Using synthetic data.")
        return generate_synthetic_data(
            symbol=symbol, interval="1d", n_points=1000,
            base_price=150.0, volatility=0.015
        )


def create_sequences(
    data: MarketData,
    seq_len: int = 256,
    target_horizon: int = 1,
) -> tuple:
    """
    Create sequences for training S4 model.

    Args:
        data: MarketData object
        seq_len: Sequence length for input
        target_horizon: How many steps ahead to predict

    Returns:
        X: Input sequences (n_samples, seq_len, n_features)
        y: Target labels (n_samples,)
    """
    features = data.to_features()
    returns = data.returns

    X, y = [], []

    for i in range(seq_len, len(features) - target_horizon):
        X.append(features[i - seq_len:i])

        # Target: direction of future return
        future_return = returns[i + target_horizon - 1]
        if future_return > 0.001:
            label = 0  # BUY
        elif future_return < -0.001:
            label = 2  # SELL
        else:
            label = 1  # HOLD

        y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Demo: fetch Bybit data
    print("Data Loader Demo")
    print("=" * 50)

    loader = BybitDataLoader()
    data = loader.fetch_klines("BTCUSDT", interval="60", limit=500)

    print(f"\nData source: {data.source}")
    print(f"Symbol: {data.symbol}")
    print(f"Shape: {data.df.shape}")
    print(f"Columns: {list(data.df.columns)}")

    print(f"\nFirst 3 rows:")
    print(data.df.head(3))

    print(f"\nClose stats: mean={data.close.mean():.2f}, std={data.close.std():.2f}")
    print(f"Returns stats: mean={data.returns.mean():.6f}, std={data.returns.std():.6f}")

    # Test features
    features = data.to_features()
    print(f"\nFeatures shape: {features.shape}")
    print(f"Features sample:\n{features[:3]}")

    # Test sequence creation
    X, y = create_sequences(data, seq_len=64)
    print(f"\nSequences: X={X.shape}, y={y.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    print("\n" + "=" * 50)
    print("Demo complete!")
