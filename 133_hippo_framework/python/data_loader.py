"""
Data loading utilities for HiPPO trading experiments.

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
from typing import Optional
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
        return self.df["close"].values

    @property
    def returns(self) -> np.ndarray:
        return self.df["close"].pct_change().fillna(0).values

    @property
    def volume(self) -> np.ndarray:
        return self.df["volume"].values

    @property
    def volatility(self) -> np.ndarray:
        """Rolling squared returns as volatility proxy."""
        ret = self.returns
        return ret ** 2


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
            interval: Kline interval in minutes (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M).
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
    base_price: float = 50000.0,
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
        base_price: Starting price.
        volatility: Base volatility per step.
        seed: Random seed for reproducibility.

    Returns:
        MarketData with synthetic DataFrame.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate returns with volatility clustering (simplified GARCH-like)
    vol = np.ones(n_points) * volatility
    returns = np.zeros(n_points)
    for i in range(1, n_points):
        vol[i] = 0.9 * vol[i - 1] + 0.1 * volatility * (1 + abs(returns[i - 1]) / volatility)
        returns[i] = np.random.randn() * vol[i]

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
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        if "adj close" in df.columns:
            df = df.drop(columns=["adj close"])
        logger.info(f"Loaded {len(df)} data points for {symbol} from yfinance")
        return MarketData(df=df, symbol=symbol, interval="1d", source="yfinance")
    except ImportError:
        logger.warning("yfinance not installed. Using synthetic data.")
        return generate_synthetic_data(symbol=symbol, interval="1d", n_points=1000,
                                       base_price=150.0, volatility=0.015)


if __name__ == "__main__":
    # Demo: fetch Bybit data
    loader = BybitDataLoader()
    data = loader.fetch_klines("BTCUSDT", interval="60", limit=200)
    print(f"\nData source: {data.source}")
    print(f"Shape: {data.df.shape}")
    print(f"Columns: {list(data.df.columns)}")
    print(f"\nFirst 5 rows:")
    print(data.df.head())
    print(f"\nClose stats: mean={data.close.mean():.2f}, std={data.close.std():.2f}")
