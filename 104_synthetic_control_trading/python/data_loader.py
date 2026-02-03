"""
Data loading utilities for Synthetic Control Method trading.

Supports data from:
- Bybit (cryptocurrency via public REST API)
- Yahoo Finance (stocks via yfinance)

Generates features and prepares data for SCM analysis.
"""

import numpy as np
import pandas as pd
import requests
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "D",
    limit: int = 200,
) -> pd.DataFrame:
    """
    Fetch kline data from Bybit public API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
        interval: Candle interval ("1", "5", "15", "60", "240", "D", "W")
        limit: Number of candles to fetch (max 1000)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode") != 0:
            logger.warning(f"Bybit API error: {data.get('retMsg')}")
            return _generate_synthetic_data(limit, symbol)

        rows = data["result"]["list"]
        if not rows:
            logger.warning(f"No data returned for {symbol}")
            return _generate_synthetic_data(limit, symbol)

        df = pd.DataFrame(rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["symbol"] = symbol
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch Bybit data for {symbol}: {e}. Using synthetic data.")
        return _generate_synthetic_data(limit, symbol)


def fetch_yfinance_data(
    symbol: str = "AAPL",
    period: str = "2y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch stock data using yfinance.

    Args:
        symbol: Stock ticker (e.g., "AAPL", "MSFT")
        period: Time period (e.g., "1y", "2y", "5y")
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)

        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date)
        else:
            df = ticker.history(period=period)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return _generate_synthetic_data(500, symbol)

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df["symbol"] = symbol
        return df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]
    except ImportError:
        logger.warning("yfinance not installed. Using synthetic data.")
        return _generate_synthetic_data(500, symbol)
    except Exception as e:
        logger.warning(f"Error fetching {symbol}: {e}. Using synthetic data.")
        return _generate_synthetic_data(500, symbol)


def _generate_synthetic_data(n: int = 200, symbol: str = "SYNTHETIC") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(hash(symbol) % 2**32)

    # Generate realistic price series with trend and volatility
    returns = np.random.randn(n) * 0.02
    prices = 100.0 * np.exp(np.cumsum(returns))
    volumes = np.random.lognormal(mean=15, sigma=1.5, size=n)

    # Generate OHLC from close prices
    highs = prices * (1 + np.abs(np.random.randn(n) * 0.015))
    lows = prices * (1 - np.abs(np.random.randn(n) * 0.015))
    opens = lows + (highs - lows) * np.random.rand(n)

    return pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=n, freq="D"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "symbol": symbol,
    })


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from price data.

    Args:
        df: DataFrame with 'close' column

    Returns:
        DataFrame with added 'returns' column
    """
    df = df.copy()
    df["returns"] = np.log(df["close"] / df["close"].shift(1))
    return df


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling volatility from returns.

    Args:
        df: DataFrame with 'returns' column
        window: Rolling window size

    Returns:
        DataFrame with added 'volatility' column
    """
    df = df.copy()
    if "returns" not in df.columns:
        df = compute_returns(df)
    df["volatility"] = df["returns"].rolling(window).std() * np.sqrt(252)
    return df


def compute_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute volume ratio (current volume / average volume).

    Args:
        df: DataFrame with 'volume' column
        window: Rolling window for average

    Returns:
        DataFrame with added 'volume_ratio' column
    """
    df = df.copy()
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(window).mean() + 1e-10)
    return df


@dataclass
class SCMDataConfig:
    """Configuration for SCM data loading."""
    pre_treatment_days: int = 60
    post_treatment_days: int = 30
    min_pre_treatment_days: int = 30
    features: Tuple[str, ...] = ("returns", "volatility", "volume_ratio")


class SCMDataLoader:
    """
    Data loader for Synthetic Control Method analysis.

    Loads data for treated and donor units, computes features,
    and prepares data for SCM fitting.
    """

    def __init__(
        self,
        treated_symbol: str,
        donor_symbols: List[str],
        source: str = "yfinance",
        pre_treatment_days: int = 60,
        post_treatment_days: int = 30,
        interval: str = "D",
    ):
        """
        Initialize SCM data loader.

        Args:
            treated_symbol: Symbol of the treated unit (e.g., "AAPL")
            donor_symbols: List of donor symbols (e.g., ["MSFT", "GOOGL"])
            source: Data source ("yfinance" or "bybit")
            pre_treatment_days: Days before treatment for fitting
            post_treatment_days: Days after treatment for evaluation
            interval: Data interval (for Bybit: "1", "60", "D", etc.)
        """
        self.treated_symbol = treated_symbol
        self.donor_symbols = donor_symbols
        self.source = source
        self.pre_treatment_days = pre_treatment_days
        self.post_treatment_days = post_treatment_days
        self.interval = interval

    def _fetch_data(self, symbol: str, limit: int) -> pd.DataFrame:
        """Fetch data for a single symbol."""
        if self.source == "bybit":
            return fetch_bybit_klines(symbol, self.interval, limit)
        elif self.source == "yfinance":
            return fetch_yfinance_data(symbol, period="2y")
        else:
            return _generate_synthetic_data(limit, symbol)

    def load_event_data(
        self,
        event_date: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for event study analysis.

        Args:
            event_date: Date of the treatment event (YYYY-MM-DD)

        Returns:
            Dictionary with keys:
            - treated_pre: Treated unit data before treatment
            - treated_post: Treated unit data after treatment
            - donors_pre: Donor units data before treatment (dict by symbol)
            - donors_post: Donor units data after treatment (dict by symbol)
        """
        event_dt = pd.to_datetime(event_date)
        total_days = self.pre_treatment_days + self.post_treatment_days + 10

        # Fetch treated data
        treated_df = self._fetch_data(self.treated_symbol, total_days)
        treated_df = self._add_features(treated_df)

        # Fetch donor data
        donor_dfs = {}
        for symbol in self.donor_symbols:
            df = self._fetch_data(symbol, total_days)
            df = self._add_features(df)
            donor_dfs[symbol] = df

        # Split by event date
        result = {
            "treated_pre": treated_df[treated_df["timestamp"] < event_dt].tail(self.pre_treatment_days),
            "treated_post": treated_df[treated_df["timestamp"] >= event_dt].head(self.post_treatment_days),
            "donors_pre": {},
            "donors_post": {},
        }

        for symbol, df in donor_dfs.items():
            result["donors_pre"][symbol] = df[df["timestamp"] < event_dt].tail(self.pre_treatment_days)
            result["donors_post"][symbol] = df[df["timestamp"] >= event_dt].head(self.post_treatment_days)

        return result

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed features to dataframe."""
        df = compute_returns(df)
        df = compute_volatility(df)
        df = compute_volume_ratio(df)
        return df

    def load_data_matrix(
        self,
        event_date: str,
        feature: str = "returns",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data as matrices suitable for SCM fitting.

        Args:
            event_date: Date of the treatment event
            feature: Feature to use (e.g., "returns", "close")

        Returns:
            Tuple of (treated_pre, donors_pre, treated_post, donors_post)
            where each is a numpy array
        """
        data = self.load_event_data(event_date)

        # Extract treated unit data
        treated_pre = data["treated_pre"][feature].values
        treated_post = data["treated_post"][feature].values

        # Extract donor data as matrix (time x donors)
        donor_symbols = list(data["donors_pre"].keys())
        n_pre = len(data["treated_pre"])
        n_post = len(data["treated_post"])
        n_donors = len(donor_symbols)

        donors_pre = np.zeros((n_pre, n_donors))
        donors_post = np.zeros((n_post, n_donors))

        for i, symbol in enumerate(donor_symbols):
            pre_data = data["donors_pre"][symbol][feature].values
            post_data = data["donors_post"][symbol][feature].values

            # Handle length mismatches
            donors_pre[:len(pre_data), i] = pre_data[:n_pre]
            donors_post[:len(post_data), i] = post_data[:n_post]

        return treated_pre, donors_pre, treated_post, donors_post


def load_multiple_events(
    events: List[Dict],
    source: str = "yfinance",
) -> List[Dict]:
    """
    Load data for multiple events.

    Args:
        events: List of dicts with keys: treated_symbol, donor_symbols, event_date
        source: Data source ("yfinance" or "bybit")

    Returns:
        List of data dictionaries for each event
    """
    results = []
    for event in events:
        loader = SCMDataLoader(
            treated_symbol=event["treated_symbol"],
            donor_symbols=event["donor_symbols"],
            source=source,
        )
        data = loader.load_event_data(event["event_date"])
        data["event_info"] = event
        results.append(data)
    return results
