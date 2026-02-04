"""
Data Fetcher for Cryptocurrency Markets

This module provides utilities for fetching cryptocurrency data
from Bybit exchange using the CCXT library.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BybitDataFetcher:
    """
    Fetch cryptocurrency data from Bybit exchange using CCXT.

    Supports:
    - OHLCV (candlestick) data
    - Multiple timeframes
    - Multiple symbols
    - Return calculation
    """

    def __init__(
        self,
        testnet: bool = False,
        rate_limit: bool = True
    ):
        """
        Initialize the Bybit data fetcher.

        Args:
            testnet: Whether to use Bybit testnet
            rate_limit: Whether to enable rate limiting
        """
        self.exchange = ccxt.bybit({
            'enableRateLimit': rate_limit,
            'options': {
                'defaultType': 'spot',  # or 'linear' for perpetuals
            }
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

        # Load markets
        self.markets = self.exchange.load_markets()
        logger.info(f"Loaded {len(self.markets)} markets from Bybit")

    def get_available_symbols(self, quote: str = "USDT") -> List[str]:
        """
        Get list of available trading symbols.

        Args:
            quote: Quote currency filter (e.g., "USDT")

        Returns:
            List of symbol strings
        """
        symbols = [
            s for s in self.markets.keys()
            if s.endswith(f"/{quote}")
        ]
        return sorted(symbols)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle timeframe (e.g., "1m", "1h", "1d")
            since: Start datetime (default: limit candles ago)
            limit: Maximum number of candles

        Returns:
            DataFrame with OHLCV data
        """
        if since is not None:
            since_ts = int(since.timestamp() * 1000)
        else:
            since_ts = None

        logger.info(f"Fetching {symbol} {timeframe} data (limit={limit})")

        ohlcv = self.exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ts,
            limit=limit
        )

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def fetch_ohlcv_extended(
        self,
        symbol: str,
        timeframe: str = "1d",
        start_date: datetime = None,
        end_date: datetime = None,
        max_requests: int = 10
    ) -> pd.DataFrame:
        """
        Fetch extended OHLCV data by making multiple requests.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start_date: Start datetime
            end_date: End datetime (default: now)
            max_requests: Maximum number of API requests

        Returns:
            DataFrame with extended OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            # Default to 1 year ago
            start_date = end_date - timedelta(days=365)

        all_data = []
        current_since = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        for _ in range(max_requests):
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=1000
            )

            if not ohlcv:
                break

            all_data.extend(ohlcv)

            # Move to next batch
            last_ts = ohlcv[-1][0]
            if last_ts >= end_ts:
                break

            current_since = last_ts + 1
            time.sleep(0.1)  # Rate limiting

        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df[df.index <= end_date]

        logger.info(f"Fetched {len(df)} candles for {symbol}")

        return df

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            limit: Maximum candles per symbol

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}

        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, timeframe, limit=limit)
                data[symbol] = df
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

        return data

    def calculate_returns(
        self,
        df: pd.DataFrame,
        return_type: str = "log",
        column: str = "close"
    ) -> pd.Series:
        """
        Calculate returns from price data.

        Args:
            df: DataFrame with price data
            return_type: "log" for log returns, "simple" for simple returns
            column: Price column to use

        Returns:
            Series of returns
        """
        prices = df[column]

        if return_type == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        return returns.dropna()

    def prepare_training_data(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        limit: int = 1000,
        return_type: str = "log",
        standardize: bool = True,
        winsorize: float = 0.01
    ) -> np.ndarray:
        """
        Prepare return data for training a normalizing flow.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            limit: Number of candles
            return_type: Type of returns
            standardize: Whether to standardize returns
            winsorize: Winsorization percentile (0 to disable)

        Returns:
            NumPy array of processed returns
        """
        # Fetch data
        df = self.fetch_ohlcv(symbol, timeframe, limit=limit)

        # Calculate returns
        returns = self.calculate_returns(df, return_type).values

        # Winsorize (clip extreme values)
        if winsorize > 0:
            lower = np.percentile(returns, winsorize * 100)
            upper = np.percentile(returns, (1 - winsorize) * 100)
            returns = np.clip(returns, lower, upper)

        # Standardize
        if standardize:
            mean = returns.mean()
            std = returns.std()
            returns = (returns - mean) / std

        return returns.reshape(-1, 1).astype(np.float32)

    def get_return_statistics(
        self,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute descriptive statistics for returns.

        Args:
            returns: Array of returns

        Returns:
            Dictionary of statistics
        """
        from scipy import stats as sp_stats

        returns = returns.flatten()

        return {
            'count': len(returns),
            'mean': returns.mean(),
            'std': returns.std(),
            'min': returns.min(),
            'max': returns.max(),
            'skewness': sp_stats.skew(returns),
            'kurtosis': sp_stats.kurtosis(returns),
            'jarque_bera_stat': sp_stats.jarque_bera(returns)[0],
            'jarque_bera_pvalue': sp_stats.jarque_bera(returns)[1],
            'percentile_1': np.percentile(returns, 1),
            'percentile_5': np.percentile(returns, 5),
            'percentile_25': np.percentile(returns, 25),
            'percentile_50': np.percentile(returns, 50),
            'percentile_75': np.percentile(returns, 75),
            'percentile_95': np.percentile(returns, 95),
            'percentile_99': np.percentile(returns, 99),
        }


def fetch_sample_data(
    symbol: str = "BTC/USDT",
    days: int = 365
) -> np.ndarray:
    """
    Quick helper to fetch sample return data for examples.

    Args:
        symbol: Trading pair
        days: Number of days of data

    Returns:
        Array of log returns
    """
    fetcher = BybitDataFetcher()

    # Fetch data
    df = fetcher.fetch_ohlcv(
        symbol=symbol,
        timeframe="1d",
        limit=days + 10  # Extra for return calculation
    )

    # Calculate log returns
    returns = fetcher.calculate_returns(df, return_type="log")

    # Basic preprocessing
    returns = returns.values[-days:]  # Last N days

    return returns.reshape(-1, 1).astype(np.float32)


def create_synthetic_returns(
    n_samples: int = 1000,
    distribution: str = "fat_tails"
) -> np.ndarray:
    """
    Create synthetic return data for testing.

    Args:
        n_samples: Number of samples
        distribution: Type of distribution
            - "normal": Standard normal
            - "fat_tails": Student-t distribution
            - "skewed": Skewed distribution
            - "mixture": Mixture of normals (regime switching)

    Returns:
        Array of synthetic returns
    """
    np.random.seed(42)

    if distribution == "normal":
        returns = np.random.normal(0, 0.02, n_samples)

    elif distribution == "fat_tails":
        # Student-t with 4 degrees of freedom
        returns = np.random.standard_t(df=4, size=n_samples) * 0.02

    elif distribution == "skewed":
        # Log-normal based skewed distribution
        returns = np.random.lognormal(mean=-0.001, sigma=0.02, size=n_samples) - 1

    elif distribution == "mixture":
        # Mixture of two normals (calm and volatile regimes)
        regime = np.random.binomial(1, 0.2, n_samples)  # 20% volatile
        calm = np.random.normal(0.001, 0.01, n_samples)
        volatile = np.random.normal(-0.002, 0.04, n_samples)
        returns = np.where(regime, volatile, calm)

    else:
        returns = np.random.normal(0, 0.02, n_samples)

    return returns.reshape(-1, 1).astype(np.float32)
