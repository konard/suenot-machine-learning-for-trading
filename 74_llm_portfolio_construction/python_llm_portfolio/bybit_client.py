"""
Bybit Data Client Module

This module provides classes for fetching cryptocurrency market data
from the Bybit exchange API for portfolio construction.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

import requests
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OHLCV:
    """OHLCV (Open, High, Low, Close, Volume) candlestick data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "datetime": self.datetime.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class BybitClient:
    """
    Bybit API client for fetching market data.

    Supports fetching klines (candlesticks), tickers, and order book data
    from Bybit's public API for portfolio construction.
    """

    BASE_URL = "https://api.bybit.com"

    # Interval mapping
    INTERVALS = {
        "1": "1",      # 1 minute
        "3": "3",      # 3 minutes
        "5": "5",      # 5 minutes
        "15": "15",    # 15 minutes
        "30": "30",    # 30 minutes
        "60": "60",    # 1 hour
        "120": "120",  # 2 hours
        "240": "240",  # 4 hours
        "360": "360",  # 6 hours
        "720": "720",  # 12 hours
        "D": "D",      # 1 day
        "W": "W",      # 1 week
        "M": "M",      # 1 month
    }

    def __init__(self, timeout: int = 30):
        """
        Initialize Bybit client.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })

    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request."""
        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise ValueError(f"API error: {data.get('retMsg', 'Unknown error')}")

            return data.get("result", {})

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "D",
        limit: int = 200,
    ) -> List[OHLCV]:
        """
        Fetch kline/candlestick data.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval (1, 5, 15, 30, 60, 240, D, W, M)
            limit: Number of candles (max 1000)

        Returns:
            List of OHLCV objects
        """
        if interval not in self.INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Valid: {list(self.INTERVALS.keys())}")

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        result = self._request("/v5/market/kline", params)
        klines = result.get("list", [])

        # Bybit returns data in reverse chronological order
        ohlcv_list = []
        for k in reversed(klines):
            ohlcv_list.append(OHLCV(
                timestamp=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            ))

        return ohlcv_list

    def fetch_historical_klines(
        self,
        symbol: str,
        interval: str = "D",
        days: int = 365,
    ) -> List[OHLCV]:
        """
        Fetch historical kline data for a specified number of days.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            days: Number of days of history

        Returns:
            List of OHLCV objects
        """
        # Calculate how many candles we need
        interval_minutes = {
            "1": 1, "3": 3, "5": 5, "15": 15, "30": 30,
            "60": 60, "120": 120, "240": 240, "360": 360, "720": 720,
            "D": 1440, "W": 10080, "M": 43200,
        }

        minutes = interval_minutes.get(interval, 1440)
        total_candles = (days * 24 * 60) // minutes

        # Fetch in batches if needed
        all_klines = []
        end_time = int(time.time() * 1000)

        while len(all_klines) < total_candles:
            batch_size = min(1000, total_candles - len(all_klines))

            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": batch_size,
                "end": end_time,
            }

            result = self._request("/v5/market/kline", params)
            klines = result.get("list", [])

            if not klines:
                break

            for k in reversed(klines):
                all_klines.append(OHLCV(
                    timestamp=int(k[0]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                ))

            # Update end time for next batch
            end_time = int(klines[-1][0]) - 1

            # Rate limiting
            time.sleep(0.1)

        # Sort by timestamp and remove duplicates
        all_klines.sort(key=lambda x: x.timestamp)
        seen = set()
        unique_klines = []
        for k in all_klines:
            if k.timestamp not in seen:
                seen.add(k.timestamp)
                unique_klines.append(k)

        return unique_klines[-total_candles:]

    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data dictionary
        """
        params = {
            "category": "linear",
            "symbol": symbol,
        }

        result = self._request("/v5/market/tickers", params)
        tickers = result.get("list", [])

        if not tickers:
            raise ValueError(f"Ticker not found for {symbol}")

        return tickers[0]

    def fetch_multiple_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch tickers for multiple symbols.

        Args:
            symbols: List of trading pair symbols

        Returns:
            Dict mapping symbol to ticker data
        """
        tickers = {}
        for symbol in symbols:
            try:
                tickers[symbol] = self.fetch_ticker(symbol)
                time.sleep(0.05)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch ticker for {symbol}: {e}")
        return tickers

    def to_dataframe(self, klines: List[OHLCV]) -> pd.DataFrame:
        """
        Convert OHLCV list to pandas DataFrame.

        Args:
            klines: List of OHLCV objects

        Returns:
            DataFrame with OHLCV columns
        """
        data = [k.to_dict() for k in klines]
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df

    def calculate_returns(self, klines: List[OHLCV]) -> np.ndarray:
        """
        Calculate daily returns from OHLCV data.

        Args:
            klines: List of OHLCV objects

        Returns:
            Array of daily returns
        """
        closes = np.array([k.close for k in klines])
        returns = np.diff(closes) / closes[:-1]
        return returns

    def calculate_volatility(self, klines: List[OHLCV], annualize: bool = True) -> float:
        """
        Calculate volatility from OHLCV data.

        Args:
            klines: List of OHLCV objects
            annualize: Whether to annualize the volatility

        Returns:
            Volatility value
        """
        returns = self.calculate_returns(klines)
        vol = np.std(returns)
        if annualize:
            vol *= np.sqrt(365)  # Crypto trades 365 days
        return vol


class PortfolioDataFetcher:
    """
    Fetches and prepares data for portfolio construction.
    """

    def __init__(self, client: Optional[BybitClient] = None):
        """
        Initialize fetcher.

        Args:
            client: BybitClient instance (creates new if None)
        """
        self.client = client or BybitClient()

    def fetch_portfolio_data(
        self,
        symbols: List[str],
        days: int = 365,
        interval: str = "D"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all portfolio symbols.

        Args:
            symbols: List of trading symbols
            days: Number of days of history
            interval: Data interval

        Returns:
            Dict mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            try:
                logger.info(f"Fetching {symbol}...")
                klines = self.client.fetch_historical_klines(symbol, interval, days)
                data[symbol] = self.client.to_dataframe(klines)
                logger.info(f"  Got {len(klines)} candles for {symbol}")
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        return data

    def build_price_matrix(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Build a price matrix from individual DataFrames.

        Args:
            data: Dict mapping symbol to DataFrame

        Returns:
            DataFrame with prices for all symbols
        """
        prices = {}
        for symbol, df in data.items():
            prices[symbol] = df["close"]

        price_df = pd.DataFrame(prices)
        price_df = price_df.dropna()
        return price_df

    def calculate_returns_matrix(
        self,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate returns matrix from price matrix.

        Args:
            price_df: DataFrame with prices

        Returns:
            DataFrame with returns
        """
        return price_df.pct_change().dropna()

    def calculate_covariance_matrix(
        self,
        returns_df: pd.DataFrame,
        annualize: bool = True
    ) -> pd.DataFrame:
        """
        Calculate covariance matrix.

        Args:
            returns_df: DataFrame with returns
            annualize: Whether to annualize

        Returns:
            Covariance matrix DataFrame
        """
        cov = returns_df.cov()
        if annualize:
            cov *= 365  # Crypto trades 365 days
        return cov

    def calculate_expected_returns(
        self,
        returns_df: pd.DataFrame,
        method: str = "mean",
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate expected returns.

        Args:
            returns_df: DataFrame with returns
            method: Calculation method ('mean', 'ewma')
            annualize: Whether to annualize

        Returns:
            Series with expected returns
        """
        if method == "mean":
            expected = returns_df.mean()
        elif method == "ewma":
            expected = returns_df.ewm(span=60).mean().iloc[-1]
        else:
            raise ValueError(f"Unknown method: {method}")

        if annualize:
            expected *= 365  # Crypto trades 365 days

        return expected


# Example usage
if __name__ == "__main__":
    print("Bybit Portfolio Data Client Demo")
    print("=" * 50)

    client = BybitClient()

    # Crypto portfolio symbols
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

    # Fetch recent data
    print("\nFetching portfolio data...")
    fetcher = PortfolioDataFetcher(client)
    data = fetcher.fetch_portfolio_data(symbols, days=30, interval="D")

    # Build matrices
    price_df = fetcher.build_price_matrix(data)
    returns_df = fetcher.calculate_returns_matrix(price_df)
    cov_matrix = fetcher.calculate_covariance_matrix(returns_df)
    expected_returns = fetcher.calculate_expected_returns(returns_df)

    print("\nExpected Annual Returns:")
    for symbol, ret in expected_returns.items():
        print(f"  {symbol}: {ret:.2%}")

    print("\nAnnualized Volatility:")
    for symbol in symbols:
        vol = returns_df[symbol].std() * np.sqrt(365)
        print(f"  {symbol}: {vol:.2%}")

    print("\nCorrelation Matrix:")
    print(returns_df.corr().round(3))
