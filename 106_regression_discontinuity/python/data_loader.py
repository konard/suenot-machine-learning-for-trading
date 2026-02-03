"""
Data loading utilities for RDD trading analysis.

Supports:
- Bybit API for cryptocurrency data
- yfinance for stock market data
- Synthetic data generation for testing
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import time


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MarketData:
    """Market data container."""
    symbol: str
    interval: str
    candles: List[Candle]

    def close_prices(self) -> np.ndarray:
        """Get close prices as numpy array."""
        return np.array([c.close for c in self.candles])

    def open_prices(self) -> np.ndarray:
        """Get open prices as numpy array."""
        return np.array([c.open for c in self.candles])

    def high_prices(self) -> np.ndarray:
        """Get high prices as numpy array."""
        return np.array([c.high for c in self.candles])

    def low_prices(self) -> np.ndarray:
        """Get low prices as numpy array."""
        return np.array([c.low for c in self.candles])

    def volumes(self) -> np.ndarray:
        """Get volumes as numpy array."""
        return np.array([c.volume for c in self.candles])

    def timestamps(self) -> np.ndarray:
        """Get timestamps as numpy array."""
        return np.array([c.timestamp for c in self.candles])

    def returns(self) -> np.ndarray:
        """Calculate simple returns."""
        closes = self.close_prices()
        rets = np.zeros(len(closes))
        rets[1:] = np.diff(closes) / closes[:-1]
        return rets

    def forward_returns(self, periods: int = 1) -> np.ndarray:
        """Calculate forward returns."""
        closes = self.close_prices()
        n = len(closes)
        rets = np.full(n, np.nan)
        for i in range(n - periods):
            if closes[i] > 0:
                rets[i] = (closes[i + periods] - closes[i]) / closes[i]
        return rets

    def __len__(self) -> int:
        return len(self.candles)


@dataclass
class FundingRate:
    """Funding rate data."""
    timestamp: int
    symbol: str
    funding_rate: float
    funding_rate_annualized: float


class BybitClient:
    """
    Bybit API client for fetching cryptocurrency market data.

    Uses the public API v5, no authentication required.
    """

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit client.

        Parameters
        ----------
        testnet : bool
            If True, use testnet API.
        """
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 200,
    ) -> MarketData:
        """
        Fetch kline (OHLCV) data from Bybit.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g., "BTCUSDT")
        interval : str
            Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        limit : int
            Number of candles (max 1000)

        Returns
        -------
        MarketData
            Market data container with candles.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required. Install with: pip install requests")

        url = f"{self.base_url}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("retCode") != 0:
            raise ValueError(f"Bybit API error: {data.get('retMsg')}")

        candles = []
        for row in data["result"]["list"]:
            if len(row) >= 6:
                candles.append(Candle(
                    timestamp=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                ))

        # Reverse to chronological order (Bybit returns newest first)
        candles.reverse()

        return MarketData(symbol=symbol, interval=interval, candles=candles)

    def fetch_funding_rates(
        self,
        symbol: str,
        limit: int = 200,
    ) -> List[FundingRate]:
        """
        Fetch historical funding rates.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g., "BTCUSDT")
        limit : int
            Number of records (max 200)

        Returns
        -------
        list of FundingRate
            Funding rate history.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required. Install with: pip install requests")

        url = f"{self.base_url}/v5/market/funding/history"
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": min(limit, 200),
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data.get("retCode") != 0:
            raise ValueError(f"Bybit API error: {data.get('retMsg')}")

        rates = []
        for item in data["result"]["list"]:
            rate = float(item["fundingRate"])
            rates.append(FundingRate(
                timestamp=int(item["fundingRateTimestamp"]),
                symbol=item["symbol"],
                funding_rate=rate,
                # Annualize: 3 payments per day * 365 days
                funding_rate_annualized=rate * 3 * 365 * 100,
            ))

        # Reverse to chronological order
        rates.reverse()
        return rates


def fetch_stock_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1h",
) -> MarketData:
    """
    Fetch stock market data using yfinance.

    Parameters
    ----------
    symbol : str
        Stock ticker (e.g., "AAPL", "SPY")
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    interval : str
        Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)

    Returns
    -------
    MarketData
        Market data container.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance library required. Install with: pip install yfinance")

    ticker = yf.Ticker(symbol)

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    df = ticker.history(start=start_date, end=end_date, interval=interval)

    candles = []
    for idx, row in df.iterrows():
        candles.append(Candle(
            timestamp=int(idx.timestamp() * 1000),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
        ))

    return MarketData(symbol=symbol, interval=interval, candles=candles)


def generate_synthetic_data(
    symbol: str = "SYNTHETIC",
    n_points: int = 1000,
    base_price: float = 100.0,
    volatility: float = 0.02,
    seed: Optional[int] = None,
) -> MarketData:
    """
    Generate synthetic OHLCV data for testing.

    Parameters
    ----------
    symbol : str
        Symbol name
    n_points : int
        Number of candles to generate
    base_price : float
        Starting price
    volatility : float
        Daily volatility (standard deviation of returns)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    MarketData
        Synthetic market data.
    """
    if seed is not None:
        np.random.seed(seed)

    candles = []
    price = base_price
    base_timestamp = 1672531200000  # 2023-01-01

    for i in range(n_points):
        ret = np.random.normal(0, volatility)
        new_price = price * (1 + ret)
        high = new_price * (1 + np.random.uniform(0, volatility))
        low = new_price * (1 - np.random.uniform(0, volatility))
        volume = np.random.uniform(100, 1000)

        candles.append(Candle(
            timestamp=base_timestamp + i * 3600000,  # Hourly
            open=price,
            high=high,
            low=low,
            close=new_price,
            volume=volume,
        ))

        price = new_price

    return MarketData(symbol=symbol, interval="60", candles=candles)


def generate_synthetic_funding_rates(
    symbol: str = "BTCUSDT",
    n_points: int = 200,
    seed: Optional[int] = None,
) -> List[FundingRate]:
    """
    Generate synthetic funding rate data for testing.

    Parameters
    ----------
    symbol : str
        Symbol name
    n_points : int
        Number of funding periods
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list of FundingRate
        Synthetic funding rate data.
    """
    if seed is not None:
        np.random.seed(seed)

    rates = []
    base_timestamp = 1672531200000  # 2023-01-01

    for i in range(n_points):
        rate = np.random.normal(0.0001, 0.0005)
        rates.append(FundingRate(
            timestamp=base_timestamp + i * 8 * 3600000,  # 8-hourly
            symbol=symbol,
            funding_rate=rate,
            funding_rate_annualized=rate * 3 * 365 * 100,
        ))

    return rates


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic data...")
    data = generate_synthetic_data("TEST", 100, 50000.0, 0.02, 42)
    print(f"Generated {len(data)} candles")
    print(f"Price range: {data.close_prices().min():.2f} - {data.close_prices().max():.2f}")

    print("\nGenerating synthetic funding rates...")
    funding = generate_synthetic_funding_rates("BTCUSDT", 50, 42)
    print(f"Generated {len(funding)} funding rate observations")
    rates = [f.funding_rate_annualized for f in funding]
    print(f"Funding rate range: {min(rates):.2f}% - {max(rates):.2f}%")
