"""
Market Data Loading Module

Utilities for loading market data from various sources including
stock markets and cryptocurrency exchanges.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class OHLCVBar:
    """OHLCV bar data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }


class MarketDataLoader:
    """
    Load market data from various sources.

    Supports both traditional stock markets (via yfinance)
    and cryptocurrency markets (via exchange APIs).

    Examples:
        >>> loader = MarketDataLoader()
        >>> df = loader.load_stock_data("AAPL", "2024-01-01", "2024-12-31")
        >>> print(df.head())
    """

    def __init__(self, use_mock: bool = True):
        """
        Initialize the data loader.

        Args:
            use_mock: If True, generate mock data for demos
        """
        self.use_mock = use_mock
        self._yf = None
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Load stock market data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.use_mock:
            df = self._generate_mock_ohlcv(symbol, start_date, end_date, interval)
        else:
            df = self._load_yfinance(symbol, start_date, end_date, interval)

        self._cache[cache_key] = df
        return df

    def load_crypto_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
        exchange: str = "bybit"
    ) -> pd.DataFrame:
        """
        Load cryptocurrency data.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            exchange: Exchange name ('bybit', 'binance')

        Returns:
            DataFrame with OHLCV data
        """
        if self.use_mock:
            return self._generate_mock_ohlcv(symbol, start_date, end_date, interval)

        # In production, use exchange-specific API
        raise NotImplementedError(f"Live data from {exchange} not implemented")

    def _load_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """Load data using yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            df.columns = df.columns.str.lower()
            return df
        except ImportError:
            print("Warning: yfinance not installed, using mock data")
            return self._generate_mock_ohlcv(symbol, start_date, end_date, interval)

    def _generate_mock_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> pd.DataFrame:
        """Generate mock OHLCV data for demonstration."""
        np.random.seed(hash(symbol) % 2**32)

        # Determine frequency
        freq = "D" if interval == "1d" else "H" if interval in ["1h", "60m"] else "D"
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Base prices for common symbols
        base_prices = {
            "AAPL": 180,
            "NVDA": 500,
            "TSLA": 250,
            "MSFT": 380,
            "GOOGL": 140,
            "BTCUSDT": 50000,
            "ETHUSDT": 3000,
            "BTC": 50000,
            "ETH": 3000,
        }
        base_price = base_prices.get(symbol.upper(), 100)

        # Generate price series
        returns = np.random.randn(len(dates)) * 0.015
        close = base_price * (1 + returns).cumprod()

        # Generate OHLCV
        high = close * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
        low = close * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
        open_price = np.roll(close, 1)
        open_price[0] = base_price
        volume = np.random.uniform(1e6, 1e7, len(dates))

        df = pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        }, index=dates)

        return df

    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Latest price
        """
        if self.use_mock:
            prices = {
                "AAPL": 185.5,
                "NVDA": 520.3,
                "TSLA": 242.8,
                "MSFT": 390.2,
                "BTCUSDT": 52340.0,
                "ETHUSDT": 2890.5,
            }
            return prices.get(symbol.upper(), 100.0)

        df = self.load_stock_data(
            symbol,
            (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )
        return df["close"].iloc[-1] if not df.empty else 0.0

    def load_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            DataFrame with close prices for all symbols
        """
        dfs = {}
        for symbol in symbols:
            df = self.load_stock_data(symbol, start_date, end_date, interval)
            dfs[symbol] = df["close"]

        return pd.DataFrame(dfs)


class NewsDataLoader:
    """
    Load financial news data for sentiment analysis.

    This is a mock implementation for demonstration purposes.
    In production, integrate with actual news APIs.
    """

    def __init__(self):
        self.mock_news = [
            {
                "symbol": "AAPL",
                "timestamp": datetime.now() - timedelta(hours=2),
                "headline": "Apple reports record quarterly revenue beating expectations",
                "source": "earnings"
            },
            {
                "symbol": "NVDA",
                "timestamp": datetime.now() - timedelta(hours=5),
                "headline": "NVIDIA AI chip demand continues to surge amid AI boom",
                "source": "news_wire"
            },
            {
                "symbol": "TSLA",
                "timestamp": datetime.now() - timedelta(hours=8),
                "headline": "Tesla faces increased competition in EV market",
                "source": "analyst"
            },
            {
                "symbol": "BTCUSDT",
                "timestamp": datetime.now() - timedelta(hours=1),
                "headline": "Bitcoin ETF sees record inflows as institutional adoption grows",
                "source": "news_wire"
            },
        ]

    def get_news(
        self,
        symbols: Optional[List[str]] = None,
        hours_back: int = 24
    ) -> List[Dict]:
        """
        Get recent news for symbols.

        Args:
            symbols: Filter by symbols (None for all)
            hours_back: How far back to look

        Returns:
            List of news items
        """
        cutoff = datetime.now() - timedelta(hours=hours_back)
        news = [n for n in self.mock_news if n["timestamp"] >= cutoff]

        if symbols:
            news = [n for n in news if n["symbol"] in symbols]

        return news


if __name__ == "__main__":
    print("=" * 70)
    print("Market Data Loader Demo")
    print("=" * 70)

    loader = MarketDataLoader(use_mock=True)

    # Load stock data
    print("\nLoading AAPL data...")
    aapl = loader.load_stock_data("AAPL", "2024-01-01", "2024-03-31")
    print(f"Loaded {len(aapl)} bars")
    print(aapl.head())

    # Load multiple symbols
    print("\n" + "-" * 50)
    print("Loading multiple symbols...")
    symbols = ["AAPL", "NVDA", "TSLA"]
    multi = loader.load_multiple(symbols, "2024-01-01", "2024-03-31")
    print(f"Shape: {multi.shape}")
    print(multi.head())

    # Get latest prices
    print("\n" + "-" * 50)
    print("Latest prices:")
    for symbol in ["AAPL", "NVDA", "BTCUSDT"]:
        price = loader.get_latest_price(symbol)
        print(f"  {symbol}: ${price:,.2f}")

    # Load news
    print("\n" + "-" * 50)
    print("Recent news:")
    news_loader = NewsDataLoader()
    news = news_loader.get_news()
    for item in news:
        print(f"  [{item['symbol']}] {item['headline'][:50]}...")
