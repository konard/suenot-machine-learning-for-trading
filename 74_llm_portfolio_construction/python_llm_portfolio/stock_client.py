"""
Stock Market Data Client Module

This module provides classes for fetching stock market data
using yfinance for portfolio construction.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StockInfo:
    """Stock information dataclass."""
    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    current_price: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None


class StockClient:
    """
    Stock market data client using yfinance.

    Fetches historical prices, fundamentals, and other data
    for stock portfolio construction.
    """

    def __init__(self):
        """Initialize stock client."""
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

    def fetch_stock_info(self, symbol: str) -> StockInfo:
        """
        Fetch stock information.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")

        Returns:
            StockInfo object
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return StockInfo(
            symbol=symbol,
            name=info.get("shortName", symbol),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=info.get("marketCap"),
            current_price=info.get("currentPrice") or info.get("regularMarketPrice"),
            pe_ratio=info.get("trailingPE"),
            dividend_yield=info.get("dividendYield"),
        )

    def fetch_historical_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical price data.

        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data found for {symbol}")

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        return df

    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks.

        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval

        Returns:
            Dict mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            try:
                logger.info(f"Fetching {symbol}...")
                df = self.fetch_historical_data(symbol, period, interval)
                data[symbol] = df
                logger.info(f"  Got {len(df)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")

        return data

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to current price
        """
        prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                price = info.get("currentPrice") or info.get("regularMarketPrice")
                if price:
                    prices[symbol] = price
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")

        return prices


class StockPortfolioDataFetcher:
    """
    Fetches and prepares stock data for portfolio construction.
    """

    def __init__(self, client: Optional[StockClient] = None):
        """
        Initialize fetcher.

        Args:
            client: StockClient instance (creates new if None)
        """
        self.client = client or StockClient()

    def fetch_portfolio_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all portfolio symbols.

        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval

        Returns:
            Dict mapping symbol to DataFrame
        """
        return self.client.fetch_multiple_stocks(symbols, period, interval)

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
            cov *= 252  # Stock market trades ~252 days
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
            method: Calculation method ('mean', 'ewma', 'capm')
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
            expected *= 252  # Stock market trades ~252 days

        return expected

    def get_fundamental_metrics(
        self,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Get fundamental metrics for stocks.

        Args:
            symbols: List of stock symbols

        Returns:
            DataFrame with fundamental metrics
        """
        metrics = []
        for symbol in symbols:
            try:
                info = self.client.fetch_stock_info(symbol)
                metrics.append({
                    "symbol": symbol,
                    "name": info.name,
                    "sector": info.sector,
                    "market_cap": info.market_cap,
                    "pe_ratio": info.pe_ratio,
                    "dividend_yield": info.dividend_yield,
                    "current_price": info.current_price,
                })
            except Exception as e:
                logger.warning(f"Failed to get fundamentals for {symbol}: {e}")

        return pd.DataFrame(metrics)


# Example usage
if __name__ == "__main__":
    print("Stock Portfolio Data Client Demo")
    print("=" * 50)

    if not YFINANCE_AVAILABLE:
        print("yfinance not installed. Install with: pip install yfinance")
        exit(1)

    client = StockClient()

    # Stock portfolio symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

    # Fetch data
    print("\nFetching portfolio data...")
    fetcher = StockPortfolioDataFetcher(client)
    data = fetcher.fetch_portfolio_data(symbols, period="1y")

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
        vol = returns_df[symbol].std() * np.sqrt(252)
        print(f"  {symbol}: {vol:.2%}")

    print("\nCorrelation Matrix:")
    print(returns_df.corr().round(3))

    # Get fundamental metrics
    print("\nFundamental Metrics:")
    fundamentals = fetcher.get_fundamental_metrics(symbols)
    print(fundamentals.to_string(index=False))
