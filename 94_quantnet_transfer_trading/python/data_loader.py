"""
Data loading and feature engineering for QuantNet transfer trading.

Supports:
- BybitDataLoader: Cryptocurrency data from Bybit exchange API
- StockDataLoader: Stock data via yfinance
- Technical feature generation: returns, log returns, volatility,
  momentum, RSI, MACD, Bollinger bands
- create_features(): Unified feature creation pipeline
"""

import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BybitDataLoader:
    """
    Data loader for cryptocurrency market data from Bybit.

    Fetches historical kline (candlestick) data from the Bybit v5 API
    and provides feature engineering utilities.

    Args:
        base_url: Bybit API base URL
    """

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or self.BASE_URL

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch historical klines from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Candle interval ('1', '5', '15', '60', '240', 'D')
            limit: Number of candles to fetch (max 1000)

        Returns:
            DataFrame with columns: open, high, low, close, volume, turnover
        """
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        response = requests.get(self.base_url, params=params)
        response.raise_for_status()

        data = response.json()["result"]["list"]
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close",
                      "volume", "turnover"],
        )

        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype(int), unit="ms",
        )
        df = df.set_index("timestamp").sort_index()

        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df

    def fetch_multi_asset(
        self,
        symbols: List[str],
        interval: str = "60",
        limit: int = 1000,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple assets.

        Args:
            symbols: List of trading pair symbols
            interval: Candle interval
            limit: Number of candles per symbol

        Returns:
            Dict mapping symbol -> OHLCV DataFrame
        """
        data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = self.fetch_klines(symbol, interval, limit)
                data[symbol] = df
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
        return data


class StockDataLoader:
    """
    Data loader for stock market data via yfinance.

    Provides the same interface as BybitDataLoader but for
    traditional equity markets.
    """

    def fetch_stock(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical stock data.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
            interval: Data interval ('1d', '1wk', '1mo')

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error(
                "yfinance not installed. "
                "Install with: pip install yfinance"
            )
            raise

        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'")

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        logger.info(f"Fetched {len(df)} bars for {ticker}")
        return df

    def fetch_multi_stock(
        self,
        tickers: List[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.

        Args:
            tickers: List of ticker symbols
            period: Data period
            interval: Data interval

        Returns:
            Dict mapping ticker -> DataFrame
        """
        data: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                df = self.fetch_stock(ticker, period, interval)
                data[ticker] = df
            except Exception as e:
                logger.error(f"Failed to load {ticker}: {e}")
        return data


def create_features(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Create technical features for QuantNet.

    Generates a comprehensive set of features from OHLCV data:
    - Returns (1-period, 5-period, 10-period)
    - Log returns
    - Rolling volatility
    - Momentum
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Band position
    - Moving average ratios (SMA, EMA)

    Args:
        df: DataFrame with at least a price column
        window: Lookback window for rolling features
        price_col: Name of the price column to use

    Returns:
        DataFrame with computed features, NaN rows dropped
    """
    prices = df[price_col]
    features = pd.DataFrame(index=df.index)

    # Returns at multiple horizons
    features["return_1d"] = prices.pct_change(1)
    features["return_5d"] = prices.pct_change(5)
    features["return_10d"] = prices.pct_change(10)

    # Log returns
    features["log_return_1d"] = np.log(prices / prices.shift(1))
    features["log_return_5d"] = np.log(prices / prices.shift(5))

    # Rolling volatility
    features["volatility"] = prices.pct_change().rolling(window).std()
    features["volatility_5d"] = prices.pct_change().rolling(5).std()

    # Momentum
    features["momentum"] = prices / prices.shift(window) - 1
    features["momentum_5d"] = prices / prices.shift(5) - 1

    # RSI (Relative Strength Index)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    features["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    features["macd"] = macd_line / prices
    features["macd_signal"] = signal_line / prices
    features["macd_hist"] = (macd_line - signal_line) / prices

    # Bollinger Bands
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    features["bb_position"] = (prices - sma) / (2 * std + 1e-10)
    features["bb_width"] = (4 * std) / (sma + 1e-10)

    # Moving average ratios
    features["sma_ratio"] = prices / sma
    features["ema_ratio"] = prices / prices.ewm(span=window).mean()

    return features.dropna()


def prepare_quantnet_dataset(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = "close",
    forward_period: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare aligned features and forward returns for QuantNet.

    Args:
        df: DataFrame with OHLCV data
        window: Feature lookback window
        price_col: Price column name
        forward_period: Number of periods ahead for return target

    Returns:
        (features_df, returns_series) with aligned indices, NaN-free
    """
    features = create_features(df, window=window, price_col=price_col)

    # Forward returns as target
    forward_returns = df[price_col].pct_change(forward_period).shift(
        -forward_period
    )
    forward_returns = forward_returns.loc[features.index].dropna()

    # Align
    common_idx = features.index.intersection(forward_returns.index)
    features = features.loc[common_idx]
    forward_returns = forward_returns.loc[common_idx]

    logger.info(
        f"Prepared dataset: {len(features)} samples, "
        f"{features.shape[1]} features"
    )
    return features, forward_returns


def prepare_multi_asset_dataset(
    asset_data: Dict[str, pd.DataFrame],
    window: int = 20,
    price_col: str = "close",
    forward_period: int = 1,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Prepare datasets for multiple assets.

    Args:
        asset_data: Dict mapping asset name -> OHLCV DataFrame
        window: Feature lookback window
        price_col: Price column name
        forward_period: Forward return period

    Returns:
        Dict mapping asset name -> (features_df, returns_series)
    """
    datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for asset, df in asset_data.items():
        try:
            features, returns = prepare_quantnet_dataset(
                df, window, price_col, forward_period,
            )
            datasets[asset] = (features, returns)
        except Exception as e:
            logger.error(f"Failed to prepare dataset for {asset}: {e}")
    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo with synthetic data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="h")
    prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.01))
    volume = np.abs(np.random.randn(500) * 1000)

    df = pd.DataFrame(
        {"close": prices, "volume": volume},
        index=dates,
    )

    features = create_features(df)
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Feature columns: {list(features.columns)}")

    features_aligned, returns = prepare_quantnet_dataset(df)
    logger.info(f"Aligned features: {features_aligned.shape}")
    logger.info(f"Returns shape: {returns.shape}")
