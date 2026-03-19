"""
Data loading utilities for DeBERTa trading.

Supports data from:
- Bybit (cryptocurrency via public REST API)
- Yahoo Finance (stocks via yfinance)

Generates synthetic data for testing when APIs are unavailable.
"""

import numpy as np
import pandas as pd
import requests
import logging
from typing import Optional
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
            return _generate_synthetic_crypto(limit, symbol)

        rows = data["result"]["list"]
        if not rows:
            logger.warning(f"No data returned for {symbol}")
            return _generate_synthetic_crypto(limit, symbol)

        df = pd.DataFrame(rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Fetched {len(df)} candles for {symbol} from Bybit")
        return df

    except requests.RequestException as e:
        logger.warning(f"Bybit API request failed: {e}. Using synthetic data.")
        return _generate_synthetic_crypto(limit, symbol)


def fetch_stock_data(
    symbol: str = "AAPL",
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker (e.g., "AAPL", "MSFT", "TSLA")
        period: Data period ("1mo", "3mo", "6mo", "1y", "2y")
        interval: Data interval ("1d", "1wk", "1mo")

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return _generate_synthetic_stock(200, symbol)

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        logger.info(f"Fetched {len(df)} candles for {symbol} from Yahoo Finance")
        return df

    except ImportError:
        logger.warning("yfinance not available. Using synthetic data.")
        return _generate_synthetic_stock(200, symbol)
    except Exception as e:
        logger.warning(f"Yahoo Finance request failed: {e}. Using synthetic data.")
        return _generate_synthetic_stock(200, symbol)


def generate_sample_headlines(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generate sample financial headlines for testing.

    Args:
        n: Number of headlines to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: timestamp, text, true_sentiment
    """
    rng = np.random.RandomState(seed)

    positive_templates = [
        "{asset} surges {pct}% on strong {metric} report",
        "{asset} beats Q{q} {metric} expectations by {pct}%",
        "Institutional investors boost {asset} holdings",
        "{asset} rally continues as {metric} hits record high",
        "Analysts upgrade {asset} outlook on {metric} growth",
        "{asset} gains momentum after positive {metric} data",
        "Record {metric} drives {asset} to new highs",
        "{asset} outperforms as market sentiment improves",
    ]
    negative_templates = [
        "{asset} drops {pct}% amid {risk} concerns",
        "{asset} misses Q{q} {metric} estimates",
        "Regulatory crackdown threatens {asset} market",
        "{asset} selloff deepens on {risk} fears",
        "Analysts downgrade {asset} citing {risk} risks",
        "{asset} plunges after {risk} announcement",
        "{risk} investigation weighs on {asset}",
        "{asset} declines as {metric} disappoints",
    ]
    neutral_templates = [
        "{asset} trades flat ahead of {metric} announcement",
        "{asset} consolidates near key {metric} level",
        "Markets await {asset} {metric} report",
        "{asset} volume unchanged as traders assess {metric}",
    ]

    assets = ["Bitcoin", "Ethereum", "Apple", "Tesla", "Microsoft", "NVIDIA"]
    metrics = ["revenue", "earnings", "growth", "demand", "adoption"]
    risks = ["regulatory", "security", "inflation", "recession", "fraud"]

    headlines = []
    for i in range(n):
        r = rng.random()
        asset = rng.choice(assets)
        metric = rng.choice(metrics)
        risk = rng.choice(risks)
        pct = rng.randint(1, 20)
        q = rng.randint(1, 5)

        if r < 0.4:
            template = rng.choice(positive_templates)
            sentiment = "positive"
        elif r < 0.8:
            template = rng.choice(negative_templates)
            sentiment = "negative"
        else:
            template = rng.choice(neutral_templates)
            sentiment = "neutral"

        text = template.format(
            asset=asset, metric=metric, risk=risk, pct=pct, q=q
        )
        headlines.append({
            "timestamp": datetime.now() - timedelta(days=n - i),
            "text": text,
            "true_sentiment": sentiment,
        })

    return pd.DataFrame(headlines)


def _generate_synthetic_crypto(n: int, symbol: str) -> pd.DataFrame:
    """Generate synthetic cryptocurrency OHLCV data."""
    logger.info(f"Generating synthetic crypto data for {symbol}")
    rng = np.random.RandomState(hash(symbol) % 2**31)

    base_price = {"BTCUSDT": 50000, "ETHUSDT": 3000, "SOLUSDT": 100}.get(
        symbol, 1000
    )

    prices = [base_price]
    for _ in range(n - 1):
        ret = rng.normal(0.0002, 0.03)
        prices.append(prices[-1] * (1 + ret))

    timestamps = pd.date_range(end=datetime.now(), periods=n, freq="D")
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        high = close * (1 + abs(rng.normal(0, 0.015)))
        low = close * (1 - abs(rng.normal(0, 0.015)))
        open_price = low + rng.random() * (high - low)
        volume = rng.uniform(1e6, 1e8)
        data.append({
            "timestamp": ts,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    return pd.DataFrame(data)


def _generate_synthetic_stock(n: int, symbol: str) -> pd.DataFrame:
    """Generate synthetic stock OHLCV data."""
    logger.info(f"Generating synthetic stock data for {symbol}")
    rng = np.random.RandomState(hash(symbol) % 2**31)

    base_price = {"AAPL": 175, "MSFT": 380, "TSLA": 250, "NVDA": 500}.get(
        symbol, 100
    )

    prices = [base_price]
    for _ in range(n - 1):
        ret = rng.normal(0.0003, 0.02)
        prices.append(prices[-1] * (1 + ret))

    timestamps = pd.date_range(end=datetime.now(), periods=n, freq="B")
    data = []
    for ts, close in zip(timestamps, prices):
        high = close * (1 + abs(rng.normal(0, 0.01)))
        low = close * (1 - abs(rng.normal(0, 0.01)))
        open_price = low + rng.random() * (high - low)
        volume = rng.uniform(1e6, 5e7)
        data.append({
            "timestamp": ts,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    return pd.DataFrame(data)
