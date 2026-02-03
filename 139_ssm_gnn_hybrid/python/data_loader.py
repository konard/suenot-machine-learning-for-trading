"""
Data loading utilities for SSM-GNN Hybrid.

Provides data fetching from:
- Yahoo Finance (stock market)
- Bybit API (cryptocurrency market)

Includes graph construction from correlation matrices.
"""

import logging
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

BYBIT_BASE_URL = "https://api.bybit.com"


def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    limit: int = 200,
) -> dict:
    """
    Fetch kline (candlestick) data from Bybit public API.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        interval: Kline interval in minutes (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M).
        limit: Number of candles to fetch (max 200).

    Returns:
        Dictionary with keys: open, high, low, close, volume, timestamp.
    """
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 200),
    }

    logger.info("Fetching Bybit klines: symbol=%s, interval=%s, limit=%d",
                symbol, interval, limit)

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("retCode") != 0:
            logger.warning("Bybit API error: %s", data.get("retMsg"))
            return _generate_simulated_data(symbol, limit)

        result = data.get("result", {}).get("list", [])
        if not result:
            logger.warning("Empty response from Bybit for %s", symbol)
            return _generate_simulated_data(symbol, limit)

        # Bybit returns newest first, reverse for chronological order
        result = list(reversed(result))

        return {
            "timestamp": [int(r[0]) for r in result],
            "open": np.array([float(r[1]) for r in result]),
            "high": np.array([float(r[2]) for r in result]),
            "low": np.array([float(r[3]) for r in result]),
            "close": np.array([float(r[4]) for r in result]),
            "volume": np.array([float(r[5]) for r in result]),
        }

    except Exception as e:
        logger.warning("Failed to fetch Bybit data for %s: %s. Using simulated data.", symbol, e)
        return _generate_simulated_data(symbol, limit)


def _generate_simulated_data(symbol: str, n_points: int) -> dict:
    """Generate simulated price data for offline/testing use."""
    rng = np.random.default_rng(hash(symbol) % (2**31))

    # Simulate geometric Brownian motion
    base_price = {"BTCUSDT": 60000, "ETHUSDT": 3000, "SOLUSDT": 150,
                  "AVAXUSDT": 35, "MATICUSDT": 0.8}.get(symbol, 100)

    returns = rng.normal(0.0001, 0.02, n_points)
    close = base_price * np.cumprod(1 + returns)

    # Generate OHLCV from close
    noise = rng.uniform(0.995, 1.005, n_points)
    open_price = close * noise
    high = np.maximum(open_price, close) * rng.uniform(1.0, 1.02, n_points)
    low = np.minimum(open_price, close) * rng.uniform(0.98, 1.0, n_points)
    volume = rng.exponential(1000, n_points) * base_price / 100

    return {
        "timestamp": list(range(n_points)),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def load_multi_asset_data(
    symbols: Optional[list] = None,
    interval: str = "60",
    limit: int = 200,
) -> tuple:
    """
    Load data for multiple assets from Bybit.

    Args:
        symbols: List of trading pair symbols. Defaults to common crypto pairs.
        interval: Kline interval.
        limit: Number of candles per asset.

    Returns:
        Tuple of (prices_dict, all_close) where:
            prices_dict: {symbol: kline_data}
            all_close: (n_assets, seq_len) close price matrix
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT"]

    prices_dict = {}
    close_arrays = []

    for sym in symbols:
        data = fetch_bybit_klines(sym, interval, limit)
        prices_dict[sym] = data
        close_arrays.append(data["close"])

    # Align to minimum length
    min_len = min(len(c) for c in close_arrays)
    all_close = np.array([c[:min_len] for c in close_arrays])

    return prices_dict, all_close


def build_correlation_graph(
    price_matrix: np.ndarray,
    threshold: float = 0.3,
) -> tuple:
    """
    Build a graph from asset correlation matrix.

    Args:
        price_matrix: (n_assets, seq_len) price matrix.
        threshold: Minimum absolute correlation to create an edge.

    Returns:
        Tuple of (edge_index, edge_weight) where:
            edge_index: (2, n_edges) array of source-target indices.
            edge_weight: (n_edges,) array of correlation values.
    """
    n_assets = price_matrix.shape[0]

    # Compute returns
    returns = np.diff(price_matrix, axis=1) / (price_matrix[:, :-1] + 1e-10)

    # Correlation matrix
    corr = np.corrcoef(returns)

    # Build edges from thresholded correlation
    sources, targets, weights = [], [], []
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j and abs(corr[i, j]) >= threshold:
                sources.append(i)
                targets.append(j)
                weights.append(abs(corr[i, j]))

    if not sources:
        # Fallback: fully connected with uniform weights
        logger.warning("No edges above threshold %.2f, using fully connected graph", threshold)
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    sources.append(i)
                    targets.append(j)
                    weights.append(1.0 / n_assets)

    edge_index = np.array([sources, targets])
    edge_weight = np.array(weights)

    logger.info("Graph constructed: %d nodes, %d edges", n_assets, len(weights))

    return edge_index, edge_weight


def prepare_features(
    prices_dict: dict,
    symbols: list,
    window: int = 14,
) -> np.ndarray:
    """
    Prepare feature matrices for all assets.

    Args:
        prices_dict: Dictionary of kline data per symbol.
        symbols: List of symbols in order.
        window: Lookback window for technical indicators.

    Returns:
        (n_assets, seq_len, n_features) feature tensor.
    """
    from python.ssm_gnn_model import compute_technical_features

    feature_list = []
    for sym in symbols:
        close = prices_dict[sym]["close"]
        feats = compute_technical_features(close, window=window)
        feature_list.append(feats)

    # Align lengths
    min_len = min(f.shape[0] for f in feature_list)
    features = np.array([f[:min_len] for f in feature_list])

    return features
