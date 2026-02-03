"""Data loading and feature engineering for Transfer Entropy trading."""

import numpy as np
import pandas as pd
import requests
from typing import Optional


def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch kline/candlestick data from Bybit API.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT').
        interval: Kline interval in minutes ('1', '5', '15', '60', '240', 'D').
        limit: Number of klines to fetch (max 1000).

    Returns:
        DataFrame with columns: open, high, low, close, volume and datetime index.
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")

    rows = data["result"]["list"]
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def generate_simulated_data(
    n_assets: int = 4,
    n_periods: int = 500,
    lead_lag_strength: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate simulated asset data with lead-lag relationships.

    The first asset (leader) influences subsequent assets with decreasing
    strength and increasing delay.

    Args:
        n_assets: Number of assets to simulate.
        n_periods: Number of time periods.
        lead_lag_strength: Strength of the lead-lag effect (0 to 1).
        seed: Random seed.

    Returns:
        Tuple of (prices_df, returns_df) with columns for each asset.
    """
    rng = np.random.default_rng(seed)
    names = [f"Asset_{i}" for i in range(n_assets)]

    # Generate leader's returns
    leader_returns = rng.normal(0.0005, 0.02, n_periods)

    all_returns = np.zeros((n_periods, n_assets))
    all_returns[:, 0] = leader_returns

    # Generate follower returns with lag
    for i in range(1, n_assets):
        lag = i  # asset i follows leader with lag i
        noise = rng.normal(0.0003, 0.02, n_periods)
        strength = lead_lag_strength / i
        for t in range(lag, n_periods):
            all_returns[t, i] = (
                strength * leader_returns[t - lag] + (1 - strength) * noise[t]
            )
        # Fill initial periods with noise
        all_returns[:lag, i] = noise[:lag]

    returns_df = pd.DataFrame(all_returns, columns=names)

    # Compute prices from returns
    prices = np.zeros((n_periods, n_assets))
    for i in range(n_assets):
        prices[0, i] = 100.0 * (1 + i * 0.5)  # different starting prices
        for t in range(1, n_periods):
            prices[t, i] = prices[t - 1, i] * (1 + all_returns[t, i])

    prices_df = pd.DataFrame(prices, columns=names)
    return prices_df, returns_df


def generate_crypto_simulated_data(
    n_periods: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate simulated crypto data with realistic lead-lag structure.

    BTC leads ETH, SOL, AVAX with different strengths and delays.

    Returns:
        Tuple of (prices_df, returns_df).
    """
    rng = np.random.default_rng(seed)
    symbols = ["BTC", "ETH", "SOL", "AVAX"]

    btc_returns = rng.normal(0.0005, 0.025, n_periods)

    all_returns = np.zeros((n_periods, 4))
    all_returns[:, 0] = btc_returns

    # ETH follows BTC with lag 1, moderate strength
    eth_noise = rng.normal(0.0003, 0.03, n_periods)
    for t in range(1, n_periods):
        all_returns[t, 1] = 0.25 * btc_returns[t - 1] + 0.75 * eth_noise[t]

    # SOL follows BTC with lag 2, weaker strength + follows ETH with lag 1
    sol_noise = rng.normal(0.0002, 0.04, n_periods)
    for t in range(2, n_periods):
        all_returns[t, 2] = (
            0.15 * btc_returns[t - 2]
            + 0.10 * all_returns[t - 1, 1]
            + 0.75 * sol_noise[t]
        )

    # AVAX follows BTC with lag 3, weakest + some SOL influence
    avax_noise = rng.normal(0.0001, 0.045, n_periods)
    for t in range(3, n_periods):
        all_returns[t, 3] = (
            0.12 * btc_returns[t - 3]
            + 0.08 * all_returns[t - 1, 2]
            + 0.80 * avax_noise[t]
        )

    returns_df = pd.DataFrame(all_returns, columns=symbols)

    # Prices
    start_prices = {"BTC": 50000.0, "ETH": 3000.0, "SOL": 100.0, "AVAX": 30.0}
    prices = np.zeros((n_periods, 4))
    for i, sym in enumerate(symbols):
        prices[0, i] = start_prices[sym]
        for t in range(1, n_periods):
            prices[t, i] = prices[t - 1, i] * (1 + all_returns[t, i])

    prices_df = pd.DataFrame(prices, columns=symbols)
    return prices_df, returns_df


def compute_rolling_te(
    returns: dict[str, np.ndarray],
    source: str,
    target: str,
    window: int = 60,
    history_length: int = 3,
    n_bins: int = 8,
) -> np.ndarray:
    """Compute Transfer Entropy on a rolling window basis.

    Args:
        returns: Dict mapping symbol to return series.
        source: Source symbol name.
        target: Target symbol name.
        window: Rolling window size.
        history_length: TE history length.
        n_bins: Number of bins for discretization.

    Returns:
        Array of rolling TE values (NaN for initial window periods).
    """
    from .te_model import TransferEntropyEstimator

    estimator = TransferEntropyEstimator(
        method="binning", n_bins=n_bins, history_length=history_length
    )

    src = returns[source]
    tgt = returns[target]
    n = len(src)

    te_values = np.full(n, np.nan)
    for t in range(window, n):
        src_window = src[t - window : t]
        tgt_window = tgt[t - window : t]
        try:
            te_values[t] = estimator.compute_te(src_window, tgt_window)
        except (ValueError, ZeroDivisionError):
            te_values[t] = 0.0

    return te_values
