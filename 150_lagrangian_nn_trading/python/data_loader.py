"""
Data loading and configuration space construction for LNN trading.

Supports:
  - Bybit V5 API (cryptocurrency: BTC/USDT, ETH/USDT, etc.)
  - Yahoo Finance (stocks: SPY, AAPL, etc.)

Constructs configuration space (q, q-dot, q-ddot) from OHLCV data:
  q:      generalized coordinates (price deviation from moving average)
  q-dot:  generalized velocities (rate of deviation change)
  q-ddot: generalized accelerations (for training targets)

Unlike HNN data (which needs q, p, dq/dt, dp/dt), LNN data uses
(q, q-dot, q-ddot) -- positions, velocities, accelerations.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import requests
from typing import Tuple, Optional, Dict, List


# =============================================================================
# Bybit Data Fetching
# =============================================================================


def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5",
    limit: int = 1000,
    category: str = "linear",
    end_time: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV kline data from Bybit V5 API.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        interval: Kline interval in minutes ("1", "3", "5", "15", "30", "60", "240", "D", "W").
        limit: Number of candles to fetch (max 1000 per request).
        category: Market category ("linear" for perpetual futures, "spot" for spot).
        end_time: End timestamp in milliseconds (optional).

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, turnover
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if end_time is not None:
        params["end"] = str(end_time)

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data["retCode"] != 0:
        raise ValueError(f"Bybit API error: {data['retMsg']}")

    rows = data["result"]["list"]
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_bybit_extended(
    symbol: str = "BTCUSDT",
    interval: str = "5",
    total_candles: int = 5000,
    category: str = "linear",
) -> pd.DataFrame:
    """
    Fetch extended history from Bybit by paginating through the API.

    Args:
        symbol: Trading pair.
        interval: Candle interval.
        total_candles: Total number of candles to fetch.
        category: Market category.

    Returns:
        DataFrame with full OHLCV history.
    """
    all_dfs: List[pd.DataFrame] = []
    end_time = None
    remaining = total_candles

    while remaining > 0:
        batch_size = min(remaining, 1000)
        df = fetch_bybit_klines(
            symbol=symbol,
            interval=interval,
            limit=batch_size,
            category=category,
            end_time=end_time,
        )

        if len(df) == 0:
            break

        all_dfs.append(df)
        remaining -= len(df)

        # Set end_time to earliest timestamp for next batch
        earliest_ts = df["timestamp"].min()
        end_time = int(earliest_ts.timestamp() * 1000) - 1

        # Rate limiting
        time.sleep(0.1)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates(subset=["timestamp"])
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


# =============================================================================
# Yahoo Finance Data Fetching
# =============================================================================


def fetch_yahoo_data(
    symbol: str = "SPY",
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.

    Args:
        symbol: Stock ticker (e.g., "SPY", "AAPL").
        period: Data period ("1mo", "3mo", "6mo", "1y", "2y", "5y").
        interval: Data interval ("1d", "1h", "5m").

    Returns:
        DataFrame with OHLCV data.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df = df.reset_index()

    # Normalize column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    elif "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})

    # Keep only needed columns
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


# =============================================================================
# Configuration Space Construction (for LNN)
# =============================================================================


def construct_config_space(
    df: pd.DataFrame,
    ma_window: int = 20,
    velocity_method: str = "gradient",
    use_volume: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert OHLCV price data to (q, q-dot, q-ddot) configuration space.

    Unlike HNN phase space (q, p), LNN uses configuration space (q, q-dot):
      q     = generalized coordinates (position)
      q-dot = generalized velocities
      q-ddot = generalized accelerations (training target)

    Args:
        df: DataFrame with at least 'close' column.
        ma_window: Window for moving average (defines equilibrium).
        velocity_method: How to compute velocity:
            - "gradient": Central difference of q
            - "returns": Raw log returns
            - "ema": Exponential moving average of gradient
        use_volume: If True, add volume deviation as additional coordinate.

    Returns:
        q: Generalized coordinates, shape (N, q_dim)
        qdot: Generalized velocities, shape (N, q_dim)
        qddot: Generalized accelerations, shape (N, q_dim)
    """
    close = df["close"].values
    log_close = np.log(close)

    # q: Price deviation from moving average
    ma = pd.Series(log_close).rolling(ma_window).mean().values
    q_price = log_close - ma

    q_cols: List[np.ndarray] = [q_price]

    if use_volume and "volume" in df.columns:
        log_vol = np.log(df["volume"].values + 1)
        vol_ma = pd.Series(log_vol).rolling(ma_window).mean().values
        q_vol = log_vol - vol_ma
        q_cols.append(q_vol)

    q = np.column_stack(q_cols)

    # q-dot: velocity
    if velocity_method == "gradient":
        qdot = np.gradient(q, axis=0)
    elif velocity_method == "returns":
        qdot = np.diff(q, axis=0, prepend=q[:1])
    elif velocity_method == "ema":
        raw_grad = np.gradient(q, axis=0)
        alpha = 2.0 / (5 + 1)
        qdot = np.zeros_like(raw_grad)
        qdot[0] = raw_grad[0]
        for i in range(1, len(raw_grad)):
            qdot[i] = alpha * raw_grad[i] + (1 - alpha) * qdot[i - 1]
    else:
        raise ValueError(f"Unknown velocity_method: {velocity_method}")

    # q-ddot: acceleration (training target for LNN)
    qddot = np.gradient(qdot, axis=0)

    # Remove NaN rows (from moving average warmup)
    valid = np.all(np.isfinite(q), axis=1)
    valid &= np.all(np.isfinite(qdot), axis=1)
    valid &= np.all(np.isfinite(qddot), axis=1)

    return q[valid], qdot[valid], qddot[valid]


def construct_multiscale_config_space(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct multi-scale configuration space with multiple MA windows.

    Args:
        df: DataFrame with 'close' column.
        windows: List of MA windows (e.g., [5, 20, 50]).

    Returns:
        q, qdot, qddot arrays with dimension = len(windows).
    """
    if windows is None:
        windows = [5, 20, 50]

    close = df["close"].values
    log_close = np.log(close)

    q_cols = []

    for w in windows:
        ma = pd.Series(log_close).rolling(w).mean().values
        q_dev = log_close - ma
        q_cols.append(q_dev)

    q = np.column_stack(q_cols)
    qdot = np.gradient(q, axis=0)
    qddot = np.gradient(qdot, axis=0)

    valid = np.all(np.isfinite(q), axis=1)
    valid &= np.all(np.isfinite(qdot), axis=1)
    valid &= np.all(np.isfinite(qddot), axis=1)

    return q[valid], qdot[valid], qddot[valid]


# =============================================================================
# Feature Engineering
# =============================================================================


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Relative Strength Index."""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = pd.Series(gains).rolling(period).mean().values
    avg_loss = pd.Series(losses).rolling(period).mean().values

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_external_features(
    df: pd.DataFrame,
    rsi_period: int = 14,
    vol_window: int = 20,
) -> np.ndarray:
    """
    Compute external features for Forced LNN.

    Returns:
        external: Array of external features (N, 3):
          - RSI deviation from 50
          - Volatility z-score
          - Volume z-score
    """
    close = df["close"].values

    # RSI deviation
    rsi = compute_rsi(close, rsi_period)
    rsi_dev = (rsi - 50.0) / 50.0  # [-1, 1]

    # Volatility z-score
    log_returns = np.diff(np.log(close), prepend=np.log(close[0]))
    vol = pd.Series(log_returns).rolling(vol_window).std().values
    vol_mean = pd.Series(vol).rolling(100).mean().values
    vol_std = pd.Series(vol).rolling(100).std().values
    vol_zscore = (vol - vol_mean) / (vol_std + 1e-10)

    # Volume z-score
    volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
    vol_ma = pd.Series(volume).rolling(vol_window).mean().values
    vol_s = pd.Series(volume).rolling(vol_window).std().values
    volume_zscore = (volume - vol_ma) / (vol_s + 1e-10)

    external = np.column_stack([rsi_dev, vol_zscore, volume_zscore])

    # Replace NaN/inf
    external = np.nan_to_num(external, nan=0.0, posinf=3.0, neginf=-3.0)

    return external


def normalize_config_space(
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Normalize configuration space to zero mean, unit variance.

    Returns:
        Normalized (q, qdot, qddot) and stats dict for denormalization.
    """
    stats = {
        "q_mean": q.mean(0),
        "q_std": q.std(0) + 1e-8,
        "qdot_mean": qdot.mean(0),
        "qdot_std": qdot.std(0) + 1e-8,
        "qddot_mean": qddot.mean(0),
        "qddot_std": qddot.std(0) + 1e-8,
    }

    q_norm = (q - stats["q_mean"]) / stats["q_std"]
    qdot_norm = (qdot - stats["qdot_mean"]) / stats["qdot_std"]
    qddot_norm = (qddot - stats["qddot_mean"]) / stats["qddot_std"]

    return q_norm, qdot_norm, qddot_norm, stats


def denormalize_config_space(
    q_norm: np.ndarray,
    qdot_norm: np.ndarray,
    stats: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reverse normalization."""
    q = q_norm * stats["q_std"] + stats["q_mean"]
    qdot = qdot_norm * stats["qdot_std"] + stats["qdot_mean"]
    return q, qdot


def train_test_split_sequential(
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
    train_ratio: float = 0.8,
) -> Tuple:
    """
    Split configuration space data into train/test (sequential, no shuffle).

    Returns:
        (q_train, qdot_train, qddot_train,
         q_test, qdot_test, qddot_test)
    """
    n = len(q)
    split = int(n * train_ratio)
    return (
        q[:split],
        qdot[:split],
        qddot[:split],
        q[split:],
        qdot[split:],
        qddot[split:],
    )


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Fetch market data for LNN trading")
    parser.add_argument(
        "--source",
        type=str,
        default="bybit",
        choices=["bybit", "yahoo"],
        help="Data source",
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument(
        "--interval", type=str, default="5", help="Candle interval (minutes or D/W)"
    )
    parser.add_argument(
        "--limit", type=int, default=5000, help="Number of candles to fetch"
    )
    parser.add_argument(
        "--ma-window", type=int, default=20, help="Moving average window"
    )
    parser.add_argument(
        "--output", type=str, default="data", help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Fetching data from {args.source}: {args.symbol}...")

    if args.source == "bybit":
        df = fetch_bybit_extended(
            symbol=args.symbol,
            interval=args.interval,
            total_candles=args.limit,
        )
    else:
        period_map = {
            5000: "5y",
            2000: "2y",
            1000: "1y",
            500: "6mo",
        }
        period = "2y"
        for threshold, p in sorted(period_map.items()):
            if args.limit <= threshold:
                period = p
                break
        df = fetch_yahoo_data(symbol=args.symbol, period=period, interval="1d")

    print(f"Fetched {len(df)} candles")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")

    # Save raw data
    raw_path = os.path.join(args.output, f"{args.symbol}_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")

    # Construct configuration space
    print("\nConstructing configuration space...")
    q, qdot, qddot = construct_config_space(
        df, ma_window=args.ma_window, velocity_method="gradient"
    )
    print(f"Config space shape: q={q.shape}, qdot={qdot.shape}, qddot={qddot.shape}")
    print(f"q range: [{q.min():.6f}, {q.max():.6f}]")
    print(f"qdot range: [{qdot.min():.6f}, {qdot.max():.6f}]")
    print(f"qddot range: [{qddot.min():.6f}, {qddot.max():.6f}]")

    # Normalize
    q_norm, qdot_norm, qddot_norm, stats = normalize_config_space(q, qdot, qddot)

    # Save configuration space data
    config_path = os.path.join(args.output, f"{args.symbol}_config_space.npz")
    np.savez(
        config_path,
        q=q_norm,
        qdot=qdot_norm,
        qddot=qddot_norm,
        q_raw=q,
        qdot_raw=qdot,
        qddot_raw=qddot,
        **{f"stats_{k}": v for k, v in stats.items()},
    )
    print(f"Saved configuration space to {config_path}")

    # Train/test split
    splits = train_test_split_sequential(q_norm, qdot_norm, qddot_norm)
    print(f"\nTrain samples: {len(splits[0])}")
    print(f"Test samples:  {len(splits[3])}")


if __name__ == "__main__":
    main()
