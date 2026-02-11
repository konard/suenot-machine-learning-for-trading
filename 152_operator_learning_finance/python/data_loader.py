"""
Data Loader for Operator Learning in Finance

Fetches and prepares data from multiple sources:
- Stock data via yfinance (yield curve proxies, option-like features)
- Crypto data from Bybit REST API (order book, OHLCV, funding rates)
- Synthetic PDE solution data (Black-Scholes, Heston)

All data is formatted as function-valued observations suitable for
operator learning: input functions u(x) and output functions s(x).
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

try:
    import yfinance as yf
except ImportError:
    yf = None


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

BYBIT_BASE_URL = "https://api.bybit.com"

@dataclass
class DataConfig:
    """Configuration for data loading."""

    # Stock data
    stock_symbols: List[str] = None
    stock_period: str = "2y"
    stock_interval: str = "1d"

    # Crypto data
    crypto_symbol: str = "BTCUSDT"
    crypto_interval: str = "60"  # 1h candles
    crypto_limit: int = 1000

    # Yield curve proxies (Treasury ETFs by maturity)
    yield_etfs: List[str] = None

    # Synthetic data
    n_synthetic_samples: int = 2000
    n_sensor_points: int = 50
    n_eval_points: int = 100

    # Processing
    sequence_length: int = 60  # lookback window
    prediction_horizon: int = 5  # prediction steps ahead
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    def __post_init__(self):
        if self.stock_symbols is None:
            self.stock_symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
        if self.yield_etfs is None:
            self.yield_etfs = [
                "SHV",   # 0-1Y
                "SHY",   # 1-3Y
                "IEI",   # 3-7Y
                "IEF",   # 7-10Y
                "TLH",   # 10-20Y
                "TLT",   # 20+Y
            ]


# ═══════════════════════════════════════════════════════════════════════════════
# Bybit Data Fetching
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    limit: int = 1000,
    category: str = "linear",
) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data from Bybit.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval ('1', '5', '15', '60', '240', 'D')
        limit: Number of candles (max 1000)
        category: 'linear' for USDT perpetual, 'spot' for spot

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    if requests is None:
        raise ImportError("requests package required: pip install requests")

    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if data["retCode"] != 0:
        raise ValueError(f"Bybit API error: {data['retMsg']}")

    rows = data["result"]["list"]
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def fetch_bybit_orderbook(
    symbol: str = "BTCUSDT",
    limit: int = 50,
    category: str = "linear",
) -> Dict[str, np.ndarray]:
    """
    Fetch order book snapshot from Bybit.

    Args:
        symbol: Trading pair
        limit: Number of price levels per side (max 200)
        category: Market category

    Returns:
        Dict with 'bids', 'asks' arrays of shape (limit, 2) = [price, qty]
        and 'mid_price' scalar.
    """
    if requests is None:
        raise ImportError("requests package required: pip install requests")

    url = f"{BYBIT_BASE_URL}/v5/market/orderbook"
    params = {
        "category": category,
        "symbol": symbol,
        "limit": min(limit, 200),
    }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if data["retCode"] != 0:
        raise ValueError(f"Bybit API error: {data['retMsg']}")

    book = data["result"]

    bids = np.array([[float(p), float(q)] for p, q in book["b"]])
    asks = np.array([[float(p), float(q)] for p, q in book["a"]])

    mid_price = (bids[0, 0] + asks[0, 0]) / 2.0

    return {
        "bids": bids,
        "asks": asks,
        "mid_price": mid_price,
        "timestamp": book.get("ts", int(time.time() * 1000)),
    }


def fetch_bybit_funding_rate(
    symbol: str = "BTCUSDT",
    limit: int = 200,
) -> pd.DataFrame:
    """
    Fetch historical funding rates from Bybit perpetual futures.

    Args:
        symbol: Trading pair
        limit: Number of records

    Returns:
        DataFrame with funding rate history
    """
    if requests is None:
        raise ImportError("requests package required: pip install requests")

    url = f"{BYBIT_BASE_URL}/v5/market/funding/history"
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": min(limit, 200),
    }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if data["retCode"] != 0:
        raise ValueError(f"Bybit API error: {data['retMsg']}")

    rows = data["result"]["list"]
    df = pd.DataFrame(rows)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["fundingRateTimestamp"] = pd.to_datetime(
        df["fundingRateTimestamp"].astype(int), unit="ms"
    )
    df = df.sort_values("fundingRateTimestamp").reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Stock / ETF Data Fetching
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_stock_data(
    symbols: List[str],
    period: str = "2y",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch stock/ETF data from Yahoo Finance.

    Args:
        symbols: List of ticker symbols
        period: Data period (e.g., '2y', '5y')
        interval: Data interval (e.g., '1d', '1h')

    Returns:
        Dict mapping symbol to DataFrame with OHLCV data
    """
    if yf is None:
        raise ImportError("yfinance package required: pip install yfinance")

    result = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                df = df.reset_index()
                result[symbol] = df
        except Exception as e:
            print(f"Warning: Could not fetch {symbol}: {e}")

    return result


def fetch_yield_curve_proxy(
    etf_symbols: Optional[List[str]] = None,
    period: str = "2y",
) -> pd.DataFrame:
    """
    Construct yield curve proxy from Treasury ETFs of different maturities.

    The idea: use ETFs spanning the maturity spectrum as proxy observations
    for the yield curve shape at each point in time.

    Args:
        etf_symbols: List of Treasury ETF tickers ordered by maturity
        period: Data period

    Returns:
        DataFrame with each column being an ETF (maturity proxy),
        rows being dates, values being normalized returns.
    """
    if etf_symbols is None:
        etf_symbols = ["SHV", "SHY", "IEI", "IEF", "TLH", "TLT"]

    data = fetch_stock_data(etf_symbols, period=period)

    # Build aligned DataFrame of close prices
    closes = {}
    for sym in etf_symbols:
        if sym in data:
            df = data[sym].set_index("Date")["Close"]
            closes[sym] = df

    if not closes:
        raise ValueError("No yield curve proxy data fetched successfully")

    combined = pd.DataFrame(closes).dropna()

    # Convert prices to yield proxies (inverse relationship)
    # Using negative log returns as yield proxy
    log_returns = np.log(combined / combined.shift(1)).dropna()
    yield_proxy = -log_returns.cumsum()

    # Normalize each column to 0-1 range for operator input
    for col in yield_proxy.columns:
        mn = yield_proxy[col].min()
        mx = yield_proxy[col].max()
        if mx - mn > 1e-10:
            yield_proxy[col] = (yield_proxy[col] - mn) / (mx - mn)

    return yield_proxy


# ═══════════════════════════════════════════════════════════════════════════════
# Data Preparation for Operator Learning
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_deeponet_data_from_prices(
    prices: pd.Series,
    sequence_length: int = 60,
    prediction_horizon: int = 5,
    n_eval_points: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare price series for DeepONet training.

    The input function u(x) is the recent price curve (normalized).
    The output function s(y) is the future price curve.
    The evaluation points y are the future timestamps.

    Args:
        prices: Price series
        sequence_length: Length of input function (sensor points)
        prediction_horizon: Length of output function
        n_eval_points: Points at which we evaluate the output

    Returns:
        branch_inputs: (N, sequence_length) - input functions at sensor points
        trunk_inputs: (N, n_eval_points, 1) - evaluation locations
        targets: (N, n_eval_points) - output function values
    """
    values = prices.values.astype(np.float64)
    n = len(values)

    branch_list = []
    trunk_list = []
    target_list = []

    total_len = sequence_length + prediction_horizon
    eval_locations = np.linspace(0, 1, n_eval_points).reshape(1, -1, 1)

    for i in range(n - total_len):
        window = values[i : i + sequence_length]
        future = values[i + sequence_length : i + total_len]

        # Normalize input function to [0, 1]
        w_min, w_max = window.min(), window.max()
        if w_max - w_min < 1e-10:
            continue
        norm_window = (window - w_min) / (w_max - w_min)

        # Normalize output using same scaling
        norm_future = (future - w_min) / (w_max - w_min)

        # Interpolate future to eval points
        future_at_eval = np.interp(
            np.linspace(0, 1, n_eval_points),
            np.linspace(0, 1, prediction_horizon),
            norm_future,
        )

        branch_list.append(norm_window)
        trunk_list.append(eval_locations[0])
        target_list.append(future_at_eval)

    branch_inputs = np.array(branch_list, dtype=np.float32)
    trunk_inputs = np.array(trunk_list, dtype=np.float32)
    targets = np.array(target_list, dtype=np.float32)

    return branch_inputs, trunk_inputs, targets


def prepare_fno_data_from_prices(
    prices: pd.Series,
    sequence_length: int = 128,
    prediction_horizon: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare price series for FNO training.

    FNO maps one function (on a grid) to another function (on a grid).
    Input: recent price curve. Output: future price curve.

    Args:
        prices: Price series
        sequence_length: Grid size for input function
        prediction_horizon: Grid size for output function

    Returns:
        inputs: (N, 1, sequence_length) - input functions
        targets: (N, 1, sequence_length) - target output functions
    """
    values = prices.values.astype(np.float64)
    n = len(values)

    inputs_list = []
    targets_list = []

    total_len = sequence_length + prediction_horizon

    for i in range(n - total_len):
        window = values[i : i + sequence_length]
        future_raw = values[i + sequence_length : i + total_len]

        w_min, w_max = window.min(), window.max()
        if w_max - w_min < 1e-10:
            continue

        norm_window = (window - w_min) / (w_max - w_min)

        # Interpolate future to same grid size as input
        future_interp = np.interp(
            np.linspace(0, 1, sequence_length),
            np.linspace(0, 1, prediction_horizon),
            future_raw,
        )
        norm_future = (future_interp - w_min) / (w_max - w_min)

        inputs_list.append(norm_window)
        targets_list.append(norm_future)

    inputs = np.array(inputs_list, dtype=np.float32)[:, np.newaxis, :]
    targets = np.array(targets_list, dtype=np.float32)[:, np.newaxis, :]

    return inputs, targets


def prepare_orderbook_operator_data(
    symbol: str = "BTCUSDT",
    n_snapshots: int = 100,
    delay_seconds: float = 1.0,
    n_book_levels: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect order book snapshots and mid-price changes for operator learning.

    NOTE: This function collects real-time data and takes time proportional
    to n_snapshots * delay_seconds. For backtesting, use synthetic data instead.

    Args:
        symbol: Trading pair on Bybit
        n_snapshots: Number of snapshots to collect
        delay_seconds: Delay between snapshots
        n_book_levels: Number of order book levels per side

    Returns:
        book_features: (N, 2 * n_book_levels + trade_features)
        eval_points: (N, 2) - (horizon_index, price_offset)
        mid_changes: (N,) - actual mid-price changes
    """
    snapshots = []
    mid_prices = []

    print(f"Collecting {n_snapshots} order book snapshots for {symbol}...")

    for i in range(n_snapshots):
        try:
            book = fetch_bybit_orderbook(symbol, limit=n_book_levels)

            # Extract bid/ask quantities as function of price level
            bid_qty = book["bids"][:n_book_levels, 1] if len(book["bids"]) >= n_book_levels else np.pad(
                book["bids"][:, 1], (0, n_book_levels - len(book["bids"])), mode="constant"
            )
            ask_qty = book["asks"][:n_book_levels, 1] if len(book["asks"]) >= n_book_levels else np.pad(
                book["asks"][:, 1], (0, n_book_levels - len(book["asks"])), mode="constant"
            )

            features = np.concatenate([bid_qty, ask_qty])
            snapshots.append(features)
            mid_prices.append(book["mid_price"])

        except Exception as e:
            print(f"  Snapshot {i} failed: {e}")
            continue

        if i < n_snapshots - 1:
            time.sleep(delay_seconds)

    if len(snapshots) < 3:
        raise ValueError("Not enough snapshots collected")

    snapshots = np.array(snapshots, dtype=np.float32)
    mid_prices = np.array(mid_prices, dtype=np.float64)

    # Compute mid-price changes
    mid_changes = np.diff(mid_prices)
    book_features = snapshots[:-1]  # Align with changes

    # Eval points: normalized horizon and price offset
    n = len(mid_changes)
    eval_points = np.column_stack([
        np.ones(n) * 0.5,  # fixed single-step horizon
        np.zeros(n),       # zero price offset
    ]).astype(np.float32)

    return book_features, eval_points, mid_changes.astype(np.float32)


def prepare_crypto_price_operator_data(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    limit: int = 1000,
    sequence_length: int = 60,
    prediction_horizon: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Fetch crypto OHLCV from Bybit and prepare for operator learning.

    Returns:
        Dict with 'branch_inputs', 'trunk_inputs', 'targets' for DeepONet,
        and 'fno_inputs', 'fno_targets' for FNO.
    """
    df = fetch_bybit_klines(symbol, interval, limit)
    prices = df["close"]

    # DeepONet format
    branch, trunk, targets = prepare_deeponet_data_from_prices(
        prices, sequence_length, prediction_horizon
    )

    # FNO format
    fno_in, fno_out = prepare_fno_data_from_prices(
        prices, min(128, sequence_length * 2), prediction_horizon
    )

    return {
        "branch_inputs": branch,
        "trunk_inputs": trunk,
        "targets": targets,
        "fno_inputs": fno_in,
        "fno_targets": fno_out,
        "raw_df": df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Data Generators (for PDE operator learning)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_operator_data(
    operator_type: str = "heat",
    n_samples: int = 2000,
    n_points: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic input/output function pairs from known PDE operators.

    Args:
        operator_type: 'heat', 'advection', or 'burgers'
        n_samples: Number of function pairs
        n_points: Spatial grid resolution

    Returns:
        inputs: (n_samples, 1, n_points)
        outputs: (n_samples, 1, n_points)
    """
    x = np.linspace(0, 2 * np.pi, n_points)
    dx = x[1] - x[0]

    inputs = np.zeros((n_samples, 1, n_points), dtype=np.float32)
    outputs = np.zeros((n_samples, 1, n_points), dtype=np.float32)

    for i in range(n_samples):
        # Random initial condition (sum of random Fourier modes)
        n_modes = np.random.randint(3, 10)
        u0 = np.zeros(n_points)
        for k in range(1, n_modes + 1):
            amp = np.random.normal(0, 1.0 / k)
            phase = np.random.uniform(0, 2 * np.pi)
            u0 += amp * np.sin(k * x + phase)

        # Normalize
        u0 = u0 / (np.max(np.abs(u0)) + 1e-10)
        inputs[i, 0] = u0

        if operator_type == "heat":
            # Heat equation: du/dt = nu * d2u/dx2
            # Analytical solution via Fourier coefficients
            nu = np.random.uniform(0.01, 0.1)
            t_final = np.random.uniform(0.1, 1.0)
            u0_fft = np.fft.rfft(u0)
            freqs = np.fft.rfftfreq(n_points, d=dx) * 2 * np.pi
            decay = np.exp(-nu * freqs ** 2 * t_final)
            u_final = np.fft.irfft(u0_fft * decay, n=n_points)
            outputs[i, 0] = u_final

        elif operator_type == "advection":
            # Advection: du/dt + c * du/dx = 0
            # Solution: u(x, t) = u0(x - c*t)
            c = np.random.uniform(0.5, 2.0)
            t_final = np.random.uniform(0.1, 1.0)
            shift = c * t_final
            shift_idx = int(shift / dx) % n_points
            outputs[i, 0] = np.roll(u0, -shift_idx)

        elif operator_type == "burgers":
            # Simplified viscous Burgers via Cole-Hopf (approximate)
            nu = np.random.uniform(0.05, 0.2)
            t_final = 0.5
            # Simple explicit Euler (for data generation)
            u = u0.copy()
            dt = 0.001
            n_steps = int(t_final / dt)
            for _ in range(n_steps):
                du_dx = np.gradient(u, dx)
                d2u_dx2 = np.gradient(du_dx, dx)
                u = u + dt * (-u * du_dx + nu * d2u_dx2)
            outputs[i, 0] = u

    return inputs, outputs


# ═══════════════════════════════════════════════════════════════════════════════
# Train/Val/Test Split
# ═══════════════════════════════════════════════════════════════════════════════

def train_val_test_split(
    *arrays: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> List[Tuple[np.ndarray, ...]]:
    """
    Time-series aware split (no shuffling, preserves temporal order).

    Args:
        *arrays: Arrays to split (all must have same first dimension)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        (test gets the remainder)

    Returns:
        List of 3 tuples: (train_arrays, val_arrays, test_arrays)
    """
    n = arrays[0].shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = []
    for split_start, split_end in [(0, train_end), (train_end, val_end), (val_end, n)]:
        split_arrays = tuple(arr[split_start:split_end] for arr in arrays)
        splits.append(split_arrays)

    return splits


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point for quick data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data(config: Optional[DataConfig] = None) -> Dict:
    """
    Load all data sources for operator learning experiments.

    Returns:
        Dict with keys:
            'stock': stock price DataFrames
            'crypto': crypto OHLCV DataFrame
            'yield_proxy': yield curve proxy DataFrame
            'bs_data': Black-Scholes synthetic data
            'synthetic_heat': synthetic heat equation data
            'deeponet_stock': prepared DeepONet data from stocks
            'deeponet_crypto': prepared DeepONet data from crypto
    """
    if config is None:
        config = DataConfig()

    result = {}

    # Synthetic data (always available, no network needed)
    from .model import generate_black_scholes_data, generate_yield_curve_data

    print("Generating synthetic Black-Scholes data...")
    vol_funcs, eval_pts, prices = generate_black_scholes_data(
        n_samples=config.n_synthetic_samples,
        n_vol_points=config.n_sensor_points,
        n_eval_points=config.n_eval_points,
    )
    result["bs_data"] = {
        "vol_functions": vol_funcs,
        "eval_points": eval_pts,
        "prices": prices,
    }

    print("Generating synthetic yield curve data...")
    curves_today, curves_future = generate_yield_curve_data(
        n_samples=config.n_synthetic_samples
    )
    result["yield_curve_data"] = {
        "today": curves_today,
        "future": curves_future,
    }

    print("Generating synthetic PDE data (heat equation)...")
    heat_in, heat_out = generate_synthetic_operator_data(
        "heat", n_samples=config.n_synthetic_samples
    )
    result["synthetic_heat"] = {"inputs": heat_in, "outputs": heat_out}

    # Try fetching real market data
    try:
        print("Fetching stock data...")
        stock_data = fetch_stock_data(config.stock_symbols, config.stock_period)
        result["stock"] = stock_data

        # Prepare DeepONet data from first stock
        first_sym = next(iter(stock_data))
        branch, trunk, targets = prepare_deeponet_data_from_prices(
            stock_data[first_sym]["Close"],
            config.sequence_length,
            config.prediction_horizon,
        )
        result["deeponet_stock"] = {
            "branch_inputs": branch,
            "trunk_inputs": trunk,
            "targets": targets,
        }
    except Exception as e:
        print(f"Warning: Stock data fetch failed: {e}")

    try:
        print("Fetching yield curve proxy data...")
        yield_proxy = fetch_yield_curve_proxy(config.yield_etfs)
        result["yield_proxy"] = yield_proxy
    except Exception as e:
        print(f"Warning: Yield proxy fetch failed: {e}")

    try:
        print(f"Fetching crypto data ({config.crypto_symbol})...")
        crypto_data = prepare_crypto_price_operator_data(
            config.crypto_symbol,
            config.crypto_interval,
            config.crypto_limit,
            config.sequence_length,
            config.prediction_horizon,
        )
        result["deeponet_crypto"] = crypto_data
    except Exception as e:
        print(f"Warning: Crypto data fetch failed: {e}")

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Operator Learning Data Loader - Chapter 152")
    print("=" * 70)

    config = DataConfig(n_synthetic_samples=500)
    data = load_all_data(config)

    print("\nLoaded data summary:")
    for key, val in data.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                if isinstance(subval, np.ndarray):
                    print(f"  {key}/{subkey}: shape={subval.shape}, dtype={subval.dtype}")
                elif isinstance(subval, pd.DataFrame):
                    print(f"  {key}/{subkey}: DataFrame {subval.shape}")
        elif isinstance(val, pd.DataFrame):
            print(f"  {key}: DataFrame {val.shape}")

    print("\nDone.")
