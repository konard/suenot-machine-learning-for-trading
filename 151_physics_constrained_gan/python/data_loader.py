"""
Data Loader Module for Physics-Constrained GAN

Provides data fetching and preprocessing for both cryptocurrency (Bybit)
and traditional stock markets.

Features:
    - Bybit REST API integration for BTC/ETH OHLCV data
    - Rolling window segmentation for time series
    - Log-return computation and normalization
    - Target statistics computation for physics constraints
    - Regime detection (simple volatility-based)
    - DataLoader creation for PyTorch training
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class DataConfig:
    """Configuration for data loading."""

    # Data source
    symbol: str = "BTCUSDT"
    interval: str = "60"  # Bybit: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
    source: str = "bybit"  # "bybit" or "synthetic"

    # Preprocessing
    seq_len: int = 256
    stride: int = 64
    normalize: bool = True

    # Regime detection
    vol_window: int = 20
    n_regimes: int = 4
    n_vol_levels: int = 4


class BybitDataLoader:
    """
    Fetches OHLCV data from Bybit REST API and preprocesses for GAN training.

    Bybit V5 API endpoint: https://api.bybit.com/v5/market/kline
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",
        category: str = "spot",
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required. Install with: pip install requests")

        self.symbol = symbol
        self.interval = interval
        self.category = category

    def fetch_klines(
        self,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch kline (candlestick) data from Bybit.

        Args:
            limit: Number of candles to fetch (max 1000 per request)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        url = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": self.category,
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                print(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                return pd.DataFrame()

            rows = data.get("result", {}).get("list", [])
            if not rows:
                print("No data returned from Bybit API")
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)

            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

        except requests.RequestException as e:
            print(f"Error fetching data from Bybit: {e}")
            return pd.DataFrame()

    def fetch_extended(self, n_candles: int = 5000) -> pd.DataFrame:
        """
        Fetch more data than the 1000 limit by paginating.

        Args:
            n_candles: Total number of candles desired

        Returns:
            DataFrame with all requested candles
        """
        all_data = []
        end_time = None
        remaining = n_candles

        while remaining > 0:
            batch_size = min(remaining, 1000)
            df = self.fetch_klines(limit=batch_size, end_time=end_time)

            if df.empty:
                break

            all_data.append(df)
            remaining -= len(df)

            if len(df) < batch_size:
                break

            # Move end_time to earliest timestamp in batch
            end_time = int(df["timestamp"].iloc[0].timestamp() * 1000) - 1
            time.sleep(0.1)  # Rate limiting

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"])
        result = result.sort_values("timestamp").reset_index(drop=True)
        return result

    def fetch_and_preprocess(
        self,
        n_candles: int = 5000,
        seq_len: int = 256,
        stride: int = 64,
    ) -> Dict:
        """
        Fetch data and preprocess for GAN training.

        Returns:
            Dict with keys: returns, windows, regimes, vol_levels, stats
        """
        df = self.fetch_extended(n_candles)
        if df.empty:
            print("No data fetched. Using synthetic data instead.")
            return generate_synthetic_data(seq_len=seq_len, n_samples=200)

        return preprocess_ohlcv(df, seq_len=seq_len, stride=stride)


def generate_synthetic_data(
    seq_len: int = 256,
    n_samples: int = 500,
    dt: float = 1.0 / 252,
    mu: float = 0.05,
    initial_vol: float = 0.2,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
) -> Dict:
    """
    Generate synthetic financial data using a Heston stochastic volatility model.

    This produces returns with realistic stylized facts:
        - Fat tails, volatility clustering, leverage effect.

    Args:
        seq_len: Length of each return sequence
        n_samples: Number of sequences to generate
        dt: Time step (fraction of year)
        mu: Drift
        initial_vol: Initial volatility level
        kappa: Mean reversion speed for variance
        theta: Long-run variance
        xi: Vol of vol
        rho: Correlation between price and vol (leverage effect)

    Returns:
        Dict with windows, regimes, vol_levels, stats
    """
    np.random.seed(42)

    windows = []
    regimes = []
    vol_levels = []

    for _ in range(n_samples):
        # Heston model simulation
        v = initial_vol ** 2  # variance
        returns = np.zeros(seq_len)

        for t in range(seq_len):
            # Correlated Brownian motions
            z1 = np.random.normal()
            z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal()

            v_pos = max(v, 1e-8)
            sigma = np.sqrt(v_pos)

            # Log return
            returns[t] = (mu - 0.5 * v_pos) * dt + sigma * np.sqrt(dt) * z1

            # Variance process (CIR)
            v = v + kappa * (theta - v) * dt + xi * np.sqrt(v_pos * dt) * z2
            v = max(v, 0.0)

        windows.append(returns)

        # Simple regime assignment based on cumulative return
        cum_ret = np.sum(returns)
        avg_vol = np.std(returns)

        if cum_ret > 0.02:
            regime = 0  # bull
        elif cum_ret < -0.02:
            regime = 1  # bear
        elif avg_vol > 0.03:
            regime = 3  # crisis
        else:
            regime = 2  # sideways

        regimes.append(regime)

        # Volatility level
        if avg_vol < 0.01:
            vol_levels.append(0)
        elif avg_vol < 0.02:
            vol_levels.append(1)
        elif avg_vol < 0.04:
            vol_levels.append(2)
        else:
            vol_levels.append(3)

    windows = np.array(windows)
    regimes = np.array(regimes)
    vol_levels = np.array(vol_levels)

    # Compute statistics
    flat = windows.flatten()
    stats = {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "kurtosis": float(_excess_kurtosis(flat)),
        "skewness": float(_skewness_np(flat)),
    }

    return {
        "windows": windows,
        "regimes": regimes,
        "vol_levels": vol_levels,
        "stats": stats,
    }


def preprocess_ohlcv(
    df: pd.DataFrame,
    seq_len: int = 256,
    stride: int = 64,
) -> Dict:
    """
    Preprocess OHLCV DataFrame into windowed log-returns with regime labels.

    Args:
        df: DataFrame with 'close' column
        seq_len: Window length
        stride: Stride between windows

    Returns:
        Dict with windows, regimes, vol_levels, stats
    """
    prices = df["close"].values
    log_returns = np.diff(np.log(prices))

    # Rolling window segmentation
    windows = []
    n = len(log_returns)
    for start in range(0, n - seq_len + 1, stride):
        window = log_returns[start:start + seq_len]
        windows.append(window)

    if not windows:
        print("Not enough data for windowing. Using synthetic data.")
        return generate_synthetic_data(seq_len=seq_len)

    windows = np.array(windows)

    # Detect regimes
    regimes = []
    vol_levels = []
    for w in windows:
        cum_ret = np.sum(w)
        vol = np.std(w)

        # Regime
        if cum_ret > np.percentile([np.sum(ww) for ww in windows], 75):
            regime = 0  # bull
        elif cum_ret < np.percentile([np.sum(ww) for ww in windows], 25):
            regime = 1  # bear
        elif vol > np.percentile([np.std(ww) for ww in windows], 75):
            regime = 3  # crisis
        else:
            regime = 2  # sideways
        regimes.append(regime)

        # Volatility level
        vol_quartiles = np.percentile([np.std(ww) for ww in windows], [25, 50, 75])
        if vol < vol_quartiles[0]:
            vol_levels.append(0)
        elif vol < vol_quartiles[1]:
            vol_levels.append(1)
        elif vol < vol_quartiles[2]:
            vol_levels.append(2)
        else:
            vol_levels.append(3)

    regimes = np.array(regimes)
    vol_levels = np.array(vol_levels)

    # Compute statistics
    flat = windows.flatten()
    stats = {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "kurtosis": float(_excess_kurtosis(flat)),
        "skewness": float(_skewness_np(flat)),
    }

    return {
        "windows": windows,
        "regimes": regimes,
        "vol_levels": vol_levels,
        "stats": stats,
    }


class ReturnWindowDataset(Dataset):
    """PyTorch Dataset for windowed return sequences."""

    def __init__(self, data_dict: Dict):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.windows = torch.tensor(data_dict["windows"], dtype=torch.float32)
        self.regimes = torch.tensor(data_dict["regimes"], dtype=torch.long)
        self.vol_levels = torch.tensor(data_dict["vol_levels"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.regimes[idx], self.vol_levels[idx]


def create_dataloader(
    data_dict: Dict,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> "DataLoader":
    """
    Create a PyTorch DataLoader from preprocessed data.

    Args:
        data_dict: Output from preprocess_ohlcv or generate_synthetic_data
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers

    Returns:
        PyTorch DataLoader
    """
    dataset = ReturnWindowDataset(data_dict)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


def _excess_kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis of an array."""
    n = len(x)
    if n < 4:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 4) - 3.0)


def _skewness_np(x: np.ndarray) -> float:
    """Compute skewness of an array."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def fetch_multiple_symbols(
    symbols: List[str],
    interval: str = "60",
    n_candles: int = 5000,
    seq_len: int = 256,
    stride: int = 64,
) -> Dict[str, Dict]:
    """
    Fetch and preprocess data for multiple symbols.

    Args:
        symbols: List of Bybit symbol strings
        interval: Candlestick interval
        n_candles: Number of candles per symbol
        seq_len: Window length
        stride: Window stride

    Returns:
        Dict mapping symbol to preprocessed data dict
    """
    results = {}
    for symbol in symbols:
        print(f"Fetching {symbol}...")
        loader = BybitDataLoader(symbol=symbol, interval=interval)
        data = loader.fetch_and_preprocess(
            n_candles=n_candles, seq_len=seq_len, stride=stride
        )
        results[symbol] = data
        time.sleep(0.5)  # Rate limiting between symbols
    return results


if __name__ == "__main__":
    # Demo: generate synthetic data and print statistics
    print("Generating synthetic Heston data...")
    data = generate_synthetic_data(seq_len=256, n_samples=200)

    print(f"Number of windows: {len(data['windows'])}")
    print(f"Window shape: {data['windows'].shape}")
    print(f"Statistics: {data['stats']}")
    print(f"Regime distribution: {np.bincount(data['regimes'])}")
    print(f"Vol level distribution: {np.bincount(data['vol_levels'])}")

    # Try Bybit fetch
    print("\nAttempting Bybit data fetch...")
    try:
        loader = BybitDataLoader(symbol="BTCUSDT", interval="60")
        df = loader.fetch_klines(limit=100)
        if not df.empty:
            print(f"Fetched {len(df)} candles from Bybit")
            print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
            print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
        else:
            print("No data returned (check network connection)")
    except Exception as e:
        print(f"Bybit fetch failed: {e}")
