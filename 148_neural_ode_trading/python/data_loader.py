"""
Data Loader Module for Neural ODE Trading

Provides data loading utilities for:
1. Bybit cryptocurrency exchange (REST API v5)
2. Synthetic stock data for testing
3. Irregularly-sampled time series dataset construction

Focuses on preserving irregular timestamps -- critical for Neural ODE models
that naturally handle non-uniform time spacing.

All data fetching is contained within this chapter's folder.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset


class BybitDataLoader:
    """
    Load cryptocurrency data from Bybit exchange.

    Supports kline data and recent trades (tick data with irregular timestamps).

    Example:
        >>> loader = BybitDataLoader()
        >>> df = loader.fetch_klines("BTCUSDT", interval="1", limit=1000)
        >>> ticks = loader.fetch_recent_trades("BTCUSDT", limit=1000)
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1",
        limit: int = 1000,
        category: str = "linear",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Kline interval in minutes ('1','3','5','15','30','60','120','240','360','720','D','W','M')
            limit: Number of klines (max 1000)
            category: Market category ('linear', 'inverse', 'spot')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with: timestamp, open, high, low, close, volume, turnover
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                print(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                return self._generate_synthetic_klines(symbol, limit)

            rows = data.get("result", {}).get("list", [])
            if not rows:
                print("No data returned, using synthetic data.")
                return self._generate_synthetic_klines(symbol, limit)

            df = pd.DataFrame(
                rows,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume", "turnover"
                ],
            )

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)

            df = df.sort_values("timestamp").reset_index(drop=True)
            return df

        except Exception as e:
            print(f"Error fetching Bybit data: {e}")
            print("Falling back to synthetic data.")
            return self._generate_synthetic_klines(symbol, limit)

    def fetch_recent_trades(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 1000,
        category: str = "linear",
    ) -> pd.DataFrame:
        """
        Fetch recent trades with IRREGULAR timestamps (tick data).

        This is the ideal data for Neural ODEs: each trade occurs at an
        irregular time point, and the model must handle the variable spacing.

        Args:
            symbol: Trading pair
            limit: Number of recent trades (max 1000)
            category: Market category

        Returns:
            DataFrame with: timestamp, price, size, side, is_block_trade
        """
        endpoint = f"{self.BASE_URL}/v5/market/recent-trade"

        params = {
            "category": category,
            "symbol": symbol,
            "limit": min(limit, 1000),
        }

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                print(f"Bybit API error: {data.get('retMsg')}")
                return self._generate_synthetic_ticks(symbol, limit)

            trades = data.get("result", {}).get("list", [])
            if not trades:
                return self._generate_synthetic_ticks(symbol, limit)

            df = pd.DataFrame(trades)

            # Parse fields
            df["timestamp"] = pd.to_datetime(df["time"].astype(int), unit="ms")
            df["price"] = df["price"].astype(float)
            df["size"] = df["size"].astype(float)
            df["side"] = df["side"].str.lower()

            # Compute irregular time gaps
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["time_delta_ms"] = df["timestamp"].diff().dt.total_seconds() * 1000
            df["time_delta_ms"] = df["time_delta_ms"].fillna(0)

            return df[["timestamp", "price", "size", "side", "time_delta_ms"]]

        except Exception as e:
            print(f"Error fetching trades: {e}")
            return self._generate_synthetic_ticks(symbol, limit)

    def fetch_multi_timeframe(
        self,
        symbol: str = "BTCUSDT",
        intervals: List[str] = None,
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data at multiple timeframes for multi-scale analysis.

        Args:
            symbol: Trading pair
            intervals: List of intervals (default: ['1', '5', '60'])
            limit: Number of candles per interval

        Returns:
            Dict mapping interval -> DataFrame
        """
        if intervals is None:
            intervals = ["1", "5", "60"]

        result = {}
        for interval in intervals:
            df = self.fetch_klines(symbol, interval=interval, limit=limit)
            result[interval] = df
            time.sleep(0.1)  # Rate limiting

        return result

    def _generate_synthetic_klines(
        self, symbol: str, n: int
    ) -> pd.DataFrame:
        """Generate synthetic kline data mimicking real market behavior."""
        np.random.seed(42)

        # Base price depends on symbol
        base_prices = {
            "BTCUSDT": 65000.0,
            "ETHUSDT": 3500.0,
            "SOLUSDT": 150.0,
        }
        base_price = base_prices.get(symbol, 100.0)

        # Generate prices with geometric Brownian motion
        dt = 1.0 / (252 * 24 * 60)  # 1-minute intervals
        mu = 0.05  # Annual drift
        sigma = 0.6  # Annual volatility

        log_returns = np.random.normal(
            (mu - 0.5 * sigma ** 2) * dt,
            sigma * np.sqrt(dt),
            size=n,
        )
        prices = base_price * np.exp(np.cumsum(log_returns))

        # Generate OHLCV
        timestamps = pd.date_range(
            end=datetime.now(), periods=n, freq="1min"
        )

        noise = np.random.uniform(0.0005, 0.003, size=n)
        opens = prices * (1 + np.random.uniform(-0.001, 0.001, n))
        highs = prices * (1 + noise)
        lows = prices * (1 - noise)
        closes = prices
        volumes = np.random.lognormal(mean=10, sigma=1.5, size=n)
        turnovers = closes * volumes

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "turnover": turnovers,
        })

    def _generate_synthetic_ticks(
        self, symbol: str, n: int
    ) -> pd.DataFrame:
        """Generate synthetic tick data with irregular timestamps."""
        np.random.seed(42)

        base_prices = {
            "BTCUSDT": 65000.0,
            "ETHUSDT": 3500.0,
            "SOLUSDT": 150.0,
        }
        base_price = base_prices.get(symbol, 100.0)

        # Irregular time gaps (exponentially distributed -- Poisson process)
        # Mean time between ticks: 200ms (high-frequency)
        time_gaps_ms = np.random.exponential(scale=200, size=n).astype(int)
        time_gaps_ms = np.clip(time_gaps_ms, 1, 5000)  # 1ms to 5s

        cumulative_ms = np.cumsum(time_gaps_ms)
        base_time = datetime.now() - timedelta(milliseconds=int(cumulative_ms[-1]))
        timestamps = [base_time + timedelta(milliseconds=int(t)) for t in cumulative_ms]

        # Generate prices (microstructure noise + trend)
        returns = np.random.normal(0, 0.0001, size=n)
        # Add occasional jumps (news events)
        jump_mask = np.random.random(n) < 0.01
        returns[jump_mask] += np.random.choice([-1, 1], size=jump_mask.sum()) * 0.005
        prices = base_price * np.exp(np.cumsum(returns))

        # Trade sizes (heavy-tailed)
        sizes = np.random.lognormal(mean=-2, sigma=1.5, size=n)
        sides = np.random.choice(["buy", "sell"], size=n, p=[0.52, 0.48])

        return pd.DataFrame({
            "timestamp": timestamps,
            "price": prices,
            "size": sizes,
            "side": sides,
            "time_delta_ms": time_gaps_ms.astype(float),
        })


class StockDataLoader:
    """
    Load stock market data for Neural ODE experiments.

    Generates realistic synthetic stock data or loads from CSV files.
    Includes support for creating irregularly-sampled versions
    (simulating missing data, trading halts, etc.).
    """

    @staticmethod
    def generate_stock_data(
        n_days: int = 500,
        n_stocks: int = 1,
        freq: str = "1h",
        add_missing: bool = True,
        missing_rate: float = 0.1,
    ) -> pd.DataFrame:
        """
        Generate synthetic stock data with optional missing observations.

        Args:
            n_days: Number of trading days
            n_stocks: Number of stocks
            freq: Data frequency ('1min', '5min', '1h', '1d')
            add_missing: Add random missing values
            missing_rate: Fraction of observations to remove

        Returns:
            DataFrame with OHLCV + irregular timestamps
        """
        np.random.seed(42)

        # Determine number of observations per day
        freq_map = {"1min": 390, "5min": 78, "15min": 26, "1h": 7, "1d": 1}
        obs_per_day = freq_map.get(freq, 7)
        n_total = n_days * obs_per_day

        # Generate timestamps (regular grid first)
        timestamps = pd.date_range(
            start="2023-01-01",
            periods=n_total,
            freq=freq,
        )

        # Filter to trading hours (9:30 AM - 4:00 PM) for realism
        if freq != "1d":
            timestamps = timestamps[
                (timestamps.hour >= 9) & (timestamps.hour < 16)
            ][:n_total]

        # Multi-factor model for realistic returns
        dt = 1.0 / (252 * obs_per_day)

        # Market factor
        market_vol = 0.20
        market_returns = np.random.normal(0, market_vol * np.sqrt(dt), len(timestamps))

        all_data = []
        for stock_idx in range(n_stocks):
            # Stock-specific parameters
            beta = np.random.uniform(0.5, 1.5)
            idio_vol = np.random.uniform(0.15, 0.40)
            mu = np.random.uniform(0.02, 0.12)

            returns = (
                mu * dt
                + beta * market_returns
                + np.random.normal(0, idio_vol * np.sqrt(dt), len(timestamps))
            )

            base_price = np.random.uniform(50, 500)
            prices = base_price * np.exp(np.cumsum(returns))

            # Generate OHLCV
            noise = np.random.uniform(0.001, 0.005, len(timestamps))
            volumes = np.random.lognormal(mean=12, sigma=1, size=len(timestamps))

            stock_df = pd.DataFrame({
                "timestamp": timestamps[:len(prices)],
                "stock_id": stock_idx,
                "open": prices * (1 + np.random.uniform(-0.002, 0.002, len(prices))),
                "high": prices * (1 + noise),
                "low": prices * (1 - noise),
                "close": prices,
                "volume": volumes,
            })

            # Add technical indicators
            stock_df["returns"] = stock_df["close"].pct_change().fillna(0)
            stock_df["log_volume"] = np.log1p(stock_df["volume"])
            stock_df["volatility"] = stock_df["returns"].rolling(20).std().fillna(0)

            all_data.append(stock_df)

        df = pd.concat(all_data, ignore_index=True)

        # Remove random observations to create irregular sampling
        if add_missing:
            n_remove = int(len(df) * missing_rate)
            remove_idx = np.random.choice(df.index[1:-1], n_remove, replace=False)
            df = df.drop(remove_idx).reset_index(drop=True)

        # Compute actual time deltas (now irregular due to missing data)
        df["time_delta"] = (
            df["timestamp"]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )

        return df


class IrregularTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for irregularly-sampled time series.

    Produces sequences with:
    - Observations at irregular time points
    - Time stamps (relative or absolute)
    - Observation masks (for missing data)

    This dataset format is designed specifically for Neural ODE models
    (LatentODE, ODE-RNN) that handle irregular spacing natively.

    Args:
        data: DataFrame with 'timestamp' column and feature columns
        feature_cols: List of feature column names
        target_col: Target column name for prediction
        seq_len: Number of observations per sequence
        pred_horizon: Number of steps ahead to predict
        normalize: Whether to normalize features
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str] = None,
        target_col: str = "close",
        seq_len: int = 60,
        pred_horizon: int = 1,
        normalize: bool = True,
    ):
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        if feature_cols is None:
            feature_cols = ["open", "high", "low", "close", "volume"]

        # Extract features and target
        self.features = data[feature_cols].values.astype(np.float32)
        self.targets = data[target_col].values.astype(np.float32)

        # Compute timestamps as seconds from start
        if "timestamp" in data.columns:
            ts = pd.to_datetime(data["timestamp"])
            self.times = (
                (ts - ts.iloc[0]).dt.total_seconds().values.astype(np.float32)
            )
        elif "time_delta" in data.columns:
            self.times = np.cumsum(
                data["time_delta"].values.astype(np.float32)
            )
        else:
            self.times = np.arange(len(data), dtype=np.float32)

        # Normalize features
        if normalize:
            self.feat_mean = np.nanmean(self.features, axis=0)
            self.feat_std = np.nanstd(self.features, axis=0) + 1e-8
            self.features = (self.features - self.feat_mean) / self.feat_std

            self.target_mean = np.nanmean(self.targets)
            self.target_std = np.nanstd(self.targets) + 1e-8
        else:
            self.feat_mean = np.zeros(len(feature_cols))
            self.feat_std = np.ones(len(feature_cols))
            self.target_mean = 0.0
            self.target_std = 1.0

        # Create observation mask (1 = valid, 0 = missing/NaN)
        self.mask = (~np.isnan(self.features)).astype(np.float32)
        self.features = np.nan_to_num(self.features, 0.0)

        self.n_samples = len(data) - seq_len - pred_horizon

    def __len__(self) -> int:
        return max(0, self.n_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                'x': Features, shape (seq_len, n_features)
                't': Timestamps, shape (seq_len,)
                'mask': Observation mask, shape (seq_len, n_features)
                'target': Future target value, shape (1,)
                'target_time': Future timestamp
        """
        # Input window
        x = self.features[idx : idx + self.seq_len]
        t = self.times[idx : idx + self.seq_len]
        mask = self.mask[idx : idx + self.seq_len]

        # Normalize time to [0, 1] within this window
        t_min = t[0]
        t_max = t[-1]
        t_range = t_max - t_min
        if t_range > 0:
            t_normalized = (t - t_min) / t_range
        else:
            t_normalized = t - t_min

        # Target (future value)
        target_idx = idx + self.seq_len + self.pred_horizon - 1
        target = self.targets[target_idx]
        target_normalized = (target - self.target_mean) / self.target_std

        target_time = self.times[target_idx]
        if t_range > 0:
            target_time_normalized = (target_time - t_min) / t_range
        else:
            target_time_normalized = target_time - t_min

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "t": torch.tensor(t_normalized, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "target": torch.tensor([target_normalized], dtype=torch.float32),
            "target_time": torch.tensor(target_time_normalized, dtype=torch.float32),
        }


def create_irregular_dataset(
    symbol: str = "BTCUSDT",
    source: str = "bybit",
    seq_len: int = 60,
    pred_horizon: int = 1,
) -> Tuple[IrregularTimeSeriesDataset, IrregularTimeSeriesDataset]:
    """
    Create train/test datasets with irregular timestamps.

    Args:
        symbol: Trading pair or stock symbol
        source: Data source ('bybit', 'stock', 'synthetic')
        seq_len: Sequence length
        pred_horizon: Prediction horizon

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if source == "bybit":
        loader = BybitDataLoader()
        df = loader.fetch_klines(symbol, interval="1", limit=1000)
    elif source == "stock":
        df = StockDataLoader.generate_stock_data(n_days=200, freq="1h")
    else:
        loader = BybitDataLoader()
        df = loader._generate_synthetic_klines(symbol, 1000)

    # Split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    feature_cols = ["open", "high", "low", "close", "volume"]

    train_ds = IrregularTimeSeriesDataset(
        train_df, feature_cols=feature_cols, seq_len=seq_len, pred_horizon=pred_horizon
    )
    test_ds = IrregularTimeSeriesDataset(
        test_df, feature_cols=feature_cols, seq_len=seq_len, pred_horizon=pred_horizon
    )

    return train_ds, test_ds
