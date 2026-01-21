"""
Data loading and feature engineering for MAML trading.

This module provides:
- BybitClient for fetching cryptocurrency data
- FeatureGenerator for computing technical indicators
- SimulatedDataGenerator for testing
"""

import numpy as np
import pandas as pd
import requests
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Kline:
    """Single candlestick data point."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'turnover': self.turnover
        }


class BybitClient:
    """Client for fetching data from Bybit exchange API."""

    def __init__(self, base_url: str = "https://api.bybit.com"):
        self.base_url = base_url
        self.session = requests.Session()

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 200
    ) -> List[Kline]:
        """
        Fetch historical klines from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval ("1", "5", "15", "60", "D")
            limit: Number of klines (max 1000)

        Returns:
            List of Kline objects sorted by timestamp ascending
        """
        url = f"{self.base_url}/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                raise ValueError(f"API error: {data.get('retMsg')}")

            klines = []
            for item in data["result"]["list"]:
                klines.append(Kline(
                    timestamp=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    turnover=float(item[6])
                ))

            # Bybit returns descending order, reverse to ascending
            klines.reverse()
            return klines

        except requests.RequestException as e:
            logger.error(f"Failed to fetch klines: {e}")
            raise

    def fetch_multi_symbol(
        self,
        symbols: List[str],
        interval: str = "60",
        limit: int = 200
    ) -> dict:
        """Fetch klines for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_klines(symbol, interval, limit)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        return results


class SimulatedDataGenerator:
    """Generate simulated market data for testing."""

    @staticmethod
    def generate_klines(
        num_klines: int,
        base_price: float = 50000.0,
        volatility: float = 0.02
    ) -> List[Kline]:
        """Generate random walk klines."""
        klines = []
        price = base_price
        base_timestamp = int((datetime.now() - timedelta(hours=num_klines)).timestamp() * 1000)

        for i in range(num_klines):
            return_pct = np.random.normal(0, volatility)
            open_price = price
            close_price = price * (1 + return_pct)
            high_price = max(open_price, close_price) * (1 + np.random.random() * 0.01)
            low_price = min(open_price, close_price) * (1 - np.random.random() * 0.01)
            volume = np.random.random() * 1000000

            klines.append(Kline(
                timestamp=base_timestamp + i * 3600000,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                turnover=volume * close_price
            ))
            price = close_price

        return klines

    @staticmethod
    def generate_trending_klines(
        num_klines: int,
        base_price: float = 50000.0,
        volatility: float = 0.02,
        trend: float = 0.0002
    ) -> List[Kline]:
        """Generate klines with a trend component."""
        klines = []
        price = base_price
        base_timestamp = int((datetime.now() - timedelta(hours=num_klines)).timestamp() * 1000)

        for i in range(num_klines):
            return_pct = np.random.normal(0, volatility) + trend
            open_price = price
            close_price = price * (1 + return_pct)
            high_price = max(open_price, close_price) * (1 + np.random.random() * 0.01)
            low_price = min(open_price, close_price) * (1 - np.random.random() * 0.01)
            volume = np.random.random() * 1000000

            klines.append(Kline(
                timestamp=base_timestamp + i * 3600000,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                turnover=volume * close_price
            ))
            price = close_price

        return klines

    @staticmethod
    def generate_regime_changing_klines(
        num_klines: int,
        base_price: float = 50000.0
    ) -> List[Kline]:
        """Generate klines with changing market regimes."""
        regimes = [
            (0.015, 0.0002, 0.2),   # Bull
            (0.025, -0.0003, 0.15), # Correction
            (0.01, 0.0, 0.15),      # Consolidation
            (0.02, 0.00015, 0.2),   # Recovery
            (0.03, -0.0001, 0.15),  # Volatility spike
            (0.012, 0.0001, 0.15),  # Calm growth
        ]

        klines = []
        price = base_price
        base_timestamp = int((datetime.now() - timedelta(hours=num_klines)).timestamp() * 1000)
        current_idx = 0

        for vol, trend, duration_frac in regimes:
            regime_len = int(num_klines * duration_frac)
            for _ in range(regime_len):
                if current_idx >= num_klines:
                    break

                return_pct = np.random.normal(0, vol) + trend
                open_price = price
                close_price = price * (1 + return_pct)
                high_price = max(open_price, close_price) * (1 + np.random.random() * 0.01)
                low_price = min(open_price, close_price) * (1 - np.random.random() * 0.01)
                volume = np.random.random() * 1000000 * (1 + vol * 10)

                klines.append(Kline(
                    timestamp=base_timestamp + current_idx * 3600000,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    turnover=volume * close_price
                ))
                price = close_price
                current_idx += 1

            if current_idx >= num_klines:
                break

        # Fill remaining with neutral regime
        while len(klines) < num_klines:
            return_pct = np.random.normal(0, 0.015)
            open_price = price
            close_price = price * (1 + return_pct)
            high_price = max(open_price, close_price) * (1 + np.random.random() * 0.01)
            low_price = min(open_price, close_price) * (1 - np.random.random() * 0.01)
            volume = np.random.random() * 1000000

            klines.append(Kline(
                timestamp=base_timestamp + len(klines) * 3600000,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                turnover=volume * close_price
            ))
            price = close_price

        return klines


class FeatureGenerator:
    """Generate technical features from price data."""

    def __init__(self, window: int = 20):
        self.window = window

    def compute_features(self, klines: List[Kline]) -> np.ndarray:
        """
        Compute all features from kline data.

        Returns array of shape (N, 11) where N is number of valid data points.
        Features: returns_1d, returns_5d, returns_10d, sma_ratio, ema_ratio,
                  volatility, momentum, rsi, macd, bb_position, volume_sma_ratio
        """
        if len(klines) < self.window + 10:
            return np.array([])

        closes = np.array([k.close for k in klines])
        volumes = np.array([k.volume for k in klines])

        # Compute individual features
        returns_1 = self._compute_returns(closes, 1)
        returns_5 = self._compute_returns(closes, 5)
        returns_10 = self._compute_returns(closes, 10)
        sma_ratio = self._compute_sma_ratio(closes)
        ema_ratio = self._compute_ema_ratio(closes)
        volatility = self._compute_volatility(closes)
        momentum = self._compute_momentum(closes)
        rsi = self._compute_rsi(closes)
        macd = self._compute_macd(closes)
        bb_position = self._compute_bollinger_position(closes)
        volume_sma_ratio = self._compute_volume_sma_ratio(volumes)

        # Find common length
        min_len = min(
            len(returns_1), len(returns_5), len(returns_10),
            len(sma_ratio), len(ema_ratio), len(volatility),
            len(momentum), len(rsi), len(macd), len(bb_position),
            len(volume_sma_ratio)
        )

        if min_len == 0:
            return np.array([])

        # Stack features
        features = np.column_stack([
            returns_1[-min_len:],
            returns_5[-min_len:],
            returns_10[-min_len:],
            sma_ratio[-min_len:],
            ema_ratio[-min_len:],
            volatility[-min_len:],
            momentum[-min_len:],
            rsi[-min_len:],
            macd[-min_len:],
            bb_position[-min_len:],
            volume_sma_ratio[-min_len:]
        ])

        return features

    def _compute_returns(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Compute percentage returns over given period."""
        if len(closes) <= period:
            return np.array([])
        returns = closes[period:] / closes[:-period] - 1
        return returns

    def _compute_sma_ratio(self, closes: np.ndarray) -> np.ndarray:
        """Compute ratio of price to SMA."""
        if len(closes) < self.window:
            return np.array([])
        sma = np.convolve(closes, np.ones(self.window)/self.window, mode='valid')
        ratio = closes[self.window-1:] / sma - 1
        return ratio

    def _compute_ema_ratio(self, closes: np.ndarray) -> np.ndarray:
        """Compute ratio of price to EMA."""
        if len(closes) < self.window:
            return np.array([])

        alpha = 2 / (self.window + 1)
        ema = np.zeros(len(closes))
        ema[0] = closes[0]

        for i in range(1, len(closes)):
            ema[i] = alpha * closes[i] + (1 - alpha) * ema[i-1]

        ratio = closes / ema - 1
        return ratio[self.window-1:]

    def _compute_volatility(self, closes: np.ndarray) -> np.ndarray:
        """Compute rolling volatility."""
        if len(closes) < self.window + 1:
            return np.array([])

        returns = np.diff(np.log(closes))
        volatility = np.array([
            np.std(returns[max(0, i-self.window+1):i+1])
            for i in range(self.window-1, len(returns))
        ])
        return volatility

    def _compute_momentum(self, closes: np.ndarray) -> np.ndarray:
        """Compute price momentum."""
        if len(closes) < self.window:
            return np.array([])
        momentum = closes[self.window-1:] / closes[:-self.window+1] - 1
        return momentum[:-1] if len(momentum) > 1 else momentum

    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute Relative Strength Index."""
        if len(closes) < period + 1:
            return np.array([])

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))

        return rsi / 100  # Normalize to [0, 1]

    def _compute_macd(self, closes: np.ndarray) -> np.ndarray:
        """Compute MACD normalized."""
        if len(closes) < 26:
            return np.array([])

        # EMA 12
        alpha12 = 2 / 13
        ema12 = np.zeros(len(closes))
        ema12[0] = closes[0]
        for i in range(1, len(closes)):
            ema12[i] = alpha12 * closes[i] + (1 - alpha12) * ema12[i-1]

        # EMA 26
        alpha26 = 2 / 27
        ema26 = np.zeros(len(closes))
        ema26[0] = closes[0]
        for i in range(1, len(closes)):
            ema26[i] = alpha26 * closes[i] + (1 - alpha26) * ema26[i-1]

        macd = (ema12 - ema26) / closes
        return macd[25:]

    def _compute_bollinger_position(self, closes: np.ndarray) -> np.ndarray:
        """Compute position within Bollinger Bands."""
        if len(closes) < self.window:
            return np.array([])

        positions = []
        for i in range(self.window - 1, len(closes)):
            window_data = closes[i-self.window+1:i+1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std > 0:
                position = (closes[i] - mean) / (2 * std)
            else:
                position = 0
            positions.append(position)

        return np.array(positions)

    def _compute_volume_sma_ratio(self, volumes: np.ndarray) -> np.ndarray:
        """Compute ratio of volume to its SMA."""
        if len(volumes) < self.window:
            return np.array([])

        sma = np.convolve(volumes, np.ones(self.window)/self.window, mode='valid')
        ratio = volumes[self.window-1:] / np.where(sma != 0, sma, 1) - 1
        return ratio


def klines_to_dataframe(klines: List[Kline]) -> pd.DataFrame:
    """Convert list of Klines to pandas DataFrame."""
    return pd.DataFrame([k.to_dict() for k in klines])


if __name__ == "__main__":
    # Example usage
    print("Generating simulated data...")
    klines = SimulatedDataGenerator.generate_regime_changing_klines(500)
    print(f"Generated {len(klines)} klines")

    print("\nComputing features...")
    feature_gen = FeatureGenerator(window=20)
    features = feature_gen.compute_features(klines)
    print(f"Computed features shape: {features.shape}")

    print("\nFeature statistics:")
    feature_names = [
        "returns_1d", "returns_5d", "returns_10d", "sma_ratio", "ema_ratio",
        "volatility", "momentum", "rsi", "macd", "bb_position", "volume_sma_ratio"
    ]
    for i, name in enumerate(feature_names):
        print(f"  {name}: mean={features[:, i].mean():.6f}, std={features[:, i].std():.6f}")
