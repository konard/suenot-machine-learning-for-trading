"""
Concept extraction module for computing interpretable trading concepts.

Concepts are human-understandable features that describe market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class VolatilityRegime(Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class MomentumSignal(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class VolumeProfile(Enum):
    ABOVE_AVERAGE = "above_average"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"


@dataclass
class TradingConcepts:
    """Container for extracted trading concepts."""
    trend_direction: TrendDirection
    trend_strength: float  # 0 to 1
    volatility_regime: VolatilityRegime
    volatility_value: float
    momentum_signal: MomentumSignal
    momentum_value: float
    volume_profile: VolumeProfile
    volume_ratio: float
    rsi: float
    support_distance: float
    resistance_distance: float

    def to_dict(self) -> Dict:
        return {
            'trend_direction': self.trend_direction.value,
            'trend_strength': self.trend_strength,
            'volatility_regime': self.volatility_regime.value,
            'volatility_value': self.volatility_value,
            'momentum_signal': self.momentum_signal.value,
            'momentum_value': self.momentum_value,
            'volume_profile': self.volume_profile.value,
            'volume_ratio': self.volume_ratio,
            'rsi': self.rsi,
            'support_distance': self.support_distance,
            'resistance_distance': self.resistance_distance,
        }

    def to_vector(self) -> np.ndarray:
        """Convert concepts to numerical vector for model input."""
        trend_map = {TrendDirection.BULLISH: 1, TrendDirection.NEUTRAL: 0, TrendDirection.BEARISH: -1}
        vol_map = {VolatilityRegime.HIGH: 1, VolatilityRegime.NORMAL: 0, VolatilityRegime.LOW: -1}
        mom_map = {MomentumSignal.POSITIVE: 1, MomentumSignal.NEUTRAL: 0, MomentumSignal.NEGATIVE: -1}
        vol_prof_map = {VolumeProfile.ABOVE_AVERAGE: 1, VolumeProfile.AVERAGE: 0, VolumeProfile.BELOW_AVERAGE: -1}

        return np.array([
            trend_map[self.trend_direction],
            self.trend_strength,
            vol_map[self.volatility_regime],
            self.volatility_value,
            mom_map[self.momentum_signal],
            self.momentum_value,
            vol_prof_map[self.volume_profile],
            self.volume_ratio,
            self.rsi / 100.0,  # Normalize to 0-1
            self.support_distance,
            self.resistance_distance,
        ])


class ConceptExtractor:
    """
    Extracts interpretable trading concepts from OHLCV data.

    Concepts include:
    - Trend direction and strength
    - Volatility regime
    - Momentum signals
    - Volume profile
    - Support/resistance levels
    """

    def __init__(
        self,
        trend_window: int = 20,
        volatility_window: int = 20,
        volume_window: int = 20,
        rsi_window: int = 14,
    ):
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.rsi_window = rsi_window

    def extract(self, df: pd.DataFrame) -> TradingConcepts:
        """
        Extract trading concepts from the most recent data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            TradingConcepts object containing all extracted concepts
        """
        if len(df) < max(self.trend_window, self.volatility_window, self.rsi_window):
            raise ValueError(f"Need at least {max(self.trend_window, self.volatility_window, self.rsi_window)} data points")

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values

        trend_dir, trend_str = self._compute_trend(closes)
        vol_regime, vol_value = self._compute_volatility(closes)
        mom_signal, mom_value = self._compute_momentum(closes)
        vol_profile, vol_ratio = self._compute_volume_profile(volumes)
        rsi = self._compute_rsi(closes)
        support_dist, resist_dist = self._compute_support_resistance(closes, highs, lows)

        return TradingConcepts(
            trend_direction=trend_dir,
            trend_strength=trend_str,
            volatility_regime=vol_regime,
            volatility_value=vol_value,
            momentum_signal=mom_signal,
            momentum_value=mom_value,
            volume_profile=vol_profile,
            volume_ratio=vol_ratio,
            rsi=rsi,
            support_distance=support_dist,
            resistance_distance=resist_dist,
        )

    def extract_batch(self, df: pd.DataFrame, lookback: int = 50) -> List[TradingConcepts]:
        """
        Extract concepts for multiple time points.

        Args:
            df: DataFrame with OHLCV data
            lookback: Minimum data points needed before first extraction

        Returns:
            List of TradingConcepts for each valid time point
        """
        concepts = []
        min_required = max(lookback, self.trend_window, self.volatility_window, self.rsi_window)

        for i in range(min_required, len(df)):
            window_df = df.iloc[:i+1]
            concepts.append(self.extract(window_df))

        return concepts

    def _compute_trend(self, closes: np.ndarray) -> Tuple[TrendDirection, float]:
        """Compute trend direction and strength using linear regression."""
        window = closes[-self.trend_window:]
        x = np.arange(len(window))
        slope, _ = np.polyfit(x, window, 1)

        normalized_slope = slope / np.mean(window)
        strength = min(abs(normalized_slope) * 100, 1.0)

        if normalized_slope > 0.001:
            direction = TrendDirection.BULLISH
        elif normalized_slope < -0.001:
            direction = TrendDirection.BEARISH
        else:
            direction = TrendDirection.NEUTRAL

        return direction, strength

    def _compute_volatility(self, closes: np.ndarray) -> Tuple[VolatilityRegime, float]:
        """Compute volatility regime using rolling standard deviation."""
        returns = np.diff(closes) / closes[:-1]
        current_vol = np.std(returns[-self.volatility_window:])
        historical_vol = np.std(returns)

        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0

        if vol_ratio > 1.5:
            regime = VolatilityRegime.HIGH
        elif vol_ratio < 0.7:
            regime = VolatilityRegime.LOW
        else:
            regime = VolatilityRegime.NORMAL

        return regime, vol_ratio

    def _compute_momentum(self, closes: np.ndarray) -> Tuple[MomentumSignal, float]:
        """Compute momentum signal using rate of change."""
        roc = (closes[-1] - closes[-self.trend_window]) / closes[-self.trend_window]

        if roc > 0.02:
            signal = MomentumSignal.POSITIVE
        elif roc < -0.02:
            signal = MomentumSignal.NEGATIVE
        else:
            signal = MomentumSignal.NEUTRAL

        return signal, roc

    def _compute_volume_profile(self, volumes: np.ndarray) -> Tuple[VolumeProfile, float]:
        """Compute volume profile relative to recent average."""
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[-self.volume_window:])
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        if vol_ratio > 1.5:
            profile = VolumeProfile.ABOVE_AVERAGE
        elif vol_ratio < 0.7:
            profile = VolumeProfile.BELOW_AVERAGE
        else:
            profile = VolumeProfile.AVERAGE

        return profile, vol_ratio

    def _compute_rsi(self, closes: np.ndarray) -> float:
        """Compute Relative Strength Index."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-self.rsi_window:])
        avg_loss = np.mean(losses[-self.rsi_window:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_support_resistance(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute distance to support and resistance levels."""
        recent_high = np.max(highs[-self.trend_window:])
        recent_low = np.min(lows[-self.trend_window:])
        current_price = closes[-1]

        price_range = recent_high - recent_low
        if price_range == 0:
            return 0.5, 0.5

        support_dist = (current_price - recent_low) / price_range
        resist_dist = (recent_high - current_price) / price_range

        return support_dist, resist_dist


def generate_concept_labels(
    df: pd.DataFrame,
    future_window: int = 5,
    threshold: float = 0.02
) -> np.ndarray:
    """
    Generate concept labels based on future price movement.

    Used for training the concept predictor in supervised learning.

    Args:
        df: DataFrame with OHLCV data
        future_window: Number of periods to look ahead
        threshold: Minimum price change for non-neutral label

    Returns:
        Array of labels: 1 (bullish), 0 (neutral), -1 (bearish)
    """
    closes = df['close'].values
    labels = np.zeros(len(closes))

    for i in range(len(closes) - future_window):
        future_return = (closes[i + future_window] - closes[i]) / closes[i]

        if future_return > threshold:
            labels[i] = 1
        elif future_return < -threshold:
            labels[i] = -1

    return labels
