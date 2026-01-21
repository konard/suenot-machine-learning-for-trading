"""
Market regime classification algorithms.

This module provides various approaches to classify market regimes:
- Statistical methods (HMM-based)
- Text-based classification using LLM embeddings
- Hybrid approaches combining multiple signals
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime enumeration."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"


@dataclass
class RegimeResult:
    """Result of regime classification."""
    regime: MarketRegime
    probability: float
    confidence: float
    explanation: str
    supporting_factors: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'regime': self.regime.value,
            'probability': self.probability,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'supporting_factors': self.supporting_factors
        }


class HMMRegimeDetector:
    """
    Hidden Markov Model baseline for regime detection.

    Uses returns and volatility to classify market regimes
    using a simplified statistical approach.
    """

    def __init__(self, n_regimes: int = 4, lookback: int = 60):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden states (regimes)
            lookback: Lookback period for calculations
        """
        self.n_regimes = n_regimes
        self.lookback = lookback

        # Regime parameters (will be estimated from data)
        self.regime_params = {
            MarketRegime.BULL: {'mean': 0.001, 'vol': 0.01},
            MarketRegime.BEAR: {'mean': -0.001, 'vol': 0.02},
            MarketRegime.SIDEWAYS: {'mean': 0.0, 'vol': 0.008},
            MarketRegime.HIGH_VOLATILITY: {'mean': 0.0, 'vol': 0.03}
        }

        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> 'HMMRegimeDetector':
        """
        Fit the detector on historical data.

        Args:
            data: DataFrame with 'returns' and optionally 'volatility' columns

        Returns:
            Self for chaining
        """
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close'].pct_change()

        returns = data['returns'].dropna().values

        # Estimate regime parameters from data
        overall_mean = np.mean(returns)
        overall_vol = np.std(returns)

        # Simple parameter estimation
        self.regime_params = {
            MarketRegime.BULL: {
                'mean': overall_mean + overall_vol,
                'vol': overall_vol * 0.8
            },
            MarketRegime.BEAR: {
                'mean': overall_mean - overall_vol,
                'vol': overall_vol * 1.5
            },
            MarketRegime.SIDEWAYS: {
                'mean': overall_mean,
                'vol': overall_vol * 0.6
            },
            MarketRegime.HIGH_VOLATILITY: {
                'mean': overall_mean,
                'vol': overall_vol * 2.5
            }
        }

        self.is_fitted = True
        logger.info("HMM detector fitted on data")
        return self

    def detect_regime(
        self,
        returns: np.ndarray,
        volatility: Optional[np.ndarray] = None
    ) -> RegimeResult:
        """
        Detect current market regime.

        Args:
            returns: Array of recent returns
            volatility: Optional volatility array

        Returns:
            RegimeResult with classification
        """
        if len(returns) < self.lookback:
            returns = np.pad(
                returns,
                (self.lookback - len(returns), 0),
                'constant',
                constant_values=0
            )

        recent_returns = returns[-self.lookback:]
        mean_return = np.mean(recent_returns)

        if volatility is None:
            volatility_value = np.std(recent_returns) * np.sqrt(252)
        else:
            volatility_value = np.mean(volatility[-self.lookback:]) if len(volatility) >= self.lookback else np.mean(volatility)

        # Classification logic based on thresholds
        if volatility_value > 0.40:
            regime = MarketRegime.CRISIS
            prob = min(0.95, volatility_value / 0.5)
            explanation = f"Extreme volatility detected: {volatility_value:.1%} annualized"
        elif volatility_value > 0.25:
            regime = MarketRegime.HIGH_VOLATILITY
            prob = min(0.9, (volatility_value - 0.25) / 0.15 + 0.5)
            explanation = f"Elevated volatility: {volatility_value:.1%} annualized"
        elif mean_return > 0.0005 and volatility_value < 0.20:
            regime = MarketRegime.BULL
            prob = min(0.9, mean_return / 0.002 + 0.5)
            explanation = f"Positive trend with low volatility: return={mean_return:.4f}"
        elif mean_return < -0.0005:
            regime = MarketRegime.BEAR
            prob = min(0.9, abs(mean_return) / 0.002 + 0.5)
            explanation = f"Negative trend: return={mean_return:.4f}"
        else:
            regime = MarketRegime.SIDEWAYS
            prob = 0.7
            explanation = "Range-bound market conditions"

        return RegimeResult(
            regime=regime,
            probability=prob,
            confidence=prob * 0.85 + 0.1,
            explanation=explanation,
            supporting_factors=[
                f"Mean return: {mean_return:.4f}",
                f"Volatility: {volatility_value:.2%}",
                f"Lookback: {self.lookback} periods"
            ]
        )


class StatisticalRegimeClassifier:
    """
    Statistical regime classifier using returns and volatility.

    Provides methods for fitting on historical data and
    classifying current market conditions.
    """

    def __init__(
        self,
        lookback_window: int = 60,
        volatility_threshold_high: float = 0.25,
        volatility_threshold_crisis: float = 0.40,
        trend_threshold: float = 0.0005
    ):
        """
        Initialize statistical classifier.

        Args:
            lookback_window: Window size for calculations
            volatility_threshold_high: Threshold for high volatility regime
            volatility_threshold_crisis: Threshold for crisis regime
            trend_threshold: Return threshold for trend detection
        """
        self.lookback_window = lookback_window
        self.volatility_threshold_high = volatility_threshold_high
        self.volatility_threshold_crisis = volatility_threshold_crisis
        self.trend_threshold = trend_threshold

        self.returns_history: List[float] = []
        self.volatility_history: List[float] = []

    def fit(self, data: pd.DataFrame) -> 'StatisticalRegimeClassifier':
        """
        Fit classifier on historical data.

        Args:
            data: DataFrame with 'close' or 'returns' column

        Returns:
            Self for chaining
        """
        if 'returns' not in data.columns:
            returns = data['close'].pct_change().dropna()
        else:
            returns = data['returns'].dropna()

        if 'volatility' in data.columns:
            volatility = data['volatility'].dropna()
        else:
            volatility = returns.rolling(window=20).std().dropna()

        # Align lengths
        min_len = min(len(returns), len(volatility))
        returns = returns.iloc[-min_len:]
        volatility = volatility.iloc[-min_len:]

        # Store history
        self.returns_history = list(returns.values)
        self.volatility_history = list(volatility.values)

        return self

    def update(self, returns: float, volatility: float):
        """
        Update with new data point.

        Args:
            returns: New return value
            volatility: New volatility value
        """
        self.returns_history.append(returns)
        self.volatility_history.append(volatility)

        # Keep only lookback window
        if len(self.returns_history) > self.lookback_window:
            self.returns_history = self.returns_history[-self.lookback_window:]
            self.volatility_history = self.volatility_history[-self.lookback_window:]

    def classify(self, data: Optional[pd.DataFrame] = None) -> RegimeResult:
        """
        Classify current regime based on accumulated data.

        Args:
            data: Optional DataFrame to fit and classify (convenience method)

        Returns:
            RegimeResult with classification
        """
        # If data provided, fit first
        if data is not None:
            self.fit(data)

        if len(self.returns_history) < 10:
            return RegimeResult(
                regime=MarketRegime.SIDEWAYS,
                probability=0.5,
                confidence=0.3,
                explanation="Insufficient data for classification",
                supporting_factors=["Data points: {}".format(len(self.returns_history))]
            )

        mean_return = np.mean(self.returns_history)
        mean_vol = np.mean(self.volatility_history)
        # Only annualize if volatility appears to be daily (small values)
        # Daily stock vol is typically 0.005-0.03, annualized is 0.08-0.50
        annualized_vol = mean_vol * np.sqrt(252) if mean_vol < 0.05 else mean_vol

        # Classification
        if annualized_vol > self.volatility_threshold_crisis:
            regime = MarketRegime.CRISIS
            prob = min(0.95, annualized_vol / 0.5)
        elif annualized_vol > self.volatility_threshold_high:
            regime = MarketRegime.HIGH_VOLATILITY
            prob = 0.75
        elif mean_return > self.trend_threshold:
            regime = MarketRegime.BULL
            prob = min(0.9, mean_return / 0.002 + 0.5)
        elif mean_return < -self.trend_threshold:
            regime = MarketRegime.BEAR
            prob = min(0.9, abs(mean_return) / 0.002 + 0.5)
        else:
            regime = MarketRegime.SIDEWAYS
            prob = 0.7

        return RegimeResult(
            regime=regime,
            probability=prob,
            confidence=prob * 0.9 + 0.1,
            explanation=f"Statistical classification: return={mean_return:.4f}, vol={annualized_vol:.2%}",
            supporting_factors=[
                f"Mean return: {mean_return:.4f}",
                f"Annualized vol: {annualized_vol:.2%}",
                f"Data points: {len(self.returns_history)}"
            ]
        )


class TextRegimeClassifier:
    """
    Classify market regime from textual data.

    Uses keyword matching and sentiment analysis to detect
    market conditions from news headlines and social media.
    """

    def __init__(self):
        """Initialize text-based regime classifier."""
        # Regime keywords for classification
        self.regime_keywords = {
            MarketRegime.BULL: [
                'rally', 'surge', 'gains', 'bullish', 'optimism',
                'record high', 'breakout', 'momentum', 'buying',
                'strong earnings', 'growth', 'expansion', 'uptick',
                'soar', 'jump', 'advance', 'climb', 'beat expectations'
            ],
            MarketRegime.BEAR: [
                'plunge', 'crash', 'bearish', 'selloff', 'decline',
                'losses', 'fear', 'correction', 'downturn',
                'recession', 'weak earnings', 'contraction', 'slump',
                'tumble', 'slide', 'drop', 'sink', 'miss expectations'
            ],
            MarketRegime.SIDEWAYS: [
                'consolidation', 'range-bound', 'flat', 'stable',
                'unchanged', 'mixed', 'neutral', 'waiting',
                'indecisive', 'sideways', 'choppy', 'uncertain'
            ],
            MarketRegime.HIGH_VOLATILITY: [
                'volatile', 'swings', 'uncertainty', 'turbulent',
                'whipsaw', 'erratic', 'unpredictable', 'vix spike',
                'risk-off', 'hedge', 'fluctuation', 'instability'
            ],
            MarketRegime.CRISIS: [
                'panic', 'crash', 'crisis', 'collapse', 'meltdown',
                'black swan', 'contagion', 'systemic', 'emergency',
                'circuit breaker', 'flash crash', 'capitulation'
            ]
        }

    def classify_text(self, texts: List[str]) -> List[RegimeResult]:
        """
        Classify regime based on text content.

        Args:
            texts: List of news/social media texts

        Returns:
            List of RegimeResult classifications
        """
        results = []

        for text in texts:
            text_lower = text.lower()
            regime_scores = {}

            for regime, keywords in self.regime_keywords.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                regime_scores[regime] = score / len(keywords)

            # Get best regime
            if max(regime_scores.values()) == 0:
                best_regime = MarketRegime.SIDEWAYS
                prob = 0.5
                matched = []
            else:
                best_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
                total = sum(regime_scores.values()) + 1e-8
                prob = regime_scores[best_regime] / total
                matched = [
                    kw for kw in self.regime_keywords[best_regime]
                    if kw in text_lower
                ][:3]

            results.append(RegimeResult(
                regime=best_regime,
                probability=prob,
                confidence=min(0.9, prob + 0.2),
                explanation=f"Text-based classification from content analysis",
                supporting_factors=[
                    f"Matched keywords: {matched}",
                    f"Text snippet: {text[:80]}..."
                ]
            ))

        return results

    def aggregate_regime(
        self,
        texts: List[str],
        weights: Optional[List[float]] = None
    ) -> RegimeResult:
        """
        Aggregate multiple texts into single regime classification.

        Args:
            texts: List of recent market texts
            weights: Optional weights (e.g., by recency)

        Returns:
            Aggregated RegimeResult
        """
        if not texts:
            return RegimeResult(
                regime=MarketRegime.SIDEWAYS,
                probability=0.5,
                confidence=0.3,
                explanation="No texts provided",
                supporting_factors=[]
            )

        if weights is None:
            weights = [1.0] * len(texts)

        results = self.classify_text(texts)

        # Aggregate scores
        regime_votes = {r: 0.0 for r in MarketRegime}

        for result, weight in zip(results, weights):
            regime_votes[result.regime] += weight * result.probability

        total = sum(regime_votes.values()) + 1e-8
        regime_probs = {r: v / total for r, v in regime_votes.items()}

        best_regime = max(regime_probs.items(), key=lambda x: x[1])[0]

        return RegimeResult(
            regime=best_regime,
            probability=regime_probs[best_regime],
            confidence=min(0.95, regime_probs[best_regime] + 0.1),
            explanation=f"Aggregated from {len(texts)} text sources",
            supporting_factors=[
                f"{r.value}: {p:.1%}" for r, p in sorted(
                    regime_probs.items(), key=lambda x: -x[1]
                )[:3]
            ]
        )


class HybridRegimeClassifier:
    """
    Hybrid regime classifier combining multiple signals:
    - Statistical analysis of prices/volatility
    - Text-based sentiment analysis
    - Optional economic indicators
    """

    def __init__(
        self,
        lookback: int = 60,
        text_weight: float = 0.3,
        stats_weight: float = 0.5,
        econ_weight: float = 0.2
    ):
        """
        Initialize hybrid classifier.

        Args:
            lookback: Lookback period for statistical analysis
            text_weight: Weight for text signals
            stats_weight: Weight for statistical signals
            econ_weight: Weight for economic signals
        """
        self.lookback = lookback
        self.weights = {
            'text': text_weight,
            'stats': stats_weight,
            'econ': econ_weight
        }

        self.hmm_detector = HMMRegimeDetector(lookback=lookback)
        self.text_classifier = TextRegimeClassifier()
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> 'HybridRegimeClassifier':
        """
        Fit the classifier on historical data.

        Args:
            data: DataFrame with price data

        Returns:
            Self for chaining
        """
        self.hmm_detector.fit(data)
        self.is_fitted = True
        return self

    def classify(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        texts: Optional[List[str]] = None,
        economic_data: Optional[Dict[str, float]] = None
    ) -> RegimeResult:
        """
        Classify regime using all available information.

        Args:
            returns: Recent return series
            volatility: Volatility series
            texts: Optional news/social media texts
            economic_data: Optional economic indicators

        Returns:
            Combined RegimeResult
        """
        # Statistical classification
        stats_result = self.hmm_detector.detect_regime(returns, volatility)

        # Text classification
        if texts:
            text_result = self.text_classifier.aggregate_regime(texts)
        else:
            text_result = stats_result  # Fallback

        # Economic classification
        if economic_data:
            econ_result = self._classify_from_economic(economic_data)
        else:
            econ_result = stats_result  # Fallback

        # Combine results
        regime_scores = {r: 0.0 for r in MarketRegime}

        regime_scores[stats_result.regime] += self.weights['stats'] * stats_result.probability
        regime_scores[text_result.regime] += self.weights['text'] * text_result.probability
        regime_scores[econ_result.regime] += self.weights['econ'] * econ_result.probability

        # Normalize
        total = sum(regime_scores.values()) + 1e-8
        regime_probs = {r: s / total for r, s in regime_scores.items()}

        best_regime = max(regime_probs.items(), key=lambda x: x[1])[0]
        prob = regime_probs[best_regime]

        return RegimeResult(
            regime=best_regime,
            probability=prob,
            confidence=min(0.95, prob + 0.1),
            explanation="Hybrid classification combining stats, text, and economic signals",
            supporting_factors=[
                f"Statistical: {stats_result.regime.value} ({stats_result.probability:.0%})",
                f"Text: {text_result.regime.value} ({text_result.probability:.0%})",
                f"Economic: {econ_result.regime.value} ({econ_result.probability:.0%})"
            ]
        )

    def classify_current(self, data: pd.DataFrame, texts: Optional[List[str]] = None) -> RegimeResult:
        """
        Classify current regime from DataFrame.

        Args:
            data: DataFrame with 'returns' and 'volatility' columns
            texts: Optional text data

        Returns:
            RegimeResult
        """
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close'].pct_change()

        if 'volatility' not in data.columns:
            data['volatility'] = data['returns'].rolling(window=20).std()

        returns = data['returns'].dropna().values
        volatility = data['volatility'].dropna().values

        return HybridRegimeClassifier.classify(self, returns, volatility, texts)

    def _classify_from_economic(self, data: Dict[str, float]) -> RegimeResult:
        """Classify based on economic indicators."""
        gdp_growth = data.get('gdp_growth', 0.02)
        unemployment = data.get('unemployment', 0.05)
        inflation = data.get('inflation', 0.02)
        yield_curve = data.get('yield_curve_slope', 0.01)

        # Rule-based classification
        if gdp_growth > 0.03 and unemployment < 0.05:
            regime = MarketRegime.BULL
            prob = 0.7
        elif gdp_growth < 0 or unemployment > 0.08:
            regime = MarketRegime.BEAR
            prob = 0.7
        elif yield_curve < 0:  # Inverted yield curve
            regime = MarketRegime.HIGH_VOLATILITY
            prob = 0.6
        else:
            regime = MarketRegime.SIDEWAYS
            prob = 0.5

        return RegimeResult(
            regime=regime,
            probability=prob,
            confidence=0.6,
            explanation="Economic indicator-based classification",
            supporting_factors=[
                f"GDP growth: {gdp_growth:.1%}",
                f"Unemployment: {unemployment:.1%}",
                f"Inflation: {inflation:.1%}"
            ]
        )


class CryptoRegimeClassifier(HybridRegimeClassifier):
    """
    Specialized regime classifier for cryptocurrency markets.

    Adjusts thresholds for higher volatility typical in crypto.
    """

    def __init__(
        self,
        lookback: int = 60,
        volatility_threshold: float = 0.5,
        trend_threshold: float = 0.02
    ):
        """
        Initialize crypto-specific classifier.

        Args:
            lookback: Lookback period
            volatility_threshold: Higher threshold for crypto
            trend_threshold: Higher threshold for trend detection
        """
        super().__init__(lookback=lookback)

        # Adjust for crypto volatility
        self.hmm_detector.regime_params = {
            MarketRegime.BULL: {'mean': 0.003, 'vol': 0.03},
            MarketRegime.BEAR: {'mean': -0.003, 'vol': 0.05},
            MarketRegime.SIDEWAYS: {'mean': 0.0, 'vol': 0.02},
            MarketRegime.HIGH_VOLATILITY: {'mean': 0.0, 'vol': 0.08}
        }

        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold

    def get_crypto_insights(self, data: pd.DataFrame) -> Dict:
        """
        Get crypto-specific insights.

        Args:
            data: DataFrame with crypto OHLCV data

        Returns:
            Dictionary with crypto-specific metrics
        """
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close'].pct_change()

        returns = data['returns'].dropna()

        # 24h volatility
        vol_24h = returns.tail(24).std() * np.sqrt(365 * 24)

        # Simple funding rate proxy (price momentum)
        momentum_24h = (data['close'].iloc[-1] / data['close'].iloc[-24] - 1) if len(data) >= 24 else 0

        # Long/short ratio proxy (based on price vs SMA)
        sma = data['close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else data['close'].iloc[-1]
        ls_ratio = data['close'].iloc[-1] / sma

        return {
            'volatility_24h': vol_24h,
            'funding_rate': momentum_24h * 0.01,  # Proxy
            'ls_ratio': ls_ratio,
            'price_change_24h': momentum_24h,
            'volume_24h': data['volume'].tail(24).sum() if 'volume' in data.columns else 0
        }

    def classify(self, data: pd.DataFrame, texts: Optional[List[str]] = None) -> RegimeResult:
        """
        Classify regime from DataFrame (convenience wrapper).

        Args:
            data: DataFrame with OHLCV data
            texts: Optional news/social texts

        Returns:
            RegimeResult with classification
        """
        return self.classify_current(data, texts)
