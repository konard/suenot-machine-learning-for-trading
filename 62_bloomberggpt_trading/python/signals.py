"""
Trading Signal Generation Module

This module provides tools for converting LLM sentiment analysis
into actionable trading signals.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


class SignalType(Enum):
    """Type of trading signal."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingSignal:
    """Trading signal generated from LLM analysis."""
    timestamp: datetime
    symbol: str
    signal: float  # -1 (strong sell) to 1 (strong buy)
    confidence: float  # 0 to 1
    reasoning: str
    source_type: str  # "news", "earnings", "filing", "social"
    raw_sentiment: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def signal_type(self) -> SignalType:
        """Convert numeric signal to signal type."""
        if self.signal > 0.6:
            return SignalType.STRONG_BUY
        elif self.signal > 0.2:
            return SignalType.BUY
        elif self.signal < -0.6:
            return SignalType.STRONG_SELL
        elif self.signal < -0.2:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act on."""
        return abs(self.signal) > 0.2 and self.confidence > 0.5


class LLMSignalGenerator:
    """
    Generate trading signals from LLM sentiment analysis.

    This class converts sentiment analysis results into actionable
    trading signals, with configurable thresholds and source weighting.

    Examples:
        >>> from sentiment import FinancialSentimentAnalyzer
        >>> analyzer = FinancialSentimentAnalyzer()
        >>> generator = LLMSignalGenerator(analyzer)
        >>> signal = generator.generate_signal(
        ...     text="Apple reports record earnings",
        ...     symbol="AAPL",
        ...     source_type="earnings"
        ... )
    """

    def __init__(
        self,
        sentiment_analyzer,
        signal_threshold: float = 0.3,
        confidence_threshold: float = 0.6,
        source_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the signal generator.

        Args:
            sentiment_analyzer: Sentiment analyzer instance
            signal_threshold: Minimum signal strength to generate
            confidence_threshold: Minimum confidence to generate
            source_weights: Weight by source type (default provided)
        """
        self.analyzer = sentiment_analyzer
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold

        # Default source weights (importance)
        self.source_weights = source_weights or {
            "earnings": 1.0,      # Most important
            "filing": 0.9,        # SEC filings
            "news": 0.7,          # News articles
            "analyst": 0.8,       # Analyst reports
            "social": 0.3,        # Social media (noisy)
            "press": 0.6,         # Press releases
            "default": 0.5
        }

        # Sentiment to signal mapping
        self._sentiment_scores = {
            "POSITIVE": 1.0,
            "NEGATIVE": -1.0,
            "NEUTRAL": 0.0,
            "NOT_MENTIONED": 0.0
        }

    def generate_signal(
        self,
        text: str,
        symbol: str,
        source_type: str = "news",
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal from financial text.

        Args:
            text: Financial text to analyze
            symbol: Stock/crypto symbol
            source_type: Type of source (news, earnings, etc.)
            timestamp: Time of the text (default: now)
            metadata: Additional metadata to attach

        Returns:
            TradingSignal if confidence threshold met, else None
        """
        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        # Analyze sentiment
        sentiment = self.analyzer.analyze(text)

        # Check confidence threshold
        if sentiment.confidence < self.confidence_threshold:
            return None

        # Calculate signal strength
        base_signal = self._sentiment_scores.get(sentiment.label, 0)
        source_weight = self.source_weights.get(
            source_type, self.source_weights["default"]
        )

        # Combine sentiment score with confidence and source weight
        signal_strength = base_signal * sentiment.confidence * source_weight

        # Apply threshold
        if abs(signal_strength) < self.signal_threshold * source_weight:
            return None

        # Construct reasoning
        reasoning = self._build_reasoning(sentiment, source_type, signal_strength)

        return TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal=signal_strength,
            confidence=sentiment.confidence,
            reasoning=reasoning,
            source_type=source_type,
            raw_sentiment=sentiment.label,
            metadata=metadata
        )

    def generate_signals_batch(
        self,
        items: List[Dict]
    ) -> List[TradingSignal]:
        """
        Generate signals from multiple items.

        Args:
            items: List of dicts with 'text', 'symbol', 'source_type', etc.

        Returns:
            List of generated signals (excluding None)
        """
        signals = []
        for item in items:
            signal = self.generate_signal(
                text=item.get("text", ""),
                symbol=item.get("symbol", ""),
                source_type=item.get("source_type", "news"),
                timestamp=item.get("timestamp"),
                metadata=item.get("metadata")
            )
            if signal is not None:
                signals.append(signal)

        return signals

    def aggregate_signals(
        self,
        signals: List[TradingSignal],
        window_hours: float = 24,
        reference_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Aggregate multiple signals into position recommendations.

        Uses time decay to weight more recent signals higher.

        Args:
            signals: List of trading signals
            window_hours: Time window for aggregation
            reference_time: Reference time for decay (default: now)

        Returns:
            Dict of symbol -> recommended position (-1 to 1)
        """
        reference_time = reference_time or datetime.now()

        # Group by symbol
        symbol_signals: Dict[str, List[TradingSignal]] = {}
        for signal in signals:
            # Check if within window
            age_hours = (reference_time - signal.timestamp).total_seconds() / 3600
            if age_hours <= window_hours:
                if signal.symbol not in symbol_signals:
                    symbol_signals[signal.symbol] = []
                symbol_signals[signal.symbol].append(signal)

        # Aggregate with time decay
        positions = {}
        for symbol, sigs in symbol_signals.items():
            weighted_sum = 0
            weight_total = 0

            for sig in sigs:
                age_hours = (reference_time - sig.timestamp).total_seconds() / 3600
                # Exponential decay
                time_weight = np.exp(-age_hours / window_hours)

                weighted_sum += sig.signal * sig.confidence * time_weight
                weight_total += sig.confidence * time_weight

            if weight_total > 0:
                positions[symbol] = np.clip(weighted_sum / weight_total, -1, 1)

        return positions

    def _build_reasoning(
        self,
        sentiment,
        source_type: str,
        signal: float
    ) -> str:
        """Build human-readable reasoning for signal."""
        direction = "bullish" if signal > 0 else "bearish" if signal < 0 else "neutral"
        strength = "strongly" if abs(signal) > 0.6 else "moderately" if abs(signal) > 0.3 else "slightly"

        return (
            f"{strength.capitalize()} {direction} signal based on {source_type} analysis. "
            f"Sentiment: {sentiment.label} with {sentiment.confidence:.0%} confidence."
        )


class MultiSourceSignalGenerator(LLMSignalGenerator):
    """
    Signal generator that handles multiple sources and entity disambiguation.

    Extends the base generator with more sophisticated multi-source handling.
    """

    def __init__(self, *args, entity_mapper: Optional[Dict] = None, **kwargs):
        """
        Initialize multi-source generator.

        Args:
            entity_mapper: Dict mapping entity mentions to canonical symbols
        """
        super().__init__(*args, **kwargs)
        self.entity_mapper = entity_mapper or {}

    def map_entity(self, mention: str) -> str:
        """Map entity mention to canonical symbol."""
        mention_lower = mention.lower()
        return self.entity_mapper.get(mention_lower, mention.upper())

    def generate_multi_entity_signals(
        self,
        text: str,
        entities: List[str],
        source_type: str = "news",
        timestamp: Optional[datetime] = None
    ) -> List[TradingSignal]:
        """
        Generate signals for multiple entities in a single text.

        Args:
            text: Text containing multiple entity mentions
            entities: List of entity names to analyze
            source_type: Source type for weighting
            timestamp: Timestamp of the text

        Returns:
            List of signals for each mentioned entity
        """
        signals = []

        # Get aspect-specific sentiments
        aspect_sentiments = self.analyzer.analyze_aspects(text, entities)

        for entity, sentiment in aspect_sentiments.items():
            if sentiment.label == "NOT_MENTIONED":
                continue

            symbol = self.map_entity(entity)

            # Calculate signal
            base_signal = self._sentiment_scores.get(sentiment.label, 0)
            source_weight = self.source_weights.get(source_type, 0.5)
            signal_strength = base_signal * sentiment.confidence * source_weight

            if abs(signal_strength) >= self.signal_threshold * source_weight:
                signals.append(TradingSignal(
                    timestamp=timestamp or datetime.now(),
                    symbol=symbol,
                    signal=signal_strength,
                    confidence=sentiment.confidence,
                    reasoning=f"Aspect-specific analysis for {entity}: {sentiment.label}",
                    source_type=source_type,
                    raw_sentiment=sentiment.label,
                    metadata={"original_entity": entity}
                ))

        return signals


def signals_to_dataframe(signals: List[TradingSignal]) -> pd.DataFrame:
    """
    Convert list of signals to DataFrame for analysis.

    Args:
        signals: List of TradingSignal objects

    Returns:
        DataFrame with signal data
    """
    if not signals:
        return pd.DataFrame()

    records = []
    for sig in signals:
        records.append({
            "timestamp": sig.timestamp,
            "symbol": sig.symbol,
            "signal": sig.signal,
            "confidence": sig.confidence,
            "signal_type": sig.signal_type.value,
            "source_type": sig.source_type,
            "raw_sentiment": sig.raw_sentiment,
            "reasoning": sig.reasoning
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Demo with mock analyzer
    class MockAnalyzer:
        def analyze(self, text):
            from sentiment import SentimentResult
            # Simple mock based on keywords
            text_lower = text.lower()
            if any(w in text_lower for w in ["record", "beat", "surge", "growth"]):
                return SentimentResult("POSITIVE", 0.85, 0.10, 0.05, 0.85)
            elif any(w in text_lower for w in ["drop", "plunge", "miss", "decline"]):
                return SentimentResult("NEGATIVE", 0.10, 0.85, 0.05, 0.85)
            else:
                return SentimentResult("NEUTRAL", 0.30, 0.30, 0.40, 0.50)

    analyzer = MockAnalyzer()
    generator = LLMSignalGenerator(analyzer)

    # Test signals
    test_items = [
        {"text": "Apple reports record iPhone sales", "symbol": "AAPL", "source_type": "earnings"},
        {"text": "Tesla stock plunges on delivery miss", "symbol": "TSLA", "source_type": "news"},
        {"text": "Microsoft maintains quarterly guidance", "symbol": "MSFT", "source_type": "earnings"},
    ]

    print("Trading Signal Generation Demo\n" + "=" * 50)

    signals = generator.generate_signals_batch(test_items)
    for sig in signals:
        print(f"\n{sig.symbol}: {sig.signal_type.value}")
        print(f"  Signal: {sig.signal:+.2f}")
        print(f"  Confidence: {sig.confidence:.2%}")
        print(f"  {sig.reasoning}")

    # Aggregate
    positions = generator.aggregate_signals(signals)
    print("\nAggregated Positions:")
    for symbol, pos in positions.items():
        direction = "LONG" if pos > 0 else "SHORT" if pos < 0 else "FLAT"
        print(f"  {symbol}: {direction} ({pos:+.2f})")
