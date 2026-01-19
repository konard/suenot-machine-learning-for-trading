"""
FinGPT Trading Signal Generation Module

Generate trading signals from financial news using FinGPT sentiment analysis.
"""

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class TradingSignal:
    """Trading signal generated from news analysis."""
    timestamp: datetime
    symbol: str
    signal: float  # -1 (strong sell) to 1 (strong buy)
    confidence: float
    source: str
    news_text: str

    @property
    def action(self) -> str:
        """Get recommended action based on signal strength."""
        if self.signal > 0.5:
            return "STRONG_BUY"
        elif self.signal > 0.2:
            return "BUY"
        elif self.signal < -0.5:
            return "STRONG_SELL"
        elif self.signal < -0.2:
            return "SELL"
        return "HOLD"


@dataclass
class AggregatedSignal:
    """Aggregated trading signal from multiple news items."""
    symbol: str
    action: str  # buy, sell, hold
    strength: float  # -1 to 1
    confidence: float  # 0 to 1
    reason: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class FinGPTTradingEngine:
    """
    Generate and manage trading signals using FinGPT.

    This engine combines sentiment analysis with signal generation
    and aggregation for actionable trading decisions.

    Examples:
        >>> from fingpt_sentiment import FinGPTSentimentAnalyzer
        >>> analyzer = FinGPTSentimentAnalyzer()
        >>> engine = FinGPTTradingEngine(analyzer)
        >>> signal = engine.generate_signal("Apple beats earnings", "AAPL")
        >>> print(f"Signal: {signal.action}")
    """

    def __init__(
        self,
        sentiment_analyzer,
        signal_threshold: float = 0.3,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the trading engine.

        Args:
            sentiment_analyzer: FinGPTSentimentAnalyzer instance
            signal_threshold: Minimum signal strength to generate signal
            confidence_threshold: Minimum confidence to generate signal
        """
        self.analyzer = sentiment_analyzer
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold

        # Source importance weights
        self.source_weights = {
            "earnings": 1.0,     # Highest weight for earnings news
            "sec_filing": 0.9,  # SEC filings are reliable
            "news_wire": 0.8,   # Major news sources
            "reuters": 0.9,     # Reuters news
            "bloomberg": 0.9,   # Bloomberg news
            "analyst": 0.7,     # Analyst reports
            "twitter": 0.4,     # Social media (noisy)
            "social": 0.4,      # Social media (noisy)
        }

        # Sentiment to signal mapping
        self.sentiment_to_signal = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0
        }

    def generate_signal(
        self,
        symbol_or_text: str,
        news_items_or_symbol: any = None,
        source: str = "news_wire",
        timestamp: Optional[datetime] = None
    ) -> Optional[any]:
        """
        Generate trading signal from news.

        This method supports two calling patterns:
        1. generate_signal(symbol, news_items) - aggregates signals from list of news
        2. generate_signal(text, symbol, source) - single news item analysis

        Args:
            symbol_or_text: Either symbol (if news_items follows) or news text
            news_items_or_symbol: Either list of news dicts or symbol string
            source: Source type for weighting (only for single news)
            timestamp: News timestamp (only for single news)

        Returns:
            AggregatedSignal or TradingSignal depending on call pattern
        """
        # Check if this is an aggregated call (symbol, news_items)
        if isinstance(news_items_or_symbol, list):
            return self._generate_aggregated_signal(symbol_or_text, news_items_or_symbol)

        # Otherwise it's a single news call (text, symbol, source)
        news_text = symbol_or_text
        symbol = news_items_or_symbol if news_items_or_symbol else "UNKNOWN"
        timestamp = timestamp or datetime.now()

        # Analyze sentiment
        result = self.analyzer.analyze(news_text)

        # Check confidence threshold
        if result.confidence < self.confidence_threshold:
            return None

        # Calculate weighted signal
        base_signal = self.sentiment_to_signal.get(result.sentiment, 0.0)
        source_weight = self.source_weights.get(source, 0.5)
        signal = base_signal * result.confidence * source_weight

        # Check signal threshold
        if abs(signal) < self.signal_threshold:
            return None

        return TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal=signal,
            confidence=result.confidence,
            source=source,
            news_text=news_text
        )

    def _generate_aggregated_signal(
        self,
        symbol: str,
        news_items: List[Dict]
    ) -> AggregatedSignal:
        """
        Generate an aggregated signal from multiple news items.

        Args:
            symbol: Stock/crypto symbol
            news_items: List of dicts with 'text' and 'source' keys

        Returns:
            AggregatedSignal with aggregated sentiment
        """
        if not news_items:
            return AggregatedSignal(
                symbol=symbol,
                action="hold",
                strength=0.0,
                confidence=0.5,
                reason="No news data available"
            )

        weighted_score = 0.0
        total_weight = 0.0
        reasons = []

        for item in news_items:
            text = item.get("text", "")
            source = item.get("source", "news_wire").lower()

            result = self.analyzer.analyze(text)
            weight = self.source_weights.get(source, 0.5)

            base_signal = self.sentiment_to_signal.get(result.sentiment, 0.0)
            weighted_score += base_signal * weight * result.confidence
            total_weight += weight

            reasons.append(f"{source}: {result.sentiment}")

        avg_strength = weighted_score / total_weight if total_weight > 0 else 0.0

        if avg_strength > 0.3:
            action = "buy"
        elif avg_strength < -0.3:
            action = "sell"
        else:
            action = "hold"

        confidence = min(abs(avg_strength) + 0.5, 1.0)
        reason = f"Aggregated from {len(news_items)} sources: {', '.join(reasons[:3])}"

        return AggregatedSignal(
            symbol=symbol,
            action=action,
            strength=avg_strength,
            confidence=confidence,
            reason=reason
        )

    def generate_signals_batch(
        self,
        news_items: List[Dict]
    ) -> List[TradingSignal]:
        """
        Generate signals from multiple news items.

        Args:
            news_items: List of dicts with 'text', 'symbol', 'source' keys

        Returns:
            List of generated TradingSignal objects
        """
        signals = []
        for item in news_items:
            signal = self.generate_signal(
                news_text=item.get("text", ""),
                symbol=item.get("symbol", ""),
                source=item.get("source", "news_wire"),
                timestamp=item.get("timestamp")
            )
            if signal:
                signals.append(signal)
        return signals

    def aggregate_signals(
        self,
        signals: List[TradingSignal],
        decay_hours: float = 24.0
    ) -> Dict[str, float]:
        """
        Aggregate multiple signals with time decay.

        Args:
            signals: List of trading signals
            decay_hours: Half-life for signal decay

        Returns:
            Dict mapping symbols to aggregated position scores
        """
        now = datetime.now()
        positions = {}

        # Group by symbol
        symbol_signals: Dict[str, List[TradingSignal]] = {}
        for sig in signals:
            if sig.symbol not in symbol_signals:
                symbol_signals[sig.symbol] = []
            symbol_signals[sig.symbol].append(sig)

        # Aggregate with exponential time decay
        for symbol, sigs in symbol_signals.items():
            weighted_sum = 0.0
            weight_sum = 0.0

            for sig in sigs:
                hours_old = (now - sig.timestamp).total_seconds() / 3600
                decay = np.exp(-hours_old / decay_hours)

                weight = sig.confidence * decay
                weighted_sum += sig.signal * weight
                weight_sum += weight

            if weight_sum > 0:
                positions[symbol] = np.clip(weighted_sum / weight_sum, -1, 1)

        return positions

    def get_recommendations(
        self,
        positions: Dict[str, float]
    ) -> List[Dict]:
        """
        Convert position scores to recommendations.

        Args:
            positions: Dict of symbol -> position score

        Returns:
            List of recommendations with actions
        """
        recommendations = []

        for symbol, score in sorted(
            positions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            if score > 0.5:
                action = "STRONG_BUY"
            elif score > 0.2:
                action = "BUY"
            elif score < -0.5:
                action = "STRONG_SELL"
            elif score < -0.2:
                action = "SELL"
            else:
                action = "HOLD"

            recommendations.append({
                "symbol": symbol,
                "score": score,
                "action": action,
                "confidence": abs(score)
            })

        return recommendations


if __name__ == "__main__":
    from fingpt_sentiment import FinGPTSentimentAnalyzer

    # Demo
    analyzer = FinGPTSentimentAnalyzer(use_mock=True)
    engine = FinGPTTradingEngine(analyzer)

    news_items = [
        {
            "text": "Apple reports record iPhone sales in China, shares rally 5%",
            "symbol": "AAPL",
            "source": "earnings"
        },
        {
            "text": "Apple faces antitrust investigation in EU over App Store",
            "symbol": "AAPL",
            "source": "news_wire"
        },
        {
            "text": "NVIDIA AI chips dominate data center market, revenue beats estimates",
            "symbol": "NVDA",
            "source": "earnings"
        },
        {
            "text": "Tesla delivery numbers miss expectations, stock drops 8%",
            "symbol": "TSLA",
            "source": "earnings"
        },
        {
            "text": "Bitcoin surges past $50,000 on ETF approval optimism",
            "symbol": "BTC",
            "source": "news_wire"
        },
    ]

    print("=" * 70)
    print("FinGPT Trading Signal Generation Demo")
    print("=" * 70)

    signals = []
    for item in news_items:
        signal = engine.generate_signal(
            news_text=item["text"],
            symbol=item["symbol"],
            source=item["source"]
        )

        if signal:
            signals.append(signal)
            print(f"\n{signal.symbol}:")
            print(f"  News: {signal.news_text[:50]}...")
            print(f"  Signal: {signal.signal:+.2f}")
            print(f"  Action: {signal.action}")
            print(f"  Confidence: {signal.confidence:.1%}")

    # Aggregate signals
    print("\n" + "=" * 70)
    print("Aggregated Recommendations:")
    print("=" * 70)

    positions = engine.aggregate_signals(signals)
    recommendations = engine.get_recommendations(positions)

    for rec in recommendations:
        print(f"\n{rec['symbol']}:")
        print(f"  Action: {rec['action']}")
        print(f"  Score: {rec['score']:+.2f}")
