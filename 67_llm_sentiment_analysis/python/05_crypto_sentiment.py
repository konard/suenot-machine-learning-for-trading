"""
Chapter 67: LLM Sentiment Analysis for Trading
05_crypto_sentiment.py - Cryptocurrency-specific sentiment analysis

This module focuses on sentiment analysis for cryptocurrency markets,
which are particularly sensitive to news and social media sentiment.

Features:
- Crypto-specific terminology handling
- Social media sentiment aggregation
- On-chain metrics integration
- Fear & Greed index correlation
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoSentimentLevel(Enum):
    """Sentiment levels for crypto markets."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class CryptoSentimentResult:
    """Result of crypto-specific sentiment analysis."""
    symbol: str
    sentiment_score: float  # -1 to 1
    sentiment_level: CryptoSentimentLevel
    confidence: float
    sources: Dict[str, float]  # Source -> sentiment score
    key_topics: List[str]
    whale_activity: str
    social_volume: int
    timestamp: datetime


class CryptoTerminologyProcessor:
    """
    Processes crypto-specific terminology for sentiment analysis.

    Crypto communities use unique slang that standard sentiment
    models may not understand correctly.
    """

    # Crypto slang with sentiment mappings
    CRYPTO_SLANG = {
        # Bullish terms
        'moon': ('positive', 0.9),
        'mooning': ('positive', 0.95),
        'hodl': ('positive', 0.6),
        'diamond hands': ('positive', 0.7),
        'bullish': ('positive', 0.8),
        'pump': ('positive', 0.7),
        'ath': ('positive', 0.8),  # All-time high
        'fomo': ('positive', 0.5),  # Can be negative context
        'wagmi': ('positive', 0.7),  # We're all gonna make it

        # Bearish terms
        'rekt': ('negative', 0.9),
        'dump': ('negative', 0.8),
        'bearish': ('negative', 0.8),
        'rug': ('negative', 0.95),  # Rug pull
        'rug pull': ('negative', 0.95),
        'scam': ('negative', 0.9),
        'fud': ('negative', 0.6),  # Fear, uncertainty, doubt
        'paper hands': ('negative', 0.5),
        'ngmi': ('negative', 0.7),  # Not gonna make it
        'crash': ('negative', 0.85),

        # Neutral/context-dependent
        'dyor': ('neutral', 0.0),  # Do your own research
        'nfa': ('neutral', 0.0),  # Not financial advice
        'dca': ('neutral', 0.0),  # Dollar cost averaging
        'defi': ('neutral', 0.0),
        'dex': ('neutral', 0.0),
        'cex': ('neutral', 0.0),
    }

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text for crypto terminology.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with slang analysis
        """
        text_lower = text.lower()
        found_terms = []
        sentiment_scores = []

        for term, (sentiment, score) in self.CRYPTO_SLANG.items():
            if term in text_lower:
                found_terms.append(term)
                if sentiment == 'positive':
                    sentiment_scores.append(score)
                elif sentiment == 'negative':
                    sentiment_scores.append(-score)

        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

        return {
            'found_terms': found_terms,
            'slang_sentiment_score': avg_score,
            'term_count': len(found_terms)
        }


class SocialMediaSentimentAggregator:
    """
    Aggregates sentiment from multiple social media sources.

    Weights different sources based on their reliability and
    relevance for crypto markets.
    """

    # Source weights (higher = more influential)
    SOURCE_WEIGHTS = {
        'twitter': 1.0,
        'reddit': 0.9,
        'telegram': 0.7,
        'discord': 0.6,
        'stocktwits': 0.5,
        'news': 1.2,
    }

    def aggregate(
        self,
        source_sentiments: Dict[str, Tuple[float, int]]  # source -> (score, count)
    ) -> Dict[str, Any]:
        """
        Aggregate sentiments from multiple sources.

        Args:
            source_sentiments: Dictionary of source -> (sentiment_score, post_count)

        Returns:
            Aggregated sentiment result
        """
        weighted_sum = 0.0
        total_weight = 0.0
        total_volume = 0

        for source, (score, count) in source_sentiments.items():
            weight = self.SOURCE_WEIGHTS.get(source, 0.5) * count
            weighted_sum += score * weight
            total_weight += weight
            total_volume += count

        aggregated_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {
            'aggregated_score': aggregated_score,
            'total_volume': total_volume,
            'source_breakdown': source_sentiments,
            'dominant_source': max(source_sentiments.keys(),
                                    key=lambda s: source_sentiments[s][1])
        }


class CryptoSentimentAnalyzer:
    """
    Comprehensive sentiment analyzer for cryptocurrency markets.

    Combines:
    - LLM-based text analysis
    - Crypto terminology processing
    - Social media aggregation
    - Market metrics
    """

    def __init__(self):
        self.terminology_processor = CryptoTerminologyProcessor()
        self.social_aggregator = SocialMediaSentimentAggregator()

    def analyze(
        self,
        symbol: str,
        texts: List[str],
        source_labels: Optional[List[str]] = None
    ) -> CryptoSentimentResult:
        """
        Analyze sentiment for a cryptocurrency.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            texts: List of texts to analyze
            source_labels: Optional source labels for each text

        Returns:
            CryptoSentimentResult with comprehensive analysis
        """
        if source_labels is None:
            source_labels = ['unknown'] * len(texts)

        # Analyze each text
        text_sentiments = []
        source_data = {}
        all_topics = []

        for text, source in zip(texts, source_labels):
            # Process crypto terminology
            slang_result = self.terminology_processor.process_text(text)

            # Simple sentiment analysis (in production, use FinBERT/GPT)
            sentiment_score = self._simple_sentiment(text)

            # Combine with slang analysis
            combined_score = (sentiment_score + slang_result['slang_sentiment_score']) / 2
            combined_score = max(-1.0, min(1.0, combined_score))

            text_sentiments.append(combined_score)

            # Aggregate by source
            if source not in source_data:
                source_data[source] = (0.0, 0)
            prev_score, prev_count = source_data[source]
            source_data[source] = (
                (prev_score * prev_count + combined_score) / (prev_count + 1),
                prev_count + 1
            )

            # Collect topics
            all_topics.extend(slang_result['found_terms'])

        # Aggregate sources
        aggregated = self.social_aggregator.aggregate(source_data)

        # Determine sentiment level
        final_score = aggregated['aggregated_score']
        sentiment_level = self._score_to_level(final_score)

        # Calculate confidence based on volume and agreement
        score_std = self._calculate_std(text_sentiments)
        confidence = max(0.3, min(0.95, 0.8 - score_std))

        return CryptoSentimentResult(
            symbol=symbol,
            sentiment_score=final_score,
            sentiment_level=sentiment_level,
            confidence=confidence,
            sources=source_data,
            key_topics=list(set(all_topics))[:10],
            whale_activity=self._mock_whale_activity(symbol),
            social_volume=aggregated['total_volume'],
            timestamp=datetime.now()
        )

    def _simple_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment for demonstration."""
        text_lower = text.lower()

        positive_words = {'bullish', 'moon', 'pump', 'surge', 'rally', 'buy', 'long', 'up', 'green'}
        negative_words = {'bearish', 'crash', 'dump', 'sell', 'short', 'down', 'red', 'scam'}

        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count + neg_count == 0:
            return 0.0

        return (pos_count - neg_count) / (pos_count + neg_count)

    def _score_to_level(self, score: float) -> CryptoSentimentLevel:
        """Convert numeric score to sentiment level."""
        if score < -0.5:
            return CryptoSentimentLevel.EXTREME_FEAR
        elif score < -0.2:
            return CryptoSentimentLevel.FEAR
        elif score < 0.2:
            return CryptoSentimentLevel.NEUTRAL
        elif score < 0.5:
            return CryptoSentimentLevel.GREED
        else:
            return CryptoSentimentLevel.EXTREME_GREED

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _mock_whale_activity(self, symbol: str) -> str:
        """Mock whale activity indicator."""
        activities = ['accumulating', 'distributing', 'neutral']
        return random.choice(activities)


class FearGreedCorrelator:
    """
    Correlates sentiment with Fear & Greed index.

    The Crypto Fear & Greed Index is a popular metric that combines
    multiple factors to gauge market sentiment.
    """

    def __init__(self):
        # Mock historical data
        self.historical_index = []

    def get_current_index(self) -> Dict[str, Any]:
        """Get current Fear & Greed index (mock)."""
        # In production, fetch from actual API
        value = random.randint(20, 80)

        if value < 25:
            classification = "Extreme Fear"
        elif value < 45:
            classification = "Fear"
        elif value < 55:
            classification = "Neutral"
        elif value < 75:
            classification = "Greed"
        else:
            classification = "Extreme Greed"

        return {
            'value': value,
            'classification': classification,
            'timestamp': datetime.now().isoformat()
        }

    def compare_with_sentiment(
        self,
        sentiment_result: CryptoSentimentResult
    ) -> Dict[str, Any]:
        """
        Compare LLM sentiment with Fear & Greed index.

        Args:
            sentiment_result: CryptoSentimentResult from analyzer

        Returns:
            Comparison analysis
        """
        fg_index = self.get_current_index()

        # Convert sentiment to 0-100 scale for comparison
        sentiment_normalized = (sentiment_result.sentiment_score + 1) * 50

        difference = abs(sentiment_normalized - fg_index['value'])

        if difference < 15:
            agreement = "Strong agreement"
        elif difference < 30:
            agreement = "Moderate agreement"
        else:
            agreement = "Divergence detected"

        return {
            'fear_greed_index': fg_index,
            'llm_sentiment_normalized': sentiment_normalized,
            'difference': difference,
            'agreement_level': agreement,
            'analysis': self._generate_analysis(
                sentiment_result.sentiment_level,
                fg_index['classification'],
                agreement
            )
        }

    def _generate_analysis(
        self,
        sentiment_level: CryptoSentimentLevel,
        fg_classification: str,
        agreement: str
    ) -> str:
        """Generate analysis text."""
        if agreement == "Strong agreement":
            return (f"LLM sentiment ({sentiment_level.value}) aligns with "
                   f"Fear & Greed Index ({fg_classification}). "
                   f"Market consensus appears clear.")
        elif agreement == "Divergence detected":
            return (f"Divergence between LLM sentiment ({sentiment_level.value}) "
                   f"and Fear & Greed Index ({fg_classification}). "
                   f"This may indicate a potential trend reversal or "
                   f"local sentiment anomaly worth investigating.")
        else:
            return (f"Moderate alignment between indicators. "
                   f"Continue monitoring for clearer signals.")


def main():
    """Demonstrate crypto-specific sentiment analysis."""

    print("=" * 60)
    print("LLM Sentiment Analysis - Crypto Market Demo")
    print("=" * 60)

    # Initialize analyzer
    analyzer = CryptoSentimentAnalyzer()
    correlator = FearGreedCorrelator()

    # Example crypto-related texts
    texts = [
        "$BTC mooning! Diamond hands holding strong through the dip. WAGMI!",
        "Just bought more ETH on this dip. This is what DCA is all about.",
        "Warning: Possible rug pull on new DeFi protocol. DYOR before investing!",
        "Bitcoin breaks $50k resistance. Institutional FOMO is real.",
        "Paper hands getting rekt during this correction. Stay calm and HODL.",
        "Massive whale accumulation detected on-chain. Bullish signal!",
        "Market looking bearish. Could see more dump before recovery.",
        "NFT trading volume down 80%. Bear market vibes.",
    ]

    sources = ['twitter', 'reddit', 'twitter', 'news',
               'reddit', 'telegram', 'twitter', 'news']

    # Analyze Bitcoin sentiment
    print("\n=== Bitcoin Sentiment Analysis ===")
    btc_result = analyzer.analyze("BTC", texts[:5], sources[:5])

    print(f"\nSymbol: {btc_result.symbol}")
    print(f"Sentiment Score: {btc_result.sentiment_score:.2f}")
    print(f"Sentiment Level: {btc_result.sentiment_level.value}")
    print(f"Confidence: {btc_result.confidence:.0%}")
    print(f"Social Volume: {btc_result.social_volume} posts")
    print(f"Whale Activity: {btc_result.whale_activity}")
    print(f"Key Topics: {', '.join(btc_result.key_topics)}")

    # Source breakdown
    print("\nSource Breakdown:")
    for source, (score, count) in btc_result.sources.items():
        print(f"  {source}: {score:.2f} ({count} posts)")

    # Compare with Fear & Greed
    print("\n=== Fear & Greed Correlation ===")
    comparison = correlator.compare_with_sentiment(btc_result)

    print(f"\nFear & Greed Index: {comparison['fear_greed_index']['value']} "
          f"({comparison['fear_greed_index']['classification']})")
    print(f"LLM Sentiment (normalized): {comparison['llm_sentiment_normalized']:.0f}")
    print(f"Agreement: {comparison['agreement_level']}")
    print(f"\nAnalysis: {comparison['analysis']}")

    # Trading signal
    print("\n=== Trading Signal ===")
    score = btc_result.sentiment_score
    confidence = btc_result.confidence

    if score > 0.3 and confidence > 0.6:
        signal = "BULLISH - Consider LONG position"
        size = "Medium" if score < 0.5 else "Large"
    elif score < -0.3 and confidence > 0.6:
        signal = "BEARISH - Consider SHORT position or reduce exposure"
        size = "Medium" if score > -0.5 else "Large"
    else:
        signal = "NEUTRAL - No clear directional signal"
        size = "Small or None"

    print(f"Signal: {signal}")
    print(f"Suggested Position Size: {size}")


if __name__ == "__main__":
    main()
