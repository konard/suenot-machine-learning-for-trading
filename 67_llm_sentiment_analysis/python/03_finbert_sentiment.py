"""
Chapter 67: LLM Sentiment Analysis for Trading
03_finbert_sentiment.py - FinBERT-based sentiment analysis

This module demonstrates how to use FinBERT, a BERT model fine-tuned
on financial text, for sentiment analysis of financial news and reports.

Requirements:
    pip install transformers torch
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to import transformers, provide mock if not available
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers/torch not installed. Using mock implementation.")


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    text: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    scores: Dict[str, float]  # Score for each class


class FinBERTAnalyzer:
    """
    Sentiment analyzer using FinBERT model.

    FinBERT is a pre-trained NLP model specifically designed for
    financial sentiment analysis. It was trained on financial news,
    earnings reports, and analyst reports.

    Model: ProsusAI/finbert
    - 3 classes: positive, negative, neutral
    - Trained on Financial PhraseBank dataset
    - Based on BERT architecture
    """

    MODEL_NAME = "ProsusAI/finbert"
    LABELS = ['negative', 'neutral', 'positive']

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the FinBERT analyzer.

        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device

        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.info("Running in mock mode - install transformers for real inference")
            self.tokenizer = None
            self.model = None

    def _load_model(self):
        """Load the FinBERT model and tokenizer."""
        logger.info(f"Loading FinBERT model: {self.MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)

        # Set device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded on device: {self.device}")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Financial text to analyze

        Returns:
            SentimentResult with sentiment label and scores
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return self._mock_analyze(text)

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Extract results
        probs_list = probs[0].cpu().tolist()
        scores = {label: prob for label, prob in zip(self.LABELS, probs_list)}

        sentiment = max(scores, key=scores.get)
        confidence = scores[sentiment]

        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            scores=scores
        )

    def _mock_analyze(self, text: str) -> SentimentResult:
        """Mock analysis for demonstration when transformers not available."""
        # Simple rule-based mock for demonstration
        text_lower = text.lower()

        positive_words = {'surge', 'rise', 'gain', 'beat', 'profit', 'growth', 'bullish', 'rally'}
        negative_words = {'fall', 'drop', 'loss', 'miss', 'crash', 'bearish', 'decline', 'plunge'}

        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)

        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.7 + positive_count * 0.1, 0.95)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.7 + negative_count * 0.1, 0.95)
        else:
            sentiment = 'neutral'
            confidence = 0.6

        scores = {
            'positive': confidence if sentiment == 'positive' else (1 - confidence) / 2,
            'negative': confidence if sentiment == 'negative' else (1 - confidence) / 2,
            'neutral': confidence if sentiment == 'neutral' else (1 - confidence) / 2,
        }

        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            scores=scores
        )

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts efficiently.

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once

        Returns:
            List of SentimentResult objects
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.analyze(text) for text in batch]
            results.extend(batch_results)

            if len(texts) > batch_size:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

        return results


class SentimentAggregator:
    """
    Aggregates sentiment scores from multiple texts.

    Useful for combining sentiment from multiple news articles
    or social media posts about the same asset.
    """

    def __init__(self, decay_factor: float = 0.9):
        """
        Initialize the aggregator.

        Args:
            decay_factor: Factor for time-based decay (newer = higher weight)
        """
        self.decay_factor = decay_factor

    def aggregate(
        self,
        results: List[SentimentResult],
        method: str = 'weighted_average'
    ) -> Dict[str, Any]:
        """
        Aggregate multiple sentiment results.

        Args:
            results: List of SentimentResult objects
            method: Aggregation method ('simple_average', 'weighted_average', 'majority_vote')

        Returns:
            Dictionary with aggregated sentiment
        """
        if not results:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'count': 0}

        if method == 'simple_average':
            return self._simple_average(results)
        elif method == 'weighted_average':
            return self._weighted_average(results)
        elif method == 'majority_vote':
            return self._majority_vote(results)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _simple_average(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Simple average of scores."""
        avg_scores = {'positive': 0, 'negative': 0, 'neutral': 0}

        for result in results:
            for label, score in result.scores.items():
                avg_scores[label] += score

        for label in avg_scores:
            avg_scores[label] /= len(results)

        sentiment = max(avg_scores, key=avg_scores.get)

        return {
            'sentiment': sentiment,
            'confidence': avg_scores[sentiment],
            'scores': avg_scores,
            'count': len(results),
            'method': 'simple_average'
        }

    def _weighted_average(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Weighted average based on confidence."""
        weighted_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_weight = 0

        for i, result in enumerate(results):
            # Apply time decay (most recent first)
            time_weight = self.decay_factor ** i
            confidence_weight = result.confidence
            weight = time_weight * confidence_weight

            for label, score in result.scores.items():
                weighted_scores[label] += score * weight

            total_weight += weight

        if total_weight > 0:
            for label in weighted_scores:
                weighted_scores[label] /= total_weight

        sentiment = max(weighted_scores, key=weighted_scores.get)

        return {
            'sentiment': sentiment,
            'confidence': weighted_scores[sentiment],
            'scores': weighted_scores,
            'count': len(results),
            'method': 'weighted_average'
        }

    def _majority_vote(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Majority vote of sentiment labels."""
        votes = {'positive': 0, 'negative': 0, 'neutral': 0}

        for result in results:
            votes[result.sentiment] += 1

        sentiment = max(votes, key=votes.get)
        confidence = votes[sentiment] / len(results)

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'votes': votes,
            'count': len(results),
            'method': 'majority_vote'
        }


def main():
    """Demonstrate FinBERT sentiment analysis."""

    print("=" * 60)
    print("LLM Sentiment Analysis - FinBERT Demo")
    print("=" * 60)

    # Initialize analyzer
    analyzer = FinBERTAnalyzer()

    # Example financial texts
    texts = [
        "Apple reported record quarterly revenue, beating analyst expectations by 15%.",
        "Tesla stock plunges 8% after disappointing delivery numbers.",
        "The Federal Reserve kept interest rates unchanged as expected.",
        "Bitcoin surges past $50,000 amid growing institutional adoption.",
        "Oil prices fall sharply on global demand concerns and oversupply fears.",
        "Microsoft announces major AI partnership, shares rally 5% in pre-market.",
        "Inflation data comes in higher than expected, spooking bond markets.",
        "Ethereum network upgrade completed successfully without issues.",
    ]

    print("\nAnalyzing individual texts:")
    print("-" * 40)

    results = []
    for text in texts:
        result = analyzer.analyze(text)
        results.append(result)

        print(f"\nText: {text[:60]}...")
        print(f"  Sentiment: {result.sentiment.upper()}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Scores: pos={result.scores['positive']:.2f}, "
              f"neg={result.scores['negative']:.2f}, "
              f"neu={result.scores['neutral']:.2f}")

    # Aggregate sentiment for crypto-related texts
    print("\n" + "=" * 60)
    print("Aggregating Crypto Sentiment")
    print("=" * 60)

    crypto_texts = [
        "Bitcoin surges past $50,000 amid institutional buying",
        "Ethereum hits new all-time high as DeFi activity grows",
        "Crypto market sees $500M in liquidations during flash crash",
        "Major exchange announces new Bitcoin ETF listing",
    ]

    crypto_results = analyzer.analyze_batch(crypto_texts)
    aggregator = SentimentAggregator()

    for method in ['simple_average', 'weighted_average', 'majority_vote']:
        agg = aggregator.aggregate(crypto_results, method=method)
        print(f"\n{method}:")
        print(f"  Sentiment: {agg['sentiment'].upper()}")
        print(f"  Confidence: {agg['confidence']:.2%}")

    # Example: Trading signal generation
    print("\n" + "=" * 60)
    print("Trading Signal Example")
    print("=" * 60)

    agg_result = aggregator.aggregate(crypto_results, method='weighted_average')

    if agg_result['sentiment'] == 'positive' and agg_result['confidence'] > 0.6:
        signal = "BULLISH - Consider LONG position"
    elif agg_result['sentiment'] == 'negative' and agg_result['confidence'] > 0.6:
        signal = "BEARISH - Consider SHORT position"
    else:
        signal = "NEUTRAL - No clear signal"

    print(f"Aggregated sentiment: {agg_result['sentiment'].upper()}")
    print(f"Confidence: {agg_result['confidence']:.2%}")
    print(f"Trading Signal: {signal}")


if __name__ == "__main__":
    main()
