"""
FinGPT Financial Sentiment Analysis Module

This module provides sentiment analysis capabilities for financial text,
using FinGPT-style approaches with open-source models.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment: str  # positive, negative, neutral
    confidence: float
    score: float  # -1 to 1

    @property
    def label(self) -> str:
        """Alias for sentiment for compatibility."""
        return self.sentiment


class FinGPTSentimentAnalyzer:
    """
    Financial sentiment analyzer using FinGPT approach.

    This class provides both mock and real implementations
    for demonstration and production use.

    Examples:
        >>> analyzer = FinGPTSentimentAnalyzer()
        >>> result = analyzer.analyze("Apple reports record earnings")
        >>> print(f"Sentiment: {result.sentiment} ({result.confidence:.1%})")
    """

    def __init__(
        self,
        use_mock: bool = True,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the analyzer.

        Args:
            use_mock: If True, use rule-based mock for demos
            model_path: Path to FinGPT model weights
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.use_mock = use_mock
        self.model = None
        self.tokenizer = None
        self.device = device

        if not use_mock and model_path:
            self._load_model(model_path)

        # Financial sentiment keywords
        self.positive_keywords = {
            "beat", "beats", "beating", "exceed", "exceeds", "exceeded",
            "surge", "surges", "surged", "soar", "soars", "soared",
            "record", "growth", "grew", "profit", "profitable",
            "gain", "gains", "gained", "bullish", "upgrade", "upgrades",
            "outperform", "outperforms", "strong", "stronger", "strongest",
            "positive", "optimistic", "rally", "rallies", "rallied",
            "boom", "booming", "rise", "rises", "rising", "rose",
            "increase", "increases", "increased", "improving", "improved"
        }
        self.negative_keywords = {
            "miss", "misses", "missed", "decline", "declines", "declined",
            "drop", "drops", "dropped", "plunge", "plunges", "plunged",
            "loss", "losses", "fall", "falls", "fell", "falling",
            "bearish", "downgrade", "downgrades", "downgraded",
            "underperform", "underperforms", "weak", "weaker", "weakest",
            "negative", "pessimistic", "crash", "crashes", "crashed",
            "slump", "slumps", "slumped", "warning", "warnings",
            "concern", "concerns", "concerned", "worries", "worried",
            "layoff", "layoffs", "cut", "cuts", "cutting",
            "decrease", "decreases", "decreased", "disappointing"
        }

    def _load_model(self, model_path: str):
        """Load the actual FinGPT model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device == "cpu":
                self.model.to(self.device)

        except ImportError:
            print("Warning: transformers library required. Using mock mode.")
            self.use_mock = True
        except Exception as e:
            print(f"Warning: Could not load model, using mock: {e}")
            self.use_mock = True

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of financial text.

        Args:
            text: Financial news or document

        Returns:
            SentimentResult with sentiment label and confidence
        """
        if self.use_mock:
            return self._mock_analyze(text)
        else:
            return self._model_analyze(text)

    def _mock_analyze(self, text: str) -> SentimentResult:
        """Rule-based mock analysis for demonstration."""
        text_lower = text.lower()

        pos_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in self.negative_keywords if kw in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return SentimentResult(
                sentiment="neutral",
                confidence=0.5,
                score=0.0
            )

        pos_ratio = pos_count / total
        neg_ratio = neg_count / total

        if pos_ratio > 0.6:
            sentiment = "positive"
            score = pos_ratio
            confidence = min(0.5 + pos_count * 0.1, 0.95)
        elif neg_ratio > 0.6:
            sentiment = "negative"
            score = -neg_ratio
            confidence = min(0.5 + neg_count * 0.1, 0.95)
        else:
            sentiment = "neutral"
            score = pos_ratio - neg_ratio
            confidence = 0.6

        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            score=score
        )

    def _model_analyze(self, text: str) -> SentimentResult:
        """Analyze using the actual FinGPT model."""
        import torch

        prompt = f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.

Input: {text}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment = self._extract_sentiment(response)
        score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}[sentiment]

        return SentimentResult(
            sentiment=sentiment,
            confidence=0.85,
            score=score
        )

    def _extract_sentiment(self, response: str) -> str:
        """Extract sentiment label from model response."""
        response_lower = response.lower()
        if "positive" in response_lower:
            return "positive"
        elif "negative" in response_lower:
            return "negative"
        return "neutral"

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple texts.

        Args:
            texts: List of financial texts

        Returns:
            List of SentimentResult objects
        """
        return [self.analyze(text) for text in texts]

    def analyze_aspects(
        self,
        text: str,
        entities: List[str]
    ) -> Dict[str, SentimentResult]:
        """
        Analyze sentiment toward specific entities (aspect-based).

        This mimics FinGPT's aspect-specific sentiment capability
        by extracting sentences mentioning each entity and analyzing them.

        Args:
            text: Full text to analyze
            entities: List of entity names to analyze sentiment for

        Returns:
            Dict mapping entity names to their sentiment results
        """
        results = {}
        sentences = self._split_sentences(text)

        for entity in entities:
            entity_sentences = [
                s for s in sentences
                if entity.lower() in s.lower()
            ]

            if entity_sentences:
                entity_text = ' '.join(entity_sentences)
                results[entity] = self.analyze(entity_text)
            else:
                results[entity] = SentimentResult(
                    sentiment="not_mentioned",
                    confidence=0.0,
                    score=0.0
                )

        return results

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def analyze_sentiment(
    text: str,
    use_mock: bool = True,
    model_path: Optional[str] = None
) -> SentimentResult:
    """
    Convenience function for quick sentiment analysis.

    Args:
        text: Financial text to analyze
        use_mock: Whether to use mock analyzer
        model_path: Path to model if not using mock

    Returns:
        SentimentResult
    """
    analyzer = FinGPTSentimentAnalyzer(use_mock=use_mock, model_path=model_path)
    return analyzer.analyze(text)


if __name__ == "__main__":
    # Demo usage
    analyzer = FinGPTSentimentAnalyzer(use_mock=True)

    test_texts = [
        "Apple reported record quarterly revenue of $123.9 billion, beating analyst expectations.",
        "Tesla shares plunged 12% after disappointing delivery numbers.",
        "The Federal Reserve kept interest rates unchanged at the latest meeting.",
        "NVIDIA stock surged to new all-time highs amid strong AI chip demand.",
        "Bitcoin dropped below $40,000 as regulatory concerns mount.",
    ]

    print("=" * 70)
    print("FinGPT Financial Sentiment Analysis Demo")
    print("=" * 70)

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text[:60]}...")
        print(f"Sentiment: {result.sentiment.upper()}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Score: {result.score:+.2f}")
