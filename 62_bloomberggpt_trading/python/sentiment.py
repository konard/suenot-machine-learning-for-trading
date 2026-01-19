"""
Financial Sentiment Analysis Module

This module provides sentiment analysis capabilities for financial text,
inspired by BloombergGPT's aspect-specific sentiment analysis.
"""

import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import re


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    label: str  # POSITIVE, NEGATIVE, NEUTRAL
    positive_prob: float
    negative_prob: float
    neutral_prob: float
    confidence: float

    @property
    def score(self) -> float:
        """Return a score from -1 (negative) to 1 (positive)."""
        return self.positive_prob - self.negative_prob


class FinancialSentimentAnalyzer:
    """
    Financial sentiment analyzer using LLM-based approach.

    Since BloombergGPT is not publicly available, we use open-source
    alternatives (FinBERT) with the same interface.

    Examples:
        >>> analyzer = FinancialSentimentAnalyzer()
        >>> result = analyzer.analyze("Apple reported record earnings")
        >>> print(f"Sentiment: {result.label} ({result.confidence:.2%})")
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model name. Options:
                - "ProsusAI/finbert" (default, best for sentiment)
                - "yiyanghkust/finbert-tone" (financial tone)
                - "distilbert-base-uncased-finetuned-sst-2-english" (general)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._tokenizer = None
        self._loaded = False

        # Label mapping (model-specific)
        self._label_maps = {
            "ProsusAI/finbert": {0: "POSITIVE", 1: "NEGATIVE", 2: "NEUTRAL"},
            "default": {0: "NEGATIVE", 1: "POSITIVE"}  # For binary classifiers
        }

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._loaded:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._loaded = True

        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of financial text.

        Args:
            text: Financial news, report, or document text

        Returns:
            SentimentResult with label and probabilities
        """
        self._load_model()

        # Tokenize input
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Handle different model output formats
        if probs.shape[0] == 3:
            # FinBERT-style (positive, negative, neutral)
            positive_prob = probs[0].item()
            negative_prob = probs[1].item()
            neutral_prob = probs[2].item()
            label_map = self._label_maps.get(self.model_name, self._label_maps["ProsusAI/finbert"])
        else:
            # Binary classifier
            negative_prob = probs[0].item()
            positive_prob = probs[1].item()
            neutral_prob = 0.0
            label_map = self._label_maps["default"]

        # Determine label
        label_idx = probs.argmax().item()
        label = label_map.get(label_idx, "NEUTRAL")
        confidence = probs.max().item()

        return SentimentResult(
            label=label,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=neutral_prob,
            confidence=confidence
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts.

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

        This mimics BloombergGPT's aspect-specific sentiment capability
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
            # Find sentences mentioning the entity
            entity_sentences = [
                s for s in sentences
                if entity.lower() in s.lower()
            ]

            if entity_sentences:
                entity_text = ' '.join(entity_sentences)
                results[entity] = self.analyze(entity_text)
            else:
                # Entity not mentioned - return neutral with zero confidence
                results[entity] = SentimentResult(
                    label="NOT_MENTIONED",
                    positive_prob=0.0,
                    negative_prob=0.0,
                    neutral_prob=1.0,
                    confidence=0.0
                )

        return results

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class AspectSentimentAnalyzer(FinancialSentimentAnalyzer):
    """
    Advanced aspect-based sentiment analyzer.

    Extends the base analyzer with more sophisticated entity extraction
    and context-aware sentiment analysis.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ner_model = None

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract financial entities from text.

        Returns:
            List of dicts with entity info (text, type, position)
        """
        # Simple pattern-based extraction
        # In production, use NER model or BloombergGPT-style entity linking

        entities = []

        # Company patterns (ticker symbols)
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        for match in re.finditer(ticker_pattern, text):
            if len(match.group()) >= 2:  # Filter single letters
                entities.append({
                    "text": match.group(),
                    "type": "TICKER",
                    "start": match.start(),
                    "end": match.end()
                })

        # Money patterns
        money_pattern = r'\$[\d,]+\.?\d*[BMK]?|\d+(?:\.\d+)?%'
        for match in re.finditer(money_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "MONEY",
                "start": match.start(),
                "end": match.end()
            })

        return entities

    def analyze_with_entities(self, text: str) -> Dict:
        """
        Analyze text with automatic entity extraction.

        Returns:
            Dict with overall sentiment and per-entity sentiments
        """
        overall = self.analyze(text)
        entities = self.extract_entities(text)

        entity_names = list(set(e["text"] for e in entities if e["type"] == "TICKER"))
        aspect_sentiments = self.analyze_aspects(text, entity_names) if entity_names else {}

        return {
            "overall": overall,
            "entities": entities,
            "aspect_sentiments": aspect_sentiments
        }


# Convenience function
def analyze_sentiment(text: str, model_name: str = "ProsusAI/finbert") -> SentimentResult:
    """
    Convenience function for quick sentiment analysis.

    Args:
        text: Financial text to analyze
        model_name: Model to use

    Returns:
        SentimentResult
    """
    analyzer = FinancialSentimentAnalyzer(model_name=model_name)
    return analyzer.analyze(text)


if __name__ == "__main__":
    # Demo usage
    analyzer = FinancialSentimentAnalyzer()

    test_texts = [
        "Apple reported record quarterly revenue, beating analyst expectations.",
        "Tesla shares plunged after disappointing delivery numbers.",
        "The Federal Reserve kept interest rates unchanged.",
    ]

    print("Financial Sentiment Analysis Demo\n" + "=" * 50)

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result.label}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Score: {result.score:+.2f}")
