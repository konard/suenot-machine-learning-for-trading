"""
DeBERTa Sentiment Model for financial text analysis.

Provides:
- DeBERTa-based sentiment classification (positive/negative/neutral)
- Confidence-weighted signal generation
- Batch inference for real-time trading
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    text: str
    label: str
    score: float
    probabilities: Dict[str, float]


# Financial sentiment lexicon for rule-based fallback
_POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "surged", "rally", "rallied", "gain",
    "gains", "profit", "profits", "growth", "record", "high", "bullish",
    "upgrade", "upgraded", "outperform", "exceeded", "exceeds", "strong",
    "positive", "rises", "rose", "soar", "soars", "soared", "boom",
    "recovery", "rebounds", "optimistic", "approval", "approved",
}

_NEGATIVE_WORDS = {
    "miss", "misses", "drop", "drops", "dropped", "fall", "falls", "fell",
    "loss", "losses", "decline", "declined", "low", "bearish", "downgrade",
    "downgraded", "underperform", "crash", "crashed", "plunge", "plunged",
    "weak", "negative", "risk", "recession", "selloff", "sell-off",
    "bankruptcy", "default", "fraud", "investigation", "recall", "breach",
}


class DeBERTaSentimentModel:
    """
    DeBERTa-based sentiment model for financial text.

    Uses HuggingFace transformers when available, with a rule-based
    fallback for environments without GPU or model weights.

    Example:
        >>> model = DeBERTaSentimentModel()
        >>> results = model.predict(["Apple revenue beats expectations"])
        >>> print(results[0].label, results[0].score)
        positive 0.85
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        max_length: int = 512,
        device: Optional[str] = None,
        use_transformer: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize DeBERTa sentiment model.

        Args:
            model_name: HuggingFace model name or path
            num_labels: Number of sentiment classes (3 = pos/neg/neutral)
            max_length: Maximum token sequence length
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_transformer: Whether to try loading the transformer model
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.verbose = verbose
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

        self._model = None
        self._tokenizer = None
        self._device = device
        self._use_transformer = use_transformer
        self._initialized = False

        if use_transformer:
            self._try_load_transformer()

    def _try_load_transformer(self) -> None:
        """Attempt to load HuggingFace transformer model."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading DeBERTa model: {self.model_name} on {self._device}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
            self._model.to(self._device)
            self._model.eval()
            self._initialized = True
            logger.info("DeBERTa model loaded successfully")

        except ImportError:
            logger.warning(
                "transformers/torch not available. Using rule-based fallback."
            )
            self._initialized = False
        except Exception as e:
            logger.warning(
                f"Could not load DeBERTa model: {e}. Using rule-based fallback."
            )
            self._initialized = False

    def predict(self, texts: List[str]) -> List[SentimentResult]:
        """
        Predict sentiment for a list of texts.

        Args:
            texts: List of financial text strings

        Returns:
            List of SentimentResult with label, score, and probabilities
        """
        if self._initialized and self._model is not None:
            return self._predict_transformer(texts)
        return self._predict_rule_based(texts)

    def _predict_transformer(self, texts: List[str]) -> List[SentimentResult]:
        """Predict using HuggingFace transformer model."""
        import torch

        results = []
        with torch.no_grad():
            for text in texts:
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                ).to(self._device)

                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

                label_idx = int(np.argmax(probs))
                label = self.label_map[label_idx]
                score = float(probs[label_idx])

                probabilities = {
                    self.label_map[i]: float(probs[i])
                    for i in range(self.num_labels)
                }

                results.append(SentimentResult(
                    text=text,
                    label=label,
                    score=score,
                    probabilities=probabilities,
                ))

        return results

    def _predict_rule_based(self, texts: List[str]) -> List[SentimentResult]:
        """
        Rule-based sentiment prediction as fallback.

        Uses a financial sentiment lexicon to classify text when
        the transformer model is not available.
        """
        results = []
        for text in texts:
            words = set(text.lower().split())

            pos_count = len(words & _POSITIVE_WORDS)
            neg_count = len(words & _NEGATIVE_WORDS)
            total = pos_count + neg_count

            if total == 0:
                label = "neutral"
                probs = {"negative": 0.15, "neutral": 0.70, "positive": 0.15}
            else:
                pos_ratio = pos_count / total
                neg_ratio = neg_count / total

                if pos_ratio > 0.6:
                    label = "positive"
                    score = 0.5 + 0.4 * pos_ratio
                    probs = {
                        "positive": score,
                        "neutral": (1 - score) * 0.6,
                        "negative": (1 - score) * 0.4,
                    }
                elif neg_ratio > 0.6:
                    label = "negative"
                    score = 0.5 + 0.4 * neg_ratio
                    probs = {
                        "negative": score,
                        "neutral": (1 - score) * 0.6,
                        "positive": (1 - score) * 0.4,
                    }
                else:
                    label = "neutral"
                    probs = {
                        "neutral": 0.50,
                        "positive": 0.25 + 0.15 * pos_ratio,
                        "negative": 0.25 + 0.15 * neg_ratio,
                    }

            results.append(SentimentResult(
                text=text,
                label=label,
                score=probs[label],
                probabilities=probs,
            ))

        return results

    def get_trading_signal(
        self,
        texts: List[str],
        threshold: float = 0.6,
    ) -> Dict[str, float]:
        """
        Generate a trading signal from multiple texts.

        Args:
            texts: List of financial headlines/texts
            threshold: Minimum confidence for signal generation

        Returns:
            Dictionary with signal direction and strength:
            - 'signal': float in [-1, 1] (negative=sell, positive=buy)
            - 'confidence': float in [0, 1]
            - 'n_positive': int
            - 'n_negative': int
            - 'n_neutral': int
        """
        predictions = self.predict(texts)

        n_positive = sum(1 for p in predictions if p.label == "positive")
        n_negative = sum(1 for p in predictions if p.label == "negative")
        n_neutral = sum(1 for p in predictions if p.label == "neutral")

        # Confidence-weighted signal
        signal_score = 0.0
        total_weight = 0.0
        for pred in predictions:
            if pred.label == "positive":
                signal_score += pred.score
            elif pred.label == "negative":
                signal_score -= pred.score
            total_weight += pred.score

        if total_weight > 0:
            signal = signal_score / total_weight
        else:
            signal = 0.0

        confidence = abs(signal)

        # Apply threshold
        if confidence < threshold:
            signal = 0.0

        return {
            "signal": signal,
            "confidence": confidence,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_neutral": n_neutral,
        }
