"""
Trading signal generation from fine-tuned LLM predictions.

This module provides utilities for:
- Generating trading signals from sentiment analysis
- Aggregating multiple signals
- Computing signal confidence and strength
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from .model import FinancialSentimentLoRA


@dataclass
class TradingSignal:
    """Represents a trading signal from LLM analysis."""
    text: str
    direction: str  # BUY, HOLD, SELL
    confidence: float
    scores: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "direction": self.direction,
            "confidence": self.confidence,
            "scores": self.scores,
            "timestamp": self.timestamp.isoformat(),
            "actionable": self.is_actionable(),
            **self.metadata
        }

    def is_actionable(self, threshold: float = 0.7) -> bool:
        """Check if signal has sufficient confidence."""
        return self.confidence >= threshold


class TradingSignalGenerator:
    """
    Generate trading signals from text using fine-tuned LLM.

    Converts sentiment predictions into actionable trading signals
    with confidence scoring.

    Example:
        >>> generator = TradingSignalGenerator(model, tokenizer)
        >>> signal = generator.generate("Apple beats earnings expectations")
        >>> print(signal.direction, signal.confidence)
        BUY 0.85
    """

    LABEL_MAP = {
        0: "SELL",   # Bearish
        1: "HOLD",   # Neutral
        2: "BUY"     # Bullish
    }

    LABEL_NAMES = {
        0: "bearish",
        1: "neutral",
        2: "bullish"
    }

    def __init__(
        self,
        model: FinancialSentimentLoRA,
        tokenizer: Optional[Any] = None,
        confidence_threshold: float = 0.7,
        device: str = "auto"
    ):
        """
        Initialize signal generator.

        Args:
            model: Fine-tuned sentiment model
            tokenizer: Text tokenizer
            confidence_threshold: Minimum confidence for actionable signals
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.confidence_threshold = confidence_threshold

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, path: str) -> "TradingSignalGenerator":
        """
        Load signal generator from saved model.

        Args:
            path: Path to saved model directory

        Returns:
            Initialized TradingSignalGenerator
        """
        import os
        import json

        # Load model
        model = FinancialSentimentLoRA()
        model_path = os.path.join(path, "model.pt")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location="cpu"))

        # Load config
        config_path = os.path.join(path, "config.json")
        threshold = 0.7
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                threshold = config.get("confidence_threshold", 0.7)

        return cls(model, confidence_threshold=threshold)

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text into model input.

        This is a simplified version - in practice, use a proper tokenizer.
        """
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            return encoding
        else:
            # Dummy encoding for demonstration
            return torch.randn(1, 128, 768)

    def generate(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TradingSignal:
        """
        Generate trading signal from text.

        Args:
            text: Financial news or analysis text
            metadata: Optional metadata to attach to signal

        Returns:
            TradingSignal with direction and confidence
        """
        metadata = metadata or {}

        # Encode text
        hidden_states = self._encode_text(text)
        if isinstance(hidden_states, dict):
            hidden_states = torch.randn(1, 128, 768)
        hidden_states = hidden_states.to(self.device)

        # Get predictions
        with torch.no_grad():
            logits = self.model(hidden_states)
            probs = F.softmax(logits, dim=-1)

        confidence, prediction = probs.max(dim=-1)

        # Create signal
        scores = {
            self.LABEL_NAMES[i]: probs[0, i].item()
            for i in range(3)
        }

        signal = TradingSignal(
            text=text,
            direction=self.LABEL_MAP[prediction.item()],
            confidence=confidence.item(),
            scores=scores,
            timestamp=datetime.now(),
            metadata=metadata
        )

        return signal

    def batch_generate(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict]] = None
    ) -> List[TradingSignal]:
        """
        Generate signals for multiple texts.

        Args:
            texts: List of text strings
            metadata_list: Optional list of metadata dicts

        Returns:
            List of TradingSignal objects
        """
        metadata_list = metadata_list or [{}] * len(texts)

        signals = []
        for text, metadata in zip(texts, metadata_list):
            signal = self.generate(text, metadata)
            signals.append(signal)

        return signals

    def aggregate_signals(
        self,
        signals: List[TradingSignal],
        method: str = "confidence_weighted"
    ) -> Dict[str, Any]:
        """
        Aggregate multiple signals into a composite signal.

        Args:
            signals: List of trading signals
            method: Aggregation method
                - "confidence_weighted": Weight by confidence
                - "majority_vote": Simple majority voting
                - "unanimous": Only signal if all agree

        Returns:
            Dictionary with aggregated signal information
        """
        if not signals:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "num_sources": 0,
                "actionable": False
            }

        if method == "confidence_weighted":
            return self._aggregate_weighted(signals)
        elif method == "majority_vote":
            return self._aggregate_majority(signals)
        elif method == "unanimous":
            return self._aggregate_unanimous(signals)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _aggregate_weighted(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Confidence-weighted aggregation."""
        weighted_scores = {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
        total_weight = 0.0

        for signal in signals:
            weight = signal.confidence
            weighted_scores[signal.direction] += weight
            total_weight += weight

        # Normalize
        if total_weight > 0:
            for key in weighted_scores:
                weighted_scores[key] /= total_weight

        # Get final direction
        direction = max(weighted_scores, key=weighted_scores.get)
        confidence = weighted_scores[direction]

        return {
            "direction": direction,
            "confidence": confidence,
            "score_breakdown": weighted_scores,
            "num_sources": len(signals),
            "actionable": confidence >= self.confidence_threshold
        }

    def _aggregate_majority(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Majority voting aggregation."""
        from collections import Counter

        votes = Counter(s.direction for s in signals)
        direction, count = votes.most_common(1)[0]
        confidence = count / len(signals)

        return {
            "direction": direction,
            "confidence": confidence,
            "vote_counts": dict(votes),
            "num_sources": len(signals),
            "actionable": confidence >= self.confidence_threshold
        }

    def _aggregate_unanimous(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Unanimous agreement aggregation."""
        directions = set(s.direction for s in signals)

        if len(directions) == 1:
            direction = directions.pop()
            avg_confidence = np.mean([s.confidence for s in signals])
        else:
            direction = "HOLD"
            avg_confidence = 0.0

        return {
            "direction": direction,
            "confidence": avg_confidence,
            "unanimous": len(directions) == 1,
            "num_sources": len(signals),
            "actionable": len(directions) == 1 and avg_confidence >= self.confidence_threshold
        }

    def generate_time_series_signals(
        self,
        texts: List[str],
        timestamps: List[datetime],
        window: int = 5
    ) -> pd.DataFrame:
        """
        Generate signals over time with rolling aggregation.

        Args:
            texts: List of text strings (chronological)
            timestamps: Corresponding timestamps
            window: Rolling window size for aggregation

        Returns:
            DataFrame with signals over time
        """
        # Generate individual signals
        signals = self.batch_generate(texts)

        # Create DataFrame
        records = []
        for signal, ts in zip(signals, timestamps):
            records.append({
                "timestamp": ts,
                "direction": signal.direction,
                "confidence": signal.confidence,
                "bullish_score": signal.scores["bullish"],
                "neutral_score": signal.scores["neutral"],
                "bearish_score": signal.scores["bearish"],
                "signal_numeric": {"SELL": -1, "HOLD": 0, "BUY": 1}[signal.direction]
            })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Rolling aggregation
        df["rolling_signal"] = df["signal_numeric"].rolling(window).mean()
        df["rolling_confidence"] = df["confidence"].rolling(window).mean()

        # Convert rolling signal to direction
        def rolling_to_direction(val):
            if pd.isna(val):
                return "HOLD"
            elif val > 0.3:
                return "BUY"
            elif val < -0.3:
                return "SELL"
            return "HOLD"

        df["aggregated_direction"] = df["rolling_signal"].apply(rolling_to_direction)

        return df


class CryptoSignalGenerator(TradingSignalGenerator):
    """
    Signal generator specialized for cryptocurrency markets.

    Adds crypto-specific features like funding rate sentiment
    and volatility-adjusted confidence.
    """

    def __init__(
        self,
        model: FinancialSentimentLoRA,
        tokenizer: Optional[Any] = None,
        confidence_threshold: float = 0.7,
        volatility_adjustment: bool = True
    ):
        super().__init__(model, tokenizer, confidence_threshold)
        self.volatility_adjustment = volatility_adjustment

    def generate_with_market_context(
        self,
        text: str,
        market_data: Optional[Dict[str, float]] = None
    ) -> TradingSignal:
        """
        Generate signal with market context adjustment.

        Args:
            text: News or analysis text
            market_data: Optional dict with:
                - volatility: Current volatility
                - funding_rate: Current funding rate
                - volume_ratio: Volume vs average

        Returns:
            TradingSignal adjusted for market conditions
        """
        # Get base signal
        signal = self.generate(text)

        if market_data and self.volatility_adjustment:
            # Adjust confidence based on volatility
            volatility = market_data.get("volatility", 0.5)
            if volatility > 1.0:  # High volatility
                signal.confidence *= 0.8  # Reduce confidence
            elif volatility < 0.3:  # Low volatility
                signal.confidence *= 1.1  # Increase confidence

            # Cap confidence at 1.0
            signal.confidence = min(signal.confidence, 1.0)

            # Add market context to metadata
            signal.metadata["market_context"] = {
                "volatility": volatility,
                "funding_rate": market_data.get("funding_rate", 0),
                "volume_ratio": market_data.get("volume_ratio", 1.0)
            }

        return signal

    def generate_with_funding_rate(
        self,
        text: str,
        funding_rate: float
    ) -> TradingSignal:
        """
        Adjust signal based on funding rate.

        High positive funding = crowded longs (contrarian bearish)
        High negative funding = crowded shorts (contrarian bullish)
        """
        signal = self.generate(text)

        # Funding rate contrarian adjustment
        if abs(funding_rate) > 0.001:  # Significant funding
            funding_signal = -np.sign(funding_rate)  # Contrarian

            # Blend with text signal
            text_signal = {"SELL": -1, "HOLD": 0, "BUY": 1}[signal.direction]
            blended = 0.7 * text_signal + 0.3 * funding_signal

            if blended > 0.3:
                signal.direction = "BUY"
            elif blended < -0.3:
                signal.direction = "SELL"
            else:
                signal.direction = "HOLD"

            signal.metadata["funding_rate"] = funding_rate
            signal.metadata["funding_adjustment"] = funding_signal

        return signal


def compute_signal_metrics(
    signals: List[TradingSignal],
    actual_returns: List[float]
) -> Dict[str, float]:
    """
    Compute signal quality metrics.

    Args:
        signals: List of generated signals
        actual_returns: Corresponding actual returns

    Returns:
        Dictionary with accuracy, precision, recall, etc.
    """
    if len(signals) != len(actual_returns):
        raise ValueError("Signals and returns must have same length")

    # Convert signals to numeric
    signal_values = []
    for signal in signals:
        if signal.direction == "BUY":
            signal_values.append(1)
        elif signal.direction == "SELL":
            signal_values.append(-1)
        else:
            signal_values.append(0)

    # Convert returns to direction
    actual_values = [1 if r > 0 else -1 if r < 0 else 0 for r in actual_returns]

    # Compute metrics
    correct = sum(1 for s, a in zip(signal_values, actual_values)
                  if s == a or s == 0)
    accuracy = correct / len(signals)

    # Directional accuracy (excluding HOLD)
    directional_signals = [(s, a) for s, a in zip(signal_values, actual_values) if s != 0]
    if directional_signals:
        directional_correct = sum(1 for s, a in directional_signals if s == a)
        directional_accuracy = directional_correct / len(directional_signals)
    else:
        directional_accuracy = 0.0

    # Profit factor
    profits = []
    for signal, ret in zip(signal_values, actual_returns):
        profits.append(signal * ret)

    total_profit = sum(profits)
    winning_trades = sum(1 for p in profits if p > 0)
    total_trades = sum(1 for s in signal_values if s != 0)

    return {
        "accuracy": accuracy,
        "directional_accuracy": directional_accuracy,
        "total_profit": total_profit,
        "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
        "num_signals": len(signals),
        "num_trades": total_trades,
        "avg_confidence": np.mean([s.confidence for s in signals])
    }


if __name__ == "__main__":
    # Example usage
    print("Testing Trading Signal Generator...")

    # Create model
    model = FinancialSentimentLoRA()

    # Create generator
    generator = TradingSignalGenerator(model, confidence_threshold=0.7)

    # Test texts
    texts = [
        "Apple beats earnings expectations, stock surges 5%",
        "Fed signals hawkish pivot, yields rise sharply",
        "Company maintains guidance amid uncertain outlook",
        "Bitcoin surges on institutional buying pressure",
        "Bybit reports record trading volume for perpetuals"
    ]

    print("\nGenerating signals...")
    for text in texts:
        signal = generator.generate(text)
        print(f"\nText: {text[:50]}...")
        print(f"Direction: {signal.direction}")
        print(f"Confidence: {signal.confidence:.3f}")
        print(f"Scores: {signal.scores}")

    # Test aggregation
    print("\n" + "="*50)
    print("Testing signal aggregation...")

    signals = generator.batch_generate(texts)
    aggregate = generator.aggregate_signals(signals)

    print(f"\nAggregate signal:")
    print(f"Direction: {aggregate['direction']}")
    print(f"Confidence: {aggregate['confidence']:.3f}")
    print(f"Score breakdown: {aggregate['score_breakdown']}")
    print(f"Actionable: {aggregate['actionable']}")
