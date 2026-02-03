"""
Concept Bottleneck Model implementation for trading.

The model predicts trading signals through interpretable intermediate concepts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .concepts import ConceptExtractor, TradingConcepts


class TradingSignal(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Prediction:
    """Container for model prediction with explanation."""
    signal: TradingSignal
    confidence: float
    concepts: TradingConcepts
    concept_contributions: Dict[str, float]


class ConceptEncoder:
    """
    Neural network that predicts concepts from raw features.

    In a full implementation, this would be a PyTorch/TensorFlow model.
    This implementation uses a rule-based approach for simplicity.
    """

    def __init__(self, input_dim: int = 50, concept_dim: int = 11):
        self.input_dim = input_dim
        self.concept_dim = concept_dim
        self.extractor = ConceptExtractor()

    def forward(self, df: pd.DataFrame) -> np.ndarray:
        """Extract concept vector from market data."""
        concepts = self.extractor.extract(df)
        return concepts.to_vector()

    def forward_with_concepts(self, df: pd.DataFrame) -> Tuple[np.ndarray, TradingConcepts]:
        """Extract concept vector and return full concept object."""
        concepts = self.extractor.extract(df)
        return concepts.to_vector(), concepts


class TaskPredictor:
    """
    Predicts trading signal from concept vector.

    Uses a simple linear model for interpretability.
    Weights can be inspected to understand concept importance.
    """

    def __init__(self, concept_dim: int = 11, output_dim: int = 3):
        self.concept_dim = concept_dim
        self.output_dim = output_dim

        # Initialize weights with interpretable defaults
        # [trend_dir, trend_str, vol_regime, vol_val, mom_sig, mom_val,
        #  vol_prof, vol_ratio, rsi, support_dist, resist_dist]
        self.weights = np.array([
            [0.3, 0.2, -0.1, -0.1, 0.2, 0.15, 0.1, 0.05, -0.1, 0.1, -0.1],  # BUY
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # HOLD
            [-0.3, 0.2, 0.1, 0.1, -0.2, -0.15, -0.1, -0.05, 0.1, -0.1, 0.1], # SELL
        ])
        self.bias = np.array([0.0, 0.3, 0.0])  # Slight bias toward HOLD

    def forward(self, concept_vector: np.ndarray) -> np.ndarray:
        """Compute logits for each trading action."""
        logits = self.weights @ concept_vector + self.bias
        return logits

    def predict(self, concept_vector: np.ndarray) -> Tuple[TradingSignal, float]:
        """Predict trading signal with confidence."""
        logits = self.forward(concept_vector)
        probs = self._softmax(logits)

        action_idx = np.argmax(probs)
        confidence = probs[action_idx]

        signals = [TradingSignal.BUY, TradingSignal.HOLD, TradingSignal.SELL]
        return signals[action_idx], confidence

    def get_contributions(self, concept_vector: np.ndarray) -> Dict[str, float]:
        """Get contribution of each concept to the prediction."""
        concept_names = [
            'trend_direction', 'trend_strength', 'volatility_regime',
            'volatility_value', 'momentum_signal', 'momentum_value',
            'volume_profile', 'volume_ratio', 'rsi',
            'support_distance', 'resistance_distance'
        ]

        logits = self.forward(concept_vector)
        action_idx = np.argmax(logits)

        contributions = {}
        for i, name in enumerate(concept_names):
            contributions[name] = self.weights[action_idx, i] * concept_vector[i]

        return contributions

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def fit(
        self,
        concept_vectors: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100
    ):
        """
        Train the task predictor using gradient descent.

        Args:
            concept_vectors: Shape (n_samples, concept_dim)
            labels: Shape (n_samples,) with values 0 (sell), 1 (hold), 2 (buy)
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
        """
        n_samples = len(labels)
        one_hot = np.zeros((n_samples, self.output_dim))
        one_hot[np.arange(n_samples), labels.astype(int)] = 1

        for _ in range(epochs):
            logits = concept_vectors @ self.weights.T + self.bias
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)

            grad_logits = (probs - one_hot) / n_samples
            grad_weights = grad_logits.T @ concept_vectors
            grad_bias = grad_logits.sum(axis=0)

            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias


class ConceptBottleneckTrader:
    """
    Complete Concept Bottleneck Model for trading.

    Combines concept extraction and task prediction into a single
    interpretable trading model.
    """

    def __init__(self):
        self.concept_encoder = ConceptEncoder()
        self.task_predictor = TaskPredictor()

    def predict(self, df: pd.DataFrame) -> Tuple[TradingSignal, TradingConcepts]:
        """
        Predict trading signal from market data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            Tuple of (TradingSignal, TradingConcepts)
        """
        concept_vector, concepts = self.concept_encoder.forward_with_concepts(df)
        signal, _ = self.task_predictor.predict(concept_vector)
        return signal, concepts

    def predict_with_explanation(self, df: pd.DataFrame) -> Prediction:
        """
        Predict trading signal with full explanation.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Prediction object with signal, confidence, concepts, and contributions
        """
        concept_vector, concepts = self.concept_encoder.forward_with_concepts(df)
        signal, confidence = self.task_predictor.predict(concept_vector)
        contributions = self.task_predictor.get_contributions(concept_vector)

        return Prediction(
            signal=signal,
            confidence=confidence,
            concepts=concepts,
            concept_contributions=contributions,
        )

    def predict_with_intervention(
        self,
        df: pd.DataFrame,
        concept_overrides: Dict[str, float]
    ) -> Prediction:
        """
        Predict with manual concept overrides.

        This allows human experts to correct concept predictions
        before the final trading decision is made.

        Args:
            df: DataFrame with OHLCV data
            concept_overrides: Dictionary of concept_name -> override_value

        Returns:
            Prediction with intervention applied
        """
        concept_vector, concepts = self.concept_encoder.forward_with_concepts(df)

        concept_indices = {
            'trend_direction': 0, 'trend_strength': 1,
            'volatility_regime': 2, 'volatility_value': 3,
            'momentum_signal': 4, 'momentum_value': 5,
            'volume_profile': 6, 'volume_ratio': 7,
            'rsi': 8, 'support_distance': 9, 'resistance_distance': 10
        }

        for concept_name, value in concept_overrides.items():
            if concept_name in concept_indices:
                concept_vector[concept_indices[concept_name]] = value

        signal, confidence = self.task_predictor.predict(concept_vector)
        contributions = self.task_predictor.get_contributions(concept_vector)

        return Prediction(
            signal=signal,
            confidence=confidence,
            concepts=concepts,
            concept_contributions=contributions,
        )

    def train(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01
    ):
        """
        Train the model on historical data.

        Args:
            df: DataFrame with OHLCV data
            labels: Array of labels (0=sell, 1=hold, 2=buy)
            epochs: Training epochs
            learning_rate: Learning rate
        """
        concept_vectors = []
        valid_indices = []

        min_required = max(
            self.concept_encoder.extractor.trend_window,
            self.concept_encoder.extractor.volatility_window,
            self.concept_encoder.extractor.rsi_window
        )

        for i in range(min_required, len(df)):
            try:
                window_df = df.iloc[:i+1]
                vec = self.concept_encoder.forward(window_df)
                concept_vectors.append(vec)
                valid_indices.append(i)
            except (ValueError, IndexError):
                continue

        if not concept_vectors:
            raise ValueError("Not enough data to extract concepts")

        X = np.array(concept_vectors)
        y = labels[valid_indices]

        self.task_predictor.fit(X, y, learning_rate, epochs)


def create_pytorch_cbm():
    """
    Create a PyTorch-based Concept Bottleneck Model.

    This is a template for a more sophisticated implementation
    using deep learning frameworks.
    """
    try:
        import torch
        import torch.nn as nn

        class PyTorchConceptEncoder(nn.Module):
            def __init__(self, input_dim: int = 50, hidden_dim: int = 64, concept_dim: int = 11):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, concept_dim),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.network(x)

        class PyTorchTaskPredictor(nn.Module):
            def __init__(self, concept_dim: int = 11, output_dim: int = 3):
                super().__init__()
                self.linear = nn.Linear(concept_dim, output_dim)

            def forward(self, concepts):
                return self.linear(concepts)

        class PyTorchCBM(nn.Module):
            def __init__(self, input_dim: int = 50, concept_dim: int = 11, output_dim: int = 3):
                super().__init__()
                self.concept_encoder = PyTorchConceptEncoder(input_dim, 64, concept_dim)
                self.task_predictor = PyTorchTaskPredictor(concept_dim, output_dim)

            def forward(self, x):
                concepts = self.concept_encoder(x)
                output = self.task_predictor(concepts)
                return output, concepts

        return PyTorchCBM()

    except ImportError:
        raise ImportError("PyTorch is required for the neural network implementation. Install with: pip install torch")
