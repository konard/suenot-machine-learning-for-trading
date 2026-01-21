"""
Few-Shot Market Prediction

This module implements multiple few-shot learning approaches for financial
market prediction, including metric-based methods, MAML-inspired optimization,
and hybrid approaches.

Key Features:
- Multiple few-shot learning paradigms (metric-based, optimization-based, hybrid)
- Support for various distance/similarity functions
- N-way K-shot classification framework
- Episodic training support
- Market-specific feature extraction
- Confidence-based prediction with uncertainty estimation

Example usage:
    ```python
    from few_shot_predictor import FewShotPredictor, MetricBasedClassifier

    # Create predictor with metric-based approach
    predictor = FewShotPredictor(
        method="metric",
        input_dim=20,
        embedding_dim=32
    )

    # Train on support set
    predictor.fit(support_features, support_labels)

    # Predict on query set
    predictions, confidences = predictor.predict(query_features)
    ```
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings


# =============================================================================
# Enums and Configuration
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_UPTREND = 0
    WEAK_UPTREND = 1
    SIDEWAYS = 2
    WEAK_DOWNTREND = 3
    STRONG_DOWNTREND = 4
    VOLATILE = 5
    CRASH = 6
    RECOVERY = 7

    @property
    def trading_bias(self) -> float:
        """Get trading bias (-1 to 1)."""
        biases = {
            MarketRegime.STRONG_UPTREND: 1.0,
            MarketRegime.WEAK_UPTREND: 0.5,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.WEAK_DOWNTREND: -0.5,
            MarketRegime.STRONG_DOWNTREND: -1.0,
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.CRASH: -1.0,
            MarketRegime.RECOVERY: 0.7,
        }
        return biases[self]

    @property
    def risk_level(self) -> str:
        """Get risk level for the regime."""
        risk_map = {
            MarketRegime.STRONG_UPTREND: "low",
            MarketRegime.WEAK_UPTREND: "medium",
            MarketRegime.SIDEWAYS: "low",
            MarketRegime.WEAK_DOWNTREND: "medium",
            MarketRegime.STRONG_DOWNTREND: "high",
            MarketRegime.VOLATILE: "high",
            MarketRegime.CRASH: "extreme",
            MarketRegime.RECOVERY: "medium",
        }
        return risk_map[self]


class DistanceMetric(Enum):
    """Distance metrics for similarity computation."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    LEARNED = "learned"


class FewShotMethod(Enum):
    """Few-shot learning methods."""
    METRIC = "metric"           # Prototypical/Matching networks
    MAML = "maml"               # Model-Agnostic Meta-Learning
    HYBRID = "hybrid"           # Combination of metric and optimization
    SIAMESE = "siamese"         # Siamese networks


@dataclass
class FewShotConfig:
    """Configuration for few-shot learning."""
    method: FewShotMethod = FewShotMethod.METRIC
    n_way: int = 5                      # Number of classes per episode
    k_shot: int = 5                     # Number of examples per class
    n_query: int = 10                   # Number of query samples per class

    # Network architecture
    input_dim: int = 20
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    embedding_dim: int = 32

    # Training parameters
    learning_rate: float = 0.001
    n_episodes: int = 1000
    adaptation_steps: int = 5           # For MAML
    adaptation_lr: float = 0.01         # Inner loop learning rate for MAML

    # Distance/similarity
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN
    temperature: float = 1.0            # Softmax temperature

    # Regularization
    dropout_rate: float = 0.1
    l2_reg: float = 0.0001


# =============================================================================
# Neural Network Components
# =============================================================================

class ActivationFunctions:
    """Collection of activation functions."""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)


class EmbeddingNetwork:
    """
    Neural network for embedding features into a representation space.

    Supports both forward pass and gradient computation for MAML-style training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        embedding_dim: int,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        self.weights = []
        self.biases = []
        self.bn_params = []  # Batch norm parameters

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        dims = [self.input_dim] + self.hidden_dims + [self.embedding_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))

            w = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)

            self.weights.append(w)
            self.biases.append(b)

            if self.use_batch_norm and i < len(dims) - 2:
                self.bn_params.append({
                    'gamma': np.ones(fan_out),
                    'beta': np.zeros(fan_out),
                    'running_mean': np.zeros(fan_out),
                    'running_var': np.ones(fan_out)
                })

    def forward(
        self,
        x: np.ndarray,
        weights: Optional[List[np.ndarray]] = None,
        biases: Optional[List[np.ndarray]] = None,
        training: bool = False
    ) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input features, shape (batch_size, input_dim)
            weights: Optional weights (for MAML adaptation)
            biases: Optional biases (for MAML adaptation)
            training: Whether in training mode (affects dropout)

        Returns:
            Embeddings, shape (batch_size, embedding_dim)
        """
        if weights is None:
            weights = self.weights
        if biases is None:
            biases = self.biases

        single_sample = x.ndim == 1
        if single_sample:
            x = x.reshape(1, -1)

        h = x
        for i, (w, b) in enumerate(zip(weights, biases)):
            h = h @ w + b

            # Apply activation to all but last layer
            if i < len(weights) - 1:
                # Batch normalization
                if self.use_batch_norm and i < len(self.bn_params):
                    h = self._batch_norm(h, i, training)

                # ReLU activation
                h = ActivationFunctions.relu(h)

                # Dropout
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, h.shape)
                    h = h * mask / (1 - self.dropout_rate)

        # L2 normalize final embeddings
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        h = h / norms

        if single_sample:
            h = h.squeeze(0)

        return h

    def _batch_norm(
        self,
        x: np.ndarray,
        layer_idx: int,
        training: bool,
        momentum: float = 0.1,
        eps: float = 1e-5
    ) -> np.ndarray:
        """Apply batch normalization."""
        params = self.bn_params[layer_idx]

        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # Update running statistics
            params['running_mean'] = (
                momentum * mean + (1 - momentum) * params['running_mean']
            )
            params['running_var'] = (
                momentum * var + (1 - momentum) * params['running_var']
            )
        else:
            mean = params['running_mean']
            var = params['running_var']

        x_norm = (x - mean) / np.sqrt(var + eps)
        return params['gamma'] * x_norm + params['beta']

    def get_params(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get current network parameters."""
        return [w.copy() for w in self.weights], [b.copy() for b in self.biases]

    def set_params(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        """Set network parameters."""
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]


# =============================================================================
# Distance and Similarity Functions
# =============================================================================

class DistanceComputer:
    """Computes distances/similarities between embeddings."""

    def __init__(self, metric: DistanceMetric = DistanceMetric.EUCLIDEAN):
        self.metric = metric
        self.learned_weights: Optional[np.ndarray] = None

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Distance value (lower = more similar)
        """
        if self.metric == DistanceMetric.EUCLIDEAN:
            return np.linalg.norm(a - b)

        elif self.metric == DistanceMetric.COSINE:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                return 1.0
            return 1.0 - np.dot(a, b) / (norm_a * norm_b)

        elif self.metric == DistanceMetric.MANHATTAN:
            return np.sum(np.abs(a - b))

        elif self.metric == DistanceMetric.LEARNED:
            if self.learned_weights is not None:
                diff = a - b
                return float(diff @ self.learned_weights @ diff)
            return np.linalg.norm(a - b)

        return np.linalg.norm(a - b)

    def compute_batch(
        self,
        queries: np.ndarray,
        prototypes: np.ndarray
    ) -> np.ndarray:
        """
        Compute distances between queries and prototypes.

        Args:
            queries: Query embeddings, shape (n_queries, embedding_dim)
            prototypes: Prototype embeddings, shape (n_classes, embedding_dim)

        Returns:
            Distance matrix, shape (n_queries, n_classes)
        """
        n_queries = queries.shape[0]
        n_classes = prototypes.shape[0]
        distances = np.zeros((n_queries, n_classes))

        for i in range(n_queries):
            for j in range(n_classes):
                distances[i, j] = self.compute(queries[i], prototypes[j])

        return distances


# =============================================================================
# Few-Shot Learning Methods
# =============================================================================

class BaseFewShotLearner(ABC):
    """Abstract base class for few-shot learners."""

    @abstractmethod
    def fit(self, support_features: np.ndarray, support_labels: np.ndarray):
        """Fit on support set."""
        pass

    @abstractmethod
    def predict(
        self,
        query_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on query set."""
        pass


class MetricBasedLearner(BaseFewShotLearner):
    """
    Metric-based few-shot learner (Prototypical Networks style).

    Classifies queries based on distance to class prototypes.
    """

    def __init__(self, config: FewShotConfig):
        self.config = config
        self.network = EmbeddingNetwork(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            embedding_dim=config.embedding_dim,
            dropout_rate=config.dropout_rate
        )
        self.distance_computer = DistanceComputer(config.distance_metric)
        self.prototypes: Dict[int, np.ndarray] = {}

    def fit(self, support_features: np.ndarray, support_labels: np.ndarray):
        """
        Fit by computing prototypes for each class.

        Args:
            support_features: Support set features, shape (n_support, input_dim)
            support_labels: Support set labels, shape (n_support,)
        """
        # Embed support features
        embeddings = self.network.forward(support_features, training=False)

        # Compute prototypes as class centroids
        self.prototypes = {}
        for class_idx in np.unique(support_labels):
            mask = support_labels == class_idx
            class_embeddings = embeddings[mask]
            self.prototypes[int(class_idx)] = np.mean(class_embeddings, axis=0)

    def predict(
        self,
        query_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes for query features.

        Args:
            query_features: Query features, shape (n_query, input_dim)

        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.prototypes:
            raise ValueError("Model not fitted. Call fit() first.")

        # Embed query features
        query_embeddings = self.network.forward(query_features, training=False)

        # Stack prototypes
        class_indices = sorted(self.prototypes.keys())
        prototype_matrix = np.stack([self.prototypes[i] for i in class_indices])

        # Compute distances
        distances = self.distance_computer.compute_batch(
            query_embeddings, prototype_matrix
        )

        # Convert to probabilities using softmax on negative distances
        neg_distances = -distances / self.config.temperature
        exp_distances = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)

        # Predictions are the class with highest probability
        predictions = np.array([class_indices[i] for i in np.argmax(probabilities, axis=1)])

        return predictions, probabilities


class MAMLLearner(BaseFewShotLearner):
    """
    MAML-inspired few-shot learner.

    Uses gradient-based adaptation for quick learning on new tasks.
    Note: This is a simplified NumPy implementation for demonstration.
    """

    def __init__(self, config: FewShotConfig):
        self.config = config
        self.network = EmbeddingNetwork(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            embedding_dim=config.embedding_dim,
            dropout_rate=config.dropout_rate
        )
        self.distance_computer = DistanceComputer(config.distance_metric)

        # Store adapted parameters after fitting
        self.adapted_weights: Optional[List[np.ndarray]] = None
        self.adapted_biases: Optional[List[np.ndarray]] = None
        self.prototypes: Dict[int, np.ndarray] = {}

    def _compute_gradients(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: List[np.ndarray],
        biases: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute gradients for adaptation.

        This is a simplified gradient computation using finite differences.
        In practice, you would use automatic differentiation.
        """
        eps = 1e-5
        weight_grads = []
        bias_grads = []

        # Compute loss at current point
        embeddings = self.network.forward(features, weights, biases, training=False)
        base_loss = self._compute_loss(embeddings, labels)

        # Compute gradients for each parameter
        for layer_idx in range(len(weights)):
            # Weight gradients
            w_grad = np.zeros_like(weights[layer_idx])
            for i in range(weights[layer_idx].shape[0]):
                for j in range(weights[layer_idx].shape[1]):
                    weights[layer_idx][i, j] += eps
                    embeddings = self.network.forward(features, weights, biases, training=False)
                    loss_plus = self._compute_loss(embeddings, labels)
                    weights[layer_idx][i, j] -= eps
                    w_grad[i, j] = (loss_plus - base_loss) / eps
            weight_grads.append(w_grad)

            # Bias gradients
            b_grad = np.zeros_like(biases[layer_idx])
            for i in range(biases[layer_idx].shape[0]):
                biases[layer_idx][i] += eps
                embeddings = self.network.forward(features, weights, biases, training=False)
                loss_plus = self._compute_loss(embeddings, labels)
                biases[layer_idx][i] -= eps
                b_grad[i] = (loss_plus - base_loss) / eps
            bias_grads.append(b_grad)

        return weight_grads, bias_grads

    def _compute_loss(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute prototype-based loss.

        Uses negative log probability of correct class.
        """
        # Compute prototypes
        prototypes = {}
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            prototypes[int(class_idx)] = np.mean(embeddings[mask], axis=0)

        # Compute distances and loss
        class_indices = sorted(prototypes.keys())
        prototype_matrix = np.stack([prototypes[i] for i in class_indices])

        total_loss = 0.0
        for i, embedding in enumerate(embeddings):
            distances = np.array([
                self.distance_computer.compute(embedding, prototype_matrix[j])
                for j in range(len(class_indices))
            ])

            # Softmax probabilities
            neg_distances = -distances / self.config.temperature
            exp_distances = np.exp(neg_distances - np.max(neg_distances))
            probs = exp_distances / np.sum(exp_distances)

            # Find correct class index
            true_class = int(labels[i])
            class_idx = class_indices.index(true_class)

            # Negative log probability
            total_loss -= np.log(probs[class_idx] + 1e-8)

        return total_loss / len(embeddings)

    def fit(self, support_features: np.ndarray, support_labels: np.ndarray):
        """
        Fit using gradient-based adaptation.

        Args:
            support_features: Support set features
            support_labels: Support set labels
        """
        # Start from base network parameters
        weights, biases = self.network.get_params()

        # Perform adaptation steps
        for step in range(self.config.adaptation_steps):
            # Compute gradients
            w_grads, b_grads = self._compute_gradients(
                support_features, support_labels, weights, biases
            )

            # Update parameters
            for i in range(len(weights)):
                weights[i] = weights[i] - self.config.adaptation_lr * w_grads[i]
                biases[i] = biases[i] - self.config.adaptation_lr * b_grads[i]

        # Store adapted parameters
        self.adapted_weights = weights
        self.adapted_biases = biases

        # Compute final prototypes
        embeddings = self.network.forward(
            support_features, weights, biases, training=False
        )
        self.prototypes = {}
        for class_idx in np.unique(support_labels):
            mask = support_labels == class_idx
            self.prototypes[int(class_idx)] = np.mean(embeddings[mask], axis=0)

    def predict(
        self,
        query_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using adapted network."""
        if self.adapted_weights is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Embed using adapted parameters
        query_embeddings = self.network.forward(
            query_features,
            self.adapted_weights,
            self.adapted_biases,
            training=False
        )

        # Compute distances to prototypes
        class_indices = sorted(self.prototypes.keys())
        prototype_matrix = np.stack([self.prototypes[i] for i in class_indices])

        distances = self.distance_computer.compute_batch(
            query_embeddings, prototype_matrix
        )

        # Convert to probabilities
        neg_distances = -distances / self.config.temperature
        exp_distances = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probabilities = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)

        predictions = np.array([class_indices[i] for i in np.argmax(probabilities, axis=1)])

        return predictions, probabilities


class HybridLearner(BaseFewShotLearner):
    """
    Hybrid few-shot learner combining metric and optimization approaches.

    Uses metric-based classification with optional MAML-style adaptation.
    """

    def __init__(self, config: FewShotConfig):
        self.config = config
        self.metric_learner = MetricBasedLearner(config)
        self.maml_learner = MAMLLearner(config)

        # Weight for combining predictions
        self.metric_weight = 0.6
        self.maml_weight = 0.4

    def fit(self, support_features: np.ndarray, support_labels: np.ndarray):
        """Fit both metric and MAML learners."""
        self.metric_learner.fit(support_features, support_labels)
        self.maml_learner.fit(support_features, support_labels)

    def predict(
        self,
        query_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict by combining metric and MAML predictions."""
        # Get predictions from both methods
        metric_preds, metric_probs = self.metric_learner.predict(query_features)
        maml_preds, maml_probs = self.maml_learner.predict(query_features)

        # Combine probabilities
        combined_probs = (
            self.metric_weight * metric_probs +
            self.maml_weight * maml_probs
        )

        # Final predictions from combined probabilities
        predictions = np.argmax(combined_probs, axis=1)

        return predictions, combined_probs


# =============================================================================
# Main Few-Shot Predictor
# =============================================================================

class FewShotPredictor:
    """
    High-level interface for few-shot market prediction.

    Supports multiple few-shot learning methods and provides
    market-specific functionality like regime classification
    and confidence estimation.
    """

    def __init__(
        self,
        method: str = "metric",
        input_dim: int = 20,
        hidden_dims: Optional[List[int]] = None,
        embedding_dim: int = 32,
        n_way: int = 5,
        k_shot: int = 5,
        distance_metric: str = "euclidean",
        temperature: float = 1.0,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the few-shot predictor.

        Args:
            method: Learning method ("metric", "maml", "hybrid")
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            embedding_dim: Embedding dimension
            n_way: Number of classes (ways)
            k_shot: Examples per class (shots)
            distance_metric: Distance metric to use
            temperature: Softmax temperature
            confidence_threshold: Threshold for confident predictions
        """
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Parse method
        method_map = {
            "metric": FewShotMethod.METRIC,
            "maml": FewShotMethod.MAML,
            "hybrid": FewShotMethod.HYBRID
        }
        fs_method = method_map.get(method.lower(), FewShotMethod.METRIC)

        # Parse distance metric
        metric_map = {
            "euclidean": DistanceMetric.EUCLIDEAN,
            "cosine": DistanceMetric.COSINE,
            "manhattan": DistanceMetric.MANHATTAN
        }
        dist_metric = metric_map.get(distance_metric.lower(), DistanceMetric.EUCLIDEAN)

        # Create config
        self.config = FewShotConfig(
            method=fs_method,
            n_way=n_way,
            k_shot=k_shot,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            distance_metric=dist_metric,
            temperature=temperature
        )

        # Create learner
        if fs_method == FewShotMethod.METRIC:
            self.learner = MetricBasedLearner(self.config)
        elif fs_method == FewShotMethod.MAML:
            self.learner = MAMLLearner(self.config)
        else:
            self.learner = HybridLearner(self.config)

        self.confidence_threshold = confidence_threshold
        self.class_names: Dict[int, str] = {}

    def fit(
        self,
        support_features: np.ndarray,
        support_labels: np.ndarray,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Fit the predictor on support set.

        Args:
            support_features: Support set features
            support_labels: Support set labels
            class_names: Optional mapping of class indices to names
        """
        self.learner.fit(support_features, support_labels)

        if class_names is not None:
            self.class_names = class_names
        else:
            unique_labels = np.unique(support_labels)
            self.class_names = {int(l): f"Class_{l}" for l in unique_labels}

    def predict(
        self,
        query_features: np.ndarray,
        return_details: bool = False
    ) -> Any:
        """
        Make predictions on query features.

        Args:
            query_features: Query features
            return_details: Whether to return detailed results

        Returns:
            If return_details=False: Tuple of (predictions, confidences)
            If return_details=True: List of detailed prediction dictionaries
        """
        predictions, probabilities = self.learner.predict(query_features)
        confidences = np.max(probabilities, axis=1)

        if not return_details:
            return predictions, confidences

        # Build detailed results
        results = []
        for i, (pred, probs, conf) in enumerate(zip(predictions, probabilities, confidences)):
            # Get class probabilities
            class_probs = {
                self.class_names.get(j, f"Class_{j}"): float(probs[j])
                for j in range(len(probs))
            }

            # Sort by probability
            sorted_classes = sorted(
                class_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )

            results.append({
                "prediction": int(pred),
                "prediction_name": self.class_names.get(int(pred), f"Class_{pred}"),
                "confidence": float(conf),
                "is_confident": conf >= self.confidence_threshold,
                "probabilities": class_probs,
                "ranked_classes": sorted_classes,
                "entropy": float(-np.sum(probs * np.log(probs + 1e-8)))
            })

        return results

    def predict_regime(
        self,
        query_features: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Predict market regime with trading recommendations.

        Args:
            query_features: Query features

        Returns:
            List of regime predictions with trading info
        """
        results = self.predict(query_features, return_details=True)

        regime_results = []
        for result in results:
            pred = result["prediction"]
            conf = result["confidence"]

            # Map to MarketRegime if possible
            try:
                regime = MarketRegime(pred)
                regime_name = regime.name
                trading_bias = regime.trading_bias
                risk_level = regime.risk_level
            except ValueError:
                regime_name = result["prediction_name"]
                trading_bias = 0.0
                risk_level = "unknown"

            # Generate trading recommendation
            if conf < 0.4:
                recommendation = "HOLD - Low confidence"
            elif trading_bias > 0.5:
                recommendation = "LONG - Bullish regime"
            elif trading_bias < -0.5:
                recommendation = "SHORT - Bearish regime"
            else:
                recommendation = "NEUTRAL - Sideways/Volatile regime"

            # Position sizing based on confidence
            position_size = conf * abs(trading_bias)

            regime_results.append({
                "regime": regime_name,
                "confidence": conf,
                "trading_bias": trading_bias,
                "risk_level": risk_level,
                "recommendation": recommendation,
                "suggested_position_size": position_size,
                "probabilities": result["probabilities"]
            })

        return regime_results


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_market_features(
    prices: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    funding_rates: Optional[np.ndarray] = None,
    include_technical: bool = True,
    window_sizes: Optional[List[int]] = None
) -> np.ndarray:
    """
    Extract features from market data.

    Args:
        prices: Close prices
        volumes: Trading volumes (optional)
        funding_rates: Funding rates for perpetual futures (optional)
        include_technical: Whether to include technical indicators
        window_sizes: Window sizes for moving averages

    Returns:
        Feature array
    """
    if window_sizes is None:
        window_sizes = [5, 10, 20, 50]

    features = []

    # Ensure we have enough data
    if len(prices) < 2:
        return np.zeros(20)  # Return zero features for insufficient data

    # Returns
    returns = np.diff(prices) / prices[:-1]

    # Basic return features
    features.append(returns[-1] if len(returns) > 0 else 0.0)  # Latest return

    # Cumulative returns over different windows
    for window in window_sizes:
        if len(returns) >= window:
            features.append(np.sum(returns[-window:]))
        else:
            features.append(0.0)

    # Volatility features
    for window in [10, 20]:
        if len(returns) >= window:
            features.append(np.std(returns[-window:]))
        else:
            features.append(0.0)

    if include_technical:
        # RSI
        if len(returns) >= 14:
            gains = np.maximum(returns[-14:], 0)
            losses = -np.minimum(returns[-14:], 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - 100 / (1 + rs)
            else:
                rsi = 100.0
            features.append(rsi / 100.0 - 0.5)  # Normalize to [-0.5, 0.5]
        else:
            features.append(0.0)

        # Moving average crossovers
        for window in window_sizes:
            if len(prices) >= window:
                ma = np.mean(prices[-window:])
                features.append(prices[-1] / ma - 1.0)
            else:
                features.append(0.0)

        # Price momentum
        for window in [5, 10]:
            if len(prices) > window:
                features.append((prices[-1] / prices[-window-1]) - 1.0)
            else:
                features.append(0.0)

    # Volume features
    if volumes is not None and len(volumes) >= 20:
        vol_ma = np.mean(volumes[-20:])
        if vol_ma > 0:
            features.append(volumes[-1] / vol_ma - 1.0)
            features.append(np.std(volumes[-10:]) / vol_ma)
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])

    # Funding rate features (crypto-specific)
    if funding_rates is not None and len(funding_rates) >= 5:
        features.append(funding_rates[-1])
        features.append(np.mean(funding_rates[-5:]))
    else:
        features.extend([0.0, 0.0])

    return np.array(features)


def create_episodes(
    features: np.ndarray,
    labels: np.ndarray,
    n_way: int,
    k_shot: int,
    n_query: int,
    n_episodes: int = 100
) -> List[Dict[str, np.ndarray]]:
    """
    Create episodic training data for few-shot learning.

    Args:
        features: All available features
        labels: Corresponding labels
        n_way: Number of classes per episode
        k_shot: Support examples per class
        n_query: Query examples per class
        n_episodes: Number of episodes to create

    Returns:
        List of episodes, each containing support and query sets
    """
    unique_classes = np.unique(labels)

    if len(unique_classes) < n_way:
        warnings.warn(f"Only {len(unique_classes)} classes available, but n_way={n_way}")
        n_way = len(unique_classes)

    episodes = []

    for _ in range(n_episodes):
        # Sample classes for this episode
        episode_classes = np.random.choice(unique_classes, size=n_way, replace=False)

        support_features = []
        support_labels = []
        query_features = []
        query_labels = []

        for new_label, class_idx in enumerate(episode_classes):
            # Get all samples for this class
            class_mask = labels == class_idx
            class_features = features[class_mask]

            if len(class_features) < k_shot + n_query:
                # Not enough samples, use with replacement
                indices = np.random.choice(
                    len(class_features),
                    size=k_shot + n_query,
                    replace=True
                )
            else:
                indices = np.random.choice(
                    len(class_features),
                    size=k_shot + n_query,
                    replace=False
                )

            # Split into support and query
            support_idx = indices[:k_shot]
            query_idx = indices[k_shot:k_shot + n_query]

            support_features.append(class_features[support_idx])
            support_labels.extend([new_label] * k_shot)

            query_features.append(class_features[query_idx])
            query_labels.extend([new_label] * n_query)

        episodes.append({
            "support_features": np.vstack(support_features),
            "support_labels": np.array(support_labels),
            "query_features": np.vstack(query_features),
            "query_labels": np.array(query_labels),
            "class_mapping": {i: c for i, c in enumerate(episode_classes)}
        })

    return episodes


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Few-Shot Market Prediction - Example")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Configuration
    n_classes = 5
    n_support_per_class = 10
    n_query_per_class = 5
    input_dim = 20

    # Generate synthetic market data
    print("\n1. Generating synthetic market data...")

    def generate_regime_data(regime_idx: int, n_samples: int) -> np.ndarray:
        """Generate synthetic data for a market regime."""
        base_patterns = {
            0: np.array([0.03, 0.15, 0.3, 0.5, 0.01, 0.02, 0.3, 0.05, 0.1, 0.15,
                        0.2, 0.25, 0.08, 0.12, 0.02, 0.01, 0.15, 0.18, 0.01, 0.02]),  # Strong uptrend
            1: np.array([0.01, 0.05, 0.1, 0.2, 0.008, 0.015, 0.15, 0.03, 0.05, 0.08,
                        0.1, 0.12, 0.04, 0.06, 0.01, 0.005, 0.08, 0.1, 0.005, 0.01]),  # Weak uptrend
            2: np.array([0.0, 0.0, 0.0, 0.02, 0.01, 0.01, 0.0, 0.0, 0.01, 0.02,
                        0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0]),  # Sideways
            3: np.array([-0.01, -0.05, -0.1, -0.2, 0.01, 0.015, -0.15, -0.03, 0.05, 0.08,
                        0.1, 0.12, -0.04, -0.06, 0.01, 0.005, -0.08, -0.1, 0.005, 0.01]),  # Weak downtrend
            4: np.array([-0.03, -0.15, -0.3, -0.5, 0.03, 0.04, -0.3, -0.05, 0.15, 0.2,
                        0.25, 0.3, -0.1, -0.15, 0.03, 0.02, -0.2, -0.25, 0.02, 0.03]),  # Strong downtrend
        }

        pattern = base_patterns[regime_idx]
        noise = np.random.randn(n_samples, input_dim) * 0.02
        return pattern + noise

    # Create support and query sets
    support_features_list = []
    support_labels_list = []
    query_features_list = []
    query_labels_list = []

    class_names = {
        0: "STRONG_UPTREND",
        1: "WEAK_UPTREND",
        2: "SIDEWAYS",
        3: "WEAK_DOWNTREND",
        4: "STRONG_DOWNTREND"
    }

    for regime_idx in range(n_classes):
        support_data = generate_regime_data(regime_idx, n_support_per_class)
        query_data = generate_regime_data(regime_idx, n_query_per_class)

        support_features_list.append(support_data)
        support_labels_list.extend([regime_idx] * n_support_per_class)

        query_features_list.append(query_data)
        query_labels_list.extend([regime_idx] * n_query_per_class)

    support_features = np.vstack(support_features_list)
    support_labels = np.array(support_labels_list)
    query_features = np.vstack(query_features_list)
    query_labels = np.array(query_labels_list)

    print(f"   Support set shape: {support_features.shape}")
    print(f"   Query set shape: {query_features.shape}")

    # Test different methods
    methods = ["metric", "maml", "hybrid"]

    for method in methods:
        print(f"\n2. Testing {method.upper()} method...")

        # Create predictor
        predictor = FewShotPredictor(
            method=method,
            input_dim=input_dim,
            embedding_dim=16,
            n_way=n_classes,
            k_shot=n_support_per_class
        )

        # Fit on support set
        predictor.fit(support_features, support_labels, class_names)
        print(f"   Model fitted successfully")

        # Predict on query set
        predictions, confidences = predictor.predict(query_features)

        # Calculate accuracy
        accuracy = np.mean(predictions == query_labels)
        avg_confidence = np.mean(confidences)

        print(f"   Accuracy: {accuracy * 100:.1f}%")
        print(f"   Average confidence: {avg_confidence * 100:.1f}%")

    # Test detailed predictions
    print("\n3. Testing detailed regime predictions...")

    predictor = FewShotPredictor(
        method="metric",
        input_dim=input_dim,
        embedding_dim=16
    )
    predictor.fit(support_features, support_labels, class_names)

    regime_results = predictor.predict_regime(query_features[:3])

    for i, result in enumerate(regime_results):
        print(f"\n   Sample {i+1}:")
        print(f"     True: {class_names[query_labels[i]]}")
        print(f"     Predicted: {result['regime']}")
        print(f"     Confidence: {result['confidence']*100:.1f}%")
        print(f"     Recommendation: {result['recommendation']}")
        print(f"     Position size: {result['suggested_position_size']*100:.1f}%")

    # Test episodic training
    print("\n4. Testing episodic training setup...")

    all_features = np.vstack([support_features, query_features])
    all_labels = np.concatenate([support_labels, query_labels])

    episodes = create_episodes(
        all_features, all_labels,
        n_way=3, k_shot=5, n_query=3, n_episodes=10
    )

    print(f"   Created {len(episodes)} episodes")
    print(f"   Episode support shape: {episodes[0]['support_features'].shape}")
    print(f"   Episode query shape: {episodes[0]['query_features'].shape}")

    print("\n" + "=" * 60)
    print("Example Complete")
    print("=" * 60)
