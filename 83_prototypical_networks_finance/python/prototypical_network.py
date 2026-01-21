"""
Prototypical Networks for Market Regime Classification

This module implements prototypical networks for few-shot learning
applied to financial market regime classification.

Example usage:
    ```python
    from prototypical_network import PrototypicalNetwork, MarketRegimeClassifier

    # Create the network
    network = PrototypicalNetwork(
        input_dim=15,
        hidden_dims=[64, 32],
        embedding_dim=16
    )

    # Create classifier
    classifier = MarketRegimeClassifier(network)

    # Train on support set
    classifier.fit(support_features, support_labels)

    # Classify new data
    predictions, confidences = classifier.predict(query_features)
    ```
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings


class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_UPTREND = 0
    WEAK_UPTREND = 1
    SIDEWAYS = 2
    WEAK_DOWNTREND = 3
    STRONG_DOWNTREND = 4

    @property
    def trading_bias(self) -> float:
        """Get trading bias (-1 to 1)."""
        biases = {
            MarketRegime.STRONG_UPTREND: 1.0,
            MarketRegime.WEAK_UPTREND: 0.5,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.WEAK_DOWNTREND: -0.5,
            MarketRegime.STRONG_DOWNTREND: -1.0,
        }
        return biases[self]


class DistanceFunction(Enum):
    """Distance functions for prototype comparison."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding network."""
    input_dim: int
    hidden_dims: List[int]
    embedding_dim: int
    use_layer_norm: bool = True
    dropout_rate: float = 0.1
    activation: str = "relu"


class EmbeddingNetwork:
    """
    Neural network for embedding market features.

    This network transforms raw features into a representation space
    where prototypical learning is performed.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding network.

        Args:
            config: Network configuration
        """
        self.config = config
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        self.weights = []
        self.biases = []

        dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.embedding_dim]

        for i in range(len(dims) - 1):
            # Xavier initialization
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            w = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.config.activation == "relu":
            return np.maximum(0, x)
        elif self.config.activation == "tanh":
            return np.tanh(x)
        elif self.config.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input features, shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Embeddings, shape (batch_size, embedding_dim) or (embedding_dim,)
        """
        single_sample = x.ndim == 1
        if single_sample:
            x = x.reshape(1, -1)

        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            # Apply activation to all but last layer
            if i < len(self.weights) - 1:
                h = self._activation(h)
                if self.config.use_layer_norm:
                    h = self._layer_norm(h)

        # L2 normalize embeddings
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        h = h / norms

        if single_sample:
            h = h.squeeze(0)

        return h

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)


class PrototypeComputer:
    """
    Computes and manages class prototypes.

    Prototypes are the centroid embeddings for each class,
    computed from support set examples.
    """

    def __init__(self, distance_fn: DistanceFunction = DistanceFunction.EUCLIDEAN):
        """
        Initialize the prototype computer.

        Args:
            distance_fn: Distance function to use
        """
        self.distance_fn = distance_fn
        self.prototypes: Dict[int, np.ndarray] = {}
        self.class_embeddings: Dict[int, List[np.ndarray]] = {}

    def add_class_examples(self, class_idx: int, embeddings: np.ndarray):
        """
        Add embeddings for a class.

        Args:
            class_idx: Class index
            embeddings: Array of shape (n_samples, embedding_dim)
        """
        if class_idx not in self.class_embeddings:
            self.class_embeddings[class_idx] = []
        self.class_embeddings[class_idx].extend(embeddings)

    def compute_prototypes(self):
        """Compute prototypes as centroids of class embeddings."""
        for class_idx, embeddings in self.class_embeddings.items():
            if embeddings:
                self.prototypes[class_idx] = np.mean(embeddings, axis=0)

    def _compute_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.distance_fn == DistanceFunction.EUCLIDEAN:
            return np.linalg.norm(a - b)
        elif self.distance_fn == DistanceFunction.COSINE:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                return 1.0
            return 1.0 - np.dot(a, b) / (norm_a * norm_b)
        elif self.distance_fn == DistanceFunction.MANHATTAN:
            return np.sum(np.abs(a - b))
        else:
            return np.linalg.norm(a - b)

    def classify(self, query: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Classify a query embedding.

        Args:
            query: Query embedding

        Returns:
            Tuple of (predicted_class, probabilities)
        """
        if not self.prototypes:
            raise ValueError("No prototypes computed. Call compute_prototypes() first.")

        distances = {}
        for class_idx, prototype in self.prototypes.items():
            distances[class_idx] = self._compute_distance(query, prototype)

        # Convert distances to probabilities using softmax
        min_dist_class = min(distances, key=distances.get)

        # Softmax on negative distances
        neg_dists = np.array([-distances[i] for i in sorted(distances.keys())])
        exp_dists = np.exp(neg_dists - np.max(neg_dists))  # Numerical stability
        probabilities = exp_dists / np.sum(exp_dists)

        return min_dist_class, probabilities


class PrototypicalNetwork:
    """
    Prototypical Network for few-shot learning.

    This class combines the embedding network with prototype computation
    for few-shot classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        embedding_dim: int = 64,
        distance_fn: DistanceFunction = DistanceFunction.EUCLIDEAN,
    ):
        """
        Initialize the prototypical network.

        Args:
            input_dim: Dimension of input features
            hidden_dims: Dimensions of hidden layers
            embedding_dim: Dimension of embedding space
            distance_fn: Distance function for prototype comparison
        """
        if hidden_dims is None:
            hidden_dims = [128, 64]

        config = EmbeddingConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
        )
        self.embedding_network = EmbeddingNetwork(config)
        self.distance_fn = distance_fn

    def compute_prototypes(
        self,
        support_features: np.ndarray,
        support_labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Compute prototypes from support set.

        Args:
            support_features: Support set features, shape (n_support, input_dim)
            support_labels: Support set labels, shape (n_support,)

        Returns:
            Dictionary mapping class indices to prototype embeddings
        """
        computer = PrototypeComputer(self.distance_fn)

        # Embed support features
        embeddings = self.embedding_network.forward(support_features)

        # Group by class and add to computer
        for class_idx in np.unique(support_labels):
            mask = support_labels == class_idx
            class_embeddings = embeddings[mask]
            computer.add_class_examples(class_idx, class_embeddings)

        computer.compute_prototypes()
        return computer.prototypes

    def predict(
        self,
        query_features: np.ndarray,
        prototypes: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes for query features.

        Args:
            query_features: Query features, shape (n_query, input_dim)
            prototypes: Dictionary of class prototypes

        Returns:
            Tuple of (predictions, probabilities)
        """
        computer = PrototypeComputer(self.distance_fn)
        computer.prototypes = prototypes

        # Embed query features
        embeddings = self.embedding_network.forward(query_features)

        predictions = []
        all_probs = []

        for embedding in embeddings:
            pred, probs = computer.classify(embedding)
            predictions.append(pred)
            all_probs.append(probs)

        return np.array(predictions), np.array(all_probs)


class MarketRegimeClassifier:
    """
    Classifier for market regime detection using prototypical networks.

    This class provides a high-level interface for market regime classification
    with support for incremental updates and confidence estimation.
    """

    def __init__(
        self,
        network: Optional[PrototypicalNetwork] = None,
        input_dim: int = 15,
        embedding_dim: int = 32,
        outlier_threshold: float = 10.0,
    ):
        """
        Initialize the market regime classifier.

        Args:
            network: Pre-configured prototypical network (optional)
            input_dim: Dimension of input features (used if network not provided)
            embedding_dim: Dimension of embedding space (used if network not provided)
            outlier_threshold: Distance threshold for outlier detection
        """
        if network is None:
            network = PrototypicalNetwork(
                input_dim=input_dim,
                hidden_dims=[64, 32],
                embedding_dim=embedding_dim,
            )
        self.network = network
        self.prototypes: Optional[Dict[int, np.ndarray]] = None
        self.outlier_threshold = outlier_threshold

    def fit(self, support_features: np.ndarray, support_labels: np.ndarray):
        """
        Fit the classifier on support set.

        Args:
            support_features: Support set features
            support_labels: Support set labels (MarketRegime indices)
        """
        self.prototypes = self.network.compute_prototypes(
            support_features, support_labels
        )

    def predict(
        self,
        query_features: np.ndarray,
    ) -> Tuple[List[MarketRegime], np.ndarray]:
        """
        Predict market regimes.

        Args:
            query_features: Query features

        Returns:
            Tuple of (regimes, confidences)
        """
        if self.prototypes is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        predictions, probabilities = self.network.predict(query_features, self.prototypes)

        regimes = [MarketRegime(p) for p in predictions]
        confidences = np.max(probabilities, axis=1)

        return regimes, confidences

    def predict_with_details(
        self,
        query_features: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Predict with detailed results.

        Args:
            query_features: Query features

        Returns:
            List of prediction details for each sample
        """
        if self.prototypes is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        predictions, probabilities = self.network.predict(query_features, self.prototypes)

        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            regime = MarketRegime(pred)
            confidence = probs[pred]

            # Sort regimes by probability
            regime_probs = [
                (MarketRegime(j), probs[j])
                for j in range(len(MarketRegime))
                if j < len(probs)
            ]
            regime_probs.sort(key=lambda x: x[1], reverse=True)

            results.append({
                "regime": regime,
                "confidence": confidence,
                "probabilities": dict(regime_probs),
                "trading_bias": regime.trading_bias,
                "is_uncertain": confidence < 0.4,
            })

        return results


# Feature extraction utilities

def extract_features(
    prices: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    window_sizes: List[int] = None,
) -> np.ndarray:
    """
    Extract features from price data.

    Args:
        prices: Price array (close prices)
        volumes: Volume array (optional)
        window_sizes: Window sizes for moving averages

    Returns:
        Feature array
    """
    if window_sizes is None:
        window_sizes = [5, 10, 20]

    features = []

    # Returns
    returns = np.diff(prices) / prices[:-1]
    if len(returns) > 0:
        features.append(returns[-1])  # Latest return

        # Cumulative returns
        for window in window_sizes:
            if len(returns) >= window:
                features.append(np.sum(returns[-window:]))
            else:
                features.append(0.0)

        # Volatility
        if len(returns) >= 20:
            features.append(np.std(returns[-20:]))
        else:
            features.append(0.0)

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
            features.append(rsi / 100.0 - 0.5)
        else:
            features.append(0.0)

    # Moving average features
    for window in window_sizes:
        if len(prices) >= window:
            ma = np.mean(prices[-window:])
            features.append(prices[-1] / ma - 1.0)
        else:
            features.append(0.0)

    # Volume features
    if volumes is not None and len(volumes) >= 20:
        vol_ma = np.mean(volumes[-20:])
        if vol_ma > 0:
            features.append(volumes[-1] / vol_ma - 1.0)
        else:
            features.append(0.0)
    else:
        features.append(0.0)

    return np.array(features)


# Example usage and testing

if __name__ == "__main__":
    print("=== Prototypical Networks for Market Regime Classification ===\n")

    # Configuration
    np.random.seed(42)
    n_classes = 5
    n_support = 20
    n_query = 10
    input_dim = 15

    # Generate synthetic data
    print("1. Generating synthetic market data...")

    def generate_regime_data(regime_idx: int, n_samples: int) -> np.ndarray:
        """Generate data for a specific regime."""
        base_pattern = np.array([
            [0.03, 0.15, 0.3, 0.6, 0.02, 0.1, 0.2, 0.7, 0.15, 0.1, 0.2, 0.3, 0.1, 0.5, 0.03],   # Strong uptrend
            [0.01, 0.05, 0.1, 0.2, 0.015, 0.05, 0.1, 0.55, 0.1, 0.05, 0.15, 0.2, 0.05, 0.4, 0.01],  # Weak uptrend
            [0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.5, 0.08, 0.0, 0.1, 0.1, 0.0, 0.3, 0.0],      # Sideways
            [-0.01, -0.05, -0.1, -0.2, 0.015, -0.05, -0.1, 0.45, 0.1, -0.05, 0.15, 0.2, -0.05, 0.2, -0.01],  # Weak downtrend
            [-0.03, -0.15, -0.3, -0.6, 0.04, -0.1, -0.2, 0.3, 0.2, -0.1, 0.3, 0.4, -0.1, 0.1, -0.03],   # Strong downtrend
        ])
        pattern = base_pattern[regime_idx]
        noise = np.random.randn(n_samples, input_dim) * 0.05
        return pattern + noise

    # Create support and query sets
    support_features = []
    support_labels = []
    query_features = []
    query_labels = []

    for regime_idx in range(n_classes):
        regime_support = generate_regime_data(regime_idx, n_support)
        regime_query = generate_regime_data(regime_idx, n_query)

        support_features.append(regime_support)
        support_labels.extend([regime_idx] * n_support)

        query_features.append(regime_query)
        query_labels.extend([regime_idx] * n_query)

    support_features = np.vstack(support_features)
    query_features = np.vstack(query_features)
    support_labels = np.array(support_labels)
    query_labels = np.array(query_labels)

    print(f"   Support set: {support_features.shape}")
    print(f"   Query set: {query_features.shape}")

    # Create and fit classifier
    print("\n2. Creating and fitting classifier...")
    classifier = MarketRegimeClassifier(
        input_dim=input_dim,
        embedding_dim=16,
    )
    classifier.fit(support_features, support_labels)
    print("   Classifier fitted successfully")

    # Make predictions
    print("\n3. Making predictions...")
    regimes, confidences = classifier.predict(query_features)

    # Calculate accuracy
    predictions = [r.value for r in regimes]
    accuracy = np.mean(predictions == query_labels)
    print(f"   Accuracy: {accuracy * 100:.1f}%")
    print(f"   Average confidence: {np.mean(confidences) * 100:.1f}%")

    # Show detailed results for a few samples
    print("\n4. Sample predictions:")
    details = classifier.predict_with_details(query_features[:5])
    for i, detail in enumerate(details):
        true_regime = MarketRegime(query_labels[i])
        print(f"   Sample {i+1}: True={true_regime.name}, "
              f"Pred={detail['regime'].name}, "
              f"Conf={detail['confidence']*100:.1f}%, "
              f"Bias={detail['trading_bias']:.1f}")

    # Test with different distance functions
    print("\n5. Testing distance functions:")
    for dist_fn in [DistanceFunction.EUCLIDEAN, DistanceFunction.COSINE, DistanceFunction.MANHATTAN]:
        network = PrototypicalNetwork(
            input_dim=input_dim,
            embedding_dim=16,
            distance_fn=dist_fn,
        )
        clf = MarketRegimeClassifier(network=network)
        clf.fit(support_features, support_labels)
        _, conf = clf.predict(query_features)
        print(f"   {dist_fn.value}: Avg confidence = {np.mean(conf)*100:.1f}%")

    print("\n=== Example Complete ===")
