"""
Matching Networks for Market Pattern Classification

This module implements Matching Networks for few-shot learning
applied to financial market pattern classification.

Matching Networks use an attention mechanism over support set examples
instead of computing class prototypes, making them effective for
recognizing diverse market patterns.

Example usage:
    ```python
    from matching_network import MatchingNetwork, MarketPatternClassifier

    # Create the network
    network = MatchingNetwork(
        input_dim=15,
        hidden_dim=64,
        embedding_dim=64,
        use_fce=True,
    )

    # Create classifier
    classifier = MarketPatternClassifier(network)

    # Train on support set
    classifier.fit(support_features, support_labels)

    # Classify new patterns
    predictions, attention = classifier.predict(query_features)
    ```

References:
    - Vinyals et al., "Matching Networks for One Shot Learning", NeurIPS 2016
    - https://arxiv.org/abs/1606.04080
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import warnings


class MarketPattern(Enum):
    """Market pattern classification."""
    TREND_CONTINUATION = 0
    TREND_REVERSAL = 1
    BREAKOUT = 2
    FALSE_BREAKOUT = 3
    CONSOLIDATION = 4

    @property
    def trading_bias(self) -> float:
        """Get trading bias for the pattern (-1 to 1)."""
        biases = {
            MarketPattern.TREND_CONTINUATION: 0.8,
            MarketPattern.TREND_REVERSAL: -0.8,
            MarketPattern.BREAKOUT: 0.6,
            MarketPattern.FALSE_BREAKOUT: -0.4,
            MarketPattern.CONSOLIDATION: 0.0,
        }
        return biases[self]

    @property
    def description(self) -> str:
        """Get pattern description."""
        descriptions = {
            MarketPattern.TREND_CONTINUATION: "Trend continues in current direction",
            MarketPattern.TREND_REVERSAL: "Trend is likely to reverse",
            MarketPattern.BREAKOUT: "Price breaking out of range",
            MarketPattern.FALSE_BREAKOUT: "Fake breakout, likely to reverse",
            MarketPattern.CONSOLIDATION: "Price consolidating, wait for direction",
        }
        return descriptions[self]


class DistanceFunction(Enum):
    """Distance functions for similarity computation."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding network."""
    input_dim: int
    hidden_dims: List[int]
    embedding_dim: int
    use_layer_norm: bool = True
    dropout_rate: float = 0.1
    activation: str = "relu"


@dataclass
class FCEConfig:
    """Configuration for Full Context Embeddings."""
    use_fce: bool = True
    lstm_hidden_dim: int = 64
    fce_steps: int = 5
    bidirectional: bool = True


class EmbeddingNetwork:
    """
    Neural network for embedding market features.

    This network transforms raw features into a representation space
    where matching is performed via attention.
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

        # L2 normalize embeddings for cosine similarity
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


class SimpleLSTM:
    """
    Simple LSTM implementation for FCE.

    Note: This is a simplified implementation for demonstration.
    In production, use PyTorch or TensorFlow.
    """

    def __init__(self, input_dim: int, hidden_dim: int, bidirectional: bool = False):
        """
        Initialize LSTM.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            bidirectional: Whether to use bidirectional LSTM
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Initialize weights for input, forget, cell, output gates
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights."""
        scale = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))

        # Combined weights for all gates [i, f, g, o]
        self.W_ih = np.random.randn(4 * self.hidden_dim, self.input_dim) * scale
        self.W_hh = np.random.randn(4 * self.hidden_dim, self.hidden_dim) * scale
        self.b_ih = np.zeros(4 * self.hidden_dim)
        self.b_hh = np.zeros(4 * self.hidden_dim)

        if self.bidirectional:
            self.W_ih_rev = np.random.randn(4 * self.hidden_dim, self.input_dim) * scale
            self.W_hh_rev = np.random.randn(4 * self.hidden_dim, self.hidden_dim) * scale
            self.b_ih_rev = np.zeros(4 * self.hidden_dim)
            self.b_hh_rev = np.zeros(4 * self.hidden_dim)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _step(
        self,
        x: np.ndarray,
        h: np.ndarray,
        c: np.ndarray,
        W_ih: np.ndarray,
        W_hh: np.ndarray,
        b_ih: np.ndarray,
        b_hh: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single LSTM step.

        Args:
            x: Input at current timestep
            h: Hidden state
            c: Cell state
            W_ih, W_hh, b_ih, b_hh: Weights and biases

        Returns:
            New hidden state and cell state
        """
        gates = W_ih @ x + b_ih + W_hh @ h + b_hh

        i, f, g, o = np.split(gates, 4)

        i = self._sigmoid(i)  # Input gate
        f = self._sigmoid(f)  # Forget gate
        g = np.tanh(g)        # Cell candidate
        o = self._sigmoid(o)  # Output gate

        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)

        return h_new, c_new

    def forward(self, sequence: np.ndarray) -> np.ndarray:
        """
        Forward pass through LSTM.

        Args:
            sequence: Input sequence, shape (seq_len, input_dim)

        Returns:
            Output sequence, shape (seq_len, hidden_dim) or (seq_len, 2*hidden_dim) if bidirectional
        """
        seq_len = sequence.shape[0]

        # Forward pass
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        forward_outputs = []

        for t in range(seq_len):
            h, c = self._step(sequence[t], h, c, self.W_ih, self.W_hh, self.b_ih, self.b_hh)
            forward_outputs.append(h)

        forward_outputs = np.array(forward_outputs)

        if not self.bidirectional:
            return forward_outputs

        # Backward pass
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        backward_outputs = []

        for t in range(seq_len - 1, -1, -1):
            h, c = self._step(
                sequence[t], h, c,
                self.W_ih_rev, self.W_hh_rev,
                self.b_ih_rev, self.b_hh_rev
            )
            backward_outputs.insert(0, h)

        backward_outputs = np.array(backward_outputs)

        # Concatenate forward and backward
        return np.concatenate([forward_outputs, backward_outputs], axis=1)


class FullContextEmbedding:
    """
    Full Context Embedding (FCE) module.

    FCE conditions the support and query embeddings on the entire support set,
    allowing context-aware representations.
    """

    def __init__(self, embedding_dim: int, config: FCEConfig):
        """
        Initialize FCE module.

        Args:
            embedding_dim: Dimension of base embeddings
            config: FCE configuration
        """
        self.embedding_dim = embedding_dim
        self.config = config

        if config.use_fce:
            # Support set encoder (bidirectional LSTM)
            self.support_lstm = SimpleLSTM(
                input_dim=embedding_dim,
                hidden_dim=config.lstm_hidden_dim,
                bidirectional=config.bidirectional,
            )

            # Projection to original embedding dimension
            lstm_output_dim = config.lstm_hidden_dim * (2 if config.bidirectional else 1)
            self.support_proj = np.random.randn(lstm_output_dim, embedding_dim) * 0.1

            # Query encoder LSTM (for attention steps)
            self.query_lstm_W_ih = np.random.randn(
                4 * embedding_dim, embedding_dim * 2
            ) * 0.1
            self.query_lstm_W_hh = np.random.randn(
                4 * embedding_dim, embedding_dim
            ) * 0.1
            self.query_lstm_b = np.zeros(4 * embedding_dim)

    def encode_support(self, base_embeddings: np.ndarray) -> np.ndarray:
        """
        Encode support set with full context.

        Args:
            base_embeddings: Base embeddings, shape (n_support, embedding_dim)

        Returns:
            Context-aware embeddings, shape (n_support, embedding_dim)
        """
        if not self.config.use_fce:
            return base_embeddings

        # Pass through bidirectional LSTM
        lstm_out = self.support_lstm.forward(base_embeddings)

        # Project back to embedding dimension
        projected = lstm_out @ self.support_proj

        # Add skip connection
        return projected + base_embeddings

    def encode_query(
        self,
        base_embedding: np.ndarray,
        support_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Encode query with attention over support set.

        Args:
            base_embedding: Base query embedding, shape (embedding_dim,)
            support_embeddings: Support set embeddings, shape (n_support, embedding_dim)

        Returns:
            Context-aware query embedding, shape (embedding_dim,)
        """
        if not self.config.use_fce:
            return base_embedding

        h = base_embedding.copy()
        c = np.zeros_like(h)

        for _ in range(self.config.fce_steps):
            # Attention readout
            r = self._attention_readout(h, support_embeddings)

            # LSTM input: concatenate query base and readout
            lstm_input = np.concatenate([base_embedding, r])

            # LSTM step
            h, c = self._lstm_step(lstm_input, h, c)

            # Skip connection
            h = h + base_embedding

        return h

    def _attention_readout(
        self,
        query: np.ndarray,
        support_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute attention-weighted readout from support embeddings."""
        # Cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        support_norms = support_embeddings / (
            np.linalg.norm(support_embeddings, axis=1, keepdims=True) + 1e-8
        )

        similarities = support_norms @ query_norm

        # Softmax
        exp_sim = np.exp(similarities - np.max(similarities))
        attention = exp_sim / (np.sum(exp_sim) + 1e-8)

        # Weighted sum
        return attention @ support_embeddings

    def _lstm_step(
        self,
        x: np.ndarray,
        h: np.ndarray,
        c: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single LSTM step for query encoding."""
        gates = self.query_lstm_W_ih @ x + self.query_lstm_W_hh @ h + self.query_lstm_b

        dim = len(h)
        i, f, g, o = gates[:dim], gates[dim:2*dim], gates[2*dim:3*dim], gates[3*dim:]

        sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))

        i = sigmoid(i)
        f = sigmoid(f)
        g = np.tanh(g)
        o = sigmoid(o)

        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)

        return h_new, c_new


class MatchingNetwork:
    """
    Matching Network for few-shot learning.

    This class combines the embedding network with attention-based
    classification for few-shot pattern recognition.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        embedding_dim: int = 64,
        use_fce: bool = True,
        fce_steps: int = 5,
        distance_fn: DistanceFunction = DistanceFunction.COSINE,
    ):
        """
        Initialize the matching network.

        Args:
            input_dim: Dimension of input features
            hidden_dims: Dimensions of hidden layers
            embedding_dim: Dimension of embedding space
            use_fce: Whether to use Full Context Embeddings
            fce_steps: Number of FCE processing steps for query
            distance_fn: Distance function for similarity computation
        """
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Embedding network
        embed_config = EmbeddingConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
        )
        self.embedding_network = EmbeddingNetwork(embed_config)

        # FCE module
        fce_config = FCEConfig(
            use_fce=use_fce,
            lstm_hidden_dim=embedding_dim,
            fce_steps=fce_steps,
        )
        self.fce = FullContextEmbedding(embedding_dim, fce_config)

        self.distance_fn = distance_fn
        self.embedding_dim = embedding_dim

    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between two vectors."""
        if self.distance_fn == DistanceFunction.COSINE:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)
        elif self.distance_fn == DistanceFunction.DOT_PRODUCT:
            return np.dot(a, b)
        elif self.distance_fn == DistanceFunction.EUCLIDEAN:
            # Convert distance to similarity
            dist = np.linalg.norm(a - b)
            return 1.0 / (1.0 + dist)
        else:
            return np.dot(a, b)

    def encode_support(self, support_features: np.ndarray) -> np.ndarray:
        """
        Encode support set with optional FCE.

        Args:
            support_features: Support features, shape (n_support, input_dim)

        Returns:
            Support embeddings, shape (n_support, embedding_dim)
        """
        # Get base embeddings
        base_embeddings = self.embedding_network.forward(support_features)

        # Apply FCE
        return self.fce.encode_support(base_embeddings)

    def encode_query(
        self,
        query_features: np.ndarray,
        support_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Encode query with attention over support set.

        Args:
            query_features: Query features, shape (input_dim,) or (n_query, input_dim)
            support_embeddings: Support embeddings, shape (n_support, embedding_dim)

        Returns:
            Query embeddings, shape (embedding_dim,) or (n_query, embedding_dim)
        """
        single_query = query_features.ndim == 1
        if single_query:
            query_features = query_features.reshape(1, -1)

        # Get base embeddings
        base_embeddings = self.embedding_network.forward(query_features)

        # Apply FCE for each query
        query_embeddings = []
        for i in range(base_embeddings.shape[0]):
            q_emb = self.fce.encode_query(base_embeddings[i], support_embeddings)
            query_embeddings.append(q_emb)

        query_embeddings = np.array(query_embeddings)

        if single_query:
            return query_embeddings.squeeze(0)

        return query_embeddings

    def compute_attention(
        self,
        query_embeddings: np.ndarray,
        support_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute attention weights from query to support examples.

        Args:
            query_embeddings: Query embeddings, shape (n_query, embedding_dim)
            support_embeddings: Support embeddings, shape (n_support, embedding_dim)

        Returns:
            Attention weights, shape (n_query, n_support)
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        n_query = query_embeddings.shape[0]
        n_support = support_embeddings.shape[0]

        # Compute similarities
        similarities = np.zeros((n_query, n_support))
        for i in range(n_query):
            for j in range(n_support):
                similarities[i, j] = self._compute_similarity(
                    query_embeddings[i],
                    support_embeddings[j],
                )

        # Softmax to get attention weights
        exp_sim = np.exp(similarities - np.max(similarities, axis=1, keepdims=True))
        attention = exp_sim / (np.sum(exp_sim, axis=1, keepdims=True) + 1e-8)

        return attention

    def forward(
        self,
        support_features: np.ndarray,
        support_labels: np.ndarray,
        query_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for classification.

        Args:
            support_features: Support features, shape (n_support, input_dim)
            support_labels: Support labels, shape (n_support,)
            query_features: Query features, shape (n_query, input_dim)

        Returns:
            Tuple of (class_probabilities, attention_weights)
            - class_probs: shape (n_query, n_classes)
            - attention: shape (n_query, n_support)
        """
        # Encode support set
        support_embeddings = self.encode_support(support_features)

        # Encode queries
        query_embeddings = self.encode_query(query_features, support_embeddings)

        # Compute attention
        attention = self.compute_attention(query_embeddings, support_embeddings)

        # Aggregate by class (weighted voting)
        n_classes = int(np.max(support_labels)) + 1
        n_query = query_embeddings.shape[0] if query_embeddings.ndim == 2 else 1

        if query_embeddings.ndim == 1:
            attention = attention.reshape(1, -1)

        class_probs = np.zeros((n_query, n_classes))
        for i in range(support_features.shape[0]):
            class_probs[:, support_labels[i]] += attention[:, i]

        return class_probs, attention


class MarketPatternClassifier:
    """
    Market pattern classifier using Matching Networks.

    Provides a scikit-learn-like interface for pattern classification.
    """

    def __init__(
        self,
        network: MatchingNetwork,
        pattern_names: List[str] = None,
    ):
        """
        Initialize the classifier.

        Args:
            network: Matching network instance
            pattern_names: Names for pattern classes
        """
        self.network = network
        self.pattern_names = pattern_names or [p.name for p in MarketPattern]

        self.support_features: Optional[np.ndarray] = None
        self.support_labels: Optional[np.ndarray] = None
        self.support_embeddings: Optional[np.ndarray] = None

    def fit(
        self,
        support_features: np.ndarray,
        support_labels: np.ndarray,
    ):
        """
        Fit the classifier with support set.

        Args:
            support_features: Support features, shape (n_support, input_dim)
            support_labels: Support labels, shape (n_support,)
        """
        self.support_features = support_features
        self.support_labels = support_labels.astype(int)

        # Pre-compute support embeddings
        self.support_embeddings = self.network.encode_support(support_features)

    def predict(
        self,
        query_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict patterns for query samples.

        Args:
            query_features: Query features, shape (n_query, input_dim)

        Returns:
            Tuple of (predictions, attention_weights)
        """
        if self.support_features is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        class_probs, attention = self.network.forward(
            self.support_features,
            self.support_labels,
            query_features,
        )

        predictions = np.argmax(class_probs, axis=1)

        return predictions, attention

    def predict_proba(
        self,
        query_features: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            query_features: Query features, shape (n_query, input_dim)

        Returns:
            Class probabilities, shape (n_query, n_classes)
        """
        if self.support_features is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        class_probs, _ = self.network.forward(
            self.support_features,
            self.support_labels,
            query_features,
        )

        return class_probs

    def interpret(
        self,
        query_features: np.ndarray,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get interpretable predictions with most similar support examples.

        Args:
            query_features: Query features, shape (n_query, input_dim)
            top_k: Number of top support examples to return

        Returns:
            List of interpretation dictionaries
        """
        if self.support_features is None:
            raise ValueError("Classifier not fitted. Call fit() first.")

        class_probs, attention = self.network.forward(
            self.support_features,
            self.support_labels,
            query_features,
        )

        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
            attention = attention.reshape(1, -1)

        interpretations = []
        for i in range(query_features.shape[0]):
            # Get prediction
            pred_class = np.argmax(class_probs[i])
            confidence = class_probs[i, pred_class]

            # Get top-k most similar support examples
            top_indices = np.argsort(attention[i])[::-1][:top_k]

            similar_examples = [
                {
                    "index": int(idx),
                    "label": int(self.support_labels[idx]),
                    "pattern_name": self.pattern_names[self.support_labels[idx]],
                    "attention_weight": float(attention[i, idx]),
                }
                for idx in top_indices
            ]

            # Compute attention entropy (uncertainty measure)
            entropy = -np.sum(attention[i] * np.log(attention[i] + 1e-8))

            interpretations.append({
                "predicted_class": int(pred_class),
                "predicted_pattern": self.pattern_names[pred_class],
                "confidence": float(confidence),
                "attention_entropy": float(entropy),
                "most_similar_examples": similar_examples,
                "class_probabilities": {
                    self.pattern_names[j]: float(class_probs[i, j])
                    for j in range(len(self.pattern_names))
                },
            })

        return interpretations


class EpisodicTrainer:
    """
    Episodic training for Matching Networks.

    Generates episodes (tasks) that simulate few-shot scenarios.
    """

    def __init__(
        self,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 10,
    ):
        """
        Initialize trainer.

        Args:
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

    def generate_episode(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate one training episode.

        Args:
            features: All features, shape (n_samples, input_dim)
            labels: All labels, shape (n_samples,)

        Returns:
            Tuple of (support_features, support_labels, query_features, query_labels)
        """
        # Get unique classes
        unique_classes = np.unique(labels)
        if len(unique_classes) < self.n_way:
            raise ValueError(f"Need at least {self.n_way} classes, got {len(unique_classes)}")

        # Sample n_way classes
        sampled_classes = np.random.choice(unique_classes, self.n_way, replace=False)

        support_features = []
        support_labels = []
        query_features = []
        query_labels = []

        for new_label, original_class in enumerate(sampled_classes):
            # Get indices for this class
            class_indices = np.where(labels == original_class)[0]

            if len(class_indices) < self.k_shot + self.n_query:
                warnings.warn(
                    f"Class {original_class} has {len(class_indices)} samples, "
                    f"need {self.k_shot + self.n_query}. Using with replacement."
                )
                sampled_indices = np.random.choice(
                    class_indices,
                    self.k_shot + self.n_query,
                    replace=True,
                )
            else:
                sampled_indices = np.random.choice(
                    class_indices,
                    self.k_shot + self.n_query,
                    replace=False,
                )

            # Split into support and query
            support_indices = sampled_indices[:self.k_shot]
            query_indices = sampled_indices[self.k_shot:]

            support_features.extend(features[support_indices])
            support_labels.extend([new_label] * self.k_shot)

            query_features.extend(features[query_indices])
            query_labels.extend([new_label] * self.n_query)

        return (
            np.array(support_features),
            np.array(support_labels),
            np.array(query_features),
            np.array(query_labels),
        )


# Feature extraction utilities
def extract_market_features(
    prices: np.ndarray,
    volumes: np.ndarray = None,
    window_size: int = 20,
) -> np.ndarray:
    """
    Extract features from price and volume data.

    Args:
        prices: Price array (close prices)
        volumes: Volume array (optional)
        window_size: Lookback window for feature computation

    Returns:
        Feature vector
    """
    features = []

    # Returns at different timeframes
    for period in [1, 5, 10, 20]:
        if len(prices) > period:
            ret = (prices[-1] - prices[-period-1]) / (prices[-period-1] + 1e-8)
            features.append(ret)
        else:
            features.append(0.0)

    # Volatility (std of returns)
    if len(prices) > window_size:
        returns = np.diff(prices[-window_size-1:]) / (prices[-window_size-1:-1] + 1e-8)
        features.append(np.std(returns))
    else:
        features.append(0.0)

    # Price position in range
    if len(prices) >= window_size:
        high = np.max(prices[-window_size:])
        low = np.min(prices[-window_size:])
        if high - low > 1e-8:
            features.append((prices[-1] - low) / (high - low))
        else:
            features.append(0.5)
    else:
        features.append(0.5)

    # Momentum (price vs moving average)
    if len(prices) >= window_size:
        ma = np.mean(prices[-window_size:])
        features.append((prices[-1] - ma) / (ma + 1e-8))
    else:
        features.append(0.0)

    # RSI approximation
    if len(prices) > 14:
        deltas = np.diff(prices[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss > 1e-8:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100 if avg_gain > 0 else 50
        features.append(rsi / 100)  # Normalize to [0, 1]
    else:
        features.append(0.5)

    # Volume features
    if volumes is not None and len(volumes) >= window_size:
        # Volume relative to average
        avg_vol = np.mean(volumes[-window_size:])
        features.append(volumes[-1] / (avg_vol + 1e-8))

        # Volume change
        if len(volumes) > 1:
            vol_change = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1e-8)
            features.append(vol_change)
        else:
            features.append(0.0)
    else:
        features.extend([1.0, 0.0])

    return np.array(features)


# Example usage and demo
if __name__ == "__main__":
    print("Matching Networks for Market Pattern Classification")
    print("=" * 60)

    # Create synthetic data for demonstration
    np.random.seed(42)

    n_samples = 100
    input_dim = 10
    n_classes = 5

    # Generate random features and labels
    features = np.random.randn(n_samples, input_dim)
    labels = np.random.randint(0, n_classes, n_samples)

    # Create matching network
    network = MatchingNetwork(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        embedding_dim=32,
        use_fce=True,
        fce_steps=3,
    )

    # Create classifier
    classifier = MarketPatternClassifier(
        network,
        pattern_names=[p.name for p in MarketPattern],
    )

    # Split into support and query
    n_support = 50
    support_features = features[:n_support]
    support_labels = labels[:n_support]
    query_features = features[n_support:]
    query_labels = labels[n_support:]

    # Fit classifier
    classifier.fit(support_features, support_labels)

    # Predict
    predictions, attention = classifier.predict(query_features)

    # Accuracy
    accuracy = np.mean(predictions == query_labels)
    print(f"\nAccuracy: {accuracy:.2%}")

    # Interpretation for first query
    interpretations = classifier.interpret(query_features[:1])
    print("\nInterpretation for first query:")
    print(f"  Predicted pattern: {interpretations[0]['predicted_pattern']}")
    print(f"  Confidence: {interpretations[0]['confidence']:.2%}")
    print(f"  Attention entropy: {interpretations[0]['attention_entropy']:.4f}")
    print("  Most similar examples:")
    for ex in interpretations[0]['most_similar_examples']:
        print(f"    - Example {ex['index']}: {ex['pattern_name']} (attention: {ex['attention_weight']:.3f})")

    # Episodic training demo
    print("\n" + "=" * 60)
    print("Episodic Training Demo")
    print("=" * 60)

    trainer = EpisodicTrainer(n_way=5, k_shot=5, n_query=5)

    # Generate an episode
    ep_support, ep_support_labels, ep_query, ep_query_labels = trainer.generate_episode(
        features, labels
    )

    print(f"Support set: {ep_support.shape}")
    print(f"Query set: {ep_query.shape}")

    # Classify within episode
    classifier.fit(ep_support, ep_support_labels)
    ep_preds, _ = classifier.predict(ep_query)
    ep_accuracy = np.mean(ep_preds == ep_query_labels)
    print(f"Episode accuracy: {ep_accuracy:.2%}")
