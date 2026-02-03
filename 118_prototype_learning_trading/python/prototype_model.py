"""
Prototype Learning Model for Trading.

This module implements prototype-based learning for market regime classification
and trading signal generation. Based on Prototypical Networks for few-shot learning,
adapted for financial time series.

References:
    Snell et al., "Prototypical Networks for Few-shot Learning"
    NeurIPS 2017. arXiv:1703.05175
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
from enum import IntEnum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(IntEnum):
    """Market regime classifications."""
    BULLISH = 0
    BEARISH = 1
    SIDEWAYS = 2
    HIGH_VOLATILITY = 3
    LOW_VOLATILITY = 4


@dataclass
class PrototypeConfig:
    """Configuration for PrototypeLearner."""
    embedding_dim: int = 64
    n_prototypes: int = 5
    encoder_type: str = 'cnn'  # 'cnn', 'lstm', 'transformer'
    sequence_length: int = 60
    input_dim: int = 10
    dropout: float = 0.1
    distance_type: str = 'euclidean'  # 'euclidean', 'cosine', 'mahalanobis'


class CNNEncoder(nn.Module):
    """
    Convolutional Neural Network encoder for time series.

    Extracts feature embeddings from price sequences using 1D convolutions.
    """

    def __init__(
        self,
        input_dim: int = 10,
        sequence_length: int = 60,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize CNN encoder.

        Args:
            input_dim: Number of input features per timestep
            sequence_length: Length of input sequences
            embedding_dim: Dimension of output embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

        # Calculate output size after convolutions
        conv_out_len = sequence_length // 8  # Three pooling layers
        self.fc = nn.Linear(128 * conv_out_len, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)

        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        # Transpose to (batch, input_dim, sequence_length) for Conv1d
        x = x.transpose(1, 2)

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten and project
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class LSTMEncoder(nn.Module):
    """
    LSTM encoder for time series.

    Extracts feature embeddings using bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int = 10,
        sequence_length: int = 60,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize LSTM encoder.

        Args:
            input_dim: Number of input features per timestep
            sequence_length: Length of input sequences
            embedding_dim: Dimension of output embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)

        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x)

        # Concatenate final hidden states from both directions
        h_forward = h_n[-2, :, :]  # Last forward layer
        h_backward = h_n[-1, :, :]  # Last backward layer
        h_combined = torch.cat([h_forward, h_backward], dim=1)

        # Project to embedding space
        x = self.dropout(h_combined)
        x = self.fc(x)

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for time series.

    Uses self-attention to capture long-range dependencies.
    """

    def __init__(
        self,
        input_dim: int = 10,
        sequence_length: int = 60,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize Transformer encoder.

        Args:
            input_dim: Number of input features per timestep
            sequence_length: Length of input sequences
            embedding_dim: Dimension of output embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, embedding_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, sequence_length, embedding_dim) * 0.1
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)

        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        # Project input
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Transformer forward
        x = self.transformer(x)

        # Use mean pooling over sequence
        x = x.mean(dim=1)

        # Final projection
        x = self.dropout(x)
        x = self.fc(x)

        return x


class PrototypeLearner(nn.Module):
    """
    Prototype-based learning model for market regime classification.

    This model learns prototype representations for different market regimes
    and classifies new samples based on their distance to prototypes.

    Key features:
    - Multiple encoder architectures (CNN, LSTM, Transformer)
    - Online prototype updating for adaptation
    - Support for few-shot learning with support sets
    - Market regime detection
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_prototypes: int = 5,
        encoder_type: str = 'cnn',
        sequence_length: int = 60,
        input_dim: int = 10,
        dropout: float = 0.1,
        distance_type: str = 'euclidean',
    ):
        """
        Initialize PrototypeLearner.

        Args:
            embedding_dim: Dimension of prototype embeddings
            n_prototypes: Number of prototypes (one per class/regime)
            encoder_type: Type of encoder ('cnn', 'lstm', 'transformer')
            sequence_length: Length of input sequences
            input_dim: Number of input features
            dropout: Dropout rate
            distance_type: Distance metric ('euclidean', 'cosine', 'mahalanobis')
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_prototypes = n_prototypes
        self.encoder_type = encoder_type
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.distance_type = distance_type

        # Create encoder
        if encoder_type == 'cnn':
            self.encoder = CNNEncoder(
                input_dim=input_dim,
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(
                input_dim=input_dim,
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        elif encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                input_dim=input_dim,
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Learnable prototypes
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, embedding_dim) * 0.1
        )

        # For Mahalanobis distance
        if distance_type == 'mahalanobis':
            self.cov_matrix = nn.Parameter(
                torch.eye(embedding_dim)
            )

        # Prototype class mapping
        self.register_buffer('prototype_labels', torch.arange(n_prototypes))

        # Statistics for online updates
        self.register_buffer('prototype_counts', torch.zeros(n_prototypes))
        self.register_buffer('running_mean', torch.zeros(n_prototypes, embedding_dim))

        logger.info(
            f"PrototypeLearner initialized: encoder={encoder_type}, "
            f"embedding_dim={embedding_dim}, n_prototypes={n_prototypes}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequences to embeddings.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)

        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        return self.encoder(x)

    def compute_distances(
        self,
        embeddings: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute distances between embeddings and prototypes.

        Args:
            embeddings: Shape (batch, embedding_dim)
            prototypes: Shape (n_prototypes, embedding_dim), uses self.prototypes if None

        Returns:
            Distances of shape (batch, n_prototypes)
        """
        if prototypes is None:
            prototypes = self.prototypes

        if self.distance_type == 'euclidean':
            # Squared Euclidean distance
            distances = torch.cdist(embeddings, prototypes, p=2) ** 2
        elif self.distance_type == 'cosine':
            # Cosine distance (1 - cosine similarity)
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1)
            similarity = torch.mm(embeddings_norm, prototypes_norm.t())
            distances = 1 - similarity
        elif self.distance_type == 'mahalanobis':
            # Mahalanobis distance
            diff = embeddings.unsqueeze(1) - prototypes.unsqueeze(0)
            distances = torch.einsum('bij,jk,bik->bi', diff, self.cov_matrix, diff)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

        return distances

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)

        Returns:
            Tuple of (logits, embeddings, distances)
        """
        # Encode input
        embeddings = self.encode(x)

        # Compute distances to prototypes
        distances = self.compute_distances(embeddings)

        # Convert distances to logits (negative distance = higher similarity)
        logits = -distances

        return logits, embeddings, distances

    def fit(
        self,
        support_set: Tuple[torch.Tensor, torch.Tensor],
        n_episodes: int = 100,
        lr: float = 0.001,
    ) -> Dict[str, List[float]]:
        """
        Fit prototypes from labeled support set.

        Uses episodic training to learn prototype representations.

        Args:
            support_set: Tuple of (X, y) where X is (n_samples, seq_len, input_dim)
                        and y is (n_samples,) with class labels
            n_episodes: Number of training episodes
            lr: Learning rate

        Returns:
            Training history dictionary
        """
        X, y = support_set

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long)

        device = next(self.parameters()).device
        X = X.to(device)
        y = y.to(device)

        # Initialize prototypes from class means
        self._initialize_prototypes_from_data(X, y)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {'loss': [], 'accuracy': []}

        self.train()
        for episode in range(n_episodes):
            optimizer.zero_grad()

            # Forward pass
            logits, _, distances = self.forward(X)

            # Cross-entropy loss
            loss = F.cross_entropy(logits, y)

            # Add clustering loss (encourage samples close to their prototypes)
            cluster_loss = self._compute_cluster_loss(distances, y)
            total_loss = loss + 0.1 * cluster_loss

            # Backward
            total_loss.backward()
            optimizer.step()

            # Track metrics
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                accuracy = (preds == y).float().mean().item()

            history['loss'].append(total_loss.item())
            history['accuracy'].append(accuracy)

            if (episode + 1) % 20 == 0:
                logger.info(
                    f"Episode {episode + 1}/{n_episodes}: "
                    f"loss={total_loss.item():.4f}, accuracy={accuracy:.4f}"
                )

        return history

    def _initialize_prototypes_from_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """Initialize prototypes as class centroids."""
        with torch.no_grad():
            embeddings = self.encode(X)
            for c in range(self.n_prototypes):
                mask = y == c
                if mask.sum() > 0:
                    self.prototypes[c] = embeddings[mask].mean(dim=0)

    def _compute_cluster_loss(
        self,
        distances: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clustering loss to encourage tight clusters."""
        batch_size = distances.size(0)

        # Get distance to correct prototype for each sample
        correct_distances = distances[torch.arange(batch_size), y]

        return correct_distances.mean()

    def predict(
        self,
        query_set: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Classify query set samples by distance to prototypes.

        Args:
            query_set: Input tensor of shape (n_samples, seq_len, input_dim)

        Returns:
            Dictionary with predictions, confidences, and distances
        """
        if isinstance(query_set, np.ndarray):
            query_set = torch.tensor(query_set, dtype=torch.float32)

        device = next(self.parameters()).device
        query_set = query_set.to(device)

        self.eval()
        with torch.no_grad():
            logits, embeddings, distances = self.forward(query_set)

            # Softmax for probabilities
            probs = F.softmax(logits, dim=1)
            confidences, predictions = probs.max(dim=1)

        return {
            'predictions': predictions.cpu().numpy(),
            'confidences': confidences.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'distances': distances.cpu().numpy(),
            'embeddings': embeddings.cpu().numpy(),
        }

    def get_regime(
        self,
        market_data,  # MarketData object
        lookback: Optional[int] = None,
    ) -> Dict[str, Union[int, float, str]]:
        """
        Detect current market regime from market data.

        Args:
            market_data: MarketData object with price data
            lookback: Number of periods to use (default: sequence_length)

        Returns:
            Dictionary with regime information
        """
        if lookback is None:
            lookback = self.sequence_length

        # Get features from market data
        features = market_data.to_features(normalize=True)

        if len(features) < lookback:
            logger.warning(f"Insufficient data: {len(features)} < {lookback}")
            return {
                'regime': -1,
                'regime_name': 'UNKNOWN',
                'confidence': 0.0,
            }

        # Take the last lookback periods
        x = features[-lookback:]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        # Predict
        result = self.predict(x)

        regime_idx = int(result['predictions'][0])
        confidence = float(result['confidences'][0])

        # Map to regime name
        regime_names = ['BULLISH', 'BEARISH', 'SIDEWAYS', 'HIGH_VOLATILITY', 'LOW_VOLATILITY']
        regime_name = regime_names[regime_idx] if regime_idx < len(regime_names) else 'UNKNOWN'

        return {
            'regime': regime_idx,
            'regime_name': regime_name,
            'confidence': confidence,
            'probabilities': result['probabilities'][0].tolist(),
            'distances': result['distances'][0].tolist(),
        }

    def update_prototypes(
        self,
        new_data: Tuple[torch.Tensor, torch.Tensor],
        momentum: float = 0.1,
    ) -> None:
        """
        Online prototype updating with new data.

        Uses exponential moving average to update prototypes.

        Args:
            new_data: Tuple of (X, y) with new labeled samples
            momentum: Update momentum (0 = no update, 1 = full replacement)
        """
        X, y = new_data

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long)

        device = next(self.parameters()).device
        X = X.to(device)
        y = y.to(device)

        self.eval()
        with torch.no_grad():
            embeddings = self.encode(X)

            for c in range(self.n_prototypes):
                mask = y == c
                if mask.sum() > 0:
                    # Compute new centroid for this class
                    new_centroid = embeddings[mask].mean(dim=0)

                    # Update with momentum
                    self.prototypes[c] = (
                        (1 - momentum) * self.prototypes[c] +
                        momentum * new_centroid
                    )

                    # Update statistics
                    self.prototype_counts[c] += mask.sum()
                    self.running_mean[c] = (
                        (1 - momentum) * self.running_mean[c] +
                        momentum * new_centroid
                    )

        logger.info(f"Updated prototypes with {len(X)} new samples")

    def get_prototype_info(self) -> List[Dict]:
        """
        Get information about learned prototypes.

        Returns:
            List of dictionaries with prototype information
        """
        regime_names = ['BULLISH', 'BEARISH', 'SIDEWAYS', 'HIGH_VOLATILITY', 'LOW_VOLATILITY']

        info = []
        for i in range(self.n_prototypes):
            info.append({
                'index': i,
                'regime': regime_names[i] if i < len(regime_names) else f'CLASS_{i}',
                'count': int(self.prototype_counts[i].item()),
                'norm': float(self.prototypes[i].norm().item()),
            })

        return info

    def save(self, path: str) -> None:
        """Save model to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.embedding_dim,
                'n_prototypes': self.n_prototypes,
                'encoder_type': self.encoder_type,
                'sequence_length': self.sequence_length,
                'input_dim': self.input_dim,
                'distance_type': self.distance_type,
            }
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'PrototypeLearner':
        """Load model from file."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']

        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)

        logger.info(f"Model loaded from {path}")
        return model


def create_prototype_learner(
    config: Optional[PrototypeConfig] = None,
    **kwargs,
) -> PrototypeLearner:
    """
    Factory function to create PrototypeLearner.

    Args:
        config: PrototypeConfig object
        **kwargs: Override config values

    Returns:
        PrototypeLearner instance
    """
    if config is None:
        config = PrototypeConfig()

    return PrototypeLearner(
        embedding_dim=kwargs.get('embedding_dim', config.embedding_dim),
        n_prototypes=kwargs.get('n_prototypes', config.n_prototypes),
        encoder_type=kwargs.get('encoder_type', config.encoder_type),
        sequence_length=kwargs.get('sequence_length', config.sequence_length),
        input_dim=kwargs.get('input_dim', config.input_dim),
        dropout=kwargs.get('dropout', config.dropout),
        distance_type=kwargs.get('distance_type', config.distance_type),
    )


if __name__ == "__main__":
    # Demo
    print("Prototype Learning Model Demo")
    print("=" * 50)

    # Create model
    model = PrototypeLearner(
        embedding_dim=64,
        n_prototypes=5,
        encoder_type='cnn',
        sequence_length=60,
        input_dim=10,
    )

    print(f"\nModel configuration:")
    print(f"  Embedding dim: {model.embedding_dim}")
    print(f"  N prototypes: {model.n_prototypes}")
    print(f"  Encoder type: {model.encoder_type}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Generate synthetic data
    batch_size = 32
    n_samples = 100
    X = np.random.randn(n_samples, 60, 10).astype(np.float32)
    y = np.random.randint(0, 5, size=n_samples)

    # Fit model
    print("\nFitting model on support set...")
    history = model.fit((X, y), n_episodes=50, lr=0.001)

    # Predict
    print("\nMaking predictions...")
    query = X[:10]
    result = model.predict(query)

    print(f"Predictions: {result['predictions']}")
    print(f"Confidences: {result['confidences'].round(3)}")

    # Prototype info
    print("\nPrototype information:")
    for info in model.get_prototype_info():
        print(f"  {info['regime']}: norm={info['norm']:.3f}")

    print("\n" + "=" * 50)
    print("Demo complete!")
