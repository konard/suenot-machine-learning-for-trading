"""
Prototype Network (ProtoPNet) for Market State Classification

This module implements the ProtoPNet architecture adapted for financial
time series classification, based on "This Looks Like That" paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import IntEnum
from typing import Tuple, Optional, List


class MarketState(IntEnum):
    """Market state classifications"""
    BULLISH = 0
    BEARISH = 1
    CONSOLIDATION = 2
    BREAKOUT_UP = 3
    BREAKOUT_DOWN = 4


class ConvFeatureEncoder(nn.Module):
    """
    Convolutional encoder for time series data.

    Converts input time series of shape (batch, channels, sequence_length)
    to latent representation of shape (batch, latent_dim).
    """

    def __init__(
        self,
        input_channels: int = 10,
        sequence_length: int = 60,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Calculate output size after convolutions and pooling
        self.pool = nn.MaxPool1d(2)
        conv_output_size = sequence_length // 8  # Three pooling layers

        # Fully connected layer to latent space
        self.fc = nn.Linear(128 * conv_output_size, latent_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length)

        Returns:
            Latent representation of shape (batch, latent_dim)
        """
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

        # Flatten and project to latent space
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class PrototypeLayer(nn.Module):
    """
    Learnable prototype layer.

    Contains K prototypes per class, computes similarity between
    input representations and prototypes.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 5,
        prototypes_per_class: int = 2,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.num_prototypes = num_classes * prototypes_per_class
        self.epsilon = epsilon

        # Learnable prototypes
        self.prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, latent_dim) * 0.1
        )

        # Mapping from prototypes to classes
        # prototype_class_identity[i, j] = 1 if prototype i belongs to class j
        prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        for i in range(self.num_prototypes):
            class_idx = i // prototypes_per_class
            prototype_class_identity[i, class_idx] = 1
        self.register_buffer('prototype_class_identity', prototype_class_identity)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarity scores between latent representations and prototypes.

        Args:
            z: Latent representation of shape (batch, latent_dim)

        Returns:
            Tuple of (similarities, distances) both of shape (batch, num_prototypes)
        """
        # Compute squared L2 distances
        # z: (batch, latent_dim)
        # prototypes: (num_prototypes, latent_dim)
        distances = torch.cdist(z, self.prototypes, p=2) ** 2

        # Convert distances to similarities using log transformation
        # This gives higher values for smaller distances
        similarities = torch.log((distances + 1) / (distances + self.epsilon))

        return similarities, distances

    def get_prototype_class(self, prototype_idx: int) -> int:
        """Get the class that a prototype belongs to."""
        return prototype_idx // self.prototypes_per_class

    def get_class_prototypes(self, class_idx: int) -> torch.Tensor:
        """Get all prototypes for a given class."""
        start_idx = class_idx * self.prototypes_per_class
        end_idx = start_idx + self.prototypes_per_class
        return self.prototypes[start_idx:end_idx]


class ProtoPNet(nn.Module):
    """
    Prototype Network for Market State Classification.

    Architecture:
    1. Convolutional encoder extracts features from time series
    2. Prototype layer computes similarities to learned prototypes
    3. Classification head maps similarities to class probabilities

    The model is interpretable: predictions are based on similarity
    to learned prototypes that represent typical market patterns.
    """

    def __init__(
        self,
        input_channels: int = 10,
        sequence_length: int = 60,
        latent_dim: int = 128,
        num_classes: int = 5,
        prototypes_per_class: int = 2,
    ):
        """
        Initialize ProtoPNet.

        Args:
            input_channels: Number of input features (e.g., OHLCV + indicators)
            sequence_length: Length of input time series
            latent_dim: Dimension of latent space
            num_classes: Number of market state classes
            prototypes_per_class: Number of prototypes per class
        """
        super().__init__()
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.num_prototypes = num_classes * prototypes_per_class

        # Components
        self.encoder = ConvFeatureEncoder(
            input_channels=input_channels,
            sequence_length=sequence_length,
            latent_dim=latent_dim,
        )

        self.prototype_layer = PrototypeLayer(
            latent_dim=latent_dim,
            num_classes=num_classes,
            prototypes_per_class=prototypes_per_class,
        )

        # Classification head: linear layer from similarities to classes
        # Initialize to connect each prototype only to its class
        self.classifier = nn.Linear(self.num_prototypes, num_classes, bias=False)
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize classifier to connect prototypes to their classes."""
        with torch.no_grad():
            # Set weights so each prototype contributes only to its class
            self.classifier.weight.copy_(
                self.prototype_layer.prototype_class_identity.T
            )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through ProtoPNet.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length)

        Returns:
            Tuple of (logits, similarities, distances)
            - logits: Class logits of shape (batch, num_classes)
            - similarities: Prototype similarities of shape (batch, num_prototypes)
            - distances: Prototype distances of shape (batch, num_prototypes)
        """
        # Encode input to latent space
        z = self.encoder(x)

        # Compute similarities to prototypes
        similarities, distances = self.prototype_layer(z)

        # Classify based on similarities
        logits = self.classifier(similarities)

        return logits, similarities, distances

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length)

        Returns:
            Tuple of (predictions, confidences)
            - predictions: Predicted class indices of shape (batch,)
            - confidences: Max softmax probability of shape (batch,)
        """
        logits, similarities, _ = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)
        return predictions, confidences

    def get_prototype_similarities(
        self, x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get similarity scores for interpretation.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length)

        Returns:
            Tuple of (similarities, predictions) as numpy arrays
        """
        self.eval()
        with torch.no_grad():
            _, similarities, _ = self.forward(x)
            predictions = similarities.argmax(dim=1)
        return similarities.cpu().numpy(), predictions.cpu().numpy()

    def push_prototypes(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> List[int]:
        """
        Push prototypes to nearest training examples.

        This makes prototypes more interpretable by ensuring they
        correspond to actual market patterns in the training data.

        Args:
            dataloader: DataLoader with training data
            device: Device to run computations on

        Returns:
            List of indices of nearest training examples for each prototype
        """
        self.eval()

        # Find nearest training example for each prototype
        min_distances = [float('inf')] * self.num_prototypes
        nearest_indices = [-1] * self.num_prototypes
        nearest_latents = [None] * self.num_prototypes

        with torch.no_grad():
            global_idx = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Get latent representations
                z = self.encoder(batch_x)

                # Compute distances to all prototypes
                _, distances = self.prototype_layer(z)

                # Update nearest examples for each prototype
                for proto_idx in range(self.num_prototypes):
                    proto_class = self.prototype_layer.get_prototype_class(proto_idx)

                    # Only consider examples from the prototype's class
                    class_mask = (batch_y == proto_class)
                    if not class_mask.any():
                        continue

                    class_distances = distances[class_mask, proto_idx]
                    class_latents = z[class_mask]
                    class_indices = torch.where(class_mask)[0]

                    min_dist, min_idx = class_distances.min(dim=0)

                    if min_dist.item() < min_distances[proto_idx]:
                        min_distances[proto_idx] = min_dist.item()
                        nearest_indices[proto_idx] = global_idx + class_indices[min_idx].item()
                        nearest_latents[proto_idx] = class_latents[min_idx]

                global_idx += len(batch_x)

        # Update prototypes to nearest latent representations
        with torch.no_grad():
            for proto_idx in range(self.num_prototypes):
                if nearest_latents[proto_idx] is not None:
                    self.prototype_layer.prototypes[proto_idx] = nearest_latents[proto_idx]

        return nearest_indices


class ProtoPNetLoss(nn.Module):
    """
    Combined loss function for ProtoPNet training.

    Loss = CrossEntropy + lambda_clst * ClusteringLoss + lambda_sep * SeparationLoss

    - CrossEntropy: Standard classification loss
    - ClusteringLoss: Encourages prototypes to be close to training examples
    - SeparationLoss: Encourages prototypes of different classes to be distant
    """

    def __init__(
        self,
        lambda_clst: float = 0.8,
        lambda_sep: float = -0.08,
    ):
        """
        Initialize loss function.

        Args:
            lambda_clst: Weight for clustering loss
            lambda_sep: Weight for separation loss (typically negative)
        """
        super().__init__()
        self.lambda_clst = lambda_clst
        self.lambda_sep = lambda_sep
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        similarities: torch.Tensor,
        distances: torch.Tensor,
        targets: torch.Tensor,
        prototype_class_identity: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            logits: Class logits of shape (batch, num_classes)
            similarities: Prototype similarities of shape (batch, num_prototypes)
            distances: Prototype distances of shape (batch, num_prototypes)
            targets: Target class indices of shape (batch,)
            prototype_class_identity: Binary matrix mapping prototypes to classes

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        batch_size = logits.size(0)
        num_prototypes = distances.size(1)

        # Cross-entropy loss
        ce = self.ce_loss(logits, targets)

        # Clustering loss: min distance to correct class prototypes
        # For each sample, find the min distance to prototypes of its class
        target_prototype_mask = prototype_class_identity[targets]  # (batch, num_prototypes)

        # Set distances to wrong-class prototypes to infinity
        masked_distances = distances.clone()
        masked_distances[target_prototype_mask == 0] = float('inf')

        # Min distance to correct prototypes
        min_correct_distances = masked_distances.min(dim=1)[0]
        cluster_loss = min_correct_distances.mean()

        # Separation loss: min distance to wrong class prototypes
        # Set distances to correct-class prototypes to infinity
        sep_masked_distances = distances.clone()
        sep_masked_distances[target_prototype_mask == 1] = float('inf')

        # Min distance to wrong prototypes
        min_wrong_distances = sep_masked_distances.min(dim=1)[0]
        separation_loss = min_wrong_distances.mean()

        # Total loss
        total_loss = ce + self.lambda_clst * cluster_loss + self.lambda_sep * separation_loss

        loss_components = {
            'cross_entropy': ce.item(),
            'clustering': cluster_loss.item(),
            'separation': separation_loss.item(),
            'total': total_loss.item(),
        }

        return total_loss, loss_components
