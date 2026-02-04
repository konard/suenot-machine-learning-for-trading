#!/usr/bin/env python3
"""
Flow Model Implementation for Trading

Implements RealNVP-style normalizing flow for market data modeling.
Provides exact likelihood computation, anomaly detection, and sample generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ActNorm(nn.Module):
    """
    Activation Normalization layer with data-dependent initialization.
    Learns scale and bias parameters initialized from first batch.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        """Initialize parameters from first batch of data"""
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + 1e-6

            self.bias.data = -mean
            self.scale.data = 1.0 / std
            self.initialized = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: apply normalization"""
        if not self.initialized and self.training:
            self.initialize(x)

        y = (x + self.bias) * self.scale
        log_det = torch.sum(torch.log(torch.abs(self.scale))) * torch.ones(x.shape[0], device=x.device)

        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse pass: remove normalization"""
        x = y / self.scale - self.bias
        return x


class AffineCoupling(nn.Module):
    """
    Affine Coupling Layer for RealNVP.
    Transforms second half of input using scale and translation from first half.
    """

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2

        # Scale network
        self.scale_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.split_dim),
            nn.Tanh()  # Bounded scale for stability
        )

        # Translation network
        self.translate_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.split_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: affine transformation"""
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]

        s = self.scale_net(x1)
        t = self.translate_net(x1)

        y1 = x1
        y2 = x2 * torch.exp(s) + t

        log_det = s.sum(dim=-1)

        return torch.cat([y1, y2], dim=-1), log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse pass: reverse affine transformation"""
        y1, y2 = y[:, :self.split_dim], y[:, self.split_dim:]

        s = self.scale_net(y1)
        t = self.translate_net(y1)

        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        return torch.cat([x1, x2], dim=-1)


class Permutation(nn.Module):
    """
    Fixed random permutation layer.
    Shuffles dimensions to allow coupling layers to affect all dimensions.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Create random permutation
        perm = torch.randperm(dim)
        inv_perm = torch.argsort(perm)

        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', inv_perm)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: apply permutation"""
        return x[:, self.perm], torch.zeros(x.shape[0], device=x.device)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse: apply inverse permutation"""
        return y[:, self.inv_perm]


class NormalizingFlow(nn.Module):
    """
    Complete Normalizing Flow model for market data.
    Uses RealNVP-style architecture with ActNorm and affine coupling layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 8,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers

        # Build flow layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ActNorm(input_dim))
            self.layers.append(AffineCoupling(input_dim, hidden_dim))
            if i < num_layers - 1:
                self.layers.append(Permutation(input_dim))

        # Base distribution (standard Gaussian)
        self.register_buffer('base_mean', torch.zeros(input_dim))
        self.register_buffer('base_std', torch.ones(input_dim))

        logger.info(f"Created NormalizingFlow: dim={input_dim}, layers={num_layers}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: transform data to latent space

        Args:
            x: Input data [batch_size, input_dim]

        Returns:
            z: Latent representation [batch_size, input_dim]
            log_det: Log determinant of Jacobian [batch_size]
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x

        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det

        return z, log_det_total

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse pass: transform latent to data space

        Args:
            z: Latent representation [batch_size, input_dim]

        Returns:
            x: Reconstructed data [batch_size, input_dim]
        """
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of data

        Args:
            x: Input data [batch_size, input_dim]

        Returns:
            log_px: Log probability [batch_size]
        """
        z, log_det = self.forward(x)

        # Log probability under base distribution
        log_pz = -0.5 * (
            self.input_dim * np.log(2 * np.pi) +
            torch.sum(z ** 2, dim=-1)
        )

        # Apply change of variables formula
        log_px = log_pz + log_det

        return log_px

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the model

        Args:
            num_samples: Number of samples to generate

        Returns:
            x: Generated samples [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.input_dim, device=self.base_mean.device)
        x = self.inverse(z)
        return x

    def nll_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss

        Args:
            x: Input data [batch_size, input_dim]

        Returns:
            loss: Mean NLL loss
        """
        log_px = self.log_prob(x)
        return -log_px.mean()


class FlowModelTrainer:
    """
    Trainer for normalizing flow models.
    Handles training loop, validation, and early stopping.
    """

    def __init__(
        self,
        model: NormalizingFlow,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            x = batch[0].to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.nll_loss(x)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Compute validation loss"""
        self.model.eval()
        total_loss = 0

        for batch in dataloader:
            x = batch[0].to(self.device)
            loss = self.model.nll_loss(x)
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 128,
        patience: int = 20
    ):
        """
        Train the flow model

        Args:
            train_data: Training data [N, input_dim]
            val_data: Validation data [M, input_dim]
            epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Early stopping patience
        """
        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = None
        if val_data is not None:
            val_loader = DataLoader(
                TensorDataset(val_data),
                batch_size=batch_size,
                shuffle=False
            )

        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        logger.info("Training complete")


class AnomalyDetector:
    """
    Anomaly detection using flow model likelihood.
    Flags data points with low likelihood as anomalies.
    """

    def __init__(self, model: NormalizingFlow, threshold_percentile: float = 5.0):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None

    @torch.no_grad()
    def fit(self, reference_data: torch.Tensor):
        """
        Fit threshold on reference (normal) data

        Args:
            reference_data: Normal training data
        """
        self.model.eval()
        log_probs = self.model.log_prob(reference_data).cpu().numpy()
        self.threshold = np.percentile(log_probs, self.threshold_percentile)
        logger.info(f"Anomaly threshold set to {self.threshold:.4f}")

    @torch.no_grad()
    def detect(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in data

        Args:
            x: Data to check for anomalies

        Returns:
            is_anomaly: Boolean array indicating anomalies
            log_probs: Log probabilities for each sample
        """
        self.model.eval()
        log_probs = self.model.log_prob(x).cpu().numpy()
        is_anomaly = log_probs < self.threshold
        return is_anomaly, log_probs


class RegimeDetector:
    """
    Market regime detection using latent space clustering.
    Groups similar market states by clustering in latent space.
    """

    def __init__(self, model: NormalizingFlow, n_regimes: int = 4):
        self.model = model
        self.n_regimes = n_regimes
        self.cluster_centers = None
        self.regime_labels = None

    @torch.no_grad()
    def fit(self, data: torch.Tensor, labels: Optional[List[str]] = None):
        """
        Fit regime clusters on data

        Args:
            data: Training data
            labels: Optional regime names
        """
        self.model.eval()

        # Get latent representations
        z, _ = self.model.forward(data)
        z_np = z.cpu().numpy()

        # Simple k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        kmeans.fit(z_np)

        self.cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        self.regime_labels = labels or [f"Regime_{i}" for i in range(self.n_regimes)]

        logger.info(f"Fitted {self.n_regimes} regimes")

    @torch.no_grad()
    def detect(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect regime for new data

        Args:
            x: Data to classify

        Returns:
            regimes: Regime indices
            distances: Distances to each cluster center
        """
        self.model.eval()
        z, _ = self.model.forward(x)

        # Compute distances to cluster centers
        distances = torch.cdist(z, self.cluster_centers.to(x.device))
        regimes = distances.argmin(dim=-1).cpu().numpy()
        distances = distances.cpu().numpy()

        return regimes, distances


def create_synthetic_market_data(n_samples: int, dim: int) -> np.ndarray:
    """Create synthetic market-like data for testing"""
    # Generate mixture of Gaussians representing different market regimes
    data1 = np.random.multivariate_normal(
        mean=np.zeros(dim) + 0.001,  # Slight positive drift
        cov=np.eye(dim) * 0.02,
        size=n_samples // 2
    )
    data2 = np.random.multivariate_normal(
        mean=np.zeros(dim) - 0.001,  # Slight negative drift
        cov=np.eye(dim) * 0.03,
        size=n_samples // 2
    )
    data = np.vstack([data1, data2]).astype(np.float32)
    np.random.shuffle(data)
    return data


def main():
    """Example: Train a flow model on synthetic data"""
    print("=" * 60)
    print("Flow Model Training Example")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 5000
    dim = 8

    data = create_synthetic_market_data(n_samples, dim)

    # Split into train/val
    train_data = torch.tensor(data[:4000])
    val_data = torch.tensor(data[4000:])

    print(f"\nData shape: {data.shape}")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Create model
    model = NormalizingFlow(
        input_dim=dim,
        num_layers=6,
        hidden_dim=128
    )

    # Train model
    trainer = FlowModelTrainer(model, learning_rate=1e-3)
    trainer.train(train_data, val_data, epochs=50, batch_size=128, patience=10)

    # Test likelihood computation
    print("\n--- Likelihood Test ---")
    model.eval()
    with torch.no_grad():
        log_probs = model.log_prob(val_data[:10])
        print(f"Sample log-probs: {log_probs.numpy()}")
        print(f"Mean log-prob: {log_probs.mean().item():.4f}")

    # Test sampling
    print("\n--- Sampling Test ---")
    samples = model.sample(5)
    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample means: {samples.mean(dim=0).numpy()}")

    # Test reconstruction (should be nearly perfect for flows)
    print("\n--- Reconstruction Test ---")
    with torch.no_grad():
        x = val_data[:5]
        z, _ = model.forward(x)
        x_recon = model.inverse(z)
        recon_error = (x - x_recon).abs().mean().item()
        print(f"Reconstruction error: {recon_error:.8f}")

    # Test anomaly detection
    print("\n--- Anomaly Detection Test ---")
    detector = AnomalyDetector(model, threshold_percentile=5.0)
    detector.fit(train_data)

    # Create some anomalies (extreme values)
    anomalies = torch.tensor(np.random.randn(10, dim).astype(np.float32) * 0.5)
    is_anomaly, log_probs = detector.detect(anomalies)
    print(f"Anomaly detection rate: {is_anomaly.mean() * 100:.1f}%")

    # Test regime detection
    print("\n--- Regime Detection Test ---")
    regime_detector = RegimeDetector(model, n_regimes=4)
    regime_detector.fit(train_data)

    regimes, distances = regime_detector.detect(val_data[:10])
    print(f"Detected regimes: {regimes}")

    print("\n" + "=" * 60)
    print("Flow model training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
