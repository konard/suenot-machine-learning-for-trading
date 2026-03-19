"""
Variational Autoencoder for LOB Snapshots

Implements a VAE that learns a compact latent representation of order book states,
enabling generation of synthetic LOB data and anomaly detection via reconstruction error.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional


class LobVaeEncoder(nn.Module):
    """Encoder network: maps LOB features to latent distribution parameters."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden(x)
        return self.fc_mu(h), self.fc_log_var(h)


class LobVaeDecoder(nn.Module):
    """Decoder network: reconstructs LOB features from latent code."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LobVae(nn.Module):
    """
    Variational Autoencoder for Limit Order Book snapshots.

    Learns a smooth latent space of LOB states that respects structural
    constraints (monotonic prices, positive quantities) and enables:
    - Synthetic LOB generation via prior sampling
    - Anomaly detection via reconstruction error
    - Latent space interpolation between market states

    Args:
        input_dim: Dimensionality of LOB feature vector (num_levels * 4).
        hidden_dim: Hidden layer size.
        latent_dim: Latent space dimensionality.
        beta: KL divergence weight (beta-VAE). Default 1.0.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        beta: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = LobVaeEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LobVaeDecoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters (mu, log_var)."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to reconstructed input."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode, reparameterize, decode.

        Returns:
            (reconstructed_x, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss: reconstruction + beta * KL divergence.

        Returns:
            (total_loss, reconstruction_loss, kl_loss)
        """
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss

    def generate(self, num_samples: int = 1) -> torch.Tensor:
        """Generate synthetic LOB feature vectors by sampling from the prior."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            return self.decode(z)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score as reconstruction error."""
        self.eval()
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
            return torch.mean((x - x_recon) ** 2, dim=-1)

    def interpolate(
        self, x_a: torch.Tensor, x_b: torch.Tensor, steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two LOB states in latent space.

        Args:
            x_a: First LOB feature vector.
            x_b: Second LOB feature vector.
            steps: Number of interpolation steps.

        Returns:
            Tensor of shape (steps, input_dim) with interpolated LOB states.
        """
        self.eval()
        with torch.no_grad():
            mu_a, _ = self.encode(x_a.unsqueeze(0))
            mu_b, _ = self.encode(x_b.unsqueeze(0))

            alphas = torch.linspace(0, 1, steps).unsqueeze(1)
            z_interp = (1 - alphas) * mu_a + alphas * mu_b
            return self.decode(z_interp)


class LobVaeTrainer:
    """
    Training loop for LobVae.

    Args:
        model: LobVae model to train.
        learning_rate: Optimizer learning rate.
        device: Torch device (cpu or cuda).
    """

    def __init__(
        self,
        model: LobVae,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_epoch(self, data: np.ndarray, batch_size: int = 32) -> dict:
        """
        Train for one epoch.

        Args:
            data: Feature vectors, shape (N, input_dim).
            batch_size: Mini-batch size.

        Returns:
            Dictionary with loss statistics.
        """
        self.model.train()
        tensor_data = torch.tensor(data, dtype=torch.float32).to(self.device)
        n = len(tensor_data)

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = tensor_data[indices[start:end]]

            self.optimizer.zero_grad()
            x_recon, mu, log_var = self.model(batch)
            loss, recon, kl = self.model.loss(batch, x_recon, mu, log_var)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon / num_batches,
            "kl_loss": total_kl / num_batches,
        }

    def train(
        self, data: np.ndarray, epochs: int = 50, batch_size: int = 32
    ) -> List[dict]:
        """
        Train the VAE for multiple epochs.

        Args:
            data: Feature vectors, shape (N, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of epoch loss dictionaries.
        """
        history = []
        for epoch in range(epochs):
            stats = self.train_epoch(data, batch_size)
            history.append(stats)
        return history


if __name__ == "__main__":
    # Quick test
    input_dim = 40  # 10 levels * 4 features
    model = LobVae(input_dim=input_dim, hidden_dim=32, latent_dim=8)
    x = torch.randn(5, input_dim)

    x_recon, mu, log_var = model(x)
    loss, recon, kl = model.loss(x, x_recon, mu, log_var)
    print(f"VAE loss: {loss.item():.4f} (recon: {recon.item():.4f}, kl: {kl.item():.4f})")

    generated = model.generate(3)
    print(f"Generated shape: {generated.shape}")

    scores = model.anomaly_score(x)
    print(f"Anomaly scores: {scores.tolist()}")

    interp = model.interpolate(x[0], x[1], steps=5)
    print(f"Interpolation shape: {interp.shape}")
    print("VAE test passed!")
