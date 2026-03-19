"""
VAE Model for Volatility Surfaces.

Provides a feed-forward Variational Autoencoder that learns low-dimensional
latent representations of implied volatility surfaces.

Supports:
- Standard VAE training with ELBO loss
- beta-VAE for disentangled representations
- Surface generation from the prior
- Encode/decode for surface completion
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class VAEConfig:
    """Configuration for the VAE model."""

    input_dim: int = 77  # 11 moneyness x 7 maturities
    hidden_dim: int = 64
    latent_dim: int = 8
    beta: float = 1.0
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32


if TORCH_AVAILABLE:

    class _VAENetwork(nn.Module):
        """PyTorch VAE network for volatility surfaces."""

        def __init__(self, config: VAEConfig) -> None:
            super().__init__()
            self.config = config

            # Encoder
            self.enc_fc1 = nn.Linear(config.input_dim, config.hidden_dim)
            self.enc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
            self.enc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

            # Decoder
            self.dec_fc1 = nn.Linear(config.latent_dim, config.hidden_dim)
            self.dec_out = nn.Linear(config.hidden_dim, config.input_dim)

        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = F.relu(self.enc_fc1(x))
            return self.enc_mu(h), self.enc_logvar(h)

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            h = F.relu(self.dec_fc1(z))
            return torch.sigmoid(self.dec_out(h)) * 1.99 + 0.01

        def forward(
            self, x: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar


class VAEModel:
    """VAE for volatility surfaces with train/encode/decode/sample interface.

    Works with both PyTorch (when available) and a pure-numpy fallback
    for environments without GPU support.
    """

    def __init__(self, config: VAEConfig | None = None) -> None:
        self.config = config or VAEConfig()
        self._use_torch = TORCH_AVAILABLE

        if self._use_torch:
            self._net = _VAENetwork(self.config)
            self._optimizer = torch.optim.Adam(
                self._net.parameters(), lr=self.config.learning_rate,
            )
        else:
            self._init_numpy_weights()

    # ------------------------------------------------------------------
    # Numpy fallback (mirrors the Rust simplified SGD implementation)
    # ------------------------------------------------------------------

    def _init_numpy_weights(self) -> None:
        rng = np.random.default_rng()
        d = self.config

        def xavier(fan_in: int, fan_out: int) -> np.ndarray:
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            return rng.uniform(-scale, scale, (fan_in, fan_out))

        self._enc_w1 = xavier(d.input_dim, d.hidden_dim)
        self._enc_b1 = np.zeros(d.hidden_dim)
        self._enc_w_mu = xavier(d.hidden_dim, d.latent_dim)
        self._enc_b_mu = np.zeros(d.latent_dim)
        self._enc_w_logvar = xavier(d.hidden_dim, d.latent_dim)
        self._enc_b_logvar = np.zeros(d.latent_dim)
        self._dec_w1 = xavier(d.latent_dim, d.hidden_dim)
        self._dec_b1 = np.zeros(d.hidden_dim)
        self._dec_w2 = xavier(d.hidden_dim, d.input_dim)
        self._dec_b2 = np.zeros(d.input_dim)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _np_encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = self._relu(x @ self._enc_w1 + self._enc_b1)
        mu = h @ self._enc_w_mu + self._enc_b_mu
        logvar = h @ self._enc_w_logvar + self._enc_b_logvar
        return mu, logvar

    def _np_reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def _np_decode(self, z: np.ndarray) -> np.ndarray:
        h = self._relu(z @ self._dec_w1 + self._dec_b1)
        out = h @ self._dec_w2 + self._dec_b2
        return self._sigmoid(out) * 1.99 + 0.01

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode an input vector into (mu, log_var)."""
        if self._use_torch:
            with torch.no_grad():
                t = torch.tensor(x, dtype=torch.float32)
                mu, logvar = self._net.encode(t)
                return mu.numpy(), logvar.numpy()
        return self._np_encode(x)

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode a latent vector into a reconstructed surface (flattened)."""
        if self._use_torch:
            with torch.no_grad():
                t = torch.tensor(z, dtype=torch.float32)
                return self._net.decode(t).numpy()
        return self._np_decode(z)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full forward pass: encode -> reparameterize -> decode."""
        if self._use_torch:
            with torch.no_grad():
                t = torch.tensor(x, dtype=torch.float32)
                recon, mu, logvar = self._net(t)
                return recon.numpy(), mu.numpy(), logvar.numpy()
        mu, logvar = self._np_encode(x)
        z = self._np_reparameterize(mu, logvar)
        recon = self._np_decode(z)
        return recon, mu, logvar

    def loss(
        self,
        x: np.ndarray,
        recon: np.ndarray,
        mu: np.ndarray,
        logvar: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute VAE loss = reconstruction_loss + beta * KL_divergence.

        Returns:
            (total_loss, reconstruction_loss, kl_divergence)
        """
        recon_loss = float(np.mean((x - recon) ** 2))
        kl = float(-0.5 * np.sum(1.0 + logvar - mu ** 2 - np.exp(logvar)))
        total = recon_loss + self.config.beta * kl
        return total, recon_loss, kl

    def train(self, data: List[np.ndarray], epochs: int | None = None, lr: float | None = None) -> Tuple[float, float, float]:
        """Train the VAE on a batch of flattened surface vectors.

        Args:
            data: List of 1-D numpy arrays (flattened surfaces).
            epochs: Number of training epochs (overrides config).
            lr: Learning rate (overrides config).

        Returns:
            (final_total_loss, final_recon_loss, final_kl)
        """
        n_epochs = epochs or self.config.epochs
        learning_rate = lr or self.config.learning_rate

        if self._use_torch:
            return self._train_torch(data, n_epochs, learning_rate)
        return self._train_numpy(data, n_epochs, learning_rate)

    def _train_torch(
        self, data: List[np.ndarray], epochs: int, lr: float,
    ) -> Tuple[float, float, float]:
        self._net.train()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        dataset = torch.tensor(np.array(data), dtype=torch.float32)

        last_total = last_recon = last_kl = 0.0
        for _ in range(epochs):
            epoch_total = epoch_recon = epoch_kl = 0.0
            # Shuffle
            perm = torch.randperm(len(dataset))
            for i in range(0, len(dataset), self.config.batch_size):
                batch = dataset[perm[i : i + self.config.batch_size]]
                optimizer.zero_grad()
                recon, mu, logvar = self._net(batch)
                recon_loss = F.mse_loss(recon, batch, reduction="mean")
                kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
                loss = recon_loss + self.config.beta * kl
                loss.backward()
                optimizer.step()
                epoch_total += loss.item() * len(batch)
                epoch_recon += recon_loss.item() * len(batch)
                epoch_kl += kl.item() * len(batch)
            n = len(dataset)
            last_total = epoch_total / n
            last_recon = epoch_recon / n
            last_kl = epoch_kl / n

        self._net.eval()
        return last_total, last_recon, last_kl

    def _train_numpy(
        self, data: List[np.ndarray], epochs: int, lr: float,
    ) -> Tuple[float, float, float]:
        """Simplified SGD training (mirrors the Rust implementation)."""
        last_total = last_recon = last_kl = 0.0
        d = self.config

        for _ in range(epochs):
            epoch_total = epoch_recon = epoch_kl = 0.0
            for sample in data:
                x = np.asarray(sample, dtype=np.float64)
                mu, logvar = self._np_encode(x)
                z = self._np_reparameterize(mu, logvar)
                recon = self._np_decode(z)
                total, recon_l, kl_l = self.loss(x, recon, mu, logvar)
                epoch_total += total
                epoch_recon += recon_l
                epoch_kl += kl_l

                # Numerical gradient update (simplified SGD)
                error = x - recon
                grad_scale = lr / d.input_dim
                self._dec_b2 += error * grad_scale

                # Update encoder bias towards smaller KL
                self._enc_b_mu -= lr * d.beta * mu * 0.01
                self._enc_b_logvar -= lr * d.beta * (np.exp(logvar) - 1.0) * 0.005

            n = len(data)
            last_total = epoch_total / n
            last_recon = epoch_recon / n
            last_kl = epoch_kl / n

        return last_total, last_recon, last_kl

    def sample(self) -> np.ndarray:
        """Sample a new surface from the prior p(z) = N(0, I)."""
        z = np.random.randn(self.config.latent_dim)
        return self.decode(z)

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    @property
    def beta(self) -> float:
        return self.config.beta
