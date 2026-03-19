"""
GAN for LOB Generation

Implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) for generating
realistic limit order book snapshots. The adversarial training produces
sharper, more realistic LOB states than VAE alone.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class LobGenerator(nn.Module):
    """Generator network: maps noise to synthetic LOB feature vectors."""

    def __init__(self, noise_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LobDiscriminator(nn.Module):
    """Discriminator (critic) network: scores LOB feature vectors."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LobGan:
    """
    Wasserstein GAN with Gradient Penalty for LOB generation.

    Uses the Wasserstein distance with gradient penalty for stable training
    on the multimodal distributions found in LOB data.

    Args:
        input_dim: Dimensionality of LOB feature vector.
        noise_dim: Dimensionality of noise input to generator.
        hidden_dim: Hidden layer size for both networks.
        gp_weight: Gradient penalty weight (lambda). Default 10.0.
        n_critic: Number of critic updates per generator update. Default 5.
        device: Torch device.
    """

    def __init__(
        self,
        input_dim: int,
        noise_dim: int = 32,
        hidden_dim: int = 64,
        gp_weight: float = 10.0,
        n_critic: int = 5,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.gp_weight = gp_weight
        self.n_critic = n_critic

        self.generator = LobGenerator(noise_dim, hidden_dim, input_dim).to(self.device)
        self.discriminator = LobDiscriminator(input_dim, hidden_dim).to(self.device)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9)
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9)
        )

    def _gradient_penalty(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        d_interp = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def train_step(self, real_data: torch.Tensor) -> dict:
        """
        One training step: update critic n_critic times, then generator once.

        Args:
            real_data: Batch of real LOB feature vectors.

        Returns:
            Dictionary with loss statistics.
        """
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        # Train discriminator
        d_loss_total = 0.0
        for _ in range(self.n_critic):
            self.opt_d.zero_grad()
            z = torch.randn(batch_size, self.noise_dim, device=self.device)
            fake = self.generator(z).detach()

            d_real = self.discriminator(real_data).mean()
            d_fake = self.discriminator(fake).mean()
            gp = self._gradient_penalty(real_data, fake)
            d_loss = d_fake - d_real + self.gp_weight * gp

            d_loss.backward()
            self.opt_d.step()
            d_loss_total += d_loss.item()

        # Train generator
        self.opt_g.zero_grad()
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake = self.generator(z)
        g_loss = -self.discriminator(fake).mean()
        g_loss.backward()
        self.opt_g.step()

        return {
            "d_loss": d_loss_total / self.n_critic,
            "g_loss": g_loss.item(),
            "wasserstein_dist": (d_real - d_fake).item(),
        }

    def train(
        self, data: np.ndarray, epochs: int = 100, batch_size: int = 32
    ) -> List[dict]:
        """
        Train the GAN for multiple epochs.

        Args:
            data: Feature vectors, shape (N, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of epoch loss dictionaries.
        """
        tensor_data = torch.tensor(data, dtype=torch.float32)
        n = len(tensor_data)
        history = []

        for epoch in range(epochs):
            indices = torch.randperm(n)
            epoch_stats = {"d_loss": 0.0, "g_loss": 0.0, "wasserstein_dist": 0.0}
            num_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = tensor_data[indices[start:end]]
                if len(batch) < 2:
                    continue

                stats = self.train_step(batch)
                for k in epoch_stats:
                    epoch_stats[k] += stats[k]
                num_batches += 1

            if num_batches > 0:
                for k in epoch_stats:
                    epoch_stats[k] /= num_batches

            history.append(epoch_stats)

        return history

    def generate(self, num_samples: int = 1) -> np.ndarray:
        """Generate synthetic LOB feature vectors."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.noise_dim, device=self.device)
            fake = self.generator(z)
            return fake.cpu().numpy()


if __name__ == "__main__":
    # Quick test
    input_dim = 40
    gan = LobGan(input_dim=input_dim, noise_dim=16, hidden_dim=32)

    fake_data = np.random.randn(100, input_dim).astype(np.float32)
    history = gan.train(fake_data, epochs=3, batch_size=16)
    print(f"Training history (3 epochs): {history}")

    generated = gan.generate(5)
    print(f"Generated shape: {generated.shape}")
    print("GAN test passed!")
