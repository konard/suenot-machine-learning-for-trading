"""
Diffusion Model for LOB Generation

Implements a Denoising Diffusion Probabilistic Model (DDPM) for generating
realistic limit order book snapshots. Diffusion models excel at capturing
the multimodal, heavy-tailed distributions characteristic of LOB data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class NoisePredictor(nn.Module):
    """
    Noise prediction network for the diffusion model.

    Predicts the noise added to the LOB feature vector at a given timestep.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_timesteps: int):
        super().__init__()
        self.time_embed = nn.Embedding(num_timesteps, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise given noisy input and timestep.

        Args:
            x_t: Noisy LOB features, shape (batch, input_dim).
            t: Timestep indices, shape (batch,).

        Returns:
            Predicted noise, shape (batch, input_dim).
        """
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x_t, t_emb], dim=-1))


class LobDiffusion:
    """
    DDPM for LOB snapshot generation.

    Defines a forward noising process that gradually adds Gaussian noise
    to LOB data, and learns a reverse denoising process to generate new samples.

    Args:
        input_dim: Dimensionality of LOB feature vector.
        hidden_dim: Hidden layer size.
        num_timesteps: Number of diffusion steps. Default 100.
        beta_start: Starting noise schedule value.
        beta_end: Ending noise schedule value.
        device: Torch device.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        self.model = NoisePredictor(input_dim, hidden_dim, num_timesteps).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def forward_diffusion(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple:
        """
        Forward diffusion: add noise to clean data.

        Args:
            x_0: Clean data, shape (batch, input_dim).
            t: Timestep indices, shape (batch,).

        Returns:
            (noisy_data, noise) tuple.
        """
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def train_step(self, batch: torch.Tensor) -> float:
        """One training step: predict noise at random timesteps."""
        self.model.train()
        batch = batch.to(self.device)
        batch_size = batch.size(0)

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        x_t, noise = self.forward_diffusion(batch, t)

        self.optimizer.zero_grad()
        predicted_noise = self.model(x_t, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self, data: np.ndarray, epochs: int = 50, batch_size: int = 32
    ) -> List[float]:
        """
        Train the diffusion model.

        Args:
            data: Feature vectors, shape (N, input_dim).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.

        Returns:
            List of average losses per epoch.
        """
        tensor_data = torch.tensor(data, dtype=torch.float32)
        n = len(tensor_data)
        history = []

        for epoch in range(epochs):
            indices = torch.randperm(n)
            total_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = tensor_data[indices[start:end]]
                total_loss += self.train_step(batch)
                num_batches += 1

            history.append(total_loss / max(num_batches, 1))

        return history

    @torch.no_grad()
    def generate(self, num_samples: int = 1) -> np.ndarray:
        """
        Generate synthetic LOB feature vectors via reverse diffusion.

        Starts from pure noise and iteratively denoises to produce
        realistic LOB snapshots.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Generated feature vectors, shape (num_samples, input_dim).
        """
        self.model.eval()
        x = torch.randn(num_samples, self.input_dim, device=self.device)

        for t_val in reversed(range(self.num_timesteps)):
            t = torch.full((num_samples,), t_val, dtype=torch.long, device=self.device)
            predicted_noise = self.model(x, t)

            alpha_t = self.alphas[t_val]
            alpha_bar_t = self.alpha_bar[t_val]
            beta_t = self.betas[t_val]

            # Reverse step
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            )

            if t_val > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

        return x.cpu().numpy()


if __name__ == "__main__":
    # Quick test
    input_dim = 40
    diffusion = LobDiffusion(input_dim=input_dim, hidden_dim=32, num_timesteps=50)

    fake_data = np.random.randn(100, input_dim).astype(np.float32)
    history = diffusion.train(fake_data, epochs=3, batch_size=16)
    print(f"Training losses: {history}")

    generated = diffusion.generate(5)
    print(f"Generated shape: {generated.shape}")
    print("Diffusion test passed!")
