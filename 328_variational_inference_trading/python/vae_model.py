"""
Variational Autoencoder (VAE) for Market Data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np

from config import ModelConfig


class Encoder(nn.Module):
    """
    Encoder network that maps input sequences to latent distribution parameters

    Takes market data sequences and outputs mu and log_var for the latent space
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        sequence_length: int = 60,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=2,
            batch_first=True,
            dropout=dropout if len(hidden_dims) > 1 else 0,
            bidirectional=True
        )

        # Fully connected layers
        layers = []
        prev_dim = hidden_dims[0] * 2  # *2 for bidirectional

        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.fc = nn.Sequential(*layers) if layers else nn.Identity()

        # Output layers for mu and log_var
        final_dim = hidden_dims[-1] if len(hidden_dims) > 1 else hidden_dims[0] * 2
        self.mu_layer = nn.Linear(final_dim, latent_dim)
        self.logvar_layer = nn.Linear(final_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Tuple of (mu, log_var) each of shape (batch, latent_dim)
        """
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden states from both directions
        h_forward = h_n[-2, :, :]  # Last layer, forward
        h_backward = h_n[-1, :, :]  # Last layer, backward
        h = torch.cat([h_forward, h_backward], dim=1)

        # Fully connected layers
        h = self.fc(h)

        # Output mu and log_var
        mu = self.mu_layer(h)
        log_var = self.logvar_layer(h)

        return mu, log_var


class Decoder(nn.Module):
    """
    Decoder network that maps latent codes back to output space

    Takes latent vectors and predicts:
    - Reconstructed input features
    - Return distribution (mu, sigma)
    - Regime probabilities
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        sequence_length: int = 60,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        n_regimes: int = 3
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.n_regimes = n_regimes

        # Reverse hidden dims for decoder
        hidden_dims = hidden_dims[::-1]

        # Initial projection
        layers = []
        prev_dim = latent_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.fc = nn.Sequential(*layers)

        # LSTM for temporal decoding
        self.lstm = nn.LSTM(
            input_size=hidden_dims[-1],
            hidden_size=hidden_dims[-1],
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Output heads
        self.reconstruction_head = nn.Linear(hidden_dims[-1], output_dim)

        # Return prediction head (outputs mu and log_var for returns)
        self.return_mu_head = nn.Linear(hidden_dims[-1], 1)
        self.return_logvar_head = nn.Linear(hidden_dims[-1], 1)

        # Regime classification head
        self.regime_head = nn.Linear(hidden_dims[-1], n_regimes)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through decoder

        Args:
            z: Latent tensor of shape (batch, latent_dim)

        Returns:
            Tuple of:
            - x_recon: Reconstructed features (batch, seq_len, output_dim)
            - return_mu: Predicted return mean (batch, 1)
            - return_logvar: Predicted return log variance (batch, 1)
            - regime_logits: Regime classification logits (batch, n_regimes)
        """
        # Project latent code
        h = self.fc(z)

        # Expand for sequence
        h = h.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # LSTM decoding
        lstm_out, _ = self.lstm(h)

        # Reconstruction
        x_recon = self.reconstruction_head(lstm_out)

        # Return prediction (from final timestep)
        final_h = lstm_out[:, -1, :]
        return_mu = self.return_mu_head(final_h)
        return_logvar = self.return_logvar_head(final_h)

        # Regime classification (from final timestep)
        regime_logits = self.regime_head(final_h)

        return x_recon, return_mu, return_logvar, regime_logits


class MarketVAE(nn.Module):
    """
    Variational Autoencoder for Market Data

    Learns latent representations of market states and provides:
    - Uncertainty-aware return predictions
    - Market regime detection
    - Anomaly detection via reconstruction error
    """

    def __init__(self, config: Optional[ModelConfig] = None, sequence_length: int = 60):
        super().__init__()

        self.config = config or ModelConfig()
        self.sequence_length = sequence_length

        self.encoder = Encoder(
            input_dim=self.config.input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims,
            sequence_length=sequence_length,
            dropout=self.config.dropout,
            use_batch_norm=self.config.use_batch_norm
        )

        self.decoder = Decoder(
            latent_dim=self.config.latent_dim,
            output_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            sequence_length=sequence_length,
            dropout=self.config.dropout,
            use_batch_norm=self.config.use_batch_norm
        )

        self.beta = self.config.beta

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution

        z = mu + sigma * epsilon, where epsilon ~ N(0, 1)

        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Full forward pass through VAE

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Tuple of (x_recon, return_mu, return_logvar, regime_logits, mu, log_var)
        """
        # Encode
        mu, log_var = self.encoder(x)

        # Sample latent code
        z = self.reparameterize(mu, log_var)

        # Decode
        x_recon, return_mu, return_logvar, regime_logits = self.decoder(z)

        return x_recon, return_mu, return_logvar, regime_logits, mu, log_var

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Decode latent code to outputs"""
        return self.decoder(z)

    def sample_predictions(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample multiple predictions for uncertainty estimation

        Args:
            x: Input tensor
            n_samples: Number of samples to draw

        Returns:
            Tuple of (return_samples, regime_samples, latent_samples)
        """
        self.set_to_eval_mode()
        with torch.no_grad():
            mu, log_var = self.encode(x)

            return_samples = []
            regime_samples = []
            latent_samples = []

            for _ in range(n_samples):
                z = self.reparameterize(mu, log_var)
                _, return_mu, return_logvar, regime_logits = self.decode(z)

                # Sample from return distribution
                return_std = torch.exp(0.5 * return_logvar)
                return_sample = return_mu + return_std * torch.randn_like(return_std)

                return_samples.append(return_sample.cpu().numpy())
                regime_samples.append(F.softmax(regime_logits, dim=-1).cpu().numpy())
                latent_samples.append(z.cpu().numpy())

        return (
            np.array(return_samples),
            np.array(regime_samples),
            np.array(latent_samples)
        )

    def set_to_eval_mode(self):
        """Set model to evaluation mode"""
        self.eval()

    def set_to_train_mode(self):
        """Set model to training mode"""
        self.train()

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        return_mu: torch.Tensor,
        return_logvar: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target_returns: Optional[torch.Tensor] = None,
        beta: Optional[float] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss (ELBO)

        Loss = Reconstruction Loss + Return Prediction Loss + beta * KL Divergence

        Args:
            x: Original input
            x_recon: Reconstructed input
            return_mu: Predicted return mean
            return_logvar: Predicted return log variance
            mu: Latent mean
            log_var: Latent log variance
            target_returns: Ground truth returns (optional)
            beta: KL weight (uses self.beta if not provided)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        beta = beta if beta is not None else self.beta
        batch_size = x.size(0)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size

        # KL divergence (analytical for Gaussian)
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size

        # Return prediction loss (if targets provided)
        if target_returns is not None:
            # Negative log likelihood of Gaussian
            return_std = torch.exp(0.5 * return_logvar)
            return_loss = F.gaussian_nll_loss(
                return_mu.squeeze(),
                target_returns,
                return_std.squeeze() ** 2,
                reduction='mean'
            )
        else:
            return_loss = torch.tensor(0.0, device=x.device)

        # Total loss
        total_loss = recon_loss + return_loss + beta * kl_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'return_loss': return_loss.item(),
            'beta': beta
        }

        return total_loss, loss_dict


class BetaVAE(MarketVAE):
    """
    Beta-VAE variant with adjustable KL weight for disentanglement
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        sequence_length: int = 60,
        beta: float = 4.0
    ):
        super().__init__(config, sequence_length)
        self.beta = beta


def create_model(
    config: Optional[ModelConfig] = None,
    sequence_length: int = 60,
    device: str = "auto"
) -> MarketVAE:
    """
    Factory function to create VAE model

    Args:
        config: Model configuration
        sequence_length: Input sequence length
        device: Device to use ("auto", "cuda", "mps", "cpu")

    Returns:
        Initialized MarketVAE model
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = MarketVAE(config, sequence_length)
    model = model.to(device)

    return model


if __name__ == "__main__":
    # Test model
    config = ModelConfig(
        input_dim=13,
        latent_dim=16,
        hidden_dims=[128, 64]
    )

    model = create_model(config, sequence_length=60, device="cpu")
    print(f"Model architecture:\n{model}")

    # Test forward pass
    batch_size = 32
    seq_len = 60
    input_dim = 13

    x = torch.randn(batch_size, seq_len, input_dim)
    target_returns = torch.randn(batch_size)

    x_recon, return_mu, return_logvar, regime_logits, mu, log_var = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Return mu shape: {return_mu.shape}")
    print(f"Return logvar shape: {return_logvar.shape}")
    print(f"Regime logits shape: {regime_logits.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent log_var shape: {log_var.shape}")

    # Test loss computation
    loss, loss_dict = model.compute_loss(
        x, x_recon, return_mu, return_logvar, mu, log_var, target_returns
    )
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")

    # Test uncertainty estimation
    print("\nTesting uncertainty estimation...")
    return_samples, regime_samples, latent_samples = model.sample_predictions(x[:1], n_samples=50)
    print(f"Return samples shape: {return_samples.shape}")
    print(f"Return mean: {return_samples.mean():.4f}, std: {return_samples.std():.4f}")
