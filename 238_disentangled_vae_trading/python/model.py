"""Disentangled VAE models for financial time series analysis.

This module provides several VAE variants designed to learn disentangled
latent representations of market data. Each latent dimension ideally
captures an independent factor such as trend direction, volatility regime,
or momentum strength.

Classes:
    DisentangledVAEConfig -- configuration dataclass for all model variants
    Encoder -- convolutional encoder network
    Decoder -- transposed-convolutional decoder network
    BetaVAE -- beta-VAE with adjustable KL weight
    FactorVAEDiscriminator -- MLP discriminator for Factor-VAE TC estimation
    FactorVAE -- Factor-VAE with adversarial Total Correlation penalty
    DisentangledVAEForecaster -- prediction head on top of any disentangled VAE
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DisentangledVAEConfig:
    """Configuration for disentangled VAE models.

    Attributes:
        input_features: Number of input feature channels (e.g. OHLCV + indicators).
        latent_dim: Dimensionality of the latent space.
        hidden_dims: Sizes of hidden layers in the encoder / decoder.
        beta: Weight applied to the KL divergence term.
        beta_schedule: Strategy for annealing beta during training.
        disentanglement_method: Which VAE variant to use.
        seq_len: Length of input time-series windows.
        prediction_horizon: Number of future steps to predict.
        gamma: Weight for Factor-VAE TC penalty or DIP-VAE regulariser.
        learning_rate: Optimiser learning rate.
    """

    input_features: int = 6
    latent_dim: int = 10
    hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128])
    beta: float = 4.0
    beta_schedule: Literal["warmup", "constant", "cyclical"] = "warmup"
    disentanglement_method: Literal[
        "beta_vae", "factor_vae", "dip_vae", "beta_tcvae"
    ] = "beta_vae"
    seq_len: int = 60
    prediction_horizon: int = 5
    gamma: float = 10.0
    learning_rate: float = 1e-3


# ---------------------------------------------------------------------------
# Encoder / Decoder building blocks
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """1-D convolutional encoder that maps (batch, features, seq_len) to
    a mean and log-variance vector in latent space."""

    def __init__(self, config: DisentangledVAEConfig) -> None:
        super().__init__()
        self.config = config

        layers: List[nn.Module] = []
        in_channels = config.input_features
        for h_dim in config.hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = h_dim
        self.conv = nn.Sequential(*layers)

        # Compute flattened size after convolutions
        dummy = torch.zeros(1, config.input_features, config.seq_len)
        with torch.no_grad():
            conv_out = self.conv(dummy)
        self._flat_size = conv_out.view(1, -1).size(1)

        self.fc_mu = nn.Linear(self._flat_size, config.latent_dim)
        self.fc_logvar = nn.Linear(self._flat_size, config.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) given input tensor of shape (B, C, T)."""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Transposed-convolutional decoder that maps a latent vector back to
    the original (batch, features, seq_len) shape."""

    def __init__(self, config: DisentangledVAEConfig, flat_size: int) -> None:
        super().__init__()
        self.config = config

        # Infer spatial size after encoder convolutions
        self._last_hidden = config.hidden_dims[-1]
        self._spatial = flat_size // self._last_hidden

        self.fc = nn.Linear(config.latent_dim, flat_size)

        layers: List[nn.Module] = []
        hidden_dims_rev = list(reversed(config.hidden_dims))
        for i in range(len(hidden_dims_rev) - 1):
            layers.extend([
                nn.ConvTranspose1d(
                    hidden_dims_rev[i], hidden_dims_rev[i + 1],
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                nn.BatchNorm1d(hidden_dims_rev[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Final layer projects back to input_features channels
        layers.append(
            nn.ConvTranspose1d(
                hidden_dims_rev[-1], config.input_features,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            )
        )
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector *z* to reconstructed time series."""
        h = self.fc(z)
        h = h.view(h.size(0), self._last_hidden, self._spatial)
        out = self.deconv(h)
        # Trim or pad to match the original sequence length
        target_len = self.config.seq_len
        if out.size(2) > target_len:
            out = out[:, :, :target_len]
        elif out.size(2) < target_len:
            out = F.pad(out, (0, target_len - out.size(2)))
        return out


# ---------------------------------------------------------------------------
# Beta-VAE
# ---------------------------------------------------------------------------

class BetaVAE(nn.Module):
    """Beta-VAE with tuneable KL weight and optional beta schedule.

    During training, ``current_beta`` is adjusted via ``update_beta()``
    according to the chosen schedule.
    """

    def __init__(self, config: DisentangledVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, self.encoder._flat_size)
        self.current_beta: float = 0.0 if config.beta_schedule == "warmup" else config.beta
        self._step: int = 0

    # ----- reparameterisation trick -----
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z from q(z|x) using the reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ----- forward -----
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            Tuple of (reconstruction, mu, logvar, z).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    # ----- beta scheduling -----
    def update_beta(self, epoch: int, total_epochs: int) -> float:
        """Update and return current_beta based on the schedule."""
        self._step = epoch
        schedule = self.config.beta_schedule
        target = self.config.beta

        if schedule == "constant":
            self.current_beta = target
        elif schedule == "warmup":
            warmup_epochs = max(total_epochs // 4, 1)
            self.current_beta = min(target, target * epoch / warmup_epochs)
        elif schedule == "cyclical":
            cycle_len = max(total_epochs // 4, 1)
            phase = (epoch % cycle_len) / cycle_len
            self.current_beta = target * phase
        else:
            self.current_beta = target

        return self.current_beta

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method to encode without decoding."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Convenience method to decode a latent vector."""
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Factor-VAE discriminator
# ---------------------------------------------------------------------------

class FactorVAEDiscriminator(nn.Module):
    """MLP discriminator for Factor-VAE.

    Distinguishes between samples from q(z) (the aggregate posterior)
    and permuted samples where each latent dimension is independently
    shuffled, thereby estimating Total Correlation.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 2),  # binary classification
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (B, 2)."""
        return self.net(z)


# ---------------------------------------------------------------------------
# Factor-VAE (VAE + discriminator)
# ---------------------------------------------------------------------------

class FactorVAE(nn.Module):
    """Factor-VAE: Beta-VAE augmented with an adversarial discriminator
    that penalises the Total Correlation of the aggregate posterior."""

    def __init__(self, config: DisentangledVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.vae = BetaVAE(config)
        self.discriminator = FactorVAEDiscriminator(config.latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE component."""
        return self.vae(x)

    def permute_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Permute each latent dimension independently across the batch."""
        z_perm = z.clone()
        batch_size = z.size(0)
        for dim in range(z.size(1)):
            perm_idx = torch.randperm(batch_size, device=z.device)
            z_perm[:, dim] = z[perm_idx, dim]
        return z_perm


# ---------------------------------------------------------------------------
# Disentangled VAE Forecaster
# ---------------------------------------------------------------------------

class DisentangledVAEForecaster(nn.Module):
    """Wraps a disentangled VAE with a prediction head that maps latent z
    to a trading signal (directional forecast) over a prediction horizon."""

    def __init__(self, config: DisentangledVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.vae = BetaVAE(config)
        self.prediction_head = nn.Sequential(
            nn.Linear(config.latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, config.prediction_horizon),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (prediction, reconstruction, mu, logvar, z)
        """
        recon, mu, logvar, z = self.vae(x)
        prediction = self.prediction_head(z)
        return prediction, recon, mu, logvar, z

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate directional predictions from input data."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            z = self.vae.reparameterize(mu, logvar)
            return self.prediction_head(z)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def beta_vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the beta-VAE loss.

    Args:
        recon: Reconstructed input, shape (B, C, T).
        x: Original input, shape (B, C, T).
        mu: Latent mean, shape (B, D).
        logvar: Latent log-variance, shape (B, D).
        beta: KL divergence weight.

    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def compute_total_correlation(
    z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Estimate Total Correlation using minibatch-weighted sampling.

    TC = KL[ q(z) || prod_j q(z_j) ]

    Uses the log-density-ratio trick from Chen et al. (2018).
    """
    batch_size, latent_dim = z.shape

    # log q(z) -- aggregate posterior evaluated at each sample
    # For each sample z_i, compute log q(z_i) = logsumexp over j of log q(z_i|x_j) - log N
    log_qz_given_x = -0.5 * (
        (z.unsqueeze(1) - mu.unsqueeze(0)).pow(2) / logvar.unsqueeze(0).exp()
        + logvar.unsqueeze(0)
        + math.log(2 * math.pi)
    )  # (B, B, D)

    # log q(z) -- joint
    log_qz = torch.logsumexp(log_qz_given_x.sum(dim=2), dim=1) - math.log(batch_size)

    # log prod_j q(z_j) -- marginals
    log_qz_marginals = (
        torch.logsumexp(log_qz_given_x, dim=1) - math.log(batch_size)
    ).sum(dim=1)

    tc = (log_qz - log_qz_marginals).mean()
    return tc


def factor_vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    discriminator: FactorVAEDiscriminator,
    z: torch.Tensor,
    gamma: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Factor-VAE loss (VAE side).

    The discriminator is trained separately; here we only use its output
    to estimate the TC penalty.

    Returns:
        (total_loss, recon_loss, kl_loss, tc_estimate)
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # TC estimated via discriminator
    d_logits = discriminator(z)
    tc_estimate = (d_logits[:, 0] - d_logits[:, 1]).mean()

    total = recon_loss + kl_loss + gamma * tc_estimate
    return total, recon_loss, kl_loss, tc_estimate


def dip_vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    lambda_diag: float = 10.0,
    lambda_offdiag: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the DIP-VAE-II loss.

    Encourages the covariance of the inferred mean to match the identity matrix,
    penalising off-diagonal (correlation) terms more heavily.

    Returns:
        (total_loss, recon_loss, kl_loss, dip_loss)
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Covariance of the mean vectors
    mu_centered = mu - mu.mean(dim=0, keepdim=True)
    cov_mu = (mu_centered.T @ mu_centered) / (mu.size(0) - 1)

    # Expectation of the variance
    exp_var = torch.mean(logvar.exp(), dim=0)
    cov_q = cov_mu + torch.diag(exp_var)

    # Penalties
    diag_penalty = (torch.diagonal(cov_q) - 1).pow(2).sum()
    off_diag = cov_q - torch.diag(torch.diagonal(cov_q))
    offdiag_penalty = off_diag.pow(2).sum()

    dip_loss = lambda_diag * diag_penalty + lambda_offdiag * offdiag_penalty
    total = recon_loss + kl_loss + dip_loss
    return total, recon_loss, kl_loss, dip_loss


def beta_tcvae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    beta: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the beta-TCVAE loss.

    Decomposes the KL term into:
        KL = Index-Code MI + beta * TC + Dimension-wise KL

    Returns:
        (total_loss, recon_loss, tc, dwkl)
    """
    batch_size, latent_dim = z.shape
    recon_loss = F.mse_loss(recon, x, reduction="mean")

    # log q(z|x)
    log_qz_given_x_per_dim = -0.5 * (
        (z - mu).pow(2) / logvar.exp() + logvar + math.log(2 * math.pi)
    )  # (B, D)
    log_qz_given_x = log_qz_given_x_per_dim.sum(dim=1)  # (B,)

    # log q(z) -- joint
    _diff = z.unsqueeze(1) - mu.unsqueeze(0)  # (B, B, D)
    _log_comp = -0.5 * (
        _diff.pow(2) / logvar.unsqueeze(0).exp()
        + logvar.unsqueeze(0)
        + math.log(2 * math.pi)
    )
    log_qz = torch.logsumexp(_log_comp.sum(dim=2), dim=1) - math.log(batch_size)

    # log prod_j q(z_j)
    log_qzj = (
        torch.logsumexp(_log_comp, dim=1) - math.log(batch_size)
    ).sum(dim=1)

    # log p(z) -- standard normal
    log_pz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=1)

    # Decomposition
    mi = (log_qz_given_x - log_qz).mean()
    tc = (log_qz - log_qzj).mean()
    dwkl = (log_qzj - log_pz).mean()

    total = recon_loss + mi + beta * tc + dwkl
    return total, recon_loss, tc, dwkl
