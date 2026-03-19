import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesEncoder(nn.Module):
    """
    1D-CNN encoder that maps a raw time-series window to a latent embedding.
    Used as the backbone for InfoNCE contrastive learning.
    """
    def __init__(self, input_dim=1, embed_dim=64):
        super(TimeSeriesEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # Projection head: maps encoder output to a space where InfoNCE operates
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim, window_size)
        Returns:
            z: (batch_size, embed_dim)
        """
        h = self.conv(x).squeeze(-1)
        z = self.projection(h)
        return z


class InfoNCEModel(nn.Module):
    """
    Full InfoNCE model for time-series contrastive learning.

    Positive pairs are constructed from augmented views of the same window.
    Augmentations include: jitter, scaling, and time shifting.
    """
    def __init__(self, input_dim=1, embed_dim=64, temperature=0.07):
        super(InfoNCEModel, self).__init__()
        self.encoder = TimeSeriesEncoder(input_dim, embed_dim)
        self.temperature = temperature

    def augment(self, x):
        """
        Applies random augmentations to create a different 'view' of the same window.
        """
        # Gaussian jitter
        noise = torch.randn_like(x) * 0.05
        x_aug = x + noise

        # Random scaling
        scale = 0.9 + torch.rand(x.size(0), 1, 1, device=x.device) * 0.2
        x_aug = x_aug * scale

        # Random time shift (roll along time axis)
        shift = torch.randint(-3, 4, (1,)).item()
        x_aug = torch.roll(x_aug, shifts=shift, dims=-1)

        return x_aug

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim, window_size) — raw time-series windows
        Returns:
            z_anchor: (batch_size, embed_dim) — anchor embeddings
            z_positive: (batch_size, embed_dim) — positive embeddings (augmented views)
        """
        x_aug = self.augment(x)
        z_anchor = self.encoder(x)
        z_positive = self.encoder(x_aug)
        return z_anchor, z_positive
