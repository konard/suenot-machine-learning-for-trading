import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DEncoder(nn.Module):
    """
    Standard 1D-CNN Encoder for capturing local temporal patterns.
    """
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        h = self.conv_block(x)
        h = self.adaptive_pool(h).squeeze(-1)
        return h

class MLPProjector(nn.Module):
    """
    3-layer Projector as recommended in the VICReg paper.
    """
    def __init__(self, in_dim, hidden_dim=256, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class VICRegLoss(nn.Module):
    """
    Implementation of Variance-Invariance-Covariance Regularization Loss.
    """
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, gamma=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    def forward(self, z1, z2):
        # 1. Invariance Loss (Mean Squared Error)
        sim_loss = F.mse_loss(z1, z2)

        # 2. Variance Loss (Standard Deviation Regularization)
        # We want the std along the batch dimension to be >= gamma
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-04)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-04)
        std_loss = torch.mean(F.relu(self.gamma - std_z1)) + torch.mean(F.relu(self.gamma - std_z2))

        # 3. Covariance Loss (Off-diagonal decorrelation)
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        
        batch_size = z1.size(0)
        dim = z1.size(1)

        cov_z1 = (z1.T @ z1) / (batch_size - 1)
        cov_z2 = (z2.T @ z2) / (batch_size - 1)
        
        cov_loss = self.off_diagonal(cov_z1).pow_(2).sum().div(dim) + \
                   self.off_diagonal(cov_z2).pow_(2).sum().div(dim)

        return self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICReg(nn.Module):
    """
    Unified VICReg model combining Encoder and Projector.
    """
    def __init__(self, base_encoder_dim=64, projection_dim=256):
        super().__init__()
        self.encoder = CNN1DEncoder(hidden_dim=base_encoder_dim)
        self.projector = MLPProjector(base_encoder_dim, out_dim=projection_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z
