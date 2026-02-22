import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConEncoder(nn.Module):
    """
    CNN Encoder for Supervised Contrastive Learning.
    Features:
    1. Base Encoder: Extracts features from time-series windows (x -> h).
    2. Projection Head: Non-linear MLP mapping features to a space where contrastive loss is applied (h -> z).
    """
    def __init__(self, input_dim=1, feature_dim=128, projection_dim=64):
        super(SupConEncoder, self).__init__()
        
        # Base Encoder: 1D-CNN
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Projection Head: 2-layer MLP as suggested in SimCLR/SupCon papers
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x):
        # x shape: (batch, input_dim, window_size)
        h = self.encoder(x).squeeze(-1)
        z = self.projector(h)
        # Normalize to unit hypersphere for cosine similarity
        z = F.normalize(z, p=2, dim=1)
        return h, z
