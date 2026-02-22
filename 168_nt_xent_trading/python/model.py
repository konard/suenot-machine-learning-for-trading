import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    Standard CNN Encoder with Projection Head for Contrastive Learning.
    """
    def __init__(self, input_dim=1, feature_dim=64, projection_dim=32):
        super(CNNEncoder, self).__init__()
        
        # Base Encoder: Extracts representations from raw time-series
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Projection Head: Maps representations to the space where NT-Xent is applied
        self.projector = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x).squeeze(-1)
        z = self.projector(h)
        return z
