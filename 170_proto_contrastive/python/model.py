import torch
import torch.nn as nn
import torch.nn.functional as F

class PCLCNNEncoder(nn.Module):
    """
    CNN Encoder for Prototypical Contrastive Learning.
    Matches the architecture of the contrastive series for consistency.
    """
    def __init__(self, input_dim=1, feature_dim=64, projection_dim=32):
        super(PCLCNNEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.projector = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x).squeeze(-1)
        z = self.projector(h)
        # Normalize for cosine similarity (important for PCL and InfoNCE)
        z = F.normalize(z, p=2, dim=1)
        return h, z
