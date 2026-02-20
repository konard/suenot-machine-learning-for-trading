import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DEncoder(nn.Module):
    """
    Standard 1D-CNN Encoder for capturing local temporal patterns.
    Outputs a continuous representation vector.
    """
    def __init__(self, in_channels=1, hidden_dim=128):
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
        
        # Linear layer to normalize embedding space to a specific dimension
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        h = self.conv_block(x)
        h = self.adaptive_pool(h).squeeze(-1)
        z = self.fc(h)
        # Normalize the embeddings to lay on a unit hypersphere
        # This makes Euclidean distance directly proportional to Cosine distance.
        return F.normalize(z, p=2, dim=1)

class TripletNet(nn.Module):
    """
    A wrapper network that handles triples: Anchor, Positive, Negative.
    Returns the embeddings for all three inputs.
    """
    def __init__(self, encoder: CNN1DEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, anchor, positive, negative):
        z_a = self.encoder(anchor)
        z_p = self.encoder(positive)
        z_n = self.encoder(negative)
        return z_a, z_p, z_n
