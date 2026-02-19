import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DEncoder(nn.Module):
    """
    1D-CNN Encoder to extract features from stock price windows.
    Input: (Batch, Channels, SeqLen)
    Output: Latent feature vector `h`.
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

class SimCLR(nn.Module):
    """
    Full SimCLR model with Encoder and Projection Head.
    """
    def __init__(self, encoder, hidden_dim=64, projection_dim=32):
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return F.normalize(z, dim=1)

if __name__ == "__main__":
    encoder = CNN1DEncoder()
    model = SimCLR(encoder)
    
    dummy_input = torch.randn(8, 1, 128)
    z = model(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Learned Representation (z): {z.shape}")
