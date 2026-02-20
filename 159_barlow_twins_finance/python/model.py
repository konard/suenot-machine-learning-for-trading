import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1DEncoder(nn.Module):
    """
    Base 1D-CNN Encoder for stock price windows.
    Returns representation y.
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
    Projection head mapping representations to the space where 
    the cross-correlation loss is applied.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class BarlowTwinsLoss(nn.Module):
    """
    Computes the Barlow Twins loss:
    1. Invariance (diagonal pushing to 1)
    2. Redundancy Reduction (off-diagonal pushing to 0)
    """
    def __init__(self, lambda_param=0.0051):
        super().__init__()
        self.lambda_param = lambda_param

    def forward(self, z1, z2):
        """
        z1: (Batch, Dim) embeddings of augmented view 1
        z2: (Batch, Dim) embeddings of augmented view 2
        """
        N, D = z1.size()

        # Empirical cross-correlation matrix
        # 1. Zero mean mapping
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-5)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-5)

        # 2. Cross-correlation matrix C (size D x D)
        c = torch.mm(z1_norm.T, z2_norm) / N

        # 3. Loss = (Invariance Term) + lambda * (Redundancy Reduction Term)
        # Invariance (Diagonal of C must be close to 1)
        c_diff = (c - torch.eye(D, device=c.device)).pow(2)
        
        # Multiply off-diagonal by lambda
        off_diagonal_mask = 1 - torch.eye(D, device=c.device)
        c_diff = c_diff * torch.where(off_diagonal_mask.bool(), self.lambda_param, 1.0)

        # Sum over all elements
        loss = c_diff.sum()
        
        return loss

class BarlowTwins(nn.Module):
    def __init__(self, base_encoder_dim=64, projection_dim=128, hidden_dim=256, lambda_param=0.0051):
        super().__init__()
        self.encoder = CNN1DEncoder(hidden_dim=base_encoder_dim)
        self.projector = MLPProjector(base_encoder_dim, hidden_dim, projection_dim)
        self.criterion = BarlowTwinsLoss(lambda_param=lambda_param)
        
    def forward(self, v1, v2):
        """
        The Barlow Twins architecture is symmetric. Both views pass through identical networks.
        """
        z1 = self.projector(self.encoder(v1))
        z2 = self.projector(self.encoder(v2))
        
        loss = self.criterion(z1, z2)
        return loss

if __name__ == "__main__":
    print("Testing Symmetric Barlow Twins Architecture...")
    model = BarlowTwins()
    x1 = torch.randn(8, 1, 128)
    x2 = torch.randn(8, 1, 128)
    loss = model(x1, x2)
    print(f"Total Loss forward pass: {loss.item():.4f}")
    assert loss.requires_grad == True
    print("Barlow Twins initialized correctly. Cross-correlation matrix logic operational.")
