import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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

class MLP(nn.Module):
    """
    Multi-Layer Perceptron used for Projectors and Predictors.
    In BYOL, this predicts z -> q(z).
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

class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent architecture.
    """
    def __init__(self, base_encoder_dim=64, projection_dim=128, hidden_dim=256, m=0.9):
        super().__init__()
        self.m = m # Exponential Moving Average momentum
        
        # Online network: Encoder -> Projector -> Predictor
        self.online_encoder = CNN1DEncoder(hidden_dim=base_encoder_dim)
        self.online_projector = MLP(base_encoder_dim, hidden_dim, projection_dim)
        self.online_predictor = MLP(projection_dim, hidden_dim, projection_dim)
        
        # Target network: Encoder -> Projector (NO PREDICTOR)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Target network doesn't get updated by gradients
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        """
        Momentum update of target network weights.
        """
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = target_param.data * self.m + online_param.data * (1.0 - self.m)
            
        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data = target_param.data * self.m + online_param.data * (1.0 - self.m)

    def forward_online(self, x):
        """
        Input -> Encoder (y) -> Projector (z) -> Predictor (q)
        """
        y = self.online_encoder(x)
        z = self.online_projector(y)
        q = self.online_predictor(z)
        return F.normalize(q, dim=-1)

    @torch.no_grad()
    def forward_target(self, x):
        """
        Input -> Target Encoder -> Target Projector (Stop Gradient)
        """
        y = self.target_encoder(x)
        z = self.target_projector(y)
        return F.normalize(z, dim=-1).detach() # Crucial: detach (stop-gradient)

    def forward(self, v1, v2):
        """
        Returns the symmetrized BYOL loss.
        """
        # View 1 -> Online, View 2 -> Target
        q1 = self.forward_online(v1)
        z2_target = self.forward_target(v2)
        loss_1 = 2 - 2 * (q1 * z2_target).sum(dim=-1).mean()
        
        # View 2 -> Online, View 1 -> Target (Symmetrized)
        q2 = self.forward_online(v2)
        z1_target = self.forward_target(v1)
        loss_2 = 2 - 2 * (q2 * z1_target).sum(dim=-1).mean()
        
        return loss_1 + loss_2

if __name__ == "__main__":
    print("Testing Asymmetric Master-Apprentice (BYOL) logic...")
    model = BYOL()
    x1 = torch.randn(8, 1, 128)
    x2 = torch.randn(8, 1, 128)
    loss = model(x1, x2)
    print(f"Total Loss forward pass: {loss.item():.4f}")
    assert loss.requires_grad == True
    print("BYOL module initialized correctly.")
