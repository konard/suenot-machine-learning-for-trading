import torch
import torch.nn as nn

class TradingNN(nn.Module):
    """
    Standard MLP for return prediction, used in the Asynchronous FL simulation.
    """
    def __init__(self, input_dim=20, hidden_dim=64):
        super(TradingNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)
