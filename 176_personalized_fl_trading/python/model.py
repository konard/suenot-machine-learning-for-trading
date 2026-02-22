import torch
import torch.nn as nn

class TradingBody(nn.Module):
    """
    Shared 'Body' of the model. Learns universal market patterns.
    """
    def __init__(self, input_dim=20, hidden_dim=64):
        super(TradingBody, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class TradingHead(nn.Module):
    """
    Personalized 'Head' of the model. Learns local asset specifics.
    """
    def __init__(self, hidden_dim=64):
        super(TradingHead, self).__init__()
        self.net = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.net(x)

class PersonalizedTradingNN(nn.Module):
    """
    Modular network combining shared body and personalized head.
    """
    def __init__(self, input_dim=20, hidden_dim=64):
        super(PersonalizedTradingNN, self).__init__()
        self.body = TradingBody(input_dim, hidden_dim)
        self.head = TradingHead(hidden_dim)

    def forward(self, x):
        features = self.body(x)
        return self.head(features)
