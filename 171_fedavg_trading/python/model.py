import torch
import torch.nn as nn

class TradingNN(nn.Module):
    """
    Simple MLP for return prediction, used as the global model in FedAvg.
    """
    def __init__(self, input_dim=20, hidden_dim=64):
        super(TradingNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Predicting return sign or magnitude
        )

    def forward(self, x):
        return self.net(x)

def get_model_weights(model):
    """Deep copy of model weights."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def set_model_weights(model, weights):
    """Load weights into model."""
    model.load_state_dict(weights)
