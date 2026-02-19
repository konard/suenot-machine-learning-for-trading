import torch
import torch.nn as nn
import numpy as np

class SirenLayer(nn.Module):
    """
    Sinusoidal Representation Layer (SIREN).
    Periodic activation allows for high-fidelity representation of 
    continuous signals and their derivatives.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        """
        Specific initialization for SIREN to ensure stable activation distribution.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SirenNet(nn.Module):
    """
    Multilayer SIREN network for Implicit Neural Representation.
    In Finance, this maps coordinates (e.g., moneyness, time-to-maturity) 
    to values (e.g., implied volatility).
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, omega_0=30):
        super().__init__()
        self.net = []
        # First layer
        self.net.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))

        # Hidden layers
        for _ in range(hidden_layers):
            self.net.append(SirenLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))

        # Final linear layer (no activation)
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6 / hidden_features) / omega_0,
                np.sqrt(6 / hidden_features) / omega_0
            )
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        """
        coords: (batch, in_features) -> e.g., [moneyness, maturity]
        """
        return self.net(coords)

if __name__ == "__main__":
    print("Initializing SIREN Network for Volatility Surface Modeling...")
    # Mapping 2D coords (Moneyness, Maturity) to 1D value (Volatility)
    model = SirenNet(in_features=2, hidden_features=128, hidden_layers=3, out_features=1)
    
    dummy_input = torch.randn(10, 2)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("SIREN Architecture initialized successfully.")
