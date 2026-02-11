"""
Hull-White PINN Architecture

Physics-Informed Neural Network for solving the Hull-White bond pricing PDE:
    dP/dt + [theta(t) - a*r] * dP/dr + 0.5 * sigma^2 * d^2P/dr^2 - r*P = 0

The network takes (r, t, T) as input and outputs the bond price P(r, t; T).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class SinActivation(nn.Module):
    """Sinusoidal activation function for improved spectral bias."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class AdaptiveActivation(nn.Module):
    """Learnable activation function: a * tanh(b * x) with trainable a, b."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * torch.tanh(self.b * x)


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "sin": SinActivation(),
        "adaptive": AdaptiveActivation(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


class HullWhitePINN(nn.Module):
    """
    Physics-Informed Neural Network for Hull-White bond pricing.

    The network learns the mapping (r, t, T) -> P(r, t; T) where:
        r = short rate
        t = current time
        T = bond maturity
        P = zero-coupon bond price

    The Hull-White PDE is enforced through the loss function.

    Parameters:
        a: float - Mean reversion speed
        sigma: float - Short rate volatility
        hidden_layers: List[int] - Hidden layer sizes (default: [64, 64, 64, 32])
        activation: str - Activation function name (default: 'tanh')
        use_fourier_features: bool - Whether to use Fourier feature encoding
        n_fourier: int - Number of Fourier features
    """

    def __init__(
        self,
        a: float = 0.1,
        sigma: float = 0.01,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        use_fourier_features: bool = False,
        n_fourier: int = 32,
    ):
        super().__init__()

        self.a = a
        self.sigma = sigma
        self.use_fourier_features = use_fourier_features
        self.n_fourier = n_fourier

        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 32]

        # Input dimension: (r, t, T) = 3 or with Fourier features
        if use_fourier_features:
            input_dim = 3 + 2 * n_fourier * 3  # original + fourier for each input
            # Random frequencies for Fourier features (frozen)
            self.register_buffer(
                "fourier_B", torch.randn(3, n_fourier) * 2.0
            )
        else:
            input_dim = 3

        # Build network layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier uniform
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier uniform initialization for better convergence."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Fourier feature encoding for input."""
        # x: (batch, 3)
        proj = x @ self.fourier_B  # (batch, n_fourier)
        features = torch.cat([
            x,
            torch.sin(2 * np.pi * proj),
            torch.cos(2 * np.pi * proj),
        ], dim=-1)
        return features

    def forward(
        self, r: torch.Tensor, t: torch.Tensor, T: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: compute bond price P(r, t; T).

        Args:
            r: Short rate values, shape (batch, 1)
            t: Current time values, shape (batch, 1)
            T: Maturity values, shape (batch, 1)

        Returns:
            Bond prices, shape (batch, 1)
        """
        x = torch.cat([r, t, T], dim=-1)

        if self.use_fourier_features:
            x = self._fourier_features(x)

        raw_output = self.network(x)

        # Apply softplus to ensure positive bond prices
        price = torch.nn.functional.softplus(raw_output)

        return price

    def predict(self, r: float, t: float, T: float) -> float:
        """
        Predict a single bond price.

        Args:
            r: Short rate
            t: Current time
            T: Bond maturity

        Returns:
            Bond price as float
        """
        self.eval()
        with torch.no_grad():
            r_t = torch.tensor([[r]], dtype=torch.float32)
            t_t = torch.tensor([[t]], dtype=torch.float32)
            T_t = torch.tensor([[T]], dtype=torch.float32)
            price = self.forward(r_t, t_t, T_t)
        return price.item()

    def predict_batch(
        self, r: np.ndarray, t: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        """
        Predict bond prices for a batch of inputs.

        Args:
            r: Array of short rates
            t: Array of current times
            T: Array of maturities

        Returns:
            Array of bond prices
        """
        self.eval()
        with torch.no_grad():
            r_t = torch.tensor(r.reshape(-1, 1), dtype=torch.float32)
            t_t = torch.tensor(t.reshape(-1, 1), dtype=torch.float32)
            T_t = torch.tensor(T.reshape(-1, 1), dtype=torch.float32)
            prices = self.forward(r_t, t_t, T_t)
        return prices.numpy().flatten()

    def compute_pde_residual(
        self,
        r: torch.Tensor,
        t: torch.Tensor,
        T: torch.Tensor,
        theta_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Hull-White PDE residual using automatic differentiation.

        PDE: dP/dt + [theta(t) - a*r] * dP/dr + 0.5*sigma^2 * d^2P/dr^2 - r*P = 0

        Args:
            r: Short rate, shape (batch, 1), requires_grad=True
            t: Time, shape (batch, 1), requires_grad=True
            T: Maturity, shape (batch, 1)
            theta_t: Time-dependent drift theta(t), shape (batch, 1)

        Returns:
            PDE residual, shape (batch, 1)
        """
        P = self.forward(r, t, T)

        # Compute partial derivatives via autodiff
        dP_dt = torch.autograd.grad(
            P, t, grad_outputs=torch.ones_like(P),
            create_graph=True, retain_graph=True
        )[0]

        dP_dr = torch.autograd.grad(
            P, r, grad_outputs=torch.ones_like(P),
            create_graph=True, retain_graph=True
        )[0]

        d2P_dr2 = torch.autograd.grad(
            dP_dr, r, grad_outputs=torch.ones_like(dP_dr),
            create_graph=True, retain_graph=True
        )[0]

        # Hull-White PDE residual
        drift = theta_t - self.a * r
        residual = dP_dt + drift * dP_dr + 0.5 * self.sigma**2 * d2P_dr2 - r * P

        return residual

    def compute_greeks(
        self, r: float, t: float, T: float
    ) -> dict:
        """
        Compute bond price sensitivities (Greeks) using automatic differentiation.

        Returns:
            dict with keys: price, duration, convexity, theta, dP_dr, d2P_dr2
        """
        r_t = torch.tensor([[r]], dtype=torch.float32, requires_grad=True)
        t_t = torch.tensor([[t]], dtype=torch.float32, requires_grad=True)
        T_t = torch.tensor([[T]], dtype=torch.float32)

        P = self.forward(r_t, t_t, T_t)

        # dP/dr
        dP_dr = torch.autograd.grad(
            P, r_t, create_graph=True, retain_graph=True
        )[0]

        # d^2P/dr^2
        d2P_dr2 = torch.autograd.grad(
            dP_dr, r_t, create_graph=True, retain_graph=True
        )[0]

        # dP/dt
        dP_dt = torch.autograd.grad(
            P, t_t, create_graph=True, retain_graph=True
        )[0]

        price = P.item()
        return {
            "price": price,
            "duration": -dP_dr.item() / price if price > 1e-10 else 0.0,
            "convexity": d2P_dr2.item() / price if price > 1e-10 else 0.0,
            "theta": dP_dt.item(),
            "dP_dr": dP_dr.item(),
            "d2P_dr2": d2P_dr2.item(),
        }


class HullWhitePINNMultiMaturity(nn.Module):
    """
    Multi-output PINN that prices bonds for multiple maturities simultaneously.

    Instead of a single output, this network produces prices for a fixed set
    of maturities, enabling more efficient yield curve construction.

    Parameters:
        a: float - Mean reversion speed
        sigma: float - Short rate volatility
        maturities: List[float] - Fixed maturity schedule
        hidden_layers: List[int] - Hidden layer sizes
        activation: str - Activation function name
    """

    def __init__(
        self,
        a: float = 0.1,
        sigma: float = 0.01,
        maturities: Optional[List[float]] = None,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
    ):
        super().__init__()

        self.a = a
        self.sigma = sigma
        self.maturities = maturities or [
            0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0
        ]
        n_maturities = len(self.maturities)

        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 32]

        # Input: (r, t) = 2 dimensions
        layers = []
        prev_dim = 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_maturities))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute bond prices for all maturities.

        Args:
            r: Short rate, shape (batch, 1)
            t: Current time, shape (batch, 1)

        Returns:
            Bond prices, shape (batch, n_maturities)
        """
        x = torch.cat([r, t], dim=-1)
        raw = self.network(x)
        prices = torch.nn.functional.softplus(raw)
        return prices

    def yield_curve(self, r: float, t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the yield curve at a given rate and time.

        Returns:
            (maturities, yields) tuple
        """
        self.eval()
        with torch.no_grad():
            r_t = torch.tensor([[r]], dtype=torch.float32)
            t_t = torch.tensor([[t]], dtype=torch.float32)
            prices = self.forward(r_t, t_t).numpy().flatten()

        maturities = np.array(self.maturities)
        tau = maturities - t
        tau = np.maximum(tau, 1e-6)
        yields = -np.log(prices) / tau
        return maturities, yields


if __name__ == "__main__":
    # Quick test
    model = HullWhitePINN(a=0.1, sigma=0.01)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test forward pass
    r = torch.tensor([[0.03]], requires_grad=True)
    t = torch.tensor([[0.0]], requires_grad=True)
    T = torch.tensor([[5.0]])
    price = model(r, t, T)
    print(f"Bond price P(r=0.03, t=0, T=5): {price.item():.6f}")

    # Test PDE residual
    theta = torch.tensor([[0.01]])
    residual = model.compute_pde_residual(r, t, T, theta)
    print(f"PDE residual: {residual.item():.6e}")

    # Test Greeks
    greeks = model.compute_greeks(0.03, 0.0, 5.0)
    print(f"Greeks: {greeks}")

    # Test multi-maturity model
    multi_model = HullWhitePINNMultiMaturity(a=0.1, sigma=0.01)
    mats, ylds = multi_model.yield_curve(0.03)
    print(f"Yield curve maturities: {mats}")
    print(f"Yield curve yields: {ylds}")
