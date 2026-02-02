"""
HiPPO (High-order Polynomial Projection Operators) model for trading.

This module provides:
- HiPPOLegS: Scaled Legendre measure (full history, uniform weight)
- HiPPOLagT: Translated Laguerre measure (exponential decay)
- HiPPOTradingModel: Complete trading model with HiPPO feature extraction
- HiPPOPredictor: Neural network head for signal prediction

Reference:
    Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections"
    NeurIPS 2020. arXiv:2008.07669
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HiPPOLegS(nn.Module):
    """
    HiPPO-LegS (Scaled Legendre) operator.

    Uses the measure mu(t) = 1/t * I_{[0,t]}, which uniformly weights
    the entire history. The state matrix A is lower-triangular.

    Args:
        N: Number of polynomial coefficients (state dimension).
    """

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        A, B = self._build_matrices(N)
        self.register_buffer('A', A)
        self.register_buffer('B', B)

    @staticmethod
    def _build_matrices(N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the HiPPO-LegS A and B matrices."""
        A = torch.zeros(N, N)
        B = torch.zeros(N, 1)
        for n in range(N):
            B[n, 0] = (2 * n + 1) ** 0.5
            for k in range(n + 1):
                if n > k:
                    A[n, k] = -(2 * n + 1) ** 0.5 * (2 * k + 1) ** 0.5
                elif n == k:
                    A[n, k] = -(n + 1)
        return A, B

    def discretize(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize the continuous system using the bilinear (Tustin) transform.

        Args:
            dt: Time step size.

        Returns:
            Tuple of (A_bar, B_bar) discrete-time matrices.
        """
        I = torch.eye(self.N, device=self.A.device)
        # Bilinear transform: (I - dt/2 * A)^{-1} (I + dt/2 * A)
        A_bar = torch.linalg.solve(I - dt / 2 * self.A, I + dt / 2 * self.A)
        B_bar = torch.linalg.solve(I - dt / 2 * self.A, dt * self.B)
        return A_bar, B_bar

    def forward(self, inputs: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Process a sequence through HiPPO-LegS dynamics.

        Args:
            inputs: Input tensor of shape (batch, seq_len) or (batch, seq_len, 1).
            dt: Time step size.

        Returns:
            Coefficient tensor of shape (batch, seq_len, N).
        """
        if inputs.dim() == 3:
            inputs = inputs.squeeze(-1)

        batch_size, seq_len = inputs.shape
        A_bar, B_bar = self.discretize(dt)

        c = torch.zeros(batch_size, self.N, device=inputs.device)
        outputs = []

        for t in range(seq_len):
            f = inputs[:, t:t + 1]  # (batch, 1)
            # c_{k+1} = A_bar @ c_k + B_bar * f_k
            c = (A_bar @ c.unsqueeze(-1)).squeeze(-1) + (B_bar * f.unsqueeze(-1)).squeeze(-1)
            outputs.append(c)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, N)


class HiPPOLagT(nn.Module):
    """
    HiPPO-LagT (Translated Laguerre) operator.

    Uses exponentially decaying measure mu(t) = e^{-(t-s)} * I_{[0,t]}.
    The A matrix is lower-triangular with all entries = -1.

    Args:
        N: Number of polynomial coefficients (state dimension).
    """

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        A, B = self._build_matrices(N)
        self.register_buffer('A', A)
        self.register_buffer('B', B)

    @staticmethod
    def _build_matrices(N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the HiPPO-LagT A and B matrices."""
        A = torch.zeros(N, N)
        B = torch.ones(N, 1)
        for n in range(N):
            for k in range(n + 1):
                A[n, k] = -1.0
        return A, B

    def discretize(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discretize using bilinear transform."""
        I = torch.eye(self.N, device=self.A.device)
        A_bar = torch.linalg.solve(I - dt / 2 * self.A, I + dt / 2 * self.A)
        B_bar = torch.linalg.solve(I - dt / 2 * self.A, dt * self.B)
        return A_bar, B_bar

    def forward(self, inputs: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Process a sequence through HiPPO-LagT dynamics.

        Args:
            inputs: Input tensor of shape (batch, seq_len) or (batch, seq_len, 1).
            dt: Time step size.

        Returns:
            Coefficient tensor of shape (batch, seq_len, N).
        """
        if inputs.dim() == 3:
            inputs = inputs.squeeze(-1)

        batch_size, seq_len = inputs.shape
        A_bar, B_bar = self.discretize(dt)

        c = torch.zeros(batch_size, self.N, device=inputs.device)
        outputs = []

        for t in range(seq_len):
            f = inputs[:, t:t + 1]
            c = (A_bar @ c.unsqueeze(-1)).squeeze(-1) + (B_bar * f.unsqueeze(-1)).squeeze(-1)
            outputs.append(c)

        return torch.stack(outputs, dim=1)


class HiPPOPredictor(nn.Module):
    """
    Neural network prediction head for trading signals.

    Takes HiPPO coefficient features and predicts trading signals.

    Args:
        input_size: Total feature dimension (N * num_features).
        hidden_size: Hidden layer size.
        dropout: Dropout probability.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class HiPPOConfig:
    """Configuration for HiPPO Trading Model."""
    n_hippo: int = 64
    variant: str = "legs"  # "legs" or "lagt"
    input_features: int = 5  # OHLCV
    hidden_size: int = 64
    dropout: float = 0.1
    dt: float = 1.0


class HiPPOTradingModel(nn.Module):
    """
    Complete HiPPO-based trading model.

    Extracts polynomial features from multiple input channels (price, volume,
    returns, etc.) using HiPPO operators, then feeds them into a prediction
    head for signal generation.

    Args:
        config: HiPPOConfig with model hyperparameters.
    """

    def __init__(self, config: Optional[HiPPOConfig] = None):
        super().__init__()
        if config is None:
            config = HiPPOConfig()
        self.config = config

        # Create HiPPO operator for each input feature
        HiPPOClass = HiPPOLegS if config.variant == "legs" else HiPPOLagT
        self.hippo_layers = nn.ModuleList([
            HiPPOClass(config.n_hippo) for _ in range(config.input_features)
        ])

        # Prediction head: takes concatenated HiPPO features
        total_features = config.n_hippo * config.input_features
        self.predictor = HiPPOPredictor(
            total_features, config.hidden_size, config.dropout
        )

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract HiPPO features from multi-channel input.

        Args:
            inputs: (batch, seq_len, n_features) tensor.

        Returns:
            (batch, seq_len, n_hippo * n_features) feature tensor.
        """
        features = []
        for i, hippo in enumerate(self.hippo_layers):
            channel = inputs[:, :, i]  # (batch, seq_len)
            coeffs = hippo(channel, dt=self.config.dt)  # (batch, seq_len, N)
            features.append(coeffs)

        return torch.cat(features, dim=-1)  # (batch, seq_len, N * n_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features and predict signals.

        Args:
            inputs: (batch, seq_len, n_features) tensor of OHLCV data.

        Returns:
            (batch, seq_len) tensor of trading signals in [-1, 1].
        """
        features = self.extract_features(inputs)
        batch, seq_len, feat_dim = features.shape

        # Apply predictor at each time step
        features_flat = features.reshape(-1, feat_dim)
        signals_flat = self.predictor(features_flat)
        signals = signals_flat.reshape(batch, seq_len)

        return signals


def demo():
    """Run a standalone demo of the HiPPO model."""
    logger.info("=== HiPPO Framework Demo ===")

    # Generate synthetic price-like data
    np.random.seed(42)
    T = 200
    trend = np.cumsum(np.random.randn(T) * 0.5) + 100
    noise = np.random.randn(T) * 0.5
    prices = trend + noise

    # Normalize
    prices_norm = (prices - prices.mean()) / prices.std()
    inputs = torch.tensor(prices_norm, dtype=torch.float32).unsqueeze(0)  # (1, T)

    # Test HiPPO-LegS
    logger.info("Testing HiPPO-LegS (N=16)...")
    hippo_legs = HiPPOLegS(N=16)
    coeffs_legs = hippo_legs(inputs, dt=1.0)
    logger.info(f"  Input shape: {inputs.shape}")
    logger.info(f"  Output shape: {coeffs_legs.shape}")
    logger.info(f"  Coefficients at t=100: {coeffs_legs[0, 100, :4].detach().numpy()}")

    # Test HiPPO-LagT
    logger.info("Testing HiPPO-LagT (N=16)...")
    hippo_lagt = HiPPOLagT(N=16)
    coeffs_lagt = hippo_lagt(inputs, dt=1.0)
    logger.info(f"  Output shape: {coeffs_lagt.shape}")
    logger.info(f"  Coefficients at t=100: {coeffs_lagt[0, 100, :4].detach().numpy()}")

    # Test full trading model
    logger.info("Testing HiPPOTradingModel...")
    config = HiPPOConfig(n_hippo=16, input_features=3, hidden_size=32)
    model = HiPPOTradingModel(config)

    # Synthetic OHLCV-like data (using 3 features for demo)
    multi_input = torch.randn(2, 100, 3)  # (batch=2, seq=100, features=3)
    signals = model(multi_input)
    logger.info(f"  Input shape: {multi_input.shape}")
    logger.info(f"  Signal shape: {signals.shape}")
    logger.info(f"  Signal range: [{signals.min().item():.3f}, {signals.max().item():.3f}]")

    logger.info("=== Demo Complete ===")


if __name__ == "__main__":
    demo()
