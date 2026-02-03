"""
S4 (Structured State Space) model implementation for trading.

This module implements the S4 architecture for sequence modeling in financial
applications. S4 efficiently handles long-range dependencies with linear
computational complexity.

References:
    Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces"
    ICLR 2022. arXiv:2111.00396
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class S4Config:
    """Configuration for S4 model."""
    d_model: int = 64
    d_state: int = 64
    n_layers: int = 4
    dropout: float = 0.1
    bidirectional: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1


def hippo_initializer(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize HiPPO-LegS matrices for optimal sequence memorization.

    The HiPPO framework provides an initialization that enables optimal
    memorization of input history using Legendre polynomials.

    Args:
        n: State dimension

    Returns:
        A: State transition matrix (n, n)
        B: Input matrix (n, 1)
    """
    # HiPPO-LegS diagonal approximation
    # A eigenvalues: -(n + 1/2) + i * sqrt(n*(n+1))
    A = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        A[i, i] = -(i + 0.5) + 1j * np.sqrt(i * (i + 1))

    # B: sqrt(2n + 1)
    B = np.sqrt(2 * np.arange(n) + 1).reshape(-1, 1)

    return A, B


class S4Layer(nn.Module):
    """
    S4 Layer implementing structured state space.

    The layer processes sequences through the state space equations:
        x_k = A_bar * x_{k-1} + B_bar * u_k
        y_k = C * x_k + D * u_k

    Args:
        d_model: Input/output dimension
        d_state: State dimension (N)
        dropout: Dropout rate
        dt_min: Minimum step size
        dt_max: Maximum step size
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Initialize HiPPO matrices
        A, B = hippo_initializer(d_state)

        # Store as parameters (diagonal A)
        self.A_real = nn.Parameter(torch.tensor(A.real.diagonal(), dtype=torch.float32))
        self.A_imag = nn.Parameter(torch.tensor(A.imag.diagonal(), dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(B.real.flatten(), dtype=torch.float32))

        # Learnable C and D
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))

        # Learnable step size (log scale)
        log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        self.dropout = nn.Dropout(dropout)

        # Cache for discretized matrices
        self._A_bar = None
        self._B_bar = None

    def _discretize(self):
        """Discretize continuous-time SSM to discrete-time using ZOH."""
        dt = self.log_dt.exp()  # (d_model,)

        # A_bar = exp(A * dt) for diagonal A
        # Shape: (d_model, d_state)
        A = self.A_real + 1j * self.A_imag  # (d_state,)
        dt_expanded = dt.unsqueeze(-1)  # (d_model, 1)
        A_expanded = A.unsqueeze(0)  # (1, d_state)

        A_bar = torch.exp(A_expanded * dt_expanded)  # (d_model, d_state), complex

        # B_bar = (exp(A*dt) - 1) / A * B
        B_expanded = self.B.unsqueeze(0)  # (1, d_state)
        denom = A_expanded + 1e-8
        B_bar = ((A_bar - 1) / denom) * B_expanded  # (d_model, d_state), complex

        self._A_bar = A_bar
        self._B_bar = B_bar

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Process input sequence.

        Args:
            u: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = u.shape

        # Discretize
        self._discretize()

        # Initialize state
        x_real = torch.zeros(batch_size, d_model, self.d_state, device=u.device)
        x_imag = torch.zeros(batch_size, d_model, self.d_state, device=u.device)

        outputs = []

        # Recurrent computation
        for t in range(seq_len):
            u_t = u[:, t, :]  # (batch, d_model)

            # x_k = A_bar * x_{k-1} + B_bar * u_k
            A_bar_real = self._A_bar.real  # (d_model, d_state)
            A_bar_imag = self._A_bar.imag
            B_bar_real = self._B_bar.real
            B_bar_imag = self._B_bar.imag

            u_expanded = u_t.unsqueeze(-1)  # (batch, d_model, 1)

            new_x_real = (A_bar_real * x_real - A_bar_imag * x_imag +
                         B_bar_real * u_expanded)
            new_x_imag = (A_bar_real * x_imag + A_bar_imag * x_real +
                         B_bar_imag * u_expanded)

            x_real = new_x_real
            x_imag = new_x_imag

            # y_k = Re(C * x_k) + D * u_k
            # C: (d_model, d_state), x: (batch, d_model, d_state)
            y = (self.C_real * x_real - self.C_imag * x_imag).sum(dim=-1)
            y = y + self.D * u_t

            outputs.append(y)

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return self.dropout(output)

    def step(
        self,
        u_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process a single timestep (for inference).

        Args:
            u_t: Input tensor of shape (batch, d_model)
            state: Previous state (x_real, x_imag) or None

        Returns:
            y_t: Output tensor of shape (batch, d_model)
            new_state: Updated state
        """
        batch_size, d_model = u_t.shape

        if state is None:
            x_real = torch.zeros(batch_size, d_model, self.d_state, device=u_t.device)
            x_imag = torch.zeros(batch_size, d_model, self.d_state, device=u_t.device)
        else:
            x_real, x_imag = state

        # Discretize if needed
        if self._A_bar is None:
            self._discretize()

        A_bar_real = self._A_bar.real
        A_bar_imag = self._A_bar.imag
        B_bar_real = self._B_bar.real
        B_bar_imag = self._B_bar.imag

        u_expanded = u_t.unsqueeze(-1)

        new_x_real = (A_bar_real * x_real - A_bar_imag * x_imag +
                     B_bar_real * u_expanded)
        new_x_imag = (A_bar_real * x_imag + A_bar_imag * x_real +
                     B_bar_imag * u_expanded)

        y_t = (self.C_real * new_x_real - self.C_imag * new_x_imag).sum(dim=-1)
        y_t = y_t + self.D * u_t

        return y_t, (new_x_real, new_x_imag)


class S4DLayer(nn.Module):
    """
    Simplified Diagonal S4 (S4D) layer.

    Uses purely diagonal A matrix for simpler implementation.
    Often achieves competitive performance with less complexity.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Initialize diagonal A (negative for stability)
        A_diag = -torch.arange(d_state, dtype=torch.float32) - 0.5
        self.A = nn.Parameter(A_diag)

        # Initialize B
        B = torch.sqrt(2 * torch.arange(d_state, dtype=torch.float32) + 1)
        self.B = nn.Parameter(B)

        # Learnable C and D
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))

        # Learnable step size
        log_dt = torch.rand(d_model) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Process input sequence."""
        batch_size, seq_len, d_model = u.shape
        dt = self.log_dt.exp()

        # Discretize
        A_bar = torch.exp(self.A.unsqueeze(0) * dt.unsqueeze(-1))  # (d_model, d_state)
        B_bar = ((A_bar - 1) / (self.A.unsqueeze(0) + 1e-8)) * self.B.unsqueeze(0)

        # Initialize state
        x = torch.zeros(batch_size, d_model, self.d_state, device=u.device)

        outputs = []
        for t in range(seq_len):
            u_t = u[:, t, :].unsqueeze(-1)  # (batch, d_model, 1)
            x = A_bar * x + B_bar * u_t
            y = (self.C * x).sum(dim=-1) + self.D * u[:, t, :]
            outputs.append(y)

        return self.dropout(torch.stack(outputs, dim=1))


class S4Block(nn.Module):
    """S4 block with residual connection, normalization, and GLU."""

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s4 = S4Layer(d_model, d_state, dropout)
        self.linear = nn.Linear(d_model, d_model * 2)  # For GLU
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual and GLU."""
        residual = x
        x = self.norm(x)
        x = self.s4(x)
        x = self.linear(x)
        x = F.glu(x, dim=-1)
        x = self.dropout(x)
        return x + residual


class S4TradingModel(nn.Module):
    """
    S4-based trading signal generator.

    Architecture:
        Input -> Embedding -> [S4Block]×N -> Pooling -> Output Head
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        d_state: int = 64,
        n_layers: int = 4,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # S4 layers
        self.layers = nn.ModuleList([
            S4Block(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, seq_len, input_dim)

        Returns:
            Logits of shape (batch, n_classes)
        """
        # Embed
        x = self.embedding(x)

        # S4 layers
        for layer in self.layers:
            x = layer(x)

        # Pool (use last timestep)
        x = x[:, -1, :]

        # Output
        x = self.output_norm(x)
        return self.output(x)

    def predict(self, features: np.ndarray) -> dict:
        """
        Make prediction from numpy features.

        Args:
            features: Features of shape (seq_len, input_dim)

        Returns:
            Dictionary with signal, confidence, and probabilities
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

        signal_map = {0: "BUY", 1: "HOLD", 2: "SELL"}
        signal_idx = np.argmax(probs)

        return {
            "signal": signal_map[signal_idx],
            "confidence": float(probs[signal_idx]),
            "probabilities": {
                "BUY": float(probs[0]),
                "HOLD": float(probs[1]),
                "SELL": float(probs[2]),
            }
        }

    def get_state(self, features: np.ndarray) -> np.ndarray:
        """Get hidden state from features."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return x[:, -1, :].squeeze(0).numpy()


def create_model(config: S4Config) -> S4TradingModel:
    """Create S4 trading model from config."""
    return S4TradingModel(
        input_dim=5,
        d_model=config.d_model,
        d_state=config.d_state,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )


if __name__ == "__main__":
    # Demo
    print("S4 Trading Model Demo")
    print("=" * 50)

    # Create model
    config = S4Config(d_model=64, d_state=64, n_layers=4)
    model = create_model(config)

    print(f"\nModel configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_state: {config.d_state}")
    print(f"  n_layers: {config.n_layers}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Generate random input
    batch_size = 2
    seq_len = 256
    input_dim = 5

    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")

    # Test prediction
    features = np.random.randn(256, 5).astype(np.float32)
    prediction = model.predict(features)

    print(f"\nPrediction:")
    print(f"  Signal: {prediction['signal']}")
    print(f"  Confidence: {prediction['confidence']:.2%}")
    print(f"  Probabilities: {prediction['probabilities']}")

    print("\n" + "=" * 50)
    print("Demo complete!")
