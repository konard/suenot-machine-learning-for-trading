"""
TimeParticle SSM - Core Model Implementation

This module implements the TimeParticle State Space Model architecture
for multiscale time series forecasting in financial markets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict


class Particle(nn.Module):
    """
    Single particle operating at a specific time scale.

    Each particle captures temporal dynamics at its designated scale using
    a gated state space model with scale-specific convolutions.

    Args:
        d_input: Input feature dimension
        d_hidden: Hidden dimension
        d_state: State dimension for the SSM
        scale_factor: Scale factor (1, 2, 3, ...) determines temporal resolution
    """

    def __init__(self, d_input: int, d_hidden: int, d_state: int, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
        self.d_hidden = d_hidden
        self.d_state = d_state

        # Scale-specific convolution with dilation
        kernel_size = min(2 ** scale_factor, 16)  # Cap kernel size
        padding = (kernel_size - 1) * (2 ** (scale_factor - 1)) // 2

        self.scale_conv = nn.Conv1d(
            d_input, d_hidden,
            kernel_size=kernel_size,
            padding=padding,
            dilation=2 ** (scale_factor - 1)
        )

        # Gated state space parameters
        self.W_gate = nn.Linear(d_hidden + d_state, d_state)
        self.W_state = nn.Linear(d_hidden, d_state)
        self.U_state = nn.Linear(d_state, d_state, bias=False)

        # Message integration (for cross-scale communication)
        self.message_proj = nn.Linear(d_hidden, d_hidden)

        # Output projection
        self.output_proj = nn.Linear(d_state, d_hidden)

        # Layer normalization for stability
        self.norm = nn.LayerNorm(d_hidden)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.xavier_uniform_(self.W_state.weight)
        nn.init.orthogonal_(self.U_state.weight)
        nn.init.zeros_(self.W_gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        prev_state: torch.Tensor,
        message: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the particle.

        Args:
            x: Input tensor [batch, seq_len, d_input]
            prev_state: Previous hidden state [batch, d_state]
            message: Cross-scale message from other particles [batch, seq_len, d_hidden]

        Returns:
            output: Particle output [batch, seq_len, d_hidden]
            final_state: Final hidden state [batch, d_state]
        """
        batch, seq_len, _ = x.shape

        # Apply scale-specific convolution
        x_conv = self.scale_conv(x.transpose(1, 2)).transpose(1, 2)

        # Ensure sequence length matches
        x_scaled = x_conv[:, :seq_len, :]
        x_scaled = self.norm(x_scaled)

        # Process sequence with gated SSM
        states = []
        state = prev_state

        for t in range(seq_len):
            x_t = x_scaled[:, t, :]

            # Integrate cross-scale message if provided
            if message is not None:
                msg_t = self.message_proj(message[:, t, :])
                x_t = x_t + msg_t

            # Compute gate
            gate_input = torch.cat([x_t, state], dim=-1)
            gate = torch.sigmoid(self.W_gate(gate_input))

            # Compute state candidate
            state_candidate = torch.tanh(
                self.W_state(x_t) + self.U_state(state)
            )

            # Gated state update
            state = gate * state_candidate + (1 - gate) * state
            states.append(state)

        # Stack states and project to output
        states = torch.stack(states, dim=1)
        output = self.output_proj(states)

        return output, state


class CrossScaleAttention(nn.Module):
    """
    Attention mechanism for cross-scale particle communication.

    Enables particles to share information across different time scales
    using multi-head attention.

    Args:
        d_hidden: Hidden dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_hidden: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_hidden % n_heads == 0, "d_hidden must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = d_hidden // n_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(d_hidden, d_hidden)
        self.W_k = nn.Linear(d_hidden, d_hidden)
        self.W_v = nn.Linear(d_hidden, d_hidden)
        self.W_o = nn.Linear(d_hidden, d_hidden)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        keys: List[torch.Tensor],
        values: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute cross-scale attention.

        Args:
            query: Query from current particle [batch, seq_len, d_hidden]
            keys: List of keys from other particles
            values: List of values from other particles

        Returns:
            Attended features [batch, seq_len, d_hidden]
        """
        batch, seq_len, d_hidden = query.shape
        n_particles = len(keys)

        if n_particles == 0:
            return torch.zeros_like(query)

        # Project query
        Q = self.W_q(query).view(batch, seq_len, self.n_heads, self.head_dim)

        # Stack and project keys/values from other particles
        K_list = [self.W_k(k) for k in keys]
        V_list = [self.W_v(v) for v in values]

        K = torch.stack(K_list, dim=2)  # [batch, seq, n_particles, d_hidden]
        V = torch.stack(V_list, dim=2)

        K = K.view(batch, seq_len, n_particles, self.n_heads, self.head_dim)
        V = V.view(batch, seq_len, n_particles, self.n_heads, self.head_dim)

        # Compute attention scores
        Q = Q.unsqueeze(2)  # [batch, seq, 1, heads, dim]
        attn = torch.einsum('bsphd,bskhd->bsphk', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum('bsphk,bskhd->bsphd', attn, V)
        out = out.squeeze(2).reshape(batch, seq_len, d_hidden)

        return self.W_o(out)


class TimeParticleSSM(nn.Module):
    """
    TimeParticle State Space Model for time series forecasting.

    Combines multiple particles operating at different time scales with
    cross-scale attention for comprehensive temporal pattern capture.

    Args:
        n_features: Number of input features
        d_hidden: Hidden dimension
        d_state: State dimension for each particle
        n_particles: Number of particles (time scales)
        n_layers: Number of processing layers
        n_heads: Number of attention heads for cross-scale communication
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_features: int,
        d_hidden: int = 64,
        d_state: int = 32,
        n_particles: int = 4,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_particles = n_particles
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_hidden = d_hidden

        # Input projection
        self.input_proj = nn.Linear(n_features, d_hidden)

        # Create particles at different scales for each layer
        self.particles = nn.ModuleList([
            nn.ModuleList([
                Particle(d_hidden, d_hidden, d_state, scale_factor=k + 1)
                for k in range(n_particles)
            ])
            for _ in range(n_layers)
        ])

        # Cross-scale attention for each layer
        self.cross_attention = nn.ModuleList([
            CrossScaleAttention(d_hidden, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Aggregation layers
        self.aggregation = nn.ModuleList([
            nn.Linear(n_particles * d_hidden, d_hidden)
            for _ in range(n_layers)
        ])

        # Final output layers
        self.norm = nn.LayerNorm(d_hidden)
        self.output_head = nn.Linear(d_hidden, 3)  # Buy, Hold, Sell

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_particles: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through TimeParticle model.

        Args:
            x: Input features [batch, seq_len, n_features]
            return_particles: If True, also return individual particle outputs

        Returns:
            logits: Classification logits [batch, 3]
            particle_outputs: (optional) List of particle outputs
        """
        batch = x.shape[0]
        device = x.device

        # Input projection
        x = self.input_proj(x)

        # Initialize particle states
        states = [
            [torch.zeros(batch, self.d_state, device=device)
             for _ in range(self.n_particles)]
            for _ in range(self.n_layers)
        ]

        # Process through layers
        particle_outputs = []

        for layer_idx in range(self.n_layers):
            layer_outputs = []

            # First pass: evolve particles independently
            for k, particle in enumerate(self.particles[layer_idx]):
                out, states[layer_idx][k] = particle(
                    x, states[layer_idx][k]
                )
                layer_outputs.append(out)

            # Second pass: cross-scale communication
            refined_outputs = []
            for k in range(self.n_particles):
                other_outputs = [
                    layer_outputs[j]
                    for j in range(self.n_particles) if j != k
                ]
                if other_outputs:
                    message = self.cross_attention[layer_idx](
                        layer_outputs[k], other_outputs, other_outputs
                    )
                    refined = layer_outputs[k] + self.dropout(message)
                else:
                    refined = layer_outputs[k]
                refined_outputs.append(refined)

            particle_outputs = refined_outputs

            # Aggregate for next layer input
            concat_features = torch.cat(refined_outputs, dim=-1)
            x = self.aggregation[layer_idx](concat_features)
            x = self.dropout(x)

        # Final aggregation and output
        final_concat = torch.cat(particle_outputs, dim=-1)
        final_features = self.aggregation[-1](final_concat)

        # Use last timestep for classification
        out = self.norm(final_features[:, -1, :])
        logits = self.output_head(out)

        if return_particles:
            return logits, particle_outputs
        return logits

    def get_particle_states(self) -> List[Dict[str, torch.Tensor]]:
        """Get the current state of all particles for analysis."""
        return [
            {
                'scale': k + 1,
                'layer': layer_idx
            }
            for layer_idx in range(self.n_layers)
            for k in range(self.n_particles)
        ]


class TimeParticleTradingModel(nn.Module):
    """
    Complete TimeParticle model for trading applications.

    Wraps the core TimeParticle SSM with training utilities and
    prediction methods suitable for trading strategies.

    Args:
        n_features: Number of input features
        d_hidden: Hidden dimension
        d_state: State dimension
        n_particles: Number of particles
        n_layers: Number of layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_features: int,
        d_hidden: int = 64,
        d_state: int = 32,
        n_particles: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.core = TimeParticleSSM(
            n_features, d_hidden, d_state, n_particles, n_layers, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        logits = self.core(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels (0=Sell, 1=Hold, 2=Buy)."""
        probs = self.predict_proba(x)
        return probs.argmax(dim=-1)

    def get_multiscale_analysis(
        self,
        x: torch.Tensor,
        scale_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get analysis at each time scale.

        Args:
            x: Input features [batch, seq_len, n_features]
            scale_names: Names for each scale

        Returns:
            Dictionary with analysis for each scale
        """
        if scale_names is None:
            scale_names = ['fast', 'medium', 'slow', 'trend']
            scale_names = scale_names[:self.core.n_particles]

        self.eval()
        with torch.no_grad():
            _, particle_outputs = self.core(x, return_particles=True)

        analysis = {}
        for k, output in enumerate(particle_outputs):
            name = scale_names[k] if k < len(scale_names) else f'scale_{k}'

            # Extract trend metrics
            recent = output[0, -10:, :].mean(dim=0)
            older = output[0, -20:-10, :].mean(dim=0) if output.shape[1] >= 20 else recent

            momentum = (recent - older).mean().item()
            volatility = output[0, -20:, :].std().item() if output.shape[1] >= 20 else 0.1

            analysis[name] = {
                'momentum': momentum,
                'volatility': volatility,
                'signal': 'bullish' if momentum > 0 else 'bearish',
                'strength': min(abs(momentum) / max(volatility, 1e-6), 1.0)
            }

        return analysis


def generate_signals(
    model: TimeParticleTradingModel,
    features: torch.Tensor,
    threshold: float = 0.6
) -> List[Tuple[str, float]]:
    """
    Generate trading signals from model predictions.

    Args:
        model: Trained TimeParticle model
        features: Input features [batch, seq_len, n_features]
        threshold: Confidence threshold for buy/sell signals

    Returns:
        List of (signal, confidence) tuples
    """
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(features)

    signals = []
    for prob in probs:
        buy_prob = prob[2].item()
        sell_prob = prob[0].item()
        hold_prob = prob[1].item()

        if buy_prob > threshold:
            signals.append(('BUY', buy_prob))
        elif sell_prob > threshold:
            signals.append(('SELL', sell_prob))
        else:
            signals.append(('HOLD', hold_prob))

    return signals


if __name__ == "__main__":
    # Quick test
    print("Testing TimeParticle SSM...")

    # Create model
    model = TimeParticleTradingModel(
        n_features=20,
        d_hidden=64,
        d_state=32,
        n_particles=4,
        n_layers=3
    )

    # Generate dummy data
    batch_size = 4
    seq_len = 100
    n_features = 20

    x = torch.randn(batch_size, seq_len, n_features)

    # Forward pass
    logits = model(x)
    print(f"Output shape: {logits.shape}")

    # Get predictions
    probs = model.predict_proba(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0].numpy()}")

    # Get multiscale analysis
    analysis = model.get_multiscale_analysis(x)
    print(f"Multiscale analysis: {analysis}")

    # Generate signals
    signals = generate_signals(model, x)
    print(f"Signals: {signals}")

    print("All tests passed!")
