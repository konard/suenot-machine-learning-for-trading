"""
FNet Model Implementation

This module implements the FNet architecture which replaces self-attention
with Fourier Transform for O(n log n) computational complexity.

Reference: "FNet: Mixing Tokens with Fourier Transforms" (2021)
https://arxiv.org/abs/2105.03824
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class FourierLayer(nn.Module):
    """
    Fourier Transform layer that replaces self-attention.

    Uses 2D FFT to mix tokens across both sequence and hidden dimensions.
    This layer has NO learnable parameters, making it extremely efficient.

    The Fourier transform provides:
    1. Global mixing - each output contains information from all inputs
    2. O(n log n) complexity vs O(n²) for attention
    3. Natural handling of periodic patterns in financial data
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D FFT and return real part.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Fourier-transformed tensor (real part only) [batch, seq_len, d_model]
        """
        # Convert to float for FFT (handles mixed precision training)
        x_float = x.float()

        # Apply 2D FFT
        # dim=-2: sequence dimension (temporal mixing)
        # dim=-1: hidden dimension (feature mixing)
        x_fft = torch.fft.fft2(x_float)

        # Return real part only
        # Imaginary part could be used but adds complexity
        return x_fft.real.type_as(x)


class FNetEncoderBlock(nn.Module):
    """
    Single FNet encoder block.

    Architecture:
    1. Fourier Layer → Residual → LayerNorm
    2. Feed-Forward → Residual → LayerNorm

    This is analogous to a Transformer encoder block but with
    Fourier Transform replacing self-attention.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension (hidden size)
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()

        # Fourier layer (no parameters)
        self.fourier = FourierLayer()
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU activation as in original paper
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_frequencies: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through encoder block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_frequencies: Whether to return Fourier output for analysis

        Returns:
            output: Transformed tensor [batch, seq_len, d_model]
            fourier_out: Optional Fourier output for frequency analysis
        """
        # Fourier sublayer with residual connection
        fourier_out = self.fourier(x)
        x = self.norm1(x + self.dropout(fourier_out))

        # Feed-forward sublayer with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        if return_frequencies:
            return x, fourier_out
        return x, None


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.

    Uses sine and cosine functions of different frequencies to encode
    position information, allowing the model to understand sequence order.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FNet(nn.Module):
    """
    Complete FNet model for financial time series prediction.

    This model replaces Transformer's self-attention with Fourier Transform,
    achieving significant speedups while maintaining competitive accuracy.

    Features:
    - O(n log n) complexity instead of O(n²)
    - 80% faster training on GPU
    - No attention matrices to store
    - Natural handling of periodic patterns
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        output_dim: int = 1,
        output_type: str = 'regression'
    ):
        """
        Args:
            n_features: Number of input features per time step
            d_model: Model dimension
            n_layers: Number of encoder blocks
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            output_dim: Output dimension
            output_type: 'regression', 'classification', or 'allocation'
        """
        super().__init__()

        self.d_model = d_model
        self.output_type = output_type

        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # FNet encoder blocks
        self.encoder_blocks = nn.ModuleList([
            FNetEncoderBlock(d_model, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output head depends on task type
        self.output_head = self._create_output_head(d_model, output_dim, dropout)

    def _create_output_head(
        self,
        d_model: int,
        output_dim: int,
        dropout: float
    ) -> nn.Module:
        """Create appropriate output head based on task type."""
        if self.output_type == 'regression':
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim)
            )
        elif self.output_type == 'classification':
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim),
                nn.Sigmoid()
            )
        elif self.output_type == 'allocation':
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim),
                nn.Tanh()  # Output bounded to [-1, 1]
            )
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")

    def forward(
        self,
        x: torch.Tensor,
        return_frequencies: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through FNet.

        Args:
            x: Input tensor [batch, seq_len, n_features]
            return_frequencies: Whether to return frequency maps for analysis

        Returns:
            output: Predictions [batch, output_dim]
            freq_maps: Optional list of frequency maps from each layer
        """
        batch_size, seq_len, _ = x.shape

        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply FNet encoder blocks
        freq_maps = []
        for block in self.encoder_blocks:
            x, freq = block(x, return_frequencies=return_frequencies)
            if freq is not None:
                freq_maps.append(freq)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [batch, d_model]

        # Generate predictions
        output = self.output_head(x)

        if return_frequencies:
            return output, freq_maps
        return output

    def get_frequency_analysis(
        self,
        x: torch.Tensor
    ) -> dict:
        """
        Analyze frequency components in the model's representations.

        Args:
            x: Input tensor [batch, seq_len, n_features]

        Returns:
            Dictionary with frequency analysis results
        """
        self.eval()
        with torch.no_grad():
            _, freq_maps = self.forward(x, return_frequencies=True)

        analysis = {}
        for i, freq_map in enumerate(freq_maps):
            # Get magnitude spectrum
            magnitude = torch.abs(freq_map)

            # Average over batch
            avg_magnitude = magnitude.mean(dim=0)

            # Find dominant frequencies
            flat_mag = avg_magnitude.flatten()
            top_k_values, top_k_indices = torch.topk(flat_mag, k=min(10, len(flat_mag)))

            analysis[f"layer_{i+1}"] = {
                "mean_magnitude": avg_magnitude.mean().item(),
                "max_magnitude": avg_magnitude.max().item(),
                "top_frequencies": top_k_indices.tolist(),
                "top_magnitudes": top_k_values.tolist()
            }

        return analysis


class MultiFNet(nn.Module):
    """
    Multi-asset FNet for portfolio prediction.

    Extends FNet to handle multiple assets simultaneously, with:
    1. Temporal FNet: Captures patterns within each asset
    2. Cross-Asset FNet: Captures relationships between assets
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        """
        Args:
            n_assets: Number of assets to predict
            n_features: Features per asset per time step
            d_model: Model dimension
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.n_assets = n_assets
        self.d_model = d_model

        # Per-asset input embeddings
        self.asset_embeddings = nn.ModuleList([
            nn.Linear(n_features, d_model)
            for _ in range(n_assets)
        ])

        # Shared positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Temporal FNet encoder (within each asset)
        self.temporal_encoder = nn.ModuleList([
            FNetEncoderBlock(d_model, d_ff, dropout)
            for _ in range(n_layers // 2)
        ])

        # Cross-asset FNet encoder (between assets)
        self.cross_asset_encoder = nn.ModuleList([
            FNetEncoderBlock(d_model, d_ff, dropout)
            for _ in range(n_layers // 2)
        ])

        # Per-asset prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
            for _ in range(n_assets)
        ])

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for multi-asset prediction.

        Args:
            x: Input tensor [batch, seq_len, n_assets, n_features]

        Returns:
            predictions: Predictions for each asset [batch, n_assets]
        """
        batch_size, seq_len, n_assets, _ = x.shape

        # Embed each asset separately
        asset_features = []
        for i in range(self.n_assets):
            asset_x = self.asset_embeddings[i](x[:, :, i, :])
            asset_x = self.positional_encoding(asset_x)
            asset_features.append(asset_x)

        # Stack: [batch, seq_len, n_assets, d_model]
        x = torch.stack(asset_features, dim=2)

        # Apply temporal encoder to each asset
        for block in self.temporal_encoder:
            temporal_outputs = []
            for i in range(self.n_assets):
                out, _ = block(x[:, :, i, :])
                temporal_outputs.append(out)
            x = torch.stack(temporal_outputs, dim=2)

        # Apply cross-asset encoder
        # Reshape for cross-asset attention: [batch * seq_len, n_assets, d_model]
        x_reshaped = x.view(batch_size * seq_len, n_assets, -1)

        for block in self.cross_asset_encoder:
            x_reshaped, _ = block(x_reshaped)

        # Reshape back: [batch, seq_len, n_assets, d_model]
        x = x_reshaped.view(batch_size, seq_len, n_assets, -1)

        # Global average pooling over time
        x = x.mean(dim=1)  # [batch, n_assets, d_model]

        # Generate predictions for each asset
        predictions = []
        for i in range(self.n_assets):
            pred = self.prediction_heads[i](x[:, i, :])
            predictions.append(pred)

        return torch.cat(predictions, dim=1)


class HybridFNetTransformer(nn.Module):
    """
    Hybrid model combining FNet and Transformer layers.

    Uses FNet for early layers (efficiency) and Transformer for
    final layers (accuracy on complex patterns).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_fnet_layers: int = 2,
        n_transformer_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        output_dim: int = 1
    ):
        super().__init__()

        self.input_projection = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # FNet layers (fast, efficient)
        self.fnet_blocks = nn.ModuleList([
            FNetEncoderBlock(d_model, d_ff, dropout)
            for _ in range(n_fnet_layers)
        ])

        # Transformer layers (accurate, detailed)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid model."""
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # FNet layers for efficient global mixing
        for block in self.fnet_blocks:
            x, _ = block(x)

        # Transformer layers for detailed pattern learning
        x = self.transformer_encoder(x)

        # Global pooling and output
        x = x.mean(dim=1)
        return self.output_head(x)


if __name__ == "__main__":
    # Test FNet model
    print("Testing FNet model...")

    batch_size = 32
    seq_len = 168
    n_features = 7

    model = FNet(
        n_features=n_features,
        d_model=256,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=512,
        output_dim=1
    )

    # Create dummy input
    x = torch.randn(batch_size, seq_len, n_features)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test with frequency analysis
    output, freq_maps = model(x, return_frequencies=True)
    print(f"Number of frequency maps: {len(freq_maps)}")

    # Test frequency analysis
    analysis = model.get_frequency_analysis(x[:4])
    print("\nFrequency analysis (first layer):")
    print(analysis["layer_1"])

    # Test MultiFNet
    print("\n\nTesting MultiFNet model...")
    multi_model = MultiFNet(
        n_assets=5,
        n_features=n_features,
        d_model=256
    )

    x_multi = torch.randn(batch_size, seq_len, 5, n_features)
    output_multi = multi_model(x_multi)
    print(f"Multi-asset input shape: {x_multi.shape}")
    print(f"Multi-asset output shape: {output_multi.shape}")

    print("\nAll tests passed!")
