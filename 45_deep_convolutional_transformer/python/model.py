"""
Deep Convolutional Transformer (DCT) Model Implementation

Based on the paper:
"Deep Convolutional Transformer Network for Stock Movement Prediction"
(MDPI Electronics, 2024)

Architecture:
- Inception Convolutional Token Embedding
- Multi-Head Self-Attention
- Separable Feed-Forward Network
- Movement Classification Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from enum import Enum


class MovementClass(Enum):
    """Stock movement classification."""
    UP = 0
    DOWN = 1
    STABLE = 2


@dataclass
class DCTConfig:
    """
    Configuration for Deep Convolutional Transformer model.

    Example:
        config = DCTConfig(
            seq_len=30,
            input_features=10,
            d_model=64
        )
    """
    # Input dimensions
    seq_len: int = 30  # Look-back window (30 days as per paper)
    input_features: int = 5  # OHLCV + technical indicators

    # Model architecture
    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 256
    num_encoder_layers: int = 2
    dropout: float = 0.1

    # Inception module
    inception_channels: int = 64
    kernel_sizes: List[int] = field(default_factory=lambda: [1, 3, 5])

    # Output
    num_classes: int = 3  # Up, Down, Stable
    movement_threshold: float = 0.005  # 0.5% threshold

    # Positional encoding
    use_positional_encoding: bool = True
    max_seq_len: int = 512

    def validate(self):
        """Validate configuration."""
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.dropout >= 0 and self.dropout <= 1, "dropout must be in [0, 1]"
        assert len(self.kernel_sizes) >= 2, "Need at least 2 kernel sizes"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


class InceptionConvEmbedding(nn.Module):
    """
    Inception-style convolutional embedding for time series.

    Uses parallel convolutions with different kernel sizes to capture
    patterns at multiple temporal scales:
    - kernel_size=1: Point-wise features
    - kernel_size=3: Short-term patterns
    - kernel_size=5: Medium-term patterns
    - MaxPool branch: Most prominent features
    """

    def __init__(self, config: DCTConfig):
        super().__init__()

        in_channels = config.input_features
        out_channels = config.inception_channels

        # Number of branches (including max pool)
        num_branches = len(config.kernel_sizes) + 1
        branch_channels = out_channels // num_branches

        # Create parallel convolutional branches
        self.branches = nn.ModuleList()
        for k in config.kernel_sizes:
            padding = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=padding),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU()
                )
            )

        # MaxPool branch
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.GELU()
        )

        # Channel reduction after concatenation
        total_channels = branch_channels * num_branches
        self.reduce = nn.Sequential(
            nn.Conv1d(total_channels, config.d_model, kernel_size=1),
            nn.BatchNorm1d(config.d_model),
            nn.GELU()
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Embedded tensor [batch, seq_len, d_model]
        """
        # Transpose for Conv1d: [batch, features, seq_len]
        x = x.transpose(1, 2)

        # Apply all branches
        branch_outputs = [branch(x) for branch in self.branches]
        branch_outputs.append(self.pool_branch(x))

        # Concatenate along channel dimension
        x = torch.cat(branch_outputs, dim=1)

        # Reduce channels
        x = self.reduce(x)
        x = self.dropout(x)

        # Transpose back: [batch, seq_len, d_model]
        return x.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for temporal dependencies.

    Each attention head can learn to focus on different aspects:
    - Recent time steps (short-term momentum)
    - Periodic patterns (seasonality)
    - Key events (earnings, announcements)
    """

    def __init__(self, config: DCTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention: Optional [batch, num_heads, seq_len, seq_len]
        """
        B, L, D = x.shape

        # Compute Q, K, V in single projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_dropped = self.dropout(attn)

        # Aggregate values
        out = (attn_dropped @ v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)

        if return_attention:
            return out, attn
        return out, None


class SeparableFFN(nn.Module):
    """
    Separable feed-forward network.

    Decomposes the standard FFN into:
    1. Depthwise operation: processes each channel independently
    2. Pointwise operation: mixes information across channels

    Reduces parameters from O(d_model * d_ff) to O(d_model + d_ff)
    """

    def __init__(self, config: DCTConfig):
        super().__init__()

        # Depthwise-style processing (groups=d_model would be true depthwise)
        # Using smaller groups for efficiency
        groups = min(config.d_model, 8)
        self.depthwise = nn.Conv1d(
            config.d_model, config.d_model,
            kernel_size=3, padding=1, groups=groups
        )

        # Pointwise transformation
        self.pointwise = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Apply depthwise conv
        residual = x
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.depthwise(x)
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]

        # Apply pointwise transformation
        x = self.pointwise(x)

        return self.norm(x + residual)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with attention and FFN."""

    def __init__(self, config: DCTConfig):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(config)
        self.ffn = SeparableFFN(config)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention: Optional attention weights
        """
        # Self-attention with residual
        attn_out, attn_weights = self.self_attention(x, return_attention)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual (handled in SeparableFFN)
        x = self.ffn(x)

        return x, attn_weights


class MovementClassifier(nn.Module):
    """
    Classification head for stock movement prediction.

    Output classes:
    - Up: Price increase > threshold
    - Down: Price decrease > threshold
    - Stable: Price change within threshold
    """

    def __init__(self, config: DCTConfig):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, d_model] (after pooling)

        Returns:
            Logits tensor [batch, num_classes]
        """
        return self.classifier(x)


class DCTModel(nn.Module):
    """
    Deep Convolutional Transformer for Stock Movement Prediction.

    Combines:
    - Inception convolutional embedding for multi-scale feature extraction
    - Transformer encoder for capturing temporal dependencies
    - Movement classifier for up/down/stable prediction

    Example:
        config = DCTConfig(seq_len=30, input_features=10)
        model = DCTModel(config)

        x = torch.randn(2, 30, 10)  # [batch, seq_len, features]
        output = model(x)
        print(output['logits'].shape)  # [2, 3]
    """

    def __init__(self, config: DCTConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Embedding layers
        self.inception_embedding = InceptionConvEmbedding(config)

        if config.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                config.d_model, config.max_seq_len, config.dropout
            )
        else:
            self.positional_encoding = None

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.num_encoder_layers)
        ])

        # Global pooling and classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = MovementClassifier(config)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, features]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
                - logits: [batch, num_classes]
                - probabilities: [batch, num_classes]
                - predictions: [batch] class indices
                - attention: Optional dict of attention weights
        """
        # Inception embedding
        x = self.inception_embedding(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        # Transformer encoder
        all_attention = {}
        for i, layer in enumerate(self.encoder_layers):
            x, attn = layer(x, return_attention)
            if attn is not None:
                all_attention[f'layer_{i}'] = attn

        # Global average pooling
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch, d_model]

        # Classification
        logits = self.classifier(x)  # [batch, num_classes]
        probabilities = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        result = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions
        }

        if return_attention:
            result['attention'] = all_attention

        return result

    def predict_movement(self, x: torch.Tensor) -> Dict[str, any]:
        """
        Predict stock movement with class names.

        Args:
            x: Input tensor [batch, seq_len, features]

        Returns:
            Dictionary with predictions and class names
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

            class_names = ['Up', 'Down', 'Stable']
            predictions = [class_names[p.item()] for p in output['predictions']]

            return {
                'predictions': predictions,
                'probabilities': output['probabilities'].cpu().numpy(),
                'class_names': class_names
            }


def create_movement_labels(
    prices: torch.Tensor,
    threshold: float = 0.005,
    horizon: int = 1
) -> torch.Tensor:
    """
    Create movement labels from price series.

    Args:
        prices: Price tensor [batch, seq_len] or [seq_len]
        threshold: Movement threshold (default 0.5%)
        horizon: Prediction horizon

    Returns:
        Labels tensor: 0=Up, 1=Down, 2=Stable
    """
    if prices.dim() == 1:
        prices = prices.unsqueeze(0)

    # Calculate returns
    returns = (prices[:, horizon:] - prices[:, :-horizon]) / prices[:, :-horizon]

    labels = torch.full_like(returns, 2, dtype=torch.long)  # Stable by default
    labels[returns > threshold] = 0  # Up
    labels[returns < -threshold] = 1  # Down

    return labels


if __name__ == "__main__":
    # Test the model
    print("Testing DCT model...")

    config = DCTConfig(
        seq_len=30,
        input_features=10,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2
    )

    model = DCTModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 30, 10)
    output = model(x, return_attention=True)

    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probabilities shape: {output['probabilities'].shape}")
    print(f"Predictions: {output['predictions']}")
    print(f"Attention available: {output['attention'] is not None}")

    # Test prediction
    pred = model.predict_movement(x)
    print(f"Predicted movements: {pred['predictions']}")
    print(f"Probabilities:\n{pred['probabilities']}")
