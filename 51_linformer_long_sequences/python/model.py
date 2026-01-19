"""
Linformer: Self-Attention with Linear Complexity

Implementation of the Linformer architecture for efficient processing
of long financial time series sequences.

Reference:
    Wang et al., "Linformer: Self-Attention with Linear Complexity"
    https://arxiv.org/abs/2006.04768
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LinformerAttention(nn.Module):
    """
    Linformer Self-Attention with Linear Complexity.

    Projects keys and values to a lower dimension k using learned
    projection matrices E and F, reducing complexity from O(n²) to O(n×k).

    The key insight is that self-attention matrices are often low-rank,
    meaning the essential information can be captured in a much smaller
    dimensional space.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        seq_len: Fixed sequence length (required for projection matrices)
        k: Projection dimension (k << seq_len for efficiency)
        dropout: Dropout rate
        share_kv: If True, share projection matrix between K and V

    Example:
        >>> attention = LinformerAttention(d_model=256, n_heads=8, seq_len=2048, k=128)
        >>> x = torch.randn(32, 2048, 256)
        >>> output, _ = attention(x)
        >>> output.shape
        torch.Size([32, 2048, 256])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        k: int = 128,
        dropout: float = 0.1,
        share_kv: bool = True
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len
        self.k = k
        self.scale = math.sqrt(self.d_k)
        self.share_kv = share_kv

        # Query, Key, Value linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Linear projection matrices E and F
        # E: [n_heads, k, seq_len] projects keys from seq_len to k
        # F: [n_heads, k, seq_len] projects values from seq_len to k
        self.E = nn.Parameter(torch.randn(n_heads, k, seq_len) * 0.02)

        if share_kv:
            # Share projection between K and V (more parameter efficient)
            self.F = self.E
        else:
            # Separate projections for K and V (potentially more expressive)
            self.F = nn.Parameter(torch.randn(n_heads, k, seq_len) * 0.02)

        self.dropout = nn.Dropout(dropout)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with linear complexity attention.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            mask: Optional attention mask of shape [batch, seq_len] or [batch, 1, seq_len]
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights):
                - output: [batch, seq_len, d_model]
                - attention_weights: [batch, n_heads, seq_len, k] if return_attention else None

        Complexity:
            Standard attention: O(n² × d)
            Linformer attention: O(n × k × d) where k << n
        """
        batch_size, seq_len, _ = x.shape

        # Handle variable sequence lengths (pad to max or truncate)
        if seq_len != self.seq_len:
            if seq_len < self.seq_len:
                # Pad shorter sequences
                padding = torch.zeros(
                    batch_size, self.seq_len - seq_len, self.d_model,
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=1)
                seq_len = self.seq_len
            else:
                # Truncate longer sequences
                x = x[:, :self.seq_len, :]
                seq_len = self.seq_len

        # Linear projections for Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)  # [batch, seq_len, d_model]
        V = self.W_v(x)  # [batch, seq_len, d_model]

        # Reshape for multi-head attention
        # [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Project K and V to lower dimension using E and F matrices
        # This is the key step that reduces complexity from O(n²) to O(n×k)
        # E @ K: [n_heads, k, seq_len] × [batch, n_heads, seq_len, d_k]
        #        → [batch, n_heads, k, d_k]
        K_proj = torch.einsum('hkn,bhnd->bhkd', self.E, K)
        V_proj = torch.einsum('hkn,bhnd->bhkd', self.F, V)

        # Compute attention scores
        # Q @ K_proj.T: [batch, n_heads, seq_len, d_k] × [batch, n_heads, d_k, k]
        #              → [batch, n_heads, seq_len, k]
        attention_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            # Create projected mask
            # We need to handle masking differently since K is projected
            # For simplicity, we mask the attention scores directly
            attention_scores = attention_scores.masked_fill(
                mask[:, :, :, :self.k] == 0, -1e9
            )

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to projected values
        # [batch, n_heads, seq_len, k] × [batch, n_heads, k, d_k]
        # → [batch, n_heads, seq_len, d_k]
        context = torch.matmul(attention_weights, V_proj)

        # Reshape back to [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # Output projection
        output = self.W_o(context)

        if return_attention:
            return output, attention_weights
        return output, None


class LinformerEncoderLayer(nn.Module):
    """
    Linformer encoder layer with linear attention, feed-forward network,
    and residual connections.

    Architecture:
        x → LayerNorm → LinformerAttention → Dropout → + → LayerNorm → FFN → Dropout → +
           ↑─────────────────────────────────────────────↑─────────────────────────────↑

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        seq_len: Fixed sequence length
        k: Projection dimension for linear attention
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        k: int = 128,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = LinformerAttention(
            d_model=d_model,
            n_heads=n_heads,
            seq_len=seq_len,
            k=k,
            dropout=dropout
        )

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization (Pre-LN transformer style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual connection (Pre-LN)
        attn_out, _ = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual connection (Pre-LN)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out

        return x


class Linformer(nn.Module):
    """
    Complete Linformer model for financial time series forecasting.

    This model efficiently handles long sequences (2000+ timesteps) with
    O(n) complexity instead of O(n²), making it ideal for analyzing
    extended historical data in trading applications.

    Args:
        n_features: Number of input features per timestep
        seq_len: Fixed sequence length (should match training data)
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        k: Projection dimension for linear attention (k << seq_len)
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        output_type: One of 'regression', 'classification', or 'allocation'
        n_outputs: Number of output values

    Example:
        >>> model = Linformer(
        ...     n_features=10,
        ...     seq_len=2048,
        ...     d_model=256,
        ...     n_heads=8,
        ...     n_layers=4,
        ...     k=128,
        ...     output_type='regression',
        ...     n_outputs=1
        ... )
        >>> x = torch.randn(32, 2048, 10)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 1])
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        k: int = 128,
        d_ff: int = 1024,
        dropout: float = 0.1,
        output_type: str = 'regression',
        n_outputs: int = 1
    ):
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.k = k
        self.output_type = output_type
        self.n_outputs = n_outputs

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(seq_len, d_model)

        # Linformer encoder layers
        self.encoder_layers = nn.ModuleList([
            LinformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                seq_len=seq_len,
                k=k,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # Output head based on task type
        if output_type == 'regression':
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_outputs)
            )
            self.loss_fn = nn.MSELoss()

        elif output_type == 'classification':
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_outputs)
            )
            self.loss_fn = nn.BCEWithLogitsLoss()

        elif output_type == 'allocation':
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_outputs),
                nn.Tanh()  # Bound allocations to [-1, 1]
            )
            self.loss_fn = None  # Custom loss for allocation

        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    def _create_positional_encoding(
        self,
        seq_len: int,
        d_model: int
    ) -> nn.Parameter:
        """
        Create sinusoidal positional encoding.

        Uses the formula from "Attention Is All You Need":
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Linformer model.

        Args:
            x: Input tensor of shape [batch, seq_len, n_features]
            mask: Optional attention mask

        Returns:
            Predictions of shape [batch, n_outputs]
        """
        batch_size, seq_len, _ = x.shape

        # Input projection to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        if seq_len <= self.seq_len:
            x = x + self.positional_encoding[:, :seq_len, :]
        else:
            # For longer sequences, we need to extrapolate positional encoding
            x = x + self.positional_encoding[:, :self.seq_len, :].repeat(
                1, (seq_len + self.seq_len - 1) // self.seq_len, 1
            )[:, :seq_len, :]

        x = self.dropout(x)

        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)

        # Final layer norm
        x = self.norm(x)

        # Use last token for prediction (common for classification/regression)
        x = x[:, -1, :]

        # Output head
        output = self.output_head(x)

        return output

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss based on output type.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Loss tensor
        """
        if self.output_type == 'regression':
            return self.loss_fn(predictions, targets)

        elif self.output_type == 'classification':
            return self.loss_fn(predictions, targets)

        elif self.output_type == 'allocation':
            # For allocation, maximize returns: negative mean of (allocation * returns)
            return -torch.mean(predictions * targets)

        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")

    def get_complexity_info(self) -> dict:
        """
        Get computational complexity information.

        Returns:
            Dictionary with complexity metrics
        """
        n = self.seq_len
        k = self.k
        d = self.d_model

        standard_attention_ops = n * n * d
        linformer_attention_ops = n * k * d

        return {
            'seq_len': n,
            'projection_dim': k,
            'd_model': d,
            'n_layers': self.n_layers,
            'standard_attention_ops': standard_attention_ops,
            'linformer_attention_ops': linformer_attention_ops,
            'speedup_factor': standard_attention_ops / linformer_attention_ops,
            'memory_reduction': (n - k) / n * 100
        }


def create_linformer(
    n_features: int,
    seq_len: int = 2048,
    config: str = 'base'
) -> Linformer:
    """
    Factory function to create Linformer models with predefined configurations.

    Args:
        n_features: Number of input features
        seq_len: Sequence length
        config: Configuration name ('small', 'base', 'large')

    Returns:
        Configured Linformer model
    """
    configs = {
        'small': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'k': 64,
            'd_ff': 512,
            'dropout': 0.1
        },
        'base': {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'k': 128,
            'd_ff': 1024,
            'dropout': 0.1
        },
        'large': {
            'd_model': 512,
            'n_heads': 16,
            'n_layers': 6,
            'k': 256,
            'd_ff': 2048,
            'dropout': 0.1
        }
    }

    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Choose from {list(configs.keys())}")

    return Linformer(
        n_features=n_features,
        seq_len=seq_len,
        **configs[config]
    )
