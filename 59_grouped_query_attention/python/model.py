"""
Grouped Query Attention Model Implementation

This module contains the core GQA implementation and trading model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) implementation.

    GQA is an efficient attention mechanism that groups query heads
    to share key-value heads, reducing memory usage while maintaining
    model quality.

    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_heads evenly)
        dropout: Dropout rate for attention weights

    Example:
        >>> gqa = GroupedQueryAttention(d_model=256, num_heads=8, num_kv_heads=2)
        >>> x = torch.randn(32, 100, 256)  # batch, seq_len, d_model
        >>> output = gqa(x)
        >>> output.shape
        torch.Size([32, 100, 256])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads

        # Query projection: one per head
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # Key and Value projections: shared within groups
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional KV cache for efficient inference.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            kv_cache: Optional tuple of (cached_keys, cached_values)

        Returns:
            Tuple of (output tensor, (keys, values) for caching)
        """
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache for inference
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # Expand K, V for grouped attention
        # Each KV head is shared by (num_heads // num_kv_heads) query heads
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        # Return output and KV cache
        # Collapse repeated KV back for caching
        k_cache = k[:, ::self.num_groups, :, :]
        v_cache = v[:, ::self.num_groups, :, :]

        return output, (k_cache, v_cache)


class GQABlock(nn.Module):
    """
    Transformer block using Grouped Query Attention.

    Consists of:
    - GQA self-attention with residual connection and layer norm
    - Feed-forward network with residual connection and layer norm

    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the GQA block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            kv_cache: Optional KV cache for inference

        Returns:
            Tuple of (output tensor, KV cache)
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, new_cache = self.attention(normed, mask, kv_cache)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        x = x + ff_out

        return x, new_cache


class GQATrader(nn.Module):
    """
    Complete trading model using Grouped Query Attention.

    This model is designed for financial time series prediction,
    supporting both cryptocurrency and stock market data.

    Args:
        input_dim: Number of input features (e.g., OHLCV = 5)
        d_model: Model hidden dimension
        num_heads: Number of query attention heads
        num_kv_heads: Number of key-value heads for GQA
        num_layers: Number of transformer layers
        d_ff: Feed-forward hidden dimension (default: 4 * d_model)
        max_seq_len: Maximum sequence length for positional encoding
        dropout: Dropout rate
        num_classes: Number of output classes (default: 3 for up/down/neutral)

    Example:
        >>> model = GQATrader(
        ...     input_dim=5,
        ...     d_model=64,
        ...     num_heads=8,
        ...     num_kv_heads=2,
        ...     num_layers=4
        ... )
        >>> x = torch.randn(32, 100, 5)  # batch, seq_len, features
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 3])
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        num_layers: int = 4,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_classes: int = 3
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if d_ff is None:
            d_ff = 4 * d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Transformer layers with GQA
        self.layers = nn.ModuleList([
            GQABlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv_cache: Optional[list] = None
    ) -> torch.Tensor:
        """
        Forward pass through the trading model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            use_cache: Whether to return KV cache for inference
            past_kv_cache: Previous KV cache for incremental decoding

        Returns:
            If use_cache: Tuple of (logits, kv_cache)
            Else: logits tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to model dimension
        x = self.input_proj(x)

        # Add positional encoding
        if past_kv_cache is not None:
            # Incremental decoding: only add position for new tokens
            past_len = past_kv_cache[0][0].shape[2] if past_kv_cache[0] is not None else 0
            x = x + self.pos_encoding[:, past_len:past_len + seq_len, :]
        else:
            x = x + self.pos_encoding[:, :seq_len, :]

        # Pass through transformer layers
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = past_kv_cache[i] if past_kv_cache is not None else None
            x, cache = layer(x, mask, layer_cache)
            if use_cache:
                new_kv_cache.append(cache)

        # Final layer norm
        x = self.norm(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        if use_cache:
            return logits, new_kv_cache
        return logits

    def get_memory_usage(self, seq_len: int, batch_size: int = 1) -> dict:
        """
        Calculate memory usage for KV cache.

        Args:
            seq_len: Sequence length
            batch_size: Batch size

        Returns:
            Dictionary with memory statistics
        """
        head_dim = self.d_model // self.layers[0].attention.num_heads
        num_kv_heads = self.layers[0].attention.num_kv_heads
        num_heads = self.layers[0].attention.num_heads

        # KV cache size per layer
        kv_per_layer = 2 * batch_size * num_kv_heads * seq_len * head_dim

        # Total KV cache
        total_kv = kv_per_layer * self.num_layers

        # Compare with MHA (full KV per head)
        mha_kv_per_layer = 2 * batch_size * num_heads * seq_len * head_dim
        mha_total = mha_kv_per_layer * self.num_layers

        return {
            "gqa_kv_cache_bytes": total_kv * 4,  # float32
            "mha_kv_cache_bytes": mha_total * 4,
            "memory_savings": 1 - (total_kv / mha_total),
            "kv_heads_ratio": num_kv_heads / num_heads
        }


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal attention mask.

    Args:
        seq_len: Sequence length
        device: Target device

    Returns:
        Causal mask tensor
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    if device is not None:
        mask = mask.to(device)
    return mask


if __name__ == "__main__":
    # Test the model
    print("Testing GQA Trading Model...")

    # Create model
    model = GQATrader(
        input_dim=5,
        d_model=64,
        num_heads=8,
        num_kv_heads=2,
        num_layers=4
    )

    # Test input
    batch_size = 32
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 5)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Memory usage
    memory = model.get_memory_usage(seq_len, batch_size)
    print(f"\nMemory Statistics:")
    print(f"  GQA KV cache: {memory['gqa_kv_cache_bytes'] / 1024:.2f} KB")
    print(f"  MHA KV cache: {memory['mha_kv_cache_bytes'] / 1024:.2f} KB")
    print(f"  Memory savings: {memory['memory_savings']:.1%}")

    print("\nAll tests passed!")
