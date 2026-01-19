#!/usr/bin/env python3
"""
Basic Positional Encoding Example

This example demonstrates the fundamental positional encoding types:
sinusoidal, learned, relative, and rotary (RoPE).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RelativePositionalEncoding,
    RotaryPositionalEncoding,
)


def main():
    print("Basic Positional Encoding Example")
    print("=" * 50)

    d_model = 64
    max_len = 100
    batch_size = 2
    seq_len = 10

    # 1. Sinusoidal Encoding
    print("\n1. Sinusoidal Positional Encoding")
    print("-" * 40)

    sinusoidal = SinusoidalPositionalEncoding(d_model, max_len)
    x = torch.zeros(batch_size, seq_len, d_model)
    encoded = sinusoidal(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {encoded.shape}")
    print(f"Position 0, dims 0-3: {encoded[0, 0, :4].numpy().round(4)}")
    print(f"Position 1, dims 0-3: {encoded[0, 1, :4].numpy().round(4)}")
    print("Note: At position 0, sin(0)=0 and cos(0)=1 pattern")

    # 2. Learned Encoding
    print("\n2. Learned Positional Encoding")
    print("-" * 40)

    learned = LearnedPositionalEncoding(d_model, max_len)
    encoded = learned(x)

    print(f"Trainable parameters: {sum(p.numel() for p in learned.parameters())}")
    print(f"Output shape: {encoded.shape}")
    print(f"Embedding norm at pos 0: {torch.norm(learned.embedding.weight[0]).item():.4f}")
    print("Note: These would be trained during model training")

    # 3. Relative Encoding
    print("\n3. Relative Positional Encoding")
    print("-" * 40)

    n_heads = 4
    head_dim = d_model // n_heads
    relative = RelativePositionalEncoding(d_model, n_heads, max_relative_position=50)

    # Create Q, K, V tensors [batch, n_heads, seq_len, head_dim]
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim)

    # Apply relative attention
    output = relative(q, k, v)

    print(f"Q/K/V shape: [batch={batch_size}, heads={n_heads}, seq={seq_len}, head_dim={head_dim}]")
    print(f"Output shape: {output.shape}")
    print(f"Max relative position: 50")
    print("Note: Modifies attention scores based on relative positions")

    # 4. Rotary Encoding (RoPE)
    print("\n4. Rotary Positional Encoding (RoPE)")
    print("-" * 40)

    # RoPE expects [batch, n_heads, seq_len, head_dim]
    rope = RotaryPositionalEncoding(d_model, n_heads, max_len=max_len)
    q_rope = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k_rope = torch.randn(batch_size, n_heads, seq_len, head_dim)

    q_rot, k_rot = rope(q_rope, k_rope)

    print(f"Query/Key shape: [batch={batch_size}, heads={n_heads}, seq={seq_len}, head_dim={head_dim}]")
    print(f"Rotated Q/K shape: {q_rot.shape}")

    # Verify norm preservation
    q_norm_before = torch.norm(q_rope, dim=-1).mean()
    q_norm_after = torch.norm(q_rot, dim=-1).mean()
    print(f"Query norm before rotation: {q_norm_before:.4f}")
    print(f"Query norm after rotation: {q_norm_after:.4f}")
    print("Note: RoPE preserves vector magnitude (rotation property)")

    # 5. Comparison
    print("\n5. Encoding Comparison")
    print("-" * 40)
    print("| Encoding   | Learnable | Relative | Extrapolation |")
    print("|------------|-----------|----------|---------------|")
    print("| Sinusoidal | No        | No       | Good          |")
    print("| Learned    | Yes       | No       | Poor          |")
    print("| Relative   | No        | Yes      | Good          |")
    print("| RoPE       | No        | Yes      | Excellent     |")

    print("\nRecommendations:")
    print("- Short fixed sequences: Sinusoidal or Learned")
    print("- Variable length: Relative")
    print("- Long sequences: RoPE")
    print("- Trading/Time series: RoPE + Calendar encoding")


if __name__ == "__main__":
    main()
