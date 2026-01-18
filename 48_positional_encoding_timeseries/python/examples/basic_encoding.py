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
    print(f"Embedding norm at pos 0: {torch.norm(learned.position_embeddings.weight[0]).item():.4f}")
    print("Note: These would be trained during model training")

    # 3. Relative Encoding
    print("\n3. Relative Positional Encoding")
    print("-" * 40)

    relative = RelativePositionalEncoding(d_model, max_distance=50)

    # Compute relative position matrix
    pos = torch.arange(seq_len)
    rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)  # (seq_len, seq_len)

    # Get embeddings
    rel_emb = relative.get_embedding(rel_pos)

    print(f"Relative position range: [{rel_pos.min().item()}, {rel_pos.max().item()}]")
    print(f"Embedding shape: {rel_emb.shape}")
    print(f"Same distance embeddings have similar norms:")
    print(f"  Distance +3: norm = {torch.norm(relative.get_embedding(torch.tensor([[3]]))).item():.4f}")
    print(f"  Distance -3: norm = {torch.norm(relative.get_embedding(torch.tensor([[-3]]))).item():.4f}")

    # 4. Rotary Encoding (RoPE)
    print("\n4. Rotary Positional Encoding (RoPE)")
    print("-" * 40)

    rope = RotaryPositionalEncoding(d_model, max_len)
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)

    q_rot, k_rot = rope(q, k)

    print(f"Query/Key shape: {q.shape}")
    print(f"Rotated Q/K shape: {q_rot.shape}")

    # Verify norm preservation
    q_norm_before = torch.norm(q, dim=-1).mean()
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
