"""
Deep Convolutional Transformer (DCT) for Stock Movement Prediction

Provides:
- DCTConfig: Model configuration
- DCTModel: Main DCT model
- InceptionConvEmbedding: Multi-scale convolutional embedding
- MultiHeadSelfAttention: Transformer attention mechanism
"""

from .model import (
    DCTConfig,
    DCTModel,
    InceptionConvEmbedding,
    MultiHeadSelfAttention,
    SeparableFFN,
    MovementClassifier,
)

__all__ = [
    "DCTConfig",
    "DCTModel",
    "InceptionConvEmbedding",
    "MultiHeadSelfAttention",
    "SeparableFFN",
    "MovementClassifier",
]
