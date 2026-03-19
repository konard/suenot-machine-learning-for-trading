"""
Cross-Modal Contrastive Learning (CLIP-style) for Financial Markets.

This module implements a dual-encoder architecture that aligns price time-series
and textual news/events into a shared latent space, enabling zero-shot retrieval
and semantic understanding of market dynamics.

Supports both stock market and cryptocurrency (Bybit) data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeSeriesEncoder(nn.Module):
    """
    1D-CNN Encoder for Financial Price Windows.
    Input:  (B, C, SeqLen) where C = number of channels (e.g. 1 for close price,
            5 for OHLCV).
    Output: (B, projection_dim)
    """

    def __init__(self, in_channels=1, hidden_dim=64, projection_dim=128):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Projection Head specific to the Time-Series modality
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        h = self.conv_block(x)
        h = self.adaptive_pool(h).squeeze(-1)
        v = self.projector(h)
        return v


class TextEncoder(nn.Module):
    """
    Simple Token Embedding + Mean Pooling Encoder for Text inputs.
    Input:  (B, SeqLen) — token indices
    Output: (B, projection_dim)
    """

    def __init__(self, vocab_size=5000, embed_dim=64, projection_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Projection Head specific to the Text modality
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
        )

    def forward(self, x):
        # x is a batch of token indices: (B, SeqLen)
        embeds = self.embedding(x)

        # Mean pooling across the token sequence length, ignoring padding (0)
        mask = (x != 0).float().unsqueeze(-1)
        sum_embeds = (embeds * mask).sum(dim=1)
        valid_tokens = mask.sum(dim=1).clamp(min=1.0)
        h = sum_embeds / valid_tokens

        # LayerNorm stabilizes text embeddings before projection
        h = F.layer_norm(h, h.shape[1:])

        v = self.projector(h)
        return v


class CrossModalCLIPModel(nn.Module):
    """
    Dual-encoder CLIP-style model for aligning price charts with text events.
    Holds both the TimeSeriesEncoder and TextEncoder with a learnable
    temperature parameter for the contrastive loss.
    """

    def __init__(self, vocab_size=5000, projection_dim=32, in_channels=1):
        super().__init__()
        self.ts_encoder = TimeSeriesEncoder(
            in_channels=in_channels, projection_dim=projection_dim
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, projection_dim=projection_dim
        )

        # Learnable temperature parameter (as in original CLIP)
        # Initiated with log(1 / 0.07) ≈ 2.6592
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, x_price, x_text):
        v_price = self.ts_encoder(x_price)
        v_text = self.text_encoder(x_text)
        return v_price, v_text


class CLIPLoss(nn.Module):
    """
    Contrastive Loss using CosineEmbeddingLoss.
    For each (Price, Text) positive pair, we also pass a (Price, Negative_Text) pair.
    """

    def __init__(self, margin=0.2):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, v_price, v_text, v_text_neg):
        # Normalize features for cosine similarity in [-1, 1]
        v_price = F.normalize(v_price, p=2, dim=1)
        v_text = F.normalize(v_text, p=2, dim=1)
        v_text_neg = F.normalize(v_text_neg, p=2, dim=1)

        # Targets: 1 for positive pairs, -1 for negative pairs
        target_pos = torch.ones(v_price.size(0), device=v_price.device)
        target_neg = -torch.ones(v_price.size(0), device=v_price.device)

        # Pull positives together
        loss_pos = self.loss_fn(v_price, v_text, target_pos)
        # Push negatives apart
        loss_neg = self.loss_fn(v_price, v_text_neg, target_neg)

        return loss_pos + loss_neg


class SymmetricCLIPLoss(nn.Module):
    """
    Symmetric InfoNCE Loss (full CLIP Loss).
    Uses the NxN similarity matrix and cross-entropy in both directions:
      - Price -> Text (which text matches each chart?)
      - Text -> Price (which chart matches each text?)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, v_price, v_text, logit_scale=None):
        # L2 normalize
        v_price = F.normalize(v_price, p=2, dim=1)
        v_text = F.normalize(v_text, p=2, dim=1)

        # Scale by learnable temperature
        if logit_scale is not None:
            scale = logit_scale.exp()
        else:
            scale = 1.0 / self.temperature

        # NxN similarity matrix
        logits_per_price = scale * v_price @ v_text.T
        logits_per_text = logits_per_price.T

        # Ground truth: diagonal entries are the positives
        labels = torch.arange(v_price.size(0), device=v_price.device)

        loss_p2t = F.cross_entropy(logits_per_price, labels)
        loss_t2p = F.cross_entropy(logits_per_text, labels)

        return (loss_p2t + loss_t2p) / 2.0
