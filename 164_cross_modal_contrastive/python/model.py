import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeSeriesEncoder(nn.Module):
    """
    1D-CNN Encoder for Financial Price Windows.
    Takes (B, 1, SeqLen) and outputs (B, Projection_Dim)
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
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection Head specific to the Time-Series modality
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        h = self.conv_block(x)
        h = self.adaptive_pool(h).squeeze(-1)
        v = self.projector(h)
        # Return raw projected similarities instead of forcing them onto a rigid hypersphere
        # The logit_scale parameter will handle scaling
        return v

class TextEncoder(nn.Module):
    """
    Simple Token Embedding + Mean Pooling Encoder for Text inputs.
    Takes (B, Tokens) and outputs (B, Projection_Dim)
    """
    def __init__(self, vocab_size=5000, embed_dim=64, projection_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Projection Head specific to the Text modality
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )
        
    def forward(self, x):
        # x is a batch of token indices: (B, SeqLen)
        embeds = self.embedding(x)
        
        # Mean pooling across the token sequence length, ignoring padding (0)
        mask = (x != 0).float().unsqueeze(-1)
        sum_embeds = (embeds * mask).sum(dim=1)
        valid_tokens = mask.sum(dim=1).clamp(min=1.0)
        h = sum_embeds / valid_tokens
        
        # LayerNorm helps stabilize text embeddings before projection
        h = F.layer_norm(h, h.shape[1:])
        
        v = self.projector(h)
        
        # L2 Normalize constraint removed to prevent vanishing gradients
        return v

class CrossModalCLIPModel(nn.Module):
    """
    Wrapper holding both the TS and Text encoders.
    """
    def __init__(self, vocab_size=5000, projection_dim=32):
        super().__init__()
        self.ts_encoder = TimeSeriesEncoder(projection_dim=projection_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, projection_dim=projection_dim)
        
        # Learnable temperature parameter (as in original CLIP)
        # Initiated with log(1 / 0.07) = 2.6592
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, x_price, x_text):
        v_price = self.ts_encoder(x_price)
        v_text = self.text_encoder(x_text)
        return v_price, v_text

class CLIPLoss(nn.Module):
    """
    Simplified Contrastive Loss using Cosine Embedding.
    For each (Price, Text) positive pair, we also pass a (Price, Negative_Text) pair.
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, v_price, v_text, v_text_neg):
        # Normalize features to guarantee Cosine Similarity [-1, 1]
        v_price = F.normalize(v_price, p=2, dim=1)
        v_text = F.normalize(v_text, p=2, dim=1)
        v_text_neg = F.normalize(v_text_neg, p=2, dim=1)
        
        # Targets: 1 for positive pairs, -1 for negative pairs
        target_pos = torch.ones(v_price.size(0), device=v_price.device)
        target_neg = -torch.ones(v_price.size(0), device=v_price.device)
        
        # Calculate loss for pulling positives together
        loss_pos = self.loss_fn(v_price, v_text, target_pos)
        
        # Calculate loss for pushing negatives apart
        loss_neg = self.loss_fn(v_price, v_text_neg, target_neg)
        
        return loss_pos + loss_neg
