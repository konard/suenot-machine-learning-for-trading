import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeSeriesEncoder(nn.Module):
    """
    1D CNN Encoder for financial time-series windows (e.g., prices).
    """
    def __init__(self, input_dim=1, hidden_dim=64, projection_dim=32):
        super(TimeSeriesEncoder, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        # x shape: (batch, input_dim, window_size)
        h = self.conv_net(x).squeeze(-1)
        v = self.projector(h)
        return v

class TextEncoder(nn.Module):
    """
    Simple MLP for text embeddings (e.g., news sentiment features or tokens).
    Uses mean-pooling over token embeddings.
    """
    def __init__(self, vocab_size, embed_dim=64, projection_dim=32):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim)
        )

    def forward(self, tokens):
        # tokens shape: (batch, seq_len)
        mask = (tokens != 0).unsqueeze(-1).float()
        embeds = self.embedding(tokens) * mask
        sum_embeds = embeds.sum(dim=1)
        valid_tokens = mask.sum(dim=1).clamp(min=1e-9)
        h = sum_embeds / valid_tokens
        h = self.layer_norm(h)
        v = self.projector(h)
        return v

class DualEncoderModel(nn.Module):
    """
    Wraps both encoders into a single model.
    """
    def __init__(self, vocab_size, projection_dim=32):
        super(DualEncoderModel, self).__init__()
        self.price_encoder = TimeSeriesEncoder(projection_dim=projection_dim)
        self.text_encoder = TextEncoder(vocab_size, projection_dim=projection_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, price, text):
        v_price = self.price_encoder(price)
        v_text = self.text_encoder(text)
        return v_price, v_text
