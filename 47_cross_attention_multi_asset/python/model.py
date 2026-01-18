"""
Cross-Attention Multi-Asset Model Implementation

Provides transformer-based model with cross-attention mechanism
for modeling relationships between multiple financial assets.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from enum import Enum


class OutputType(Enum):
    """Type of model output"""
    REGRESSION = "regression"      # Predict returns
    CLASSIFICATION = "classification"  # Predict direction (up/down/neutral)
    PORTFOLIO = "portfolio"        # Predict portfolio weights


@dataclass
class CrossAttentionConfig:
    """
    Configuration for Cross-Attention Multi-Asset model.

    Example:
        config = CrossAttentionConfig(
            n_assets=5,
            n_features=6,
            seq_len=168
        )
    """
    # Data dimensions
    n_assets: int = 5
    n_features: int = 6
    seq_len: int = 168

    # Model architecture
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    # Attention settings
    use_temporal_attention: bool = True
    use_cross_asset_attention: bool = True
    max_lag: int = 5  # For temporal cross-attention

    # Output settings
    output_type: OutputType = OutputType.REGRESSION
    n_classes: int = 3  # For classification: down, neutral, up

    def validate(self):
        """Validate configuration parameters"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_assets > 0, "n_assets must be positive"
        assert self.n_features > 0, "n_features must be positive"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Token embedding using 1D convolution.

    Extracts local patterns from raw features.
    """

    def __init__(self, n_features: int, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_features]

        Returns:
            [batch, seq_len, d_model]
        """
        x = x.transpose(1, 2)  # [batch, n_features, seq_len]
        x = self.conv(x)       # [batch, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        x = self.activation(x)
        return self.norm(x)


class AssetEmbedding(nn.Module):
    """Learnable embedding for each asset."""

    def __init__(self, n_assets: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(n_assets, d_model)

    def forward(self, x: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        """
        Add asset-specific embedding.

        Args:
            x: [batch, n_assets, seq_len, d_model]
            asset_ids: [n_assets]

        Returns:
            [batch, n_assets, seq_len, d_model]
        """
        asset_emb = self.embedding(asset_ids)  # [n_assets, d_model]
        return x + asset_emb.unsqueeze(0).unsqueeze(2)


class CrossAssetAttention(nn.Module):
    """
    Cross-Asset Attention Mechanism.

    Allows each asset to attend to all other assets,
    learning inter-asset dependencies and relationships.
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Cross-asset attention forward pass.

        Args:
            x: [batch, n_assets, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, n_assets, seq_len, d_model]
            attention: [batch, n_heads, n_assets, n_assets] (optional)
        """
        batch, n_assets, seq_len, d_model = x.shape
        residual = x

        # Pool temporal dimension for cross-asset attention
        x_pooled = x.mean(dim=2)  # [batch, n_assets, d_model]

        # Project to Q, K, V
        Q = self.q_proj(x_pooled)  # [batch, n_assets, d_model]
        K = self.k_proj(x_pooled)
        V = self.v_proj(x_pooled)

        # Reshape for multi-head attention
        Q = Q.view(batch, n_assets, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, n_assets, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, n_assets, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, n_heads, n_assets, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch, n_heads, n_assets, n_assets]

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)
        # [batch, n_heads, n_assets, head_dim]

        # Reshape and project
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, n_assets, d_model)
        context = self.out_proj(context)

        # Broadcast back to sequence length
        context = context.unsqueeze(2).expand(-1, -1, seq_len, -1)

        # Residual connection and normalization
        output = self.norm(residual + self.dropout(context))

        if return_attention:
            return output, attention.mean(dim=1)  # Average over heads
        return output, None


class TemporalSelfAttention(nn.Module):
    """
    Temporal Self-Attention within each asset.

    Models temporal dependencies in the time series.
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Temporal self-attention forward pass.

        Args:
            x: [batch, n_assets, seq_len, d_model]

        Returns:
            output: [batch, n_assets, seq_len, d_model]
            attention: Optional attention weights
        """
        batch, n_assets, seq_len, d_model = x.shape

        # Process each asset separately
        x_flat = x.view(batch * n_assets, seq_len, d_model)

        # Self-attention
        attn_out, attn_weights = self.mha(
            x_flat, x_flat, x_flat,
            need_weights=return_attention
        )

        # Reshape back
        attn_out = attn_out.view(batch, n_assets, seq_len, d_model)
        output = self.norm(x + self.dropout(attn_out))

        if return_attention:
            attn_weights = attn_weights.view(batch, n_assets, seq_len, seq_len)
            return output, attn_weights
        return output, None


class TemporalCrossAttention(nn.Module):
    """
    Temporal Cross-Attention with lead-lag relationships.

    Models how past values of one asset affect future values of another.
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.max_lag = config.max_lag
        self.cross_attention = CrossAssetAttention(config)

        # Learnable lag weights
        self.lag_weights = nn.Parameter(
            torch.ones(config.max_lag + 1) / (config.max_lag + 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Temporal cross-attention with multiple lags.

        Args:
            x: [batch, n_assets, seq_len, d_model]

        Returns:
            output: Attended representation with temporal alignment
            attention: Cross-asset attention at different lags
        """
        outputs = []
        attentions = []

        # Compute attention at different lags
        for lag in range(self.max_lag + 1):
            if lag == 0:
                x_lagged = x
            else:
                # Shift backward by lag steps
                x_lagged = F.pad(x[:, :, :-lag], (0, 0, lag, 0))

            out, attn = self.cross_attention(x_lagged, return_attention)
            outputs.append(out)
            if return_attention:
                attentions.append(attn)

        # Weighted combination across lags
        lag_weights = F.softmax(self.lag_weights, dim=0)
        output = sum(w * o for w, o in zip(lag_weights, outputs))

        if return_attention:
            return output, torch.stack(attentions, dim=1)
        return output, None


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class CrossAttentionEncoderLayer(nn.Module):
    """
    Encoder layer with temporal self-attention and cross-asset attention.
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()

        # Temporal attention within each asset
        self.temporal_attention = TemporalSelfAttention(config)

        # Cross-asset attention
        self.cross_asset_attention = CrossAssetAttention(config)

        # Feed-forward
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Encoder layer forward pass.

        Args:
            x: [batch, n_assets, seq_len, d_model]

        Returns:
            output: [batch, n_assets, seq_len, d_model]
            attention_dict: Dictionary of attention weights
        """
        attention_dict = {}

        # Temporal self-attention
        x, temporal_attn = self.temporal_attention(x, return_attention)
        if temporal_attn is not None:
            attention_dict['temporal'] = temporal_attn

        # Cross-asset attention
        x, cross_attn = self.cross_asset_attention(x, return_attention)
        if cross_attn is not None:
            attention_dict['cross_asset'] = cross_attn

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm(x + self.dropout(ff_out))

        return x, attention_dict


class CrossAttentionMultiAsset(nn.Module):
    """
    Cross-Attention Multi-Asset Model.

    Transformer-based model that uses cross-attention to model
    relationships between multiple financial assets.

    Example:
        config = CrossAttentionConfig(n_assets=5, n_features=6)
        model = CrossAttentionMultiAsset(config)

        x = torch.randn(2, 5, 168, 6)  # [batch, assets, seq, features]
        output = model(x)
        print(output['predictions'].shape)  # [2, 5]
    """

    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Embedding layers
        self.token_embedding = TokenEmbedding(
            config.n_features, config.d_model
        )
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.seq_len * 2, config.dropout
        )
        self.asset_embedding = AssetEmbedding(config.n_assets, config.d_model)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            CrossAttentionEncoderLayer(config)
            for _ in range(config.n_layers)
        ])

        # Output head
        self.output_head = self._build_output_head(config)

        # Register asset IDs buffer
        self.register_buffer('asset_ids', torch.arange(config.n_assets))

    def _build_output_head(self, config: CrossAttentionConfig) -> nn.Module:
        """Build output projection based on output type."""
        if config.output_type == OutputType.CLASSIFICATION:
            return nn.Linear(config.d_model, config.n_classes)
        else:
            return nn.Linear(config.d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, n_assets, seq_len, n_features]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
                - predictions: Model predictions
                - attention_weights: Optional attention weights
        """
        batch, n_assets, seq_len, n_features = x.shape

        # Token embedding for each asset
        embedded = []
        for a in range(n_assets):
            emb = self.token_embedding(x[:, a])  # [batch, seq_len, d_model]
            emb = self.pos_encoding(emb)
            embedded.append(emb)
        x = torch.stack(embedded, dim=1)  # [batch, n_assets, seq_len, d_model]

        # Add asset embedding
        x = self.asset_embedding(x, self.asset_ids)

        # Encoder layers
        all_attention = {}
        for i, layer in enumerate(self.encoder_layers):
            x, attn_dict = layer(x, return_attention)
            if attn_dict:
                all_attention[f'layer_{i}'] = attn_dict

        # Pool temporal dimension (take last timestep)
        x = x[:, :, -1, :]  # [batch, n_assets, d_model]

        # Output projection
        predictions = self._compute_output(x)

        result = {
            'predictions': predictions,
            'attention_weights': all_attention if return_attention else None
        }

        return result

    def _compute_output(self, x: torch.Tensor) -> torch.Tensor:
        """Compute output predictions based on output type."""
        batch, n_assets, d_model = x.shape

        if self.config.output_type == OutputType.PORTFOLIO:
            # Portfolio weights via softmax
            logits = self.output_head(x).squeeze(-1)  # [batch, n_assets]
            return F.softmax(logits, dim=-1)

        elif self.config.output_type == OutputType.CLASSIFICATION:
            # Per-asset direction classification
            logits = self.output_head(x)  # [batch, n_assets, n_classes]
            return logits

        else:  # REGRESSION
            # Per-asset return prediction
            return self.output_head(x).squeeze(-1)  # [batch, n_assets]

    @property
    def output_type(self) -> str:
        """Return output type string for compatibility."""
        return self.config.output_type.value


def create_model(
    n_assets: int = 5,
    n_features: int = 6,
    seq_len: int = 168,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    output_type: str = 'regression',
    **kwargs
) -> CrossAttentionMultiAsset:
    """
    Factory function to create Cross-Attention model.

    Args:
        n_assets: Number of assets
        n_features: Number of features per asset
        seq_len: Sequence length
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        output_type: One of 'regression', 'classification', 'portfolio'

    Returns:
        CrossAttentionMultiAsset model
    """
    output_type_enum = OutputType(output_type)

    config = CrossAttentionConfig(
        n_assets=n_assets,
        n_features=n_features,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        output_type=output_type_enum,
        **kwargs
    )

    return CrossAttentionMultiAsset(config)


if __name__ == "__main__":
    # Test the model
    print("Testing Cross-Attention Multi-Asset model...")

    config = CrossAttentionConfig(
        n_assets=5,
        n_features=6,
        seq_len=48,
        d_model=32,
        n_heads=4,
        n_layers=2
    )

    model = CrossAttentionMultiAsset(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 5, 48, 6)
    output = model(x, return_attention=True)

    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Attention weights available: {output['attention_weights'] is not None}")

    # Test different output types
    for output_type in OutputType:
        config.output_type = output_type
        model = CrossAttentionMultiAsset(config)
        output = model(x)
        print(f"{output_type.value}: predictions shape = {output['predictions'].shape}")

    print("\nAll tests passed!")
