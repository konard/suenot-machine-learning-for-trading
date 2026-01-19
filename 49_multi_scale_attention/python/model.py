"""
Multi-Scale Attention Model for Financial Time Series.

This module implements the core multi-scale attention architecture that
processes financial data at multiple temporal resolutions simultaneously.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learnable)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ScaleEncoder(nn.Module):
    """
    Encoder for a specific time scale.

    Each scale has its own encoder optimized for that temporal resolution.
    Short scales focus on local patterns while long scales capture global trends.
    """

    def __init__(
        self,
        scale_name: str,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        use_learnable_pe: bool = True,
    ):
        """
        Args:
            scale_name: Name of the scale (e.g., '1min', '1H')
            input_dim: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            dropout: Dropout rate
            use_learnable_pe: Whether to use learnable positional encoding
        """
        super().__init__()
        self.scale_name = scale_name
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Positional encoding
        if use_learnable_pe:
            self.pos_encoding = LearnablePositionalEncoding(d_model, dropout=dropout)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a sequence at this scale.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            mask: Optional attention mask [batch, seq_len]

        Returns:
            Encoded representation [batch, seq_len, d_model]
        """
        # Project input to model dimension
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer encoder
        if mask is not None:
            x = self.encoder(x, src_key_padding_mask=mask)
        else:
            x = self.encoder(x)

        return self.output_norm(x)


class MultiResolutionAttention(nn.Module):
    """
    Cross-scale attention mechanism.

    Enables each scale to attend to information from all other scales,
    discovering cross-scale dependencies and temporal relationships.
    """

    def __init__(
        self, d_model: int, n_heads: int, n_scales: int, dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_scales: Number of scales
            dropout: Dropout rate
        """
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model
        self.n_heads = n_heads

        # Cross-attention for each scale
        self.cross_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_scales)
            ]
        )

        # Layer norms for residual connections
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_scales)]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self, scale_features: List[torch.Tensor], return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-scale attention.

        Args:
            scale_features: List of encoded features [batch, seq_len_i, d_model]
            return_attention: Whether to return attention weights

        Returns:
            fused: Fused representation [batch, d_model]
            attention: Optional attention weights [batch, n_scales, total_len]
        """
        batch_size = scale_features[0].shape[0]

        # Concatenate all scales for keys/values
        all_features = torch.cat(scale_features, dim=1)

        attended = []
        attention_weights = []

        for i, features in enumerate(scale_features):
            # Query: last position of this scale (most recent)
            query = features[:, -1:, :]

            # Attend to all scales
            attn_out, attn_weight = self.cross_attention[i](
                query, all_features, all_features, need_weights=return_attention
            )

            # Residual connection with last position
            residual = features[:, -1, :]
            attn_out = self.layer_norms[i](attn_out.squeeze(1) + residual)

            attended.append(attn_out)
            if return_attention:
                attention_weights.append(attn_weight.squeeze(1))

        # Combine attended features from all scales
        combined = torch.cat(attended, dim=-1)
        output = self.output_proj(combined)

        if return_attention:
            return output, torch.stack(attention_weights, dim=1)
        return output, None


class CrossScaleFusion(nn.Module):
    """
    Learnable fusion of multi-scale representations.

    Combines information from multiple scales using learned importance
    weights and dynamic gating based on input content.
    """

    def __init__(self, d_model: int, n_scales: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_scales: Number of scales
            dropout: Dropout rate
        """
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model

        # Learnable scale weights (initialized uniformly)
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        # Dynamic gating network
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_scales),
            nn.Softmax(dim=-1),
        )

        # Scale-specific transformations
        self.scale_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(n_scales)
            ]
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self, scale_features: List[torch.Tensor], return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Fuse multi-scale features.

        Args:
            scale_features: List of [batch, d_model] from each scale
            return_weights: Whether to return fusion weights

        Returns:
            fused: Fused representation [batch, d_model]
            weights: Optional tuple of (static_weights, dynamic_gates)
        """
        # Transform each scale
        transformed = [
            self.scale_transforms[i](feat) for i, feat in enumerate(scale_features)
        ]

        # Compute dynamic gates
        concat = torch.cat(scale_features, dim=-1)
        dynamic_gates = self.gate_network(concat)

        # Combine static weights with dynamic gates
        static_weights = torch.softmax(self.scale_weights, dim=0)
        combined_weights = static_weights * dynamic_gates
        combined_weights = combined_weights / (
            combined_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Weighted sum of transformed features
        stacked = torch.stack(transformed, dim=1)
        weighted = (stacked * combined_weights.unsqueeze(-1)).sum(dim=1)

        # Final fusion
        output = self.fusion(weighted)

        if return_weights:
            return output, (static_weights, dynamic_gates)
        return output, None


class MultiScaleAttention(nn.Module):
    """
    Multi-Scale Attention Network for financial time series prediction.

    Processes data at multiple temporal resolutions and learns to combine
    insights from different time scales for improved prediction accuracy.
    """

    def __init__(
        self,
        scale_configs: Dict[str, Dict],
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        """
        Args:
            scale_configs: Configuration for each scale
                {
                    'scale_name': {
                        'input_dim': int,
                        'seq_len': int  # Expected sequence length
                    }
                }
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers per scale
            dropout: Dropout rate
            output_dim: Output dimension
        """
        super().__init__()
        self.scale_names = list(scale_configs.keys())
        self.n_scales = len(self.scale_names)
        self.d_model = d_model
        self.scale_configs = scale_configs

        # Scale-specific encoders
        self.scale_encoders = nn.ModuleDict()
        for name, config in scale_configs.items():
            self.scale_encoders[name] = ScaleEncoder(
                scale_name=name,
                input_dim=config["input_dim"],
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_encoder_layers,
                dropout=dropout,
            )

        # Multi-resolution attention
        self.multi_res_attention = MultiResolutionAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_scales=self.n_scales,
            dropout=dropout,
        )

        # Cross-scale fusion
        self.cross_scale_fusion = CrossScaleFusion(
            d_model=d_model,
            n_scales=self.n_scales,
            dropout=dropout,
        )

        # Prediction heads
        self.short_term_head = self._make_prediction_head(d_model, output_dim, dropout)
        self.medium_term_head = self._make_prediction_head(d_model, output_dim, dropout)
        self.long_term_head = self._make_prediction_head(d_model, output_dim, dropout)

        # Direction classification head
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

    def _make_prediction_head(
        self, d_model: int, output_dim: int, dropout: float
    ) -> nn.Module:
        """Create a prediction head."""
        return nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(
        self,
        scale_inputs: Dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            scale_inputs: Dictionary mapping scale name to input tensor
                Each tensor has shape [batch, seq_len, input_dim]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with predictions:
                - 'short_term': Short-term prediction
                - 'medium_term': Medium-term prediction
                - 'long_term': Long-term prediction
                - 'direction': Direction probability (0-1)
                - 'uncertainty': Prediction uncertainty
                - 'attention': Optional attention weights
        """
        # Encode each scale
        scale_encodings = []
        for name in self.scale_names:
            x = scale_inputs[name]
            encoded = self.scale_encoders[name](x)
            scale_encodings.append(encoded)

        # Multi-resolution attention
        attention_output, attention_weights = self.multi_res_attention(
            scale_encodings, return_attention=return_attention
        )

        # Cross-scale fusion
        scale_summaries = [enc[:, -1, :] for enc in scale_encodings]
        fusion_output, fusion_weights = self.cross_scale_fusion(
            scale_summaries, return_weights=return_attention
        )

        # Combine attention and fusion outputs
        combined = attention_output + fusion_output

        # Generate predictions
        result = {
            "short_term": self.short_term_head(combined),
            "medium_term": self.medium_term_head(combined),
            "long_term": self.long_term_head(combined),
            "direction": self.direction_head(combined),
            "uncertainty": self.uncertainty_head(combined),
        }

        if return_attention:
            result["attention"] = attention_weights
            result["fusion_weights"] = fusion_weights

        return result

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        target_horizon: str = "short_term",
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            predictions: Model predictions
            targets: Target values [batch, 1]
            target_horizon: Which horizon to use as primary target

        Returns:
            Total loss
        """
        # Regression loss
        mse_loss = F.mse_loss(predictions[target_horizon], targets)

        # Direction loss
        direction_target = (targets > 0).float()
        direction_loss = F.binary_cross_entropy(
            predictions["direction"], direction_target
        )

        # Uncertainty-weighted loss (negative log-likelihood under Gaussian)
        uncertainty = predictions["uncertainty"]
        nll_loss = (
            0.5 * torch.log(uncertainty)
            + 0.5 * (targets - predictions[target_horizon]) ** 2 / uncertainty
        ).mean()

        # Combine losses
        total_loss = mse_loss + 0.5 * direction_loss + 0.1 * nll_loss

        return total_loss

    def get_scale_importance(self) -> Dict[str, float]:
        """Get learned scale importance weights."""
        weights = torch.softmax(
            self.cross_scale_fusion.scale_weights, dim=0
        ).detach().cpu().numpy()
        return {name: float(w) for name, w in zip(self.scale_names, weights)}
