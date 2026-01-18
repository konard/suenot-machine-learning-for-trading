"""
Informer Model Implementation with ProbSparse Attention

Based on: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
Paper: https://arxiv.org/abs/2012.07436

This module provides:
- InformerConfig: Model configuration
- ProbSparseAttention: O(L·log(L)) attention mechanism
- AttentionDistilling: Sequence length reduction
- InformerModel: Complete model for time series forecasting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class OutputType(Enum):
    """Type of model output"""
    REGRESSION = "regression"
    DIRECTION = "direction"
    QUANTILE = "quantile"


@dataclass
class InformerConfig:
    """
    Configuration for Informer model with ProbSparse attention

    Example:
        config = InformerConfig(
            seq_len=96,
            pred_len=24,
            d_model=64
        )
    """
    # Input/Output
    seq_len: int = 96          # Input sequence length
    label_len: int = 48        # Start token length for decoder
    pred_len: int = 24         # Prediction horizon
    input_features: int = 6    # Number of input features

    # Architecture
    d_model: int = 64          # Model dimension
    n_heads: int = 4           # Number of attention heads
    d_ff: int = 256            # Feed-forward dimension
    n_encoder_layers: int = 2  # Number of encoder layers
    n_decoder_layers: int = 1  # Number of decoder layers
    dropout: float = 0.1       # Dropout rate

    # ProbSparse Attention
    sampling_factor: float = 5.0  # Controls sparsity (u = c·log(L))
    use_distilling: bool = True   # Enable attention distilling

    # Output
    output_type: OutputType = OutputType.REGRESSION
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Embedding
    kernel_size: int = 3
    use_positional_encoding: bool = True

    def validate(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.kernel_size % 2 == 1, "kernel_size must be odd"
        assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"
        assert self.sampling_factor > 0, "sampling_factor must be positive"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class TokenEmbedding(nn.Module):
    """
    Token embedding using 1D convolution

    Converts [batch, seq_len, features] to [batch, seq_len, d_model]
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=config.input_features,
            out_channels=config.d_model,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.conv(x)       # [batch, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        return self.activation(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention Mechanism

    Achieves O(L·log(L)) complexity by selecting only the most
    informative queries for full attention computation.

    Key idea: Not all queries contribute equally to attention.
    "Active" queries (with spiky attention distributions) are more
    informative than "lazy" queries (with uniform distributions).

    The sparsity measurement M(q, K) = max(qK^T) - mean(qK^T)
    approximates KL-divergence from uniform distribution.
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.sampling_factor = config.sampling_factor
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def _get_probsparse_scores(
        self,
        Q: torch.Tensor,
        K: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Calculate Query Sparsity Measurement M(q, K)

        M(q_i, K) = max_j(q_i · k_j^T / sqrt(d)) - mean_j(q_i · k_j^T / sqrt(d))

        Args:
            Q: Queries [batch, n_heads, seq_len, head_dim]
            K: Keys [batch, n_heads, seq_len, head_dim]

        Returns:
            M: Sparsity measurements [batch, n_heads, seq_len]
            u: Number of top queries to select
        """
        batch, n_heads, seq_len, head_dim = Q.shape

        # Calculate u (number of active queries)
        u = max(1, min(seq_len, int(self.sampling_factor * math.log(seq_len + 1))))

        # Sample keys for efficient M computation
        U_part = min(
            int(self.sampling_factor * seq_len * math.log(seq_len + 1)),
            seq_len
        )

        # Random sample of keys
        sample_idx = torch.randint(0, seq_len, (U_part,), device=Q.device)
        K_sample = K[:, :, sample_idx, :]  # [batch, n_heads, U_part, head_dim]

        # Q·K_sample^T / sqrt(d)
        scores_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / self.scale
        # [batch, n_heads, seq_len, U_part]

        # M(q) = max(scores) - mean(scores)
        M = scores_sample.max(dim=-1)[0] - scores_sample.mean(dim=-1)
        # [batch, n_heads, seq_len]

        return M, u

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with ProbSparse attention

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attn_mask: Optional attention mask
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention: Optional [batch, n_heads, u, seq_len]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # [batch, n_heads, seq_len, head_dim]

        # Calculate sparsity measurement and number of active queries
        M, u = self._get_probsparse_scores(Q, K)

        # Select top-u queries based on sparsity measurement
        _, M_top_indices = M.topk(u, dim=-1, sorted=False)
        # [batch, n_heads, u]

        # Gather selected queries
        batch_idx = torch.arange(batch, device=x.device)[:, None, None]
        head_idx = torch.arange(self.n_heads, device=x.device)[None, :, None]
        Q_reduce = Q[batch_idx, head_idx, M_top_indices]
        # [batch, n_heads, u, head_dim]

        # Full attention only for selected queries
        attn_scores = torch.matmul(Q_reduce, K.transpose(-2, -1)) / self.scale
        # [batch, n_heads, u, seq_len]

        if attn_mask is not None:
            # Expand mask for selected queries
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute context for selected queries
        context = torch.matmul(attn_probs, V)
        # [batch, n_heads, u, head_dim]

        # Initialize output with mean values (for non-selected queries)
        # This is a form of "lazy query" approximation
        V_mean = V.mean(dim=2, keepdim=True)  # [batch, n_heads, 1, head_dim]
        output = V_mean.expand(-1, -1, seq_len, -1).clone()

        # Fill in computed values for selected queries
        M_top_indices_expanded = M_top_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        output.scatter_(2, M_top_indices_expanded, context)
        # [batch, n_heads, seq_len, head_dim]

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.out_proj(output)

        attention = attn_probs if return_attention else None
        return output, attention


class AttentionDistilling(nn.Module):
    """
    Distilling layer that reduces sequence length by half.

    Uses Conv1d + ELU + MaxPool to extract salient features
    while discarding redundant information.

    This operation:
    1. Applies 1D convolution to blend neighboring features
    2. Uses ELU activation for non-linearity
    3. Applies MaxPool(2) to reduce sequence length by half

    Result: Each layer's output has half the sequence length,
    reducing overall memory from O(N·L²) to O(N·L·log(L))
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular'  # Better for time series
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len//2, d_model]
        """
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x.transpose(1, 2)  # [batch, seq_len//2, d_model]


class EncoderLayer(nn.Module):
    """
    Informer Encoder Layer with ProbSparse attention

    Structure:
    1. ProbSparse Self-Attention
    2. Add & Norm
    3. Feed-Forward Network
    4. Add & Norm
    5. (Optional) Distilling
    """

    def __init__(self, config: InformerConfig, use_distilling: bool = True):
        super().__init__()
        self.use_distilling = use_distilling

        self.self_attention = ProbSparseAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

        self.distilling = AttentionDistilling(config.d_model) if use_distilling else None
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len or seq_len//2, d_model]
            attention: Optional attention weights
        """
        # Self-attention
        attn_out, attn_weights = self.self_attention(x, attn_mask, return_attention)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        # Distilling (reduce sequence length)
        if self.distilling is not None:
            x = self.distilling(x)

        return x, attn_weights


class InformerModel(nn.Module):
    """
    Informer: Efficient Transformer for Long Sequence Time-Series Forecasting

    Key features:
    1. ProbSparse Self-Attention: O(L·log(L)) complexity
    2. Self-Attention Distilling: Progressive sequence length reduction
    3. Multi-step output generation: Efficient long horizon prediction

    Example:
        config = InformerConfig(seq_len=96, pred_len=24)
        model = InformerModel(config)

        x = torch.randn(2, 96, 6)  # [batch, seq_len, features]
        output = model(x)
        print(output['predictions'].shape)  # [2, 24]
    """

    def __init__(self, config: InformerConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Embedding layers
        self.token_embedding = TokenEmbedding(config)
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.seq_len * 2, config.dropout
        ) if config.use_positional_encoding else None

        # Encoder layers with progressive distilling
        self.encoder_layers = nn.ModuleList()
        for i in range(config.n_encoder_layers):
            # Last layer doesn't distill
            use_distilling = config.use_distilling and (i < config.n_encoder_layers - 1)
            self.encoder_layers.append(EncoderLayer(config, use_distilling))

        # Calculate final sequence length after distilling
        final_seq_len = config.seq_len
        if config.use_distilling:
            for _ in range(config.n_encoder_layers - 1):
                final_seq_len = final_seq_len // 2

        # Output projection
        self.output_head = self._build_output_head(config, final_seq_len)

    def _build_output_head(self, config: InformerConfig, final_seq_len: int) -> nn.Module:
        """Build output projection layer"""
        flatten_dim = config.d_model * final_seq_len

        if config.output_type == OutputType.QUANTILE:
            output_dim = config.pred_len * len(config.quantiles)
        elif config.output_type == OutputType.DIRECTION:
            output_dim = config.pred_len * 3  # down, neutral, up
        else:  # REGRESSION
            output_dim = config.pred_len

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> dict:
        """
        Forward pass

        Args:
            x: Input tensor [batch, seq_len, features]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
                - predictions: [batch, pred_len] or [batch, pred_len * num_outputs]
                - attention_weights: Optional attention weights from each layer
        """
        # Token embedding
        x = self.token_embedding(x)  # [batch, seq_len, d_model]

        # Positional encoding
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        # Encoder layers
        all_attention = {}
        for i, layer in enumerate(self.encoder_layers):
            x, attn = layer(x, return_attention=return_attention)
            if attn is not None:
                all_attention[f'layer_{i}'] = attn

        # Output projection
        predictions = self.output_head(x)

        result = {
            'predictions': predictions,
            'attention_weights': all_attention if return_attention else None
        }

        # Reshape for different output types
        if self.config.output_type == OutputType.QUANTILE:
            batch = predictions.size(0)
            predictions = predictions.view(batch, self.config.pred_len, len(self.config.quantiles))
            result['predictions'] = predictions
            result['confidence'] = self._compute_confidence(predictions)

        elif self.config.output_type == OutputType.DIRECTION:
            batch = predictions.size(0)
            predictions = predictions.view(batch, self.config.pred_len, 3)
            result['predictions'] = F.softmax(predictions, dim=-1)

        return result

    def _compute_confidence(self, quantile_predictions: torch.Tensor) -> torch.Tensor:
        """Compute confidence from quantile predictions"""
        # Narrower prediction intervals = higher confidence
        interval_width = (quantile_predictions[:, :, -1] - quantile_predictions[:, :, 0]).abs()
        confidence = 1.0 / (1.0 + interval_width)
        return confidence


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing Informer model with ProbSparse attention...")

    config = InformerConfig(
        seq_len=96,
        pred_len=24,
        input_features=6,
        d_model=64,
        n_heads=4,
        n_encoder_layers=3,
        use_distilling=True
    )

    model = InformerModel(config)
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(2, 96, 6)  # [batch, seq_len, features]
    output = model(x, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Attention weights available: {output['attention_weights'] is not None}")

    # Test different output types
    print("\nTesting different output types:")
    for output_type in OutputType:
        config.output_type = output_type
        model = InformerModel(config)
        output = model(x)
        print(f"  {output_type.value}: predictions shape = {output['predictions'].shape}")

    # Test sequence length reduction from distilling
    print("\nSequence length through layers (with distilling):")
    config.output_type = OutputType.REGRESSION
    config.use_distilling = True
    model = InformerModel(config)

    seq_len = config.seq_len
    print(f"  Input: {seq_len}")
    for i in range(config.n_encoder_layers):
        if i < config.n_encoder_layers - 1:
            seq_len = seq_len // 2
        print(f"  After layer {i+1}: {seq_len}")

    print("\nAll tests passed!")
