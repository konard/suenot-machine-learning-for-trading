"""
BigBird Sparse Attention Model Implementation

Provides:
- BigBirdConfig: Model configuration
- BigBirdSparseAttention: Sparse attention mechanism combining random, window, and global
- BigBirdEncoderLayer: Transformer encoder layer with sparse attention
- BigBirdForTrading: Complete model for financial time series prediction
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
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    QUANTILE = "quantile"
    PORTFOLIO = "portfolio"


@dataclass
class BigBirdConfig:
    """
    Configuration for BigBird model.

    Args:
        seq_len: Input sequence length
        input_dim: Number of input features
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        window_size: Size of local attention window
        num_random: Number of random attention connections per query
        num_global: Number of global tokens
        global_tokens: Position of global tokens ('first', 'last', 'both')
        output_dim: Output dimension
        output_type: Type of output (regression, classification, quantile, portfolio)
        quantiles: Quantiles for quantile regression

    Example:
        config = BigBirdConfig(
            seq_len=256,
            input_dim=6,
            d_model=128,
            n_heads=8,
            window_size=7,
            num_random=3,
            num_global=2
        )
    """
    # Architecture
    seq_len: int = 512
    input_dim: int = 6
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # BigBird specific
    window_size: int = 7
    num_random: int = 3
    num_global: int = 2
    global_tokens: str = 'first'  # 'first', 'last', 'both'

    # Output
    output_dim: int = 1
    output_type: OutputType = OutputType.REGRESSION
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    def validate(self):
        """Validate configuration parameters"""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.window_size % 2 == 1, \
            f"window_size ({self.window_size}) must be odd"
        assert 0 <= self.dropout <= 1, \
            f"dropout ({self.dropout}) must be in [0, 1]"
        assert self.global_tokens in ['first', 'last', 'both'], \
            f"global_tokens must be 'first', 'last', or 'both'"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def final_output_dim(self) -> int:
        if self.output_type == OutputType.CLASSIFICATION:
            return 3  # down, neutral, up
        elif self.output_type == OutputType.QUANTILE:
            return len(self.quantiles)
        elif self.output_type == OutputType.PORTFOLIO:
            return self.output_dim
        return self.output_dim


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

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
        """
        Args:
            x: Tensor of shape [batch, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BigBirdSparseAttention(nn.Module):
    """
    BigBird sparse attention mechanism.

    Combines three types of attention:
    1. Random attention: Each query attends to r random keys
    2. Window attention: Each query attends to w/2 tokens on each side
    3. Global attention: Designated tokens attend to all and are attended by all

    This reduces complexity from O(n^2) to O(n).
    """

    def __init__(self, config: BigBirdConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Pre-compute attention pattern
        self.register_buffer(
            'attention_mask',
            self._create_attention_mask(config.seq_len)
        )

    def _create_attention_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create BigBird sparse attention mask.

        Returns:
            Boolean mask of shape [seq_len, seq_len] where True means attend
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

        # 1. Window (local) attention
        half_w = self.config.window_size // 2
        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(seq_len, i + half_w + 1)
            mask[i, start:end] = True

        # 2. Random attention (fixed pattern for reproducibility)
        torch.manual_seed(42)
        for i in range(seq_len):
            # Find positions not yet in the mask
            non_mask_positions = torch.where(~mask[i])[0]
            if len(non_mask_positions) > 0:
                num_random = min(self.config.num_random, len(non_mask_positions))
                random_indices = non_mask_positions[
                    torch.randperm(len(non_mask_positions))[:num_random]
                ]
                mask[i, random_indices] = True

        # 3. Global attention
        if self.config.global_tokens == 'first':
            global_indices = list(range(self.config.num_global))
        elif self.config.global_tokens == 'last':
            global_indices = list(range(seq_len - self.config.num_global, seq_len))
        else:  # 'both'
            half = self.config.num_global // 2
            global_indices = (
                list(range(half)) +
                list(range(seq_len - (self.config.num_global - half), seq_len))
            )

        for idx in global_indices:
            if idx < seq_len:
                mask[idx, :] = True  # Global token attends to all
                mask[:, idx] = True  # All tokens attend to global

        return mask

    def _get_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get attention mask, potentially creating a new one if seq_len differs."""
        if seq_len == self.attention_mask.size(0):
            return self.attention_mask

        # Create new mask for different sequence length
        return self._create_attention_mask(seq_len).to(device)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sparse attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention: [batch, n_heads, seq_len, seq_len] if return_attention
        """
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply sparse mask
        mask = self._get_attention_mask(seq_len, x.device)
        scores = scores.masked_fill(
            ~mask.unsqueeze(0).unsqueeze(0),
            float('-inf')
        )

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle any NaN from all-masked rows
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.out_proj(out)

        return out, attn if return_attention else None


class BigBirdEncoderLayer(nn.Module):
    """
    BigBird transformer encoder layer.

    Contains:
    - BigBird sparse self-attention
    - Layer normalization
    - Feed-forward network with GELU activation
    """

    def __init__(self, config: BigBirdConfig):
        super().__init__()

        self.attention = BigBirdSparseAttention(config)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            attention: Attention weights if return_attention
        """
        # Self-attention with residual and pre-norm
        attn_out, attn_weights = self.attention(self.norm1(x), return_attention)
        x = x + attn_out

        # Feed-forward with residual and pre-norm
        x = x + self.ff(self.norm2(x))

        return x, attn_weights


class BigBirdForTrading(nn.Module):
    """
    BigBird model for financial time series prediction.

    Supports:
    - Regression: Predict continuous values (e.g., returns)
    - Classification: Predict direction (down, neutral, up)
    - Quantile: Predict distribution quantiles
    - Portfolio: Predict allocation weights

    Example:
        config = BigBirdConfig(
            seq_len=256,
            input_dim=6,
            d_model=128,
            n_heads=8,
            window_size=7,
            num_random=3,
            num_global=2
        )
        model = BigBirdForTrading(config)

        x = torch.randn(32, 256, 6)  # [batch, seq_len, features]
        output = model(x)
        print(output['predictions'].shape)  # [32, 1]
    """

    def __init__(self, config: BigBirdConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model,
            max_len=config.seq_len * 2,
            dropout=config.dropout
        )

        # Encoder layers
        self.layers = nn.ModuleList([
            BigBirdEncoderLayer(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Output head
        self.output_head = self._build_output_head(config)

        # Initialize weights
        self._init_weights()

    def _build_output_head(self, config: BigBirdConfig) -> nn.Module:
        """Build output projection layer based on output type"""
        if config.output_type == OutputType.CLASSIFICATION:
            return nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 3)  # down, neutral, up
            )
        elif config.output_type == OutputType.QUANTILE:
            return nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, len(config.quantiles))
            )
        elif config.output_type == OutputType.PORTFOLIO:
            return nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, config.output_dim)
            )
        else:  # REGRESSION
            return nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, config.output_dim)
            )

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - predictions: Model predictions
                - attention_weights: Attention weights per layer (if return_attention)
                - hidden_states: Final hidden states
        """
        batch, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through encoder layers
        all_attention = {}
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, return_attention)
            if attn is not None:
                all_attention[f'layer_{i}'] = attn

        # Final layer norm
        x = self.final_norm(x)

        # Pool: use last position
        pooled = x[:, -1, :]  # [batch, d_model]

        # Output projection
        predictions = self.output_head(pooled)

        # Apply appropriate activation for output type
        if self.config.output_type == OutputType.CLASSIFICATION:
            predictions = F.softmax(predictions, dim=-1)
        elif self.config.output_type == OutputType.PORTFOLIO:
            predictions = F.softmax(predictions, dim=-1)

        result = {
            'predictions': predictions,
            'hidden_states': x
        }

        if return_attention:
            result['attention_weights'] = all_attention

        return result

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss based on output type.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            mask: Optional mask for valid positions

        Returns:
            Loss value
        """
        if self.config.output_type == OutputType.CLASSIFICATION:
            loss = F.cross_entropy(predictions, targets.long())
        elif self.config.output_type == OutputType.QUANTILE:
            loss = self._quantile_loss(predictions, targets)
        elif self.config.output_type == OutputType.PORTFOLIO:
            # Portfolio loss: negative Sharpe ratio proxy
            returns = torch.sum(predictions * targets, dim=-1)
            loss = -returns.mean() / (returns.std() + 1e-8)
        else:  # REGRESSION
            loss = F.mse_loss(predictions, targets)

        return loss

    def _quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute pinball loss for quantile regression"""
        quantiles = torch.tensor(
            self.config.quantiles,
            device=predictions.device
        )

        # Expand dimensions for broadcasting
        targets = targets.unsqueeze(-1)  # [batch, 1]
        errors = targets - predictions   # [batch, n_quantiles]

        loss = torch.max(
            quantiles * errors,
            (quantiles - 1) * errors
        ).mean()

        return loss


def create_model(
    seq_len: int = 256,
    input_dim: int = 6,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    window_size: int = 7,
    num_random: int = 3,
    num_global: int = 2,
    output_type: str = 'regression',
    **kwargs
) -> BigBirdForTrading:
    """
    Factory function to create BigBird model.

    Args:
        seq_len: Input sequence length
        input_dim: Number of input features
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        window_size: Size of local attention window
        num_random: Number of random attention connections
        num_global: Number of global tokens
        output_type: Type of output ('regression', 'classification', 'quantile', 'portfolio')
        **kwargs: Additional configuration parameters

    Returns:
        Configured BigBirdForTrading model
    """
    output_type_enum = OutputType(output_type)

    config = BigBirdConfig(
        seq_len=seq_len,
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        window_size=window_size,
        num_random=num_random,
        num_global=num_global,
        output_type=output_type_enum,
        **kwargs
    )

    return BigBirdForTrading(config)


if __name__ == "__main__":
    # Test the model
    print("Testing BigBird model...")

    config = BigBirdConfig(
        seq_len=128,
        input_dim=6,
        d_model=64,
        n_heads=4,
        n_layers=2,
        window_size=7,
        num_random=3,
        num_global=2
    )

    model = BigBirdForTrading(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(4, 128, 6)
    output = model(x, return_attention=True)

    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Hidden states shape: {output['hidden_states'].shape}")
    print(f"Attention weights available: {'attention_weights' in output}")

    # Test different output types
    for output_type in OutputType:
        config.output_type = output_type
        model = BigBirdForTrading(config)
        output = model(x)
        print(f"{output_type.value}: predictions shape = {output['predictions'].shape}")

    print("\nAll tests passed!")
