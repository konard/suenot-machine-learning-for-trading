"""
Time Series Transformer Model with Positional Encoding

This module provides a complete transformer model for time series
prediction with configurable positional encoding methods.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum

from .positional_encoding import (
    SinusoidalPositionalEncoding,
    TimeSeriesSinusoidalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    RoPETimeSeriesAttention,
    CalendarEncoding,
    MarketSessionEncoding,
)


class EncodingType(Enum):
    """Available positional encoding types."""
    SINUSOIDAL = "sinusoidal"
    TIMESERIES_SINUSOIDAL = "timeseries_sinusoidal"
    LEARNED = "learned"
    ROPE = "rope"
    NONE = "none"


class OutputType(Enum):
    """Model output types."""
    REGRESSION = "regression"
    DIRECTION = "direction"
    QUANTILE = "quantile"


@dataclass
class TimeSeriesTransformerConfig:
    """
    Configuration for Time Series Transformer.

    Attributes:
        input_features: Number of input features
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        max_len: Maximum sequence length
        encoding_type: Type of positional encoding
        output_type: Type of output (regression, direction, quantile)
        horizon: Prediction horizon
        use_calendar: Whether to use calendar encoding
        use_market_session: Whether to use market session encoding
        market_type: Market type for session encoding ('crypto' or 'stock')
        quantiles: Quantiles for quantile regression
    """
    input_features: int = 6
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 1024
    encoding_type: EncodingType = EncodingType.ROPE
    output_type: OutputType = OutputType.REGRESSION
    horizon: int = 1
    use_calendar: bool = False
    use_market_session: bool = False
    market_type: str = 'crypto'
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    def validate(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.dropout >= 0 and self.dropout < 1, "dropout must be in [0, 1)"


class TokenEmbedding(nn.Module):
    """
    Token embedding using 1D convolution.

    Converts raw input features to model dimension.
    """

    def __init__(self, input_features: int, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_features,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_features]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.conv(x)       # [batch, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        return self.activation(x)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with optional RoPE attention.
    """

    def __init__(self, config: TimeSeriesTransformerConfig, use_rope: bool = False):
        super().__init__()
        self.use_rope = use_rope

        if use_rope:
            self.self_attention = RoPETimeSeriesAttention(
                config.d_model, config.n_heads, config.dropout, config.max_len
            )
        else:
            self.self_attention = nn.MultiheadAttention(
                config.d_model, config.n_heads,
                dropout=config.dropout, batch_first=True
            )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask
        """
        # Self-attention
        if self.use_rope:
            attn_out = self.self_attention(x, mask)
        else:
            attn_out, _ = self.self_attention(x, x, x, attn_mask=mask)

        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series prediction.

    Supports multiple positional encoding methods and output types.

    Args:
        config: Model configuration

    Example:
        >>> config = TimeSeriesTransformerConfig(
        ...     input_features=6,
        ...     d_model=64,
        ...     n_heads=4,
        ...     encoding_type=EncodingType.ROPE
        ... )
        >>> model = TimeSeriesTransformer(config)
        >>> x = torch.randn(2, 168, 6)  # [batch, seq_len, features]
        >>> output = model(x)
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Token embedding
        self.token_embedding = TokenEmbedding(
            config.input_features, config.d_model
        )

        # Positional encoding
        self.pos_encoding = self._build_positional_encoding(config)

        # Optional calendar encoding
        if config.use_calendar:
            self.calendar_encoding = CalendarEncoding(config.d_model, config.dropout)
            self.calendar_proj = nn.Linear(2 * config.d_model, config.d_model)
        else:
            self.calendar_encoding = None

        # Optional market session encoding
        if config.use_market_session:
            self.session_encoding = MarketSessionEncoding(
                config.d_model, config.market_type, config.dropout
            )
            self.session_proj = nn.Linear(2 * config.d_model, config.d_model)
        else:
            self.session_encoding = None

        # Transformer layers
        use_rope = config.encoding_type == EncodingType.ROPE
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config, use_rope)
            for _ in range(config.n_layers)
        ])

        # Output head
        self.output_head = self._build_output_head(config)

    def _build_positional_encoding(self, config: TimeSeriesTransformerConfig):
        """Build positional encoding based on config."""
        if config.encoding_type == EncodingType.SINUSOIDAL:
            return SinusoidalPositionalEncoding(
                config.d_model, config.max_len, config.dropout
            )
        elif config.encoding_type == EncodingType.TIMESERIES_SINUSOIDAL:
            return TimeSeriesSinusoidalEncoding(
                config.d_model, dropout=config.dropout
            )
        elif config.encoding_type == EncodingType.LEARNED:
            return LearnedPositionalEncoding(
                config.d_model, config.max_len, config.dropout
            )
        elif config.encoding_type == EncodingType.ROPE:
            return None  # RoPE is applied in attention
        else:
            return None

    def _build_output_head(self, config: TimeSeriesTransformerConfig):
        """Build output head based on output type."""
        if config.output_type == OutputType.REGRESSION:
            return nn.Linear(config.d_model, config.horizon)
        elif config.output_type == OutputType.DIRECTION:
            return nn.Linear(config.d_model, config.horizon * 3)  # down/neutral/up
        elif config.output_type == OutputType.QUANTILE:
            return nn.Linear(config.d_model, config.horizon * len(config.quantiles))
        else:
            return nn.Linear(config.d_model, config.horizon)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_features]
            timestamps: Optional dict with calendar features
            mask: Optional attention mask

        Returns:
            Dictionary with predictions and optional attention weights
        """
        # Token embedding
        x = self.token_embedding(x)

        # Positional encoding (if not RoPE)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Calendar encoding
        if self.calendar_encoding is not None and timestamps is not None:
            calendar_emb = self.calendar_encoding(
                timestamps.get('dayofweek', torch.zeros_like(timestamps['hour'])),
                timestamps.get('month', torch.zeros_like(timestamps['hour'])),
                timestamps.get('quarter', torch.zeros_like(timestamps['hour'])),
                timestamps['hour'],
                torch.zeros_like(timestamps['hour'])  # session placeholder
            )
            x = self.calendar_proj(torch.cat([x, calendar_emb], dim=-1))

        # Market session encoding
        if self.session_encoding is not None and timestamps is not None:
            if self.config.market_type == "crypto":
                session_emb = self.session_encoding(timestamps['hour'])
            else:
                # Stock market requires session and time_in_session
                if 'session' not in timestamps or 'time_in_session' not in timestamps:
                    raise ValueError("Stock market session encoding requires 'session' and 'time_in_session' in timestamps")
                session_emb = self.session_encoding(
                    timestamps['hour'],
                    timestamps['session'],
                    timestamps['time_in_session']
                )
            x = self.session_proj(torch.cat([x, session_emb], dim=-1))

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Pool: use last timestep
        x = x[:, -1, :]  # [batch, d_model]

        # Output projection
        predictions = self.output_head(x)

        # Format output based on output type
        result = self._format_output(predictions)

        return result

    def _format_output(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Format predictions based on output type."""
        batch = predictions.size(0)

        if self.config.output_type == OutputType.REGRESSION:
            return {'predictions': predictions}

        elif self.config.output_type == OutputType.DIRECTION:
            # Reshape to [batch, horizon, 3]
            logits = predictions.view(batch, self.config.horizon, 3)
            probs = F.softmax(logits, dim=-1)
            return {
                'predictions': logits.argmax(dim=-1) - 1,  # -1, 0, 1
                'probabilities': probs,
                'logits': logits  # Return logits for loss computation
            }

        elif self.config.output_type == OutputType.QUANTILE:
            n_quantiles = len(self.config.quantiles)
            quantiles = predictions.view(batch, self.config.horizon, n_quantiles)
            return {
                'predictions': quantiles[:, :, n_quantiles // 2],  # Median
                'quantiles': quantiles
            }

        return {'predictions': predictions}

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss based on output type.

        Args:
            predictions: Model output dictionary
            targets: Target values

        Returns:
            Loss tensor
        """
        if self.config.output_type == OutputType.REGRESSION:
            return F.mse_loss(predictions['predictions'], targets)

        elif self.config.output_type == OutputType.DIRECTION:
            # Convert targets to direction labels
            direction = torch.sign(targets).long() + 1  # 0, 1, 2
            if targets.dim() > 1:
                direction = direction[:, 0]  # Use first horizon step
            return F.cross_entropy(
                predictions['logits'][:, 0, :],  # First horizon step (use logits, not probabilities)
                direction
            )

        elif self.config.output_type == OutputType.QUANTILE:
            return self._quantile_loss(predictions['quantiles'], targets)

        return F.mse_loss(predictions['predictions'], targets)

    def _quantile_loss(
        self,
        quantiles: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute quantile loss (pinball loss)."""
        # Ensure targets have shape [batch, horizon] for broadcasting
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)  # [batch] -> [batch, 1]

        losses = []
        for i, q in enumerate(self.config.quantiles):
            # quantiles: [batch, horizon, n_quantiles]
            # targets: [batch, horizon]
            error = targets - quantiles[:, :, i]  # [batch, horizon]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        return torch.stack(losses).mean()


if __name__ == "__main__":
    # Test the model
    print("Testing Time Series Transformer...")
    print("=" * 60)

    batch_size = 4
    seq_len = 168
    input_features = 6
    horizon = 24

    # Test different configurations
    configs = [
        ("Sinusoidal", EncodingType.SINUSOIDAL),
        ("Learned", EncodingType.LEARNED),
        ("RoPE", EncodingType.ROPE),
    ]

    x = torch.randn(batch_size, seq_len, input_features)
    timestamps = {
        'hour': torch.randint(0, 24, (batch_size, seq_len)),
        'dayofweek': torch.randint(0, 7, (batch_size, seq_len)),
        'month': torch.randint(0, 12, (batch_size, seq_len)),
        'quarter': torch.randint(0, 4, (batch_size, seq_len)),
    }

    for name, encoding_type in configs:
        print(f"\n{name} Encoding:")

        config = TimeSeriesTransformerConfig(
            input_features=input_features,
            d_model=64,
            n_heads=4,
            n_layers=2,
            encoding_type=encoding_type,
            horizon=horizon
        )

        model = TimeSeriesTransformer(config)
        output = model(x, timestamps)

        print(f"  Predictions shape: {output['predictions'].shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with calendar encoding
    print("\n\nWith Calendar Encoding:")
    config = TimeSeriesTransformerConfig(
        input_features=input_features,
        d_model=64,
        n_heads=4,
        encoding_type=EncodingType.ROPE,
        use_calendar=True,
        use_market_session=True,
        market_type='crypto',
        horizon=horizon
    )

    model = TimeSeriesTransformer(config)
    output = model(x, timestamps)
    print(f"  Predictions shape: {output['predictions'].shape}")

    # Test loss computation
    print("\n\nLoss Computation:")
    targets = torch.randn(batch_size, horizon)
    loss = model.compute_loss(output, targets)
    print(f"  MSE Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("All model tests passed!")
