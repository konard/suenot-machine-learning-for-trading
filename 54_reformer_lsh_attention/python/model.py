"""
Reformer Model Implementation in PyTorch

Provides:
- ReformerConfig: Model configuration
- ReformerModel: Main transformer model with LSH attention
- LSHAttention: Locality-sensitive hashing attention mechanism
- ReversibleBlock: Memory-efficient reversible layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class AttentionType(Enum):
    """Type of attention mechanism"""
    FULL = "full"
    LSH = "lsh"


class OutputType(Enum):
    """Type of model output"""
    REGRESSION = "regression"
    DIRECTION = "direction"
    PORTFOLIO = "portfolio"
    QUANTILE = "quantile"


@dataclass
class ReformerConfig:
    """
    Configuration for Reformer model

    Example:
        config = ReformerConfig(
            num_tickers=5,
            seq_len=4096,
            d_model=256
        )
    """
    # Architecture
    num_tickers: int = 5
    seq_len: int = 4096
    input_features: int = 6
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_layers: int = 6
    dropout: float = 0.1

    # LSH Attention
    n_buckets: int = 64
    n_rounds: int = 4
    chunk_size: int = 64
    causal: bool = True

    # Reversible layers
    use_reversible: bool = True

    # Output
    output_type: OutputType = OutputType.REGRESSION
    prediction_horizon: int = 1
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Embedding
    kernel_size: int = 3
    use_positional_encoding: bool = True

    def validate(self):
        """Validate configuration"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.kernel_size % 2 == 1, "kernel_size must be odd"
        assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"
        assert self.n_buckets > 0, "n_buckets must be positive"
        assert self.n_rounds > 0, "n_rounds must be positive"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def output_dim(self) -> int:
        if self.output_type == OutputType.REGRESSION:
            return self.num_tickers
        elif self.output_type == OutputType.DIRECTION:
            return self.num_tickers * 3
        elif self.output_type == OutputType.PORTFOLIO:
            return self.num_tickers
        elif self.output_type == OutputType.QUANTILE:
            return self.num_tickers * len(self.quantiles)
        return self.num_tickers


class TokenEmbedding(nn.Module):
    """
    Token embedding using 1D convolution

    Converts [batch, seq_len, features] to [batch, seq_len, d_model]
    """

    def __init__(self, config: ReformerConfig):
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

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
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


class TickerEncoding(nn.Module):
    """Learnable ticker-specific encoding"""

    def __init__(self, num_tickers: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tickers, d_model)

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_tickers, seq_len, d_model]
        # ticker_ids: [num_tickers]
        ticker_emb = self.embedding(ticker_ids)  # [num_tickers, d_model]
        return x + ticker_emb.unsqueeze(0).unsqueeze(2)


class LSHAttention(nn.Module):
    """
    Locality-Sensitive Hashing Attention

    Achieves O(L log L) complexity by:
    1. Hashing queries/keys into buckets
    2. Attending only within buckets
    3. Using multiple hash rounds for accuracy

    Reference: Kitaev et al., "Reformer: The Efficient Transformer" (ICLR 2020)
    """

    def __init__(self, config: ReformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_buckets = config.n_buckets
        self.n_rounds = config.n_rounds
        self.chunk_size = config.chunk_size
        self.causal = config.causal

        self.scale = math.sqrt(self.head_dim)

        # Shared QK projection (Reformer uses same vectors for Q and K)
        self.qk_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Random rotation matrices for hashing (one per round per head)
        self.register_buffer(
            'random_rotations',
            torch.randn(config.n_rounds, config.n_heads, config.head_dim, config.n_buckets // 2)
        )

    def hash_vectors(
        self,
        vectors: torch.Tensor,
        round_idx: int
    ) -> torch.Tensor:
        """
        Hash vectors using random rotation projection.

        h(x) = argmax([x @ R; -x @ R])

        This ensures vectors pointing in similar directions
        get similar hash values with high probability.

        Args:
            vectors: [batch, n_heads, seq_len, head_dim]
            round_idx: Which hashing round (0 to n_rounds-1)

        Returns:
            buckets: [batch, n_heads, seq_len] bucket assignments
        """
        batch, n_heads, seq_len, head_dim = vectors.shape

        # Get rotation matrix for this round: [n_heads, head_dim, n_buckets//2]
        rotation = self.random_rotations[round_idx]

        # Project vectors: einsum for batch matrix multiplication
        # vectors: [batch, n_heads, seq_len, head_dim]
        # rotation: [n_heads, head_dim, n_buckets//2]
        rotated = torch.einsum('bhld,hdk->bhlk', vectors, rotation)

        # Concatenate with negation for n_buckets total
        rotated = torch.cat([rotated, -rotated], dim=-1)  # [B, H, L, n_buckets]

        # Hash is the argmax bucket
        hash_values = rotated.argmax(dim=-1)  # [B, H, L]

        return hash_values

    def sort_by_bucket(
        self,
        qk: torch.Tensor,
        v: torch.Tensor,
        buckets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sort sequences by bucket assignment for efficient attention.

        Args:
            qk: [batch, n_heads, seq_len, head_dim]
            v: [batch, n_heads, seq_len, head_dim]
            buckets: [batch, n_heads, seq_len]

        Returns:
            Sorted qk, v, buckets, and unsort indices
        """
        batch, n_heads, seq_len, _ = qk.shape
        device = qk.device

        # Create sort indices: sort by (bucket, original_position) for stability
        # This ensures items in the same bucket stay in original relative order
        position_offset = torch.arange(seq_len, device=device).view(1, 1, -1)
        sort_keys = buckets * seq_len + position_offset
        sort_indices = sort_keys.argsort(dim=-1)

        # Expand indices for gathering
        expand_indices = sort_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)

        # Sort QK and V
        qk_sorted = qk.gather(2, expand_indices)
        v_sorted = v.gather(2, expand_indices)
        buckets_sorted = buckets.gather(-1, sort_indices)

        # Compute unsort indices for reversing the sort later
        unsort_indices = sort_indices.argsort(dim=-1)

        return qk_sorted, v_sorted, buckets_sorted, unsort_indices

    def chunked_attention(
        self,
        qk: torch.Tensor,
        v: torch.Tensor,
        buckets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention within chunks.

        Each query attends to keys in:
        1. Same chunk
        2. Previous chunk (lookback for better coverage)

        This is more efficient than full attention while maintaining
        good coverage thanks to the LSH sorting.

        Args:
            qk: Sorted QK tensor [batch, n_heads, seq_len, head_dim]
            v: Sorted V tensor [batch, n_heads, seq_len, head_dim]
            buckets: Sorted bucket assignments [batch, n_heads, seq_len]

        Returns:
            Attention output [batch, n_heads, seq_len, head_dim]
        """
        batch, n_heads, seq_len, head_dim = qk.shape
        device = qk.device

        # Pad sequence to be divisible by chunk_size
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            qk = F.pad(qk, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            buckets = F.pad(buckets, (0, pad_len), value=-1)  # -1 for padding

        padded_len = qk.size(2)
        n_chunks = padded_len // self.chunk_size

        # Reshape into chunks: [batch, n_heads, n_chunks, chunk_size, head_dim]
        qk = qk.view(batch, n_heads, n_chunks, self.chunk_size, head_dim)
        v = v.view(batch, n_heads, n_chunks, self.chunk_size, head_dim)
        buckets = buckets.view(batch, n_heads, n_chunks, self.chunk_size)

        # Create lookback: concatenate previous chunk with current
        # For first chunk, pad with zeros
        qk_prev = F.pad(qk[:, :, :-1], (0, 0, 0, 0, 1, 0))  # Shift and pad
        v_prev = F.pad(v[:, :, :-1], (0, 0, 0, 0, 1, 0))
        buckets_prev = F.pad(buckets[:, :, :-1], (0, 0, 1, 0), value=-1)

        # Concatenate current with previous for key-value pairs
        k = torch.cat([qk_prev, qk], dim=3)  # [B, H, C, 2*chunk, D]
        v_extended = torch.cat([v_prev, v], dim=3)
        buckets_kv = torch.cat([buckets_prev, buckets], dim=3)

        # Query is just current chunk
        q = qk  # [B, H, C, chunk, D]
        buckets_q = buckets

        # Compute attention scores
        scores = torch.einsum('bhcqd,bhckd->bhcqk', q, k) / self.scale

        # Create bucket mask: only attend to same bucket
        # buckets_q: [B, H, C, chunk]
        # buckets_kv: [B, H, C, 2*chunk]
        bucket_mask = (buckets_q.unsqueeze(-1) != buckets_kv.unsqueeze(-2))

        # Mask padding tokens (bucket == -1)
        padding_mask = (buckets_kv == -1).unsqueeze(-2)
        bucket_mask = bucket_mask | padding_mask

        # Causal mask (if enabled)
        if self.causal:
            # Each position in chunk can only attend to positions
            # at or before it (accounting for the lookback)
            causal_mask = torch.triu(
                torch.ones(self.chunk_size, 2 * self.chunk_size, device=device, dtype=torch.bool),
                diagonal=self.chunk_size + 1
            )
            bucket_mask = bucket_mask | causal_mask

        # Apply mask
        scores = scores.masked_fill(bucket_mask, float('-inf'))

        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(bucket_mask, 0.0)  # Zero out masked positions
        attn = self.dropout(attn)

        # Compute output
        out = torch.einsum('bhcqk,bhckd->bhcqd', attn, v_extended)

        # Reshape back: [batch, n_heads, seq_len, head_dim]
        out = out.view(batch, n_heads, padded_len, head_dim)

        # Remove padding
        if pad_len > 0:
            out = out[:, :, :seq_len]

        return out

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with multi-round LSH attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention info

        Returns:
            output: [batch, seq_len, d_model]
            attention_info: Optional dict with hash buckets for visualization
        """
        batch, seq_len, d_model = x.shape

        # Project to QK (shared) and V
        qk = self.qk_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Normalize QK for better hashing (unit vectors)
        qk = F.normalize(qk, dim=-1)

        # Multi-round LSH attention
        outputs = []
        all_buckets = [] if return_attention else None

        for round_idx in range(self.n_rounds):
            # Hash vectors into buckets
            buckets = self.hash_vectors(qk, round_idx)

            if return_attention:
                all_buckets.append(buckets.clone())

            # Sort by bucket
            qk_sorted, v_sorted, buckets_sorted, unsort_indices = \
                self.sort_by_bucket(qk, v, buckets)

            # Chunked attention within buckets
            out_sorted = self.chunked_attention(qk_sorted, v_sorted, buckets_sorted)

            # Unsort back to original order
            unsort_expand = unsort_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
            out = out_sorted.gather(2, unsort_expand)

            outputs.append(out)

        # Average across rounds
        output = torch.mean(torch.stack(outputs), dim=0)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.out_proj(output)

        # Prepare attention info if requested
        attention_info = None
        if return_attention:
            attention_info = {
                'buckets': torch.stack(all_buckets, dim=0),  # [n_rounds, B, H, L]
                'n_buckets': self.n_buckets,
                'n_rounds': self.n_rounds
            }

        return output, attention_info


class ChunkedFeedForward(nn.Module):
    """
    Feed-forward network processed in chunks for memory efficiency.

    For very long sequences, processing the entire FF layer at once
    can exhaust memory. Chunking processes it piece by piece.
    """

    def __init__(self, config: ReformerConfig):
        super().__init__()
        self.chunk_size = config.chunk_size * 16  # Larger chunks for FF

        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        # For short sequences, process normally
        if seq_len <= self.chunk_size:
            return self.ff(x)

        # Process in chunks for memory efficiency
        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size]
            outputs.append(self.ff(chunk))

        return torch.cat(outputs, dim=1)


class ReversibleBlock(nn.Module):
    """
    Reversible residual block for memory efficiency.

    Forward:
        Y1 = X1 + F(X2)
        Y2 = X2 + G(Y1)

    Reverse (recompute during backprop):
        X2 = Y2 - G(Y1)
        X1 = Y1 - F(X2)

    This allows recomputation of activations during backward pass,
    reducing memory usage from O(N*L*d) to O(L*d) where N is layers.

    Reference: Gomez et al., "The Reversible Residual Network" (NeurIPS 2017)
    """

    def __init__(self, f: nn.Module, g: nn.Module):
        super().__init__()
        self.f = f  # Usually attention
        self.g = g  # Usually feed-forward

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Forward pass through reversible block.

        Args:
            x1, x2: Split input tensors [batch, seq_len, d_model//2]
            return_attention: Whether to return attention info

        Returns:
            y1, y2: Output tensors
            attention_info: Optional attention information
        """
        f_out, attention_info = self.f(x2, return_attention)
        y1 = x1 + f_out
        y2 = x2 + self.g(y1)
        return y1, y2, attention_info

    def reverse(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse pass to recover inputs from outputs.

        Used during backward pass to recompute activations.
        """
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)[0]
        return x1, x2


class EncoderLayer(nn.Module):
    """
    Reformer encoder layer with LSH attention and optional reversibility.
    """

    def __init__(self, config: ReformerConfig):
        super().__init__()
        self.use_reversible = config.use_reversible

        self.attention = LSHAttention(config)
        self.feed_forward = ChunkedFeedForward(config)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        if config.use_reversible:
            # Wrap attention to include norm
            self.f = lambda x, ret_attn=False: (
                self.dropout(self.attention(self.norm1(x), ret_attn)[0]),
                self.attention(self.norm1(x), ret_attn)[1]
            )
            self.g = lambda x: self.dropout(self.feed_forward(self.norm2(x)))

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention: Whether to return attention info

        Returns:
            output: [batch, seq_len, d_model]
            attention_info: Optional attention information
        """
        # Standard (non-reversible) forward pass
        attn_out, attention_info = self.attention(self.norm1(x), return_attention)
        x = x + self.dropout(attn_out)

        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x, attention_info


class ReformerModel(nn.Module):
    """
    Reformer: Efficient Transformer with LSH Attention

    Designed for processing long financial time series with
    O(L log L) complexity instead of O(L^2).

    Example:
        config = ReformerConfig(num_tickers=5, seq_len=4096)
        model = ReformerModel(config)

        x = torch.randn(2, 5, 4096, 6)  # [batch, tickers, seq_len, features]
        output = model(x)
        print(output['predictions'].shape)  # [2, 5]
    """

    def __init__(self, config: ReformerConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Embedding layers
        self.token_embedding = TokenEmbedding(config)
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.seq_len * 2, config.dropout
        ) if config.use_positional_encoding else None
        self.ticker_encoding = TickerEncoding(config.num_tickers, config.d_model)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Output head
        self.output_head = self._build_output_head(config)

        # Register ticker IDs
        self.register_buffer('ticker_ids', torch.arange(config.num_tickers))

    def _build_output_head(self, config: ReformerConfig) -> nn.Module:
        """Build output projection layer based on output type"""
        if config.output_type == OutputType.QUANTILE:
            return nn.Linear(config.d_model, len(config.quantiles))
        elif config.output_type == OutputType.DIRECTION:
            return nn.Linear(config.d_model, 3)
        else:
            return nn.Linear(config.d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> dict:
        """
        Forward pass through Reformer.

        Args:
            x: Input tensor [batch, num_tickers, seq_len, features]
            return_attention: Whether to return attention info

        Returns:
            Dictionary with:
                - predictions: [batch, output_dim]
                - attention_info: Optional LSH attention information
                - confidence: Optional confidence scores (for quantile output)
        """
        batch, num_tickers, seq_len, features = x.shape

        # Token embedding for each ticker
        x_embedded = []
        for t in range(num_tickers):
            emb = self.token_embedding(x[:, t])  # [batch, seq_len, d_model]
            x_embedded.append(emb)
        x = torch.stack(x_embedded, dim=1)  # [batch, num_tickers, seq_len, d_model]

        # Add positional encoding
        if self.positional_encoding is not None:
            for t in range(num_tickers):
                x[:, t] = self.positional_encoding(x[:, t])

        # Add ticker encoding
        x = self.ticker_encoding(x, self.ticker_ids)

        # Process each ticker through encoder
        all_attention = {}
        encoded = []

        for t in range(num_tickers):
            ticker_x = x[:, t]  # [batch, seq_len, d_model]

            for i, layer in enumerate(self.encoder_layers):
                ticker_x, attn_info = layer(ticker_x, return_attention)
                if attn_info and return_attention:
                    all_attention[f'ticker_{t}_layer_{i}'] = attn_info

            ticker_x = self.final_norm(ticker_x)
            encoded.append(ticker_x)

        # Stack encoded outputs: [batch, num_tickers, seq_len, d_model]
        x = torch.stack(encoded, dim=1)

        # Pool: take last timestep for prediction
        x = x[:, :, -1, :]  # [batch, num_tickers, d_model]

        # Output projection
        predictions = self._compute_output(x)

        result = {
            'predictions': predictions,
            'attention_info': all_attention if return_attention else None
        }

        # Add confidence for quantile regression
        if self.config.output_type == OutputType.QUANTILE:
            result['confidence'] = self._compute_confidence(predictions)

        return result

    def _compute_output(self, x: torch.Tensor) -> torch.Tensor:
        """Compute output predictions based on output type"""
        batch, num_tickers, d_model = x.shape

        if self.config.output_type == OutputType.PORTFOLIO:
            # Portfolio weights sum to 1
            logits = self.output_head(x).squeeze(-1)  # [batch, num_tickers]
            return F.softmax(logits, dim=-1)

        elif self.config.output_type == OutputType.DIRECTION:
            # Per-ticker classification: down, neutral, up
            logits = self.output_head(x)  # [batch, num_tickers, 3]
            probs = F.softmax(logits, dim=-1)
            return probs.view(batch, -1)  # [batch, num_tickers * 3]

        elif self.config.output_type == OutputType.QUANTILE:
            # Per-ticker quantile prediction
            quantiles = self.output_head(x)  # [batch, num_tickers, num_quantiles]
            return quantiles.view(batch, -1)

        else:  # REGRESSION
            return self.output_head(x).squeeze(-1)  # [batch, num_tickers]

    def _compute_confidence(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute confidence from quantile predictions"""
        batch = predictions.size(0)
        num_quantiles = len(self.config.quantiles)
        num_tickers = self.config.num_tickers

        predictions = predictions.view(batch, num_tickers, num_quantiles)
        interval_width = (predictions[:, :, -1] - predictions[:, :, 0]).abs()
        confidence = 1.0 / (1.0 + interval_width)

        return confidence


if __name__ == "__main__":
    # Test the model
    print("Testing Reformer model...")

    config = ReformerConfig(
        num_tickers=5,
        seq_len=512,  # Shorter for testing
        input_features=6,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_buckets=32,
        n_rounds=2,
        chunk_size=32
    )

    model = ReformerModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 5, 512, 6)
    output = model(x, return_attention=True)

    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Attention info available: {output['attention_info'] is not None}")

    # Test different output types
    for output_type in OutputType:
        config.output_type = output_type
        model = ReformerModel(config)
        output = model(x)
        print(f"{output_type.value}: predictions shape = {output['predictions'].shape}")

    print("\nAll tests passed!")
