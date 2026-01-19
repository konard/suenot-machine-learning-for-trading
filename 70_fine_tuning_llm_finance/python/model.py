"""
LoRA, QLoRA, and Prefix-Tuning implementations for financial LLMs.

This module provides parameter-efficient fine-tuning methods for adapting
pre-trained language models to financial tasks like sentiment analysis
and market prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import math


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["query", "value"]


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (quantized LoRA)."""
    bits: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"


@dataclass
class PrefixConfig:
    """Configuration for prefix-tuning."""
    num_prefix_tokens: int = 20
    prefix_projection: bool = True
    prefix_hidden_dim: int = 512


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning.

    Instead of updating the full weight matrix W, LoRA learns a low-rank
    decomposition: W' = W + BA, where B and A are much smaller matrices.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of the low-rank decomposition
        alpha: Scaling factor (alpha/rank)
        dropout: Dropout probability for regularization

    Example:
        >>> layer = LoRALayer(768, 768, rank=8)
        >>> x = torch.randn(32, 768)
        >>> output = layer(x)
        >>> print(output.shape)
        torch.Size([32, 768])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        base_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Base weight (frozen, from pre-trained model)
        if base_weight is not None:
            self.weight = nn.Parameter(base_weight.clone(), requires_grad=False)
        else:
            self.weight = nn.Parameter(
                torch.randn(out_features, in_features) * 0.02,
                requires_grad=False
            )

        # LoRA trainable matrices
        # A: down-projection, B: up-projection
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize: A with Kaiming, B with zeros
        # Starting from zero ensures initial output matches pre-trained model
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self._enabled = True

    def enable_lora(self):
        """Enable LoRA adaptation."""
        self._enabled = True

    def disable_lora(self):
        """Disable LoRA adaptation (use base model only)."""
        self._enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (batch, ..., in_features)

        Returns:
            Output tensor of shape (batch, ..., out_features)
        """
        # Base transformation (frozen)
        result = F.linear(x, self.weight)

        # LoRA adaptation
        if self._enabled:
            # Apply dropout, then low-rank transformation
            lora_input = self.dropout(x)
            # x @ A.T @ B.T = (x @ A.T) @ B.T
            lora_output = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
            result = result + self.scaling * lora_output

        return result

    def merge_weights(self) -> torch.Tensor:
        """
        Merge LoRA weights into base weights for inference.

        Returns:
            Merged weight matrix W + (alpha/r) * BA
        """
        delta_w = self.scaling * (self.lora_B @ self.lora_A)
        return self.weight + delta_w

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}"
        )


class FinancialSentimentLoRA(nn.Module):
    """
    Financial sentiment classifier with LoRA-adapted transformer layers.

    This model uses LoRA to adapt a pre-trained transformer for financial
    sentiment classification (Bullish, Neutral, Bearish).

    Args:
        base_dim: Hidden dimension of the transformer
        num_heads: Number of attention heads
        lora_rank: Rank for LoRA adaptation
        num_classes: Number of sentiment classes
        dropout: Dropout probability

    Example:
        >>> model = FinancialSentimentLoRA(base_dim=768, lora_rank=8)
        >>> hidden_states = torch.randn(32, 128, 768)
        >>> logits = model(hidden_states)
        >>> print(logits.shape)
        torch.Size([32, 3])
    """

    def __init__(
        self,
        base_dim: int = 768,
        num_heads: int = 12,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.base_dim = base_dim
        self.num_heads = num_heads
        self.head_dim = base_dim // num_heads

        # LoRA-adapted attention projections
        self.query_lora = LoRALayer(base_dim, base_dim, lora_rank, lora_alpha, dropout)
        self.key_lora = LoRALayer(base_dim, base_dim, lora_rank, lora_alpha, dropout)
        self.value_lora = LoRALayer(base_dim, base_dim, lora_rank, lora_alpha, dropout)
        self.output_lora = LoRALayer(base_dim, base_dim, lora_rank, lora_alpha, dropout)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(base_dim)
        self.layer_norm2 = nn.LayerNorm(base_dim)

        # Feed-forward network (fully trainable)
        self.ffn = nn.Sequential(
            nn.Linear(base_dim, base_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_dim * 4, base_dim),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(base_dim, base_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_dim // 2, num_classes)
        )

        # Sentiment labels
        self.labels = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

    def attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-head self-attention with LoRA-adapted projections.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V using LoRA layers
        query = self.query_lora(hidden_states)
        key = self.key_lora(hidden_states)
        value = self.value_lora(hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        attention_probs = F.softmax(scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.base_dim)

        # Output projection
        output = self.output_lora(context)

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for sentiment classification.

        Args:
            hidden_states: Transformer hidden states (batch, seq_len, dim)
            attention_mask: Optional attention mask

        Returns:
            Classification logits (batch, num_classes)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output

        # FFN with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output

        # Pool: use [CLS] token (first position) or mean pooling
        pooled = hidden_states[:, 0]  # CLS token

        # Classification
        logits = self.classifier(pooled)

        return logits

    def predict(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """
        Get prediction with confidence scores.

        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        with torch.no_grad():
            logits = self.forward(hidden_states)
            probs = F.softmax(logits, dim=-1)
            confidence, prediction = probs.max(dim=-1)

        return {
            "prediction": self.labels[prediction.item()],
            "confidence": confidence.item(),
            "probabilities": {
                self.labels[i]: probs[0, i].item()
                for i in range(len(self.labels))
            }
        }

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


class PrefixTuningLayer(nn.Module):
    """
    Prefix-tuning layer for task-specific LLM adaptation.

    Instead of modifying model weights, prefix-tuning prepends learnable
    continuous vectors (prefixes) to the input, steering model behavior.

    Args:
        num_prefix_tokens: Number of prefix tokens to prepend
        hidden_dim: Hidden dimension of the transformer
        num_layers: Number of transformer layers
        prefix_projection: Whether to use MLP projection
        prefix_hidden_dim: Hidden dimension for MLP projection

    Example:
        >>> prefix = PrefixTuningLayer(num_prefix_tokens=20, hidden_dim=768)
        >>> keys, values = prefix(batch_size=32)
        >>> print(keys.shape)
        torch.Size([12, 32, 20, 768])
    """

    def __init__(
        self,
        num_prefix_tokens: int = 20,
        hidden_dim: int = 768,
        num_layers: int = 12,
        prefix_projection: bool = True,
        prefix_hidden_dim: int = 512
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.prefix_projection = prefix_projection

        if prefix_projection:
            # Reparameterization: embedding -> MLP -> prefix
            self.prefix_embedding = nn.Embedding(
                num_prefix_tokens,
                prefix_hidden_dim
            )
            self.prefix_mlp = nn.Sequential(
                nn.Linear(prefix_hidden_dim, prefix_hidden_dim),
                nn.Tanh(),
                nn.Linear(prefix_hidden_dim, num_layers * 2 * hidden_dim)
            )
        else:
            # Direct prefix parameters
            self.prefix_embedding = nn.Embedding(
                num_prefix_tokens,
                num_layers * 2 * hidden_dim
            )
            self.prefix_mlp = nn.Identity()

        # Token indices for forward pass
        self.register_buffer(
            "prefix_tokens",
            torch.arange(num_prefix_tokens)
        )

    def forward(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prefix key-value pairs for all layers.

        Args:
            batch_size: Current batch size

        Returns:
            Tuple of (prefix_keys, prefix_values) tensors
            Each has shape (num_layers, batch_size, num_prefix_tokens, hidden_dim)
        """
        # Expand prefix tokens for batch
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)

        # Get prefix embeddings
        prefix_embeds = self.prefix_embedding(prefix_tokens)

        # Project through MLP
        prefix_output = self.prefix_mlp(prefix_embeds)

        # Reshape: (batch, num_prefix, layers * 2 * hidden)
        # -> (batch, num_prefix, layers, 2, hidden)
        prefix_output = prefix_output.view(
            batch_size,
            self.num_prefix_tokens,
            self.num_layers,
            2,
            self.hidden_dim
        )

        # Reorder to (layers, 2, batch, num_prefix, hidden)
        prefix_output = prefix_output.permute(2, 3, 0, 1, 4)

        # Split into keys and values
        prefix_keys = prefix_output[:, 0]    # (layers, batch, prefix, hidden)
        prefix_values = prefix_output[:, 1]

        return prefix_keys, prefix_values

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FinancialPrefixClassifier(nn.Module):
    """
    Financial text classifier using prefix-tuning.

    Adapts a frozen transformer for financial classification by learning
    task-specific prefix tokens that steer model behavior.

    Args:
        hidden_dim: Hidden dimension of the transformer
        num_layers: Number of transformer layers
        num_prefix_tokens: Number of prefix tokens
        num_classes: Number of output classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_prefix_tokens: int = 20,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        # Prefix tuning layer
        self.prefix_tuning = PrefixTuningLayer(
            num_prefix_tokens=num_prefix_tokens,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.num_prefix_tokens = num_prefix_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with prefix-augmented attention.

        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask

        Returns:
            Classification logits
        """
        batch_size = hidden_states.size(0)

        # Get prefix key-values (would be used in actual transformer)
        prefix_keys, prefix_values = self.prefix_tuning(batch_size)

        # For this simplified version, we just classify based on input
        # In full implementation, prefix would be prepended to KV cache
        pooled = hidden_states[:, 0]  # CLS token
        logits = self.classifier(pooled)

        return logits


def create_lora_model(
    base_model: nn.Module,
    config: LoRAConfig,
    freeze_base: bool = True
) -> nn.Module:
    """
    Add LoRA adapters to an existing model.

    Args:
        base_model: Pre-trained model to adapt
        config: LoRA configuration
        freeze_base: Whether to freeze base model parameters

    Returns:
        Model with LoRA adapters
    """
    if freeze_base:
        for param in base_model.parameters():
            param.requires_grad = False

    # This is a simplified version - full implementation would
    # traverse the model and replace target modules with LoRA versions
    return base_model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {
        "trainable": trainable,
        "total": total,
        "frozen": total - trainable,
        "trainable_percent": 100 * trainable / total if total > 0 else 0
    }


if __name__ == "__main__":
    # Example usage
    print("Testing LoRA Layer...")
    layer = LoRALayer(768, 768, rank=8)
    x = torch.randn(2, 10, 768)
    output = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("\nTesting Financial Sentiment LoRA...")
    model = FinancialSentimentLoRA(base_dim=768, lora_rank=8)
    hidden = torch.randn(2, 128, 768)
    logits = model(hidden)
    print(f"Hidden states shape: {hidden.shape}")
    print(f"Logits shape: {logits.shape}")

    params = count_parameters(model)
    print(f"\nParameter counts:")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable %: {params['trainable_percent']:.2f}%")

    print("\nTesting Prefix Tuning...")
    prefix = PrefixTuningLayer(num_prefix_tokens=20, hidden_dim=768)
    keys, values = prefix(batch_size=2)
    print(f"Prefix keys shape: {keys.shape}")
    print(f"Prefix values shape: {values.shape}")
