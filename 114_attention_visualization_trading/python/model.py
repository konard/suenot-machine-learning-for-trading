"""Transformer model with attention weight extraction for financial prediction.

Implements a Transformer encoder that returns attention weights alongside
predictions, enabling visualization and interpretation of model decisions.

References:
    Vaswani et al. (2017) - "Attention Is All You Need"
    Kobayashi et al. (2020) - "Attention is Not Only a Weight"
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention with attention weight extraction.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention weight extraction.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
            - output: (batch, seq_len, d_model)
            - attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention weights (this is what we visualize)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights_dropped = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights_dropped, V)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with attention extraction.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int = 256, dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention extraction.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual
        attn_out, attention_weights = self.attention(x, mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x, attention_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AttentionTransformer(nn.Module):
    """Transformer model for financial prediction with attention extraction.

    This model processes OHLCV and technical indicator data to predict:
    - Expected return (regression)
    - Direction probability (classification)
    - Volatility forecast (regression)

    Additionally, it returns attention weights from all layers for visualization.

    Args:
        input_dim: Number of input features (e.g., OHLCV + indicators)
        d_model: Model dimension
        n_heads: Number of attention heads per layer
        n_layers: Number of transformer encoder layers
        output_dim: Number of output targets (default 3)
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        return_attention: Whether to return attention weights
    """

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        output_dim: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        return_attention: bool = True,
    ):
        super().__init__()
        self.return_attention = return_attention
        self.n_layers = n_layers

        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output heads
        self.return_head = nn.Linear(d_model, 1)
        self.direction_head = nn.Linear(d_model, 1)
        self.volatility_head = nn.Linear(d_model, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass through the transformer.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Tuple of (pred_return, pred_direction, pred_volatility, attention_weights)
            - pred_return: (batch, 1) predicted return
            - pred_direction: (batch, 1) direction probability
            - pred_volatility: (batch, 1) predicted volatility
            - attention_weights: dict with keys 'layer_0', 'layer_1', etc.
              Each value is (batch, n_heads, seq_len, seq_len)
              Only returned if return_attention=True
        """
        # Project input to model dimension
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        # Process through transformer layers, collecting attention
        attention_weights = {}
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, mask)
            if self.return_attention:
                attention_weights[f"layer_{i}"] = attn

        # Use last timestep for prediction
        last_hidden = x[:, -1, :]

        # Prediction heads
        pred_return = self.return_head(last_hidden)
        pred_direction = torch.sigmoid(self.direction_head(last_hidden))
        pred_volatility = F.softplus(self.volatility_head(last_hidden))

        if self.return_attention:
            return pred_return, pred_direction, pred_volatility, attention_weights
        return pred_return, pred_direction, pred_volatility, None


class AttentionTrainer:
    """Trainer for the AttentionTransformer model.

    Supports multi-task learning with uncertainty-based loss weighting.
    """

    def __init__(
        self,
        model: AttentionTransformer,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

        # Learnable loss weights (uncertainty weighting)
        self.log_var_return = nn.Parameter(torch.zeros(1, device=device))
        self.log_var_direction = nn.Parameter(torch.zeros(1, device=device))
        self.log_var_volatility = nn.Parameter(torch.zeros(1, device=device))

        self.loss_optimizer = torch.optim.Adam(
            [self.log_var_return, self.log_var_direction, self.log_var_volatility],
            lr=1e-3,
        )

    def compute_loss(
        self,
        pred_return: torch.Tensor,
        pred_direction: torch.Tensor,
        pred_volatility: torch.Tensor,
        target_return: torch.Tensor,
        target_direction: torch.Tensor,
        target_volatility: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-task loss with uncertainty weighting."""
        loss_ret = F.mse_loss(pred_return.squeeze(), target_return)
        loss_dir = F.binary_cross_entropy(
            pred_direction.squeeze(), target_direction
        )
        loss_vol = F.mse_loss(pred_volatility.squeeze(), target_volatility)

        # Uncertainty-weighted combination
        precision_ret = torch.exp(-self.log_var_return)
        precision_dir = torch.exp(-self.log_var_direction)
        precision_vol = torch.exp(-self.log_var_volatility)

        total_loss = (
            precision_ret * loss_ret + self.log_var_return
            + precision_dir * loss_dir + self.log_var_direction
            + precision_vol * loss_vol + self.log_var_volatility
        )
        return total_loss

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            features, target_ret, target_dir, target_vol = [
                b.to(self.device) for b in batch
            ]

            self.optimizer.zero_grad()
            self.loss_optimizer.zero_grad()

            pred_ret, pred_dir, pred_vol, _ = self.model(features)

            loss = self.compute_loss(
                pred_ret, pred_dir, pred_vol,
                target_ret, target_dir, target_vol,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.loss_optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()

        return {
            "loss": total_loss / max(num_batches, 1),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> dict:
        """Evaluate model on validation/test data."""
        self.model.eval()
        all_pred_ret, all_target_ret = [], []
        all_pred_dir, all_target_dir = [], []
        all_pred_vol, all_target_vol = [], []

        for batch in dataloader:
            features, target_ret, target_dir, target_vol = [
                b.to(self.device) for b in batch
            ]
            pred_ret, pred_dir, pred_vol, _ = self.model(features)

            all_pred_ret.append(pred_ret.squeeze().cpu())
            all_target_ret.append(target_ret.cpu())
            all_pred_dir.append(pred_dir.squeeze().cpu())
            all_target_dir.append(target_dir.cpu())
            all_pred_vol.append(pred_vol.squeeze().cpu())
            all_target_vol.append(target_vol.cpu())

        pred_ret = torch.cat(all_pred_ret)
        target_ret = torch.cat(all_target_ret)
        pred_dir = torch.cat(all_pred_dir)
        target_dir = torch.cat(all_target_dir)
        pred_vol = torch.cat(all_pred_vol)
        target_vol = torch.cat(all_target_vol)

        mse_ret = F.mse_loss(pred_ret, target_ret).item()
        mae_ret = F.l1_loss(pred_ret, target_ret).item()

        direction_correct = ((pred_dir > 0.5).float() == target_dir).float()
        accuracy = direction_correct.mean().item()

        mse_vol = F.mse_loss(pred_vol, target_vol).item()

        return {
            "mse_return": mse_ret,
            "mae_return": mae_ret,
            "direction_accuracy": accuracy,
            "mse_volatility": mse_vol,
        }

    @torch.no_grad()
    def get_attention_weights(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights for visualization.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Dictionary mapping layer names to attention weight tensors
        """
        self.model.eval()
        x = x.to(self.device)
        _, _, _, attention_weights = self.model(x)
        return attention_weights
