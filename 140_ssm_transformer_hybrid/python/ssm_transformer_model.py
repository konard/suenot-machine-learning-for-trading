"""
SSM-Transformer Hybrid model for trading.

This module provides:
- SSMBlock: Mamba-style selective state space layer
- TransformerBlock: Multi-head self-attention layer
- SSMTransformerHybrid: Complete hybrid architecture with multi-task outputs
- SSMHybridTrainer: Training loop with uncertainty-based task weighting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSMBlock(nn.Module):
    """
    Simplified Mamba-style Selective State Space block.

    Implements input-dependent SSM with selective scan:
    - Input projection to create B, C, delta parameters
    - Discretization with softplus for delta
    - Sequential scan through the state space
    - Output gating with SiLU activation
    """

    def __init__(self, d_model: int, state_size: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size

        # Input projection: expand then contract
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)

        # SSM parameters (input-dependent)
        self.B_proj = nn.Linear(d_model, state_size, bias=False)
        self.C_proj = nn.Linear(d_model, state_size, bias=False)
        self.delta_proj = nn.Linear(d_model, d_model, bias=True)

        # Diagonal A matrix (log-space for stability)
        A = torch.arange(1, state_size + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_model, -1).clone())

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SSM block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        batch, seq_len, _ = x.shape

        # Input projection and split into x_main and gate
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)

        # Compute input-dependent SSM parameters
        B = self.B_proj(x_main)  # (batch, seq_len, state_size)
        C = self.C_proj(x_main)  # (batch, seq_len, state_size)
        delta = F.softplus(self.delta_proj(x_main))  # (batch, seq_len, d_model)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_model, state_size)

        # Selective scan
        y = self._selective_scan(x_main, delta, A, B, C)

        # Gated output
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y + residual

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform selective scan (sequential processing through SSM).

        Args:
            x: Input (batch, seq_len, d_model)
            delta: Step sizes (batch, seq_len, d_model)
            A: State matrix (d_model, state_size)
            B: Input-dependent B (batch, seq_len, state_size)
            C: Input-dependent C (batch, seq_len, state_size)

        Returns:
            Output (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # Discretize: dA = exp(delta * A), dB = delta * B
        delta_A = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq, d_model, state)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq, d_model, state)

        # Sequential scan
        h = torch.zeros(batch, d_model, self.state_size, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, d_model)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head self-attention and FFN.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm
        residual = x
        x_norm = self.attn_norm(x)

        # Causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x = residual + attn_out

        # FFN with pre-norm
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))

        return x


class SSMTransformerHybrid(nn.Module):
    """
    Hybrid model combining SSM (Mamba-style) and Transformer layers.

    Architecture follows the Jamba pattern: mostly SSM layers with
    periodic Transformer layers interleaved.

    Multi-task outputs:
    - direction: Binary classification (up/down)
    - volatility: Regression (next-bar volatility)
    - return_mag: Regression (return magnitude)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_ssm_layers: int = 6,
        n_attn_layers: int = 2,
        ssm_state_size: int = 16,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_outputs: int = 3,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Build interleaved layers
        total_layers = n_ssm_layers + n_attn_layers
        attn_interval = max(1, total_layers // max(1, n_attn_layers))

        self.layers = nn.ModuleList()
        attn_placed = 0
        for i in range(total_layers):
            if (i + 1) % attn_interval == 0 and attn_placed < n_attn_layers:
                self.layers.append(TransformerBlock(d_model, n_heads, d_ff, dropout))
                attn_placed += 1
            else:
                self.layers.append(SSMBlock(d_model, ssm_state_size, dropout))

        self.final_norm = nn.LayerNorm(d_model)

        # Task-specific output heads
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Uncertainty-based task weighting (log variance)
        self.log_vars = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid model.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Dictionary with keys: direction, volatility, return_mag
        """
        # Project input
        x = self.input_proj(x)

        # Pass through interleaved SSM/Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # Use last timestep for prediction
        x_last = x[:, -1, :]

        return {
            "direction": torch.sigmoid(self.direction_head(x_last)).squeeze(-1),
            "volatility": F.softplus(self.volatility_head(x_last)).squeeze(-1),
            "return_mag": self.return_head(x_last).squeeze(-1),
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SSMHybridTrainer:
    """
    Trainer for the SSM-Transformer Hybrid model.

    Features:
    - Multi-task loss with uncertainty weighting
    - Gradient clipping
    - Learning rate scheduling
    - Validation-based early stopping
    """

    def __init__(
        self,
        model: SSMTransformerHybrid,
        learning_rate: float = 1e-3,
        task_weights: Optional[Dict[str, float]] = None,
        clip_grad_norm: float = 1.0,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.task_weights = task_weights or {
            "direction": 1.0,
            "volatility": 0.5,
            "return_mag": 0.5,
        }
        self.clip_grad_norm = clip_grad_norm
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute multi-task loss with uncertainty weighting.

        Uses the learned log_vars from the model for automatic task balancing.
        """
        losses = {}
        losses["direction"] = F.binary_cross_entropy(
            outputs["direction"], targets["direction"]
        )
        losses["volatility"] = F.mse_loss(outputs["volatility"], targets["volatility"])
        losses["return_mag"] = F.mse_loss(outputs["return_mag"], targets["return_mag"])

        # Uncertainty-weighted combination
        total = torch.tensor(0.0, device=outputs["direction"].device)
        for i, (task_name, loss) in enumerate(losses.items()):
            precision = torch.exp(-self.model.log_vars[i])
            w = self.task_weights.get(task_name, 1.0)
            total = total + w * precision * loss + self.model.log_vars[i]

        return total

    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            features, targets = batch
            self.optimizer.zero_grad()

            outputs = self.model(features)
            loss = self.compute_loss(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            features, targets = batch
            outputs = self.model(features)
            loss = self.compute_loss(outputs, targets)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience

        Returns:
            Dictionary with training and validation loss history
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
