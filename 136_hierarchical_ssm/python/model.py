"""Hierarchical State Space Model (HiSS) for financial time series prediction.

Implements a multi-scale SSM architecture that processes financial data at
multiple temporal resolutions simultaneously, inspired by:
  Bhirangi et al. (2024) - "Hierarchical State Space Models for Continuous
  Sequence-to-Sequence Modeling" (arXiv:2402.10211)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMLayer(nn.Module):
    """Single State Space Model layer with discretized continuous dynamics.

    Implements the discrete SSM:
        x_k = A_bar * x_{k-1} + B_bar * u_k
        y_k = C * x_k + D * u_k

    Uses a diagonal state matrix for computational efficiency.
    """

    def __init__(self, input_dim: int, state_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim

        # Learnable SSM parameters
        # Initialize A with negative values for stability (HiPPO-inspired)
        self.log_A = nn.Parameter(
            torch.log(0.5 * torch.ones(state_dim) + 0.5 * torch.rand(state_dim))
        )
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.zeros(output_dim, input_dim))

        # Learnable discretization step
        self.log_delta = nn.Parameter(torch.log(torch.tensor(0.01)))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SSM layer.

        Args:
            u: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = u.shape

        # Compute discretized parameters
        A = -torch.exp(self.log_A)  # Negative diagonal for stability
        delta = torch.exp(self.log_delta)

        # Zero-order hold discretization
        A_bar = torch.exp(delta * A)  # (state_dim,)
        B_bar = (1.0 / A) * (A_bar - 1.0) * delta  # (state_dim,)
        B_bar = B_bar.unsqueeze(1) * self.B  # (state_dim, input_dim)

        # Sequential scan
        x = torch.zeros(batch_size, self.state_dim, device=u.device)
        outputs = []

        for t in range(seq_len):
            x = A_bar.unsqueeze(0) * x + (u[:, t, :] @ B_bar.T)
            y = x @ self.C.T + u[:, t, :] @ self.D.T
            outputs.append(y)

        return torch.stack(outputs, dim=1)


class SSMBlock(nn.Module):
    """SSM block with layer normalization, activation, and residual connection."""

    def __init__(self, dim: int, state_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = SSMLayer(dim, state_dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x + residual


class HierarchicalSSM(nn.Module):
    """Hierarchical State Space Model for multi-scale financial prediction.

    Stacks SSM blocks at multiple temporal resolutions to capture patterns
    from tick-level microstructure to weekly macro trends.

    Args:
        input_dim: Number of input features (e.g., OHLCV + indicators)
        hidden_dim: Hidden dimension for SSM blocks
        output_dim: Number of prediction targets
        num_levels: Number of hierarchy levels
        downsample_factors: List of downsampling factors between levels.
            Length must be num_levels - 1.
        ssm_state_dim: Dimension of the SSM latent state
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 3,
        num_levels: int = 3,
        downsample_factors: Optional[List[int]] = None,
        ssm_state_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.downsample_factors = downsample_factors or [4] * (num_levels - 1)
        assert len(self.downsample_factors) == num_levels - 1

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # SSM blocks at each hierarchy level
        self.ssm_blocks = nn.ModuleList([
            SSMBlock(hidden_dim, ssm_state_dim, dropout)
            for _ in range(num_levels)
        ])

        # Downsampling layers (average pooling via strided conv)
        self.downsamplers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, stride=k)
            for k in self.downsample_factors
        ])

        # Upsampling layers (transposed convolution)
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=k, stride=k)
            for k in self.downsample_factors
        ])

        # Fusion layer: combines features from all levels
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Prediction heads
        self.return_head = nn.Linear(hidden_dim, 1)  # Return prediction
        self.direction_head = nn.Linear(hidden_dim, 1)  # Direction (sigmoid)
        self.volatility_head = nn.Linear(hidden_dim, 1)  # Volatility prediction

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the hierarchical SSM.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Tuple of (predicted_return, direction_prob, predicted_volatility),
            each of shape (batch, 1)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to hidden dimension
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Process each hierarchy level
        level_outputs = []

        # Level 0: full resolution
        h0 = self.ssm_blocks[0](h)
        level_outputs.append(h0)

        current = h0
        for level in range(1, self.num_levels):
            # Downsample: (batch, seq, dim) -> (batch, dim, seq) -> conv -> (batch, dim, seq/k)
            ds = self.downsamplers[level - 1](
                current.transpose(1, 2)
            ).transpose(1, 2)

            # Process at this resolution
            hl = self.ssm_blocks[level](ds)
            level_outputs.append(hl)
            current = hl

        # Upsample coarser levels back to original resolution
        upsampled = [level_outputs[0]]
        for level in range(1, self.num_levels):
            up = self.upsamplers[level - 1](
                level_outputs[level].transpose(1, 2)
            ).transpose(1, 2)

            # Match sequence length (handle rounding)
            target_len = level_outputs[0].shape[1]
            if up.shape[1] > target_len:
                up = up[:, :target_len, :]
            elif up.shape[1] < target_len:
                pad = target_len - up.shape[1]
                up = F.pad(up, (0, 0, 0, pad))
            upsampled.append(up)

        # Take the last timestep from each level for prediction
        last_features = [u[:, -1, :] for u in upsampled]

        # Fuse features from all levels
        fused = torch.cat(last_features, dim=-1)  # (batch, hidden_dim * num_levels)
        fused = self.fusion(fused)  # (batch, hidden_dim)

        # Prediction heads
        pred_return = self.return_head(fused)  # (batch, 1)
        direction = torch.sigmoid(self.direction_head(fused))  # (batch, 1)
        pred_vol = F.softplus(self.volatility_head(fused))  # (batch, 1), positive

        return pred_return, direction, pred_vol


class HiSSTrainer:
    """Trainer for the Hierarchical SSM model.

    Supports multi-task learning with uncertainty-based loss weighting.
    """

    def __init__(
        self,
        model: HierarchicalSSM,
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

        # Learnable loss weights (uncertainty weighting, Kendall et al. 2018)
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

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> dict:
        """Train for one epoch.

        Args:
            dataloader: DataLoader yielding (features, return, direction, volatility)

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            features, target_ret, target_dir, target_vol = [
                b.to(self.device) for b in batch
            ]

            self.optimizer.zero_grad()
            self.loss_optimizer.zero_grad()

            pred_ret, pred_dir, pred_vol = self.model(features)

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
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> dict:
        """Evaluate model on validation/test data.

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_pred_ret, all_target_ret = [], []
        all_pred_dir, all_target_dir = [], []
        all_pred_vol, all_target_vol = [], []

        for batch in dataloader:
            features, target_ret, target_dir, target_vol = [
                b.to(self.device) for b in batch
            ]
            pred_ret, pred_dir, pred_vol = self.model(features)

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
