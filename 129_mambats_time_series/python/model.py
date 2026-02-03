"""
MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting

This module implements the full MambaTS architecture, combining:
- Temporal patch embedding
- Variable-aware scanning
- Stacked Mamba layers
- Multi-horizon prediction heads

Based on the paper "MambaTS: Improved Selective State Space Models for Long-term Forecasting"
(arXiv:2405.16440, 2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from einops import rearrange, repeat

from .mamba_block import MambaBlock, VariableAwareScanning


class TemporalPatchEmbedding(nn.Module):
    """
    Converts time series into patches for efficient processing.

    Instead of processing individual time steps, this module groups
    consecutive time steps into patches, reducing sequence length
    while preserving temporal information.
    """

    def __init__(
        self,
        n_variables: int,
        patch_size: int,
        d_model: int,
        stride: Optional[int] = None,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_variables = n_variables
        self.patch_size = patch_size
        self.stride = stride or patch_size  # Non-overlapping by default
        self.d_model = d_model
        self.padding_mode = padding_mode

        # Projection from patch to embedding
        self.projection = nn.Linear(patch_size * n_variables, d_model)

        # Learnable position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_variables)

        Returns:
            Tuple of:
            - Embedded patches of shape (batch, n_patches, d_model)
            - Original sequence length for reconstruction
        """
        batch, seq_len, n_vars = x.shape

        # Pad sequence if necessary
        pad_len = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            # Replicate padding
            if self.padding_mode == "replicate":
                x = F.pad(x.transpose(1, 2), (0, pad_len), mode="replicate").transpose(
                    1, 2
                )
            else:
                x = F.pad(x, (0, 0, 0, pad_len))

        # Create patches
        padded_len = seq_len + pad_len
        n_patches = (padded_len - self.patch_size) // self.stride + 1

        # Extract patches using unfold
        x = x.unfold(1, self.patch_size, self.stride)  # (batch, n_patches, n_vars, patch_size)
        x = rearrange(x, "b n v p -> b n (p v)")

        # Project to d_model
        x = self.projection(x)

        # Add position embedding
        x = x + self.pos_embedding[:, :n_patches, :]

        return self.norm(x), seq_len


class ChannelMixing(nn.Module):
    """
    Channel mixing module for cross-variable information exchange.

    Uses a simple MLP with GELU activation to mix information
    across different input variables.
    """

    def __init__(self, n_channels: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = n_channels * expansion_factor
        self.fc1 = nn.Linear(n_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (..., n_channels)

        Returns:
            Output of same shape
        """
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class MambaEncoder(nn.Module):
    """
    Stack of Mamba blocks forming the encoder.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all encoder layers.
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class PredictionHead(nn.Module):
    """
    Prediction head for various forecasting tasks.

    Supports:
    - Single-step regression
    - Multi-step regression
    - Classification (up/down)
    - Multi-horizon forecasting
    """

    def __init__(
        self,
        d_model: int,
        n_variables: int,
        forecast_horizon: int,
        task: str = "regression",
        n_classes: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_variables = n_variables
        self.forecast_horizon = forecast_horizon
        self.task = task

        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, n_classes),
            )
        elif task == "regression":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, forecast_horizon * n_variables),
            )
        elif task == "multi_horizon":
            self.heads = nn.ModuleList(
                [
                    nn.Linear(d_model, n_variables)
                    for _ in range(forecast_horizon)
                ]
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Encoder output of shape (batch, n_patches, d_model)

        Returns:
            Predictions based on task type
        """
        # Use the last patch representation
        x = x[:, -1, :]  # (batch, d_model)

        if self.task == "classification":
            return self.head(x)
        elif self.task == "regression":
            out = self.head(x)
            return out.view(-1, self.forecast_horizon, self.n_variables)
        elif self.task == "multi_horizon":
            return torch.stack([head(x) for head in self.heads], dim=1)


class MambaTS(nn.Module):
    """
    MambaTS: Full model for time series forecasting.

    Combines temporal patch embedding, Mamba encoder, and prediction head
    for efficient long-term forecasting of financial time series.

    Args:
        n_variables: Number of input variables (features)
        d_model: Model dimension (embedding size)
        n_layers: Number of Mamba encoder layers
        patch_size: Size of temporal patches
        forecast_horizon: Number of steps to forecast
        d_state: State dimension for SSM
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        dropout: Dropout rate
        task: Task type ('regression', 'classification', 'multi_horizon')
        n_classes: Number of classes for classification task
    """

    def __init__(
        self,
        n_variables: int,
        d_model: int = 256,
        n_layers: int = 4,
        patch_size: int = 16,
        forecast_horizon: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        task: str = "regression",
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_variables = n_variables
        self.d_model = d_model
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.forecast_horizon = forecast_horizon
        self.task = task

        # Patch embedding
        self.patch_embed = TemporalPatchEmbedding(
            n_variables=n_variables,
            patch_size=patch_size,
            d_model=d_model,
        )

        # Channel mixing before encoding
        self.channel_mix = ChannelMixing(d_model, dropout=dropout)

        # Mamba encoder
        self.encoder = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Prediction head
        self.pred_head = PredictionHead(
            d_model=d_model,
            n_variables=n_variables,
            forecast_horizon=forecast_horizon,
            task=task,
            n_classes=n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_variables)

        Returns:
            Predictions based on task type
        """
        # Patch embedding
        x, orig_len = self.patch_embed(x)

        # Channel mixing
        x = self.channel_mix(x)

        # Mamba encoding
        x = self.encoder(x)

        # Prediction
        return self.pred_head(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (inference mode).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "MambaTS":
        """
        Load a pretrained model.
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device)

    def save(self, path: str):
        """
        Save model checkpoint.
        """
        config = {
            "n_variables": self.n_variables,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "patch_size": self.patch_size,
            "forecast_horizon": self.forecast_horizon,
            "task": self.task,
        }
        torch.save(
            {
                "config": config,
                "model_state_dict": self.state_dict(),
            },
            path,
        )


class MambaTSTrainer:
    """
    Trainer class for MambaTS model.

    Handles training loop, validation, early stopping, and mixed precision training.
    """

    def __init__(
        self,
        model: MambaTS,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device == "cuda"

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        if model.task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")

    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader) -> float:
        """
        Validate the model.
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0

        for batch in val_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            loss = self.criterion(pred, y)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        patience: int = 10,
        save_path: Optional[str] = None,
    ) -> dict:
        """
        Full training loop with early stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self.model.save(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return history


if __name__ == "__main__":
    # Test the model
    batch_size = 8
    seq_len = 720  # 30 days of hourly data
    n_variables = 10
    forecast_horizon = 24

    # Create random input
    x = torch.randn(batch_size, seq_len, n_variables)

    # Test regression model
    model = MambaTS(
        n_variables=n_variables,
        d_model=128,
        n_layers=4,
        patch_size=16,
        forecast_horizon=forecast_horizon,
        task="regression",
    )

    y = model(x)
    print(f"Regression: input {x.shape} -> output {y.shape}")
    print(f"Expected output shape: ({batch_size}, {forecast_horizon}, {n_variables})")

    # Test classification model
    model_cls = MambaTS(
        n_variables=n_variables,
        d_model=128,
        n_layers=4,
        patch_size=16,
        forecast_horizon=1,
        task="classification",
        n_classes=2,
    )

    y_cls = model_cls(x)
    print(f"Classification: input {x.shape} -> output {y_cls.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
