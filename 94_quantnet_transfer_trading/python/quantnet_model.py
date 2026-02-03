"""
QuantNet Transfer Trading model and trainer.

This module implements the QuantNet architecture for cross-asset transfer
learning in trading. The model learns a shared representation across
multiple assets via an autoencoder pretraining phase, then finetunes
asset-specific trading heads using a differentiable Sharpe ratio loss.

Provides:
- QuantNetEncoder: Shared feature encoder with BatchNorm, GELU, Dropout
- QuantNetDecoder: Mirror decoder for reconstruction pretraining
- TradingHead: Asset-specific signal generator (tanh output in [-1, 1])
- QuantNet: Complete model with encoder, decoder, and ModuleDict of heads
- SharpeRatioLoss: Differentiable Sharpe ratio for gradient-based optimization
- QuantNetTrainer: Two-phase training loop (pretrain + finetune)

Reference:
    Zhang et al., "QuantNet: Transferring Learning Across Trading Strategies"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantNetEncoder(nn.Module):
    """
    Shared feature encoder for QuantNet.

    Learns a common latent representation across multiple assets.
    Architecture: [Linear -> BatchNorm -> GELU -> Dropout] x N

    Args:
        input_size: Dimensionality of input features
        hidden_sizes: List of hidden layer dimensions
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        self.encoder = nn.Sequential(*layers)
        self.output_size = hidden_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features into latent representation."""
        return self.encoder(x)


class QuantNetDecoder(nn.Module):
    """
    Decoder for QuantNet autoencoder pretraining.

    Mirrors the encoder architecture to reconstruct the original input,
    forcing the latent space to retain useful information.

    Args:
        latent_size: Dimensionality of the latent representation
        hidden_sizes: List of hidden layer dimensions (reverse of encoder)
        output_size: Dimensionality of reconstructed output
        dropout: Dropout probability
    """

    def __init__(
        self,
        latent_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_size = latent_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from latent representation."""
        return self.decoder(z)


class TradingHead(nn.Module):
    """
    Asset-specific trading signal generator.

    Produces continuous position signals in [-1, 1] via tanh activation,
    where -1 is full short, 0 is flat, and +1 is full long.

    Args:
        input_size: Dimensionality of the latent representation
        hidden_size: Hidden layer dimension
    """

    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate trading signal from latent features."""
        return self.head(z)


class QuantNet(nn.Module):
    """
    Complete QuantNet model for cross-asset transfer trading.

    Architecture:
        Input -> Encoder -> Latent -> { Decoder (reconstruction),
                                         TradingHead_asset1,
                                         TradingHead_asset2, ... }

    The encoder is shared across all assets during both pretraining
    (autoencoder reconstruction) and finetuning (Sharpe maximization).

    Args:
        input_size: Number of input features
        encoder_hidden: Hidden layer sizes for encoder
        decoder_hidden: Hidden layer sizes for decoder (reverse of encoder)
        head_hidden: Hidden layer size for trading heads
        asset_names: List of asset identifiers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        encoder_hidden: List[int],
        decoder_hidden: Optional[List[int]] = None,
        head_hidden: int = 32,
        asset_names: Optional[List[str]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if decoder_hidden is None:
            decoder_hidden = list(reversed(encoder_hidden[:-1]))

        self.encoder = QuantNetEncoder(input_size, encoder_hidden, dropout)
        latent_size = self.encoder.output_size

        self.decoder = QuantNetDecoder(
            latent_size, decoder_hidden, input_size, dropout,
        )

        if asset_names is None:
            asset_names = ["default"]
        self.asset_names = asset_names

        self.trading_heads = nn.ModuleDict({
            name: TradingHead(latent_size, head_hidden)
            for name in asset_names
        })

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstructed input."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        asset: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            x: Input features (batch_size, input_size)
            asset: If specified, only compute signal for this asset.
                   Otherwise, compute signals for all assets.

        Returns:
            Dict with keys 'reconstruction' and per-asset signal keys.
        """
        z = self.encoder(x)
        reconstruction = self.decoder(z)

        outputs: Dict[str, torch.Tensor] = {"reconstruction": reconstruction}

        if asset is not None:
            outputs[asset] = self.trading_heads[asset](z)
        else:
            for name, head in self.trading_heads.items():
                outputs[name] = head(z)

        return outputs

    def get_signals(
        self,
        features: np.ndarray,
        asset: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate trading signals from numpy features.

        Args:
            features: Input features as numpy array
            asset: Optional asset name to filter

        Returns:
            Dict mapping asset name -> signal array
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            z = self.encoder(x)
            signals = {}
            heads = (
                {asset: self.trading_heads[asset]}
                if asset else self.trading_heads
            )
            for name, head in heads.items():
                signals[name] = head(z).numpy().flatten()
        return signals

    def add_asset(self, asset_name: str, head_hidden: int = 32) -> None:
        """
        Add a new asset-specific trading head.

        The shared encoder weights are preserved; only a new head is
        initialized for the new asset.

        Args:
            asset_name: Identifier for the new asset
            head_hidden: Hidden size for the new trading head
        """
        if asset_name in self.trading_heads:
            logger.warning(f"Asset '{asset_name}' already exists, skipping.")
            return
        latent_size = self.encoder.output_size
        self.trading_heads[asset_name] = TradingHead(latent_size, head_hidden)
        self.asset_names.append(asset_name)
        logger.info(f"Added trading head for asset '{asset_name}'")


class SharpeRatioLoss(nn.Module):
    """
    Differentiable Sharpe ratio loss for gradient-based optimization.

    Computes the negative Sharpe ratio of strategy returns so that
    minimizing this loss maximizes the realized Sharpe ratio.

    The Sharpe ratio is computed as:
        SR = mean(returns) / (std(returns) + eps)

    Args:
        eps: Small constant for numerical stability
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        signals: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative Sharpe ratio loss.

        Args:
            signals: Trading signals in [-1, 1], shape (batch_size, 1)
            returns: Actual asset returns, shape (batch_size, 1)

        Returns:
            Scalar negative Sharpe ratio (lower is better)
        """
        strategy_returns = signals * returns
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std() + self.eps
        sharpe = mean_return / std_return
        return -sharpe


class QuantNetTrainer:
    """
    Two-phase trainer for the QuantNet model.

    Phase 1 (Pretraining): Autoencoder reconstruction across all assets.
        The encoder-decoder pair learns a universal feature representation.

    Phase 2 (Finetuning): Sharpe ratio maximization per asset.
        Asset-specific trading heads are optimized while the encoder
        is optionally frozen or updated with a reduced learning rate.

    Args:
        model: QuantNet model instance
        pretrain_lr: Learning rate for pretraining phase
        finetune_lr: Learning rate for finetuning phase
        encoder_finetune_lr: Learning rate for encoder during finetuning
            (set to 0 to freeze encoder)
    """

    def __init__(
        self,
        model: QuantNet,
        pretrain_lr: float = 1e-3,
        finetune_lr: float = 5e-4,
        encoder_finetune_lr: float = 1e-5,
    ):
        self.model = model
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.encoder_finetune_lr = encoder_finetune_lr

        # Pretraining optimizer (encoder + decoder)
        self.pretrain_optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=pretrain_lr,
        )

        # Finetuning optimizer with differential learning rates
        param_groups = [
            {"params": model.encoder.parameters(), "lr": encoder_finetune_lr},
            {"params": model.trading_heads.parameters(), "lr": finetune_lr},
        ]
        self.finetune_optimizer = torch.optim.Adam(param_groups)

        self.reconstruction_loss_fn = nn.MSELoss()
        self.sharpe_loss_fn = SharpeRatioLoss()

    def pretrain_step(
        self,
        features: torch.Tensor,
    ) -> Dict[str, float]:
        """
        One pretraining step: minimize reconstruction loss.

        Args:
            features: Input feature tensor (batch_size, input_size)

        Returns:
            Dict with 'reconstruction_loss' value
        """
        self.model.train()
        self.pretrain_optimizer.zero_grad()

        z = self.model.encode(features)
        reconstruction = self.model.decode(z)
        loss = self.reconstruction_loss_fn(reconstruction, features)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.pretrain_optimizer.step()

        return {"reconstruction_loss": loss.item()}

    def finetune_step(
        self,
        features: torch.Tensor,
        returns: torch.Tensor,
        asset: str,
    ) -> Dict[str, float]:
        """
        One finetuning step: maximize Sharpe ratio for a specific asset.

        Args:
            features: Input feature tensor (batch_size, input_size)
            returns: Actual asset returns (batch_size, 1)
            asset: Asset identifier

        Returns:
            Dict with 'sharpe_loss' and 'mean_signal' values
        """
        self.model.train()
        self.finetune_optimizer.zero_grad()

        outputs = self.model(features, asset=asset)
        signals = outputs[asset]

        sharpe_loss = self.sharpe_loss_fn(signals, returns)
        sharpe_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.finetune_optimizer.step()

        return {
            "sharpe_loss": sharpe_loss.item(),
            "mean_signal": signals.mean().item(),
        }

    def pretrain_epoch(
        self,
        features: torch.Tensor,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Run one full pretraining epoch with mini-batches.

        Args:
            features: Full feature tensor (concatenated across assets)
            batch_size: Mini-batch size

        Returns:
            Average reconstruction loss over the epoch
        """
        n = features.shape[0]
        indices = torch.randperm(n)
        epoch_losses: List[float] = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            batch = features[idx]
            step_result = self.pretrain_step(batch)
            epoch_losses.append(step_result["reconstruction_loss"])

        return {"reconstruction_loss": float(np.mean(epoch_losses))}

    def finetune_epoch(
        self,
        asset_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        batch_size: int = 64,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run one full finetuning epoch across all assets.

        Args:
            asset_data: Dict mapping asset name -> (features, returns) tensors
            batch_size: Mini-batch size

        Returns:
            Dict mapping asset name -> average losses
        """
        results: Dict[str, Dict[str, List[float]]] = {}

        for asset, (features, returns) in asset_data.items():
            if asset not in self.model.trading_heads:
                logger.warning(f"No trading head for '{asset}', skipping.")
                continue

            n = features.shape[0]
            indices = torch.randperm(n)
            asset_losses: Dict[str, List[float]] = {
                "sharpe_loss": [],
                "mean_signal": [],
            }

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]
                step_result = self.finetune_step(
                    features[idx], returns[idx], asset,
                )
                for k, v in step_result.items():
                    asset_losses[k].append(v)

            results[asset] = {
                k: float(np.mean(v)) for k, v in asset_losses.items()
            }

        return results


@dataclass
class QuantNetConfig:
    """Configuration for QuantNet training."""
    input_size: int = 10
    encoder_hidden: List[int] = field(default_factory=lambda: [64, 32, 16])
    decoder_hidden: Optional[List[int]] = None
    head_hidden: int = 32
    dropout: float = 0.1
    pretrain_lr: float = 1e-3
    finetune_lr: float = 5e-4
    encoder_finetune_lr: float = 1e-5
    pretrain_epochs: int = 100
    finetune_epochs: int = 200
    batch_size: int = 64


def main():
    """Example usage of the QuantNet model."""
    logger.info("QuantNet Transfer Trading Model Demo")

    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # Shared latent factors
    latent = np.random.randn(n_samples, 3).astype(np.float32)

    # Generate correlated features per asset
    asset_features = {}
    asset_returns = {}
    all_features = []
    for asset in assets:
        noise = np.random.randn(n_samples, n_features).astype(np.float32) * 0.3
        W = np.random.randn(3, n_features).astype(np.float32) * 0.5
        X = latent @ W + noise
        ret = (X[:, 0] * 0.3 + X[:, 1] * 0.2
               + np.random.randn(n_samples).astype(np.float32) * 0.05)
        asset_features[asset] = torch.tensor(X)
        asset_returns[asset] = torch.tensor(ret).unsqueeze(1)
        all_features.append(X)

    all_features_tensor = torch.tensor(
        np.concatenate(all_features, axis=0)
    )

    # Create model
    model = QuantNet(
        input_size=n_features,
        encoder_hidden=[64, 32, 16],
        head_hidden=32,
        asset_names=assets,
        dropout=0.1,
    )
    trainer = QuantNetTrainer(model)

    # Phase 1: Pretrain autoencoder
    logger.info("Phase 1: Pretraining autoencoder...")
    for epoch in range(50):
        losses = trainer.pretrain_epoch(all_features_tensor, batch_size=64)
        if epoch % 10 == 0:
            logger.info(
                f"  Epoch {epoch}: "
                f"recon_loss={losses['reconstruction_loss']:.4f}"
            )

    # Phase 2: Finetune trading heads
    logger.info("Phase 2: Finetuning trading heads...")
    ft_data = {
        asset: (asset_features[asset], asset_returns[asset])
        for asset in assets
    }
    for epoch in range(50):
        results = trainer.finetune_epoch(ft_data, batch_size=64)
        if epoch % 10 == 0:
            for asset, metrics in results.items():
                logger.info(
                    f"  Epoch {epoch} [{asset}]: "
                    f"sharpe_loss={metrics['sharpe_loss']:.4f}, "
                    f"mean_signal={metrics['mean_signal']:.4f}"
                )

    # Inference
    for asset in assets:
        signals = model.get_signals(asset_features[asset].numpy(), asset=asset)
        logger.info(
            f"Signals for {asset}: "
            f"mean={signals[asset].mean():.4f}, "
            f"std={signals[asset].std():.4f}"
        )

    # Demonstrate transfer: add new asset
    model.add_asset("AVAXUSDT")
    logger.info("Added new asset AVAXUSDT with fresh trading head")


if __name__ == "__main__":
    main()
