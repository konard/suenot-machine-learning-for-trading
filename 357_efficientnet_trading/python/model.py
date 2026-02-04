"""
EfficientNet Trading Model Module

This module provides the EfficientNet-based model for trading signal prediction
using time series to image transformations.

Features:
- Pre-trained EfficientNet backbone (B0-B7)
- Custom trading head with direction and magnitude prediction
- Transfer learning utilities
- Grad-CAM attention visualization
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch torchvision")


@dataclass
class ModelConfig:
    """Configuration for EfficientNet Trading Model."""

    # EfficientNet variant (b0, b1, b2, b3, b4, b5, b6, b7)
    variant: str = "b2"

    # Number of output classes (up, neutral, down)
    num_classes: int = 3

    # Input image size (should match EfficientNet variant)
    image_size: int = 260  # B2 default

    # Use pre-trained ImageNet weights
    pretrained: bool = True

    # Dropout rate for trading head
    dropout: float = 0.3

    # Hidden dimension for trading head
    hidden_dim: int = 512

    # Whether to predict magnitude in addition to direction
    predict_magnitude: bool = True

    # Freeze early layers for transfer learning
    freeze_backbone: bool = False

    # Number of backbone blocks to freeze (if freeze_backbone is False)
    freeze_blocks: int = 3


if TORCH_AVAILABLE:

    class Swish(nn.Module):
        """Swish activation function (x * sigmoid(x))."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)


    class SqueezeExcitation(nn.Module):
        """
        Squeeze-and-Excitation block.

        Learns to emphasize informative channels and suppress less useful ones.
        """

        def __init__(self, in_channels: int, reduction_ratio: int = 4):
            super().__init__()
            reduced_channels = max(1, in_channels // reduction_ratio)

            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Linear(in_channels, reduced_channels, bias=False),
                Swish(),
                nn.Linear(reduced_channels, in_channels, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, channels, _, _ = x.size()

            # Squeeze: Global average pooling
            squeezed = self.squeeze(x).view(batch, channels)

            # Excitation: Channel attention
            attention = self.excitation(squeezed).view(batch, channels, 1, 1)

            return x * attention


    class MBConvBlock(nn.Module):
        """
        Mobile Inverted Bottleneck Convolution Block.

        Key components:
        1. Expansion phase (increase channels)
        2. Depthwise convolution (spatial mixing)
        3. Squeeze-and-Excitation attention
        4. Projection phase (reduce channels)
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            expansion_ratio: int = 6,
            se_ratio: float = 0.25,
        ):
            super().__init__()

            self.use_residual = (stride == 1) and (in_channels == out_channels)
            expanded_channels = in_channels * expansion_ratio

            # Expansion
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish(),
            ) if expansion_ratio != 1 else nn.Identity()

            # Depthwise convolution
            padding = (kernel_size - 1) // 2
            self.depthwise = nn.Sequential(
                nn.Conv2d(
                    expanded_channels if expansion_ratio != 1 else in_channels,
                    expanded_channels if expansion_ratio != 1 else in_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=expanded_channels if expansion_ratio != 1 else in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded_channels if expansion_ratio != 1 else in_channels),
                Swish(),
            )

            # Squeeze-and-Excitation
            se_channels = expanded_channels if expansion_ratio != 1 else in_channels
            self.se = SqueezeExcitation(se_channels, int(1 / se_ratio))

            # Projection
            self.project = nn.Sequential(
                nn.Conv2d(
                    expanded_channels if expansion_ratio != 1 else in_channels,
                    out_channels,
                    1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = x

            # Expansion
            out = self.expand(x)

            # Depthwise
            out = self.depthwise(out)

            # Squeeze-and-Excitation
            out = self.se(out)

            # Projection
            out = self.project(out)

            # Residual connection
            if self.use_residual:
                out = out + identity

            return out


    class TradingHead(nn.Module):
        """
        Custom trading head for direction and magnitude prediction.

        Outputs:
        - Direction probabilities: [P(down), P(neutral), P(up)]
        - Magnitude: Expected return magnitude
        """

        def __init__(
            self,
            in_features: int,
            num_classes: int = 3,
            hidden_dim: int = 512,
            dropout: float = 0.3,
            predict_magnitude: bool = True,
        ):
            super().__init__()

            self.predict_magnitude = predict_magnitude

            # Direction prediction head
            self.direction_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim, num_classes),
            )

            # Magnitude prediction head
            if predict_magnitude:
                self.magnitude_head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, hidden_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim // 2, 1),
                )

        def forward(
            self,
            features: torch.Tensor,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Forward pass.

            Args:
                features: Backbone features of shape (batch, in_features)

            Returns:
                If predict_magnitude: (direction_logits, magnitude)
                Else: direction_logits
            """
            direction_logits = self.direction_head(features)

            if self.predict_magnitude:
                magnitude = self.magnitude_head(features)
                return direction_logits, magnitude

            return direction_logits


    class EfficientNetTrader(nn.Module):
        """
        EfficientNet-based trading model.

        Combines pre-trained EfficientNet backbone with custom trading head.

        Example:
            >>> model = EfficientNetTrader(config)
            >>> images = torch.randn(8, 3, 260, 260)
            >>> direction_probs, magnitude = model(images)
            >>> signals = model.generate_signals(direction_probs, magnitude)
        """

        # EfficientNet configurations: (width_mult, depth_mult, resolution, dropout)
        CONFIGS = {
            "b0": (1.0, 1.0, 224, 0.2),
            "b1": (1.0, 1.1, 240, 0.2),
            "b2": (1.1, 1.2, 260, 0.3),
            "b3": (1.2, 1.4, 300, 0.3),
            "b4": (1.4, 1.8, 380, 0.4),
            "b5": (1.6, 2.2, 456, 0.4),
            "b6": (1.8, 2.6, 528, 0.5),
            "b7": (2.0, 3.1, 600, 0.5),
        }

        def __init__(self, config: Optional[ModelConfig] = None):
            super().__init__()

            if config is None:
                config = ModelConfig()

            self.config = config

            # Try to load pretrained EfficientNet from torchvision
            self._build_model()

        def _build_model(self):
            """Build the model architecture."""
            try:
                from torchvision import models

                # Get pretrained EfficientNet
                variant = self.config.variant.lower()
                weights = "IMAGENET1K_V1" if self.config.pretrained else None

                model_builders = {
                    "b0": models.efficientnet_b0,
                    "b1": models.efficientnet_b1,
                    "b2": models.efficientnet_b2,
                    "b3": models.efficientnet_b3,
                    "b4": models.efficientnet_b4,
                    "b5": models.efficientnet_b5,
                    "b6": models.efficientnet_b6,
                    "b7": models.efficientnet_b7,
                }

                if variant not in model_builders:
                    raise ValueError(f"Unknown variant: {variant}")

                self.backbone = model_builders[variant](weights=weights)

                # Get feature dimension
                in_features = self.backbone.classifier[1].in_features

                # Remove original classifier
                self.backbone.classifier = nn.Identity()

                # Add trading head
                self.trading_head = TradingHead(
                    in_features=in_features,
                    num_classes=self.config.num_classes,
                    hidden_dim=self.config.hidden_dim,
                    dropout=self.config.dropout,
                    predict_magnitude=self.config.predict_magnitude,
                )

                # Freeze layers if specified
                if self.config.freeze_backbone:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                elif self.config.freeze_blocks > 0:
                    self._freeze_early_blocks(self.config.freeze_blocks)

            except ImportError:
                print("torchvision not available. Using simplified model.")
                self._build_simple_model()

        def _build_simple_model(self):
            """Build a simplified CNN model without torchvision."""
            # Simple CNN backbone
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                Swish(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                Swish(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                Swish(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                Swish(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

            self.trading_head = TradingHead(
                in_features=256,
                num_classes=self.config.num_classes,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
                predict_magnitude=self.config.predict_magnitude,
            )

        def _freeze_early_blocks(self, n_blocks: int):
            """Freeze first n blocks of the backbone."""
            blocks_frozen = 0
            for name, child in self.backbone.features.named_children():
                if blocks_frozen >= n_blocks:
                    break
                for param in child.parameters():
                    param.requires_grad = False
                blocks_frozen += 1

        def forward(
            self,
            x: torch.Tensor,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Forward pass.

            Args:
                x: Input images of shape (batch, 3, H, W)

            Returns:
                If predict_magnitude: (direction_logits, magnitude)
                Else: direction_logits
            """
            # Extract features
            features = self.backbone(x)

            # Trading head
            return self.trading_head(features)

        def predict(
            self,
            x: torch.Tensor,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Make predictions with probability outputs.

            Args:
                x: Input images

            Returns:
                (direction_probs, magnitude)
            """
            self.set_mode_inference()
            with torch.no_grad():
                output = self(x)

                if self.config.predict_magnitude:
                    direction_logits, magnitude = output
                else:
                    direction_logits = output
                    magnitude = None

                direction_probs = F.softmax(direction_logits, dim=1)

            return direction_probs, magnitude

        def set_mode_inference(self):
            """Set model to inference mode."""
            self.train(False)

        def generate_signals(
            self,
            direction_probs: torch.Tensor,
            magnitude: Optional[torch.Tensor] = None,
            buy_threshold: float = 0.6,
            sell_threshold: float = 0.6,
            min_magnitude: float = 0.005,
        ) -> List[Dict]:
            """
            Generate trading signals from predictions.

            Args:
                direction_probs: Probability tensor of shape (batch, 3)
                magnitude: Expected return magnitude (optional)
                buy_threshold: Probability threshold for buy signal
                sell_threshold: Probability threshold for sell signal
                min_magnitude: Minimum expected return for signal

            Returns:
                List of signal dictionaries
            """
            signals = []

            direction_probs = direction_probs.cpu().numpy()
            if magnitude is not None:
                magnitude = magnitude.cpu().numpy().squeeze()

            for i in range(len(direction_probs)):
                prob_down, prob_neutral, prob_up = direction_probs[i]
                mag = magnitude[i] if magnitude is not None else 0.0

                if prob_up > buy_threshold:
                    if magnitude is None or mag > min_magnitude:
                        signals.append({
                            "signal": "BUY",
                            "confidence": float(prob_up),
                            "expected_return": float(mag),
                        })
                    else:
                        signals.append({"signal": "HOLD", "confidence": 0.0})
                elif prob_down > sell_threshold:
                    if magnitude is None or mag < -min_magnitude:
                        signals.append({
                            "signal": "SELL",
                            "confidence": float(prob_down),
                            "expected_return": float(mag),
                        })
                    else:
                        signals.append({"signal": "HOLD", "confidence": 0.0})
                else:
                    signals.append({"signal": "HOLD", "confidence": 0.0})

            return signals


    class GradCAM:
        """
        Grad-CAM visualization for EfficientNet.

        Visualizes which parts of the input image the model
        focuses on for its predictions.
        """

        def __init__(self, model: EfficientNetTrader, target_layer: str = "features.8"):
            """
            Initialize Grad-CAM.

            Args:
                model: EfficientNet model
                target_layer: Name of target layer for visualization
            """
            self.model = model
            self.target_layer = target_layer

            self.activations = None
            self.gradients = None

            # Register hooks
            self._register_hooks()

        def _register_hooks(self):
            """Register forward and backward hooks."""
            def forward_hook(module, input, output):
                self.activations = output.detach()

            def backward_hook(module, grad_in, grad_out):
                self.gradients = grad_out[0].detach()

            # Find target layer
            target = None
            for name, module in self.model.backbone.named_modules():
                if name == self.target_layer:
                    target = module
                    break

            if target is None:
                # Try to find a suitable layer
                for name, module in self.model.backbone.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target = module

            if target is not None:
                target.register_forward_hook(forward_hook)
                target.register_full_backward_hook(backward_hook)

        def generate_cam(
            self,
            image: torch.Tensor,
            target_class: Optional[int] = None,
        ) -> np.ndarray:
            """
            Generate Grad-CAM visualization.

            Args:
                image: Input image tensor of shape (1, 3, H, W)
                target_class: Target class (if None, uses predicted class)

            Returns:
                CAM heatmap of shape (H, W)
            """
            self.model.set_mode_inference()

            # Forward pass
            output = self.model(image)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Get target class
            if target_class is None:
                target_class = logits.argmax(dim=1).item()

            # Backward pass
            self.model.zero_grad()
            logits[0, target_class].backward()

            # Compute CAM
            if self.activations is None or self.gradients is None:
                return np.zeros((image.shape[2], image.shape[3]))

            # Global average pooling of gradients
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)

            # Weighted combination of activation maps
            cam = (weights * self.activations).sum(dim=1, keepdim=True)

            # ReLU and resize
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

            # Normalize
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            return cam


def create_model(
    variant: str = "b2",
    pretrained: bool = True,
    num_classes: int = 3,
) -> "EfficientNetTrader":
    """
    Convenience function to create an EfficientNet trading model.

    Args:
        variant: EfficientNet variant (b0-b7)
        pretrained: Use ImageNet pretrained weights
        num_classes: Number of output classes

    Returns:
        EfficientNetTrader model
    """
    config = ModelConfig(
        variant=variant,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return EfficientNetTrader(config)


if __name__ == "__main__" and TORCH_AVAILABLE:
    # Example usage
    print("Testing EfficientNet Trading Model...")

    # Create model
    config = ModelConfig(variant="b0", pretrained=False)  # No pretrained for speed
    model = EfficientNetTrader(config)

    print(f"Model created: EfficientNet-{config.variant.upper()}")

    # Test forward pass
    batch_size = 4
    image_size = 224
    images = torch.randn(batch_size, 3, image_size, image_size)

    print(f"Input shape: {images.shape}")

    direction_logits, magnitude = model(images)
    print(f"Direction logits shape: {direction_logits.shape}")
    print(f"Magnitude shape: {magnitude.shape}")

    # Test prediction
    direction_probs, magnitude = model.predict(images)
    print(f"Direction probs: {direction_probs[0]}")

    # Generate signals
    signals = model.generate_signals(direction_probs, magnitude)
    print(f"Generated signals: {signals}")

    # Test Grad-CAM
    print("\nTesting Grad-CAM...")
    cam = GradCAM(model)
    single_image = torch.randn(1, 3, image_size, image_size)
    heatmap = cam.generate_cam(single_image)
    print(f"CAM heatmap shape: {heatmap.shape}")

    print("\nAll tests passed!")
