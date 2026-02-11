"""
Operator Learning Models for Finance

This module implements neural operator architectures for financial applications:
- DeepONet: Deep Operator Network for learning operators between function spaces
- FNO: Fourier Neural Operator for spectral-domain operator learning

Both architectures learn mappings G: u(x) -> s(x) between function spaces,
enabling resolution-invariant financial modeling.

Typical usage:
    config = DeepONetConfig(branch_input_dim=50, trunk_input_dim=2)
    model = DeepONet(config)
    output = model(branch_input, trunk_input)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


class OperatorType(Enum):
    """Supported operator learning architectures."""
    DEEPONET = "deeponet"
    FNO = "fno"
    DEEPONET_POD = "deeponet_pod"


# ═══════════════════════════════════════════════════════════════════════════════
# DeepONet Configuration and Architecture
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeepONetConfig:
    """Configuration for DeepONet model."""

    # Branch network: encodes the input function at sensor points
    branch_input_dim: int = 50  # Number of sensor points
    branch_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    branch_type: str = "mlp"  # "mlp", "cnn", "transformer"

    # Trunk network: encodes the evaluation location
    trunk_input_dim: int = 2  # e.g., (maturity, time) for yield curves
    trunk_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])

    # Shared parameters
    latent_dim: int = 128  # Number of basis functions p
    activation: str = "gelu"
    dropout: float = 0.1
    use_bias: bool = True

    # POD-DeepONet settings
    use_pod: bool = False
    n_pod_modes: int = 64

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64


if TORCH_AVAILABLE:

    class BranchNetMLP(nn.Module):
        """MLP-based branch network for encoding input functions."""

        def __init__(self, config: DeepONetConfig):
            super().__init__()
            layers = []
            in_dim = config.branch_input_dim
            for hidden_dim in config.branch_hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(self._get_activation(config.activation))
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, config.latent_dim))
            self.net = nn.Sequential(*layers)

        def _get_activation(self, name: str) -> nn.Module:
            activations = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "tanh": nn.Tanh(),
                "silu": nn.SiLU(),
            }
            return activations.get(name, nn.GELU())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass. x: (batch, branch_input_dim)."""
            return self.net(x)

    class BranchNetCNN(nn.Module):
        """CNN-based branch network for encoding input functions."""

        def __init__(self, config: DeepONetConfig):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(16),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * 8, 256),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, config.latent_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass. x: (batch, branch_input_dim)."""
            x = x.unsqueeze(1)  # (batch, 1, branch_input_dim)
            x = self.conv_layers(x)
            x = x.flatten(1)
            return self.fc(x)

    class TrunkNet(nn.Module):
        """Trunk network for encoding evaluation locations."""

        def __init__(self, config: DeepONetConfig):
            super().__init__()
            layers = []
            in_dim = config.trunk_input_dim
            for hidden_dim in config.trunk_hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(self._get_activation(config.activation))
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, config.latent_dim))
            self.net = nn.Sequential(*layers)

        def _get_activation(self, name: str) -> nn.Module:
            activations = {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "tanh": nn.Tanh(),
                "silu": nn.SiLU(),
            }
            return activations.get(name, nn.GELU())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass. x: (batch, trunk_input_dim)."""
            return self.net(x)

    class DeepONet(nn.Module):
        """
        Deep Operator Network for learning operators between function spaces.

        Architecture:
            Branch net encodes input function u at sensor points -> [b_1, ..., b_p]
            Trunk net encodes evaluation point y -> [t_1, ..., t_p]
            Output: G(u)(y) = sum(b_k * t_k) + bias

        Args:
            config: DeepONetConfig with architecture hyperparameters

        Example:
            >>> config = DeepONetConfig(branch_input_dim=50, trunk_input_dim=2)
            >>> model = DeepONet(config)
            >>> # branch_input: yield curve at 50 maturities
            >>> # trunk_input: (maturity, time_horizon) for evaluation
            >>> branch_input = torch.randn(32, 50)
            >>> trunk_input = torch.randn(32, 2)
            >>> output = model(branch_input, trunk_input)
        """

        def __init__(self, config: DeepONetConfig):
            super().__init__()
            self.config = config

            # Build branch network
            if config.branch_type == "cnn":
                self.branch_net = BranchNetCNN(config)
            else:
                self.branch_net = BranchNetMLP(config)

            # Build trunk network
            self.trunk_net = TrunkNet(config)

            # Optional bias
            if config.use_bias:
                self.bias = nn.Parameter(torch.zeros(1))
            else:
                self.bias = None

        def forward(
            self,
            branch_input: torch.Tensor,
            trunk_input: torch.Tensor,
        ) -> torch.Tensor:
            """
            Forward pass of DeepONet.

            Args:
                branch_input: Input function values at sensor points (batch, m)
                trunk_input: Evaluation locations (batch, d) or (batch, n_points, d)

            Returns:
                Operator output at evaluation points (batch,) or (batch, n_points)
            """
            # Branch: (batch, p)
            b = self.branch_net(branch_input)

            # Handle multiple evaluation points
            if trunk_input.dim() == 3:
                # trunk_input: (batch, n_points, d)
                batch_size, n_points, d = trunk_input.shape
                trunk_flat = trunk_input.reshape(-1, d)
                t = self.trunk_net(trunk_flat)  # (batch * n_points, p)
                t = t.reshape(batch_size, n_points, -1)  # (batch, n_points, p)
                # Dot product: (batch, n_points)
                output = torch.einsum("bp,bnp->bn", b, t)
            else:
                # trunk_input: (batch, d)
                t = self.trunk_net(trunk_input)  # (batch, p)
                # Dot product: (batch,)
                output = torch.sum(b * t, dim=-1)

            if self.bias is not None:
                output = output + self.bias

            return output

        def predict_function(
            self,
            branch_input: torch.Tensor,
            eval_grid: torch.Tensor,
        ) -> torch.Tensor:
            """
            Predict the output function on a grid of evaluation points.

            Args:
                branch_input: Input function values (1, m) or (batch, m)
                eval_grid: Grid of evaluation points (n_points, d)

            Returns:
                Function values at grid points (batch, n_points)
            """
            batch_size = branch_input.shape[0]
            n_points = eval_grid.shape[0]

            # Expand eval_grid for batch
            eval_expanded = eval_grid.unsqueeze(0).expand(batch_size, -1, -1)

            return self.forward(branch_input, eval_expanded)

    # ═══════════════════════════════════════════════════════════════════════════
    # Fourier Neural Operator
    # ═══════════════════════════════════════════════════════════════════════════

    @dataclass
    class FNOConfig:
        """Configuration for Fourier Neural Operator."""

        # Input/output dimensions
        input_dim: int = 1  # Number of input channels
        output_dim: int = 1  # Number of output channels

        # FNO architecture
        hidden_channels: int = 64
        n_fourier_layers: int = 4
        n_fourier_modes: int = 16  # Number of Fourier modes to keep

        # 1D or 2D
        spatial_dim: int = 1  # 1 for time series, 2 for surfaces

        # Training
        activation: str = "gelu"
        dropout: float = 0.0
        learning_rate: float = 1e-3
        weight_decay: float = 1e-4
        batch_size: int = 32

    class SpectralConv1d(nn.Module):
        """1D Spectral convolution layer for FNO."""

        def __init__(self, in_channels: int, out_channels: int, modes: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes = modes

            # Complex-valued weights for Fourier modes
            scale = 1.0 / (in_channels * out_channels)
            self.weights = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass with spectral convolution.
            x: (batch, channels, spatial_dim)
            """
            batch_size = x.shape[0]
            n_points = x.shape[-1]

            # FFT
            x_ft = torch.fft.rfft(x, dim=-1)

            # Multiply relevant Fourier modes
            out_ft = torch.zeros(
                batch_size,
                self.out_channels,
                n_points // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            modes = min(self.modes, n_points // 2 + 1)
            out_ft[:, :, :modes] = torch.einsum(
                "bim,iom->bom", x_ft[:, :, :modes], self.weights[:, :, :modes]
            )

            # IFFT
            return torch.fft.irfft(out_ft, n=n_points, dim=-1)

    class SpectralConv2d(nn.Module):
        """2D Spectral convolution layer for FNO (e.g., volatility surfaces)."""

        def __init__(
            self, in_channels: int, out_channels: int, modes1: int, modes2: int
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes1 = modes1
            self.modes2 = modes2

            scale = 1.0 / (in_channels * out_channels)
            self.weights1 = nn.Parameter(
                scale
                * torch.randn(
                    in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
                )
            )
            self.weights2 = nn.Parameter(
                scale
                * torch.randn(
                    in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
                )
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (batch, channels, dim1, dim2)
            """
            batch_size = x.shape[0]
            n1, n2 = x.shape[-2], x.shape[-1]

            x_ft = torch.fft.rfft2(x, dim=(-2, -1))

            out_ft = torch.zeros(
                batch_size,
                self.out_channels,
                n1,
                n2 // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )

            m1 = min(self.modes1, n1)
            m2 = min(self.modes2, n2 // 2 + 1)

            # Positive frequencies
            out_ft[:, :, :m1, :m2] = torch.einsum(
                "bimn,iomn->bomn",
                x_ft[:, :, :m1, :m2],
                self.weights1[:, :, :m1, :m2],
            )
            # Negative frequencies
            out_ft[:, :, -m1:, :m2] = torch.einsum(
                "bimn,iomn->bomn",
                x_ft[:, :, -m1:, :m2],
                self.weights2[:, :, :m1, :m2],
            )

            return torch.fft.irfft2(out_ft, s=(n1, n2), dim=(-2, -1))

    class FNOBlock1d(nn.Module):
        """Single FNO block: spectral convolution + linear bypass + activation."""

        def __init__(self, channels: int, modes: int, activation: str = "gelu"):
            super().__init__()
            self.spectral_conv = SpectralConv1d(channels, channels, modes)
            self.linear = nn.Conv1d(channels, channels, kernel_size=1)
            self.norm = nn.InstanceNorm1d(channels)
            self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, channels, spatial)"""
            x1 = self.spectral_conv(x)
            x2 = self.linear(x)
            out = self.norm(x1 + x2)
            return self.activation(out)

    class FNOBlock2d(nn.Module):
        """2D FNO block for surface-valued operators."""

        def __init__(
            self, channels: int, modes1: int, modes2: int, activation: str = "gelu"
        ):
            super().__init__()
            self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
            self.linear = nn.Conv2d(channels, channels, kernel_size=1)
            self.norm = nn.InstanceNorm2d(channels)
            self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, channels, dim1, dim2)"""
            x1 = self.spectral_conv(x)
            x2 = self.linear(x)
            out = self.norm(x1 + x2)
            return self.activation(out)

    class FourierNeuralOperator(nn.Module):
        """
        Fourier Neural Operator for learning operators in spectral domain.

        The FNO learns global operators through spectral convolutions in Fourier space.
        Each layer applies:
            v_{l+1}(x) = sigma(W_l * v_l(x) + K_l(v_l)(x))
        where K_l is a kernel integral operator parameterized in Fourier space.

        Args:
            config: FNOConfig with architecture hyperparameters

        Example:
            >>> config = FNOConfig(input_dim=1, output_dim=1, n_fourier_modes=16)
            >>> model = FourierNeuralOperator(config)
            >>> x = torch.randn(32, 1, 128)  # batch of 1D functions on 128-point grid
            >>> y = model(x)  # (32, 1, 128) output functions
        """

        def __init__(self, config: FNOConfig):
            super().__init__()
            self.config = config

            if config.spatial_dim == 1:
                # Lifting layer
                self.lifting = nn.Conv1d(
                    config.input_dim, config.hidden_channels, kernel_size=1
                )

                # Fourier layers
                self.fourier_layers = nn.ModuleList(
                    [
                        FNOBlock1d(
                            config.hidden_channels,
                            config.n_fourier_modes,
                            config.activation,
                        )
                        for _ in range(config.n_fourier_layers)
                    ]
                )

                # Projection layer
                self.projection = nn.Sequential(
                    nn.Conv1d(config.hidden_channels, 128, kernel_size=1),
                    nn.GELU(),
                    nn.Conv1d(128, config.output_dim, kernel_size=1),
                )
            else:
                # 2D case
                self.lifting = nn.Conv2d(
                    config.input_dim, config.hidden_channels, kernel_size=1
                )

                modes = config.n_fourier_modes
                self.fourier_layers = nn.ModuleList(
                    [
                        FNOBlock2d(
                            config.hidden_channels, modes, modes, config.activation
                        )
                        for _ in range(config.n_fourier_layers)
                    ]
                )

                self.projection = nn.Sequential(
                    nn.Conv2d(config.hidden_channels, 128, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(128, config.output_dim, kernel_size=1),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of FNO.

            Args:
                x: Input function values on grid
                   1D: (batch, channels, n_points)
                   2D: (batch, channels, n1, n2)

            Returns:
                Output function values on same grid
            """
            # Lift to higher-dimensional channel space
            x = self.lifting(x)

            # Apply Fourier layers
            for layer in self.fourier_layers:
                x = layer(x)

            # Project back to output dimension
            x = self.projection(x)

            return x

        def predict_superresolution(
            self,
            x: torch.Tensor,
            target_resolution: int,
        ) -> torch.Tensor:
            """
            Predict at higher resolution than training (zero-shot super-resolution).

            Args:
                x: Input on coarse grid (batch, channels, n_coarse)
                target_resolution: Desired output resolution

            Returns:
                Output on fine grid (batch, output_dim, target_resolution)
            """
            # Interpolate input to target resolution
            if self.config.spatial_dim == 1:
                x_fine = F.interpolate(
                    x, size=target_resolution, mode="linear", align_corners=True
                )
            else:
                x_fine = F.interpolate(
                    x, size=target_resolution, mode="bilinear", align_corners=True
                )

            return self.forward(x_fine)

    # ═══════════════════════════════════════════════════════════════════════════
    # Financial-specific Operator Models
    # ═══════════════════════════════════════════════════════════════════════════

    class YieldCurveOperator(nn.Module):
        """
        Operator for yield curve evolution: G: r_0(T) -> r_t(T).

        Maps today's yield curve to future yield curve under specified conditions.
        Uses DeepONet architecture with market condition encoding.
        """

        def __init__(
            self,
            n_maturities: int = 30,
            n_market_features: int = 5,
            latent_dim: int = 128,
        ):
            super().__init__()
            self.n_maturities = n_maturities

            # Branch: encode current yield curve + market features
            config = DeepONetConfig(
                branch_input_dim=n_maturities + n_market_features,
                trunk_input_dim=2,  # (maturity, time_horizon)
                latent_dim=latent_dim,
                branch_hidden_dims=[256, 256, 256],
                trunk_hidden_dims=[256, 256, 256],
            )
            self.operator = DeepONet(config)

        def forward(
            self,
            yield_curve: torch.Tensor,
            market_features: torch.Tensor,
            eval_points: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                yield_curve: Current yield curve (batch, n_maturities)
                market_features: Market conditions (batch, n_market_features)
                eval_points: (maturity, horizon) pairs (batch, n_eval, 2)

            Returns:
                Predicted yields (batch, n_eval)
            """
            branch_input = torch.cat([yield_curve, market_features], dim=-1)
            return self.operator(branch_input, eval_points)

    class VolSurfaceOperator(nn.Module):
        """
        Operator for implied volatility surface dynamics using 2D FNO.

        Maps current vol surface sigma(K, T) to future vol surface.
        """

        def __init__(
            self,
            n_strikes: int = 20,
            n_maturities: int = 10,
            hidden_channels: int = 64,
            n_layers: int = 4,
            n_modes: int = 8,
        ):
            super().__init__()

            config = FNOConfig(
                input_dim=2,  # vol surface + moneyness grid
                output_dim=1,
                hidden_channels=hidden_channels,
                n_fourier_layers=n_layers,
                n_fourier_modes=n_modes,
                spatial_dim=2,
            )
            self.fno = FourierNeuralOperator(config)

        def forward(self, vol_surface: torch.Tensor) -> torch.Tensor:
            """
            Args:
                vol_surface: Current implied vol surface (batch, 2, n_strikes, n_mats)
                            Channel 0: implied vols, Channel 1: moneyness coordinates

            Returns:
                Predicted next-step vol surface (batch, 1, n_strikes, n_mats)
            """
            return self.fno(vol_surface)

    class OptionPricingOperator(nn.Module):
        """
        Operator for learning PDE solution operators for option pricing.

        Maps (volatility_function, rate, payoff_type) -> price_surface V(S, t).
        Replaces finite difference / Monte Carlo for fast real-time pricing.
        """

        def __init__(
            self,
            n_vol_points: int = 50,
            n_params: int = 5,
            latent_dim: int = 256,
        ):
            super().__init__()

            # Branch: encode volatility function + model parameters
            config = DeepONetConfig(
                branch_input_dim=n_vol_points + n_params,
                trunk_input_dim=2,  # (spot/strike ratio, time to expiry)
                latent_dim=latent_dim,
                branch_hidden_dims=[512, 512, 256],
                trunk_hidden_dims=[256, 256, 256],
            )
            self.operator = DeepONet(config)

        def forward(
            self,
            vol_function: torch.Tensor,
            model_params: torch.Tensor,
            eval_points: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                vol_function: Local/implied vol at grid points (batch, n_vol_points)
                model_params: Model parameters [r, q, ...] (batch, n_params)
                eval_points: (S/K ratio, T) pairs (batch, n_eval, 2) or (batch, 2)

            Returns:
                Option prices at evaluation points (batch,) or (batch, n_eval)
            """
            branch_input = torch.cat([vol_function, model_params], dim=-1)
            return self.operator(branch_input, eval_points)

    class CryptoOrderBookOperator(nn.Module):
        """
        Operator for learning order book dynamics on crypto exchanges (Bybit).

        Maps (bid_profile, ask_profile, trade_features) -> mid_price_change distribution.
        """

        def __init__(
            self,
            n_book_levels: int = 25,
            n_trade_features: int = 10,
            latent_dim: int = 128,
            n_output_bins: int = 50,
        ):
            super().__init__()
            self.n_output_bins = n_output_bins

            # Branch: encode order book snapshot
            branch_dim = 2 * n_book_levels + n_trade_features
            self.branch = nn.Sequential(
                nn.Linear(branch_dim, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, latent_dim),
            )

            # Trunk: encode prediction horizon and price level
            self.trunk = nn.Sequential(
                nn.Linear(2, 128),  # (horizon, price_offset)
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, latent_dim),
            )

            self.bias = nn.Parameter(torch.zeros(1))

            # Classification head for direction
            self.direction_head = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.GELU(),
                nn.Linear(64, 3),  # up, neutral, down
            )

            # Regression head for magnitude
            self.magnitude_head = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(
            self,
            order_book: torch.Tensor,
            eval_points: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                order_book: Flattened order book + trade features (batch, branch_dim)
                eval_points: (horizon, price_offset) (batch, 2)

            Returns:
                Dict with 'direction_logits', 'magnitude', 'operator_output'
            """
            b = self.branch(order_book)
            t = self.trunk(eval_points)

            # Operator output
            operator_out = torch.sum(b * t, dim=-1) + self.bias

            # Direction classification
            direction_logits = self.direction_head(b)

            # Magnitude regression
            magnitude = self.magnitude_head(b).squeeze(-1)

            return {
                "direction_logits": direction_logits,
                "magnitude": magnitude,
                "operator_output": operator_out,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions (available without PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

def relative_l2_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute relative L2 error between prediction and target functions."""
    return np.linalg.norm(prediction - target) / (np.linalg.norm(target) + 1e-10)


def generate_black_scholes_data(
    n_samples: int = 1000,
    n_vol_points: int = 50,
    n_eval_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training data for Black-Scholes operator learning.

    Creates pairs of (volatility_function, evaluation_points, option_prices)
    using the analytical Black-Scholes formula.

    Returns:
        vol_functions: (n_samples, n_vol_points) - volatility at grid points
        eval_points: (n_samples, n_eval_points, 2) - (S/K, T) pairs
        prices: (n_samples, n_eval_points) - option prices
    """
    from scipy.stats import norm

    vol_functions = np.zeros((n_samples, n_vol_points))
    eval_points = np.zeros((n_samples, n_eval_points, 2))
    prices = np.zeros((n_samples, n_eval_points))

    s_grid = np.linspace(0.5, 2.0, n_vol_points)  # S/K ratios for vol

    for i in range(n_samples):
        # Random volatility function (smooth)
        base_vol = np.random.uniform(0.1, 0.6)
        skew = np.random.uniform(-0.3, 0.1)
        curvature = np.random.uniform(0.0, 0.2)
        vol_func = base_vol + skew * (s_grid - 1.0) + curvature * (s_grid - 1.0) ** 2
        vol_func = np.clip(vol_func, 0.05, 1.0)
        vol_functions[i] = vol_func

        # Random evaluation points
        s_k_ratios = np.random.uniform(0.6, 1.5, n_eval_points)
        times = np.random.uniform(0.01, 2.0, n_eval_points)
        eval_points[i, :, 0] = s_k_ratios
        eval_points[i, :, 1] = times

        # Compute BS prices with local vol (interpolated)
        r = 0.03
        for j in range(n_eval_points):
            s_k = s_k_ratios[j]
            t = times[j]
            # Interpolate volatility
            sigma = np.interp(s_k, s_grid, vol_func)
            # Black-Scholes formula
            d1 = (np.log(s_k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t) + 1e-10)
            d2 = d1 - sigma * np.sqrt(t)
            prices[i, j] = s_k * norm.cdf(d1) - np.exp(-r * t) * norm.cdf(d2)

    return vol_functions, eval_points, prices


def generate_yield_curve_data(
    n_samples: int = 1000,
    n_maturities: int = 30,
    horizon_days: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic yield curve evolution data.

    Returns:
        curves_today: (n_samples, n_maturities)
        curves_future: (n_samples, n_maturities)
    """
    maturities = np.linspace(0.25, 30, n_maturities)  # 3M to 30Y
    curves_today = np.zeros((n_samples, n_maturities))
    curves_future = np.zeros((n_samples, n_maturities))

    for i in range(n_samples):
        # Nelson-Siegel parameters for today
        beta0 = np.random.uniform(0.02, 0.06)
        beta1 = np.random.uniform(-0.04, 0.02)
        beta2 = np.random.uniform(-0.03, 0.03)
        tau = np.random.uniform(1.0, 5.0)

        # Nelson-Siegel yield curve
        factor1 = (1 - np.exp(-maturities / tau)) / (maturities / tau)
        factor2 = factor1 - np.exp(-maturities / tau)
        curves_today[i] = beta0 + beta1 * factor1 + beta2 * factor2

        # Evolve parameters (random walk with mean reversion)
        d_beta0 = np.random.normal(0, 0.002)
        d_beta1 = np.random.normal(0, 0.003)
        d_beta2 = np.random.normal(0, 0.002)
        curves_future[i] = (
            (beta0 + d_beta0)
            + (beta1 + d_beta1) * factor1
            + (beta2 + d_beta2) * factor2
        )
        # Add noise
        curves_future[i] += np.random.normal(0, 0.0005, n_maturities)

    return curves_today, curves_future
