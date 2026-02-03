"""
Mamba4Cast Model Implementation for Zero-Shot Time Series Forecasting

This module implements the Mamba4Cast architecture, a zero-shot forecasting
model based on selective state space models.

References:
    - Ekambaram, V., et al. (2024). "Mamba4Cast: Efficient Zero-Shot Time
      Series Forecasting with State Space Models." arXiv:2410.09385
    - Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling
      with Selective State Spaces." arXiv:2312.00752
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Mamba4CastBlock(nn.Module):
    """
    Mamba4Cast block for zero-shot time series forecasting.

    Key differences from standard Mamba:
    1. Horizon-aware output projection for non-autoregressive forecasting
    2. Multi-scale temporal encoding
    3. Scale-invariant processing

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (N in paper)
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        max_horizon: Maximum forecast horizon
        dt_rank: Rank for delta projection (default: auto)
        dt_min: Minimum delta value
        dt_max: Maximum delta value
        bias: Whether to use bias in linear layers
        conv_bias: Whether to use bias in convolution
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_horizon: int = 96,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.max_horizon = max_horizon
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Input projection: projects to 2*d_inner for x and z (gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Causal convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # SSM parameters projection
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )

        # Delta projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize delta projection
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter (diagonal, learned in log space)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Horizon embedding for non-autoregressive output
        self.horizon_embed = nn.Embedding(max_horizon, d_model)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        horizon_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of Mamba4Cast block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            horizon_indices: Optional horizon indices for embedding

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Causal convolution
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = F.silu(x)

        # Selective SSM
        y = self.ssm(x)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        # Add horizon embedding if provided
        if horizon_indices is not None:
            h_embed = self.horizon_embed(horizon_indices)
            output = output + h_embed.unsqueeze(0)

        return output

    def ssm(self, x: Tensor) -> Tensor:
        """
        Selective State Space Model computation.

        Args:
            x: Input after convolution, shape (batch, seq_len, d_inner)

        Returns:
            SSM output, shape (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape

        # Project for dt, B, C
        x_dbl = self.x_proj(x)
        dt, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Compute delta with softplus
        dt = F.softplus(self.dt_proj(dt))

        # Get A from log space
        A = -torch.exp(self.A_log)

        # Discretize
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)

        # Selective scan
        h = torch.zeros(
            batch, d_inner, self.d_state, device=x.device, dtype=x.dtype
        )
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t:t+1, :].transpose(1, 2)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y + x * self.D

        return y


class Mamba4CastLayer(nn.Module):
    """
    Full Mamba4Cast layer with normalization and residual connection.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        max_horizon: Maximum forecast horizon
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_horizon: int = 96,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba4CastBlock(
            d_model, d_state, d_conv, expand, max_horizon
        )

    def forward(
        self,
        x: Tensor,
        horizon_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward with pre-norm and residual."""
        return x + self.mamba(self.norm(x), horizon_indices)


class Mamba4CastForecaster(nn.Module):
    """
    Complete Mamba4Cast model for zero-shot time series forecasting.

    Features:
    - Non-autoregressive multi-horizon forecasting
    - Context-based pattern recognition
    - Scale-invariant predictions through normalization

    Args:
        n_features: Number of input features
        d_model: Model dimension
        n_layers: Number of Mamba layers
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        max_horizon: Maximum forecast horizon
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_features: int = 1,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_horizon: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.max_horizon = max_horizon

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Positional encoding
        self.register_buffer(
            'pos_encoding',
            self._create_positional_encoding(2048, d_model)
        )

        # Mamba layers
        self.layers = nn.ModuleList([
            Mamba4CastLayer(d_model, d_state, d_conv, expand, max_horizon)
            for _ in range(n_layers)
        ])

        # Output normalization
        self.norm = nn.LayerNorm(d_model)

        # Forecast head (produces all horizons at once)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, max_horizon * n_features),
        )

    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int,
    ) -> Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(
        self,
        x: Tensor,
        horizon: Optional[int] = None,
    ) -> Tensor:
        """
        Generate forecasts for given context.

        Args:
            x: Context time series (batch, context_len, n_features)
            horizon: Number of steps to forecast (default: max_horizon)

        Returns:
            Forecasts of shape (batch, horizon, n_features)
        """
        if horizon is None:
            horizon = self.max_horizon

        batch, seq_len, _ = x.shape

        # Normalize input for scale invariance
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_normalized = (x - x_mean) / x_std

        # Project input
        x = self.input_proj(x_normalized)
        x = self.input_dropout(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Process through Mamba layers
        for layer in self.layers:
            x = layer(x)

        # Normalize
        x = self.norm(x)

        # Use last hidden state for forecasting
        last_hidden = x[:, -1, :]

        # Generate all forecasts at once (non-autoregressive)
        forecasts = self.forecast_head(last_hidden)
        forecasts = forecasts.view(batch, self.max_horizon, self.n_features)

        # Denormalize predictions
        forecasts = forecasts * x_std + x_mean

        # Return requested horizon
        return forecasts[:, :horizon, :]

    @torch.no_grad()
    def zero_shot_forecast(
        self,
        time_series,
        context_length: int = 100,
        horizon: int = 24,
    ):
        """
        Zero-shot forecasting for any time series.

        Args:
            time_series: Input series as numpy array, list, or tensor
            context_length: Number of historical points to use
            horizon: Forecast horizon

        Returns:
            Forecasted values as numpy array
        """
        self.eval()

        # Prepare input
        if not isinstance(time_series, torch.Tensor):
            time_series = torch.tensor(time_series, dtype=torch.float32)

        # Ensure correct shape
        if time_series.dim() == 1:
            time_series = time_series.unsqueeze(-1)
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(0)

        # Use last context_length points
        context = time_series[:, -context_length:, :]

        # Generate forecast
        forecast = self.forward(context, horizon=horizon)

        return forecast.squeeze(0).cpu().numpy()

    def generate_trading_signals(
        self,
        time_series,
        context_length: int = 100,
        horizon: int = 24,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
    ) -> dict:
        """
        Generate trading signals from zero-shot forecasts.

        Args:
            time_series: Input time series
            context_length: Historical context length
            horizon: Forecast horizon
            buy_threshold: Return threshold for buy signal
            sell_threshold: Return threshold for sell signal

        Returns:
            Dictionary with signals and forecasts
        """
        forecast = self.zero_shot_forecast(
            time_series, context_length, horizon
        )

        # Get current price
        if isinstance(time_series, torch.Tensor):
            current_price = time_series[-1, 0].item()
        else:
            current_price = time_series[-1] if len(time_series.shape) == 1 else time_series[-1, 0]

        # Calculate expected returns
        predicted_prices = forecast[:, 0]
        expected_returns = (predicted_prices - current_price) / current_price

        # Generate signals for each horizon
        signals = []
        for h, (pred_price, exp_ret) in enumerate(zip(predicted_prices, expected_returns)):
            if exp_ret > buy_threshold:
                signal = 'BUY'
            elif exp_ret < sell_threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            signals.append({
                'horizon': h + 1,
                'signal': signal,
                'predicted_price': float(pred_price),
                'expected_return': float(exp_ret),
            })

        return {
            'current_price': current_price,
            'signals': signals,
            'forecast': forecast,
        }


def create_mamba4cast_model(
    n_features: int = 1,
    preset: str = "default",
    **kwargs,
) -> Mamba4CastForecaster:
    """
    Factory function to create Mamba4Cast models with presets.

    Args:
        n_features: Number of input features
        preset: Model preset ('small', 'default', 'large')
        **kwargs: Override preset parameters

    Returns:
        Configured Mamba4CastForecaster
    """
    presets = {
        "small": {
            "d_model": 32,
            "n_layers": 2,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
            "max_horizon": 48,
        },
        "default": {
            "d_model": 64,
            "n_layers": 4,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "max_horizon": 96,
        },
        "large": {
            "d_model": 128,
            "n_layers": 6,
            "d_state": 32,
            "d_conv": 4,
            "expand": 2,
            "max_horizon": 192,
        },
    }

    config = presets.get(preset, presets["default"])
    config.update(kwargs)

    return Mamba4CastForecaster(n_features=n_features, **config)
