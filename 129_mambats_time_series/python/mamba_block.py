"""
Core Mamba Block implementation for selective state space modeling.

Based on the paper "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
by Albert Gu and Tri Dao (2023).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - the core innovation of Mamba.

    Unlike traditional SSMs with fixed parameters, this module makes
    B, C, and delta (discretization step) input-dependent, enabling
    content-aware reasoning.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters projection
        # Projects to: delta, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Delta (discretization step) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialize A (diagonal state matrix)
        # Using the S4D initialization: A_n = -1/2 + ni
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the selective SSM.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Convolution for local context
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        # Compute input-dependent SSM parameters
        x_ssm = self.x_proj(x)
        delta = x_ssm[..., :1]
        B = x_ssm[..., 1 : self.d_state + 1]
        C = x_ssm[..., self.d_state + 1 :]

        # Process delta
        delta = F.softplus(self.dt_proj(delta))

        # Get A from log space
        A = -torch.exp(self.A_log)

        # Selective scan
        y = self.selective_scan(x, delta, A, B, C)

        # Gating
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs the selective scan operation.

        This is a simplified implementation. For production use,
        consider using the optimized CUDA kernels from the mamba-ssm package.
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize A and B
        delta_A = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))
        delta_B = delta * B.unsqueeze(-1)

        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        # Sequential scan (for clarity; can be parallelized)
        outputs = []
        for t in range(seq_len):
            # State update: h = A_bar * h + B_bar * x
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t : t + 1, :].transpose(1, 2)
            # Output: y = C * h
            y = torch.einsum("bdn,bdn->bd", C[:, t : t + 1].expand(-1, d_inner, -1), h)
            outputs.append(y)

        y = torch.stack(outputs, dim=1)

        # Add skip connection with D
        y = y + x * self.D

        return y


class MambaBlock(nn.Module):
    """
    Complete Mamba block with normalization and residual connection.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-norm and residual connection.
        """
        return x + self.dropout(self.ssm(self.norm(x)))


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba block that processes sequences in both directions.

    Useful for capturing both past and future context in time series
    where we have access to the full sequence (e.g., during training).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.forward_block = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.backward_block = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.fusion = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence in both directions and fuse results.
        """
        # Forward direction
        h_fwd = self.forward_block(x)

        # Backward direction
        x_rev = torch.flip(x, dims=[1])
        h_bwd = self.backward_block(x_rev)
        h_bwd = torch.flip(h_bwd, dims=[1])

        # Fuse
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.fusion(h)


class VariableAwareScanning(nn.Module):
    """
    Variable-aware scanning module for multivariate time series.

    Processes both temporal and cross-variable dependencies using
    separate Mamba blocks, then fuses the results.
    """

    def __init__(
        self,
        d_model: int,
        n_variables: int,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_variables = n_variables

        self.temporal_mamba = MambaBlock(d_model, d_state, dropout=dropout)
        self.variable_mamba = MambaBlock(d_model, d_state, dropout=dropout)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with variable-aware scanning.

        Args:
            x: Input of shape (batch, seq_len, n_vars, d_model)

        Returns:
            Output of shape (batch, seq_len, n_vars, d_model)
        """
        batch, seq_len, n_vars, d_model = x.shape

        # Temporal scanning: process each variable's time series
        x_temp = rearrange(x, "b l v d -> (b v) l d")
        h_temp = self.temporal_mamba(x_temp)
        h_temp = rearrange(h_temp, "(b v) l d -> b l v d", b=batch, v=n_vars)

        # Variable scanning: process cross-variable dependencies at each time step
        x_var = rearrange(x, "b l v d -> (b l) v d")
        h_var = self.variable_mamba(x_var)
        h_var = rearrange(h_var, "(b l) v d -> b l v d", b=batch, l=seq_len)

        # Fuse temporal and variable information
        h = torch.cat([h_temp, h_var], dim=-1)
        h = self.fusion(h)

        return self.norm(h + x)


if __name__ == "__main__":
    # Test the Mamba block
    batch_size = 4
    seq_len = 128
    d_model = 64

    x = torch.randn(batch_size, seq_len, d_model)

    # Test MambaBlock
    block = MambaBlock(d_model)
    y = block(x)
    print(f"MambaBlock: input {x.shape} -> output {y.shape}")

    # Test BidirectionalMambaBlock
    bi_block = BidirectionalMambaBlock(d_model)
    y_bi = bi_block(x)
    print(f"BidirectionalMambaBlock: input {x.shape} -> output {y_bi.shape}")

    # Test VariableAwareScanning
    n_vars = 5
    x_mv = torch.randn(batch_size, seq_len, n_vars, d_model)
    va_block = VariableAwareScanning(d_model, n_vars)
    y_va = va_block(x_mv)
    print(f"VariableAwareScanning: input {x_mv.shape} -> output {y_va.shape}")
