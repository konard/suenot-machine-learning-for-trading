"""
ODE-RNN Hybrid Architecture for Trading

Combines the strengths of ODE solvers and RNNs:
- Between observations: hidden state evolves via an ODE (continuous dynamics)
- At observations: hidden state is updated via an RNN cell (discrete jumps)

This is particularly powerful for financial data because:
1. Markets evolve continuously between observations (the ODE part)
2. New information arrives at discrete (irregular) times (the RNN part)
3. The model naturally handles irregular timestamps without interpolation

Architecture:
    For each observation (x_i, t_i):
        1. Evolve h(t_{i-1}) to h(t_i^-) using ODE: dh/dt = f_theta(h, t)
        2. Update h(t_i^-) -> h(t_i) using RNN: h(t_i) = GRU(x_i, h(t_i^-))

References:
    - Rubanova et al., "Latent ODEs for Irregularly-Sampled Time Series", NeurIPS 2019
    - De Brouwer et al., "GRU-ODE-Bayes", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:
    raise ImportError("torchdiffeq required. Install: pip install torchdiffeq")

from .neural_ode import ODEFunc


class ODERNNCell(nn.Module):
    """
    A single ODE-RNN cell that combines continuous ODE dynamics
    with discrete RNN updates.

    Between observations: evolve hidden state with ODE
    At observations: update hidden state with GRU

    Args:
        input_dim: Dimension of input observations
        hidden_dim: Dimension of hidden state
        ode_layers: Number of layers in ODE dynamics
        solver: ODE solver method
        use_adjoint: Use adjoint for backpropagation
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        ode_layers: int = 2,
        solver: str = "dopri5",
        use_adjoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.use_adjoint = use_adjoint

        # ODE dynamics for continuous evolution between observations
        self.ode_func = ODEFunc(
            hidden_dim=hidden_dim,
            n_layers=ode_layers,
            activation="tanh",
            time_dependent=True,
        )

        # GRU cell for discrete updates at observations
        self.gru_cell = nn.GRUCell(
            input_size=input_dim,
            hidden_size=hidden_dim,
        )

    def ode_step(
        self,
        h: torch.Tensor,
        t_start: float,
        t_end: float,
    ) -> torch.Tensor:
        """
        Evolve hidden state from t_start to t_end using ODE solver.

        Args:
            h: Current hidden state, shape (batch_size, hidden_dim)
            t_start: Start time
            t_end: End time

        Returns:
            Hidden state at t_end
        """
        if abs(t_end - t_start) < 1e-8:
            return h

        t_span = torch.tensor(
            [t_start, t_end], device=h.device, dtype=h.dtype
        )

        ode_solver = odeint_adjoint if self.use_adjoint else odeint

        h_traj = ode_solver(
            self.ode_func,
            h,
            t_span,
            method=self.solver,
            rtol=1e-4,
            atol=1e-5,
        )

        return h_traj[-1]

    def rnn_step(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update hidden state with observation using GRU.

        If mask is provided, only update for observed entries.

        Args:
            x: Observation, shape (batch_size, input_dim)
            h: Hidden state before update, shape (batch_size, hidden_dim)
            mask: Observation mask, shape (batch_size,). 1 = observed

        Returns:
            Updated hidden state
        """
        h_new = self.gru_cell(x, h)

        if mask is not None:
            # Only update hidden state for observed samples
            mask = mask.unsqueeze(-1)  # (batch_size, 1)
            h_new = mask * h_new + (1 - mask) * h
        return h_new

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        t_start: float,
        t_end: float,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One step of ODE-RNN: evolve to t_end via ODE, then update with x.

        Args:
            x: Observation at t_end, shape (batch_size, input_dim)
            h: Hidden state at t_start, shape (batch_size, hidden_dim)
            t_start: Time of previous state
            t_end: Time of current observation
            mask: Observation mask

        Returns:
            Updated hidden state at t_end
        """
        # 1. Continuous evolution: h(t_start) -> h(t_end^-)
        h_ode = self.ode_step(h, t_start, t_end)

        # 2. Discrete update with observation: h(t_end^-) -> h(t_end)
        h_new = self.rnn_step(x, h_ode, mask)

        return h_new


class ODERNN(nn.Module):
    """
    Complete ODE-RNN model for irregularly-sampled financial time series.

    Processes a sequence of observations at irregular time points, producing
    predictions at desired future times.

    The model alternates between:
    - ODE integration (continuous dynamics between observations)
    - GRU updates (incorporating new observations)

    For trading:
    - Naturally handles tick data with irregular timestamps
    - Continuous dynamics capture market microstructure
    - No need for time series interpolation or resampling
    - Can make predictions at any future time point

    Args:
        input_dim: Features per observation (e.g., price, volume, bid, ask)
        hidden_dim: Hidden state dimension
        output_dim: Output features (1 for regression, 3 for classification)
        ode_layers: Layers in ODE dynamics
        solver: ODE solver method
        use_adjoint: Use adjoint method for training
        task: 'regression' or 'classification'
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 1,
        ode_layers: int = 2,
        solver: str = "dopri5",
        use_adjoint: bool = True,
        task: str = "regression",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.task = task

        # ODE-RNN cell
        self.cell = ODERNNCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            ode_layers=ode_layers,
            solver=solver,
            use_adjoint=use_adjoint,
        )

        # Output layer
        if task == "classification":
            self.output_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        else:
            self.output_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Process an irregularly-sampled sequence.

        Args:
            x: Observations, shape (batch_size, n_obs, input_dim)
            t: Observation times, shape (n_obs,) or (batch_size, n_obs)
            mask: Observation mask, shape (batch_size, n_obs)
            return_hidden: If True, also return hidden state trajectory

        Returns:
            output: Prediction at the last time point, shape (batch_size, output_dim)
            hidden_trajectory (optional): shape (n_obs, batch_size, hidden_dim)
        """
        batch_size = x.shape[0]
        n_obs = x.shape[1]
        device = x.device

        # Handle time dimensions
        if t.dim() == 1:
            times = t
        else:
            times = t[0]  # Assume same times for all in batch

        # Normalize time to [0, 1] for numerical stability
        t_min = times.min()
        t_max = times.max()
        t_range = t_max - t_min
        if t_range > 0:
            times_norm = (times - t_min) / t_range
        else:
            times_norm = times - t_min

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        hidden_trajectory = []

        # Process each observation
        prev_t = 0.0
        for i in range(n_obs):
            curr_t = times_norm[i].item()
            x_i = x[:, i, :]  # (batch_size, input_dim)
            mask_i = mask[:, i] if mask is not None else None

            # ODE step + RNN update
            h = self.cell(x_i, h, prev_t, curr_t, mask_i)
            hidden_trajectory.append(h)

            prev_t = curr_t

        # Stack hidden trajectory
        hidden_trajectory = torch.stack(hidden_trajectory)  # (n_obs, batch, hidden)

        # Output from final hidden state
        output = self.output_net(h)

        if return_hidden:
            return output, hidden_trajectory
        return output

    def predict_future(
        self,
        x: torch.Tensor,
        t_obs: torch.Tensor,
        t_future: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process observations and predict at future time points.

        Args:
            x: Observations, shape (batch_size, n_obs, input_dim)
            t_obs: Observation times, shape (n_obs,)
            t_future: Future time points, shape (n_future,)
            mask: Observation mask

        Returns:
            Predictions at future times, shape (n_future, batch_size, output_dim)
        """
        batch_size = x.shape[0]
        device = x.device

        # First, process all observations
        output, _ = self.forward(x, t_obs, mask, return_hidden=True)

        # Get final hidden state by re-running (or cache it)
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        times = t_obs if t_obs.dim() == 1 else t_obs[0]

        t_all = torch.cat([times, t_future])
        t_min = t_all.min()
        t_range = t_all.max() - t_min
        times_norm = (times - t_min) / t_range if t_range > 0 else times - t_min
        t_future_norm = (t_future - t_min) / t_range if t_range > 0 else t_future - t_min

        # Process observations
        prev_t = 0.0
        for i in range(x.shape[1]):
            curr_t = times_norm[i].item()
            x_i = x[:, i, :]
            mask_i = mask[:, i] if mask is not None else None
            h = self.cell(x_i, h, prev_t, curr_t, mask_i)
            prev_t = curr_t

        # Evolve via ODE to future time points
        future_preds = []
        for j in range(len(t_future)):
            curr_t = t_future_norm[j].item()
            h = self.cell.ode_step(h, prev_t, curr_t)
            pred = self.output_net(h)
            future_preds.append(pred)
            prev_t = curr_t

        return torch.stack(future_preds)  # (n_future, batch, output_dim)


class GRUODEBayes(nn.Module):
    """
    GRU-ODE-Bayes model (De Brouwer et al., 2019).

    A Bayesian variant of ODE-RNN that uses a GRU-like ODE for continuous
    dynamics. The hidden state evolves as a GRU-style ODE between observations.

    The continuous GRU-ODE dynamics:
        dh/dt = (1 - z(t)) * (h_tilde(t) - h(t))

    where z(t) and h_tilde(t) are computed similar to GRU gates but as
    continuous functions of the hidden state.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden state dimension
        output_dim: Output dimension
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Continuous GRU dynamics
        self.gru_ode = GRUODEFunc(hidden_dim)

        # Discrete update (standard GRU cell)
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

        # Output mapping
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim * 2),  # mean + log_var
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Returns dict with 'mean', 'logvar', 'hidden_trajectory'.
        """
        batch_size = x.shape[0]
        n_obs = x.shape[1]
        device = x.device

        times = t if t.dim() == 1 else t[0]

        # Normalize times
        t_min, t_max = times.min(), times.max()
        t_range = t_max - t_min
        times_norm = (times - t_min) / t_range if t_range > 0 else times - t_min

        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        hidden_traj = []

        prev_t = 0.0
        for i in range(n_obs):
            curr_t = times_norm[i].item()

            # Continuous ODE evolution
            if abs(curr_t - prev_t) > 1e-8:
                t_span = torch.tensor([prev_t, curr_t], device=device)
                h_traj = odeint(self.gru_ode, h, t_span, method="dopri5")
                h = h_traj[-1]

            # Discrete GRU update at observation
            if mask is not None:
                obs_mask = mask[:, i].unsqueeze(-1)
                h_new = self.gru_cell(x[:, i], h)
                h = obs_mask * h_new + (1 - obs_mask) * h
            else:
                h = self.gru_cell(x[:, i], h)

            hidden_traj.append(h)
            prev_t = curr_t

        hidden_traj = torch.stack(hidden_traj)

        # Output with uncertainty
        out_params = self.output_net(h)
        mean = out_params[:, : self.output_net[-1].out_features // 2]
        logvar = out_params[:, self.output_net[-1].out_features // 2 :]

        return {
            "mean": mean,
            "logvar": logvar,
            "hidden_trajectory": hidden_traj,
        }


class GRUODEFunc(nn.Module):
    """
    GRU-style ODE dynamics for continuous hidden state evolution.

    dh/dt = (1 - z) * (h_tilde - h)

    where:
        z = sigmoid(W_z * h + b_z)      (update gate)
        r = sigmoid(W_r * h + b_r)      (reset gate)
        h_tilde = tanh(W_h * (r * h) + b_h)  (candidate)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_z = nn.Linear(hidden_dim, hidden_dim)
        self.W_r = nn.Linear(hidden_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.W_z(h))
        r = torch.sigmoid(self.W_r(h))
        h_tilde = torch.tanh(self.W_h(r * h))
        dh = (1 - z) * (h_tilde - h)
        return dh
