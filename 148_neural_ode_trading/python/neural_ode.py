"""
Neural ODE Module for Trading

Implements Neural Ordinary Differential Equations (Chen et al., 2018) for
continuous-time financial time series modeling.

Core idea: Instead of discrete layers h_{l+1} = h_l + f(h_l),
we model the continuous dynamics dh/dt = f_theta(h(t), t) and solve the
ODE with numerical solvers (RK4, Dormand-Prince).

The adjoint sensitivity method enables constant-memory backpropagation,
making it practical for long financial time series.

References:
    - Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018
    - Rubanova et al., "Latent ODEs for Irregularly-Sampled Time Series", NeurIPS 2019
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:
    raise ImportError(
        "torchdiffeq is required. Install with: pip install torchdiffeq"
    )


class ODEFunc(nn.Module):
    """
    Neural network that defines the ODE dynamics: dh/dt = f_theta(h(t), t).

    This is the core building block of a Neural ODE. The function f_theta
    parameterized by a neural network defines how the hidden state evolves
    continuously in time.

    For trading: the dynamics capture how market state evolves continuously
    rather than at discrete intervals.

    Args:
        hidden_dim: Dimension of the hidden state h(t)
        n_layers: Number of hidden layers in the dynamics network
        activation: Activation function ('tanh', 'relu', 'softplus', 'elu')
        time_dependent: If True, concatenate t to input (time-varying dynamics)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        activation: str = "tanh",
        time_dependent: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations (for monitoring solver efficiency)

        # Choose activation
        act_map = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "softplus": nn.Softplus,
            "elu": nn.ELU,
            "silu": nn.SiLU,
        }
        act_cls = act_map.get(activation, nn.Tanh)

        # Input dimension: hidden_dim + 1 if time-dependent (concatenate t)
        input_dim = hidden_dim + 1 if time_dependent else hidden_dim

        # Build the dynamics network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_cls())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*layers)

        # Initialize weights to be small for stable ODE dynamics
        self._initialize_weights()

    def _initialize_weights(self):
        """Small initialization for stable ODE integration."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt = f_theta(h(t), t).

        Args:
            t: Current time (scalar tensor)
            h: Hidden state at time t, shape (batch_size, hidden_dim)

        Returns:
            dh/dt: Time derivative of hidden state, same shape as h
        """
        self.nfe += 1

        if self.time_dependent:
            # Expand t to match batch dimension and concatenate
            t_expand = t.expand(h.shape[0], 1)
            h_aug = torch.cat([h, t_expand], dim=-1)
        else:
            h_aug = h

        return self.net(h_aug)


class ODENet(nn.Module):
    """
    Neural ODE network for sequence modeling.

    Encodes input features into a hidden state, evolves it through continuous
    dynamics via an ODE solver, and decodes to output predictions.

    Architecture:
        Input -> Encoder -> ODE Solver (continuous dynamics) -> Decoder -> Output

    For trading: processes market features (price, volume, indicators) through
    continuous-time dynamics to predict future states.

    Args:
        input_dim: Number of input features (e.g., OHLCV = 5)
        hidden_dim: Dimension of the ODE hidden state
        output_dim: Number of output features (e.g., 1 for price prediction)
        n_layers: Number of layers in the ODE function
        solver: ODE solver method ('dopri5', 'rk4', 'euler', 'midpoint', 'adaptive_heun')
        rtol: Relative tolerance for adaptive solvers
        atol: Absolute tolerance for adaptive solvers
        use_adjoint: If True, use adjoint method for O(1) memory backprop
        activation: Activation function for ODE dynamics
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_layers: int = 2,
        solver: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-5,
        use_adjoint: bool = True,
        activation: str = "tanh",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint

        # Encoder: map input features to hidden state
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ODE dynamics function
        self.ode_func = ODEFunc(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            time_dependent=True,
        )

        # Decoder: map hidden state to output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_span: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the Neural ODE.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            t_span: Time points to evaluate, shape (n_time_points,).
                     Default: [0, 1] (integrate from 0 to 1)
            return_trajectory: If True, return full trajectory

        Returns:
            If return_trajectory: tensor of shape (n_time_points, batch_size, output_dim)
            Else: tensor of shape (batch_size, output_dim) at final time
        """
        # Encode input to initial hidden state h(0)
        h0 = self.encoder(x)

        # Default integration interval
        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)

        # Choose ODE solver (adjoint for training, regular for inference)
        ode_solver = odeint_adjoint if self.use_adjoint else odeint
        self.ode_func.nfe = 0

        # Solve the ODE: integrate from t_span[0] to t_span[-1]
        # Returns trajectory at all time points in t_span
        h_trajectory = ode_solver(
            self.ode_func,
            h0,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        # h_trajectory shape: (n_time_points, batch_size, hidden_dim)

        if return_trajectory:
            # Decode entire trajectory
            out = self.decoder(h_trajectory)
            return out
        else:
            # Take hidden state at final time point
            h_final = h_trajectory[-1]
            return self.decoder(h_final)

    def get_nfe(self) -> int:
        """Return the number of function evaluations in the last forward pass."""
        return self.ode_func.nfe


class NeuralODE(nn.Module):
    """
    Complete Neural ODE model for trading with sequence input.

    Processes a window of market observations through a Neural ODE to produce
    trading signals or price predictions.

    This wraps ODENet with additional sequence handling: it can process
    a time series window, encode it, and evolve the encoded state through
    continuous-time dynamics.

    Args:
        input_dim: Number of features per time step
        hidden_dim: Hidden state dimension
        output_dim: Output dimension (1 for regression, 3 for buy/hold/sell)
        seq_len: Length of the input sequence window
        n_ode_layers: Number of layers in ODE dynamics
        solver: ODE solver ('dopri5', 'rk4', 'euler', 'adaptive_heun')
        use_adjoint: Use adjoint method for backpropagation
        task: 'regression' or 'classification'
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 1,
        seq_len: int = 60,
        n_ode_layers: int = 2,
        solver: str = "dopri5",
        use_adjoint: bool = True,
        task: str = "regression",
    ):
        super().__init__()
        self.task = task
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Sequence encoder: process the lookback window
        self.seq_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # ODE dynamics
        self.ode_func = ODEFunc(
            hidden_dim=hidden_dim,
            n_layers=n_ode_layers,
            activation="tanh",
            time_dependent=True,
        )

        # Decoder
        if task == "classification":
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        self.use_adjoint = use_adjoint
        self.solver = solver

    def forward(
        self,
        x: torch.Tensor,
        t_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequence, shape (batch_size, seq_len, input_dim)
            t_pred: Time points for prediction. Default: [0, 1] (one step ahead)

        Returns:
            Prediction tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Encode sequence into initial hidden state
        _, h0 = self.seq_encoder(x)
        h0 = h0.squeeze(0)  # (batch_size, hidden_dim)

        # Default: evolve one unit of time into the future
        if t_pred is None:
            t_pred = torch.tensor([0.0, 1.0], device=x.device, dtype=x.dtype)

        # Solve ODE forward
        ode_solver = odeint_adjoint if self.use_adjoint else odeint
        self.ode_func.nfe = 0

        h_trajectory = ode_solver(
            self.ode_func,
            h0,
            t_pred,
            method=self.solver,
            rtol=1e-4,
            atol=1e-5,
        )

        # Take final state for prediction
        h_final = h_trajectory[-1]  # (batch_size, hidden_dim)

        # Decode to output
        out = self.decoder(h_final)

        if self.task == "classification":
            return out  # Raw logits; apply softmax/sigmoid outside
        return out

    def predict_trajectory(
        self,
        x: torch.Tensor,
        t_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the full trajectory at multiple future time points.

        Useful for multi-step ahead forecasting.

        Args:
            x: Input sequence, shape (batch_size, seq_len, input_dim)
            t_points: Time points to evaluate, shape (n_points,)

        Returns:
            Predictions at each time point, shape (n_points, batch_size, output_dim)
        """
        _, h0 = self.seq_encoder(x)
        h0 = h0.squeeze(0)

        ode_solver = odeint_adjoint if self.use_adjoint else odeint

        h_traj = ode_solver(
            self.ode_func,
            h0,
            t_points,
            method=self.solver,
            rtol=1e-4,
            atol=1e-5,
        )

        return self.decoder(h_traj)


class ContinuousNormalizingFlow(nn.Module):
    """
    Continuous Normalizing Flow (CNF) for density estimation.

    Uses Neural ODEs to transform a simple base distribution (e.g., Gaussian)
    into a complex target distribution. For trading, this can model the
    distribution of returns, enabling probabilistic forecasting and risk estimation.

    The key insight: the change in log-probability is computed via the
    instantaneous change of variables formula:
        d log p(z(t)) / dt = -tr(df/dz)

    Args:
        input_dim: Dimension of the data
        hidden_dim: Hidden dimension of the dynamics network
        n_layers: Number of layers in dynamics
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Dynamics network that also outputs trace of Jacobian
        self.dynamics = CNFDynamics(input_dim, hidden_dim, n_layers)

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform samples and compute log-probability change.

        Args:
            x: Input samples, shape (batch_size, input_dim)
            reverse: If True, map from latent to data space

        Returns:
            z: Transformed samples
            delta_logp: Change in log-probability
        """
        # Augment state with log-probability accumulator
        logp_init = torch.zeros(x.shape[0], 1, device=x.device)
        state0 = torch.cat([x, logp_init], dim=-1)

        if reverse:
            t_span = torch.tensor([1.0, 0.0], device=x.device)
        else:
            t_span = torch.tensor([0.0, 1.0], device=x.device)

        state_traj = odeint(
            self.dynamics,
            state0,
            t_span,
            method="dopri5",
            rtol=1e-5,
            atol=1e-5,
        )

        state_final = state_traj[-1]
        z = state_final[:, : self.input_dim]
        delta_logp = state_final[:, self.input_dim :]

        return z, delta_logp

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x) using the CNF.

        Args:
            x: Data samples, shape (batch_size, input_dim)

        Returns:
            Log-probabilities, shape (batch_size,)
        """
        z, delta_logp = self.forward(x, reverse=False)

        # Log-probability under the base distribution (standard normal)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1, keepdim=True)

        # Change of variables: log p(x) = log p(z) - delta_logp
        log_px = log_pz - delta_logp

        return log_px.squeeze(-1)


class CNFDynamics(nn.Module):
    """
    Dynamics function for Continuous Normalizing Flows.

    Outputs both the state derivative and the trace of the Jacobian
    (for the log-probability computation).

    Uses the Hutchinson trace estimator for efficient computation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        layers.append(nn.Linear(input_dim + 1, hidden_dim))  # +1 for time
        layers.append(nn.Tanh())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

        # Small initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamics and trace for augmented state.

        Args:
            t: Current time
            state: Augmented state [x, logp], shape (batch_size, input_dim + 1)

        Returns:
            d[x, logp]/dt
        """
        x = state[:, : self.input_dim]
        batch_size = x.shape[0]

        # Enable gradient tracking for trace computation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            t_expand = t.expand(batch_size, 1)
            x_t = torch.cat([x, t_expand], dim=-1)
            dx = self.net(x_t)

            # Compute trace of Jacobian using Hutchinson estimator
            # trace(df/dx) = E[e^T (df/dx) e] where e ~ N(0, I)
            e = torch.randn_like(x)
            e_dx = torch.autograd.grad(
                dx, x, e, create_graph=True, retain_graph=True
            )[0]
            trace_jac = (e_dx * e).sum(dim=-1, keepdim=True)

        # d(logp)/dt = -trace(df/dx)
        d_logp = -trace_jac

        return torch.cat([dx, d_logp], dim=-1)
