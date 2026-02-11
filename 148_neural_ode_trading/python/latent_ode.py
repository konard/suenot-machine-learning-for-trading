"""
Latent ODE Module for Irregularly-Sampled Time Series

Implements the Latent ODE model (Rubanova et al., 2019) which is specifically
designed for irregularly-sampled time series -- a common scenario in financial
data (tick data, missing observations, different trading hours).

Architecture:
    1. Recognition/Encoder RNN processes observations backward in time
    2. Encoder outputs parameters of the approximate posterior q(z0 | x)
    3. Sample z0 from the posterior
    4. Solve the ODE forward: dz/dt = f_theta(z(t), t)
    5. Decode latent state at desired time points

This is essentially a VAE where the generative model uses an ODE in latent space.

Key advantage for trading:
    - Naturally handles irregular timestamps (tick data, gaps, holidays)
    - Continuous-time model can be evaluated at any time point
    - Uncertainty quantification through the VAE framework

References:
    - Rubanova et al., "Latent ODEs for Irregularly-Sampled Time Series", NeurIPS 2019
    - Kidger et al., "Neural CDEs for Long Time Series via the Log-ODE Method", ICML 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:
    raise ImportError("torchdiffeq required. Install: pip install torchdiffeq")

from .neural_ode import ODEFunc


class RecognitionRNN(nn.Module):
    """
    Backward-running RNN that encodes an irregularly-sampled time series
    into a distribution over initial latent states q(z0 | x_{1:T}).

    Processes observations from t_T back to t_1 (reverse time order).
    At each observation, it updates its hidden state with both the observed
    value and the time gap since the next observation.

    For trading: encodes a history of market observations (possibly with
    missing data or irregular intervals) into a latent representation.

    Args:
        input_dim: Number of observed features per time step
        hidden_dim: RNN hidden state dimension
        latent_dim: Dimension of the latent ODE state z
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # GRU cell for processing observations
        # Input: [observed_value, time_gap, observation_mask]
        self.gru = nn.GRUCell(
            input_size=input_dim + 1,  # +1 for delta_t
            hidden_size=hidden_dim,
        )

        # Map from RNN hidden state to latent distribution parameters
        self.hidden_to_z0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * latent_dim),  # mean + log_var
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observations backward in time to get q(z0).

        Args:
            x: Observations, shape (batch_size, n_obs, input_dim)
               Observations should be in FORWARD time order
            t: Observation times, shape (batch_size, n_obs) or (n_obs,)
            mask: Binary mask, shape (batch_size, n_obs, input_dim)
                  1 = observed, 0 = missing

        Returns:
            z0_mean: Mean of q(z0), shape (batch_size, latent_dim)
            z0_logvar: Log-variance of q(z0), shape (batch_size, latent_dim)
        """
        batch_size = x.shape[0]
        n_obs = x.shape[1]

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Handle time tensor shape
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(batch_size, -1)

        # Process observations in reverse time order
        for i in reversed(range(n_obs)):
            # Compute time gap to previous (in reverse) observation
            if i < n_obs - 1:
                delta_t = t[:, i + 1] - t[:, i]
            else:
                delta_t = torch.zeros(batch_size, device=x.device)

            delta_t = delta_t.unsqueeze(-1)  # (batch_size, 1)

            # Current observation
            x_i = x[:, i, :]  # (batch_size, input_dim)

            # Apply mask if provided (zero out missing values)
            if mask is not None:
                x_i = x_i * mask[:, i, :]

            # Concatenate observation with time gap
            gru_input = torch.cat([x_i, delta_t], dim=-1)

            # Update hidden state
            h = self.gru(gru_input, h)

        # Map to latent distribution parameters
        z0_params = self.hidden_to_z0(h)
        z0_mean = z0_params[:, : self.latent_dim]
        z0_logvar = z0_params[:, self.latent_dim :]

        return z0_mean, z0_logvar


class Decoder(nn.Module):
    """
    Decoder that maps latent states z(t) to observed space x(t).

    For trading: maps continuous latent dynamics back to observable
    market features (price, volume, etc.) at any desired time point.

    Args:
        latent_dim: Dimension of latent state z
        hidden_dim: Decoder hidden dimension
        output_dim: Number of output features
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 5,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to observation space.

        Args:
            z: Latent state, shape (..., latent_dim)

        Returns:
            Decoded observation, shape (..., output_dim)
        """
        return self.net(z)


class LatentODE(nn.Module):
    """
    Latent ODE model for irregularly-sampled financial time series.

    Combines:
    1. A recognition network (backward RNN) to encode observations
    2. A latent ODE that governs dynamics in latent space
    3. A decoder to map back to observation space

    This is a variational autoencoder (VAE) where the decoder's generative
    model uses an ODE solver in latent space.

    For trading applications:
    - Handles irregular timestamps natively (tick data, trading halts)
    - Provides uncertainty estimates through the VAE framework
    - Can interpolate and extrapolate at any time point
    - Continuous-time dynamics capture smooth market evolution

    Args:
        input_dim: Number of observed features (e.g., OHLCV = 5)
        latent_dim: Dimension of the latent ODE state
        hidden_dim: Hidden dimension for all sub-networks
        output_dim: Number of output features to reconstruct
        ode_layers: Number of layers in the ODE dynamics network
        solver: ODE solver method
        use_adjoint: Use adjoint method for memory-efficient backprop
    """

    def __init__(
        self,
        input_dim: int = 5,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 5,
        ode_layers: int = 2,
        solver: str = "dopri5",
        use_adjoint: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.solver = solver
        self.use_adjoint = use_adjoint

        # 1. Recognition network (encoder)
        self.encoder = RecognitionRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        # 2. Latent ODE dynamics: dz/dt = f_theta(z(t), t)
        self.ode_func = ODEFunc(
            hidden_dim=latent_dim,
            n_layers=ode_layers,
            activation="tanh",
            time_dependent=True,
        )

        # 3. Decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def reparameterize(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = mean + std * epsilon."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        return mean

    def forward(
        self,
        x: torch.Tensor,
        t_obs: torch.Tensor,
        t_pred: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode, solve ODE, decode.

        Args:
            x: Observations, shape (batch_size, n_obs, input_dim)
            t_obs: Observation times, shape (batch_size, n_obs) or (n_obs,)
            t_pred: Time points for prediction. If None, uses t_obs
            mask: Observation mask, shape (batch_size, n_obs, input_dim)

        Returns:
            Dictionary with:
                'x_pred': Predicted observations at t_pred
                'z0_mean': Posterior mean
                'z0_logvar': Posterior log-variance
                'z_trajectory': Latent trajectory
        """
        # 1. Encode observations into latent posterior q(z0 | x)
        z0_mean, z0_logvar = self.encoder(x, t_obs, mask)

        # 2. Sample z0 from posterior
        z0 = self.reparameterize(z0_mean, z0_logvar)

        # 3. Determine prediction times
        if t_pred is None:
            t_pred = t_obs if t_obs.dim() == 1 else t_obs[0]

        # Normalize time to [0, 1] for stable ODE integration
        t_min = t_pred.min()
        t_max = t_pred.max()
        t_range = t_max - t_min
        if t_range > 0:
            t_normalized = (t_pred - t_min) / t_range
        else:
            t_normalized = t_pred - t_min

        # 4. Solve ODE forward in latent space
        ode_solver = odeint_adjoint if self.use_adjoint else odeint

        z_trajectory = ode_solver(
            self.ode_func,
            z0,
            t_normalized,
            method=self.solver,
            rtol=1e-4,
            atol=1e-5,
        )
        # z_trajectory: (n_time_points, batch_size, latent_dim)

        # 5. Decode latent trajectory to observation space
        x_pred = self.decoder(z_trajectory)
        # x_pred: (n_time_points, batch_size, output_dim)

        return {
            "x_pred": x_pred,
            "z0_mean": z0_mean,
            "z0_logvar": z0_logvar,
            "z_trajectory": z_trajectory,
        }

    def predict(
        self,
        x: torch.Tensor,
        t_obs: torch.Tensor,
        t_future: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate predictions at future time points.

        Args:
            x: Historical observations, shape (batch_size, n_obs, input_dim)
            t_obs: Observation times
            t_future: Future times to predict at
            mask: Observation mask
            n_samples: Number of samples for uncertainty estimation

        Returns:
            Predictions, shape (n_samples, n_future, batch_size, output_dim)
        """
        self.eval()
        with torch.no_grad():
            z0_mean, z0_logvar = self.encoder(x, t_obs, mask)

            all_preds = []
            for _ in range(n_samples):
                z0 = self.reparameterize(z0_mean, z0_logvar)

                # Combine historical and future times
                if t_obs.dim() > 1:
                    t_all = torch.cat([t_obs[0], t_future])
                else:
                    t_all = torch.cat([t_obs, t_future])

                t_min = t_all.min()
                t_range = t_all.max() - t_min
                t_norm = (t_all - t_min) / t_range if t_range > 0 else t_all - t_min

                z_traj = odeint(
                    self.ode_func,
                    z0,
                    t_norm,
                    method=self.solver,
                )

                # Take only future time points
                n_obs = t_obs.shape[-1]
                z_future = z_traj[n_obs:]
                x_future = self.decoder(z_future)
                all_preds.append(x_future)

            return torch.stack(all_preds)

    @staticmethod
    def compute_loss(
        x_true: torch.Tensor,
        result: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        kl_weight: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss = Reconstruction + KL Divergence.

        Args:
            x_true: True observations, shape (batch_size, n_obs, output_dim)
            result: Forward pass output dictionary
            mask: Observation mask
            kl_weight: Weight for KL divergence term (beta-VAE style)

        Returns:
            Dictionary with 'total', 'recon', and 'kl' losses
        """
        x_pred = result["x_pred"]  # (n_time, batch, output_dim)
        z0_mean = result["z0_mean"]
        z0_logvar = result["z0_logvar"]

        # Reconstruction loss (MSE)
        # x_pred is (n_time, batch, dim), x_true is (batch, n_time, dim)
        x_pred_t = x_pred.permute(1, 0, 2)  # (batch, n_time, dim)

        if mask is not None:
            recon_loss = ((x_pred_t - x_true) ** 2 * mask).sum() / mask.sum()
        else:
            recon_loss = F.mse_loss(x_pred_t, x_true)

        # KL divergence: KL(q(z0) || p(z0)) where p(z0) = N(0, I)
        kl_loss = -0.5 * torch.sum(
            1 + z0_logvar - z0_mean.pow(2) - z0_logvar.exp()
        ) / z0_mean.shape[0]

        total_loss = recon_loss + kl_weight * kl_loss

        return {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
        }
