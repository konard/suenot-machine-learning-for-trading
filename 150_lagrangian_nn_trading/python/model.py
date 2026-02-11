"""
Lagrangian Neural Networks for Trading.

Implements:
  - LagrangianNN: Conservative LNN that learns L_theta(q, q-dot)
  - DissipativeLNN: LNN with Rayleigh dissipation for transaction costs
  - ForcedLNN: LNN with external forces for market shocks
  - MultiScaleLNN: Multi-scale Lagrangian for cross-timeframe dynamics

The key insight: d/dt(dL/dq-dot) - dL/dq = 0 gives dynamics.
Unlike HNNs, LNNs work in (q, q-dot) space -- no Legendre transform needed.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    activation: str = "softplus",
) -> nn.Sequential:
    """Build a multi-layer perceptron with smooth activations."""
    act_fn = {
        "softplus": nn.Softplus,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
    }[activation]

    layers = [nn.Linear(input_dim, hidden_dim), act_fn()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act_fn()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def compute_mass_matrix(
    l_net: nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the mass matrix M = d^2L / dq-dot^2.

    Args:
        l_net: Network that computes L(q, q-dot) as a scalar.
        q: Generalized coordinates (batch, dim).
        qdot: Generalized velocities (batch, dim).

    Returns:
        M: Mass matrix (batch, dim, dim).
    """
    q = q.requires_grad_(True)
    qdot = qdot.requires_grad_(True)

    x = torch.cat([q, qdot], dim=-1)
    L = l_net(x)

    # dL/dq-dot
    dL_dqdot = torch.autograd.grad(
        L.sum(), qdot, create_graph=True, retain_graph=True
    )[0]

    dim = q.shape[-1]
    rows = []
    for i in range(dim):
        row = torch.autograd.grad(
            dL_dqdot[:, i].sum(), qdot, create_graph=True, retain_graph=True
        )[0]
        rows.append(row)

    M = torch.stack(rows, dim=-2)  # (batch, dim, dim)
    return M


def euler_lagrange_acceleration(
    l_net: nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    mass_reg: float = 1e-4,
) -> torch.Tensor:
    """
    Compute acceleration q-ddot via the Euler-Lagrange equations.

    q-ddot = M^{-1} * [dL/dq - C * q-dot]

    where:
      M = d^2L/dq-dot^2     (mass matrix)
      C = d^2L/dq-dot dq    (Coriolis/centrifugal matrix)

    Args:
        l_net: Network computing L(q, q-dot).
        q: Generalized coordinates (batch, dim).
        qdot: Generalized velocities (batch, dim).
        mass_reg: Regularization for mass matrix invertibility.

    Returns:
        qddot: Predicted acceleration (batch, dim).
    """
    q = q.requires_grad_(True)
    qdot = qdot.requires_grad_(True)

    x = torch.cat([q, qdot], dim=-1)
    L = l_net(x)

    # First derivatives
    dL_dq = torch.autograd.grad(
        L.sum(), q, create_graph=True, retain_graph=True
    )[0]
    dL_dqdot = torch.autograd.grad(
        L.sum(), qdot, create_graph=True, retain_graph=True
    )[0]

    dim = q.shape[-1]
    batch = q.shape[0]

    # Second derivatives: mass matrix M and cross term C
    M_rows = []
    C_rows = []
    for i in range(dim):
        # d(dL/dq-dot_i) / dq-dot  --> row of M
        row_M = torch.autograd.grad(
            dL_dqdot[:, i].sum(), qdot, create_graph=True, retain_graph=True
        )[0]
        # d(dL/dq-dot_i) / dq  --> row of C
        row_C = torch.autograd.grad(
            dL_dqdot[:, i].sum(), q, create_graph=True, retain_graph=True
        )[0]
        M_rows.append(row_M)
        C_rows.append(row_C)

    M = torch.stack(M_rows, dim=-2)  # (batch, dim, dim)
    C = torch.stack(C_rows, dim=-2)  # (batch, dim, dim)

    # Regularize mass matrix for invertibility
    M = M + mass_reg * torch.eye(dim, device=q.device).unsqueeze(0)

    # RHS = dL/dq - C * q-dot
    rhs = dL_dq - torch.bmm(C, qdot.unsqueeze(-1)).squeeze(-1)

    # Solve M * q-ddot = rhs
    qddot = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)

    return qddot


class LagrangianNN(nn.Module):
    """
    Lagrangian Neural Network.

    Learns L_theta(q, q-dot) as a scalar function.
    Dynamics derived via Euler-Lagrange equations using double autograd.

    Unlike HNN:
      - Works in (q, q-dot) space, not (q, p)
      - No Legendre transform needed
      - Handles non-separable kinetic/potential coupling naturally
    """

    def __init__(
        self,
        coord_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "softplus",
        mass_reg: float = 1e-4,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.mass_reg = mass_reg

        input_dim = 2 * coord_dim
        self.l_net = build_mlp(input_dim, hidden_dim, 1, num_layers, activation)

    def lagrangian(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Compute the Lagrangian L(q, q-dot)."""
        x = torch.cat([q, qdot], dim=-1)
        return self.l_net(x)

    def energy(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """
        Compute the conserved energy E = q-dot * dL/dq-dot - L.

        This is conserved when L has no explicit time dependence (Noether's theorem).
        """
        q = q.requires_grad_(True)
        qdot = qdot.requires_grad_(True)

        L = self.lagrangian(q, qdot)
        dL_dqdot = torch.autograd.grad(
            L.sum(), qdot, create_graph=True, retain_graph=True
        )[0]

        # E = q-dot * dL/dq-dot - L
        E = (qdot * dL_dqdot).sum(dim=-1, keepdim=True) - L
        return E

    def time_derivative(
        self, q: torch.Tensor, qdot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (dq/dt, dq-dot/dt) = (q-dot, q-ddot).

        dq/dt = q-dot (by definition)
        dq-dot/dt = q-ddot from Euler-Lagrange equations
        """
        qddot = euler_lagrange_acceleration(
            self.l_net, q, qdot, mass_reg=self.mass_reg
        )
        return qdot, qddot

    def forward(
        self, q: torch.Tensor, qdot: torch.Tensor
    ) -> torch.Tensor:
        """Compute acceleration q-ddot."""
        return euler_lagrange_acceleration(
            self.l_net, q, qdot, mass_reg=self.mass_reg
        )

    def mass_matrix(
        self, q: torch.Tensor, qdot: torch.Tensor
    ) -> torch.Tensor:
        """Compute the mass matrix M = d^2L/dq-dot^2."""
        return compute_mass_matrix(self.l_net, q, qdot)


class DissipativeLNN(nn.Module):
    """
    Dissipative Lagrangian Neural Network.

    Extends LNN with a Rayleigh dissipation function D(q, q-dot) >= 0:
      Modified Euler-Lagrange:
        d/dt(dL/dq-dot) - dL/dq = -dD/dq-dot

    In trading context:
      - D models transaction costs, slippage, market impact
      - Larger velocity (rapid price changes) --> more friction
    """

    def __init__(
        self,
        coord_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "softplus",
        mass_reg: float = 1e-4,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.mass_reg = mass_reg

        input_dim = 2 * coord_dim
        self.l_net = build_mlp(input_dim, hidden_dim, 1, num_layers, activation)
        self.d_net = build_mlp(input_dim, hidden_dim, 1, num_layers, activation)

    def lagrangian(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Compute the Lagrangian L(q, q-dot)."""
        x = torch.cat([q, qdot], dim=-1)
        return self.l_net(x)

    def dissipation(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Compute Rayleigh dissipation D(q, q-dot) >= 0."""
        x = torch.cat([q, qdot], dim=-1)
        return nn.functional.softplus(self.d_net(x))

    def energy(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Compute E = q-dot * dL/dq-dot - L (decreasing due to dissipation)."""
        q = q.requires_grad_(True)
        qdot = qdot.requires_grad_(True)
        L = self.lagrangian(q, qdot)
        dL_dqdot = torch.autograd.grad(
            L.sum(), qdot, create_graph=True, retain_graph=True
        )[0]
        E = (qdot * dL_dqdot).sum(dim=-1, keepdim=True) - L
        return E

    def time_derivative(
        self, q: torch.Tensor, qdot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (dq/dt, dq-dot/dt) with dissipation."""
        q_grad = q.requires_grad_(True)
        qdot_grad = qdot.requires_grad_(True)

        # Conservative acceleration
        qddot = euler_lagrange_acceleration(
            self.l_net, q_grad, qdot_grad, mass_reg=self.mass_reg
        )

        # Dissipation gradient
        D = self.dissipation(q_grad, qdot_grad)
        dD_dqdot = torch.autograd.grad(
            D.sum(), qdot_grad, create_graph=True, retain_graph=True
        )[0]

        # Mass matrix for converting dissipation force to acceleration
        M = compute_mass_matrix(self.l_net, q_grad, qdot_grad)
        M = M + self.mass_reg * torch.eye(
            self.coord_dim, device=q.device
        ).unsqueeze(0)

        # Dissipative correction
        dissip_accel = torch.linalg.solve(
            M, dD_dqdot.unsqueeze(-1)
        ).squeeze(-1)

        qddot_total = qddot - dissip_accel
        return qdot, qddot_total

    def forward(
        self, q: torch.Tensor, qdot: torch.Tensor
    ) -> torch.Tensor:
        """Compute acceleration q-ddot with dissipation."""
        _, qddot = self.time_derivative(q, qdot)
        return qddot


class ForcedLNN(nn.Module):
    """
    Forced Lagrangian Neural Network.

    Extends DissipativeLNN with learned external forces Q_theta(t, q, q-dot, u):
      d/dt(dL/dq-dot) - dL/dq = -dD/dq-dot + Q(q, q-dot, u)

    In trading:
      - Q models external shocks (news, earnings, FOMC, etc.)
      - u = external features (sentiment, macro indicators)
    """

    def __init__(
        self,
        coord_dim: int = 1,
        external_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "softplus",
        mass_reg: float = 1e-4,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.external_dim = external_dim
        self.mass_reg = mass_reg

        input_dim = 2 * coord_dim
        self.l_net = build_mlp(input_dim, hidden_dim, 1, num_layers, activation)
        self.d_net = build_mlp(input_dim, hidden_dim, 1, num_layers, activation)
        self.force_net = build_mlp(
            input_dim + external_dim, hidden_dim, coord_dim, num_layers, activation
        )

    def lagrangian(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Compute L(q, q-dot)."""
        x = torch.cat([q, qdot], dim=-1)
        return self.l_net(x)

    def dissipation(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Compute D(q, q-dot) >= 0."""
        x = torch.cat([q, qdot], dim=-1)
        return nn.functional.softplus(self.d_net(x))

    def external_force(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute generalized force Q(q, q-dot, u)."""
        x = torch.cat([q, qdot, u], dim=-1)
        return self.force_net(x)

    def energy(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Compute E = q-dot * dL/dq-dot - L."""
        q = q.requires_grad_(True)
        qdot = qdot.requires_grad_(True)
        L = self.lagrangian(q, qdot)
        dL_dqdot = torch.autograd.grad(
            L.sum(), qdot, create_graph=True, retain_graph=True
        )[0]
        E = (qdot * dL_dqdot).sum(dim=-1, keepdim=True) - L
        return E

    def time_derivative(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        u: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (dq/dt, dq-dot/dt) with dissipation and external force."""
        q_grad = q.requires_grad_(True)
        qdot_grad = qdot.requires_grad_(True)

        # Conservative + dissipative acceleration
        qddot = euler_lagrange_acceleration(
            self.l_net, q_grad, qdot_grad, mass_reg=self.mass_reg
        )

        # Dissipation
        D = self.dissipation(q_grad, qdot_grad)
        dD_dqdot = torch.autograd.grad(
            D.sum(), qdot_grad, create_graph=True, retain_graph=True
        )[0]

        M = compute_mass_matrix(self.l_net, q_grad, qdot_grad)
        M = M + self.mass_reg * torch.eye(
            self.coord_dim, device=q.device
        ).unsqueeze(0)

        dissip_accel = torch.linalg.solve(
            M, dD_dqdot.unsqueeze(-1)
        ).squeeze(-1)

        qddot_total = qddot - dissip_accel

        # External force
        if u is not None:
            Q = self.external_force(q_grad, qdot_grad, u)
            force_accel = torch.linalg.solve(
                M, Q.unsqueeze(-1)
            ).squeeze(-1)
            qddot_total = qddot_total + force_accel

        return qdot, qddot_total

    def forward(
        self,
        q: torch.Tensor,
        qdot: torch.Tensor,
        u: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute acceleration q-ddot."""
        _, qddot = self.time_derivative(q, qdot, u)
        return qddot


class MultiScaleLNN(nn.Module):
    """
    Multi-Scale Lagrangian Neural Network.

    Learns L(q_1, ..., q_K, q-dot_1, ..., q-dot_K) where each q_k
    represents deviation at a different timescale (MA window).

    This captures cross-scale interactions automatically.
    """

    def __init__(
        self,
        n_scales: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: str = "softplus",
        mass_reg: float = 1e-4,
    ):
        super().__init__()
        self.coord_dim = n_scales
        self.mass_reg = mass_reg

        input_dim = 2 * n_scales
        self.l_net = build_mlp(input_dim, hidden_dim, 1, num_layers, activation)

    def lagrangian(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Compute L(q, q-dot) for multi-scale coordinates."""
        x = torch.cat([q, qdot], dim=-1)
        return self.l_net(x)

    def energy(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Compute E = q-dot * dL/dq-dot - L."""
        q = q.requires_grad_(True)
        qdot = qdot.requires_grad_(True)
        L = self.lagrangian(q, qdot)
        dL_dqdot = torch.autograd.grad(
            L.sum(), qdot, create_graph=True, retain_graph=True
        )[0]
        E = (qdot * dL_dqdot).sum(dim=-1, keepdim=True) - L
        return E

    def time_derivative(
        self, q: torch.Tensor, qdot: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (dq/dt, dq-dot/dt) for multi-scale system."""
        qddot = euler_lagrange_acceleration(
            self.l_net, q, qdot, mass_reg=self.mass_reg
        )
        return qdot, qddot

    def forward(
        self, q: torch.Tensor, qdot: torch.Tensor
    ) -> torch.Tensor:
        """Compute acceleration q-ddot."""
        return euler_lagrange_acceleration(
            self.l_net, q, qdot, mass_reg=self.mass_reg
        )


# =============================================================================
# Loss Functions
# =============================================================================


def compute_lnn_loss(
    model: nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    qddot_target: torch.Tensor,
    energy_reg: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute training loss for LNN.

    Loss = MSE(q-ddot_pred, q-ddot_target) + energy_reg * Var(E)

    Args:
        model: LNN model (LagrangianNN, DissipativeLNN, etc.)
        q: Coordinates (batch, dim).
        qdot: Velocities (batch, dim).
        qddot_target: Target accelerations (batch, dim).
        energy_reg: Weight for energy conservation regularization.

    Returns:
        loss: Total loss.
        metrics: Dictionary of loss components.
    """
    qddot_pred = model(q, qdot)

    # Acceleration matching loss
    loss_accel = ((qddot_pred - qddot_target) ** 2).mean()

    metrics = {"loss_accel": loss_accel.item()}

    total_loss = loss_accel

    # Energy conservation regularization
    if energy_reg > 0:
        E = model.energy(q, qdot)
        energy_var = E.var()
        total_loss = total_loss + energy_reg * energy_var
        metrics["energy_var"] = energy_var.item()

    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


def compute_dissipative_loss(
    model: DissipativeLNN,
    q: torch.Tensor,
    qdot: torch.Tensor,
    qddot_target: torch.Tensor,
    dissipation_reg: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute training loss for Dissipative LNN.

    Adds a regularization term that encourages non-trivial dissipation.
    """
    qddot_pred = model(q, qdot)
    loss_accel = ((qddot_pred - qddot_target) ** 2).mean()

    # Encourage dissipation to be positive but not too large
    D = model.dissipation(q, qdot)
    dissip_loss = dissipation_reg * (D.mean() - D.var())

    total_loss = loss_accel + dissip_loss.abs()

    metrics = {
        "loss_accel": loss_accel.item(),
        "dissipation_mean": D.mean().item(),
        "dissipation_var": D.var().item(),
        "loss_total": total_loss.item(),
    }
    return total_loss, metrics


def compute_forced_loss(
    model: ForcedLNN,
    q: torch.Tensor,
    qdot: torch.Tensor,
    qddot_target: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    force_reg: float = 0.001,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute training loss for Forced LNN.

    Includes force regularization to prevent overfitting to external signal.
    """
    qddot_pred = model(q, qdot, u)
    loss_accel = ((qddot_pred - qddot_target) ** 2).mean()

    total_loss = loss_accel

    metrics = {"loss_accel": loss_accel.item()}

    # Force regularization
    if u is not None and force_reg > 0:
        Q = model.external_force(q, qdot, u)
        force_penalty = force_reg * (Q ** 2).mean()
        total_loss = total_loss + force_penalty
        metrics["force_magnitude"] = (Q ** 2).mean().item()

    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


# =============================================================================
# Integration (for prediction)
# =============================================================================


def integrate_trajectory(
    model: nn.Module,
    q0: torch.Tensor,
    qdot0: torch.Tensor,
    dt: float = 0.1,
    n_steps: int = 100,
    method: str = "rk4",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate a trajectory using the learned Lagrangian dynamics.

    Args:
        model: LNN model with time_derivative method.
        q0: Initial coordinates (1, dim).
        qdot0: Initial velocities (1, dim).
        dt: Time step.
        n_steps: Number of integration steps.
        method: Integration method ("euler", "rk4", "leapfrog").

    Returns:
        traj_q: Trajectory of q, shape (n_steps+1, dim).
        traj_qdot: Trajectory of q-dot, shape (n_steps+1, dim).
    """
    traj_q = [q0.detach()]
    traj_qdot = [qdot0.detach()]

    q = q0.clone()
    qdot = qdot0.clone()

    for _ in range(n_steps):
        if method == "euler":
            q, qdot = _euler_step(model, q, qdot, dt)
        elif method == "rk4":
            q, qdot = _rk4_step(model, q, qdot, dt)
        elif method == "leapfrog":
            q, qdot = _leapfrog_step(model, q, qdot, dt)
        else:
            raise ValueError(f"Unknown method: {method}")

        traj_q.append(q.detach())
        traj_qdot.append(qdot.detach())

    return torch.stack(traj_q, dim=0).squeeze(1), torch.stack(traj_qdot, dim=0).squeeze(1)


def _euler_step(
    model: nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Euler integration step."""
    with torch.enable_grad():
        _, qddot = model.time_derivative(q, qdot)
    q_new = q + qdot * dt
    qdot_new = qdot + qddot * dt
    return q_new.detach().requires_grad_(True), qdot_new.detach().requires_grad_(True)


def _rk4_step(
    model: nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runge-Kutta 4th order integration step."""
    def deriv(q_in, qdot_in):
        with torch.enable_grad():
            _, qddot = model.time_derivative(q_in, qdot_in)
        return qdot_in.detach(), qddot.detach()

    k1_q, k1_v = deriv(q, qdot)
    k2_q, k2_v = deriv(q + 0.5 * dt * k1_q, qdot + 0.5 * dt * k1_v)
    k3_q, k3_v = deriv(q + 0.5 * dt * k2_q, qdot + 0.5 * dt * k2_v)
    k4_q, k4_v = deriv(q + dt * k3_q, qdot + dt * k3_v)

    q_new = q + (dt / 6) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    qdot_new = qdot + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return q_new.detach().requires_grad_(True), qdot_new.detach().requires_grad_(True)


def _leapfrog_step(
    model: nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Leapfrog (Stormer-Verlet) integration step."""
    with torch.enable_grad():
        _, qddot_0 = model.time_derivative(q, qdot)
    qddot_0 = qddot_0.detach()

    # Half-step velocity
    qdot_half = qdot + 0.5 * dt * qddot_0

    # Full-step position
    q_new = q + dt * qdot_half

    # Full-step acceleration
    with torch.enable_grad():
        _, qddot_1 = model.time_derivative(
            q_new.requires_grad_(True), qdot_half.requires_grad_(True)
        )
    qddot_1 = qddot_1.detach()

    # Half-step velocity
    qdot_new = qdot_half + 0.5 * dt * qddot_1

    return q_new.requires_grad_(True), qdot_new.requires_grad_(True)


def compute_energy_along_trajectory(
    model: nn.Module,
    traj_q: torch.Tensor,
    traj_qdot: torch.Tensor,
) -> torch.Tensor:
    """Compute energy at each point along a trajectory."""
    energies = []
    for i in range(traj_q.shape[0]):
        q_i = traj_q[i:i+1]
        qdot_i = traj_qdot[i:i+1]
        with torch.enable_grad():
            E = model.energy(q_i, qdot_i)
        energies.append(E.detach())
    return torch.cat(energies, dim=0)
