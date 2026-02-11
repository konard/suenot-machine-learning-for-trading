"""
PINN Architecture for CIR Bond Pricing PDE
============================================

Implements a Physics-Informed Neural Network that solves the CIR PDE:

    dP/dt + kappa*(theta - r)*dP/dr + 0.5*sigma^2*r*d^2P/dr^2 - r*P = 0

with terminal condition P(r, T) = 1 and appropriate boundary conditions.

Key design decisions:
- Include sqrt(r) as auxiliary input to capture the CIR volatility structure
- Sigmoid output to enforce P in (0, 1]
- Tanh activations for smooth second derivatives
- sqrt(r)-aware collocation point sampling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


class CIRPINNNetwork(nn.Module):
    """
    Neural network backbone for the CIR PINN.

    Takes (r_norm, t_norm, sqrt_r_norm) as inputs and outputs bond price P.

    Parameters
    ----------
    hidden_layers : list of int
        Number of neurons in each hidden layer.
    activation : str
        Activation function: 'tanh' or 'sin'.
    use_sqrt_r_input : bool
        Whether to include sqrt(r) as auxiliary input.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        activation: str = "tanh",
        use_sqrt_r_input: bool = True,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 128, 128, 64]

        self.use_sqrt_r_input = use_sqrt_r_input
        input_dim = 3 if use_sqrt_r_input else 2

        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sin":
                layers.append(SinActivation())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_dim = hidden_dim

        # Output layer with sigmoid to ensure P in (0, 1)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, r_norm: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        r_norm : Tensor of shape (N,) or (N, 1)
            Normalized rate values in [0, 1].
        t_norm : Tensor of shape (N,) or (N, 1)
            Normalized time values in [0, 1].

        Returns
        -------
        Tensor of shape (N, 1)
            Predicted bond prices.
        """
        r_norm = r_norm.view(-1, 1)
        t_norm = t_norm.view(-1, 1)

        if self.use_sqrt_r_input:
            sqrt_r_norm = torch.sqrt(r_norm + 1e-8)
            x = torch.cat([r_norm, t_norm, sqrt_r_norm], dim=1)
        else:
            x = torch.cat([r_norm, t_norm], dim=1)

        return self.network(x)


class SinActivation(nn.Module):
    """Sinusoidal activation function: sin(x). Good for periodic features."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class CIRPINN:
    """
    Physics-Informed Neural Network for the CIR bond pricing PDE.

    Solves:
        dP/dt + kappa*(theta - r)*dP/dr + 0.5*sigma^2*r*d^2P/dr^2 - r*P = 0

    with P(r, T) = 1.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run mean rate.
    sigma : float
        Volatility coefficient.
    T : float
        Bond maturity.
    r_max : float
        Maximum rate for the computational domain.
    hidden_layers : list of int, optional
        Architecture specification.
    device : str
        Torch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        kappa: float = 0.5,
        theta: float = 0.05,
        sigma: float = 0.1,
        T: float = 5.0,
        r_max: float = 0.20,
        hidden_layers: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.T = T
        self.r_max = r_max
        self.device = torch.device(device)

        # Feller condition check
        feller_lhs = 2.0 * kappa * theta
        feller_rhs = sigma ** 2
        self.feller_satisfied = feller_lhs >= feller_rhs
        if not self.feller_satisfied:
            print(
                f"WARNING: Feller condition violated! "
                f"2*kappa*theta={feller_lhs:.4f} < sigma^2={feller_rhs:.4f}"
            )

        # Create network
        self.net = CIRPINNNetwork(
            hidden_layers=hidden_layers,
            activation="tanh",
            use_sqrt_r_input=True,
        ).to(self.device)

    def _normalize_r(self, r: torch.Tensor) -> torch.Tensor:
        """Normalize rate to [0, 1]."""
        return r / self.r_max

    def _normalize_t(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize time to [0, 1]."""
        return t / self.T

    def predict(
        self,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict bond price P(r, t).

        Parameters
        ----------
        r : Tensor
            Short rate values.
        t : Tensor
            Time values.

        Returns
        -------
        Tensor
            Predicted bond prices.
        """
        self.net.eval()
        with torch.no_grad():
            r_norm = self._normalize_r(r.to(self.device))
            t_norm = self._normalize_t(t.to(self.device))
            P = self.net(r_norm, t_norm)
        return P.squeeze(-1)

    def compute_pde_residual(
        self,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the CIR PDE residual at given points.

        PDE: dP/dt + kappa*(theta - r)*dP/dr + 0.5*sigma^2*r*d^2P/dr^2 - r*P = 0

        Parameters
        ----------
        r : Tensor of shape (N,), requires_grad=True
            Short rate values.
        t : Tensor of shape (N,), requires_grad=True
            Time values.

        Returns
        -------
        Tuple[Tensor, Tensor]
            (P, residual) where P is the predicted price and residual is the PDE residual.
        """
        r_norm = self._normalize_r(r)
        t_norm = self._normalize_t(t)

        P = self.net(r_norm, t_norm)

        # Compute derivatives via autograd
        # dP/dr = dP/d(r_norm) * d(r_norm)/dr = dP/d(r_norm) / r_max
        # But since r_norm is a function of r through the normalization,
        # and we need chain rule through the network, we compute directly.
        dP_dr_norm = torch.autograd.grad(
            P, r, grad_outputs=torch.ones_like(P), create_graph=True
        )[0]

        dP_dt_norm = torch.autograd.grad(
            P, t, grad_outputs=torch.ones_like(P), create_graph=True
        )[0]

        d2P_dr2_norm = torch.autograd.grad(
            dP_dr_norm, r, grad_outputs=torch.ones_like(dP_dr_norm), create_graph=True
        )[0]

        # CIR PDE residual
        # dP/dt + kappa*(theta - r)*dP/dr + 0.5*sigma^2*r*d2P/dr2 - r*P = 0
        residual = (
            dP_dt_norm
            + self.kappa * (self.theta - r) * dP_dr_norm
            + 0.5 * self.sigma ** 2 * r * d2P_dr2_norm
            - r * P.squeeze(-1)
        )

        return P.squeeze(-1), residual

    def generate_collocation_points(
        self,
        n_interior: int = 5000,
        n_boundary: int = 500,
        n_terminal: int = 500,
        sqrt_sampling: bool = True,
    ) -> dict:
        """
        Generate collocation points for PINN training.

        Uses sqrt-aware sampling to concentrate points near r = 0.

        Parameters
        ----------
        n_interior : int
            Number of interior collocation points.
        n_boundary : int
            Number of boundary points (each boundary).
        n_terminal : int
            Number of terminal condition points.
        sqrt_sampling : bool
            If True, use quadratic sampling to concentrate near r = 0.

        Returns
        -------
        dict
            Dictionary with keys 'interior', 'terminal', 'lower_boundary', 'upper_boundary'.
        """
        device = self.device

        # Interior points: (r, t) with sqrt(r)-aware sampling
        if sqrt_sampling:
            # Quadratic sampling concentrates near r = 0
            u = torch.rand(n_interior, device=device)
            r_interior = self.r_max * (u ** 2)
        else:
            r_interior = torch.rand(n_interior, device=device) * self.r_max

        t_interior = torch.rand(n_interior, device=device) * self.T
        r_interior.requires_grad_(True)
        t_interior.requires_grad_(True)

        # Terminal condition: P(r, T) = 1
        if sqrt_sampling:
            u = torch.rand(n_terminal, device=device)
            r_terminal = self.r_max * (u ** 2)
        else:
            r_terminal = torch.rand(n_terminal, device=device) * self.r_max
        t_terminal = torch.full((n_terminal,), self.T, device=device)

        # Lower boundary: r = 0
        r_lower = torch.zeros(n_boundary, device=device)
        t_lower = torch.rand(n_boundary, device=device) * self.T
        r_lower.requires_grad_(True)
        t_lower.requires_grad_(True)

        # Upper boundary: r = r_max, P -> 0
        r_upper = torch.full((n_boundary,), self.r_max, device=device)
        t_upper = torch.rand(n_boundary, device=device) * self.T

        return {
            "interior": (r_interior, t_interior),
            "terminal": (r_terminal, t_terminal),
            "lower_boundary": (r_lower, t_lower),
            "upper_boundary": (r_upper, t_upper),
        }

    def compute_loss(
        self,
        collocation: dict,
        w_pde: float = 1.0,
        w_terminal: float = 10.0,
        w_boundary: float = 1.0,
        w_analytical: float = 0.0,
        analytical_points: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total PINN loss.

        Parameters
        ----------
        collocation : dict
            Output of generate_collocation_points().
        w_pde, w_terminal, w_boundary, w_analytical : float
            Loss weights.
        analytical_points : tuple, optional
            (r, t, P_analytical) for supervised anchoring.

        Returns
        -------
        Tuple[Tensor, dict]
            Total loss and dictionary of individual loss components.
        """
        losses = {}

        # 1. PDE residual loss (interior)
        r_int, t_int = collocation["interior"]
        P_int, residual = self.compute_pde_residual(r_int, t_int)

        # Weight residuals more near r = 0
        weights = 1.0 / (r_int.detach() / self.r_max + 0.1)
        weights = weights / weights.mean()
        losses["pde"] = torch.mean(weights * residual ** 2)

        # 2. Terminal condition loss: P(r, T) = 1
        r_term, t_term = collocation["terminal"]
        r_term_norm = self._normalize_r(r_term)
        t_term_norm = self._normalize_t(t_term)
        P_term = self.net(r_term_norm, t_term_norm).squeeze(-1)
        losses["terminal"] = torch.mean((P_term - 1.0) ** 2)

        # 3. Boundary condition loss
        # Lower boundary: reduced PDE at r = 0
        r_low, t_low = collocation["lower_boundary"]
        P_low, residual_low = self.compute_pde_residual(r_low, t_low)
        # At r=0, the PDE reduces to: dP/dt + kappa*theta*dP/dr = 0
        # We enforce this through the PDE residual which naturally simplifies at r=0
        losses["lower_bc"] = torch.mean(residual_low ** 2)

        # Upper boundary: P(r_max, t) -> 0 (high rates make bonds worthless)
        r_up, t_up = collocation["upper_boundary"]
        r_up_norm = self._normalize_r(r_up)
        t_up_norm = self._normalize_t(t_up)
        P_up = self.net(r_up_norm, t_up_norm).squeeze(-1)
        # Target: bond price at very high rates should be very small
        from .analytical import cir_bond_price as cir_price_np
        target_up = cir_price_np(
            self.r_max, 0.0, self.T, self.kappa, self.theta, self.sigma
        )
        losses["upper_bc"] = torch.mean((P_up - target_up) ** 2)

        # 4. Analytical constraint loss (optional)
        if w_analytical > 0 and analytical_points is not None:
            r_a, t_a, P_a = analytical_points
            r_a_norm = self._normalize_r(r_a.to(self.device))
            t_a_norm = self._normalize_t(t_a.to(self.device))
            P_pred = self.net(r_a_norm, t_a_norm).squeeze(-1)
            losses["analytical"] = torch.mean((P_pred - P_a.to(self.device)) ** 2)
        else:
            losses["analytical"] = torch.tensor(0.0, device=self.device)

        # Total loss
        total = (
            w_pde * losses["pde"]
            + w_terminal * losses["terminal"]
            + w_boundary * (losses["lower_bc"] + losses["upper_bc"])
            + w_analytical * losses["analytical"]
        )

        losses["total"] = total
        return total, losses

    def get_parameters(self) -> list:
        """Return network parameters for optimizer."""
        return list(self.net.parameters())

    def save(self, path: str):
        """Save model to file."""
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "kappa": self.kappa,
                "theta": self.theta,
                "sigma": self.sigma,
                "T": self.T,
                "r_max": self.r_max,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.kappa = checkpoint["kappa"]
        self.theta = checkpoint["theta"]
        self.sigma = checkpoint["sigma"]
        self.T = checkpoint["T"]
        self.r_max = checkpoint["r_max"]
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Quick test
    pinn = CIRPINN(kappa=0.5, theta=0.05, sigma=0.1, T=5.0)
    print(f"PINN created with {sum(p.numel() for p in pinn.get_parameters())} parameters")
    print(f"Feller condition satisfied: {pinn.feller_satisfied}")

    # Test prediction
    r_test = torch.linspace(0.01, 0.15, 10)
    t_test = torch.zeros(10)
    P_pred = pinn.predict(r_test, t_test)
    print(f"\nPredicted bond prices (untrained): {P_pred}")

    # Test collocation
    collocation = pinn.generate_collocation_points(n_interior=100, n_boundary=20, n_terminal=20)
    print(f"\nCollocation points generated:")
    for key, (r, t) in collocation.items():
        print(f"  {key}: {r.shape[0]} points, r in [{r.min():.4f}, {r.max():.4f}]")
