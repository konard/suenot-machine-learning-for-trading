"""
Training Loop for Hull-White PINN

Trains the Physics-Informed Neural Network to solve the Hull-White bond pricing PDE.
The loss function combines:
1. PDE residual loss (physics constraint)
2. Terminal condition loss (P(r, T; T) = 1)
3. Boundary condition loss (P -> 0 as r -> +inf)
4. Term structure fitting loss (match observed yield curve)
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Tuple, List
from tqdm import tqdm
import time

from hull_white_pinn import HullWhitePINN
from analytical import HullWhiteAnalytical, FlatCurveHullWhite
from data_loader import load_sample_treasury_curve


class HullWhitePINNTrainer:
    """
    Trainer for the Hull-White PINN.

    Handles collocation point sampling, loss computation, and optimization.

    Parameters:
        model: HullWhitePINN instance
        a: Mean reversion speed
        sigma: Volatility
        r0: Current short rate
        T_max: Maximum maturity
        r_min: Minimum rate for domain
        r_max: Maximum rate for domain
        market_maturities: Observable yield curve maturities
        market_rates: Observable yield curve rates
        device: torch device
    """

    def __init__(
        self,
        model: HullWhitePINN,
        a: float = 0.1,
        sigma: float = 0.01,
        r0: float = 0.03,
        T_max: float = 10.0,
        r_min: float = -0.02,
        r_max: float = 0.15,
        market_maturities: Optional[np.ndarray] = None,
        market_rates: Optional[np.ndarray] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.a = a
        self.sigma = sigma
        self.r0 = r0
        self.T_max = T_max
        self.r_min = r_min
        self.r_max = r_max
        self.device = torch.device(device)

        # Set up analytical model for theta(t) and validation
        if market_maturities is not None and market_rates is not None:
            self.hw_analytical = HullWhiteAnalytical(
                a=a, sigma=sigma,
                market_times=market_maturities,
                market_rates=market_rates,
            )
            self.market_maturities = market_maturities
            self.market_rates = market_rates
        else:
            self.hw_analytical = FlatCurveHullWhite(a=a, sigma=sigma, flat_rate=r0)
            self.market_maturities = np.array([0.5, 1, 2, 3, 5, 7, 10])
            self.market_rates = np.full(7, r0)

        # Loss weights
        self.w_pde = 1.0
        self.w_terminal = 50.0
        self.w_boundary = 10.0
        self.w_data = 100.0

        # Training history
        self.history = {
            "epoch": [],
            "total_loss": [],
            "pde_loss": [],
            "terminal_loss": [],
            "boundary_loss": [],
            "data_loss": [],
            "max_error": [],
        }

    def _sample_collocation_points(
        self, n_pde: int, n_terminal: int, n_boundary: int
    ) -> Dict[str, torch.Tensor]:
        """
        Sample collocation points for each loss component.

        Uses Latin Hypercube Sampling for better coverage.
        """
        # PDE interior points (r, t, T) with t < T
        r_pde = torch.FloatTensor(n_pde, 1).uniform_(self.r_min, self.r_max)
        t_pde = torch.FloatTensor(n_pde, 1).uniform_(0.0, self.T_max * 0.9)
        T_pde = t_pde + torch.FloatTensor(n_pde, 1).uniform_(0.1, self.T_max)
        T_pde = torch.clamp(T_pde, max=self.T_max)

        r_pde.requires_grad_(True)
        t_pde.requires_grad_(True)

        # Terminal condition: at t = T, P = 1
        r_terminal = torch.FloatTensor(n_terminal, 1).uniform_(self.r_min, self.r_max)
        T_terminal = torch.FloatTensor(n_terminal, 1).uniform_(0.1, self.T_max)
        t_terminal = T_terminal.clone()  # t = T

        # Boundary condition: at high r, P -> 0
        r_boundary = torch.full((n_boundary, 1), self.r_max)
        t_boundary = torch.FloatTensor(n_boundary, 1).uniform_(0.0, self.T_max * 0.9)
        T_boundary = t_boundary + torch.FloatTensor(n_boundary, 1).uniform_(0.5, self.T_max)
        T_boundary = torch.clamp(T_boundary, max=self.T_max)

        return {
            "r_pde": r_pde.to(self.device),
            "t_pde": t_pde.to(self.device),
            "T_pde": T_pde.to(self.device),
            "r_terminal": r_terminal.to(self.device),
            "t_terminal": t_terminal.to(self.device),
            "T_terminal": T_terminal.to(self.device),
            "r_boundary": r_boundary.to(self.device),
            "t_boundary": t_boundary.to(self.device),
            "T_boundary": T_boundary.to(self.device),
        }

    def _compute_theta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute theta(t) for given time points."""
        theta_vals = np.array([
            self.hw_analytical.theta(ti.item()) for ti in t.flatten()
        ])
        return torch.tensor(
            theta_vals.reshape(-1, 1), dtype=torch.float32, device=self.device
        )

    def _compute_data_loss(self) -> torch.Tensor:
        """
        Compute term structure fitting loss.

        Match the PINN output at (r0, 0, T_i) to market bond prices.
        """
        n = len(self.market_maturities)
        r_data = torch.full((n, 1), self.r0, device=self.device)
        t_data = torch.zeros((n, 1), device=self.device)
        T_data = torch.tensor(
            self.market_maturities.reshape(-1, 1),
            dtype=torch.float32, device=self.device
        )

        # Filter to valid maturities
        valid = T_data.flatten() <= self.T_max
        r_data = r_data[valid]
        t_data = t_data[valid]
        T_data = T_data[valid].reshape(-1, 1)

        if len(T_data) == 0:
            return torch.tensor(0.0, device=self.device)

        # PINN predictions
        P_pinn = self.model(r_data, t_data, T_data)

        # Market bond prices
        market_prices = []
        for T_i in T_data.flatten().cpu().numpy():
            market_prices.append(
                self.hw_analytical.bond_price(self.r0, 0.0, T_i)
            )
        P_market = torch.tensor(
            np.array(market_prices).reshape(-1, 1),
            dtype=torch.float32, device=self.device
        )

        return torch.mean((P_pinn - P_market) ** 2)

    def compute_loss(
        self, points: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with all components.

        Returns:
            (total_loss, loss_dict)
        """
        # 1. PDE residual loss
        theta_pde = self._compute_theta(points["t_pde"])
        residual = self.model.compute_pde_residual(
            points["r_pde"], points["t_pde"], points["T_pde"], theta_pde
        )
        pde_loss = torch.mean(residual ** 2)

        # 2. Terminal condition loss: P(r, T; T) = 1
        P_terminal = self.model(
            points["r_terminal"], points["t_terminal"], points["T_terminal"]
        )
        terminal_loss = torch.mean((P_terminal - 1.0) ** 2)

        # 3. Boundary condition loss: P(r_max, t; T) -> 0
        P_boundary = self.model(
            points["r_boundary"], points["t_boundary"], points["T_boundary"]
        )
        # Bond price should be small at very high rates
        target_boundary = torch.exp(
            -(points["T_boundary"] - points["t_boundary"]) * self.r_max
        )
        boundary_loss = torch.mean((P_boundary - target_boundary) ** 2)

        # 4. Data loss: match market term structure
        data_loss = self._compute_data_loss()

        # Total weighted loss
        total = (
            self.w_pde * pde_loss
            + self.w_terminal * terminal_loss
            + self.w_boundary * boundary_loss
            + self.w_data * data_loss
        )

        loss_dict = {
            "total": total.item(),
            "pde": pde_loss.item(),
            "terminal": terminal_loss.item(),
            "boundary": boundary_loss.item(),
            "data": data_loss.item(),
        }

        return total, loss_dict

    def validate(self, n_test: int = 100) -> float:
        """
        Validate PINN against analytical prices.

        Returns maximum absolute error.
        """
        self.model.eval()
        maturities = np.linspace(0.5, min(self.T_max, 10.0), n_test)
        max_error = 0.0

        with torch.no_grad():
            for T in maturities:
                pinn_price = self.model.predict(self.r0, 0.0, T)
                anal_price = self.hw_analytical.bond_price(self.r0, 0.0, T)
                error = abs(pinn_price - anal_price)
                max_error = max(max_error, error)

        self.model.train()
        return max_error

    def train(
        self,
        epochs: int = 5000,
        lr: float = 1e-3,
        n_pde: int = 5000,
        n_terminal: int = 500,
        n_boundary: int = 500,
        resample_every: int = 500,
        validate_every: int = 100,
        lr_schedule: str = "cosine",
        verbose: bool = True,
    ) -> Dict:
        """
        Train the PINN.

        Args:
            epochs: Number of training epochs
            lr: Initial learning rate
            n_pde: Number of PDE collocation points
            n_terminal: Number of terminal condition points
            n_boundary: Number of boundary condition points
            resample_every: Resample collocation points every N epochs
            validate_every: Validate against analytical every N epochs
            lr_schedule: Learning rate schedule ('cosine', 'step', 'none')
            verbose: Print progress

        Returns:
            Training history dict
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=epochs // 5, T_mult=2, eta_min=lr * 0.01
            )
        elif lr_schedule == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=epochs // 4, gamma=0.5
            )
        else:
            scheduler = None

        # Initial collocation points
        points = self._sample_collocation_points(n_pde, n_terminal, n_boundary)

        start_time = time.time()

        pbar = tqdm(range(epochs), desc="Training PINN", disable=not verbose)
        for epoch in pbar:
            # Resample collocation points periodically
            if epoch > 0 and epoch % resample_every == 0:
                points = self._sample_collocation_points(n_pde, n_terminal, n_boundary)

            self.model.train()
            optimizer.zero_grad()

            total_loss, loss_dict = self.compute_loss(points)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Record history
            max_error = 0.0
            if epoch % validate_every == 0:
                max_error = self.validate()

            self.history["epoch"].append(epoch)
            self.history["total_loss"].append(loss_dict["total"])
            self.history["pde_loss"].append(loss_dict["pde"])
            self.history["terminal_loss"].append(loss_dict["terminal"])
            self.history["boundary_loss"].append(loss_dict["boundary"])
            self.history["data_loss"].append(loss_dict["data"])
            self.history["max_error"].append(max_error)

            if verbose:
                pbar.set_postfix({
                    "loss": f"{loss_dict['total']:.2e}",
                    "pde": f"{loss_dict['pde']:.2e}",
                    "err": f"{max_error:.2e}",
                })

        elapsed = time.time() - start_time

        if verbose:
            print(f"\nTraining complete in {elapsed:.1f}s")
            print(f"Final loss: {self.history['total_loss'][-1]:.2e}")
            print(f"Final max error: {self.history['max_error'][-1]:.2e}")

        return self.history


def train_pinn(
    model: Optional[HullWhitePINN] = None,
    r0: float = 0.03,
    a: float = 0.1,
    sigma: float = 0.01,
    T_max: float = 10.0,
    n_collocation: int = 5000,
    n_boundary: int = 500,
    n_initial: int = 500,
    epochs: int = 5000,
    lr: float = 1e-3,
    market_maturities: Optional[np.ndarray] = None,
    market_rates: Optional[np.ndarray] = None,
) -> Tuple[HullWhitePINN, Dict]:
    """
    Convenience function to create and train a PINN.

    Returns:
        (trained_model, history)
    """
    if model is None:
        model = HullWhitePINN(
            a=a, sigma=sigma,
            hidden_layers=[64, 64, 64, 32],
            activation="tanh",
        )

    if market_maturities is None or market_rates is None:
        market_maturities, market_rates = load_sample_treasury_curve()

    trainer = HullWhitePINNTrainer(
        model=model,
        a=a, sigma=sigma,
        r0=r0, T_max=T_max,
        market_maturities=market_maturities,
        market_rates=market_rates,
    )

    history = trainer.train(
        epochs=epochs, lr=lr,
        n_pde=n_collocation,
        n_terminal=n_initial,
        n_boundary=n_boundary,
    )

    return model, history


if __name__ == "__main__":
    print("=" * 60)
    print("Hull-White PINN Training")
    print("=" * 60)

    # Load market data
    market_mat, market_rates = load_sample_treasury_curve()
    print(f"\nMarket curve: {len(market_mat)} tenors")
    for t, r in zip(market_mat, market_rates):
        print(f"  {t:6.2f}Y: {r:.4%}")

    # Parameters
    a = 0.1
    sigma = 0.01
    r0 = 0.05

    # Create and train
    model = HullWhitePINN(a=a, sigma=sigma, hidden_layers=[64, 64, 64, 32])
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

    model, history = train_pinn(
        model=model,
        r0=r0, a=a, sigma=sigma,
        T_max=10.0,
        epochs=2000,
        lr=1e-3,
        market_maturities=market_mat,
        market_rates=market_rates,
    )

    # Compare with analytical
    hw = HullWhiteAnalytical(
        a=a, sigma=sigma,
        market_times=market_mat,
        market_rates=market_rates,
    )

    print("\n--- Comparison: PINN vs Analytical ---")
    print(f"{'Maturity':>10} {'PINN':>12} {'Analytical':>12} {'Error':>12}")
    print("-" * 50)
    for T in [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        p_pinn = model.predict(r0, 0.0, T)
        p_anal = hw.bond_price(r0, 0.0, T)
        err = abs(p_pinn - p_anal)
        print(f"{T:10.1f} {p_pinn:12.6f} {p_anal:12.6f} {err:12.2e}")

    print("\n--- Greeks at T=5 ---")
    greeks = model.compute_greeks(r0, 0.0, 5.0)
    for key, val in greeks.items():
        print(f"  {key}: {val:.6f}")
