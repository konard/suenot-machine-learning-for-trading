"""
Training Loop for CIR PINN
============================

Implements multi-phase training with:
- Phase 1: Warm start with analytical solutions
- Phase 2: PDE-focused training
- Phase 3: Fine-tuning with adaptive collocation

Features:
- sqrt(r)-aware collocation point sampling
- Cosine annealing learning rate schedule
- Loss component tracking and logging
- Adaptive residual-based resampling
"""

import torch
import torch.optim as optim
import numpy as np
import argparse
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

from .cir_pinn import CIRPINN
from .analytical import cir_bond_price


def generate_analytical_anchors(
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    r_max: float,
    n_points: int = 200,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate analytical CIR bond prices for supervised anchoring.

    Parameters
    ----------
    kappa, theta, sigma : float
        CIR parameters.
    T : float
        Bond maturity.
    r_max : float
        Maximum rate.
    n_points : int
        Number of anchor points.
    device : str
        Torch device.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        (r, t, P_analytical) tensors.
    """
    r_vals = np.random.uniform(0.001, r_max, n_points)
    t_vals = np.random.uniform(0.0, T * 0.99, n_points)

    P_vals = np.array([
        cir_bond_price(r, t, T, kappa, theta, sigma)
        for r, t in zip(r_vals, t_vals)
    ])

    return (
        torch.tensor(r_vals, dtype=torch.float32, device=device),
        torch.tensor(t_vals, dtype=torch.float32, device=device),
        torch.tensor(P_vals, dtype=torch.float32, device=device),
    )


def train_pinn(
    pinn: CIRPINN,
    epochs: int = 5000,
    lr: float = 1e-3,
    n_interior: int = 5000,
    n_boundary: int = 500,
    n_terminal: int = 500,
    n_analytical: int = 200,
    resample_every: int = 500,
    log_every: int = 100,
    save_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Train the CIR PINN with multi-phase schedule.

    Phase 1 (0-20%): Warm start with heavy analytical supervision
    Phase 2 (20-80%): PDE-focused training
    Phase 3 (80-100%): Fine-tuning with balanced weights

    Parameters
    ----------
    pinn : CIRPINN
        The PINN model to train.
    epochs : int
        Total number of training epochs.
    lr : float
        Initial learning rate.
    n_interior : int
        Number of interior collocation points.
    n_boundary : int
        Number of boundary points per boundary.
    n_terminal : int
        Number of terminal condition points.
    n_analytical : int
        Number of analytical anchor points.
    resample_every : int
        Resample collocation points every N epochs.
    log_every : int
        Print loss every N epochs.
    save_path : str, optional
        Path to save the trained model.

    Returns
    -------
    Dict[str, List[float]]
        Dictionary of loss histories for each component.
    """
    pinn.net.train()

    optimizer = optim.Adam(pinn.get_parameters(), lr=lr)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=epochs // 5, T_mult=2, eta_min=lr * 0.01
    )

    # Generate analytical anchor points
    analytical_points = generate_analytical_anchors(
        pinn.kappa, pinn.theta, pinn.sigma, pinn.T, pinn.r_max,
        n_points=n_analytical, device=str(pinn.device),
    )

    # Training phase boundaries
    phase1_end = int(epochs * 0.2)
    phase2_end = int(epochs * 0.8)

    # Loss history
    history = {
        "total": [], "pde": [], "terminal": [],
        "lower_bc": [], "upper_bc": [], "analytical": [],
    }

    # Generate initial collocation points
    collocation = pinn.generate_collocation_points(
        n_interior=n_interior,
        n_boundary=n_boundary,
        n_terminal=n_terminal,
        sqrt_sampling=True,
    )

    pbar = tqdm(range(epochs), desc="Training CIR PINN")
    for epoch in pbar:
        # Determine phase-dependent weights
        if epoch < phase1_end:
            # Phase 1: Heavy analytical supervision
            w_pde = 0.1
            w_terminal = 10.0
            w_boundary = 1.0
            w_analytical = 10.0
        elif epoch < phase2_end:
            # Phase 2: PDE-focused
            w_pde = 10.0
            w_terminal = 5.0
            w_boundary = 2.0
            w_analytical = 1.0
        else:
            # Phase 3: Balanced fine-tuning
            w_pde = 5.0
            w_terminal = 5.0
            w_boundary = 2.0
            w_analytical = 2.0

        # Resample collocation points periodically
        if epoch > 0 and epoch % resample_every == 0:
            collocation = pinn.generate_collocation_points(
                n_interior=n_interior,
                n_boundary=n_boundary,
                n_terminal=n_terminal,
                sqrt_sampling=True,
            )

        # Forward pass and loss computation
        optimizer.zero_grad()
        total_loss, losses = pinn.compute_loss(
            collocation,
            w_pde=w_pde,
            w_terminal=w_terminal,
            w_boundary=w_boundary,
            w_analytical=w_analytical,
            analytical_points=analytical_points,
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(pinn.get_parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Record losses
        for key in history:
            if key in losses:
                history[key].append(losses[key].item())

        # Logging
        if epoch % log_every == 0 or epoch == epochs - 1:
            lr_current = optimizer.param_groups[0]["lr"]
            phase = (
                "Phase 1 (warm)" if epoch < phase1_end
                else "Phase 2 (PDE)" if epoch < phase2_end
                else "Phase 3 (fine)"
            )
            pbar.set_postfix({
                "loss": f"{total_loss.item():.6f}",
                "pde": f"{losses['pde'].item():.6f}",
                "term": f"{losses['terminal'].item():.6f}",
                "lr": f"{lr_current:.6f}",
                "phase": phase,
            })

    # Save model
    if save_path:
        pinn.save(save_path)

    return history


def evaluate_pinn(
    pinn: CIRPINN,
    n_test: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate PINN accuracy against analytical CIR bond prices.

    Parameters
    ----------
    pinn : CIRPINN
        Trained PINN model.
    n_test : int
        Number of test points.

    Returns
    -------
    Dict[str, float]
        Evaluation metrics.
    """
    pinn.net.eval()

    # Generate test points
    r_test = torch.linspace(0.001, pinn.r_max * 0.9, n_test)
    t_test = torch.rand(n_test) * pinn.T * 0.95

    # PINN predictions
    P_pinn = pinn.predict(r_test, t_test).numpy()

    # Analytical prices
    P_analytical = np.array([
        cir_bond_price(r.item(), t.item(), pinn.T, pinn.kappa, pinn.theta, pinn.sigma)
        for r, t in zip(r_test, t_test)
    ])

    # Compute metrics
    abs_error = np.abs(P_pinn - P_analytical)
    rel_error = abs_error / (np.abs(P_analytical) + 1e-10)

    metrics = {
        "max_abs_error": float(np.max(abs_error)),
        "mean_abs_error": float(np.mean(abs_error)),
        "max_rel_error": float(np.max(rel_error)),
        "mean_rel_error": float(np.mean(rel_error)),
        "rmse": float(np.sqrt(np.mean(abs_error ** 2))),
    }

    return metrics


def main():
    """Command-line training interface."""
    parser = argparse.ArgumentParser(description="Train CIR PINN")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--kappa", type=float, default=0.5, help="Mean-reversion speed")
    parser.add_argument("--theta", type=float, default=0.05, help="Long-run mean")
    parser.add_argument("--sigma", type=float, default=0.1, help="Volatility")
    parser.add_argument("--T", type=float, default=5.0, help="Bond maturity")
    parser.add_argument("--r-max", type=float, default=0.20, help="Max rate")
    parser.add_argument("--n-interior", type=int, default=5000, help="Interior points")
    parser.add_argument("--save", type=str, default=None, help="Save path")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    print("=" * 70)
    print("CIR PINN Training")
    print("=" * 70)
    print(f"Parameters: kappa={args.kappa}, theta={args.theta}, sigma={args.sigma}")
    print(f"Maturity: T={args.T}, r_max={args.r_max}")
    print(f"Feller condition: 2*kappa*theta = {2*args.kappa*args.theta:.4f} "
          f"vs sigma^2 = {args.sigma**2:.4f}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print("=" * 70)

    # Create PINN
    pinn = CIRPINN(
        kappa=args.kappa,
        theta=args.theta,
        sigma=args.sigma,
        T=args.T,
        r_max=args.r_max,
        device=args.device,
    )

    n_params = sum(p.numel() for p in pinn.get_parameters())
    print(f"Network parameters: {n_params}")

    # Train
    history = train_pinn(
        pinn,
        epochs=args.epochs,
        lr=args.lr,
        n_interior=args.n_interior,
        save_path=args.save,
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation")
    print("=" * 70)
    metrics = evaluate_pinn(pinn)
    for key, val in metrics.items():
        print(f"  {key}: {val:.8f}")

    # Sample predictions
    print("\nSample predictions vs analytical:")
    rates = [0.01, 0.03, 0.05, 0.07, 0.10]
    for r in rates:
        r_t = torch.tensor([r])
        t_t = torch.tensor([0.0])
        P_pinn_val = pinn.predict(r_t, t_t).item()
        P_analytical_val = cir_bond_price(r, 0.0, args.T, args.kappa, args.theta, args.sigma)
        error = abs(P_pinn_val - P_analytical_val)
        print(f"  r={r:.2f}: PINN={P_pinn_val:.6f}, Analytical={P_analytical_val:.6f}, "
              f"Error={error:.6f}")


if __name__ == "__main__":
    main()
