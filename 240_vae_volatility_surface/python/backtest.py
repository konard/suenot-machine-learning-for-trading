"""
Backtesting framework for VAE-based volatility trading strategies.

Implements a relative-value strategy that:
1. Encodes observed volatility surfaces with the trained VAE
2. Decodes to get the VAE's "fair" surface
3. Trades mispricings: sell overpriced vol, buy underpriced vol
4. Tracks PnL and computes Sharpe/Sortino/Max Drawdown

Usage:
    python -m python.backtest
"""

import argparse
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .model import VAEConfig, VAEModel
from .surface_generator import SurfaceGenerator, default_maturity_days, default_moneyness_grid
from .vol_surface import ArbitrageChecker, VolSurface


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    pnl_series: List[float] = field(default_factory=list)
    cumulative_pnl: List[float] = field(default_factory=list)
    n_trades: int = 0
    n_surfaces: int = 0

    @property
    def total_pnl(self) -> float:
        return sum(self.pnl_series)

    @property
    def sharpe_ratio(self) -> float:
        if len(self.pnl_series) < 2:
            return 0.0
        arr = np.array(self.pnl_series)
        std = arr.std()
        if std < 1e-12:
            return 0.0
        return float(arr.mean() / std * np.sqrt(252))

    @property
    def sortino_ratio(self) -> float:
        if len(self.pnl_series) < 2:
            return 0.0
        arr = np.array(self.pnl_series)
        downside = arr[arr < 0]
        if len(downside) == 0 or downside.std() < 1e-12:
            return 0.0
        return float(arr.mean() / downside.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        if not self.cumulative_pnl:
            return 0.0
        cum = np.array(self.cumulative_pnl)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        return float(dd.max())


def run_backtest(
    vae: VAEModel,
    surfaces: List[VolSurface],
    threshold: float = 0.02,
    position_size: float = 1.0,
) -> BacktestResult:
    """Run a relative-value backtest using VAE fair-value estimation.

    Strategy:
    - For each surface, encode then decode to get VAE's "fair" surface
    - Compute mispricing = observed - fair for each grid point
    - If |mispricing| > threshold: trade (sell overpriced, buy underpriced)
    - PnL is proportional to the mispricing that mean-reverts

    Args:
        vae: Trained VAE model.
        surfaces: Time-ordered list of volatility surfaces.
        threshold: Minimum mispricing to trigger a trade (in vol points).
        position_size: Notional per trade.

    Returns:
        BacktestResult with PnL series and metrics.
    """
    result = BacktestResult(n_surfaces=len(surfaces))
    cum_pnl = 0.0

    for i in range(len(surfaces) - 1):
        observed = surfaces[i].flatten()
        recon, _, _ = vae.forward(observed)
        mispricing = observed - recon

        # Trade signals: sell where observed > fair, buy where observed < fair
        period_pnl = 0.0
        next_observed = surfaces[i + 1].flatten()
        next_recon, _, _ = vae.forward(next_observed)
        next_mispricing = next_observed - next_recon

        for j in range(len(mispricing)):
            if abs(mispricing[j]) > threshold:
                # Direction: sell if overpriced (mispricing > 0), buy if underpriced
                direction = -np.sign(mispricing[j])
                # PnL: mean-reversion profit
                pnl = direction * (next_mispricing[j] - mispricing[j]) * position_size
                period_pnl += pnl
                result.n_trades += 1

        result.pnl_series.append(period_pnl)
        cum_pnl += period_pnl
        result.cumulative_pnl.append(cum_pnl)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest VAE volatility strategy")
    parser.add_argument("--n-train", type=int, default=500, help="Training surfaces")
    parser.add_argument("--n-test", type=int, default=100, help="Test surfaces")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--threshold", type=float, default=0.02, help="Trade threshold (vol pts)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    moneyness = default_moneyness_grid()
    maturities = default_maturity_days() / 365.0

    print("=" * 60)
    print("VAE Volatility Surface — Backtest")
    print("=" * 60)

    # Generate data
    gen = SurfaceGenerator(seed=args.seed)
    train_surfaces = gen.generate_batch(args.n_train, moneyness, maturities)
    test_surfaces = gen.generate_batch(args.n_test, moneyness, maturities)

    train_data = [s.flatten() for s in train_surfaces]
    input_dim = len(train_data[0])

    # Train VAE
    config = VAEConfig(input_dim=input_dim, epochs=args.epochs)
    vae = VAEModel(config)
    print(f"\nTraining VAE on {args.n_train} surfaces for {args.epochs} epochs...")
    total, recon_l, kl_l = vae.train(train_data, epochs=args.epochs)
    print(f"Final loss: total={total:.4f} (recon={recon_l:.4f}, kl={kl_l:.4f})")

    # Run backtest
    print(f"\nRunning backtest on {args.n_test} test surfaces (threshold={args.threshold})...")
    result = run_backtest(vae, test_surfaces, threshold=args.threshold)

    # Report
    print("\n" + "-" * 60)
    print("Backtest Results:")
    print(f"  Surfaces processed: {result.n_surfaces}")
    print(f"  Total trades: {result.n_trades}")
    print(f"  Total PnL: {result.total_pnl:.4f}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.4f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.4f}")
    print(f"  Max Drawdown: {result.max_drawdown:.4f}")

    # Check arbitrage on reconstructed surfaces
    print("\n" + "-" * 60)
    print("Arbitrage check on VAE-reconstructed surfaces:")
    bf_total = cal_total = 0
    for i, surf in enumerate(test_surfaces[:10]):
        recon, _, _ = vae.forward(surf.flatten())
        recon_surf = VolSurface(
            moneyness=moneyness,
            maturities=maturities,
            ivols=recon.reshape(len(maturities), len(moneyness)),
        )
        bf, _ = ArbitrageChecker.check_butterfly(recon_surf)
        cal, _ = ArbitrageChecker.check_calendar(recon_surf)
        bf_total += bf
        cal_total += cal
    print(f"  Butterfly violations (10 surfaces): {bf_total}")
    print(f"  Calendar violations (10 surfaces):  {cal_total}")

    print("\nDONE")


if __name__ == "__main__":
    main()
