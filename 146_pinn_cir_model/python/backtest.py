"""
Backtesting CIR-Based Trading Strategies
==========================================

Implements trading strategies based on the CIR model:
1. Funding Rate Mean-Reversion: Trade when funding rates deviate from CIR equilibrium
2. Bond Duration Trading: Trade bonds based on CIR yield curve predictions
3. Credit Spread Trading: Trade based on CIR-implied default probabilities

Data sources: Bybit funding rates, synthetic rate data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import argparse

from .analytical import cir_conditional_mean, cir_conditional_variance, cir_bond_price
from .calibration import calibrate_cir_mle, calibrate_cir_ols
from .data_loader import (
    fetch_bybit_funding_rates,
    generate_synthetic_funding_rates,
    prepare_rate_data_for_calibration,
)


# =============================================================================
# Strategy 1: Funding Rate Mean-Reversion
# =============================================================================

def funding_rate_mean_reversion(
    df: pd.DataFrame,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float = 8.0 / (24.0 * 365.0),
    z_threshold: float = 2.0,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """
    Mean-reversion strategy on funding rates using CIR model.

    When the funding rate is significantly above theta, expect it to revert down.
    When below, expect it to revert up.

    Signal:
        z_score = (rate - theta) / conditional_std
        If z > threshold: short signal (rate will drop)
        If z < -threshold: long signal (rate will rise)

    Parameters
    ----------
    df : pd.DataFrame
        Funding rate data with 'rate' column.
    kappa, theta, sigma : float
        Calibrated CIR parameters.
    dt : float
        Time between observations (8h in years for Bybit).
    z_threshold : float
        Z-score threshold for signal generation.
    position_size : float
        Base position size.

    Returns
    -------
    pd.DataFrame
        DataFrame with signals, positions, and PnL.
    """
    rates = df["rate"].values
    n = len(rates)

    signals = np.zeros(n)
    positions = np.zeros(n)
    pnl = np.zeros(n)

    for i in range(1, n):
        r = max(abs(rates[i]), 1e-8)

        # CIR expected rate and conditional std
        expected = cir_conditional_mean(r, dt, kappa, theta)
        variance = cir_conditional_variance(r, dt, kappa, theta, sigma)
        std = np.sqrt(max(variance, 1e-16))

        # Z-score: how far is current rate from long-run mean?
        z_score = (r - theta) / max(std, 1e-10)

        # Generate signal
        if z_score > z_threshold:
            signals[i] = -1.0  # Short: rate will revert down
        elif z_score < -z_threshold:
            signals[i] = 1.0   # Long: rate will revert up
        else:
            signals[i] = 0.0

        # Position (simple: signal * size)
        positions[i] = signals[i] * position_size

        # PnL: profit from rate change
        # If short and rate drops -> profit; if long and rate rises -> profit
        rate_change = rates[i] - rates[i - 1]
        pnl[i] = positions[i - 1] * (-rate_change) * 10000  # scale to bps

    result_df = df.copy()
    result_df["signal"] = signals
    result_df["position"] = positions
    result_df["pnl"] = pnl
    result_df["cumulative_pnl"] = np.cumsum(pnl)

    return result_df


# =============================================================================
# Strategy 2: Bond Duration Trading
# =============================================================================

def bond_duration_strategy(
    rates: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    T: float = 5.0,
    dt: float = 1.0 / 252.0,
    threshold: float = 0.005,
) -> pd.DataFrame:
    """
    Trade bonds based on CIR yield predictions.

    When the model predicts rates will fall, go long bonds (rising prices).
    When rates will rise, short bonds (falling prices).

    Parameters
    ----------
    rates : ndarray
        Observed short rate series.
    kappa, theta, sigma : float
        CIR parameters.
    T : float
        Bond maturity for trading.
    dt : float
        Time step.
    threshold : float
        Minimum expected change to trigger trade.

    Returns
    -------
    pd.DataFrame
        Trading results.
    """
    n = len(rates)

    positions = np.zeros(n)
    bond_prices = np.zeros(n)
    pnl = np.zeros(n)

    for i in range(n):
        # Current bond price
        bond_prices[i] = cir_bond_price(
            max(rates[i], 1e-8), 0.0, T, kappa, theta, sigma
        )

        if i == 0:
            continue

        # CIR-predicted rate change
        expected_rate = cir_conditional_mean(rates[i], dt, kappa, theta)
        expected_change = expected_rate - rates[i]

        # If rates expected to drop -> bond prices will rise -> go long
        if expected_change < -threshold:
            positions[i] = 1.0
        elif expected_change > threshold:
            positions[i] = -1.0
        else:
            positions[i] = 0.0

        # PnL from bond price change
        bond_return = (bond_prices[i] - bond_prices[i - 1]) / bond_prices[i - 1]
        pnl[i] = positions[i - 1] * bond_return * 10000  # bps

    df = pd.DataFrame({
        "rate": rates,
        "bond_price": bond_prices,
        "position": positions,
        "pnl": pnl,
        "cumulative_pnl": np.cumsum(pnl),
    })

    return df


# =============================================================================
# Performance Metrics
# =============================================================================

def compute_metrics(pnl: np.ndarray, periods_per_year: float = 252.0) -> Dict:
    """
    Compute strategy performance metrics.

    Parameters
    ----------
    pnl : ndarray
        PnL series (per-period returns).
    periods_per_year : float
        Number of trading periods per year.

    Returns
    -------
    Dict
        Performance metrics.
    """
    total_pnl = np.sum(pnl)
    n_periods = len(pnl)
    n_years = n_periods / periods_per_year

    # Returns
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)

    # Sharpe ratio (annualized)
    sharpe = (mean_pnl / std_pnl * np.sqrt(periods_per_year)) if std_pnl > 0 else 0

    # Sortino ratio
    downside = pnl[pnl < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-10
    sortino = (mean_pnl / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0

    # Maximum drawdown
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns)

    # Win rate
    trades = pnl[pnl != 0]
    win_rate = np.mean(trades > 0) if len(trades) > 0 else 0

    # Profit factor
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = abs(np.sum(pnl[pnl < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "total_pnl_bps": total_pnl,
        "annualized_return_bps": total_pnl / max(n_years, 0.01),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_bps": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": len(trades),
        "n_periods": n_periods,
        "mean_pnl_bps": mean_pnl,
        "std_pnl_bps": std_pnl,
    }


def run_backtest(
    symbol: str = "BTCUSDT",
    days: int = 90,
    strategy: str = "mean_reversion",
    z_threshold: float = 2.0,
    calibration_window: int = 60,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Run a full backtest pipeline.

    1. Fetch data
    2. Calibrate CIR on training window
    3. Generate signals on test window
    4. Compute performance metrics

    Parameters
    ----------
    symbol : str
        Bybit symbol.
    days : int
        Total days of data.
    strategy : str
        'mean_reversion' or 'bond_duration'.
    z_threshold : float
        Signal threshold for mean-reversion strategy.
    calibration_window : int
        Days for calibration (rest is test).

    Returns
    -------
    Tuple[DataFrame, Dict, Dict]
        (results_df, metrics, cir_params)
    """
    # Fetch data
    print(f"\n{'='*60}")
    print(f"Backtest: {strategy} on {symbol}")
    print(f"{'='*60}")

    try:
        df = fetch_bybit_funding_rates(symbol, days=days)
    except Exception:
        print("Bybit fetch failed, using synthetic data.")
        df = generate_synthetic_funding_rates(symbol, days=days)

    # Prepare rates
    rates = prepare_rate_data_for_calibration(df, "rate", ensure_positive=True)

    # Split into calibration and test
    cal_periods = calibration_window * 3  # 3 funding periods per day
    cal_rates = rates[:cal_periods]
    test_df = df.iloc[cal_periods:].copy().reset_index(drop=True)

    # Calibrate CIR
    print(f"\nCalibrating on {len(cal_rates)} observations ({calibration_window} days)...")
    dt = 8.0 / (24.0 * 365.0)  # 8 hours in years

    try:
        cal_results = calibrate_cir_mle(cal_rates, dt, method="moment")
        kappa = cal_results["kappa"]
        theta = cal_results["theta"]
        sigma = cal_results["sigma"]
    except Exception as e:
        print(f"MLE failed ({e}), using OLS.")
        kappa, theta, sigma = calibrate_cir_ols(cal_rates, dt)

    cir_params = {"kappa": kappa, "theta": theta, "sigma": sigma}
    print(f"\nCalibrated: kappa={kappa:.4f}, theta={theta:.8f}, sigma={sigma:.4f}")

    # Run strategy
    if strategy == "mean_reversion":
        results_df = funding_rate_mean_reversion(
            test_df, kappa, theta, sigma, dt=dt, z_threshold=z_threshold,
        )
        periods_per_year = 3 * 365  # 3 funding periods per day
    else:
        test_rates = prepare_rate_data_for_calibration(test_df, "rate", ensure_positive=True)
        results_df = bond_duration_strategy(
            test_rates, kappa, theta, sigma, T=5.0, dt=dt,
        )
        periods_per_year = 3 * 365

    # Compute metrics
    metrics = compute_metrics(results_df["pnl"].values, periods_per_year=periods_per_year)

    # Print results
    print(f"\n{'='*60}")
    print("Backtest Results")
    print(f"{'='*60}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    return results_df, metrics, cir_params


def main():
    """Command-line backtest interface."""
    parser = argparse.ArgumentParser(description="CIR Backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol")
    parser.add_argument("--days", type=int, default=90, help="Days of data")
    parser.add_argument("--strategy", type=str, default="mean_reversion",
                        choices=["mean_reversion", "bond_duration"],
                        help="Trading strategy")
    parser.add_argument("--z-threshold", type=float, default=2.0,
                        help="Z-score threshold for signals")
    parser.add_argument("--cal-window", type=int, default=60,
                        help="Calibration window (days)")
    args = parser.parse_args()

    results_df, metrics, cir_params = run_backtest(
        symbol=args.symbol,
        days=args.days,
        strategy=args.strategy,
        z_threshold=args.z_threshold,
        calibration_window=args.cal_window,
    )


if __name__ == "__main__":
    main()
