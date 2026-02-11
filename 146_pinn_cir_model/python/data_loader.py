"""
Data Loading for CIR Model
============================

Fetches interest rate data from multiple sources:
- US Treasury yields (synthetic / public data)
- Bybit perpetual funding rates
- Synthetic CIR data for testing

Bybit API endpoints for funding rates:
- GET /v5/market/funding/history
"""

import numpy as np
import pandas as pd
import requests
import time
import argparse
from typing import Optional, List, Tuple
from datetime import datetime, timedelta


# =============================================================================
# Bybit Funding Rate Data
# =============================================================================

def fetch_bybit_funding_rates(
    symbol: str = "BTCUSDT",
    days: int = 180,
    category: str = "linear",
) -> pd.DataFrame:
    """
    Fetch funding rate history from Bybit.

    Bybit perpetual futures have funding every 8 hours (3 times per day).

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT').
    days : int
        Number of days of history to fetch.
    category : str
        Market category ('linear' for USDT perpetual).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, symbol, rate, rate_annualized.
    """
    base_url = "https://api.bybit.com/v5/market/funding/history"
    all_records = []
    limit = 200  # max records per request

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    cursor_time = end_time

    print(f"Fetching {symbol} funding rates from Bybit ({days} days)...")

    while cursor_time > start_time:
        params = {
            "category": category,
            "symbol": symbol,
            "limit": limit,
            "endTime": cursor_time,
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()

            if data.get("retCode") != 0:
                print(f"API error: {data.get('retMsg', 'Unknown error')}")
                break

            records = data.get("result", {}).get("list", [])
            if not records:
                break

            for record in records:
                all_records.append({
                    "timestamp": pd.Timestamp(
                        int(record["fundingRateTimestamp"]), unit="ms"
                    ),
                    "symbol": record["symbol"],
                    "rate": float(record["fundingRate"]),
                })

            # Move cursor to before the oldest record
            oldest_ts = min(int(r["fundingRateTimestamp"]) for r in records)
            cursor_time = oldest_ts - 1

            time.sleep(0.1)  # rate limiting

        except requests.RequestException as e:
            print(f"Request error: {e}")
            break

    if not all_records:
        print("No data fetched. Generating synthetic funding rates.")
        return generate_synthetic_funding_rates(symbol, days)

    df = pd.DataFrame(all_records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset=["timestamp"])

    # Annualize: 3 funding periods per day, 365 days per year
    df["rate_annualized"] = df["rate"] * 3 * 365

    print(f"Fetched {len(df)} funding rate records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Mean rate: {df['rate'].mean():.6f} ({df['rate_annualized'].mean():.4f} annualized)")

    return df


def generate_synthetic_funding_rates(
    symbol: str = "BTCUSDT",
    days: int = 180,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic funding rates that mimic Bybit-like dynamics.

    Uses a discretized CIR-like process with occasional regime changes.

    Parameters
    ----------
    symbol : str
        Symbol name for the DataFrame.
    days : int
        Number of days to generate.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Synthetic funding rate DataFrame.
    """
    np.random.seed(seed)

    n_periods = days * 3  # 3 funding periods per day (every 8 hours)
    dt = 8.0 / (24.0 * 365.0)  # 8 hours in years

    # CIR-like parameters for funding rates
    kappa = 50.0  # fast mean-reversion
    theta = 0.0001  # baseline funding rate ~0.01%
    sigma = 0.005  # moderate volatility

    rates = np.zeros(n_periods)
    rates[0] = theta

    for i in range(1, n_periods):
        r = max(rates[i - 1], 1e-8)
        drift = kappa * (theta - r) * dt
        diffusion = sigma * np.sqrt(r) * np.random.normal(0, np.sqrt(dt))
        rates[i] = max(r + drift + diffusion, -0.001)  # allow slightly negative

    # Add regime changes (bull/bear markets)
    regime_shifts = np.random.choice(n_periods, size=5, replace=False)
    for shift in regime_shifts:
        shift_size = np.random.uniform(-0.001, 0.003)
        decay = np.exp(-0.01 * np.arange(n_periods - shift))
        rates[shift:] += shift_size * decay[:n_periods - shift]

    timestamps = pd.date_range(
        end=datetime.now(),
        periods=n_periods,
        freq="8h",
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": symbol,
        "rate": rates,
        "rate_annualized": rates * 3 * 365,
    })

    print(f"Generated {len(df)} synthetic funding rate records")
    return df


# =============================================================================
# Treasury Rate Data
# =============================================================================

def fetch_treasury_rates(
    maturity: str = "3m",
    days: int = 365,
) -> pd.DataFrame:
    """
    Fetch or generate US Treasury rate data.

    Note: For a production system, use FRED API or similar.
    This implementation generates realistic synthetic data.

    Parameters
    ----------
    maturity : str
        Treasury maturity: '3m', '6m', '1y', '2y', '5y', '10y'.
    days : int
        Number of days of history.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, rate.
    """
    # Typical yield levels and volatilities by maturity
    maturity_params = {
        "3m": {"mean": 0.045, "vol": 0.005, "kappa": 0.5},
        "6m": {"mean": 0.047, "vol": 0.005, "kappa": 0.4},
        "1y": {"mean": 0.048, "vol": 0.006, "kappa": 0.35},
        "2y": {"mean": 0.042, "vol": 0.007, "kappa": 0.3},
        "5y": {"mean": 0.040, "vol": 0.008, "kappa": 0.25},
        "10y": {"mean": 0.042, "vol": 0.008, "kappa": 0.2},
    }

    params = maturity_params.get(maturity, maturity_params["3m"])
    theta = params["mean"]
    vol = params["vol"]
    kappa = params["kappa"]

    np.random.seed(hash(maturity) % 2**31)

    rates = np.zeros(days)
    rates[0] = theta

    dt = 1.0 / 252.0  # daily

    for i in range(1, days):
        r = max(rates[i - 1], 0.001)
        drift = kappa * (theta - r) * dt
        diffusion = vol * np.sqrt(r) * np.random.normal(0, np.sqrt(dt))
        rates[i] = max(r + drift + diffusion, 0.001)

    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")

    df = pd.DataFrame({
        "date": dates[:len(rates)],
        "rate": rates,
        "maturity": maturity,
    })

    print(f"Generated {len(df)} {maturity} Treasury rate records")
    print(f"Mean rate: {df['rate'].mean():.4f}, Std: {df['rate'].std():.4f}")

    return df


# =============================================================================
# Synthetic CIR Data
# =============================================================================

def generate_synthetic_cir_data(
    kappa: float = 0.5,
    theta: float = 0.05,
    sigma: float = 0.1,
    r0: float = 0.03,
    T: float = 10.0,
    n_steps: int = 2520,
    n_paths: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic CIR rate paths for model testing.

    Uses the exact non-central chi-squared sampling method.

    Parameters
    ----------
    kappa, theta, sigma : float
        CIR parameters.
    r0 : float
        Initial rate.
    T : float
        Time horizon in years.
    n_steps : int
        Number of time steps (e.g., 252 per year for daily).
    n_paths : int
        Number of simulated paths.
    seed : int
        Random seed.

    Returns
    -------
    Tuple[ndarray, ndarray]
        (time_grid, paths) where paths has shape (n_paths, n_steps + 1).
    """
    np.random.seed(seed)

    dt = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0

    c = sigma ** 2 * (1.0 - np.exp(-kappa * dt)) / (4.0 * kappa)
    df = 4.0 * kappa * theta / (sigma ** 2)

    for i in range(n_steps):
        nc = paths[:, i] * np.exp(-kappa * dt) / c
        nc = np.maximum(nc, 1e-10)
        paths[:, i + 1] = c * np.random.noncentral_chisquare(df, nc)

    print(f"Generated {n_paths} CIR paths with {n_steps} steps")
    print(f"Parameters: kappa={kappa}, theta={theta}, sigma={sigma}")
    print(f"Min rate: {paths.min():.6f}, Max rate: {paths.max():.6f}")

    return time_grid, paths


# =============================================================================
# Data Preprocessing
# =============================================================================

def prepare_rate_data_for_calibration(
    df: pd.DataFrame,
    rate_column: str = "rate",
    ensure_positive: bool = True,
    min_rate: float = 1e-8,
) -> np.ndarray:
    """
    Prepare rate data for CIR calibration.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with rate column.
    rate_column : str
        Name of the rate column.
    ensure_positive : bool
        If True, clip rates to be positive (required for CIR).
    min_rate : float
        Minimum rate value when ensuring positivity.

    Returns
    -------
    ndarray
        Cleaned rate array.
    """
    rates = df[rate_column].values.copy()

    # Remove NaN values
    rates = rates[~np.isnan(rates)]

    if ensure_positive:
        n_negative = np.sum(rates <= 0)
        if n_negative > 0:
            print(f"Warning: {n_negative} non-positive rates clipped to {min_rate}")
            rates = np.maximum(rates, min_rate)

    return rates


def main():
    """Command-line data fetching."""
    parser = argparse.ArgumentParser(description="Fetch rate data")
    parser.add_argument("--source", type=str, default="bybit",
                        choices=["bybit", "treasury", "synthetic"],
                        help="Data source")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Bybit symbol")
    parser.add_argument("--days", type=int, default=180,
                        help="Number of days")
    parser.add_argument("--maturity", type=str, default="3m",
                        help="Treasury maturity")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file")
    args = parser.parse_args()

    if args.source == "bybit":
        df = fetch_bybit_funding_rates(args.symbol, args.days)
    elif args.source == "treasury":
        df = fetch_treasury_rates(args.maturity, args.days)
    else:
        _, paths = generate_synthetic_cir_data(n_paths=1, T=args.days / 252.0)
        df = pd.DataFrame({
            "date": pd.date_range(end=datetime.now(), periods=paths.shape[1], freq="B"),
            "rate": paths[0],
        })

    print(f"\nData shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nStatistics:")
    print(df.describe())

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
