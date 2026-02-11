"""
Data Loader for Hull-White PINN

Loads interest rate data from two sources:
1. US Treasury yield curve data (traditional finance)
2. Bybit perpetual futures funding rates (crypto)
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import json
import time


# ═══════════════════════════════════════════════════════════════════════════════
# US TREASURY YIELD CURVE DATA
# ═══════════════════════════════════════════════════════════════════════════════

# Standard US Treasury yield curve tenors (in years)
TREASURY_TENORS = {
    "1M": 1 / 12,
    "2M": 2 / 12,
    "3M": 3 / 12,
    "6M": 6 / 12,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}


def load_sample_treasury_curve() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a sample US Treasury yield curve.
    Returns realistic data for demonstration/testing when API is unavailable.

    Returns:
        (maturities_years, zero_rates) tuple
    """
    # Representative US Treasury yields (as of a typical date)
    sample_yields = {
        1 / 12: 0.0525,    # 1M
        2 / 12: 0.0530,    # 2M
        3 / 12: 0.0535,    # 3M
        6 / 12: 0.0528,    # 6M
        1.0: 0.0505,       # 1Y
        2.0: 0.0465,       # 2Y
        3.0: 0.0440,       # 3Y
        5.0: 0.0420,       # 5Y
        7.0: 0.0418,       # 7Y
        10.0: 0.0415,      # 10Y
        20.0: 0.0445,      # 20Y
        30.0: 0.0435,      # 30Y
    }

    maturities = np.array(sorted(sample_yields.keys()))
    rates = np.array([sample_yields[t] for t in maturities])
    return maturities, rates


def fetch_treasury_yields_xml(date: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch US Treasury yield curve data from the US Treasury website.
    Falls back to sample data if the API is unavailable.

    Args:
        date: Date string in 'YYYY-MM-DD' format (default: recent date)

    Returns:
        (maturities_years, zero_rates) tuple
    """
    try:
        if date is None:
            # Use a recent business day
            today = datetime.now()
            if today.weekday() >= 5:
                today -= timedelta(days=today.weekday() - 4)
            date = today.strftime("%Y-%m-%d")

        year = date[:4]
        month = date[5:7]

        url = (
            f"https://home.treasury.gov/resource-center/data-chart-center/"
            f"interest-rates/daily-treasury-rates.csv/{year}/{month}"
            f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
        )

        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Treasury API returned {response.status_code}, using sample data")
            return load_sample_treasury_curve()

        lines = response.text.strip().split("\n")
        if len(lines) < 2:
            return load_sample_treasury_curve()

        # Parse CSV header and last row
        header = lines[0].split(",")
        last_row = lines[-1].split(",")

        tenor_map = {
            "1 Mo": 1 / 12, "2 Mo": 2 / 12, "3 Mo": 3 / 12,
            "4 Mo": 4 / 12, "6 Mo": 6 / 12,
            "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0, "5 Yr": 5.0,
            "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0,
        }

        maturities = []
        rates = []

        for i, col in enumerate(header):
            col_clean = col.strip().strip('"')
            if col_clean in tenor_map and i < len(last_row):
                try:
                    rate = float(last_row[i].strip().strip('"'))
                    maturities.append(tenor_map[col_clean])
                    rates.append(rate / 100.0)  # Convert from percent
                except (ValueError, IndexError):
                    continue

        if len(maturities) < 3:
            return load_sample_treasury_curve()

        return np.array(maturities), np.array(rates)

    except Exception as e:
        print(f"Error fetching Treasury data: {e}. Using sample data.")
        return load_sample_treasury_curve()


def generate_synthetic_yield_curve(
    short_rate: float = 0.03,
    long_rate: float = 0.045,
    hump_rate: float = 0.05,
    hump_maturity: float = 2.0,
    n_points: int = 50,
    T_max: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic yield curve with a specified shape.

    Args:
        short_rate: Short end of the curve
        long_rate: Long end of the curve
        hump_rate: Peak rate (for humped curves, set > long_rate)
        hump_maturity: Maturity at which the hump occurs
        n_points: Number of points
        T_max: Maximum maturity

    Returns:
        (maturities, rates) tuple
    """
    maturities = np.linspace(0.1, T_max, n_points)

    # Nelson-Siegel-Svensson style parametric curve
    beta0 = long_rate
    beta1 = short_rate - long_rate
    beta2 = (hump_rate - long_rate) * np.exp(1)
    tau = hump_maturity

    factor1 = (1 - np.exp(-maturities / tau)) / (maturities / tau)
    factor2 = factor1 - np.exp(-maturities / tau)

    rates = beta0 + beta1 * factor1 + beta2 * factor2

    return maturities, rates


# ═══════════════════════════════════════════════════════════════════════════════
# BYBIT FUNDING RATE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_bybit_funding_rates(
    symbol: str = "BTCUSDT",
    limit: int = 200,
) -> pd.DataFrame:
    """
    Fetch historical funding rates from Bybit API v5.

    Funding rates on perpetual futures are analogous to short-term interest rates:
    - Positive funding: longs pay shorts (bullish market)
    - Negative funding: shorts pay longs (bearish market)

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        limit: Number of records (max 200 per request)

    Returns:
        DataFrame with columns: timestamp, datetime, funding_rate
    """
    url = "https://api.bybit.com/v5/market/funding/history"

    all_records = []
    end_time = None

    # Fetch multiple pages to get more history
    n_pages = max(1, limit // 200)
    remaining = limit

    for _ in range(n_pages):
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": min(remaining, 200),
        }
        if end_time is not None:
            params["endTime"] = str(end_time)

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data.get("retCode") != 0:
                print(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                break

            items = data.get("result", {}).get("list", [])
            if not items:
                break

            for item in items:
                ts = int(item["fundingRateTimestamp"])
                all_records.append({
                    "timestamp": ts,
                    "datetime": datetime.fromtimestamp(ts / 1000),
                    "funding_rate": float(item["fundingRate"]),
                })

            remaining -= len(items)
            if remaining <= 0:
                break

            # Set end_time for next page (go further back)
            end_time = int(items[-1]["fundingRateTimestamp"]) - 1
            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"Error fetching Bybit data: {e}")
            break

    if not all_records:
        print("No Bybit data fetched. Using synthetic funding rates.")
        return generate_synthetic_funding_rates(symbol=symbol, n_periods=limit)

    df = pd.DataFrame(all_records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_synthetic_funding_rates(
    symbol: str = "BTCUSDT",
    n_periods: int = 500,
    mean_rate: float = 0.0001,
    mean_reversion: float = 0.3,
    volatility: float = 0.0003,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic funding rate data mimicking Bybit perpetual futures.

    Uses an Ornstein-Uhlenbeck process to simulate mean-reverting funding rates.

    Args:
        symbol: Symbol name for labeling
        n_periods: Number of 8-hour funding periods
        mean_rate: Long-run mean funding rate
        mean_reversion: Speed of mean reversion
        volatility: Volatility of funding rate
        seed: Random seed

    Returns:
        DataFrame with columns: timestamp, datetime, funding_rate
    """
    np.random.seed(seed)

    dt = 1.0  # 1 period (8 hours)
    rates = np.zeros(n_periods)
    rates[0] = mean_rate

    for i in range(1, n_periods):
        dr = mean_reversion * (mean_rate - rates[i - 1]) * dt + \
            volatility * np.sqrt(dt) * np.random.randn()
        rates[i] = rates[i - 1] + dr

    # Create timestamps (every 8 hours going back from now)
    now = datetime.now()
    timestamps = []
    for i in range(n_periods):
        t = now - timedelta(hours=8 * (n_periods - 1 - i))
        timestamps.append(t)

    df = pd.DataFrame({
        "timestamp": [int(t.timestamp() * 1000) for t in timestamps],
        "datetime": timestamps,
        "funding_rate": rates,
    })

    return df


def funding_rates_to_annualized(
    df: pd.DataFrame,
    periods_per_day: int = 3,
) -> pd.DataFrame:
    """
    Convert 8-hour funding rates to annualized rates.

    Bybit funding occurs every 8 hours (3 times per day).
    Annualized rate = funding_rate * periods_per_day * 365

    Args:
        df: DataFrame with 'funding_rate' column
        periods_per_day: Number of funding periods per day (default: 3)

    Returns:
        DataFrame with additional 'annualized_rate' column
    """
    df = df.copy()
    df["annualized_rate"] = df["funding_rate"] * periods_per_day * 365
    return df


def compute_funding_rate_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for funding rate data.

    Returns dict with mean, std, min, max, autocorrelation, etc.
    """
    rates = df["funding_rate"].values

    # Autocorrelation at lag 1
    if len(rates) > 1:
        autocorr = np.corrcoef(rates[:-1], rates[1:])[0, 1]
    else:
        autocorr = 0.0

    # Estimate OU parameters via MLE
    if len(rates) > 2:
        dr = np.diff(rates)
        r_prev = rates[:-1]
        # dr = (theta - a*r)*dt + sigma*dW => linear regression
        # dr = c0 + c1 * r_prev + noise
        X = np.column_stack([np.ones(len(r_prev)), r_prev])
        coeffs = np.linalg.lstsq(X, dr, rcond=None)[0]
        c0, c1 = coeffs
        a_est = -c1  # mean reversion speed
        theta_est = c0 / a_est if abs(a_est) > 1e-10 else rates.mean()
        residuals = dr - X @ coeffs
        sigma_est = np.std(residuals)
    else:
        a_est = 0.0
        theta_est = rates.mean()
        sigma_est = 0.0

    return {
        "n_observations": len(rates),
        "mean": float(np.mean(rates)),
        "std": float(np.std(rates)),
        "min": float(np.min(rates)),
        "max": float(np.max(rates)),
        "median": float(np.median(rates)),
        "skewness": float(pd.Series(rates).skew()),
        "kurtosis": float(pd.Series(rates).kurtosis()),
        "autocorrelation_lag1": float(autocorr),
        "estimated_mean_reversion": float(a_est),
        "estimated_long_run_mean": float(theta_est),
        "estimated_volatility": float(sigma_est),
        "annualized_mean": float(np.mean(rates) * 3 * 365),
        "annualized_std": float(np.std(rates) * np.sqrt(3 * 365)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data(
    treasury_date: Optional[str] = None,
    bybit_symbol: str = "BTCUSDT",
    bybit_limit: int = 500,
    use_synthetic_if_unavailable: bool = True,
) -> Dict:
    """
    Load both treasury and crypto funding rate data.

    Returns:
        Dict with keys:
            'treasury_maturities': np.ndarray
            'treasury_rates': np.ndarray
            'funding_rates': pd.DataFrame
            'funding_stats': dict
    """
    print("Loading Treasury yield curve data...")
    try:
        mat, rates = fetch_treasury_yields_xml(treasury_date)
        print(f"  Loaded {len(mat)} treasury tenors")
    except Exception as e:
        print(f"  Failed: {e}")
        if use_synthetic_if_unavailable:
            mat, rates = load_sample_treasury_curve()
            print(f"  Using sample data: {len(mat)} tenors")
        else:
            raise

    print(f"\nLoading Bybit {bybit_symbol} funding rates...")
    try:
        funding_df = fetch_bybit_funding_rates(bybit_symbol, bybit_limit)
        print(f"  Loaded {len(funding_df)} funding rate observations")
    except Exception as e:
        print(f"  Failed: {e}")
        if use_synthetic_if_unavailable:
            funding_df = generate_synthetic_funding_rates(bybit_symbol, bybit_limit)
            print(f"  Using synthetic data: {len(funding_df)} observations")
        else:
            raise

    funding_df = funding_rates_to_annualized(funding_df)
    stats = compute_funding_rate_statistics(funding_df)

    return {
        "treasury_maturities": mat,
        "treasury_rates": rates,
        "funding_rates": funding_df,
        "funding_stats": stats,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Hull-White PINN Data Loader")
    print("=" * 60)

    data = load_all_data(bybit_symbol="BTCUSDT", bybit_limit=200)

    print("\n--- Treasury Yield Curve ---")
    for t, r in zip(data["treasury_maturities"], data["treasury_rates"]):
        print(f"  {t:6.2f}Y: {r:.4%}")

    print("\n--- Bybit Funding Rate Statistics ---")
    for key, val in data["funding_stats"].items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")

    print("\n--- Recent Funding Rates ---")
    print(data["funding_rates"].tail(10).to_string(index=False))
