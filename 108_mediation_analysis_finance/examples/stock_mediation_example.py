#!/usr/bin/env python3
"""
Stock Market Mediation Analysis Example

This example demonstrates mediation analysis using stock market data:
- Treatment (X): SPY (S&P 500) returns
- Mediator (M): VIX (volatility index)
- Outcome (Y): Individual stock returns (e.g., AAPL)

The hypothesis: Market movements affect individual stocks both directly
and indirectly through volatility changes.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import yfinance, fall back to synthetic data if not available
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("yfinance not installed. Using synthetic data instead.")

from model import baron_kenny_mediation, bootstrap_mediation, CausalMediationAnalysis
from strategy import MediationTradingStrategy, MediationBacktester, BacktestConfig


def generate_synthetic_stock_data(n: int = 500) -> pd.DataFrame:
    """Generate synthetic stock market data for demonstration"""
    np.random.seed(42)

    dates = pd.date_range('2023-01-01', periods=n, freq='B')

    # SPY returns (treatment)
    spy_returns = np.random.randn(n) * 0.01 + 0.0003  # Slight positive drift

    # VIX level (mediator) - increases with market volatility
    # VIX tends to spike when market goes down
    vix_base = 20 + np.cumsum(np.random.randn(n) * 0.5)
    vix_spike = -spy_returns * 500  # VIX spikes on negative returns
    vix = vix_base + vix_spike
    vix = np.maximum(vix, 10)  # Floor at 10

    # AAPL returns (outcome)
    # Affected by both market directly and through volatility
    direct_effect = 0.9 * spy_returns  # High beta to SPY
    vol_effect = -0.002 * (vix - vix.mean())  # Higher VIX hurts returns
    noise = np.random.randn(n) * 0.008
    aapl_returns = direct_effect + vol_effect + noise

    return pd.DataFrame({
        'X': spy_returns,
        'M': vix,
        'Y': aapl_returns,
        'SPY_return': spy_returns,
        'VIX': vix,
        'AAPL_return': aapl_returns
    }, index=dates)


def fetch_real_stock_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch real stock data from Yahoo Finance"""
    print("Fetching data from Yahoo Finance...")

    # Download data
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    aapl = yf.download('AAPL', start=start_date, end=end_date, progress=False)

    # Calculate returns
    spy_returns = spy['Adj Close'].pct_change()
    aapl_returns = aapl['Adj Close'].pct_change()
    vix_level = vix['Close']

    # Align and combine
    df = pd.DataFrame({
        'SPY_return': spy_returns,
        'VIX': vix_level,
        'AAPL_return': aapl_returns
    }).dropna()

    # Lag treatment and mediator
    df['X'] = df['SPY_return'].shift(1)
    df['M'] = df['VIX'].shift(1)
    df['Y'] = df['AAPL_return']

    return df.dropna()


def run_analysis():
    print("=" * 70)
    print("STOCK MARKET MEDIATION ANALYSIS")
    print("Hypothesis: SPY returns → VIX changes → AAPL returns")
    print("=" * 70)
    print()

    # Get data
    if HAS_YFINANCE:
        try:
            data = fetch_real_stock_data('2022-01-01', '2024-01-01')
            print(f"Using real market data: {len(data)} observations")
        except Exception as e:
            print(f"Failed to fetch real data: {e}")
            print("Falling back to synthetic data")
            data = generate_synthetic_stock_data()
    else:
        data = generate_synthetic_stock_data()
        print(f"Using synthetic data: {len(data)} observations")

    print()

    # Run Baron-Kenny mediation analysis
    print("=" * 70)
    print("BARON-KENNY MEDIATION ANALYSIS")
    print("=" * 70)

    results = baron_kenny_mediation(
        data['X'].values,
        data['M'].values,
        data['Y'].values
    )
    print(results)

    # Bootstrap confidence intervals
    print("=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS (n=1000)")
    print("=" * 70)

    boot_results = bootstrap_mediation(
        data['X'].values,
        data['M'].values,
        data['Y'].values,
        n_bootstrap=1000,
        random_state=42
    )

    print(f"\nIndirect Effect:")
    print(f"  Mean: {boot_results['indirect_mean']:.6f}")
    print(f"  95% CI: [{boot_results['indirect_ci_lower']:.6f}, {boot_results['indirect_ci_upper']:.6f}]")
    print(f"  Significant: {boot_results['indirect_significant']}")

    print(f"\nDirect Effect:")
    print(f"  Mean: {boot_results['direct_mean']:.6f}")
    print(f"  95% CI: [{boot_results['direct_ci_lower']:.6f}, {boot_results['direct_ci_upper']:.6f}]")

    print(f"\nProportion Mediated:")
    print(f"  Mean: {boot_results['proportion_mediated_mean']:.2%}")
    print(f"  95% CI: [{boot_results['proportion_mediated_ci_lower']:.2%}, {boot_results['proportion_mediated_ci_upper']:.2%}]")

    # Causal mediation with sensitivity analysis
    print()
    print("=" * 70)
    print("CAUSAL MEDIATION ANALYSIS WITH SENSITIVITY")
    print("=" * 70)

    cma = CausalMediationAnalysis(
        data['X'].values,
        data['M'].values,
        data['Y'].values
    )

    effects = cma.estimate_effects(n_simulations=500, random_state=42)

    print("\nCausal Effects:")
    for effect_name, values in effects.items():
        print(f"\n{effect_name}:")
        print(f"  Estimate: {values['estimate']:.6f}")
        print(f"  95% CI: [{values['ci_lower']:.6f}, {values['ci_upper']:.6f}]")
        if 'pvalue' in values:
            print(f"  P-value: {values['pvalue']:.4f}")

    # Sensitivity analysis
    sensitivity = cma.sensitivity_analysis(n_simulations=200, random_state=42)

    print("\nSensitivity Analysis:")
    if sensitivity['breakdown_rho'] is not None:
        print(f"  Breakdown rho: {sensitivity['breakdown_rho']:.2f}")
        print("  (Correlation at which indirect effect crosses zero)")
    else:
        print("  No breakdown point found - results robust to confounding")

    # Run trading strategy backtest
    print()
    print("=" * 70)
    print("TRADING STRATEGY BACKTEST")
    print("=" * 70)

    strategy = MediationTradingStrategy(
        lookback_window=60,
        mediation_threshold=0.25,
        signal_threshold=0.3
    )

    config = BacktestConfig(
        initial_capital=100000,
        transaction_cost_bps=10,
        max_position_size=0.2
    )

    backtester = MediationBacktester(strategy, config)
    bt_results = backtester.run(data)
    metrics = backtester.analyze_results(bt_results)

    print("\nBacktest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'return' in key.lower() or 'ratio' in key.lower() or 'drawdown' in key.lower() or 'rate' in key.lower():
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Plot results
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Portfolio value
        axes[0].plot(bt_results.index, bt_results['portfolio_value'])
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].set_title('Mediation Strategy Equity Curve')
        axes[0].grid(True, alpha=0.3)

        # Mediation dynamics
        axes[1].plot(bt_results.index, bt_results['proportion_mediated'].rolling(20).mean(), label='Prop. Mediated (20d MA)')
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Proportion Mediated')
        axes[1].set_title('Mediation Strength Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Drawdown
        axes[2].fill_between(bt_results.index, 0, -bt_results['drawdown'], color='red', alpha=0.5)
        axes[2].set_ylabel('Drawdown')
        axes[2].set_title('Underwater Plot')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('stock_mediation_results.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved to stock_mediation_results.png")
        plt.show()
    except Exception as e:
        print(f"\nCould not create plot: {e}")


if __name__ == "__main__":
    run_analysis()
