#!/usr/bin/env python3
"""
Cryptocurrency Mediation Analysis Example (Bybit Data)

This example demonstrates mediation analysis using crypto market data:
- Treatment (X): BTC returns
- Mediator (M): Funding rate / Open interest
- Outcome (Y): ETH returns

The hypothesis: BTC movements affect ETH both directly
and indirectly through derivatives market sentiment indicators.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import pandas as pd

from model import baron_kenny_mediation, bootstrap_mediation
from strategy import MediationTradingStrategy, MediationBacktester, BacktestConfig


def generate_synthetic_crypto_data(n: int = 500) -> pd.DataFrame:
    """
    Generate synthetic crypto data that mimics real market dynamics.

    Simulates:
    - BTC returns with high volatility and momentum
    - Funding rates that respond to BTC trends
    - ETH returns affected by both BTC and funding rates
    """
    np.random.seed(42)

    dates = pd.date_range('2024-01-01', periods=n, freq='H')

    # BTC returns (treatment) - higher volatility than stocks
    # Include momentum and mean reversion
    btc_innovations = np.random.randn(n) * 0.02
    btc_returns = np.zeros(n)

    for i in range(1, n):
        # Mean reversion + momentum + innovation
        btc_returns[i] = -0.05 * btc_returns[i-1] + btc_innovations[i]

    # Funding rate (mediator)
    # Funding tends to follow recent price action
    # When prices go up, longs pay shorts (positive funding)
    funding_base = 0.0001  # 0.01% base rate
    funding_rates = np.zeros(n)

    for i in range(8, n):
        # Funding responds to recent 8h returns
        recent_return = btc_returns[i-8:i].sum()
        funding_rates[i] = funding_base + recent_return * 0.5 + np.random.randn() * 0.0001

    # ETH returns (outcome)
    # High correlation with BTC, but also affected by funding rates
    # High positive funding → overheated longs → potential correction

    a_effect = 0.4   # BTC → Funding
    b_effect = -50   # Funding → ETH (negative: high funding = bearish)
    direct_effect = 0.85  # Direct BTC → ETH correlation

    eth_returns = np.zeros(n)
    for i in range(1, n):
        eth_returns[i] = (
            direct_effect * btc_returns[i] +
            b_effect * funding_rates[i] +
            np.random.randn() * 0.015
        )

    return pd.DataFrame({
        'timestamp': dates,
        'X': btc_returns,
        'M': funding_rates,
        'Y': eth_returns,
        'BTC_return': btc_returns,
        'Funding_rate': funding_rates,
        'ETH_return': eth_returns
    }).set_index('timestamp')


def run_analysis():
    print("=" * 70)
    print("CRYPTOCURRENCY MEDIATION ANALYSIS (BYBIT-STYLE DATA)")
    print("Hypothesis: BTC returns → Funding Rate → ETH returns")
    print("=" * 70)
    print()

    # Generate synthetic data (in production, use prepare_crypto_data_bybit)
    print("Generating synthetic crypto data...")
    data = generate_synthetic_crypto_data(n=1000)
    print(f"Data shape: {data.shape}")
    print()

    # Display data summary
    print("Data Summary:")
    print(data[['BTC_return', 'Funding_rate', 'ETH_return']].describe())
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
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    boot_results = bootstrap_mediation(
        data['X'].values,
        data['M'].values,
        data['Y'].values,
        n_bootstrap=1000,
        random_state=42
    )

    print(f"\nIndirect Effect (BTC → Funding → ETH):")
    print(f"  Mean: {boot_results['indirect_mean']:.6f}")
    print(f"  95% CI: [{boot_results['indirect_ci_lower']:.6f}, {boot_results['indirect_ci_upper']:.6f}]")
    print(f"  Significant: {boot_results['indirect_significant']}")

    print(f"\nDirect Effect (BTC → ETH):")
    print(f"  Mean: {boot_results['direct_mean']:.6f}")
    print(f"  95% CI: [{boot_results['direct_ci_lower']:.6f}, {boot_results['direct_ci_upper']:.6f}]")

    print(f"\nTotal Effect:")
    print(f"  Mean: {boot_results['total_mean']:.6f}")
    print(f"  95% CI: [{boot_results['total_ci_lower']:.6f}, {boot_results['total_ci_upper']:.6f}]")

    # Trading interpretation
    print()
    print("=" * 70)
    print("TRADING INTERPRETATION")
    print("=" * 70)

    print("""
Based on the mediation analysis:

1. DIRECT EFFECT (BTC → ETH):
   - ETH has high beta to BTC
   - When BTC moves, ETH follows almost immediately
   - This is the "correlation" traders typically see

2. INDIRECT EFFECT (BTC → Funding → ETH):
   - When BTC rises, funding rates increase (longs pay shorts)
   - High funding indicates crowded long positions
   - This creates pressure that can dampen or reverse ETH returns

3. TRADING SIGNALS:
   - When BTC is up AND funding is normal: Expect full positive ETH move
   - When BTC is up AND funding is high: Expect muted ETH response
   - Monitor funding as a "sentiment dampener" indicator
""")

    # Run trading strategy backtest
    print("=" * 70)
    print("MEDIATION-AWARE TRADING STRATEGY BACKTEST")
    print("=" * 70)

    strategy = MediationTradingStrategy(
        lookback_window=100,  # More data for crypto
        mediation_threshold=0.2,  # Lower threshold for derivatives
        signal_threshold=0.2
    )

    config = BacktestConfig(
        initial_capital=10000,  # Smaller capital for crypto
        transaction_cost_bps=5,  # Lower fees on Bybit
        max_position_size=0.5  # Higher position size
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

    # Mediation regime analysis
    print()
    print("=" * 70)
    print("MEDIATION REGIME ANALYSIS")
    print("=" * 70)

    if 'mediator_active' in bt_results.columns:
        active = bt_results[bt_results['mediator_active'] == True]
        inactive = bt_results[bt_results['mediator_active'] == False]

        if len(active) > 0:
            active_return = active['return'].mean() * 252 * 24  # Annualize hourly
            print(f"\nWhen mediator (funding) is ACTIVE:")
            print(f"  Observations: {len(active)}")
            print(f"  Annualized return: {active_return:.2%}")

        if len(inactive) > 0:
            inactive_return = inactive['return'].mean() * 252 * 24
            print(f"\nWhen mediator (funding) is INACTIVE:")
            print(f"  Observations: {len(inactive)}")
            print(f"  Annualized return: {inactive_return:.2%}")

    # Save results
    print()
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save backtest results to CSV
    bt_results.to_csv('crypto_mediation_backtest.csv')
    print("Backtest results saved to crypto_mediation_backtest.csv")

    # Try to create visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # Portfolio value
        axes[0].plot(bt_results.index, bt_results['portfolio_value'])
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].set_title('Crypto Mediation Strategy - Equity Curve')
        axes[0].grid(True, alpha=0.3)

        # BTC and ETH returns
        axes[1].plot(bt_results.index, data['BTC_return'].rolling(24).mean(), label='BTC (24h MA)', alpha=0.7)
        axes[1].plot(bt_results.index, data['ETH_return'].rolling(24).mean(), label='ETH (24h MA)', alpha=0.7)
        axes[1].set_ylabel('Return')
        axes[1].set_title('BTC and ETH Returns')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Funding rate (mediator)
        axes[2].plot(bt_results.index, data['Funding_rate'].rolling(24).mean() * 10000, label='Funding Rate (bps, 24h MA)')
        axes[2].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='1 bps')
        axes[2].set_ylabel('Funding Rate (bps)')
        axes[2].set_title('Funding Rate (Mediator)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Mediation proportion
        axes[3].plot(bt_results.index, bt_results['proportion_mediated'].rolling(48).mean(), color='purple')
        axes[3].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[3].set_ylabel('Proportion Mediated')
        axes[3].set_title('Mediation Strength Over Time')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('crypto_mediation_results.png', dpi=150, bbox_inches='tight')
        print("Plot saved to crypto_mediation_results.png")
        plt.show()

    except ImportError:
        print("matplotlib not installed, skipping visualization")
    except Exception as e:
        print(f"Could not create plot: {e}")


if __name__ == "__main__":
    run_analysis()
