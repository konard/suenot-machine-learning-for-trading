#!/usr/bin/env python3
"""
Example 3: Backtesting Anomaly-Based Trading Strategies

This example demonstrates how to backtest trading strategies
based on anomaly detection signals.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import YahooFinanceLoader, BybitDataLoader
from detector import StatisticalAnomalyDetector, EnsembleAnomalyDetector
from signals import AnomalySignalGenerator, MultiStrategySignalGenerator
from backtest import AnomalyBacktester, BacktestResult


def backtest_contrarian_strategy():
    """Backtest a contrarian strategy on SPY."""
    print("=" * 60)
    print("CONTRARIAN STRATEGY BACKTEST - SPY")
    print("=" * 60)

    # Load SPY data
    loader = YahooFinanceLoader()
    data = loader.get_ohlcv("SPY", interval="1d", limit=1000)

    if data.empty or len(data) < 200:
        print("Insufficient data for backtest")
        return

    data = loader.compute_features(data)
    print(f"Loaded {len(data)} days of SPY data")

    # Create detector and signal generator
    detector = StatisticalAnomalyDetector(
        z_threshold=2.5,
        contamination=0.05,
    )

    signal_gen = AnomalySignalGenerator(
        strategy="contrarian",
        min_anomaly_score=0.6,
        position_sizing="score_based",
    )

    # Create backtester
    backtester = AnomalyBacktester(
        detector=detector,
        signal_generator=signal_gen,
        initial_capital=100000,
        position_size=0.1,
        max_positions=3,
        transaction_cost=0.001,
    )

    # Run backtest
    print("\nRunning backtest...")
    result = backtester.run(
        data,
        train_period=200,
        stop_loss=0.05,
        take_profit=0.10,
        walk_forward=True,
        retrain_period=50,
    )

    result.print_summary()

    # Show some example trades
    if result.trades:
        print("\nExample Trades:")
        print("-" * 60)
        for trade in result.trades[:5]:
            direction = "LONG" if trade.direction > 0 else "SHORT"
            print(f"  {direction}: Entry ${trade.entry_price:.2f} -> "
                  f"Exit ${trade.exit_price:.2f} = ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
            print(f"    Reason: {trade.reason}")

    return result


def backtest_crypto_strategy():
    """Backtest anomaly strategy on BTC."""
    print("\n" + "=" * 60)
    print("CRYPTO STRATEGY BACKTEST - BTCUSDT (Bybit)")
    print("=" * 60)

    # Load BTC data from Bybit
    loader = BybitDataLoader()
    data = loader.get_ohlcv("BTCUSDT", interval="4h", limit=1000)

    if data.empty or len(data) < 200:
        print("Insufficient data for backtest")
        return

    data = loader.compute_features(data)
    print(f"Loaded {len(data)} 4-hour candles of BTCUSDT")

    # Create detector with crypto-optimized settings
    detector = StatisticalAnomalyDetector(
        z_threshold=3.0,  # Higher threshold for volatile crypto
        contamination=0.03,
        methods=["zscore", "isolation_forest", "lof"],
    )

    # Use risk-based signal generation for crypto
    signal_gen = AnomalySignalGenerator(
        strategy="risk",  # Risk management focus
        min_anomaly_score=0.7,
        position_sizing="fixed",
        base_position_size=0.05,  # Smaller positions for crypto
    )

    # Create backtester with crypto-appropriate parameters
    backtester = AnomalyBacktester(
        detector=detector,
        signal_generator=signal_gen,
        initial_capital=10000,  # Smaller capital for crypto demo
        position_size=0.05,
        max_positions=2,
        transaction_cost=0.001,  # Bybit trading fee
        slippage=0.001,  # Higher slippage for crypto
    )

    # Run backtest
    print("\nRunning backtest...")
    result = backtester.run(
        data,
        train_period=200,
        stop_loss=0.08,  # Wider stops for crypto volatility
        take_profit=0.15,
        walk_forward=True,
        retrain_period=100,
    )

    result.print_summary()
    return result


def compare_strategies():
    """Compare different strategies on the same data."""
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)

    # Load data
    loader = BybitDataLoader()
    data = loader.get_ohlcv("ETHUSDT", interval="1h", limit=1000)

    if data.empty or len(data) < 300:
        print("Insufficient data")
        return

    data = loader.compute_features(data)
    print(f"Loaded {len(data)} hours of ETHUSDT data")

    strategies = ["contrarian", "momentum", "risk"]
    results = {}

    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")

        detector = StatisticalAnomalyDetector(z_threshold=2.5)
        signal_gen = AnomalySignalGenerator(
            strategy=strategy,
            min_anomaly_score=0.6,
        )

        backtester = AnomalyBacktester(
            detector=detector,
            signal_generator=signal_gen,
            initial_capital=10000,
            position_size=0.1,
        )

        result = backtester.run(
            data,
            train_period=200,
            stop_loss=0.05,
            take_profit=0.10,
        )

        results[strategy] = result

    # Print comparison
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Strategy':<15} {'Return %':<12} {'Sharpe':<10} {'MaxDD %':<12} {'Win Rate':<10} {'Trades'}")
    print("-" * 70)

    for strategy, result in results.items():
        print(f"{strategy:<15} {result.total_return_pct:>10.2f}% {result.sharpe_ratio:>10.2f} "
              f"{result.max_drawdown_pct:>10.2f}% {result.win_rate:>9.1f}% {result.num_trades:>6}")

    return results


def backtest_ensemble():
    """Backtest using ensemble of detectors."""
    print("\n" + "=" * 60)
    print("ENSEMBLE DETECTOR BACKTEST")
    print("=" * 60)

    # Load data
    loader = BybitDataLoader()
    data = loader.get_ohlcv("BTCUSDT", interval="1h", limit=800)

    if data.empty or len(data) < 300:
        print("Insufficient data")
        return

    data = loader.compute_features(data)
    print(f"Loaded {len(data)} hours of BTCUSDT data")

    # Create ensemble detector
    ensemble = EnsembleAnomalyDetector(
        detectors=[
            StatisticalAnomalyDetector(z_threshold=2.0),
            StatisticalAnomalyDetector(z_threshold=3.0, methods=["isolation_forest"]),
            StatisticalAnomalyDetector(z_threshold=2.5, methods=["lof"]),
        ],
        voting="soft",
        threshold=0.5,
    )

    # Multi-strategy signal generator
    signal_gen = MultiStrategySignalGenerator(
        strategies=["contrarian", "risk"],
        weights={"contrarian": 0.6, "risk": 0.4},
    )

    # Backtest
    backtester = AnomalyBacktester(
        detector=ensemble,
        signal_generator=AnomalySignalGenerator(strategy="contrarian"),
        initial_capital=10000,
    )

    print("\nRunning ensemble backtest...")
    result = backtester.run(
        data,
        train_period=200,
        stop_loss=0.06,
        take_profit=0.12,
    )

    result.print_summary()
    return result


def parameter_sensitivity_analysis():
    """Analyze sensitivity to detector parameters."""
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Load data
    loader = BybitDataLoader()
    data = loader.get_ohlcv("BTCUSDT", interval="1h", limit=600)

    if data.empty or len(data) < 300:
        print("Insufficient data")
        return

    data = loader.compute_features(data)

    # Test different z-score thresholds
    z_thresholds = [1.5, 2.0, 2.5, 3.0, 3.5]
    results = {}

    print("\nTesting different z-score thresholds...")
    print("-" * 50)

    for z in z_thresholds:
        detector = StatisticalAnomalyDetector(z_threshold=z)
        signal_gen = AnomalySignalGenerator(strategy="contrarian")

        backtester = AnomalyBacktester(
            detector=detector,
            signal_generator=signal_gen,
            initial_capital=10000,
        )

        result = backtester.run(
            data,
            train_period=200,
        )

        results[z] = result
        print(f"Z={z}: Return={result.total_return_pct:>7.2f}%, "
              f"Sharpe={result.sharpe_ratio:>6.2f}, "
              f"Trades={result.num_trades:>3}")

    # Find best threshold
    best_z = max(results, key=lambda z: results[z].sharpe_ratio)
    print(f"\nBest z-score threshold: {best_z} (Sharpe: {results[best_z].sharpe_ratio:.2f})")

    return results


if __name__ == "__main__":
    print("Anomaly-Based Trading Strategy Backtest Examples")
    print("=" * 60)

    # Run backtests
    spy_result = backtest_contrarian_strategy()
    btc_result = backtest_crypto_strategy()
    comparison = compare_strategies()
    ensemble_result = backtest_ensemble()
    sensitivity = parameter_sensitivity_analysis()

    print("\n" + "=" * 60)
    print("All backtests completed!")
    print("=" * 60)
