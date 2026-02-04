#!/usr/bin/env python3
"""
Deep Ensembles Trading - Complete Example

This script demonstrates the full pipeline of using Deep Ensembles
for uncertainty-aware cryptocurrency trading.

Usage:
    python example.py [--live]  # --live for real data from Bybit
"""

import argparse
import numpy as np
from typing import Optional

from data_fetcher import (
    BybitDataFetcher,
    prepare_features,
    generate_synthetic_data
)
from deep_ensemble import DeepEnsemble, compute_calibration_error
from strategy import (
    UncertaintyAwareStrategy,
    MarketRegimeDetector,
    SignalType,
    compute_trading_metrics
)
from backtest import (
    BacktestEngine,
    generate_synthetic_prices,
    generate_synthetic_features,
    print_backtest_report
)


def run_synthetic_example():
    """Run example with synthetic data."""
    print("\n" + "=" * 70)
    print("DEEP ENSEMBLES TRADING - SYNTHETIC DATA EXAMPLE")
    print("=" * 70)

    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic market data...")
    np.random.seed(42)

    n_samples = 2000
    n_features = 8

    # Generate features and targets
    features, targets = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        noise_level=0.3
    )

    # Split data
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)

    X_train = features[:train_size]
    y_train = targets[:train_size]
    X_val = features[train_size:train_size+val_size]
    y_val = targets[train_size:train_size+val_size]
    X_test = features[train_size+val_size:]
    y_test = targets[train_size+val_size:]

    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")

    # Step 2: Create and train Deep Ensemble
    print("\n[Step 2] Creating Deep Ensemble with 5 models...")
    ensemble = DeepEnsemble(
        input_dim=n_features,
        hidden_dims=[128, 64, 32],
        num_models=5,
        dropout_rate=0.1
    )

    print("\n[Step 3] Training ensemble (this may take a moment)...")
    history = ensemble.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=64,
        learning_rate=0.005,
        early_stopping_patience=15,
        verbose=True
    )

    # Step 4: Evaluate model
    print("\n[Step 4] Evaluating model on test set...")
    predictions = ensemble.predict(X_test)

    mse = np.mean((predictions.mean - y_test) ** 2)
    mae = np.mean(np.abs(predictions.mean - y_test))
    ece = compute_calibration_error(predictions, y_test)

    print(f"\n   Model Performance:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   Expected Calibration Error: {ece:.4f}")

    # Step 5: Analyze uncertainty
    print("\n[Step 5] Uncertainty Analysis:")
    print(f"   Average Total Uncertainty: {predictions.total_std.mean():.4f}")
    print(f"   Average Epistemic Uncertainty: {predictions.epistemic_std.mean():.4f}")
    print(f"   Average Aleatoric Uncertainty: {predictions.aleatoric_std.mean():.4f}")
    print(f"   Epistemic/Total Ratio: {(predictions.epistemic_std / predictions.total_std).mean():.2%}")

    # Show predictions for a few samples
    print("\n   Sample Predictions (first 5 test samples):")
    print("   " + "-" * 60)
    print(f"   {'True':>10} | {'Predicted':>10} | {'Std':>8} | {'Epist':>8} | {'Aleat':>8}")
    print("   " + "-" * 60)
    for i in range(5):
        print(f"   {y_test[i]:>10.4f} | {predictions.mean[i]:>10.4f} | "
              f"{predictions.total_std[i]:>8.4f} | {predictions.epistemic_std[i]:>8.4f} | "
              f"{predictions.aleatoric_std[i]:>8.4f}")

    # Step 6: Create trading strategy
    print("\n[Step 6] Creating uncertainty-aware trading strategy...")
    strategy = UncertaintyAwareStrategy(
        ensemble=ensemble,
        confidence_threshold=0.6,
        epistemic_threshold=0.5,
        max_position_size=0.1,
        signal_strength_threshold=1.2
    )

    # Generate signals
    symbols = ["BTC/USDT"]
    signals = strategy.generate_signals(X_test, symbols * len(X_test))

    # Count signal types
    long_signals = sum(1 for s in signals if s.signal_type == SignalType.LONG)
    short_signals = sum(1 for s in signals if s.signal_type == SignalType.SHORT)
    hold_signals = sum(1 for s in signals if s.signal_type == SignalType.HOLD)

    print(f"\n   Signal Distribution:")
    print(f"   LONG:  {long_signals} ({long_signals/len(signals)*100:.1f}%)")
    print(f"   SHORT: {short_signals} ({short_signals/len(signals)*100:.1f}%)")
    print(f"   HOLD:  {hold_signals} ({hold_signals/len(signals)*100:.1f}%)")

    # Step 7: Run backtest
    print("\n[Step 7] Running backtest simulation...")

    # Generate price series for backtest
    prices = generate_synthetic_prices(
        n_steps=len(X_test),
        initial_price=45000,  # BTC-like price
        drift=0.0001,
        volatility=0.02
    )

    engine = BacktestEngine(
        initial_capital=100000,
        trading_fee=0.001,
        slippage=0.0005
    )

    result = engine.run(
        strategy=strategy,
        features=X_test,
        prices=prices,
        symbols=["BTC/USDT"],
        verbose=False
    )

    # Print backtest report
    print_backtest_report(result)

    # Step 8: Save model
    print("\n[Step 8] Saving model...")
    ensemble.save("deep_ensemble_model.json")
    print("   Model saved to: deep_ensemble_model.json")

    # Demonstrate loading
    print("   Testing model loading...")
    loaded_ensemble = DeepEnsemble.load("deep_ensemble_model.json")
    loaded_predictions = loaded_ensemble.predict(X_test[:5])
    print(f"   Loaded model predictions match: {np.allclose(predictions.mean[:5], loaded_predictions.mean)}")

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)


def run_live_example():
    """Run example with live data from Bybit."""
    print("\n" + "=" * 70)
    print("DEEP ENSEMBLES TRADING - LIVE DATA EXAMPLE")
    print("=" * 70)

    # Step 1: Fetch data from Bybit
    print("\n[Step 1] Fetching data from Bybit...")
    fetcher = BybitDataFetcher()

    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    data = fetcher.fetch_multiple_symbols(symbols, timeframe='1h', limit=500)

    if not data:
        print("   ERROR: Could not fetch data. Running synthetic example instead.")
        run_synthetic_example()
        return

    for symbol, df in data.items():
        print(f"   {symbol}: {len(df)} candles, "
              f"last price: ${df['close'].iloc[-1]:.2f}")

    # Step 2: Prepare features
    print("\n[Step 2] Preparing features...")
    symbol = 'BTC/USDT'
    df = data[symbol]

    features, targets = prepare_features(df, lookback=60)
    print(f"   Features shape: {features.shape}")
    print(f"   Targets shape: {targets.shape}")

    # Split data
    train_size = int(0.6 * len(features))
    val_size = int(0.2 * len(features))

    X_train = features[:train_size]
    y_train = targets[:train_size]
    X_val = features[train_size:train_size+val_size]
    y_val = targets[train_size:train_size+val_size]
    X_test = features[train_size+val_size:]
    y_test = targets[train_size+val_size:]

    print(f"   Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # Step 3: Train ensemble
    print("\n[Step 3] Training Deep Ensemble...")
    ensemble = DeepEnsemble(
        input_dim=features.shape[1],
        hidden_dims=[64, 32],
        num_models=5
    )

    ensemble.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32,
        verbose=True
    )

    # Step 4: Make predictions
    print("\n[Step 4] Making predictions on test set...")
    predictions = ensemble.predict(X_test)

    print(f"\n   Prediction Statistics:")
    print(f"   Mean prediction: {predictions.mean.mean():.4f}")
    print(f"   Total uncertainty: {predictions.total_std.mean():.4f}")
    print(f"   Epistemic uncertainty: {predictions.epistemic_std.mean():.4f}")
    print(f"   Aleatoric uncertainty: {predictions.aleatoric_std.mean():.4f}")

    # Step 5: Generate current signal
    print("\n[Step 5] Generating current trading signal...")
    strategy = UncertaintyAwareStrategy(
        ensemble=ensemble,
        confidence_threshold=0.6,
        epistemic_threshold=0.5,
        max_position_size=0.1
    )

    # Use latest data point
    latest_features = X_test[-1:] if len(X_test) > 0 else features[-1:]
    signals = strategy.generate_signals(latest_features, [symbol])

    if signals:
        signal = signals[0]
        print(f"\n   Current Signal for {symbol}:")
        print(f"   Signal Type: {signal.signal_type.value}")
        print(f"   Confidence: {signal.confidence:.2%}")
        print(f"   Predicted Return: {signal.predicted_return*100:.2f}%")
        print(f"   Total Uncertainty: {signal.total_uncertainty:.4f}")
        print(f"   Epistemic Uncertainty: {signal.epistemic_uncertainty:.4f}")

        if signal.signal_type == SignalType.HOLD:
            print("\n   Recommendation: NO TRADE (high uncertainty or weak signal)")
        elif signal.signal_type == SignalType.LONG:
            position_size = strategy.calculate_position_size(signal, 100000)
            print(f"\n   Recommendation: BUY with position size ${position_size:.2f}")
        else:
            position_size = strategy.calculate_position_size(signal, 100000)
            print(f"\n   Recommendation: SELL/SHORT with position size ${abs(position_size):.2f}")

    # Step 6: Run backtest
    print("\n[Step 6] Running backtest on test period...")
    prices_test = df['close'].values[train_size+val_size:train_size+val_size+len(X_test)]

    if len(prices_test) == len(X_test):
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(
            strategy=strategy,
            features=X_test,
            prices=prices_test,
            symbols=[symbol],
            verbose=False
        )
        print_backtest_report(result)
    else:
        print("   Could not run backtest due to data length mismatch")

    print("\n" + "=" * 70)
    print("LIVE EXAMPLE COMPLETE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Deep Ensembles Trading Example')
    parser.add_argument('--live', action='store_true',
                       help='Use live data from Bybit instead of synthetic data')
    args = parser.parse_args()

    print("\n" + "*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "     DEEP ENSEMBLES FOR CRYPTOCURRENCY TRADING".center(68) + "*")
    print("*" + "     Uncertainty-Aware Machine Learning".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    print("\nThis example demonstrates:")
    print("  1. Training a Deep Ensemble model")
    print("  2. Estimating prediction uncertainty (epistemic vs aleatoric)")
    print("  3. Using uncertainty for trading decisions")
    print("  4. Position sizing based on confidence")
    print("  5. Backtesting the strategy")

    if args.live:
        try:
            run_live_example()
        except ImportError as e:
            print(f"\nError: {e}")
            print("Running synthetic example instead...")
            run_synthetic_example()
    else:
        run_synthetic_example()

    print("\n" + "-" * 70)
    print("DISCLAIMER: This is for educational purposes only.")
    print("Cryptocurrency trading involves substantial risk.")
    print("Do not trade with money you cannot afford to lose.")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
