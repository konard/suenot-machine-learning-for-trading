"""
MC Dropout Trading - Main Entry Point
======================================

This script demonstrates the complete MC Dropout trading workflow:
1. Fetch data from Bybit
2. Create features
3. Train MC Dropout model
4. Generate trading signals with uncertainty
5. Run backtest

Usage:
    python main.py
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from loguru import logger
import sys

from bybit_client import BybitClient, fetch_training_data, DEFAULT_SYMBOLS
from mc_dropout_model import create_mc_dropout_model, MCDropoutRegressor
from concrete_dropout import create_concrete_dropout_model
from trading_strategy import MCDropoutTradingStrategy, StrategyConfig, SignalType
from backtest import BacktestEngine, run_backtest_example


def create_features(df: pd.DataFrame) -> np.ndarray:
    """
    Create features from OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        Feature array
    """
    features = pd.DataFrame(index=df.index)

    # Price returns at different lookbacks
    for period in [1, 5, 10, 20]:
        features[f'return_{period}'] = df['close'].pct_change(period)

    # Volatility
    features['volatility_10'] = df['close'].pct_change().rolling(10).std()
    features['volatility_20'] = df['close'].pct_change().rolling(20).std()

    # Volume features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_change'] = df['volume'].pct_change()

    # Price relative to moving averages
    for period in [10, 20, 50]:
        ma = df['close'].rolling(period).mean()
        features[f'price_ma_ratio_{period}'] = df['close'] / ma - 1

    # High-Low range
    features['range'] = (df['high'] - df['low']) / df['close']
    features['range_ma'] = features['range'].rolling(10).mean()

    # Price position in range
    features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    # Momentum
    features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    features['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    # RSI approximation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    features['rsi'] = (features['rsi'] - 50) / 50  # Normalize to [-1, 1]

    # Drop NaN rows
    features = features.dropna()

    return features.values.astype(np.float32), features.index


def create_targets(df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
    """
    Create target variable (future return).

    Args:
        df: DataFrame with close prices
        horizon: Prediction horizon in bars

    Returns:
        Target array
    """
    return df['close'].pct_change(horizon).shift(-horizon).values.astype(np.float32)


def run_live_example():
    """Run a live example fetching data from Bybit."""
    logger.info("=== MC Dropout Trading - Live Example ===")

    # Initialize client
    client = BybitClient()

    # Fetch data
    symbol = 'BTC/USDT:USDT'
    logger.info(f"Fetching data for {symbol}...")

    try:
        df = client.get_ohlcv(symbol, timeframe='1h', limit=500)
        logger.info(f"Fetched {len(df)} candles")
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        logger.info("Running with synthetic data instead...")
        return run_synthetic_example()

    # Create features and targets
    logger.info("Creating features...")
    features, feature_index = create_features(df)
    targets = create_targets(df, horizon=1)

    # Align features and targets
    df_aligned = df.loc[feature_index]
    targets_aligned = targets[df.index.get_indexer(feature_index)]

    # Remove NaN targets
    valid_mask = ~np.isnan(targets_aligned)
    features = features[valid_mask]
    targets_aligned = targets_aligned[valid_mask]
    prices = df_aligned['close'].values[valid_mask]
    timestamps = df_aligned.index[valid_mask]

    logger.info(f"Feature shape: {features.shape}")

    # Split data
    train_size = int(0.7 * len(features))
    X_train = features[:train_size]
    y_train = targets_aligned[:train_size]
    X_test = features[train_size:]
    y_test = targets_aligned[train_size:]
    prices_test = prices[train_size:]
    timestamps_test = timestamps[train_size:]

    # Train model
    logger.info("Training MC Dropout model...")
    model = create_mc_dropout_model(
        input_dim=features.shape[1],
        hidden_dims=[128, 64, 32],
        dropout_rate=0.1,
        n_mc_samples=50,
        task='regression'
    )
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, patience=10)

    # Create strategy
    config = StrategyConfig(
        confidence_threshold=1.5,
        uncertainty_threshold=0.02,
        max_position_size=0.1,
        use_kelly=True,
        n_mc_samples=50
    )
    strategy = MCDropoutTradingStrategy(model, config)

    # Generate signals for last few samples
    logger.info("\n=== Recent Trading Signals ===")
    for i in range(-5, 0):
        signal = strategy.generate_signal(X_test[i])
        actual = y_test[i]

        logger.info(f"Signal: {signal.signal_type.value:5s} | "
                   f"Pred: {signal.predicted_return*100:+.3f}% | "
                   f"Uncert: {signal.uncertainty*100:.3f}% | "
                   f"Conf: {signal.confidence:.2f} | "
                   f"Size: {signal.position_size:.3f} | "
                   f"Actual: {actual*100:+.3f}%")

    # Run backtest
    logger.info("\n=== Running Backtest ===")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        holding_period=1
    )

    result = engine.run(
        strategy=strategy,
        features=X_test,
        prices=prices_test,
        timestamps=timestamps_test,
        symbol=symbol
    )

    result.print_summary()

    return result


def run_synthetic_example():
    """Run example with synthetic data."""
    logger.info("=== MC Dropout Trading - Synthetic Data Example ===")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Generate features
    features = np.random.randn(n_samples, n_features).astype(np.float32)

    # Generate price series with signal
    signal = features[:, 0] * 0.002 + features[:, 1] * 0.001
    noise = np.random.randn(n_samples) * 0.005
    returns = signal + noise
    prices = 100 * np.cumprod(1 + returns)

    # Create targets (next period return)
    targets = returns[1:]
    features = features[:-1]
    prices = prices[:-1]

    # Split data
    train_size = int(0.7 * len(features))
    X_train = features[:train_size]
    y_train = targets[:train_size]
    X_test = features[train_size:]
    y_test = targets[train_size:]
    prices_test = prices[train_size:]

    # Train model
    logger.info("Training MC Dropout model...")
    model = create_mc_dropout_model(
        input_dim=n_features,
        hidden_dims=[64, 32],
        dropout_rate=0.1,
        n_mc_samples=50,
        task='regression'
    )
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, patience=10)

    # Create strategy
    config = StrategyConfig(
        confidence_threshold=1.0,
        uncertainty_threshold=0.03,
        max_position_size=0.1,
        use_kelly=True,
        n_mc_samples=50
    )
    strategy = MCDropoutTradingStrategy(model, config)

    # Generate signals
    logger.info("\n=== Sample Trading Signals ===")
    for i in range(5):
        signal = strategy.generate_signal(X_test[i])
        actual = y_test[i]

        logger.info(f"Signal: {signal.signal_type.value:5s} | "
                   f"Pred: {signal.predicted_return*100:+.4f}% | "
                   f"Uncert: {signal.uncertainty*100:.4f}% | "
                   f"Conf: {signal.confidence:.2f} | "
                   f"Actual: {actual*100:+.4f}%")

    # Run backtest
    logger.info("\n=== Running Backtest ===")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        holding_period=1
    )

    result = engine.run(
        strategy=strategy,
        features=X_test,
        prices=prices_test,
        symbol="SYNTHETIC"
    )

    result.print_summary()

    return result


def compare_models():
    """Compare MC Dropout with Concrete Dropout."""
    logger.info("=== Model Comparison: MC Dropout vs Concrete Dropout ===")

    # Generate data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    features = np.random.randn(n_samples, n_features).astype(np.float32)
    signal = features[:, 0] * 0.002 + features[:, 1] * 0.001
    noise = np.random.randn(n_samples) * 0.005
    targets = (signal + noise).astype(np.float32)

    train_size = int(0.8 * len(features))
    X_train, X_val = features[:train_size], features[train_size:]
    y_train, y_val = targets[:train_size], targets[train_size:]

    # Train MC Dropout
    logger.info("\nTraining MC Dropout model...")
    mc_model = create_mc_dropout_model(
        input_dim=n_features,
        hidden_dims=[64, 32],
        dropout_rate=0.1,
        n_mc_samples=50,
        task='regression'
    )
    mc_model.fit(X_train, y_train, X_val, y_val, epochs=50, patience=10)

    # Train Concrete Dropout
    logger.info("\nTraining Concrete Dropout model...")
    concrete_model = create_concrete_dropout_model(
        input_dim=n_features,
        hidden_dims=[64, 32],
        n_mc_samples=50
    )
    concrete_model.fit(X_train, y_train, X_val, y_val, epochs=50, patience=10)

    # Compare predictions
    X_test = torch.FloatTensor(X_val[:10])

    logger.info("\n=== Prediction Comparison ===")
    mc_result = mc_model.predict_with_uncertainty(X_test)
    concrete_result = concrete_model.predict_with_uncertainty(X_test)

    logger.info("\nMC Dropout predictions:")
    for i in range(5):
        logger.info(f"  Sample {i+1}: {mc_result['mean'][i].item():.4f} +/- {mc_result['std'][i].item():.4f}")

    logger.info("\nConcrete Dropout predictions:")
    for i in range(5):
        logger.info(f"  Sample {i+1}: {concrete_result['mean'][i].item():.4f} +/- {concrete_result['std'][i].item():.4f}")

    logger.info(f"\nConcrete Dropout learned rates: {concrete_model.get_dropout_rates()}")


def main():
    """Main entry point."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("=" * 60)
    print("MC Dropout Trading - Machine Learning for Trading")
    print("=" * 60)
    print()

    print("Select mode:")
    print("1. Live data from Bybit")
    print("2. Synthetic data example")
    print("3. Model comparison (MC vs Concrete Dropout)")
    print()

    try:
        choice = input("Enter choice (1/2/3) [default: 2]: ").strip() or "2"

        if choice == "1":
            run_live_example()
        elif choice == "2":
            run_synthetic_example()
        elif choice == "3":
            compare_models()
        else:
            print("Invalid choice. Running synthetic example...")
            run_synthetic_example()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
