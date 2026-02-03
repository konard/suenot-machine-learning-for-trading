"""
Example usage of counterfactual explanations for trading.

This script demonstrates how to:
1. Load and prepare trading data
2. Train a trading classifier
3. Generate counterfactual explanations
4. Use counterfactuals for risk management
5. Backtest a strategy with counterfactual insights
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from model import TradingClassifier, TradingClassifierTrainer, create_trading_labels
from counterfactual import (
    CounterfactualOptimizer,
    CounterfactualGenerator,
    DiCE,
    format_explanation
)
from data import get_sample_data, BybitClient, prepare_features
from backtest import Backtester, CounterfactualRiskManager, evaluate_counterfactual_quality


def main():
    """Main example demonstrating counterfactual explanations."""
    print("=" * 60)
    print("Counterfactual Explanations for Trading - Example")
    print("=" * 60)

    # 1. Load sample data
    print("\n1. Loading sample data...")
    features, labels, feature_names = get_sample_data()
    print(f"   Features shape: {features.shape}")
    print(f"   Feature names: {feature_names}")
    print(f"   Label distribution: SELL={sum(labels==0)}, HOLD={sum(labels==1)}, BUY={sum(labels==2)}")

    # 2. Prepare data for training
    print("\n2. Preparing training data...")
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Split into train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 3. Train classifier
    print("\n3. Training trading classifier...")
    model = TradingClassifier(
        input_dim=features.shape[1],
        hidden_dim=64,
        num_classes=3
    )

    trainer = TradingClassifierTrainer(model, learning_rate=1e-3)
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=50,
        early_stopping_patience=10,
        verbose=True
    )
    print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.2%}")

    # 4. Generate counterfactual explanations
    print("\n4. Generating counterfactual explanations...")

    # Pick a sample to explain
    sample_idx = 0
    x_sample = X_val[sample_idx:sample_idx+1]
    y_sample = y_val[sample_idx].item()

    print(f"\n   Sample input (index {sample_idx}):")
    for i, name in enumerate(feature_names):
        print(f"     {name}: {x_sample[0, i].item():.4f}")

    with torch.no_grad():
        probs = model.predict_proba(x_sample)
        pred = model.predict(x_sample).item()
    print(f"\n   Model prediction: {model.get_class_name(pred)} ({probs[0, pred].item():.1%})")
    print(f"   True label: {model.get_class_name(y_sample)}")

    # Generate counterfactual to flip the prediction
    optimizer = CounterfactualOptimizer(
        model,
        lambda_proximity=1.0,
        lambda_validity=1.0,
        lambda_sparsity=0.1
    )

    # Find target class (opposite of prediction)
    target_class = 0 if pred == 2 else (2 if pred == 0 else 2)

    cf_result = optimizer.generate(
        x_sample,
        target_class,
        num_steps=100,
        lr=0.01
    )

    print("\n   Counterfactual Explanation:")
    print(format_explanation(cf_result, feature_names))

    # 5. Generate diverse counterfactuals
    print("\n5. Generating diverse counterfactuals (DiCE)...")
    dice = DiCE(model, num_counterfactuals=3, diversity_weight=0.5)
    diverse_cfs = dice.generate(x_sample, target_class, num_steps=150)

    print(f"   Generated {len(diverse_cfs)} diverse counterfactuals:")
    for i, cf in enumerate(diverse_cfs):
        print(f"\n   Counterfactual {i+1}:")
        print(f"     Valid: {cf.is_valid}")
        print(f"     Features changed: {cf.num_features_changed}")
        print(f"     L1 distance: {cf.l1_distance:.4f}")
        if cf.changed_features:
            for idx, orig, new in cf.changed_features[:3]:  # Show top 3
                print(f"     - {feature_names[idx]}: {orig:.3f} -> {new:.3f}")

    # 6. Evaluate counterfactual quality
    print("\n6. Evaluating counterfactual quality across validation set...")

    cf_results = []
    for i in range(min(50, len(X_val))):  # Evaluate first 50 samples
        x = X_val[i:i+1]
        pred = model.predict(x).item()
        if pred != 1:  # Only for non-HOLD predictions
            target = 0 if pred == 2 else 2
            result = optimizer.generate(x, target, num_steps=50)
            cf_results.append(result)

    quality_metrics = evaluate_counterfactual_quality(cf_results)
    print(f"   Validity rate: {quality_metrics['validity_rate']:.1%}")
    print(f"   Avg features changed: {quality_metrics['avg_features_changed']:.2f}")
    print(f"   Avg L1 distance: {quality_metrics['avg_l1_distance']:.4f}")

    # 7. Backtest with counterfactual-based risk management
    print("\n7. Backtesting with counterfactual risk management...")

    # Generate synthetic prices for backtesting
    np.random.seed(42)
    n_prices = len(X_val)
    returns = np.random.normal(0.001, 0.02, n_prices)
    prices = 100 * np.cumprod(1 + returns)

    # Generate predictions
    with torch.no_grad():
        predictions = model.predict(X_val).numpy()

    # Calculate counterfactual distances
    cf_distances = np.zeros(n_prices)
    for i in range(n_prices):
        x = X_val[i:i+1]
        pred = predictions[i]
        if pred != 1:
            target = 0 if pred == 2 else 2
            result = optimizer.generate(x, target, num_steps=30)
            cf_distances[i] = result.l1_distance
        else:
            cf_distances[i] = 0.0

    # Run backtest without CF filtering
    backtester = Backtester(initial_capital=10000)
    result_no_cf = backtester.run(prices, predictions.copy())

    print("\n   Results WITHOUT counterfactual filtering:")
    print(f"     Total return: {result_no_cf.total_return:.2%}")
    print(f"     Sharpe ratio: {result_no_cf.sharpe_ratio:.2f}")
    print(f"     Max drawdown: {result_no_cf.max_drawdown:.2%}")
    print(f"     Win rate: {result_no_cf.win_rate:.2%}")
    print(f"     Num trades: {result_no_cf.num_trades}")

    # Run backtest with CF filtering
    result_with_cf = backtester.run(
        prices,
        predictions.copy(),
        cf_distances=cf_distances,
        cf_threshold=0.5  # Only trade if CF distance > 0.5
    )

    print("\n   Results WITH counterfactual filtering (threshold=0.5):")
    print(f"     Total return: {result_with_cf.total_return:.2%}")
    print(f"     Sharpe ratio: {result_with_cf.sharpe_ratio:.2f}")
    print(f"     Max drawdown: {result_with_cf.max_drawdown:.2%}")
    print(f"     Win rate: {result_with_cf.win_rate:.2%}")
    print(f"     Num trades: {result_with_cf.num_trades}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


def demo_real_data():
    """Demo with real market data (requires network)."""
    print("\nFetching real market data from Bybit...")

    client = BybitClient()
    df = client.get_dataframe("BTCUSDT", "60", limit=200)

    if df.empty:
        print("Failed to fetch data. Using sample data instead.")
        return

    print(f"Fetched {len(df)} candles")
    print(df.tail())

    # Prepare features
    features, feature_names = prepare_features(df)
    print(f"Prepared {len(features)} samples with {len(feature_names)} features")


if __name__ == "__main__":
    main()

    # Uncomment to try with real data:
    # demo_real_data()
