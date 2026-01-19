#!/usr/bin/env python3
"""
Example usage of Multi-Scale Attention for financial time series prediction.

This script demonstrates the complete workflow:
1. Fetch data from Bybit
2. Create multi-scale features
3. Build and train the model
4. Make predictions
5. Backtest a trading strategy
"""

import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from our modules
from data import (
    fetch_bybit_klines_extended,
    create_multi_scale_features,
    create_sequences,
    MultiScaleDataset,
    prepare_train_val_test_split,
    generate_synthetic_data,
)
from model import MultiScaleAttention
from strategy import (
    backtest_multi_scale_strategy,
    compare_scale_strategies,
    generate_performance_report,
    plot_equity_curve,
)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict:
    """Train the multi-scale attention model."""
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    history = {"train_loss": [], "val_loss": [], "val_direction_acc": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = model.compute_loss(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        correct_directions = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = {k: v.to(device) for k, v in batch_x.items()}
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = model.compute_loss(outputs, batch_y)
                val_losses.append(loss.item())

                # Direction accuracy
                pred_direction = (outputs["direction"] > 0.5).float()
                true_direction = (batch_y > 0).float()
                correct_directions += (pred_direction == true_direction).sum().item()
                total += batch_y.size(0)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        direction_acc = correct_directions / total if total > 0 else 0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_direction_acc"].append(direction_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")
            print(f"  Direction Acc: {direction_acc:.4f}")

    return history


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, np.ndarray]:
    """Make predictions with the trained model."""
    model.eval()
    model = model.to(device)

    all_predictions = {
        "short_term": [],
        "medium_term": [],
        "long_term": [],
        "direction": [],
        "uncertainty": [],
    }

    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = {k: v.to(device) for k, v in batch_x.items()}
            outputs = model(batch_x)

            for key in all_predictions:
                all_predictions[key].append(outputs[key].cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in all_predictions.items()}


def run_synthetic_example():
    """Run example with synthetic data."""
    print("=" * 60)
    print("MULTI-SCALE ATTENTION - SYNTHETIC DATA EXAMPLE")
    print("=" * 60)

    # Configuration
    scales = ["1min", "5min", "1H"]
    lookbacks = {"1min": 60, "5min": 24, "1H": 12}
    n_features = 15

    print("\n1. Generating synthetic data...")
    X, y = generate_synthetic_data(
        n_samples=2000,
        scales=scales,
        n_features=n_features,
        lookbacks=lookbacks,
    )

    print(f"   Generated {len(y)} samples")
    for scale in scales:
        print(f"   Scale {scale}: shape {X[scale].shape}")

    # Split data
    print("\n2. Splitting data...")
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = prepare_train_val_test_split(
        X, y
    )
    print(f"   Train: {len(train_y)}, Val: {len(val_y)}, Test: {len(test_y)}")

    # Create datasets
    train_dataset = MultiScaleDataset(train_X, train_y, scales)
    val_dataset = MultiScaleDataset(val_X, val_y, scales)
    test_dataset = MultiScaleDataset(test_X, test_y, scales)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Create model
    print("\n3. Creating model...")
    scale_configs = {
        scale: {"input_dim": n_features, "seq_len": lookbacks[scale]} for scale in scales
    }

    model = MultiScaleAttention(
        scale_configs=scale_configs,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        dropout=0.1,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Train model
    print("\n4. Training model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")

    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=50,
        learning_rate=0.001,
        device=device,
        verbose=True,
    )

    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))

    # Make predictions
    print("\n5. Making predictions...")
    predictions = predict(model, test_loader, device)
    print(f"   Predictions shape: {predictions['direction'].shape}")

    # Get scale importance
    print("\n6. Scale importance:")
    importance = model.get_scale_importance()
    for scale, weight in importance.items():
        print(f"   {scale}: {weight:.4f}")

    # Simulate backtest with synthetic prices
    print("\n7. Backtesting strategy...")
    n_test = len(test_y)
    synthetic_prices = pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(np.random.randn(n_test) * 0.01))}
    )

    result = backtest_multi_scale_strategy(
        predictions,
        synthetic_prices,
        initial_capital=100000,
        transaction_cost=0.001,
        confidence_threshold=0.55,
    )

    print(generate_performance_report(result))

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return model, history, predictions, result


def run_bybit_example(
    symbol: str = "BTCUSDT",
    days: int = 7,
):
    """Run example with real Bybit data."""
    print("=" * 60)
    print(f"MULTI-SCALE ATTENTION - BYBIT {symbol} EXAMPLE")
    print("=" * 60)

    # Configuration
    scales = ["5min", "15min", "1H", "4H"]
    lookbacks = {"5min": 48, "15min": 24, "1H": 24, "4H": 12}

    print(f"\n1. Fetching {days} days of {symbol} data from Bybit...")
    try:
        df = fetch_bybit_klines_extended(
            symbol=symbol,
            interval="5",
            days=days,
            verbose=True,
        )
        print(f"   Fetched {len(df)} candles")
    except Exception as e:
        print(f"   Error fetching data: {e}")
        print("   Falling back to synthetic example...")
        return run_synthetic_example()

    if len(df) < 1000:
        print("   Not enough data, falling back to synthetic example...")
        return run_synthetic_example()

    print("\n2. Creating multi-scale features...")
    multi_scale_features = create_multi_scale_features(df, scales)

    for scale, features in multi_scale_features.items():
        print(f"   Scale {scale}: {len(features)} samples, {features.shape[1]} features")

    print("\n3. Creating sequences...")
    X, y, timestamps = create_sequences(
        multi_scale_features,
        lookback_periods=lookbacks,
        prediction_horizon=1,
        target_scale="1H",
    )

    print(f"   Created {len(y)} sequences")

    # Split data
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = prepare_train_val_test_split(
        X, y
    )

    # Create datasets
    valid_scales = list(X.keys())
    train_dataset = MultiScaleDataset(train_X, train_y, valid_scales)
    val_dataset = MultiScaleDataset(val_X, val_y, valid_scales)
    test_dataset = MultiScaleDataset(test_X, test_y, valid_scales)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Create model
    print("\n4. Creating model...")
    scale_configs = {}
    for scale in valid_scales:
        n_features = X[scale].shape[2]
        seq_len = lookbacks[scale]
        scale_configs[scale] = {"input_dim": n_features, "seq_len": seq_len}

    model = MultiScaleAttention(
        scale_configs=scale_configs,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        dropout=0.1,
    )

    # Train model
    print("\n5. Training model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=30,
        device=device,
        verbose=True,
    )

    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))

    # Make predictions
    print("\n6. Making predictions...")
    predictions = predict(model, test_loader, device)

    # Get prices for backtesting
    test_start_idx = int(len(df) * 0.85)
    test_prices = df.iloc[test_start_idx : test_start_idx + len(test_y)]

    if len(test_prices) < len(predictions["direction"]):
        test_prices = df.iloc[-len(predictions["direction"]) :]

    # Backtest
    print("\n7. Backtesting strategy...")
    result = backtest_multi_scale_strategy(
        predictions,
        test_prices,
        initial_capital=100000,
        transaction_cost=0.001,
        confidence_threshold=0.55,
    )

    print(generate_performance_report(result))

    # Compare strategies
    print("\n8. Comparing scale strategies...")
    comparison = compare_scale_strategies(predictions, test_prices)
    print(comparison)

    # Scale importance
    print("\n9. Learned scale importance:")
    importance = model.get_scale_importance()
    for scale, weight in importance.items():
        print(f"   {scale}: {weight:.4f}")

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return model, history, predictions, result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Scale Attention for Financial Time Series"
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "bybit"],
        default="synthetic",
        help="Run mode: synthetic data or real Bybit data",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading pair for Bybit mode",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of data to fetch for Bybit mode",
    )

    args = parser.parse_args()

    if args.mode == "synthetic":
        run_synthetic_example()
    else:
        run_bybit_example(symbol=args.symbol, days=args.days)


if __name__ == "__main__":
    main()
