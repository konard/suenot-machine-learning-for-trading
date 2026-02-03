#!/usr/bin/env python3
"""
Layer-wise Relevance Propagation (LRP) Example Usage

This script demonstrates the complete workflow for:
1. Loading and preparing trading data
2. Training an LRP-enabled neural network
3. Generating explanations for predictions
4. Backtesting with LRP insights

Usage:
    python example_usage.py

Requirements:
    pip install -r requirements.txt
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LRPNetwork, LRPConfig, LRPClassifier, verify_conservation
from data import prepare_trading_data, TradingDataset, create_dataloaders
from strategy import (
    backtest_with_lrp,
    BacktestConfig,
    analyze_trading_signal,
    compute_feature_importance,
    visualize_backtest_results,
    generate_trading_report
)


def train_model(
    model: LRPNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cpu"
) -> dict:
    """
    Train the LRP model.

    Args:
        model: LRP-enabled network
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on

    Returns:
        Training history dictionary
    """
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                val_batches += 1

                pred = output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += len(batch_y)

        val_loss /= val_batches
        val_acc = correct / total

        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_acc:.4f}")

    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))

    return history


def main():
    """Main example workflow."""
    print("=" * 60)
    print("Layer-wise Relevance Propagation (LRP) Example")
    print("=" * 60)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # 1. Prepare Data
    print("\n1. Preparing trading data...")

    data = prepare_trading_data(
        symbols=["BTCUSDT"],  # Using sample data if API unavailable
        source="sample",      # Use "bybit" for real data
        lookback=60,
        horizon=1
    )

    print(f"   Total samples: {len(data['X'])}")
    print(f"   Features: {len(data['feature_names'])}")
    print(f"   Feature names: {data['feature_names'][:5]}...")
    print(f"   Train/Val/Test split: "
          f"{len(data['train_idx'])}/{len(data['val_idx'])}/{len(data['test_idx'])}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(data, batch_size=32)

    # 2. Create Model
    print("\n2. Creating LRP-enabled model...")

    input_dim = data["X"].shape[1] * data["X"].shape[2]  # lookback * features

    config = LRPConfig(
        epsilon=0.01,
        gamma=0.25,
    )

    model = LRPClassifier(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        config=config
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")
    print(f"   Architecture: {input_dim} -> [128, 64, 32] -> 2")

    # 3. Train Model
    print("\n3. Training model...")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=1e-3,
        device=device
    )

    print(f"   Final Val Accuracy: {history['val_acc'][-1]:.4f}")

    # 4. Verify LRP Conservation
    print("\n4. Verifying LRP conservation property...")

    model = model.to("cpu")
    sample_batch, _ = next(iter(test_loader))
    sample = sample_batch[:1]

    conserved, error = verify_conservation(model, sample)
    print(f"   Conservation holds: {conserved}")
    print(f"   Relative error: {error:.4f}")

    # 5. Analyze Single Prediction
    print("\n5. Analyzing single prediction...")

    analysis = analyze_trading_signal(
        model=model,
        sample=sample,
        feature_names=data["feature_names"],
        threshold=0.05
    )

    print(f"   Signal: {analysis['signal']}")
    print(f"   Confidence: {analysis['confidence']:.2%}")
    print("   Top explanations:")
    for exp in analysis["explanations"][:5]:
        print(f"     - {exp['feature']}: {exp['relevance_pct']:.1f}% ({exp['direction']})")

    # 6. Compute Feature Importance
    print("\n6. Computing overall feature importance...")

    importance = compute_feature_importance(
        model=model,
        data_loader=test_loader,
        feature_names=data["feature_names"]
    )

    print("   Top 5 features:")
    for feat, imp in importance[:5]:
        print(f"     - {feat}: {imp*100:.1f}%")

    # 7. Backtest Strategy
    print("\n7. Running backtest...")

    backtest_config = BacktestConfig(
        initial_capital=100000,
        transaction_cost=0.001,
        confidence_threshold=0.55,
        use_lrp_sizing=True
    )

    results = backtest_with_lrp(
        model=model,
        test_loader=test_loader,
        feature_names=data["feature_names"],
        config=backtest_config
    )

    print(f"   Total Return: {results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Sortino Ratio: {results['sortino_ratio']:.2f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"   Win Rate: {results['win_rate']:.2%}")
    print(f"   Avg Confidence: {results['avg_confidence']:.2%}")
    print(f"   Avg Coherence: {results['avg_coherence']:.2f}")

    # 8. Generate Report
    print("\n8. Generating trading report...")

    report = generate_trading_report(
        model=model,
        data_loader=test_loader,
        feature_names=data["feature_names"]
    )

    # Save report
    with open("trading_report.md", "w") as f:
        f.write(report)
    print("   Report saved to: trading_report.md")

    # 9. Visualize Results (optional)
    print("\n9. Visualization...")
    try:
        visualize_backtest_results(results, save_path="backtest_results.png")
        print("   Results saved to: backtest_results.png")
    except Exception as e:
        print(f"   Skipping visualization: {e}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
