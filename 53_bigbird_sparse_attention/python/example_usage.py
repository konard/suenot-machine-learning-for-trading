#!/usr/bin/env python3
"""
BigBird Trading Model - Complete Example

This script demonstrates:
1. Data fetching from Bybit (crypto) and Yahoo Finance (stocks)
2. Feature engineering
3. Model training
4. Backtesting and evaluation
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# Import local modules
from model import BigBirdConfig, BigBirdForTrading, OutputType
from data import (
    fetch_bybit_data,
    fetch_stock_data,
    prepare_features,
    create_sequences,
    prepare_data_loaders,
    generate_synthetic_data
)
from strategy import (
    backtest_strategy,
    BacktestConfig,
    print_metrics,
    visualize_results
)


def train_model(
    model: BigBirdForTrading,
    train_loader,
    val_loader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    device: torch.device = None,
    save_path: str = 'best_model.pt'
):
    """
    Train the BigBird model.

    Args:
        model: BigBird model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device for training
        save_path: Path to save best model

    Returns:
        Trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output['predictions'].squeeze(), batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                val_loss += criterion(output['predictions'].squeeze(), batch_y).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Load best model
    model.load_state_dict(torch.load(save_path))
    print(f"\nBest model saved to {save_path}")

    return model


def run_crypto_example(
    symbols: list = ['BTCUSDT', 'ETHUSDT'],
    timeframe: str = '1h',
    limit: int = 2000,
    seq_len: int = 128,
    epochs: int = 30
):
    """
    Run complete example with Bybit cryptocurrency data.
    """
    print("\n" + "=" * 60)
    print(" BigBird Trading Model - Cryptocurrency Example")
    print("=" * 60)

    # Fetch and prepare data
    print(f"\n1. Fetching data from Bybit for {symbols}...")
    all_X, all_y = [], []

    for symbol in symbols:
        try:
            df = fetch_bybit_data(symbol, timeframe, limit=limit)
            print(f"   {symbol}: {len(df)} rows fetched")

            df = prepare_features(df)
            print(f"   {symbol}: {len(df)} rows after feature prep")

            X, y = create_sequences(df, seq_len=seq_len)
            all_X.append(X)
            all_y.append(y)
            print(f"   {symbol}: {len(X)} sequences created")
        except Exception as e:
            print(f"   {symbol}: Error - {e}")
            continue

    if not all_X:
        print("No data available. Running synthetic example instead.")
        return run_synthetic_example(seq_len=seq_len, epochs=epochs)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\nTotal data: X shape {X.shape}, y shape {y.shape}")

    # Prepare data loaders
    print("\n2. Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        X, y, batch_size=32
    )

    # Create model
    print("\n3. Creating BigBird model...")
    config = BigBirdConfig(
        seq_len=seq_len,
        input_dim=X.shape[-1],
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        window_size=7,
        num_random=3,
        num_global=2,
        output_type=OutputType.REGRESSION
    )
    model = BigBirdForTrading(config)

    # Train model
    print("\n4. Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(
        model, train_loader, val_loader,
        epochs=epochs, device=device
    )

    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            test_preds.extend(output['predictions'].cpu().numpy().flatten())
            test_targets.extend(batch_y.numpy().flatten())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    # Calculate prediction metrics
    mse = np.mean((test_preds - test_targets) ** 2)
    mae = np.mean(np.abs(test_preds - test_targets))
    correlation = np.corrcoef(test_preds, test_targets)[0, 1]
    directional_accuracy = np.mean(np.sign(test_preds) == np.sign(test_targets))

    print(f"\n   Test MSE: {mse:.6f}")
    print(f"   Test MAE: {mae:.6f}")
    print(f"   Correlation: {correlation:.4f}")
    print(f"   Directional Accuracy: {directional_accuracy*100:.2f}%")

    print("\n" + "=" * 60)
    print(" Cryptocurrency Example Complete!")
    print("=" * 60)


def run_stock_example(
    symbol: str = 'SPY',
    period: str = '1y',
    interval: str = '1h',
    seq_len: int = 128,
    epochs: int = 30
):
    """
    Run complete example with stock market data.
    """
    print("\n" + "=" * 60)
    print(f" BigBird Trading Model - Stock Example ({symbol})")
    print("=" * 60)

    # Fetch and prepare data
    print(f"\n1. Fetching data from Yahoo Finance for {symbol}...")
    try:
        df = fetch_stock_data(symbol, period=period, interval=interval)
        print(f"   Fetched {len(df)} rows")
    except Exception as e:
        print(f"   Error fetching data: {e}")
        print("   Running synthetic example instead.")
        return run_synthetic_example(seq_len=seq_len, epochs=epochs)

    df = prepare_features(df)
    print(f"   After feature prep: {len(df)} rows")

    # Create sequences
    X, y = create_sequences(df, seq_len=seq_len)
    print(f"   Sequences: X shape {X.shape}, y shape {y.shape}")

    # Prepare data loaders
    print("\n2. Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        X, y, batch_size=32
    )

    # Create model
    print("\n3. Creating BigBird model...")
    config = BigBirdConfig(
        seq_len=seq_len,
        input_dim=X.shape[-1],
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        window_size=7,
        num_random=3,
        num_global=2
    )
    model = BigBirdForTrading(config)

    # Train model
    print("\n4. Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(
        model, train_loader, val_loader,
        epochs=epochs, device=device
    )

    print("\n" + "=" * 60)
    print(" Stock Example Complete!")
    print("=" * 60)


def run_synthetic_example(
    n_samples: int = 5000,
    seq_len: int = 128,
    n_features: int = 6,
    epochs: int = 30
):
    """
    Run example with synthetic data (no external dependencies).
    """
    print("\n" + "=" * 60)
    print(" BigBird Trading Model - Synthetic Data Example")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=n_samples, seq_len=seq_len, n_features=n_features)
    print(f"   Data shape: X {X.shape}, y {y.shape}")

    # Prepare data loaders
    print("\n2. Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        X, y, batch_size=32
    )
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Val: {len(val_loader)} batches")
    print(f"   Test: {len(test_loader)} batches")

    # Create model
    print("\n3. Creating BigBird model...")
    config = BigBirdConfig(
        seq_len=seq_len,
        input_dim=n_features,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        window_size=7,
        num_random=3,
        num_global=2,
        dropout=0.1
    )
    model = BigBirdForTrading(config)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("\n4. Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(
        model, train_loader, val_loader,
        epochs=epochs, device=device,
        save_path='synthetic_model.pt'
    )

    # Evaluate
    print("\n5. Evaluating model...")
    model.eval()
    test_preds, test_targets = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            test_preds.extend(output['predictions'].cpu().numpy().flatten())
            test_targets.extend(batch_y.numpy().flatten())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    mse = np.mean((test_preds - test_targets) ** 2)
    mae = np.mean(np.abs(test_preds - test_targets))
    correlation = np.corrcoef(test_preds, test_targets)[0, 1]

    print(f"\n   Test MSE: {mse:.6f}")
    print(f"   Test MAE: {mae:.6f}")
    print(f"   Correlation: {correlation:.4f}")

    # Analyze attention patterns
    print("\n6. Analyzing attention patterns...")
    sample_x = torch.FloatTensor(X[:1]).to(device)
    with torch.no_grad():
        output = model(sample_x, return_attention=True)

    if 'attention_weights' in output and output['attention_weights']:
        for layer_name, attn in output['attention_weights'].items():
            sparsity = (attn == 0).float().mean().item()
            print(f"   {layer_name}: sparsity = {sparsity*100:.1f}%")

    print("\n" + "=" * 60)
    print(" Synthetic Example Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='BigBird Trading Model Examples')
    parser.add_argument('--example', type=str, default='synthetic',
                       choices=['synthetic', 'crypto', 'stock'],
                       help='Example to run (default: synthetic)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--seq-len', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--symbol', type=str, default='SPY',
                       help='Stock symbol for stock example')

    args = parser.parse_args()

    if args.example == 'crypto':
        run_crypto_example(
            symbols=['BTCUSDT', 'ETHUSDT'],
            seq_len=args.seq_len,
            epochs=args.epochs
        )
    elif args.example == 'stock':
        run_stock_example(
            symbol=args.symbol,
            seq_len=args.seq_len,
            epochs=args.epochs
        )
    else:
        run_synthetic_example(
            seq_len=args.seq_len,
            epochs=args.epochs
        )


if __name__ == "__main__":
    main()
