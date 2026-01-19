"""
Example Usage of Deep Convolutional Transformer (DCT)

This script demonstrates:
1. Loading data from Bybit (crypto) or Yahoo Finance (stocks)
2. Computing technical indicators
3. Training the DCT model
4. Running backtests and evaluating performance
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

from model import DCTModel, DCTConfig
from data import (
    load_crypto_data,
    load_stock_data,
    compute_technical_indicators,
    prepare_dataset,
    DataConfig
)
from strategy import (
    StrategyConfig,
    backtest_dct_strategy,
    print_backtest_report
)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10,
    device: str = 'cpu'
) -> nn.Module:
    """
    Train DCT model with early stopping.

    Args:
        model: DCT model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        device: Device to train on

    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs['logits'], y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            train_correct += predicted.eq(y_batch).sum().item()
            train_total += y_batch.size(0)

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs['logits'], y_batch)
                val_loss += loss.item()

                _, predicted = outputs['logits'].max(1)
                val_correct += predicted.eq(y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def example_crypto_trading():
    """Example: Trading crypto with DCT on Bybit data."""
    print("="*60)
    print("DCT Crypto Trading Example (Bybit)")
    print("="*60)

    # Load data
    print("\n1. Loading crypto data from Bybit...")
    df = load_crypto_data("BTCUSDT", interval="D", days=365)

    if df.empty:
        print("Failed to load data. Using synthetic data for demonstration.")
        # Create synthetic data
        dates = pd.date_range('2023-01-01', periods=365)
        np.random.seed(42)
        prices = 30000 * np.exp(np.cumsum(np.random.randn(365) * 0.02))
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(365) * 0.01),
            'high': prices * (1 + np.abs(np.random.randn(365) * 0.02)),
            'low': prices * (1 - np.abs(np.random.randn(365) * 0.02)),
            'close': prices,
            'volume': np.random.uniform(1e9, 5e9, 365)
        }, index=dates)

    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Compute technical indicators
    print("\n2. Computing technical indicators...")
    df_features = compute_technical_indicators(df)
    print(f"Features computed: {len(df_features.columns)} columns")

    # Prepare dataset
    print("\n3. Preparing dataset...")
    data_config = DataConfig(
        lookback=30,
        horizon=1,
        movement_threshold=0.005
    )
    dataset = prepare_dataset(df_features, data_config)

    print(f"Train: {dataset['X_train'].shape}, Val: {dataset['X_val'].shape}, Test: {dataset['X_test'].shape}")
    print(f"Label distribution (train): {np.bincount(dataset['y_train'], minlength=3)}")

    # Create data loaders
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n4. Training on device: {device}")

    train_dataset = TensorDataset(
        torch.FloatTensor(dataset['X_train']),
        torch.LongTensor(dataset['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(dataset['X_val']),
        torch.LongTensor(dataset['y_val'])
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create model
    model_config = DCTConfig(
        seq_len=30,
        input_features=dataset['X_train'].shape[2],
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        dropout=0.2
    )
    model = DCTModel(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("\n5. Training model...")
    model = train_model(model, train_loader, val_loader, epochs=30, device=device)

    # Backtest
    print("\n6. Running backtest...")

    # Get test prices (need to align with test data)
    n_test = len(dataset['X_test'])
    test_start_idx = len(df_features) - n_test - data_config.horizon
    test_prices = df_features['close'].iloc[test_start_idx:test_start_idx + n_test]

    strategy_config = StrategyConfig(
        initial_capital=100000,
        position_size=0.1,
        confidence_threshold=0.6,
        stop_loss=0.03,
        take_profit=0.06
    )

    result = backtest_dct_strategy(
        model,
        dataset['X_test'],
        test_prices.reset_index(drop=True),
        strategy_config
    )

    print_backtest_report(result)

    return model, result


def example_stock_trading():
    """Example: Trading stocks with DCT on Yahoo Finance data."""
    print("="*60)
    print("DCT Stock Trading Example (Yahoo Finance)")
    print("="*60)

    # Load data
    print("\n1. Loading stock data from Yahoo Finance...")
    try:
        df = load_stock_data("AAPL", start_date="2020-01-01")
    except Exception as e:
        print(f"Failed to load Yahoo Finance data: {e}")
        print("Using synthetic data for demonstration.")
        # Create synthetic data
        dates = pd.date_range('2020-01-01', periods=1000)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(1000) * 0.015))
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(1000) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(1000) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(1000) * 0.01)),
            'close': prices,
            'volume': np.random.uniform(1e7, 5e7, 1000)
        }, index=dates)

    print(f"Loaded {len(df)} days of data")

    # Compute indicators
    print("\n2. Computing technical indicators...")
    df_features = compute_technical_indicators(df)

    # Prepare dataset
    print("\n3. Preparing dataset...")
    data_config = DataConfig(lookback=30, horizon=1, movement_threshold=0.005)
    dataset = prepare_dataset(df_features, data_config)

    print(f"Train: {dataset['X_train'].shape}")
    print(f"Label distribution: {np.bincount(dataset['y_train'], minlength=3)}")

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = DCTConfig(
        seq_len=30,
        input_features=dataset['X_train'].shape[2],
        d_model=64,
        num_heads=4,
        num_encoder_layers=2
    )
    model = DCTModel(model_config)

    # Quick training
    print("\n4. Training model (quick demo)...")
    train_dataset = TensorDataset(
        torch.FloatTensor(dataset['X_train']),
        torch.LongTensor(dataset['y_train'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(dataset['X_val']),
        torch.LongTensor(dataset['y_val'])
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = train_model(model, train_loader, val_loader, epochs=20, device=device)

    # Backtest
    print("\n5. Running backtest...")
    n_test = len(dataset['X_test'])
    test_start_idx = len(df_features) - n_test - data_config.horizon
    test_prices = df_features['close'].iloc[test_start_idx:test_start_idx + n_test]

    result = backtest_dct_strategy(
        model,
        dataset['X_test'],
        test_prices.reset_index(drop=True)
    )

    print_backtest_report(result)

    return model, result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DCT Trading Example")
    parser.add_argument(
        "--mode",
        choices=["crypto", "stock", "both"],
        default="crypto",
        help="Which example to run"
    )

    args = parser.parse_args()

    if args.mode in ["crypto", "both"]:
        example_crypto_trading()

    if args.mode in ["stock", "both"]:
        example_stock_trading()

    print("\nDone!")
