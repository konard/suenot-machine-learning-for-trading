#!/usr/bin/env python3
"""
Complete Example: Cross-Attention Multi-Asset Trading

This script demonstrates the full workflow:
1. Load data from Bybit
2. Prepare features
3. Train cross-attention model
4. Run backtest
5. Analyze results

Usage:
    python 01_complete_example.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import CrossAttentionMultiAsset, CrossAttentionConfig, OutputType
from data import (
    BybitDataLoader,
    prepare_cross_attention_data,
    train_val_test_split,
    DataConfig
)
from backtest import (
    CrossAttentionBacktest,
    BacktestConfig,
    print_metrics_report
)


def main():
    """Main function to run complete example."""
    print("=" * 60)
    print("Cross-Attention Multi-Asset Trading - Complete Example")
    print("=" * 60)

    # Configuration
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {DEVICE}")

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 1: Loading Data from Bybit")
    print("-" * 40)

    loader = BybitDataLoader()

    # Fetch recent data (last 1000 hourly candles per symbol)
    print(f"Fetching data for: {SYMBOLS}")
    asset_data = loader.fetch_multi_asset(SYMBOLS, interval='60', limit=1000)

    # Print data summary
    for symbol, df in asset_data.items():
        if not df.empty:
            print(f"  {symbol}: {len(df)} candles, "
                  f"{df['timestamp'].min().strftime('%Y-%m-%d')} to "
                  f"{df['timestamp'].max().strftime('%Y-%m-%d')}")

    # =========================================================================
    # Step 2: Prepare Features
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 2: Preparing Features")
    print("-" * 40)

    data_config = DataConfig(
        lookback=168,  # 7 days
        horizon=24,    # 1 day ahead
        train_ratio=0.7,
        val_ratio=0.15
    )

    feature_cols = ['log_return', 'volume_ratio', 'volatility_20',
                    'rsi_14', 'macd', 'momentum_5']

    X, y, symbols, timestamps = prepare_cross_attention_data(
        asset_data,
        config=data_config,
        feature_cols=feature_cols
    )

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {feature_cols}")

    # Split data
    splits = train_val_test_split(X, y, timestamps, data_config)
    print(f"Train: {len(splits['train']['X'])} samples")
    print(f"Val:   {len(splits['val']['X'])} samples")
    print(f"Test:  {len(splits['test']['X'])} samples")

    # =========================================================================
    # Step 3: Create Model
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 3: Creating Cross-Attention Model")
    print("-" * 40)

    config = CrossAttentionConfig(
        n_assets=len(symbols),
        n_features=len(feature_cols),
        seq_len=data_config.lookback,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        output_type=OutputType.PORTFOLIO
    )

    model = CrossAttentionMultiAsset(config)
    model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Output type: {config.output_type.value}")

    # =========================================================================
    # Step 4: Train Model
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 4: Training Model")
    print("-" * 40)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(splits['train']['X']),
            torch.FloatTensor(splits['train']['y'])
        ),
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(splits['val']['X']),
            torch.FloatTensor(splits['val']['y'])
        ),
        batch_size=32,
        shuffle=False
    )

    # Loss function: maximize portfolio returns
    def portfolio_loss(weights, returns):
        """Negative portfolio return as loss."""
        portfolio_return = torch.sum(weights * returns, dim=-1)
        return -torch.mean(portfolio_return)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    n_epochs = 50
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            output = model(batch_x)
            predictions = output['predictions']

            loss = portfolio_loss(predictions, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                output = model(batch_x)
                predictions = output['predictions']

                loss = portfolio_loss(predictions, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

    # =========================================================================
    # Step 5: Run Backtest
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 5: Running Backtest")
    print("-" * 40)

    backtest_config = BacktestConfig(
        initial_capital=100000,
        transaction_cost=0.001,
        rebalance_freq=24,
        max_position=0.5,
        allow_short=False
    )

    backtest = CrossAttentionBacktest(model, backtest_config)

    results = backtest.run(
        splits['test']['X'],
        splits['test']['y'],
        splits['test']['timestamps'],
        device=DEVICE
    )

    print(f"Backtest completed: {len(results)} timesteps")

    # =========================================================================
    # Step 6: Analyze Results
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 6: Analyzing Results")
    print("-" * 40)

    metrics = backtest.compute_metrics(results)
    comparison = backtest.compare_baseline(results, splits['test']['y'])

    print_metrics_report(metrics, comparison)

    # =========================================================================
    # Step 7: Visualize Attention
    # =========================================================================
    print("\n" + "-" * 40)
    print("Step 7: Attention Analysis")
    print("-" * 40)

    model.eval()

    # Get attention for a sample batch
    with torch.no_grad():
        sample_x = torch.FloatTensor(splits['test']['X'][:10]).to(DEVICE)
        output = model(sample_x, return_attention=True)

    attention = output.get('attention_weights')
    if attention:
        print("\nCross-Asset Attention Matrix (averaged):")
        print(f"{'':>10}", end='')
        for s in symbols:
            print(f"{s:>10}", end='')
        print()

        # Get last layer cross-asset attention
        last_layer = list(attention.keys())[-1]
        cross_attn = attention[last_layer].get('cross_asset')

        if cross_attn is not None:
            avg_attn = cross_attn.mean(dim=0).cpu().numpy()

            for i, s in enumerate(symbols):
                print(f"{s:>10}", end='')
                for j in range(len(symbols)):
                    if i == j:
                        print(f"{'---':>10}", end='')
                    else:
                        print(f"{avg_attn[i, j]:>10.3f}", end='')
                print()

    print("\n" + "=" * 60)
    print("Complete Example Finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()
