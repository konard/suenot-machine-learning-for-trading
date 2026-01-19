"""
Example usage of Linformer for financial time series forecasting.

This script demonstrates:
1. Loading data from Bybit
2. Preparing long sequences
3. Training a Linformer model
4. Running backtests
5. Evaluating performance

Run with: python example_usage.py
"""

import torch
import numpy as np
import logging
from datetime import datetime

from model import Linformer, create_linformer
from data import (
    load_bybit_data,
    prepare_long_sequence_data,
    train_val_test_split,
    create_data_loaders
)
from strategy import (
    backtest_linformer_strategy,
    calculate_performance_metrics,
    plot_backtest_results
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    model: Linformer,
    train_loader,
    val_loader,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Linformer:
    """
    Train Linformer model.

    Args:
        model: Linformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        device: Computation device

    Returns:
        Trained model
    """
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_x)
            loss = model.compute_loss(predictions, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_x)
                loss = model.compute_loss(predictions, batch_y)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss={avg_train_loss:.6f}, "
            f"Val Loss={avg_val_loss:.6f}, "
            f"LR={scheduler.get_last_lr()[0]:.2e}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def main():
    """Main function demonstrating Linformer usage."""

    logger.info("="*60)
    logger.info("LINFORMER FINANCIAL TIME SERIES FORECASTING")
    logger.info("="*60)

    # Configuration
    config = {
        # Data parameters
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'lookback': 2048,  # Long sequence - Linformer's strength!
        'horizon': 24,     # Predict 24 hours ahead
        'interval': '1h',

        # Model parameters
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 4,
        'k': 128,          # Projection dimension
        'd_ff': 1024,
        'dropout': 0.1,
        'output_type': 'regression',

        # Training parameters
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 50,
        'patience': 10
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Step 1: Load and prepare data
    logger.info("\n" + "-"*40)
    logger.info("STEP 1: Loading and preparing data")
    logger.info("-"*40)

    try:
        data = prepare_long_sequence_data(
            symbols=config['symbols'],
            lookback=config['lookback'],
            horizon=config['horizon'],
            data_source='bybit',
            interval=config['interval']
        )
        logger.info(f"Loaded {len(data['X'])} samples")
        logger.info(f"Feature shape: {data['X'].shape}")
    except Exception as e:
        logger.warning(f"Could not load Bybit data: {e}")
        logger.info("Using synthetic data for demonstration...")

        # Generate synthetic data
        n_samples = 1000
        n_features = 20
        data = {
            'X': np.random.randn(n_samples, config['lookback'], n_features).astype(np.float32),
            'y': np.random.randn(n_samples, 1).astype(np.float32) * 0.02,
            'symbols': config['symbols'],
            'timestamps': np.array([datetime.now()] * n_samples)
        }

    # Step 2: Split data
    logger.info("\n" + "-"*40)
    logger.info("STEP 2: Splitting data")
    logger.info("-"*40)

    train_data, val_data, test_data = train_val_test_split(data)
    logger.info(f"Train: {len(train_data['X'])} samples")
    logger.info(f"Val: {len(val_data['X'])} samples")
    logger.info(f"Test: {len(test_data['X'])} samples")

    # Step 3: Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, val_data,
        batch_size=config['batch_size']
    )

    # Step 4: Initialize model
    logger.info("\n" + "-"*40)
    logger.info("STEP 3: Initializing Linformer model")
    logger.info("-"*40)

    n_features = data['X'].shape[-1]
    model = Linformer(
        n_features=n_features,
        seq_len=config['lookback'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        k=config['k'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        output_type=config['output_type'],
        n_outputs=1
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Print complexity info
    complexity_info = model.get_complexity_info()
    logger.info(f"Sequence length: {complexity_info['seq_len']}")
    logger.info(f"Projection dimension: {complexity_info['projection_dim']}")
    logger.info(f"Speedup factor: {complexity_info['speedup_factor']:.1f}x")
    logger.info(f"Memory reduction: {complexity_info['memory_reduction']:.1f}%")

    # Step 5: Train model
    logger.info("\n" + "-"*40)
    logger.info("STEP 4: Training model")
    logger.info("-"*40)

    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        device=device
    )

    # Save model
    torch.save(model.state_dict(), 'linformer_model.pt')
    logger.info("Model saved to linformer_model.pt")

    # Step 6: Backtest
    logger.info("\n" + "-"*40)
    logger.info("STEP 5: Running backtest")
    logger.info("-"*40)

    results = backtest_linformer_strategy(
        model,
        test_data,
        initial_capital=100000,
        transaction_cost=0.001,
        signal_threshold=0.001,
        device=device
    )

    # Save results
    results.to_csv('backtest_results.csv', index=False)
    logger.info("Results saved to backtest_results.csv")

    # Step 7: Plot results
    logger.info("\n" + "-"*40)
    logger.info("STEP 6: Plotting results")
    logger.info("-"*40)

    try:
        plot_backtest_results(results, title="Linformer Strategy Backtest")
        logger.info("Plot saved to backtest_results.png")
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")

    logger.info("\n" + "="*60)
    logger.info("DONE!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
