"""
Training script for MambaTS model.

Usage:
    python train.py --data bybit --symbol BTCUSDT --epochs 100
    python train.py --data yahoo --symbol AAPL --epochs 50
"""

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import torch

from model import MambaTS, MambaTSTrainer
from data_loader import FinancialDataLoader
from backtest import Backtester, SimpleThresholdStrategy


def parse_args():
    parser = argparse.ArgumentParser(description="Train MambaTS model")

    # Data arguments
    parser.add_argument(
        "--data", type=str, default="bybit", choices=["bybit", "yahoo"],
        help="Data source"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT",
        help="Trading symbol"
    )
    parser.add_argument(
        "--interval", type=str, default="1h",
        help="Data interval"
    )
    parser.add_argument(
        "--lookback", type=int, default=168,
        help="Lookback window (default: 168 = 1 week of hourly data)"
    )
    parser.add_argument(
        "--forecast-horizon", type=int, default=24,
        help="Forecast horizon (default: 24 = 1 day of hourly data)"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Days of historical data to use"
    )

    # Model arguments
    parser.add_argument(
        "--d-model", type=int, default=128,
        help="Model dimension"
    )
    parser.add_argument(
        "--n-layers", type=int, default=4,
        help="Number of Mamba layers"
    )
    parser.add_argument(
        "--patch-size", type=int, default=16,
        help="Temporal patch size"
    )
    parser.add_argument(
        "--task", type=str, default="regression",
        choices=["regression", "classification"],
        help="Task type"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, cuda, cpu)"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Output directory for saved models"
    )
    parser.add_argument(
        "--run-backtest", action="store_true",
        help="Run backtest after training"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=args.days)

    print(f"\nLoading data for {args.symbol} from {start_time} to {end_time}")

    # Create data loader
    loader = FinancialDataLoader(
        symbols=args.symbol,
        interval=args.interval,
        lookback=args.lookback,
        forecast_horizon=args.forecast_horizon,
        source=args.data,
        add_indicators=True,
        normalize=True,
    )

    # Fetch data
    try:
        df = loader.fetch_data(start_time, end_time)
        print(f"Loaded {len(df)} rows of data")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nGenerating synthetic data for demonstration...")

        # Generate synthetic data for demonstration
        n_samples = args.days * 24  # Hourly data
        np.random.seed(42)

        # Simulate realistic price movement
        returns = np.random.normal(0.0001, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))

        # Create synthetic features
        n_features = 12
        data = np.column_stack([
            prices,
            np.random.randn(n_samples, n_features - 1) * 0.1
        ])

        # Normalize
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

        # Create simple datasets
        from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

        n_train = int(n_samples * 0.8)
        n_val = int(n_samples * 0.1)

        def create_sequences(data, lookback, horizon):
            X, y = [], []
            for i in range(lookback, len(data) - horizon):
                X.append(data[i - lookback:i])
                y.append(data[i:i + horizon])
            return np.array(X), np.array(y)

        X, y = create_sequences(data, args.lookback, args.forecast_horizon)

        X_train = torch.tensor(X[:n_train], dtype=torch.float32)
        y_train = torch.tensor(y[:n_train], dtype=torch.float32)
        X_val = torch.tensor(X[n_train:n_train + n_val], dtype=torch.float32)
        y_val = torch.tensor(y[n_train:n_train + n_val], dtype=torch.float32)
        X_test = torch.tensor(X[n_train + n_val:], dtype=torch.float32)
        y_test = torch.tensor(y[n_train + n_val:], dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size)

        n_variables = n_features
        prices_array = prices[args.lookback + args.forecast_horizon:]
    else:
        # Prepare features and create dataloaders
        features = loader.prepare_features()
        n_variables = features.shape[1]
        print(f"Number of features: {n_variables}")

        train_loader, val_loader, test_loader = loader.get_dataloaders(
            batch_size=args.batch_size
        )
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Get prices for backtesting
        prices_array = df["close"].values[args.lookback + args.forecast_horizon:]

    # Create model
    print(f"\nCreating MambaTS model...")
    model = MambaTS(
        n_variables=n_variables,
        d_model=args.d_model,
        n_layers=args.n_layers,
        patch_size=args.patch_size,
        forecast_horizon=args.forecast_horizon,
        task=args.task,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create trainer
    trainer = MambaTSTrainer(
        model=model,
        lr=args.lr,
        device=device,
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    save_path = os.path.join(args.output_dir, f"mambats_{args.symbol.lower()}.pt")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        save_path=save_path,
    )

    print(f"\nTraining completed! Model saved to {save_path}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss = trainer.validate(test_loader)
    print(f"Test loss: {test_loss:.6f}")

    # Run backtest if requested
    if args.run_backtest and args.task == "regression":
        print("\nRunning backtest...")

        # Get predictions on test data
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                x, _ = batch
                x = x.to(device)
                pred = model(x)
                # Use first variable, first horizon prediction
                predictions.extend(pred[:, 0, 0].cpu().numpy())

        predictions = np.array(predictions)

        # Run backtest
        strategy = SimpleThresholdStrategy(
            predictions,
            long_threshold=0.01,
            short_threshold=-0.01,
        )

        backtester = Backtester(
            strategy=strategy,
            initial_capital=100000,
            commission=0.001,
        )

        # Use corresponding prices
        test_prices = prices_array[-len(predictions):]
        result = backtester.run(test_prices)

        print(result.summary())


if __name__ == "__main__":
    main()
