#!/usr/bin/env python3
"""
Main entry point for Variational Inference Trading

This script provides commands for:
- Fetching market data from Bybit
- Training VAE model
- Generating predictions
- Running backtests
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from config import (
    Config, DataConfig, ModelConfig, TrainingConfig,
    StrategyConfig, BacktestConfig, get_default_config
)
from data_fetcher import BybitDataFetcher, fetch_and_prepare_data
from vae_model import create_model, MarketVAE
from trainer import train_model, Trainer
from strategy import VariationalTradingStrategy, TradingSignal
from backtest import run_backtest, BacktestResult


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging"""
    logger.remove()
    logger.add(sys.stderr, level=level)
    if log_file:
        logger.add(log_file, level=level)


def cmd_fetch_data(args):
    """Fetch market data from Bybit"""
    logger.info("Fetching market data from Bybit...")

    config = DataConfig(
        symbols=args.symbols.split(",") if args.symbols else None,
        timeframe=args.timeframe,
        limit=args.limit
    )

    fetcher = BybitDataFetcher(config)

    # Fetch data
    data = fetcher.fetch_all_symbols()

    # Save raw data
    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for symbol, df in data.items():
        safe_symbol = symbol.replace("/", "_")
        filepath = data_dir / f"{safe_symbol}_ohlcv.csv"
        df.to_csv(filepath)
        logger.info(f"Saved {symbol} data to {filepath}")

    logger.info(f"Fetched data for {len(data)} symbols")


def cmd_prepare_data(args):
    """Prepare data for training"""
    logger.info("Preparing data for training...")

    config = DataConfig(
        symbols=args.symbols.split(",") if args.symbols else None,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon
    )

    # Fetch and prepare
    prepared_data = fetch_and_prepare_data(config)

    # Save prepared data
    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for symbol, data in prepared_data.items():
        safe_symbol = symbol.replace("/", "_")

        # Save arrays
        np.save(data_dir / f"{safe_symbol}_X_train.npy", data['X_train'])
        np.save(data_dir / f"{safe_symbol}_X_val.npy", data['X_val'])
        np.save(data_dir / f"{safe_symbol}_X_test.npy", data['X_test'])
        np.save(data_dir / f"{safe_symbol}_y_train.npy", data['y_train'])
        np.save(data_dir / f"{safe_symbol}_y_val.npy", data['y_val'])
        np.save(data_dir / f"{safe_symbol}_y_test.npy", data['y_test'])

        logger.info(f"Saved prepared data for {symbol}")

    logger.info("Data preparation complete!")


def cmd_train(args):
    """Train VAE model"""
    logger.info("Starting training...")

    # Load data
    data_dir = Path(args.data_dir)
    symbol = args.symbol.replace("/", "_")

    X_train = np.load(data_dir / f"{symbol}_X_train.npy")
    X_val = np.load(data_dir / f"{symbol}_X_val.npy")
    y_train = np.load(data_dir / f"{symbol}_y_train.npy")
    y_val = np.load(data_dir / f"{symbol}_y_val.npy")

    logger.info(f"Loaded data: train={X_train.shape}, val={X_val.shape}")

    # Configure model
    model_config = ModelConfig(
        input_dim=X_train.shape[2],
        latent_dim=args.latent_dim,
        hidden_dims=[int(x) for x in args.hidden_dims.split(",")],
        beta=args.beta,
        dropout=args.dropout
    )

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        early_stopping_patience=args.patience,
        beta_warmup_epochs=args.beta_warmup
    )

    # Train
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        model_config, training_config,
        model_dir=args.model_dir
    )

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")


def cmd_predict(args):
    """Generate predictions using trained model"""
    import torch

    logger.info("Generating predictions...")

    # Load model
    model_dir = Path(args.model_dir)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu")

    model_config = ModelConfig(**checkpoint['config']['model_config'])
    model = create_model(model_config, sequence_length=args.sequence_length, device="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create strategy
    strategy_config = StrategyConfig(
        confidence_threshold=args.confidence_threshold,
        n_samples=args.n_samples
    )
    strategy = VariationalTradingStrategy(model, strategy_config, device="cpu")

    # Load test data
    data_dir = Path(args.data_dir)
    symbol = args.symbol.replace("/", "_")
    X_test = np.load(data_dir / f"{symbol}_X_test.npy")

    logger.info(f"Generating predictions for {len(X_test)} samples...")

    # Generate signals
    signals = strategy.batch_generate_signals(X_test)

    # Print summary
    signal_counts = {}
    for s in signals:
        signal_counts[s.signal_type.value] = signal_counts.get(s.signal_type.value, 0) + 1

    logger.info(f"Signal distribution: {signal_counts}")

    # Print sample predictions
    logger.info("\nSample predictions:")
    for i, signal in enumerate(signals[:5]):
        logger.info(f"  {i}: {signal}")


def cmd_backtest(args):
    """Run backtest on historical data"""
    import torch

    logger.info("Running backtest...")

    # Load model
    model_dir = Path(args.model_dir)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu")

    model_config = ModelConfig(**checkpoint['config']['model_config'])
    model = create_model(model_config, sequence_length=args.sequence_length, device="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load data
    data_dir = Path(args.data_dir)
    symbol = args.symbol.replace("/", "_")

    X_test = np.load(data_dir / f"{symbol}_X_test.npy")
    y_test = np.load(data_dir / f"{symbol}_y_test.npy")

    # Generate synthetic prices for demo (in real use, load actual prices)
    n_samples = len(X_test)
    returns = y_test
    prices = pd.Series(1000 * np.exp(np.cumsum(returns)))
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')

    # Configure
    strategy_config = StrategyConfig(
        confidence_threshold=args.confidence_threshold,
        n_samples=args.n_samples
    )
    backtest_config = BacktestConfig(
        initial_capital=args.initial_capital,
        commission=args.commission
    )

    # Run backtest
    result = run_backtest(
        model=model,
        features=X_test,
        prices=prices,
        timestamps=timestamps,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        symbol=args.symbol
    )

    # Print results
    print(result.summary())

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save equity curve
        result.equity_curve.to_csv(output_dir / "equity_curve.csv")
        result.returns.to_csv(output_dir / "returns.csv")

        logger.info(f"Saved results to {output_dir}")


def cmd_demo(args):
    """Run demo with synthetic data"""
    logger.info("Running demo with synthetic data...")

    # Generate synthetic data
    n_samples = 1000
    seq_len = 60
    n_features = 13

    np.random.seed(42)
    X_train = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y_train = np.random.randn(n_samples).astype(np.float32) * 0.01
    X_val = np.random.randn(200, seq_len, n_features).astype(np.float32)
    y_val = np.random.randn(200).astype(np.float32) * 0.01
    X_test = np.random.randn(500, seq_len, n_features).astype(np.float32)
    y_test = np.random.randn(500).astype(np.float32) * 0.01

    # Configure
    model_config = ModelConfig(
        input_dim=n_features,
        latent_dim=8,
        hidden_dims=[64, 32],
        beta=2.0
    )
    training_config = TrainingConfig(
        batch_size=64,
        max_epochs=20,
        early_stopping_patience=5,
        beta_warmup_epochs=5
    )

    logger.info("Training model...")

    # Train
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        model_config, training_config,
        model_dir="./demo_models"
    )

    logger.info(f"Training complete. Final val loss: {history['val_loss'][-1]:.4f}")

    # Generate predictions
    strategy = VariationalTradingStrategy(model, device="cpu")
    signals = strategy.batch_generate_signals(X_test[:10])

    logger.info("\nSample predictions:")
    for i, signal in enumerate(signals):
        logger.info(f"  {i}: {signal}")

    # Run simple backtest
    prices = pd.Series(1000 * np.exp(np.cumsum(y_test)))
    timestamps = pd.date_range(start='2024-01-01', periods=len(X_test), freq='1min')

    result = run_backtest(
        model=model,
        features=X_test,
        prices=prices,
        timestamps=timestamps,
        symbol="DEMO/USDT"
    )

    print(result.summary())

    logger.info("Demo complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Variational Inference Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data from Bybit
  python main.py fetch-data --symbols BTC/USDT,ETH/USDT --limit 1000

  # Prepare data for training
  python main.py prepare-data --symbols BTC/USDT

  # Train model
  python main.py train --symbol BTC/USDT --epochs 100

  # Generate predictions
  python main.py predict --symbol BTC/USDT

  # Run backtest
  python main.py backtest --symbol BTC/USDT

  # Run demo with synthetic data
  python main.py demo
"""
    )

    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--log-file", default=None, help="Log file path")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Fetch data
    fetch_parser = subparsers.add_parser("fetch-data", help="Fetch market data from Bybit")
    fetch_parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    fetch_parser.add_argument("--timeframe", default="1m")
    fetch_parser.add_argument("--limit", type=int, default=1000)
    fetch_parser.add_argument("--output-dir", default="./data")

    # Prepare data
    prepare_parser = subparsers.add_parser("prepare-data", help="Prepare data for training")
    prepare_parser.add_argument("--symbols", default="BTC/USDT")
    prepare_parser.add_argument("--sequence-length", type=int, default=60)
    prepare_parser.add_argument("--prediction-horizon", type=int, default=5)
    prepare_parser.add_argument("--output-dir", default="./data")

    # Train
    train_parser = subparsers.add_parser("train", help="Train VAE model")
    train_parser.add_argument("--symbol", default="BTC/USDT")
    train_parser.add_argument("--data-dir", default="./data")
    train_parser.add_argument("--model-dir", default="./models")
    train_parser.add_argument("--latent-dim", type=int, default=16)
    train_parser.add_argument("--hidden-dims", default="256,128,64")
    train_parser.add_argument("--beta", type=float, default=4.0)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--patience", type=int, default=20)
    train_parser.add_argument("--beta-warmup", type=int, default=50)

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument("--symbol", default="BTC/USDT")
    predict_parser.add_argument("--data-dir", default="./data")
    predict_parser.add_argument("--model-dir", default="./models")
    predict_parser.add_argument("--sequence-length", type=int, default=60)
    predict_parser.add_argument("--confidence-threshold", type=float, default=0.6)
    predict_parser.add_argument("--n-samples", type=int, default=100)

    # Backtest
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--symbol", default="BTC/USDT")
    backtest_parser.add_argument("--data-dir", default="./data")
    backtest_parser.add_argument("--model-dir", default="./models")
    backtest_parser.add_argument("--sequence-length", type=int, default=60)
    backtest_parser.add_argument("--confidence-threshold", type=float, default=0.6)
    backtest_parser.add_argument("--n-samples", type=int, default=100)
    backtest_parser.add_argument("--initial-capital", type=float, default=10000)
    backtest_parser.add_argument("--commission", type=float, default=0.001)
    backtest_parser.add_argument("--output", default=None, help="Output directory for results")

    # Demo
    demo_parser = subparsers.add_parser("demo", help="Run demo with synthetic data")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # Execute command
    if args.command == "fetch-data":
        cmd_fetch_data(args)
    elif args.command == "prepare-data":
        cmd_prepare_data(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
