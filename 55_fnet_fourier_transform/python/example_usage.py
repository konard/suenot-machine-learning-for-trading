"""
Complete Example: FNet for Cryptocurrency Trading

This script demonstrates the full workflow:
1. Loading data from Bybit
2. Feature engineering
3. Training FNet model
4. Generating trading signals
5. Backtesting the strategy

Usage:
    python example_usage.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import warnings

from model import FNet
from data import BybitDataLoader, calculate_features, create_sequences, normalize_features
from strategy import FNetTradingStrategy, Backtester, print_backtest_results

warnings.filterwarnings('ignore')


def main():
    """Main training and backtesting pipeline."""

    print("=" * 60)
    print("FNet for Cryptocurrency Trading")
    print("=" * 60)

    # Configuration
    config = {
        'symbols': ['BTCUSDT'],
        'interval': '60',  # 1 hour
        'limit': 2000,
        'seq_len': 168,  # 7 days
        'horizon': 24,   # 1 day ahead
        'd_model': 256,
        'n_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Feature columns
    feature_cols = [
        "log_return", "volatility", "volume_ratio",
        "momentum_5", "momentum_10", "momentum_20",
        "rsi_normalized", "bb_position"
    ]

    print(f"\n1. Loading Data...")

    # Load data
    loader = BybitDataLoader()
    all_dfs = []

    for symbol in config['symbols']:
        print(f"   Fetching {symbol}...")
        df = loader.fetch_klines(symbol, config['interval'], config['limit'])

        if df.empty:
            print(f"   Warning: Could not load data for {symbol}")
            continue

        # Calculate features
        df = calculate_features(df)
        df = df.dropna()

        print(f"   Loaded {len(df)} candles with features")
        all_dfs.append(df)

    if not all_dfs:
        print("\nError: No data loaded. Using synthetic data for demonstration.")
        # Generate synthetic data for testing
        n_samples = 2000
        np.random.seed(42)

        dates = np.arange(n_samples)
        close = 50000 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.01))

        df = {
            'timestamp': dates,
            'open': close * (1 + np.random.randn(n_samples) * 0.001),
            'high': close * (1 + np.abs(np.random.randn(n_samples)) * 0.01),
            'low': close * (1 - np.abs(np.random.randn(n_samples)) * 0.01),
            'close': close,
            'volume': np.random.exponential(1000, n_samples)
        }
        df = calculate_features(type('DataFrame', (), df)())

        # Create synthetic dataframe
        import pandas as pd
        df = pd.DataFrame({
            'log_return': np.random.randn(n_samples) * 0.01,
            'volatility': np.random.exponential(0.02, n_samples),
            'volume_ratio': np.random.exponential(1, n_samples),
            'momentum_5': np.random.randn(n_samples) * 0.05,
            'momentum_10': np.random.randn(n_samples) * 0.08,
            'momentum_20': np.random.randn(n_samples) * 0.1,
            'rsi_normalized': np.random.randn(n_samples) * 0.3,
            'bb_position': np.random.randn(n_samples) * 0.5,
            'close': close
        })
        all_dfs = [df]

    # Combine all data
    combined_df = all_dfs[0] if len(all_dfs) == 1 else all_dfs[0]

    print(f"\n2. Creating Sequences...")

    # Create sequences
    X, y = create_sequences(
        combined_df,
        feature_cols,
        'log_return',
        seq_len=config['seq_len'],
        horizon=config['horizon']
    )

    print(f"   Total sequences: {len(X)}")
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {y.shape}")

    # Split data (70% train, 15% val, 15% test)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Normalize using training statistics
    X_train, mean, std = normalize_features(X_train)
    X_val, _, _ = normalize_features(X_val, mean, std)
    X_test, _, _ = normalize_features(X_test, mean, std)

    print(f"\n   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    print(f"\n3. Building Model...")

    # Initialize model
    model = FNet(
        n_features=len(feature_cols),
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=config['seq_len'],
        output_dim=1
    ).to(config['device'])

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {n_params:,}")

    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    print(f"\n4. Training Model...")

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(config['device'])
            batch_y = batch_y.to(config['device'])

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(config['device'])
                batch_y = batch_y.to(config['device'])

                predictions = model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_fnet_model.pt')
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{config['epochs']}: "
                  f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if patience_counter >= max_patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load('best_fnet_model.pt', weights_only=True))
    print(f"\n   Best validation loss: {best_val_loss:.6f}")

    print(f"\n5. Frequency Analysis...")

    # Analyze frequency patterns
    model.eval()
    sample_X = torch.FloatTensor(X_test[:10]).to(config['device'])
    analysis = model.get_frequency_analysis(sample_X)

    print("   Layer 1 frequency analysis:")
    print(f"     Mean magnitude: {analysis['layer_1']['mean_magnitude']:.4f}")
    print(f"     Max magnitude: {analysis['layer_1']['max_magnitude']:.4f}")
    print(f"     Top frequency indices: {analysis['layer_1']['top_frequencies'][:5]}")

    print(f"\n6. Backtesting Strategy...")

    # Get prices for test period
    if 'close' in combined_df.columns:
        prices_test = combined_df['close'].values[
            config['seq_len'] + config['horizon'] - 1 + val_end:
        ][:len(X_test)]
    else:
        # Use synthetic prices
        prices_test = 50000 * np.exp(np.cumsum(y_test))

    # Ensure prices array matches test data length
    prices_test = prices_test[:len(X_test)]

    if len(prices_test) < len(X_test):
        X_test = X_test[:len(prices_test)]
        y_test = y_test[:len(prices_test)]

    # Create strategy
    model.to('cpu')  # Move to CPU for backtesting
    strategy = FNetTradingStrategy(
        model=model,
        threshold=0.001,
        confidence_threshold=0.4,
        position_size=1.0,
        stop_loss=0.02,
        take_profit=0.04,
        max_holding_period=24
    )

    # Run backtest
    backtester = Backtester(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )

    result = backtester.run(strategy, X_test, prices_test)

    # Print results
    print_backtest_results(result)

    # Compare with buy-and-hold
    buy_hold_return = (prices_test[-1] / prices_test[0]) - 1
    strategy_return = result.metrics['total_return']

    print(f"\nStrategy Comparison:")
    print(f"  FNet Strategy:  {strategy_return*100:>8.2f}%")
    print(f"  Buy & Hold:     {buy_hold_return*100:>8.2f}%")
    print(f"  Outperformance: {(strategy_return - buy_hold_return)*100:>8.2f}%")

    # Save results
    print(f"\n7. Saving Results...")

    try:
        from strategy import plot_results
        plot_results(result, save_path='backtest_results.png')
    except Exception as e:
        print(f"   Could not save plot: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
