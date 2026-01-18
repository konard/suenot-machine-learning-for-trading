#!/usr/bin/env python3
"""
Bybit Data Loading Example

This example demonstrates loading cryptocurrency data from Bybit exchange
and applying positional encodings for time series analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data import BybitDataLoader, prepare_features, create_sequences
from positional_encoding import (
    SinusoidalPositionalEncoding,
    CalendarEncoding,
    MarketSessionEncoding,
    MultiScaleTemporalEncoding,
)
from strategy import TradingStrategy, run_backtest, calculate_buy_and_hold


def main():
    print("Bybit Data Loading Example")
    print("=" * 50)

    # 1. Initialize data loader
    print("\n1. Loading Data from Bybit")
    print("-" * 40)

    loader = BybitDataLoader()
    symbol = "BTCUSDT"
    interval = "60"  # 1 hour
    limit = 500

    try:
        df = loader.fetch_klines(symbol, interval, limit)
        print(f"Loaded {len(df)} candles for {symbol}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        use_real_data = True
    except Exception as e:
        print(f"Could not fetch real data: {e}")
        print("Using synthetic data instead...")

        # Generate synthetic data
        np.random.seed(42)
        n = 500
        base_time = 1704067200  # 2024-01-01 00:00:00 UTC

        returns = np.random.normal(0.0001, 0.02, n)
        closes = 45000 * np.cumprod(1 + returns)

        import pandas as pd
        df = pd.DataFrame({
            'timestamp': [base_time + i * 3600 for i in range(n)],
            'open': np.roll(closes, 1),
            'high': closes * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': closes * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': closes,
            'volume': np.random.uniform(500, 1500, n),
        })
        df['open'].iloc[0] = 45000

        print(f"Generated {len(df)} synthetic candles")
        use_real_data = False

    # 2. Prepare Features
    print("\n2. Feature Preparation")
    print("-" * 40)

    features = prepare_features(
        df['open'].values,
        df['high'].values,
        df['low'].values,
        df['close'].values,
        df['volume'].values
    )
    timestamps = df['timestamp'].values
    returns = np.diff(df['close'].values) / df['close'].values[:-1]
    returns = np.insert(returns, 0, 0)  # Pad first return as 0

    print(f"Feature matrix shape: {features.shape}")
    print(f"Features: return, log_volume, range, body_ratio, direction, return_pct")

    # 3. Apply Positional Encodings
    print("\n3. Applying Positional Encodings")
    print("-" * 40)

    d_model = 32
    n_samples = len(df)

    # Position encoding
    pos_enc = SinusoidalPositionalEncoding(d_model, n_samples)
    x = torch.zeros(1, n_samples, d_model)
    pos_features = pos_enc(x).squeeze(0).numpy()
    print(f"Position encoding: {pos_features.shape}")

    # Calendar encoding
    cal_enc = CalendarEncoding(d_model)
    ts_tensor = torch.tensor(timestamps).unsqueeze(0)
    cal_features = cal_enc(ts_tensor).squeeze(0).numpy()
    print(f"Calendar encoding: {cal_features.shape}")

    # Market session encoding (crypto)
    session_enc = MarketSessionEncoding(d_model, market_type='crypto')
    session_features = session_enc(ts_tensor).squeeze(0).numpy()
    print(f"Session encoding: {session_features.shape}")

    # Multi-scale temporal encoding
    multi_enc = MultiScaleTemporalEncoding(d_model, market_type='crypto')
    multi_features = multi_enc(ts_tensor).squeeze(0).numpy()
    print(f"Multi-scale encoding: {multi_features.shape}")

    # Combine all features
    all_features = np.concatenate([
        features,
        pos_features,
        cal_features,
        session_features
    ], axis=1)
    print(f"\nCombined feature dimension: {all_features.shape[1]}")

    # 4. Create Sequences
    print("\n4. Creating Training Sequences")
    print("-" * 40)

    seq_length = 24  # 24-hour lookback
    target_length = 1  # 1-hour prediction

    sequences, targets, seq_timestamps = create_sequences(
        features, returns, timestamps, seq_length, target_length
    )

    print(f"Number of sequences: {len(sequences)}")
    print(f"Sequence shape: {sequences[0].shape}")
    print(f"Target shape: {targets[0].shape}")

    # 5. Trading Strategy Backtest
    print("\n5. Trading Strategy Backtest")
    print("-" * 40)

    # Simple momentum strategy (in practice, use trained model predictions)
    window = 24
    predictions = np.zeros(len(returns))
    for i in range(window, len(returns)):
        ma_short = returns[i-12:i].mean()
        ma_long = returns[i-24:i].mean()
        predictions[i] = ma_short - ma_long

    strategy = TradingStrategy(threshold=0.001, transaction_cost=0.001)
    result = run_backtest(predictions, returns, strategy, periods_per_year=8760)

    print("\nStrategy Performance:")
    print(f"  Total Return: {result.total_return * 100:.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown * 100:.2f}%")
    print(f"  Win Rate: {result.win_rate * 100:.2f}%")
    print(f"  Trades: {result.n_trades}")

    # Buy and hold benchmark
    bh_result = calculate_buy_and_hold(returns, periods_per_year=8760)
    print("\nBuy & Hold Benchmark:")
    print(f"  Total Return: {bh_result.total_return * 100:.2f}%")
    print(f"  Sharpe Ratio: {bh_result.sharpe_ratio:.2f}")

    # 6. Time Pattern Analysis
    print("\n6. Time Pattern Analysis")
    print("-" * 40)

    # Analyze returns by session
    from datetime import datetime

    asia_returns = []
    europe_returns = []
    americas_returns = []

    for i, ts in enumerate(timestamps):
        if i == 0:
            continue
        dt = datetime.utcfromtimestamp(ts)
        hour = dt.hour

        if hour < 8:
            asia_returns.append(returns[i])
        elif hour < 16:
            europe_returns.append(returns[i])
        else:
            americas_returns.append(returns[i])

    print("\nMean returns by session:")
    print(f"  Asia (00-08 UTC):     {np.mean(asia_returns)*100:.4f}%")
    print(f"  Europe (08-16 UTC):   {np.mean(europe_returns)*100:.4f}%")
    print(f"  Americas (16-24 UTC): {np.mean(americas_returns)*100:.4f}%")

    # Analyze returns by day of week
    weekday_returns = {i: [] for i in range(7)}
    for i, ts in enumerate(timestamps):
        if i == 0:
            continue
        dt = datetime.utcfromtimestamp(ts)
        weekday_returns[dt.weekday()].append(returns[i])

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print("\nMean returns by day of week:")
    for day in range(7):
        if weekday_returns[day]:
            mean_ret = np.mean(weekday_returns[day])
            print(f"  {day_names[day]}: {mean_ret*100:.4f}%")

    # 7. Summary
    print("\n7. Summary")
    print("-" * 40)

    print("\nData source:", "Bybit API" if use_real_data else "Synthetic")
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval} minutes")
    print(f"Samples: {n_samples}")
    print(f"Feature dimensions: {all_features.shape[1]}")
    print(f"  - Base features: {features.shape[1]}")
    print(f"  - Position encoding: {pos_features.shape[1]}")
    print(f"  - Calendar encoding: {cal_features.shape[1]}")
    print(f"  - Session encoding: {session_features.shape[1]}")

    print("\nThis example demonstrates:")
    print("  1. Loading real/synthetic crypto data")
    print("  2. Feature engineering with positional encodings")
    print("  3. Sequence creation for transformer models")
    print("  4. Basic backtesting framework")
    print("  5. Time pattern analysis by session and day")


if __name__ == "__main__":
    main()
