#!/usr/bin/env python3
"""
Trading Strategy with Positional Encoding Example

This example shows how to combine positional encodings with trading
strategies and backtesting for financial time series.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datetime import datetime, timedelta

from positional_encoding import (
    SinusoidalPositionalEncoding,
    CalendarEncoding,
    MarketSessionEncoding,
)
from data import prepare_features, create_sequences, train_test_split
from strategy import TradingStrategy, run_backtest, calculate_buy_and_hold, compare_strategies


def generate_synthetic_data(n_samples: int, seed: int = 42) -> tuple:
    """Generate synthetic price data for demonstration."""
    np.random.seed(seed)

    base_time = datetime(2024, 1, 1, 0, 0)
    timestamps = np.array([
        int((base_time + timedelta(hours=i)).timestamp())
        for i in range(n_samples)
    ])

    # Random walk with trend and volatility
    returns = np.random.normal(0.0001, 0.02, n_samples)
    prices = 45000 * np.cumprod(1 + returns)

    # Generate OHLCV
    opens = np.roll(prices, 1)
    opens[0] = 45000
    closes = prices
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
    volumes = np.random.uniform(500, 1500, n_samples)

    return timestamps, opens, highs, lows, closes, volumes, returns


def main():
    print("Trading Strategy Example")
    print("=" * 50)

    # 1. Generate or load data
    print("\n1. Data Preparation")
    print("-" * 40)

    n_samples = 2000
    timestamps, opens, highs, lows, closes, volumes, returns = generate_synthetic_data(n_samples)

    print(f"Generated {n_samples} hourly candles")
    print(f"Price range: ${closes.min():.2f} - ${closes.max():.2f}")
    print(f"Mean return: {returns.mean():.6f}")
    print(f"Std return: {returns.std():.6f}")

    # 2. Feature Engineering
    print("\n2. Feature Engineering")
    print("-" * 40)

    # Prepare base features
    features = prepare_features(opens, highs, lows, closes, volumes)
    print(f"Base features shape: {features.shape}")
    print(f"Features: return, log_volume, range, body_ratio, direction, return_pct")

    # Add positional encoding
    d_model = 16
    pos_encoding = SinusoidalPositionalEncoding(d_model, n_samples)
    x_pos = torch.zeros(1, n_samples, d_model)
    pos_features = pos_encoding(x_pos).squeeze(0).numpy()
    print(f"Position encoding shape: {pos_features.shape}")

    # Add calendar encoding
    calendar = CalendarEncoding(d_model)
    ts_tensor = torch.tensor(timestamps).unsqueeze(0)
    cal_features = calendar(ts_tensor).squeeze(0).numpy()
    print(f"Calendar encoding shape: {cal_features.shape}")

    # Add market session encoding
    session = MarketSessionEncoding(d_model, market_type='crypto')
    session_features = session(ts_tensor).squeeze(0).numpy()
    print(f"Session encoding shape: {session_features.shape}")

    # Total features
    total_dim = features.shape[1] + pos_features.shape[1] + cal_features.shape[1] + session_features.shape[1]
    print(f"Total feature dimension: {total_dim}")

    # 3. Create Training Sequences
    print("\n3. Sequence Creation")
    print("-" * 40)

    sequence_length = 24  # 24-hour lookback
    target_length = 1     # 1-hour prediction

    sequences, targets, seq_timestamps = create_sequences(
        features, returns, timestamps, sequence_length, target_length
    )
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences[0].shape}")

    # Train/test split
    train_ratio = 0.8
    train_seqs, test_seqs = train_test_split(sequences, train_ratio)
    train_targets, test_targets = train_test_split(targets, train_ratio)
    print(f"Train: {len(train_seqs)} sequences, Test: {len(test_seqs)} sequences")

    # 4. Trading Strategy
    print("\n4. Trading Strategy")
    print("-" * 40)

    strategy = TradingStrategy(threshold=0.001, max_position=1.0, transaction_cost=0.001)

    print("Strategy configuration:")
    print(f"  Threshold: {strategy.threshold:.4f}")
    print(f"  Max Position: {strategy.max_position:.1f}")
    print(f"  Transaction Cost: {strategy.transaction_cost:.4f}")

    # Generate signals from a simple moving average crossover (simulating model predictions)
    window = 24
    predictions = np.zeros(len(returns))
    for i in range(window, len(returns)):
        ma_short = returns[i-12:i].mean()
        ma_long = returns[i-24:i].mean()
        predictions[i] = ma_short - ma_long  # Momentum signal

    signals = strategy.generate_signals(predictions)
    positions = strategy.generate_position_sizes(signals)

    long_count = np.sum(signals > 0)
    short_count = np.sum(signals < 0)
    neutral_count = np.sum(signals == 0)

    print("\nSignal distribution:")
    print(f"  Long: {long_count} ({100.0 * long_count / len(signals):.1f}%)")
    print(f"  Short: {short_count} ({100.0 * short_count / len(signals):.1f}%)")
    print(f"  Neutral: {neutral_count} ({100.0 * neutral_count / len(signals):.1f}%)")

    # 5. Backtesting
    print("\n5. Backtesting")
    print("-" * 40)

    result = run_backtest(predictions, returns, strategy, periods_per_year=8760)

    print("\nStrategy Performance:")
    print(f"  Total Return: {result.total_return * 100:.2f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown * 100:.2f}%")
    print(f"  Win Rate: {result.win_rate * 100:.2f}%")
    print(f"  Number of Trades: {result.n_trades}")

    # 6. Benchmark Comparison
    print("\n6. Benchmark Comparison")
    print("-" * 40)

    bh_result = calculate_buy_and_hold(returns, periods_per_year=8760)

    print("\nBuy & Hold Performance:")
    print(f"  Total Return: {bh_result.total_return * 100:.2f}%")
    print(f"  Sharpe Ratio: {bh_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {bh_result.max_drawdown * 100:.2f}%")

    print("\nStrategy vs Buy & Hold:")
    excess_return = result.total_return - bh_result.total_return
    sharpe_diff = result.sharpe_ratio - bh_result.sharpe_ratio
    print(f"  Excess Return: {excess_return * 100:.2f}%")
    print(f"  Sharpe Difference: {sharpe_diff:.2f}")

    # 7. Threshold Comparison
    print("\n7. Threshold Optimization")
    print("-" * 40)

    thresholds = [0.0005, 0.001, 0.002, 0.005]
    comparison = compare_strategies(predictions, returns, thresholds, periods_per_year=8760)

    print("\n| Threshold | Return   | Sharpe | MaxDD   | Trades |")
    print("|-----------|----------|--------|---------|--------|")
    for _, row in comparison.iterrows():
        print(f"| {row['threshold']:.4f}    | {row['total_return']*100:>7.2f}% | {row['sharpe_ratio']:>6.2f} | {row['max_drawdown']*100:>6.2f}% | {row['n_trades']:>6} |")

    # 8. Summary
    print("\n8. Integration with Positional Encoding")
    print("-" * 40)

    print("\nHow positional encoding helps trading strategies:")
    print()
    print("1. Sequence Position (Sinusoidal/RoPE):")
    print("   - Helps model understand temporal order of observations")
    print("   - Enables attention to focus on recent vs distant past")
    print()
    print("2. Calendar Features:")
    print("   - Captures day-of-week effects (Monday dip, Friday close)")
    print("   - Month-end rebalancing patterns")
    print("   - Seasonal/quarterly patterns")
    print()
    print("3. Market Session:")
    print("   - Regional activity patterns (Asia/Europe/Americas)")
    print("   - Session overlap volatility")
    print("   - Pre/post market behavior")
    print()
    print("Recommended Pipeline:")
    print("  1. Prepare base features (OHLCV + technical indicators)")
    print("  2. Add positional encoding (RoPE for transformers)")
    print("  3. Add calendar encoding (for time-aware predictions)")
    print("  4. Add market session encoding (for intraday)")
    print("  5. Train transformer model with combined features")
    print("  6. Generate predictions and signals")
    print("  7. Backtest with realistic transaction costs")
    print("  8. Optimize threshold and position sizing")


if __name__ == "__main__":
    main()
