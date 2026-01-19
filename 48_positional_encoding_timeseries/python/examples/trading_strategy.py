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
from data import KlineData, prepare_features, create_sequences, train_test_split_time_series
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

    # Create KlineData object for prepare_features
    kline_data = KlineData(
        timestamp=timestamps,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        volume=volumes,
        symbol="SYNTH",
        interval="1h"
    )
    features_df = prepare_features(kline_data)
    print(f"Base features shape: {features_df.shape}")
    feature_cols = ['log_return', 'volatility', 'volume_change', 'momentum', 'rsi', 'range', 'vwap_deviation']
    features_numeric = features_df[feature_cols].values
    print(f"Features: {feature_cols}")

    # Add positional encoding
    d_model = 16
    n_features_len = len(features_df)
    pos_encoding = SinusoidalPositionalEncoding(d_model, n_features_len)
    x_pos = torch.zeros(1, n_features_len, d_model)
    pos_features = pos_encoding(x_pos).squeeze(0).detach().numpy()
    print(f"Position encoding shape: {pos_features.shape}")

    # Add calendar encoding - requires discrete features
    calendar = CalendarEncoding(d_model)
    dayofweek = torch.tensor(features_df['dayofweek'].values).unsqueeze(0)
    month = torch.tensor(features_df['month'].values - 1).unsqueeze(0)  # 0-indexed
    quarter = torch.tensor(features_df['quarter'].values).unsqueeze(0)
    hour = torch.tensor(features_df['hour'].values).unsqueeze(0)
    session_idx = torch.zeros_like(hour)  # placeholder
    cal_features = calendar(dayofweek, month, quarter, hour, session_idx).squeeze(0).detach().numpy()
    print(f"Calendar encoding shape: {cal_features.shape}")

    # Add market session encoding (crypto) - requires hour tensor
    session_enc = MarketSessionEncoding(d_model, market_type='crypto')
    session_features = session_enc(hour).squeeze(0).detach().numpy()
    print(f"Session encoding shape: {session_features.shape}")

    # Total features
    total_dim = features_numeric.shape[1] + pos_features.shape[1] + cal_features.shape[1] + session_features.shape[1]
    print(f"Total feature dimension: {total_dim}")

    # 3. Create Training Sequences
    print("\n3. Sequence Creation")
    print("-" * 40)

    sequence_length = 24  # 24-hour lookback
    target_length = 1     # 1-hour prediction

    X, y, seq_timestamps = create_sequences(
        features_df, feature_cols, target_col='log_return',
        seq_len=sequence_length, horizon=target_length
    )
    print(f"Created {len(X)} sequences")
    print(f"Sequence shape: {X.shape}")

    # Train/test split
    train_ratio = 0.8
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_time_series(
        X, y, timestamps=None, train_ratio=train_ratio, val_ratio=0.1
    )
    print(f"Train: {len(X_train)} sequences, Val: {len(X_val)}, Test: {len(X_test)} sequences")

    # 4. Trading Strategy
    print("\n4. Trading Strategy")
    print("-" * 40)

    strategy = TradingStrategy(threshold=0.001, max_position=1.0, transaction_cost=0.001)

    print("Strategy configuration:")
    print(f"  Threshold: {strategy.threshold:.4f}")
    print(f"  Max Position: {strategy.max_position:.1f}")
    print(f"  Transaction Cost: {strategy.transaction_cost:.4f}")

    # Get returns from features_df for backtesting
    returns_for_backtest = features_df['log_return'].values

    # Generate signals from a simple moving average crossover (simulating model predictions)
    window = 24
    predictions = np.zeros(len(returns_for_backtest))
    for i in range(window, len(returns_for_backtest)):
        ma_short = returns_for_backtest[i-12:i].mean()
        ma_long = returns_for_backtest[i-24:i].mean()
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

    result = run_backtest(predictions, returns_for_backtest, strategy, periods_per_year=8760)

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

    bh_result = calculate_buy_and_hold(returns_for_backtest, periods_per_year=8760)

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
    comparison = compare_strategies(predictions, returns_for_backtest, thresholds, periods_per_year=8760)

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
