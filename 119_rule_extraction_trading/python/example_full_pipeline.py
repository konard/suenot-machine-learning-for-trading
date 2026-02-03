#!/usr/bin/env python3
"""
Full Pipeline Example: Rule Extraction Trading

This script demonstrates the complete workflow:
1. Fetch data from Bybit (or use simulated data)
2. Train a neural network model
3. Extract interpretable rules
4. Backtest the rule-based strategy
5. Compare with buy-and-hold benchmark
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from data_loader import BybitClient, compute_features, generate_labels
from rule_extractor import TrepanExtractor, evaluate_rules
from backtest import RuleBasedStrategy, backtest, buy_and_hold_benchmark, print_backtest_results, print_trades


def main():
    print("=" * 60)
    print("    Rule Extraction Trading - Full Pipeline Example")
    print("=" * 60)

    # Configuration
    symbol = "BTCUSDT"
    initial_capital = 100000.0

    # Step 1: Fetch or generate data
    print("\n[Step 1] Fetching market data...")

    try:
        client = BybitClient()
        df = client.get_klines(symbol, interval="60", limit=200)
        print(f"  Fetched {len(df)} candles from Bybit")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        use_real_data = True
    except Exception as e:
        print(f"  Could not fetch from Bybit: {e}")
        print("  Using simulated data instead...")
        use_real_data = False

        # Generate simulated data
        np.random.seed(42)
        n = 300

        prices = np.zeros(n)
        prices[0] = 50000  # BTC-like price

        for i in range(1, n):
            trend = 0.0001 * np.sin(i / 50)  # Cyclical trend
            noise = np.random.randn() * 0.015
            prices[i] = prices[i-1] * (1 + trend + noise)

        import pandas as pd
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='H'),
            'open': prices * (1 + np.random.randn(n) * 0.002),
            'high': prices * (1 + np.abs(np.random.randn(n) * 0.008)),
            'low': prices * (1 - np.abs(np.random.randn(n) * 0.008)),
            'close': prices,
            'volume': np.random.uniform(1000, 5000, n)
        })
        print(f"  Generated {len(df)} simulated candles")

    # Step 2: Compute features and labels
    print("\n[Step 2] Computing features...")

    features_df, feature_names = compute_features(df)
    labels = generate_labels(df)
    prices = df['close'].values

    X = features_df.values
    y = labels.values

    print(f"  Features: {feature_names}")
    print(f"  Samples: {len(X)}")
    print(f"  Label distribution: {(y == 1).sum()} BUY, {(y == -1).sum()} SELL")

    # Show latest features
    print(f"\n  Latest feature values:")
    for name, val in zip(feature_names, X[-1]):
        print(f"    {name}: {val:.4f}")

    # Step 3: Train neural network
    print("\n[Step 3] Training neural network...")

    # Split data (time-based, no shuffle)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    prices_train, prices_test = prices[:split_idx], prices[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15
    )
    nn.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, nn.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, nn.predict(X_test_scaled))

    print(f"  Training accuracy: {train_acc:.2%}")
    print(f"  Test accuracy: {test_acc:.2%}")

    # Step 4: Extract rules
    print("\n[Step 4] Extracting rules from neural network...")

    extractor = TrepanExtractor(
        feature_names=feature_names,
        max_depth=5,
        min_samples_leaf=20
    )

    # Use unscaled features for interpretable rules
    ruleset = extractor.extract(nn, X_train_scaled)

    # Update feature names in ruleset for display
    ruleset.feature_names = feature_names

    print(f"\n  Extracted {len(ruleset)} rules:")
    for i, rule in enumerate(ruleset.rules[:8]):
        print(f"    {i+1}. {rule.to_string(feature_names)}")
    if len(ruleset.rules) > 8:
        print(f"    ... ({len(ruleset.rules) - 8} more rules)")

    # Evaluate rules
    metrics = evaluate_rules(
        ruleset,
        X_train_scaled,
        y_train,
        nn.predict(X_train_scaled)
    )

    print(f"\n  Rule Quality Metrics:")
    print(f"    Fidelity (match NN): {metrics.get('fidelity', 0):.2%}")
    print(f"    Coverage: {metrics['coverage']:.2%}")
    print(f"    Accuracy: {metrics['accuracy']:.2%}")
    print(f"    Avg Complexity: {metrics['avg_complexity']:.2f} conditions/rule")

    # Step 5: Backtest strategy
    print("\n[Step 5] Backtesting rule-based strategy...")

    strategy = RuleBasedStrategy(ruleset)

    # Backtest on test data (use scaled features for consistency)
    result = backtest(
        strategy,
        X_test_scaled,
        prices_test,
        initial_capital=initial_capital,
        transaction_cost=0.001
    )

    # Buy-and-hold benchmark
    benchmark = buy_and_hold_benchmark(prices_test, initial_capital)

    # Print results
    print_backtest_results(result, benchmark, "Rule Extraction Strategy")

    # Show trades
    if result.trades:
        print("\n" + "=" * 50)
        print("  Recent Trades with Explanations")
        print("=" * 50)

        for i, trade in enumerate(result.trades[:5]):
            price_idx = min(trade.idx, len(prices_test) - 1)
            print(f"\nTrade {i+1}: {trade.trade_type} at ${trade.price:.2f}")
            print(f"  Time index: {trade.idx}")
            if trade.explanation:
                print("  Reasons:")
                for exp in trade.explanation[:3]:
                    print(f"    - {exp}")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    excess_return = result.total_return - benchmark.total_return
    if excess_return > 0:
        print(f"\n  The rule-based strategy OUTPERFORMED buy-and-hold")
        print(f"  by {excess_return*100:.2f}%")
    else:
        print(f"\n  The rule-based strategy UNDERPERFORMED buy-and-hold")
        print(f"  by {abs(excess_return)*100:.2f}%")

    if result.max_drawdown < benchmark.max_drawdown:
        print(f"\n  However, the strategy had LOWER risk:")
        print(f"    Max Drawdown: {result.max_drawdown*100:.2f}% vs {benchmark.max_drawdown*100:.2f}%")

    print("\n  Key Insights:")
    print(f"    - Neural network accuracy: {test_acc:.2%}")
    print(f"    - Rule fidelity to NN: {metrics.get('fidelity', 0):.2%}")
    print(f"    - Number of trades: {result.n_trades}")
    print(f"    - Win rate: {result.win_rate*100:.1f}%")

    print("\n  [Note: Past performance does not guarantee future results]")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
