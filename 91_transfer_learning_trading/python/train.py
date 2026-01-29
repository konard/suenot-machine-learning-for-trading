"""
Training script for Transfer Learning Trading model.

Usage:
    python train.py --source BTC/USDT --target ETH/USDT --method fine_tune
    python train.py --use-synthetic --epochs 50
"""

import argparse
import numpy as np
import sys

from model import TransferLearningPipeline

# ── Feature Extraction ─────────────────────────────────────

def compute_returns(prices):
    """Compute percentage returns from price series."""
    returns = np.diff(prices) / prices[:-1]
    return returns


def extract_features(prices, volumes, lookback=20):
    """Extract market features from price and volume data.

    Features:
    - Returns (mean, std, skew, kurtosis)
    - Momentum (1-bar, 5-bar)
    - RSI
    - SMA ratios
    - Volume ratio
    - Bollinger Band position
    """
    n_samples = len(prices) - lookback
    if n_samples <= 0:
        return np.array([]), np.array([])

    features = []
    for i in range(n_samples):
        window_prices = prices[i:i + lookback + 1]
        window_volumes = volumes[i:i + lookback + 1]
        ret = compute_returns(window_prices)

        feat = []
        # Return features
        feat.append(ret[-1] if len(ret) > 0 else 0.0)
        feat.append(np.mean(ret) if len(ret) > 0 else 0.0)
        feat.append(np.std(ret) if len(ret) > 1 else 0.0)

        # Momentum
        if len(window_prices) >= 2:
            feat.append((window_prices[-1] - window_prices[-2]) / max(window_prices[-2], 1e-10))
        else:
            feat.append(0.0)
        if len(window_prices) >= 6:
            feat.append((window_prices[-1] - window_prices[-6]) / max(window_prices[-6], 1e-10))
        else:
            feat.append(0.0)

        # RSI
        gains = np.sum(ret[ret > 0]) if len(ret) > 0 else 0.0
        losses = np.sum(np.abs(ret[ret < 0])) if len(ret) > 0 else 0.0
        if losses > 0:
            rs = gains / losses
            rsi = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi = 50.0
        feat.append(rsi / 100.0)

        # SMA ratios
        sma5 = np.mean(window_prices[-5:])
        sma10 = np.mean(window_prices[-10:]) if len(window_prices) >= 10 else sma5
        feat.append(window_prices[-1] / max(sma5, 1e-10))
        feat.append(window_prices[-1] / max(sma10, 1e-10))

        # Volume ratio
        vol_mean = np.mean(window_volumes) if len(window_volumes) > 0 else 1.0
        feat.append(window_volumes[-1] / max(vol_mean, 1e-10))

        # Bollinger Band position
        sma20 = np.mean(window_prices)
        std20 = np.std(window_prices)
        if std20 > 0:
            bb_pos = (window_prices[-1] - (sma20 - 2 * std20)) / (4 * std20)
            bb_pos = np.clip(bb_pos, 0, 1)
        else:
            bb_pos = 0.5
        feat.append(bb_pos)

        features.append(feat)

    return np.array(features)


def generate_labels(prices, lookback=20, forward=3, threshold=0.005):
    """Generate labels based on future returns.

    0 = Down, 1 = Hold, 2 = Up
    """
    n = len(prices) - lookback - forward
    if n <= 0:
        return np.array([])

    labels = []
    for i in range(n):
        current = prices[i + lookback]
        future = prices[i + lookback + forward]
        ret = (future - current) / max(current, 1e-10)

        if ret > threshold:
            labels.append(2)  # Up
        elif ret < -threshold:
            labels.append(0)  # Down
        else:
            labels.append(1)  # Hold

    return np.array(labels)


def generate_synthetic_data(n_bars=500, start_price=100.0, volatility=0.02):
    """Generate synthetic OHLCV data for testing."""
    prices = [start_price]
    volumes = []

    for _ in range(n_bars - 1):
        ret = np.random.normal(0, volatility)
        prices.append(prices[-1] * (1 + ret))
        volumes.append(1000 * (1 + np.random.random() * 2))

    volumes.append(1000 * (1 + np.random.random() * 2))
    return np.array(prices), np.array(volumes)


# ── Training ────────────────────────────────────────────────

def train_transfer_model(args):
    """Run the transfer learning training pipeline."""
    print("=" * 60)
    print("Transfer Learning for Trading - Training")
    print("=" * 60)

    # Generate or load data
    if args.use_synthetic:
        print("\nGenerating synthetic data...")
        source_prices, source_volumes = generate_synthetic_data(
            n_bars=500, start_price=100.0, volatility=0.02
        )
        target_prices, target_volumes = generate_synthetic_data(
            n_bars=150, start_price=50.0, volatility=0.04
        )
        print(f"  Source: {len(source_prices)} bars")
        print(f"  Target: {len(target_prices)} bars")
    else:
        print(f"\nSource: {args.source}")
        print(f"Target: {args.target}")
        print("Note: Real data fetching requires API integration.")
        print("Using synthetic data for demonstration.\n")
        source_prices, source_volumes = generate_synthetic_data(500, 100.0, 0.02)
        target_prices, target_volumes = generate_synthetic_data(150, 50.0, 0.04)

    # Extract features
    print("\nExtracting features...")
    lookback = 20
    source_features = extract_features(source_prices, source_volumes, lookback)
    target_features = extract_features(target_prices, target_volumes, lookback)

    source_labels = generate_labels(source_prices, lookback, forward=3, threshold=0.005)
    target_labels = generate_labels(target_prices, lookback, forward=3, threshold=0.005)

    # Align feature and label counts
    n_source = min(len(source_features), len(source_labels))
    n_target = min(len(target_features), len(target_labels))
    source_features = source_features[:n_source]
    source_labels = source_labels[:n_source]
    target_features = target_features[:n_target]
    target_labels = target_labels[:n_target]

    if n_source == 0 or n_target == 0:
        print("Not enough data. Exiting.")
        return

    n_features = source_features.shape[1]
    print(f"  Source: {n_source} samples x {n_features} features")
    print(f"  Target: {n_target} samples x {n_features} features")
    print(f"  Source label distribution: {np.bincount(source_labels, minlength=3)}")
    print(f"  Target label distribution: {np.bincount(target_labels, minlength=3)}")

    # Initialize pipeline
    print(f"\nInitializing pipeline (method={args.method})...")
    pipeline = TransferLearningPipeline(
        input_dim=n_features,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        adaptation_method=args.method,
    )

    # Step 1: Pre-train on source
    print("\nStep 1: Pre-training on source domain...")
    pretrain_history = pipeline.pretrain(
        source_features, source_labels,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
    )
    print(f"  Final loss: {pretrain_history['loss'][-1]:.4f}")
    print(f"  Final accuracy: {pretrain_history['accuracy'][-1]:.2%}")
    print(f"  Epochs trained: {len(pretrain_history['loss'])}")

    # Baseline: evaluate on target before adaptation
    baseline_preds = pipeline.predict(target_features)
    baseline_accuracy = np.mean(baseline_preds == target_labels)
    print(f"\n  Baseline target accuracy (before adaptation): {baseline_accuracy:.2%}")

    # Step 2: Adapt to target
    print("\nStep 2: Adapting to target domain...")
    adapt_history = pipeline.adapt(
        source_features, target_features,
        target_labels=target_labels,
        epochs=args.adapt_epochs,
        lr=args.adapt_lr,
        adaptation_weight=args.adapt_weight,
    )
    print(f"  Final adaptation loss: {adapt_history['adapt_loss'][-1]:.6f}")
    print(f"  Final task loss: {adapt_history['task_loss'][-1]:.4f}")

    # Step 3: Evaluate
    print("\nStep 3: Evaluation...")
    source_preds = pipeline.predict(source_features)
    target_preds = pipeline.predict(target_features)

    source_accuracy = np.mean(source_preds == source_labels)
    target_accuracy = np.mean(target_preds == target_labels)
    transfer_gain = target_accuracy - baseline_accuracy

    print(f"  Source accuracy: {source_accuracy:.2%}")
    print(f"  Target accuracy: {target_accuracy:.2%}")
    print(f"  Baseline accuracy: {baseline_accuracy:.2%}")
    print(f"  Transfer gain: {transfer_gain:+.2%}")

    # Class-wise accuracy
    for cls in range(3):
        cls_mask = target_labels == cls
        if cls_mask.sum() > 0:
            cls_acc = np.mean(target_preds[cls_mask] == target_labels[cls_mask])
            cls_name = {0: 'Down', 1: 'Hold', 2: 'Up'}[cls]
            print(f"  {cls_name} accuracy: {cls_acc:.2%} ({cls_mask.sum()} samples)")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning for Trading - Training")
    parser.add_argument("--source", type=str, default="BTC/USDT", help="Source trading pair")
    parser.add_argument("--target", type=str, default="ETH/USDT", help="Target trading pair")
    parser.add_argument("--method", type=str, default="mmd",
                        choices=["mmd", "coral", "none"], help="Adaptation method")
    parser.add_argument("--epochs", type=int, default=50, help="Pre-training epochs")
    parser.add_argument("--adapt-epochs", type=int, default=30, help="Adaptation epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Pre-training learning rate")
    parser.add_argument("--adapt-lr", type=float, default=0.0001, help="Adaptation learning rate")
    parser.add_argument("--adapt-weight", type=float, default=1.0, help="Adaptation loss weight")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--feature-dim", type=int, default=64, help="Feature embedding dimension")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data")

    args = parser.parse_args()
    train_transfer_model(args)
