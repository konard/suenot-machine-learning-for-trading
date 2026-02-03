"""
Training script for Concept Bottleneck Trading model.
"""

import argparse
import numpy as np
import pandas as pd
from typing import Optional

from data_loader import BybitDataLoader, generate_sample_data
from concepts import ConceptExtractor, generate_concept_labels
from model import ConceptBottleneckTrader, TradingSignal
from backtest import Backtester, print_backtest_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Concept Bottleneck Trading model"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Data interval (default: 1h)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples to fetch (default: 500)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Use synthetic sample data instead of real API data"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Training data split ratio (default: 0.7)"
    )
    return parser.parse_args()


def load_data(args) -> pd.DataFrame:
    """Load market data from API or generate sample data."""
    if args.use_sample_data:
        print("Generating sample data...")
        return generate_sample_data(n_samples=args.samples)

    print(f"Fetching {args.symbol} {args.interval} data from Bybit...")
    loader = BybitDataLoader()

    all_data = []
    remaining = args.samples

    while remaining > 0:
        batch_size = min(remaining, 200)
        df = loader.fetch_klines(
            symbol=args.symbol,
            interval=args.interval,
            limit=batch_size
        )
        all_data.append(df)
        remaining -= batch_size

        if len(df) < batch_size:
            break

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=['timestamp'])
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        return combined

    print("Warning: Could not fetch data from API, using sample data")
    return generate_sample_data(n_samples=args.samples)


def prepare_labels(df: pd.DataFrame, future_window: int = 5) -> np.ndarray:
    """Generate labels for supervised learning."""
    labels = generate_concept_labels(df, future_window=future_window, threshold=0.02)

    # Convert from [-1, 0, 1] to [0, 1, 2] for classification
    labels = labels + 1  # Now: 0=sell, 1=hold, 2=buy

    return labels


def train_model(
    model: ConceptBottleneckTrader,
    train_df: pd.DataFrame,
    train_labels: np.ndarray,
    epochs: int,
    learning_rate: float
):
    """Train the CBM model."""
    print(f"Training model for {epochs} epochs...")

    model.train(
        train_df,
        train_labels,
        epochs=epochs,
        learning_rate=learning_rate
    )

    print("Training complete!")


def evaluate_model(
    model: ConceptBottleneckTrader,
    test_df: pd.DataFrame,
) -> None:
    """Evaluate model on test data."""
    print("\nRunning backtest...")

    backtester = Backtester(
        initial_capital=100000.0,
        position_size=0.1,
        transaction_cost=0.001,
    )

    result = backtester.run(model, test_df, warmup_period=50)
    print_backtest_report(result)


def demonstrate_concepts(model: ConceptBottleneckTrader, df: pd.DataFrame):
    """Demonstrate concept extraction and interpretation."""
    print("\n" + "=" * 50)
    print("CONCEPT DEMONSTRATION")
    print("=" * 50)

    prediction = model.predict_with_explanation(df)

    print(f"\nTrading Signal: {prediction.signal.value.upper()}")
    print(f"Confidence: {prediction.confidence:.2%}")

    print("\nExtracted Concepts:")
    concepts = prediction.concepts.to_dict()
    for name, value in concepts.items():
        print(f"  {name}: {value}")

    print("\nConcept Contributions to Decision:")
    sorted_contributions = sorted(
        prediction.concept_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for name, contribution in sorted_contributions[:5]:
        direction = "+" if contribution > 0 else ""
        print(f"  {name}: {direction}{contribution:.4f}")


def demonstrate_intervention(model: ConceptBottleneckTrader, df: pd.DataFrame):
    """Demonstrate concept intervention capability."""
    print("\n" + "=" * 50)
    print("INTERVENTION DEMONSTRATION")
    print("=" * 50)

    # Get original prediction
    original = model.predict_with_explanation(df)
    print(f"\nOriginal Signal: {original.signal.value.upper()}")
    print(f"Original Concepts: trend={original.concepts.trend_direction.value}, "
          f"volatility={original.concepts.volatility_regime.value}")

    # Override volatility concept
    print("\n[Intervention: Setting volatility_regime to HIGH]")
    intervened = model.predict_with_intervention(
        df,
        concept_overrides={'volatility_regime': 1.0, 'volatility_value': 2.0}
    )
    print(f"Intervened Signal: {intervened.signal.value.upper()}")

    # Override trend concept
    print("\n[Intervention: Setting trend to BEARISH]")
    intervened = model.predict_with_intervention(
        df,
        concept_overrides={'trend_direction': -1.0, 'trend_strength': 0.8}
    )
    print(f"Intervened Signal: {intervened.signal.value.upper()}")


def main():
    args = parse_args()

    # Load data
    df = load_data(args)
    print(f"Loaded {len(df)} data points")

    # Split into train/test
    split_idx = int(len(df) * args.train_split)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Prepare labels
    train_labels = prepare_labels(train_df)

    # Create and train model
    model = ConceptBottleneckTrader()
    train_model(model, train_df, train_labels, args.epochs, args.learning_rate)

    # Evaluate on test set
    evaluate_model(model, test_df)

    # Demonstrate concepts
    demonstrate_concepts(model, test_df)

    # Demonstrate intervention
    demonstrate_intervention(model, test_df)

    print("\n" + "=" * 50)
    print("Training and evaluation complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
