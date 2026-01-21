#!/usr/bin/env python3
"""
Basic example: Zero-shot regime prediction with synthetic data.

This example demonstrates:
1. Creating the zero-shot model
2. Generating synthetic market features
3. Predicting market regimes

Run: python example_basic.py
"""

import numpy as np
import torch
from zero_shot_trading import (
    MarketEncoder,
    AttributeEncoder,
    ZeroShotTradingModel,
    MarketRegime,
)


def generate_synthetic_features(
    timesteps: int = 50,
    num_features: int = 11,
    trend: str = "uptrend"
) -> torch.Tensor:
    """Generate synthetic market features for demonstration."""

    features = []

    for i in range(timesteps):
        t = i / timesteps

        # Base values depend on trend
        if trend == "uptrend":
            base_return = 0.002 + t * 0.001
            rsi_base = 0.55 + t * 0.15
            sma_ratio = 1.01 + t * 0.02
        elif trend == "downtrend":
            base_return = -0.002 - t * 0.001
            rsi_base = 0.45 - t * 0.15
            sma_ratio = 0.99 - t * 0.02
        else:  # sideways
            base_return = 0.0
            rsi_base = 0.5
            sma_ratio = 1.0

        row = [
            base_return + np.random.normal(0, 0.005),      # returns
            base_return + np.random.normal(0, 0.005),      # log_returns
            0.02 + np.random.normal(0, 0.005),             # range
            0.5 + np.random.normal(0, 0.1),                # close_position
            1.0 + np.random.normal(0, 0.2),                # volume_ratio
            0.015 + np.random.normal(0, 0.003),            # volatility_20
            0.02 + np.random.normal(0, 0.005),             # volatility_5
            sma_ratio + np.random.normal(0, 0.01),         # sma_ratio
            rsi_base + np.random.normal(0, 0.05),          # rsi (normalized)
            base_return * 0.5 + np.random.normal(0, 0.001),# macd
            0.5 + base_return * 20 + np.random.normal(0, 0.1),  # bb_position
        ]
        features.append(row)

    return torch.tensor(features, dtype=torch.float32)


def main():
    print("Zero-Shot Trading: Basic Prediction Example")
    print("=" * 50)
    print()

    # Model configuration
    input_features = 11
    embedding_dim = 64
    num_regimes = 5

    # Create model components
    market_encoder = MarketEncoder(
        input_dim=input_features,
        hidden_dim=128,
        embedding_dim=embedding_dim
    )

    attribute_encoder = AttributeEncoder(
        categorical_dims=[5, 4, 4, 15],  # asset_type, market_cap, volatility, sector
        numerical_dim=4,
        embedding_dim=embedding_dim
    )

    model = ZeroShotTradingModel(
        market_encoder=market_encoder,
        attribute_encoder=attribute_encoder,
        embedding_dim=embedding_dim,
        num_regimes=num_regimes
    )

    model.eval()

    print("Model created with:")
    print(f"  - Input features: {input_features}")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Output classes: {num_regimes} market regimes")
    print()

    # Test on different market conditions
    scenarios = [
        ("Strong Uptrend", "uptrend"),
        ("Sideways Market", "sideways"),
        ("Strong Downtrend", "downtrend"),
    ]

    for scenario_name, trend in scenarios:
        print("-" * 50)
        print(f"Scenario: {scenario_name}")
        print("-" * 50)

        # Generate features
        features = generate_synthetic_features(timesteps=50, trend=trend)
        features = features.unsqueeze(0)  # Add batch dimension

        print(f"Features shape: {features.shape}")

        # Get prediction
        with torch.no_grad():
            # Encode market data
            market_embedding = model.market_encoder(features)

            # Get regime logits
            regime_logits = model.regime_head(market_embedding)

            # Get probabilities
            probs = torch.softmax(regime_logits, dim=-1).squeeze()
            predicted_idx = torch.argmax(probs).item()
            predicted_regime = MarketRegime(predicted_idx)
            confidence = probs[predicted_idx].item()

        print(f"\nPredicted Regime: {predicted_regime.name}")
        print(f"Confidence: {confidence:.1%}")
        print("\nRegime Probabilities:")

        for i, regime in enumerate(MarketRegime):
            prob = probs[i].item()
            bar_len = int(prob * 30)
            bar = "█" * bar_len
            print(f"  {regime.name:18} {prob:>6.1%} {bar}")

        print()

    print("=" * 50)
    print("✓ Example completed successfully!")


if __name__ == "__main__":
    main()
