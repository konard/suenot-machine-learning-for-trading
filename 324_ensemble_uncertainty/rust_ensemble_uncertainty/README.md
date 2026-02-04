# Ensemble Uncertainty Trading - Rust Implementation

This is the Rust implementation of the Ensemble Uncertainty trading system for cryptocurrency markets using Bybit data.

## Overview

This library provides:

- **Ensemble Models**: Random Forest, Gradient Boosting, and Stacking ensembles
- **Uncertainty Quantification**: Prediction variance, confidence intervals, calibration
- **Trading Strategy**: Uncertainty-aware position sizing and signal generation
- **Backtesting**: Full backtesting engine with performance metrics

## Project Structure

```
rust_ensemble_uncertainty/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                 # Library entry point
│   ├── api/                   # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs          # HTTP client for Bybit
│   │   └── types.rs           # API data types
│   ├── ensemble/              # Ensemble models
│   │   ├── mod.rs
│   │   ├── traits.rs          # Ensemble traits
│   │   ├── decision_tree.rs   # Decision tree implementation
│   │   ├── random_forest.rs   # Random Forest
│   │   ├── gradient_boosting.rs # Gradient Boosting
│   │   └── stacking.rs        # Stacking ensemble
│   ├── uncertainty/           # Uncertainty quantification
│   │   ├── mod.rs
│   │   ├── quantifier.rs      # Uncertainty metrics
│   │   └── calibration.rs     # Model calibration
│   ├── strategy/              # Trading strategy
│   │   ├── mod.rs
│   │   ├── signal.rs          # Signal types and generation
│   │   └── uncertainty_strategy.rs # Uncertainty-aware strategy
│   └── backtest/              # Backtesting
│       ├── mod.rs
│       ├── engine.rs          # Backtest engine
│       └── report.rs          # Performance reports
└── examples/
    ├── fetch_data.rs          # Fetch data from Bybit
    ├── train_ensemble.rs      # Train ensemble models
    └── live_trading.rs        # Live trading simulation
```

## Usage

### Building

```bash
cd rust_ensemble_uncertainty
cargo build --release
```

### Running Examples

```bash
# Fetch market data
cargo run --example fetch_data

# Train ensemble models
cargo run --example train_ensemble

# Run trading simulation
cargo run --example live_trading
```

### Library Usage

```rust
use ensemble_uncertainty_trading::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fetch data
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", "60", 1000).await?;

    // Train ensemble
    let mut rf = RandomForestEnsemble::new(100, Some(10));
    rf.fit(&features, &targets)?;

    // Get predictions with uncertainty
    let (predictions, uncertainties) = rf.predict_with_uncertainty(&test_features)?;

    // Generate trading signals
    let strategy = UncertaintyStrategy::new(0.001, 0.03, 0.6);
    let signals = strategy.generate_portfolio_signals(
        &symbols,
        &predictions,
        &uncertainties,
        &prices,
    );

    Ok(())
}
```

## Key Features

### Ensemble Models

- **Random Forest**: Bootstrap aggregating with tree disagreement uncertainty
- **Gradient Boosting**: Sequential boosting with quantile regression for intervals
- **Stacking**: Meta-learning that combines base models

### Uncertainty Metrics

- Prediction variance and standard deviation
- Coefficient of variation
- Interquartile range
- Confidence intervals (any level)
- Epistemic/aleatoric decomposition

### Trading Strategy

- Confidence-based position sizing
- Kelly criterion with uncertainty adjustment
- Dynamic stop-loss based on uncertainty
- Signal filtering by confidence threshold

## Performance

The uncertainty-aware strategy typically shows:

- **Higher Sharpe ratio**: By avoiding trades when uncertain
- **Lower drawdowns**: By reducing position size in uncertain conditions
- **Better win rates**: By trading only high-confidence signals

## Dependencies

- `ndarray`: Numerical arrays
- `reqwest`: HTTP client
- `tokio`: Async runtime
- `rayon`: Parallel processing
- `serde`: Serialization

## License

MIT License
