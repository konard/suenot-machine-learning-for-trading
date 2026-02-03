# Rust Counterfactual Explanations

A Rust implementation of counterfactual explanations for trading models.

## Overview

This crate provides tools for generating counterfactual explanations that help understand what minimal changes to input features would flip a trading model's prediction.

## Features

- **API Client**: Fetch real-time data from Bybit exchange
- **Feature Engineering**: Technical indicators (RSI, MACD, Bollinger Bands)
- **Trading Classifier**: Simple neural network for trading signals
- **Counterfactual Generation**: Gradient-based optimization for counterfactuals
- **Backtesting**: Framework for strategy evaluation with CF-based risk management

## Quick Start

```bash
# Fetch market data
cargo run --example fetch_data

# Train a trading classifier
cargo run --example train_classifier

# Generate counterfactual explanations
cargo run --example generate_cf

# Run backtest with CF risk management
cargo run --example backtest
```

## Usage

```rust
use rust_counterfactual::{
    model::TradingClassifier,
    counterfactual::CounterfactualOptimizer,
    data::get_sample_data,
};

// Load data and train model
let (features, labels, names) = get_sample_data();
let mut model = TradingClassifier::new(6, 64, 3);
model.train(&features, &labels, 50, 0.01);

// Generate counterfactual
let optimizer = CounterfactualOptimizer::new(&model);
let result = optimizer.generate(&features[0], 2, 100, 0.05);

println!("{}", result);
```

## Project Structure

```
rust_counterfactual/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Library entry point
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs        # Bybit API client
│   ├── data/
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading
│   │   └── features.rs     # Feature engineering
│   ├── model/
│   │   ├── mod.rs
│   │   ├── classifier.rs   # Trading classifier
│   │   └── config.rs       # Model configuration
│   ├── counterfactual/
│   │   ├── mod.rs
│   │   ├── generator.rs    # NN-based generator
│   │   ├── optimizer.rs    # Gradient-based optimizer
│   │   └── metrics.rs      # Proximity metrics
│   └── strategy/
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting framework
└── examples/
    ├── fetch_data.rs
    ├── train_classifier.rs
    ├── generate_cf.rs
    └── backtest.rs
```

## Dependencies

- `reqwest`: HTTP client for API calls
- `serde/serde_json`: Serialization
- `ndarray`: Numerical arrays
- `rand`: Random number generation
- `chrono`: Date/time handling
- `anyhow/thiserror`: Error handling

## License

MIT
