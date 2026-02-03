# Layer-wise Relevance Propagation (LRP) - Rust Implementation

High-performance Rust implementation of Layer-wise Relevance Propagation for explainable trading models.

## Features

- Multiple LRP rules: LRP-0, LRP-epsilon, LRP-gamma, LRP-alpha-beta
- Bybit API integration for market data
- Technical indicator calculation
- Backtesting engine with LRP insights
- Async/await support with Tokio

## Project Structure

```
rust_lrp/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Library exports
│   ├── api/                # Exchange API client
│   │   ├── mod.rs
│   │   ├── client.rs       # Bybit HTTP client
│   │   └── types.rs        # API types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset structure
│   ├── model/              # LRP model
│   │   ├── mod.rs
│   │   ├── config.rs       # Configuration
│   │   ├── linear.rs       # LRP linear layer
│   │   ├── network.rs      # Network architecture
│   │   └── lrp.rs          # LRP rules
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting
└── examples/
    ├── fetch_data.rs       # Download market data
    ├── train.rs            # Train model
    ├── explain.rs          # Generate explanations
    └── backtest.rs         # Run backtest
```

## Quick Start

```bash
# Build the project
cargo build --release

# Fetch market data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --interval 60

# Train a model
cargo run --example train -- --epochs 100 --batch-size 32

# Generate explanations
cargo run --example explain -- --model model.bin --input data.json

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Usage as Library

```rust
use rust_lrp::{
    model::{LRPNetwork, LRPConfig, LRPRule},
    data::{prepare_data, Dataset},
    strategy::{backtest, BacktestConfig},
};

// Create model
let config = LRPConfig::default()
    .with_epsilon(0.01)
    .with_gamma(0.25);

let model = LRPNetwork::new(420, vec![128, 64, 32], 2, config);

// Forward pass
let output = model.forward(&input);

// Get explanation
let relevance = model.explain(&input, Some(1));  // Explain class 1

// Backtest
let results = backtest(&model, &test_data, BacktestConfig::default());
println!("Sharpe: {:.2}", results.sharpe_ratio);
```

## LRP Rules

| Rule | Description | Best For |
|------|-------------|----------|
| LRP-0 | Basic rule, no stabilization | Output layers |
| LRP-epsilon | Stabilized with epsilon term | Middle layers |
| LRP-gamma | Emphasizes positive contributions | Input layers |
| LRP-alpha-beta | Separates positive/negative | Fine-grained control |

## Performance

The Rust implementation offers significant performance advantages:

- Zero-copy data handling where possible
- SIMD-optimized ndarray operations
- Async data fetching for parallel downloads
- Memory-efficient batch processing

## License

MIT License
