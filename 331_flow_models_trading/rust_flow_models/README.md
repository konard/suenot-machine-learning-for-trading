# Flow Models Trading - Rust Implementation

This crate provides a Rust implementation of Normalizing Flow models for cryptocurrency trading.

## Features

- **Normalizing Flow Model**: RealNVP-style architecture with ActNorm, Affine Coupling, and Permutation layers
- **Bybit API Client**: Fetch market data (OHLCV, order book, trades) from Bybit exchange
- **Feature Engineering**: Technical indicators and market features
- **Anomaly Detection**: Detect unusual market conditions via likelihood estimation
- **Regime Detection**: Cluster latent representations to identify market states
- **Trading Strategy**: Generate signals based on flow model analysis
- **Backtesting Engine**: Test strategies on historical data

## Project Structure

```
rust_flow_models/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                 # Library entry point
│   ├── api/                   # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs          # HTTP client
│   │   └── types.rs           # Data types
│   ├── flow/                  # Flow model implementation
│   │   ├── mod.rs
│   │   ├── config.rs          # Model configuration
│   │   ├── layers.rs          # Flow layers (ActNorm, Coupling, etc.)
│   │   ├── model.rs           # Complete flow model
│   │   ├── anomaly.rs         # Anomaly detection
│   │   └── regime.rs          # Regime detection
│   ├── features/              # Feature engineering
│   │   ├── mod.rs
│   │   ├── engine.rs          # Feature computation
│   │   └── indicators.rs      # Technical indicators
│   ├── strategy/              # Trading strategy
│   │   ├── mod.rs
│   │   ├── signal.rs          # Signal types
│   │   └── flow_strategy.rs   # Flow-based strategy
│   └── backtest/              # Backtesting
│       ├── mod.rs
│       ├── engine.rs          # Backtest engine
│       └── report.rs          # Performance report
└── examples/
    ├── fetch_market_data.rs   # Fetch data from Bybit
    ├── train_flow_model.rs    # Train a flow model
    ├── anomaly_detection.rs   # Detect anomalies
    ├── regime_detection.rs    # Detect market regimes
    ├── backtest.rs            # Run backtest
    └── live_signals.rs        # Generate live signals
```

## Usage

### Build

```bash
cargo build --release
```

### Run Examples

```bash
# Fetch market data from Bybit
cargo run --example fetch_market_data

# Train a flow model
cargo run --example train_flow_model

# Anomaly detection
cargo run --example anomaly_detection

# Regime detection
cargo run --example regime_detection

# Run backtest
cargo run --example backtest

# Generate live signals
cargo run --example live_signals
```

### Use as Library

```rust
use flow_models_trading::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fetch market data
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", "60", 500).await?;

    // Compute features
    let mut engine = FeatureEngine::new();
    let features = engine.compute_features(&klines);

    // Create and train flow model
    let config = FlowConfig::default().with_input_dim(9);
    let mut model = NormalizingFlow::new(config);
    model.train(&features, 50)?;

    // Create trading strategy
    let mut strategy = FlowTradingStrategy::new(model);
    strategy.fit(&features, None);

    // Generate signal
    let signal = strategy.generate_signal(&features.row(features.nrows() - 1).to_owned());
    println!("Signal: {:?}", signal);

    Ok(())
}
```

## Key Concepts

### Normalizing Flows

Normalizing flows learn invertible transformations between complex data distributions and simple base distributions (Gaussian). This enables:

- **Exact likelihood computation**: Know precisely how likely an observation is
- **Perfect reconstruction**: Transform data to latent space and back without loss
- **Anomaly detection**: Low likelihood indicates unusual observations
- **Sample generation**: Generate new samples by transforming noise

### Flow Architecture

```
Input x → [ActNorm] → [Coupling] → [Permutation] → ... → Latent z

Forward:  x → z (compute likelihood)
Inverse:  z → x (generate samples)
```

### Trading Strategy

1. **Anomaly Detection**: Flag unusual market conditions (reduce exposure)
2. **Regime Detection**: Identify current market state (Bull/Bear, High/Low Volatility)
3. **Signal Generation**: Combine regime and confidence for trading decisions

## Dependencies

- `ndarray`: Numerical arrays
- `reqwest`: HTTP client for Bybit API
- `tokio`: Async runtime
- `serde`: Serialization
- `chrono`: Date/time handling
- `rand`: Random number generation

## Disclaimer

This is for **educational purposes only**. Cryptocurrency trading involves substantial risk. The strategies here have not been validated in live trading. Past performance does not guarantee future results.
