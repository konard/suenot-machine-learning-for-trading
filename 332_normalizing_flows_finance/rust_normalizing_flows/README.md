# Rust Normalizing Flows for Finance

Rust implementation of normalizing flow models for financial risk management and density estimation.

## Features

- **RealNVP Flow**: Affine coupling-based normalizing flow
- **Bybit API Client**: Fetch cryptocurrency market data
- **Risk Metrics**: VaR, CVaR, tail probabilities
- **Synthetic Data Generation**: Generate realistic return scenarios
- **Stress Testing**: Test portfolios under extreme conditions

## Quick Start

```rust
use normalizing_flows_finance::prelude::*;

fn main() -> anyhow::Result<()> {
    // Fetch market data
    let client = BybitClient::new();
    let klines = client.get_klines_sync("BTCUSDT", "D", 500)?;

    // Calculate returns
    let returns = calculate_log_returns(&klines);

    // Create and train flow
    let config = FlowConfig::new(1).with_n_layers(6);
    let mut flow = RealNVP::new(config);
    flow.train(&returns, 100)?;

    // Compute risk metrics
    let var_95 = compute_var(&flow, 0.05, 100000);
    let (_, cvar_95) = compute_cvar(&flow, 0.05, 100000);

    println!("95% VaR: {:.4}", var_95);
    println!("95% CVaR: {:.4}", cvar_95);

    Ok(())
}
```

## Examples

Run the examples with:

```bash
# Fetch cryptocurrency data
cargo run --example fetch_data

# Train a normalizing flow
cargo run --example train_flow

# Compute VaR and CVaR
cargo run --example compute_var

# Generate synthetic scenarios
cargo run --example synthetic_generation
```

## Project Structure

```
rust_normalizing_flows/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs           # Library entry point
│   ├── api/             # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs    # REST API client
│   │   └── types.rs     # Data types
│   ├── flows/           # Normalizing flow implementations
│   │   ├── mod.rs
│   │   ├── config.rs    # Model configuration
│   │   ├── layers.rs    # Neural network layers
│   │   ├── realnvp.rs   # RealNVP implementation
│   │   └── traits.rs    # Common traits
│   ├── risk/            # Risk metrics
│   │   ├── mod.rs
│   │   └── metrics.rs   # VaR, CVaR calculations
│   └── utils/           # Utilities
│       ├── mod.rs
│       ├── returns.rs   # Return calculations
│       └── statistics.rs # Statistical functions
└── examples/
    ├── fetch_data.rs
    ├── train_flow.rs
    ├── compute_var.rs
    └── synthetic_generation.rs
```

## Dependencies

- `ndarray` - N-dimensional arrays
- `reqwest` - HTTP client for API
- `serde` - Serialization
- `rand` - Random number generation
- `statrs` - Statistical distributions

## License

MIT
