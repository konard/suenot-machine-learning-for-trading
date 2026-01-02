# Graph Pooling for Trading - Rust Implementation

High-performance Rust implementation of hierarchical graph pooling for cryptocurrency trading using Bybit exchange data.

## Features

- **Bybit API Client**: Real-time market data from Bybit cryptocurrency exchange
- **Graph Construction**: Build correlation graphs from price data
- **Graph Pooling Algorithms**:
  - DiffPool (Differentiable Pooling)
  - SAGPool (Self-Attention Graph Pooling)
  - MinCutPool (Spectral Clustering inspired)
- **Trading Strategy**: Cluster-relative momentum with regime detection
- **Backtesting**: Performance evaluation framework

## Project Structure

```
rust/
├── Cargo.toml              # Project configuration
├── README.md               # This file
└── src/
    ├── lib.rs              # Library entry point
    ├── api/                # Bybit API client
    │   ├── mod.rs
    │   ├── client.rs       # HTTP client implementation
    │   └── types.rs        # API data types
    ├── graph/              # Graph structures
    │   ├── mod.rs
    │   ├── market_graph.rs # Market graph representation
    │   └── correlation.rs  # Correlation computation
    ├── pooling/            # Graph pooling algorithms
    │   ├── mod.rs
    │   ├── diffpool.rs     # DiffPool implementation
    │   ├── sagpool.rs      # SAGPool implementation
    │   └── mincut.rs       # MinCutPool implementation
    ├── strategy/           # Trading strategy
    │   ├── mod.rs
    │   └── hierarchical.rs # Hierarchical pooling strategy
    └── bin/                # Example binaries
        ├── demo.rs         # Live demo with Bybit data
        └── backtest.rs     # Backtesting example
```

## Quick Start

### Prerequisites

- Rust 1.70+ (install from https://rustup.rs)
- OpenBLAS (for linear algebra)

On Ubuntu/Debian:
```bash
sudo apt-get install libopenblas-dev
```

On macOS:
```bash
brew install openblas
```

### Build

```bash
cd rust
cargo build --release
```

### Run Demo

Fetch live data from Bybit and generate trading signals:

```bash
cargo run --release --bin demo
```

### Run Backtest

Run backtesting simulation:

```bash
cargo run --release --bin backtest
```

## Usage Example

```rust
use graph_pooling_trading::prelude::*;
use graph_pooling_trading::StrategyConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize Bybit client
    let client = BybitClient::new(None, None);

    // Fetch market data
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let klines = client.get_klines_batch(&symbols, "60", 100).await?;

    // Build market graph
    let graph = MarketGraph::from_klines(&klines)?;
    println!("{}", graph.summary());

    // Initialize strategy
    let config = StrategyConfig {
        n_clusters_l1: 5,
        n_clusters_l2: 2,
        ..Default::default()
    };
    let mut strategy = GraphPoolingStrategy::with_config(config);

    // Generate trading signals
    let signals = strategy.generate_signals(&graph)?;

    for signal in &signals {
        println!(
            "{}: position={:.4}, cluster={}, regime={}",
            signal.symbol, signal.position, signal.cluster, signal.regime
        );
    }

    Ok(())
}
```

## API Reference

### BybitClient

```rust
// Public endpoints (no API key required)
let client = BybitClient::new(None, None);

// Get klines (candlesticks)
let klines = client.get_klines("BTCUSDT", "60", 100).await?;

// Get multiple symbols
let batch = client.get_klines_batch(&["BTCUSDT", "ETHUSDT"], "60", 100).await?;

// Get ticker
let ticker = client.get_ticker("BTCUSDT").await?;

// Get order book
let orderbook = client.get_orderbook("BTCUSDT", 50).await?;
```

### MarketGraph

```rust
// Build from klines
let graph = MarketGraph::from_klines(&klines)?;

// With custom correlation threshold
let graph = MarketGraph::from_klines_with_threshold(&klines, 0.5)?;

// Get graph statistics
println!("Assets: {}", graph.n_assets());
println!("Hub assets: {:?}", graph.hub_assets(3));
println!("Clusters: {:?}", graph.find_clusters(0.6));
```

### GraphPoolingStrategy

```rust
// Default configuration
let mut strategy = GraphPoolingStrategy::new();

// Custom configuration
let config = StrategyConfig {
    n_clusters_l1: 10,     // First level clusters
    n_clusters_l2: 3,      // Second level clusters
    prediction_weight: 0.5, // Weight for model predictions
    momentum_weight: 0.5,   // Weight for momentum signals
    max_position: 0.1,      // Maximum position per asset
    ..Default::default()
};
let mut strategy = GraphPoolingStrategy::with_config(config);

// Generate signals
let signals = strategy.generate_signals(&graph)?;

// Check regime
println!("Regime: {}", strategy.current_regime());
```

### Trading Signals

```rust
pub struct Signal {
    pub symbol: String,           // Asset symbol
    pub position: f64,            // Position size (-1 to 1)
    pub confidence: f64,          // Confidence (0 to 1)
    pub cluster: usize,           // Cluster assignment
    pub relative_momentum: f64,   // Momentum vs cluster
    pub regime: Regime,           // Market regime
}

pub enum Regime {
    Normal,              // Normal market conditions
    CorrelationBreakdown, // All assets moving together (panic)
    ClusterFormation,    // Clusters becoming distinct
}
```

## Performance

The Rust implementation provides:
- **Fast correlation computation**: Optimized matrix operations with ndarray
- **Efficient graph pooling**: O(n²) complexity for n assets
- **Low-latency API calls**: Async HTTP with connection pooling
- **Memory efficiency**: Zero-copy data structures where possible

Typical performance on modern hardware:
- Graph construction (20 assets, 100 candles): ~5ms
- Hierarchical pooling: ~2ms
- Signal generation: ~1ms

## Testing

Run tests:
```bash
cargo test
```

Run with logging:
```bash
RUST_LOG=debug cargo test
```

## Dependencies

- `tokio`: Async runtime
- `reqwest`: HTTP client
- `ndarray`: N-dimensional arrays
- `serde`: Serialization
- `tracing`: Logging

## License

MIT License

## References

- [Hierarchical Graph Representation Learning with DiffPool](https://arxiv.org/abs/1806.08804)
- [Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082)
- [Bybit API Documentation](https://bybit-exchange.github.io/docs/)
