# GNN Crypto - Graph Neural Networks for Cryptocurrency Trading

Rust implementation of Graph Neural Networks for analyzing cryptocurrency relationships and building trading strategies based on momentum propagation.

## Features

- **Bybit API Client**: Fetch historical OHLCV data for multiple cryptocurrencies
- **Graph Construction**: Build correlation graphs, k-NN graphs, and sector-based graphs
- **GNN Models**: GCN, GAT implementations using tch-rs (PyTorch bindings)
- **Trading Strategy**: Momentum propagation and lead-lag exploitation
- **Backtesting**: Historical strategy evaluation

## Project Structure

```
rust_gnn_crypto/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Public API
│   ├── main.rs             # CLI application
│   ├── data/               # Data fetching and preprocessing
│   │   ├── mod.rs
│   │   ├── bybit_client.rs # Bybit API client
│   │   ├── ohlcv.rs        # OHLCV data structures
│   │   └── features.rs     # Feature engineering
│   ├── graph/              # Graph construction
│   │   ├── mod.rs
│   │   ├── correlation.rs  # Correlation-based graphs
│   │   ├── knn.rs          # k-NN graphs
│   │   └── temporal.rs     # Time-evolving graphs
│   ├── model/              # GNN models
│   │   ├── mod.rs
│   │   ├── gcn.rs          # Graph Convolutional Network
│   │   ├── gat.rs          # Graph Attention Network
│   │   └── layers.rs       # Common graph layers
│   ├── strategy/           # Trading strategies
│   │   ├── mod.rs
│   │   └── momentum.rs     # Momentum propagation strategy
│   └── utils/              # Utilities
│       ├── mod.rs
│       └── config.rs       # Configuration
├── examples/
│   ├── fetch_data.rs       # Fetch data from Bybit
│   ├── build_graph.rs      # Build crypto graph
│   ├── train_gnn.rs        # Train GNN model
│   └── backtest.rs         # Backtest trading strategy
└── data/                   # Local data storage
    └── .gitkeep
```

## Quick Start

### 1. Fetch Data

```bash
cargo run --example fetch_data -- \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT,AVAXUSDT,MATICUSDT \
    --interval 60 \
    --days 90
```

### 2. Build Graph

```bash
cargo run --example build_graph -- \
    --method correlation \
    --threshold 0.5 \
    --window 60
```

### 3. Train Model

```bash
cargo run --example train_gnn -- \
    --model gcn \
    --hidden-dim 64 \
    --epochs 100 \
    --lr 0.001
```

### 4. Backtest Strategy

```bash
cargo run --example backtest -- \
    --model checkpoints/best_model.pt \
    --threshold 0.6 \
    --initial-capital 100000
```

## Usage as Library

```rust
use gnn_crypto::{
    data::BybitClient,
    graph::CorrelationGraph,
    model::{GCN, GNNConfig},
    strategy::MomentumStrategy,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fetch data
    let client = BybitClient::new();
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let data = client.fetch_multi_symbol(&symbols, "60", 90).await?;

    // Build graph
    let graph = CorrelationGraph::new(0.5, 60);
    let crypto_graph = graph.build(&data)?;

    // Create and train model
    let config = GNNConfig {
        num_features: 10,
        hidden_dim: 64,
        num_classes: 3,
        num_layers: 3,
        dropout: 0.3,
    };
    let model = GCN::new(&config, tch::Device::Cpu);

    // Use for trading
    let strategy = MomentumStrategy::new(model, 0.6);
    let signals = strategy.generate_signals(&crypto_graph)?;

    Ok(())
}
```

## Configuration

Create a `config.toml` file:

```toml
[data]
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT"]
interval = "60"  # 1 hour candles
history_days = 90

[graph]
method = "correlation"  # correlation, knn, sector
threshold = 0.5
window = 60
update_frequency = "daily"

[model]
architecture = "gcn"  # gcn, gat
hidden_dim = 64
num_layers = 3
dropout = 0.3
learning_rate = 0.001
epochs = 100

[strategy]
confidence_threshold = 0.6
position_size = 0.1  # 10% of capital per position
max_positions = 5
stop_loss = 0.05  # 5%
take_profit = 0.10  # 10%

[backtest]
initial_capital = 100000.0
transaction_cost = 0.001  # 0.1%
```

## Examples

### Correlation Analysis

```rust
use gnn_crypto::graph::analyze_correlations;

let correlations = analyze_correlations(&returns_data, 60)?;
println!("Strongest correlations:");
for (pair, corr) in correlations.iter().take(10) {
    println!("  {} <-> {}: {:.3}", pair.0, pair.1, corr);
}
```

### Lead-Lag Detection

```rust
use gnn_crypto::graph::detect_lead_lag;

let lead_lag_pairs = detect_lead_lag(&returns_data, 5)?;
for pair in lead_lag_pairs {
    println!(
        "{} leads {} by {} periods (p-value: {:.4})",
        pair.leader, pair.lagger, pair.lag, pair.pvalue
    );
}
```

### Feature Engineering

```rust
use gnn_crypto::data::FeatureEngineer;

let engineer = FeatureEngineer::new();
let features = engineer.compute_all_features(&ohlcv_data)?;

// Features include:
// - Momentum (1d, 7d, 30d returns)
// - Volatility (realized, ATR)
// - RSI, MACD
// - Volume anomaly
// - Price position (relative to 52-week range)
```

## Model Architectures

### Graph Convolutional Network (GCN)

Simple spectral convolution over neighbors:

```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

Best for: Small graphs with stable structure.

### Graph Attention Network (GAT)

Attention-weighted neighbor aggregation:

```
α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
h'_i = σ(Σ_j α_ij W h_j)
```

Best for: When different neighbors have different importance.

## Performance

Benchmarks on 50-coin crypto graph (M1 MacBook Pro):

| Operation | Time |
|-----------|------|
| Data fetch (90 days, 50 coins) | ~30s |
| Graph construction | ~100ms |
| Forward pass (GCN) | ~5ms |
| Forward pass (GAT) | ~15ms |
| Full training (100 epochs) | ~2min |

## Requirements

- Rust 1.70+
- LibTorch (automatically downloaded by tch-rs)
- Network access for Bybit API

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_correlation_graph

# Run with logging
RUST_LOG=debug cargo test
```

## License

MIT

## References

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [tch-rs Documentation](https://docs.rs/tch/)
