# Graph Transformer Trading - Rust Implementation

A high-performance Rust implementation of Graph Transformers for cryptocurrency trading using Bybit exchange data.

## Features

- **Graph Construction**: Build market graphs from cryptocurrency correlations and relationships
- **Graph Transformer**: Neural network with graph attention for structured market data
- **Feature Engineering**: Technical indicators and order book features
- **Signal Generation**: ML-based trading signal generation
- **Backtesting**: Complete backtesting framework with performance metrics
- **Bybit Integration**: Real-time market data from Bybit exchange

## Project Structure

```
rust_graph_transformer/
├── src/
│   ├── lib.rs              # Library entry point
│   ├── api/                # Bybit API client
│   │   ├── client.rs       # HTTP REST client
│   │   ├── types.rs        # Data structures
│   │   └── websocket.rs    # WebSocket client
│   ├── graph/              # Graph data structures
│   │   ├── node.rs         # Crypto node representation
│   │   ├── edge.rs         # Edge types and features
│   │   ├── market_graph.rs # Market graph structure
│   │   └── builder.rs      # Graph construction
│   ├── transformer/        # Graph Transformer model
│   │   ├── config.rs       # Model configuration
│   │   ├── attention.rs    # Graph attention mechanism
│   │   ├── layer.rs        # Transformer layer
│   │   ├── positional.rs   # Positional encoding
│   │   └── model.rs        # Full model
│   ├── features/           # Feature engineering
│   │   ├── indicators.rs   # Technical indicators
│   │   ├── node_features.rs# Node feature extraction
│   │   └── engine.rs       # Feature engine
│   ├── strategy/           # Trading strategy
│   │   ├── signal.rs       # Signal generation
│   │   └── portfolio.rs    # Portfolio construction
│   ├── backtest/           # Backtesting
│   │   ├── engine.rs       # Backtest engine
│   │   ├── metrics.rs      # Performance metrics
│   │   └── report.rs       # Report generation
│   └── utils/              # Utilities
│       ├── math.rs         # Math functions
│       └── time.rs         # Time utilities
└── examples/
    ├── fetch_market_data.rs   # Fetch data from Bybit
    ├── build_market_graph.rs  # Build market graph
    ├── simple_prediction.rs   # Make predictions
    ├── graph_attention.rs     # Visualize attention
    ├── backtest.rs            # Run backtest
    └── live_signals.rs        # Generate live signals
```

## Quick Start

### Prerequisites

- Rust 1.70+ (install from https://rustup.rs/)
- Internet connection (for Bybit API)

### Build

```bash
cd rust_graph_transformer
cargo build --release
```

### Run Examples

```bash
# 1. Fetch market data from Bybit
cargo run --example fetch_market_data

# 2. Build a market graph
cargo run --example build_market_graph

# 3. Make predictions with Graph Transformer
cargo run --example simple_prediction

# 4. Visualize attention weights
cargo run --example graph_attention

# 5. Run a backtest
cargo run --example backtest

# 6. Generate live trading signals
cargo run --example live_signals
```

## Usage

### Building a Market Graph

```rust
use graph_transformer_trading::prelude::*;

// Create graph builder
let builder = MarketGraphBuilder::new()
    .correlation_threshold(0.5)
    .include_sector_edges(true);

// Build from tickers
let graph = builder.build_from_tickers(&tickers);

// Access nodes and edges
println!("Nodes: {}", graph.num_nodes());
println!("Edges: {}", graph.num_edges());
```

### Making Predictions

```rust
use graph_transformer_trading::prelude::*;

// Create model
let config = GraphTransformerConfig::default();
let model = GraphTransformer::new(config);

// Make predictions
let (direction_probs, returns) = model.predict(&features, &graph);

// Get trading signals
let signal_gen = SignalGenerator::new(0.6);
let signals = signal_gen.generate(&model, &features, &graph);
```

### Running a Backtest

```rust
use graph_transformer_trading::prelude::*;

let engine = BacktestEngine::new(BacktestConfig {
    initial_capital: 100000.0,
    trading_fee: 0.001,
    slippage: 0.0005,
    rebalance_frequency: 1,
});

let report = engine.run_simple(&portfolio, &prices);
println!("{}", report.summary());
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `MarketGraph` | Graph of cryptocurrency assets |
| `CryptoNode` | Node representing a cryptocurrency |
| `MarketEdge` | Edge representing asset relationship |
| `GraphTransformer` | Graph Transformer neural network |
| `Signal` | Trading signal (long/short/hold) |
| `BacktestEngine` | Backtesting framework |

### Key Functions

```rust
// Graph construction
MarketGraphBuilder::build_from_tickers(&[Ticker]) -> MarketGraph
MarketGraphBuilder::build_from_klines(&HashMap<String, Vec<Kline>>) -> MarketGraph

// Predictions
GraphTransformer::predict(&Array2<f64>, &MarketGraph) -> (Array2<f64>, Array1<f64>)
SignalGenerator::generate(&GraphTransformer, &Array2<f64>, &MarketGraph) -> Vec<Signal>

// Backtesting
BacktestEngine::run_simple(&Portfolio, &HashMap<String, Vec<Kline>>) -> BacktestReport
```

## Performance Metrics

The backtesting framework provides these metrics:

- **Total Return**: Cumulative return
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss

## Configuration

### Model Configuration

```rust
let config = GraphTransformerConfig {
    input_dim: 32,
    hidden_dim: 64,
    output_dim: 3,
    num_layers: 4,
    num_heads: 4,
    dropout: 0.1,
    edge_dim: Some(8),
    use_edge_features: true,
    positional_encoding: PositionalEncodingType::Laplacian,
    num_pe_dims: 8,
    activation: ActivationType::GELU,
    use_layer_norm: true,
    use_residual: true,
    ff_expansion: 4,
};
```

### Graph Builder Configuration

```rust
let builder_config = GraphBuilderConfig {
    correlation_threshold: 0.5,
    correlation_window: 30,
    include_sector_edges: true,
    include_negative_correlations: true,
    max_edges_per_node: Some(10),
};
```

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run benchmarks
cargo bench
```

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for **educational purposes only**. Cryptocurrency trading involves substantial risk. Do not trade with money you cannot afford to lose. The authors are not responsible for any financial losses.
