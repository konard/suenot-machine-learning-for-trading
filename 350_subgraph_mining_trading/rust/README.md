# Subgraph Mining for Trading - Rust Implementation

High-performance Rust library for discovering and trading on subgraph patterns in cryptocurrency markets.

## Features

- **Graph Construction**: Build correlation networks from market data
- **Subgraph Mining**: Discover frequent patterns using gSpan algorithm
- **Pattern Matching**: Find triangles, stars, cliques, and chains
- **Signal Generation**: Convert patterns to trading signals
- **Backtesting**: Validate strategies on historical data
- **Bybit Integration**: Real-time cryptocurrency data

## Quick Start

```bash
# Build the project
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example basic_mining
cargo run --example bybit_live
cargo run --example backtest
```

## Project Structure

```
src/
├── lib.rs              # Library entry point
├── graph/              # Graph types and algorithms
│   ├── types.rs        # Node, Edge, FinancialGraph
│   ├── builder.rs      # Graph construction
│   └── algorithms.rs   # Pattern finding (triangles, cliques, etc.)
├── mining/             # Subgraph mining
│   ├── gspan.rs        # gSpan algorithm
│   ├── pattern.rs      # Pattern representation
│   └── matcher.rs      # Pattern matching
├── trading/            # Trading logic
│   ├── signals.rs      # Signal generation
│   ├── strategy.rs     # Trading strategy
│   └── backtest.rs     # Backtesting engine
├── data/               # Market data
│   ├── bybit.rs        # Bybit API client
│   ├── types.rs        # Candle, Ticker, etc.
│   └── cache.rs        # Data caching
└── utils/              # Utilities
    ├── stats.rs        # Statistical functions
    └── config.rs       # Configuration
```

## Usage Examples

### Build Correlation Network

```rust
use subgraph_mining_trading::{
    data::BybitClient,
    graph::GraphBuilder,
};

#[tokio::main]
async fn main() {
    let client = BybitClient::new();
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let data = client.fetch_candles_multi(&symbols, "1h", 100).await?;

    let graph = GraphBuilder::new()
        .correlation_threshold(0.7)
        .window_size(24)
        .build_correlation_network(&data)?;

    println!("Nodes: {}, Edges: {}", graph.node_count(), graph.edge_count());
}
```

### Mine Patterns

```rust
use subgraph_mining_trading::mining::GSpanMiner;

let miner = GSpanMiner::new()
    .min_support(2)
    .max_size(5);

let patterns = miner.mine(&graph)?;

for pattern in patterns {
    println!("{:?}: support={}", pattern.pattern_type, pattern.support);
}
```

### Generate Trading Signals

```rust
use subgraph_mining_trading::trading::SignalGenerator;

let mut signal_gen = SignalGenerator::new(patterns);
let signals = signal_gen.generate(&graph);

for signal in signals {
    println!("{:?}: {} - {}",
        signal.signal_type,
        signal.symbols.join(","),
        signal.reason
    );
}
```

### Run Backtest

```rust
use subgraph_mining_trading::trading::{BacktestConfig, Backtester};

let config = BacktestConfig::default();
let backtester = Backtester::new(config);
let result = backtester.run(&historical_data)?;

println!("Return: {:.2}%", result.total_return * 100.0);
println!("Sharpe: {:.2}", result.sharpe_ratio);
println!("Max DD: {:.2}%", result.max_drawdown * 100.0);
```

## Supported Patterns

| Pattern | Description | Trading Signal |
|---------|-------------|----------------|
| Triangle | 3 connected assets | Strong sector correlation |
| Star | Hub with multiple spokes | Market leader influence |
| Chain | Linear path | Information cascade |
| Clique | Fully connected group | Strong regime |

## Dependencies

- `petgraph` - Graph data structures
- `tokio` - Async runtime
- `reqwest` - HTTP client
- `serde` - Serialization
- `nalgebra` - Linear algebra
- `chrono` - Time handling

## Performance

The library is optimized for:
- Efficient graph operations using petgraph
- Async I/O for API calls
- Memory-efficient pattern mining
- Parallel processing where applicable

## Configuration

Create a `config.json`:

```json
{
  "data": {
    "api_url": "https://api.bybit.com",
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "timeframe": "1h",
    "cache_ttl": 300
  },
  "graph": {
    "correlation_threshold": 0.7,
    "window_size": 24,
    "min_volume": 0
  },
  "mining": {
    "min_support": 2,
    "max_size": 6,
    "max_patterns": 100
  },
  "trading": {
    "initial_capital": 10000,
    "max_position_size": 0.1,
    "stop_loss": 0.05,
    "take_profit": 0.10
  }
}
```

## License

MIT License
