# Linformer Rust Implementation

High-performance Rust implementation of Linformer for financial time series analysis.

## Features

- **Linear Complexity Attention**: O(n×k) instead of O(n²)
- **Memory Efficient**: Up to 98% memory reduction for long sequences
- **Bybit API Integration**: Fetch real-time cryptocurrency data
- **Technical Indicators**: RSI, MACD, Bollinger Bands, volatility
- **Backtesting Framework**: Complete strategy evaluation

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
linformer = { path = "path/to/linformer" }
```

## Quick Start

```rust
use linformer::prelude::*;

// Create model
let config = LinformerConfig::new(128, 4, 512, 64, 4)
    .with_n_features(6)
    .with_n_outputs(1);

let model = Linformer::new(config)?;

// Print model info
println!("{}", model.summary());
```

## Examples

### Fetch Data from Bybit

```bash
cargo run --example fetch_data
```

### Train Model

```bash
cargo run --example train
```

### Backtest Strategy

```bash
cargo run --example backtest
```

## Project Structure

```
rust/
├── src/
│   ├── lib.rs           # Main library exports
│   ├── api/             # Exchange API clients
│   │   ├── client.rs    # Bybit HTTP client
│   │   └── types.rs     # API response types
│   ├── data/            # Data processing
│   │   ├── loader.rs    # Data loading utilities
│   │   ├── features.rs  # Technical indicators
│   │   └── sequence.rs  # Sequence dataset
│   ├── model/           # Linformer implementation
│   │   ├── attention.rs # Linear attention layer
│   │   ├── config.rs    # Model configuration
│   │   └── linformer.rs # Full model
│   └── strategy/        # Trading strategy
│       ├── backtest.rs  # Backtesting framework
│       └── metrics.rs   # Performance metrics
└── examples/
    ├── fetch_data.rs    # Data fetching example
    ├── train.rs         # Training example
    └── backtest.rs      # Backtesting example
```

## Model Architecture

```
Input [seq_len, n_features]
    ↓
Input Projection [seq_len, d_model]
    ↓
Positional Encoding
    ↓
┌─────────────────────────────────┐
│  Linformer Encoder Layer (×N)  │
│  ├─ Linformer Attention        │
│  │  └─ Linear Projection E, F  │
│  ├─ Residual + LayerNorm       │
│  ├─ Feed-Forward Network       │
│  └─ Residual + LayerNorm       │
└─────────────────────────────────┘
    ↓
Output Projection [n_outputs]
```

## Performance

| Sequence Length | Standard Transformer | Linformer (k=64) | Memory Reduction |
|-----------------|----------------------|------------------|------------------|
| 512             | 262,144              | 32,768           | 87.5%            |
| 1,024           | 1,048,576            | 65,536           | 93.8%            |
| 2,048           | 4,194,304            | 131,072          | 96.9%            |
| 4,096           | 16,777,216           | 262,144          | 98.4%            |

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_attention
```

## Benchmarks

```bash
# Run benchmarks (requires criterion)
cargo bench
```

## Dependencies

- `ndarray` - N-dimensional arrays
- `polars` - DataFrame operations
- `reqwest` - HTTP client
- `tokio` - Async runtime
- `serde` - Serialization

## License

MIT License - See main project LICENSE.

## References

- [Linformer Paper](https://arxiv.org/abs/2006.04768)
- [Machine Learning for Trading](https://github.com/suenot/machine-learning-for-trading)
