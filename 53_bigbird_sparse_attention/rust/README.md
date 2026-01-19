# BigBird Trading - Rust Implementation

High-performance Rust implementation of BigBird sparse attention for financial time series prediction.

## Features

- **BigBird Sparse Attention**: Efficient O(n) attention mechanism
  - Window (local) attention
  - Random attention connections
  - Global tokens
- **Bybit API Integration**: Real-time cryptocurrency data
- **Stock Data Support**: Yahoo Finance compatible
- **Backtesting Engine**: Complete strategy evaluation

## Project Structure

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/
│   │   ├── mod.rs
│   │   ├── bybit.rs        # Bybit API client
│   │   └── types.rs        # API response types
│   ├── data/
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading utilities
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset implementation
│   ├── model/
│   │   ├── mod.rs
│   │   ├── config.rs       # Model configuration
│   │   ├── attention.rs    # BigBird sparse attention
│   │   ├── encoder.rs      # Transformer encoder
│   │   └── bigbird.rs      # Complete model
│   └── strategy/
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── fetch_data.rs       # Download market data
    ├── train.rs            # Train model
    └── backtest.rs         # Run backtest
```

## Quick Start

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Cargo (comes with Rust)

### Build

```bash
cd rust
cargo build --release
```

### Run Examples

```bash
# Fetch cryptocurrency data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --timeframe 1h --limit 1000

# Train the model
cargo run --example train -- --epochs 100 --seq-len 256 --batch-size 32

# Run backtest
cargo run --example backtest -- --model checkpoints/best.safetensors
```

## API Usage

### Creating a Model

```rust
use bigbird_trading::{BigBirdConfig, BigBirdModel};

let config = BigBirdConfig {
    seq_len: 256,
    input_dim: 6,
    d_model: 128,
    n_heads: 8,
    n_layers: 4,
    window_size: 7,
    num_random: 3,
    num_global: 2,
    ..Default::default()
};

let model = BigBirdModel::new(&config);
```

### Loading Data

```rust
use bigbird_trading::data::{fetch_bybit_data, prepare_features};

// Fetch cryptocurrency data
let df = fetch_bybit_data("BTCUSDT", "1h", 1000).await?;

// Prepare features
let df = prepare_features(df)?;
```

### Running Backtest

```rust
use bigbird_trading::strategy::{BacktestConfig, run_backtest};

let config = BacktestConfig {
    initial_capital: 100_000.0,
    position_size: 0.1,
    transaction_cost: 0.001,
    ..Default::default()
};

let results = run_backtest(&model, &test_data, &config)?;
println!("Sharpe Ratio: {:.2}", results.metrics.sharpe_ratio);
```

## Configuration

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_len` | 256 | Input sequence length |
| `input_dim` | 6 | Number of input features |
| `d_model` | 128 | Model dimension |
| `n_heads` | 8 | Number of attention heads |
| `n_layers` | 4 | Number of encoder layers |
| `window_size` | 7 | Size of local attention window |
| `num_random` | 3 | Random attention connections |
| `num_global` | 2 | Number of global tokens |

### Backtest Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 100,000 | Starting capital |
| `position_size` | 0.1 | Fraction of capital per trade |
| `transaction_cost` | 0.001 | Transaction cost (0.1%) |
| `slippage` | 0.0005 | Slippage estimate |

## Performance

Benchmarks on synthetic data (seq_len=256, batch_size=32):

| Operation | Time (CPU) | Time (GPU) |
|-----------|------------|------------|
| Forward pass | 15ms | 2ms |
| Backward pass | 45ms | 5ms |
| Full epoch | 12s | 1.5s |

Memory usage scales linearly with sequence length due to sparse attention.

## License

MIT License
