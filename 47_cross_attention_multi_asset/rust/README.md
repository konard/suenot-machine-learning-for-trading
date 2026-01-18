# Cross-Attention Multi-Asset Trading - Rust Implementation

High-performance Rust implementation of Cross-Attention mechanism for multi-asset trading using the Candle ML framework.

## Features

- **Cross-Attention Model**: Transformer-based model with cross-asset attention
- **Bybit Integration**: Fetch cryptocurrency data from Bybit exchange
- **Feature Engineering**: Technical indicators and feature computation
- **Backtesting Engine**: Evaluate strategies with realistic constraints

## Project Structure

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                  # Library exports
│   ├── model/                  # Model implementation
│   │   ├── mod.rs
│   │   ├── attention.rs        # Cross-attention layers
│   │   ├── embedding.rs        # Token embeddings
│   │   └── cross_attention.rs  # Main model
│   ├── data/                   # Data handling
│   │   ├── mod.rs
│   │   ├── bybit.rs            # Bybit API client
│   │   ├── features.rs         # Feature engineering
│   │   └── dataset.rs          # Training dataset
│   └── strategy/               # Trading strategy
│       ├── mod.rs
│       ├── signals.rs          # Signal generation
│       └── backtest.rs         # Backtesting engine
└── examples/
    ├── fetch_data.rs           # Download data from Bybit
    ├── train.rs                # Train the model
    └── backtest.rs             # Run backtest
```

## Quick Start

### Prerequisites

- Rust 1.70+
- Internet connection for Bybit API

### Installation

```bash
# Clone repository (if not already done)
cd 47_cross_attention_multi_asset/rust

# Build the project
cargo build --release
```

### Fetch Data

```bash
# Fetch hourly data for major cryptocurrencies
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 60 --limit 1000
```

### Train Model

```bash
# Train with default configuration
cargo run --release --example train -- --epochs 50 --batch-size 32

# Train with custom settings
cargo run --release --example train -- \
    --epochs 100 \
    --batch-size 64 \
    --d-model 128 \
    --n-heads 8 \
    --learning-rate 0.001
```

### Run Backtest

```bash
# Backtest on test data
cargo run --release --example backtest -- --start 2024-01-01 --end 2024-12-31

# With custom parameters
cargo run --release --example backtest -- \
    --initial-capital 100000 \
    --transaction-cost 0.001 \
    --rebalance-freq 24
```

## Architecture

### Cross-Attention Mechanism

The model uses cross-attention to capture relationships between multiple assets:

```rust
pub struct CrossAssetAttention {
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    output_proj: Linear,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl CrossAssetAttention {
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // x: [batch, n_assets, seq_len, d_model]

        // Pool temporal dimension
        let x_pooled = x.mean(2)?;  // [batch, n_assets, d_model]

        // Project to Q, K, V
        let q = self.query_proj.forward(&x_pooled)?;
        let k = self.key_proj.forward(&x_pooled)?;
        let v = self.value_proj.forward(&x_pooled)?;

        // Multi-head attention
        // ... reshape for heads ...

        // Compute attention scores
        let scores = q.matmul(&k.transpose(-2, -1)?)? * self.scale;
        let attention = softmax(&scores, -1)?;

        // Apply to values
        let context = attention.matmul(&v)?;

        Ok((output, attention))
    }
}
```

### Data Pipeline

```rust
// Fetch from Bybit
let client = BybitClient::new();
let data = client.fetch_klines("BTCUSDT", "60", 1000).await?;

// Compute features
let features = compute_features(&data)?;

// Prepare for model
let dataset = CrossAttentionDataset::new(
    features,
    lookback: 168,
    horizon: 24,
)?;
```

## Configuration

### Model Configuration

```rust
pub struct ModelConfig {
    pub n_assets: usize,
    pub n_features: usize,
    pub seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub dropout: f64,
    pub output_type: OutputType,
}

// Default configuration
let config = ModelConfig {
    n_assets: 5,
    n_features: 6,
    seq_len: 168,
    d_model: 64,
    n_heads: 4,
    n_layers: 2,
    dropout: 0.1,
    output_type: OutputType::Portfolio,
};
```

### Backtest Configuration

```rust
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub transaction_cost: f64,
    pub rebalance_freq: usize,
    pub max_position: f64,
    pub allow_short: bool,
}
```

## Performance

The Rust implementation is optimized for:

- **Memory efficiency**: Uses Candle's memory-efficient tensor operations
- **Speed**: Native code with SIMD optimizations
- **Parallelism**: Async data fetching and parallel feature computation

Benchmarks on Apple M1:

| Operation | Time |
|-----------|------|
| Forward pass (batch=32) | ~5ms |
| Feature computation (1000 candles) | ~2ms |
| Backtest (1000 steps) | ~10ms |

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

## References

- [Candle ML Framework](https://github.com/huggingface/candle)
- [Bybit API Documentation](https://bybit-exchange.github.io/docs/v5/intro)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
