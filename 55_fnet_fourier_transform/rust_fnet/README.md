# FNet Trading - Rust Implementation

High-performance Rust implementation of FNet (Fourier Neural Network) for cryptocurrency trading.

## Features

- **O(n log n) complexity**: Uses FFT instead of attention for efficient token mixing
- **Bybit API integration**: Fetch cryptocurrency market data
- **Feature engineering**: RSI, Bollinger Bands, momentum, volatility
- **Backtesting engine**: Full simulation with metrics (Sharpe, Sortino, drawdown)
- **Zero-copy where possible**: Efficient memory usage with ndarray

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        FNet Model                            │
├─────────────────────────────────────────────────────────────┤
│  Input: [batch, seq_len, n_features]                        │
│                    │                                         │
│           ┌───────▼───────┐                                 │
│           │ Input Linear  │                                  │
│           │ + Positional  │                                  │
│           │   Encoding    │                                  │
│           └───────┬───────┘                                 │
│                   │                                          │
│     ┌─────────────▼─────────────┐                           │
│     │   FNet Encoder Block (×N) │                           │
│     │  ┌───────────────────┐    │                           │
│     │  │ Fourier Layer     │    │ ◄── 2D FFT (no params!)   │
│     │  │ (replaces attn)   │    │                           │
│     │  └─────────┬─────────┘    │                           │
│     │  ┌─────────▼─────────┐    │                           │
│     │  │ Feed-Forward +    │    │                           │
│     │  │ Residual + Norm   │    │                           │
│     │  └───────────────────┘    │                           │
│     └─────────────┬─────────────┘                           │
│                   │                                          │
│           ┌───────▼───────┐                                 │
│           │ Output Linear │                                  │
│           └───────┬───────┘                                 │
│                   │                                          │
│  Output: [batch, 1] (prediction)                            │
└─────────────────────────────────────────────────────────────┘
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fnet_trading = { path = "path/to/rust_fnet" }
```

Or build from source:

```bash
cd rust_fnet
cargo build --release
```

## Quick Start

### Fetch Data

```rust
use fnet_trading::{BybitClient, calculate_features, FeatureConfig};

// Fetch cryptocurrency data
let client = BybitClient::new();
let klines = client.fetch_klines("BTCUSDT", "60", 1000)?;

// Calculate trading features
let config = FeatureConfig::default();
let features = calculate_features(&klines, &config);

println!("Got {} data points with 8 features", features.len());
```

### Create Model

```rust
use fnet_trading::FNet;

// Create FNet model
let model = FNet::new(
    8,    // n_features (input dimension)
    256,  // d_model (hidden dimension)
    4,    // n_layers (encoder blocks)
    1024, // d_ff (feed-forward dimension)
    512,  // max_seq_len
);

println!("Model has {} parameters", model.num_parameters());
```

### Run Backtest

```rust
use fnet_trading::{
    Backtester, BacktesterConfig, SignalGeneratorConfig
};

// Configure backtester
let backtester = Backtester::new(BacktesterConfig {
    initial_capital: 100_000.0,
    transaction_cost: 0.001,
    slippage: 0.0005,
});

// Configure signal generation
let signal_config = SignalGeneratorConfig {
    threshold: 0.001,
    confidence_threshold: 0.4,
    stop_loss: 0.02,
    take_profit: 0.04,
    ..Default::default()
};

// Run backtest
let result = backtester.run(&predictions, &prices, &timestamps, signal_config);

// Print results
println!("Total Return: {:.2}%", result.metrics.total_return * 100.0);
println!("Sharpe Ratio: {:.2}", result.metrics.sharpe_ratio);
println!("Max Drawdown: {:.2}%", result.metrics.max_drawdown * 100.0);
```

## Examples

Run the examples:

```bash
# Fetch data from Bybit
cargo run --example fetch_data -- --symbol BTCUSDT --interval 60

# Model demonstration
cargo run --example model_demo

# Full backtest (with synthetic data)
cargo run --example backtest -- --synthetic

# Full backtest (with real data)
cargo run --example backtest -- --symbol ETHUSDT --limit 2000
```

## Module Structure

```text
src/
├── lib.rs              # Main exports
├── api/
│   ├── mod.rs
│   ├── client.rs       # Bybit API client
│   └── types.rs        # API response types
├── data/
│   ├── mod.rs
│   ├── features.rs     # Feature engineering
│   └── dataset.rs      # Dataset utilities
├── model/
│   ├── mod.rs
│   ├── fourier.rs      # Fourier Transform layer
│   ├── encoder.rs      # FNet encoder block
│   └── fnet.rs         # Complete FNet model
└── strategy/
    ├── mod.rs
    ├── signals.rs      # Signal generation
    └── backtest.rs     # Backtesting engine
```

## Features

### Trading Features (8 dimensions)

| Feature | Description |
|---------|-------------|
| log_return | Log price returns |
| volatility | Rolling standard deviation of returns |
| volume_ratio | Volume relative to moving average |
| momentum_short | 5-period momentum |
| momentum_medium | 10-period momentum |
| momentum_long | 20-period momentum |
| rsi_normalized | RSI normalized to [-1, 1] |
| bb_position | Position within Bollinger Bands |

### Backtest Metrics

- **Total Return**: Overall strategy return
- **Annual Return**: Annualized return
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss

## Performance

FNet advantages over Transformer:

| Aspect | Transformer | FNet |
|--------|-------------|------|
| Token mixing complexity | O(n²) | O(n log n) |
| Learnable parameters | Many (Q,K,V) | Zero (FFT) |
| GPU memory | High | Low |
| Training speed | 1x | 5x faster |
| Accuracy | Baseline | ~92-97% of baseline |

## Testing

Run tests:

```bash
# All tests
cargo test

# With output
cargo test -- --nocapture

# Specific module
cargo test data::
cargo test model::
cargo test strategy::
```

## Dependencies

- `ndarray`: N-dimensional arrays
- `rustfft`: Fast Fourier Transform
- `reqwest`: HTTP client (blocking)
- `serde`/`serde_json`: JSON serialization
- `clap`: Command-line parsing
- `anyhow`: Error handling

## License

MIT License

## References

- [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824)
- [Bybit API Documentation](https://bybit-exchange.github.io/docs/)
