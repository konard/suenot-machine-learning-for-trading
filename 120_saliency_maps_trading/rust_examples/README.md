# Saliency Maps Trading - Rust Implementation

This directory contains a Rust implementation of saliency map computation for cryptocurrency trading.

## Features

- **Bybit API Client**: Fetch OHLCV data from Bybit exchange
- **Neural Network**: Simple feedforward network for price prediction
- **Saliency Computation**: Multiple gradient-based saliency methods
- **Trading Strategy**: Saliency-filtered signal generation
- **Backtesting**: Performance evaluation framework

## Project Structure

```
rust_examples/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Library entry point
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs        # Bybit API client
│   ├── data/
│   │   ├── mod.rs
│   │   ├── processor.rs    # Data preprocessing
│   │   └── features.rs     # Technical indicators
│   ├── models/
│   │   ├── mod.rs
│   │   └── network.rs      # Neural network
│   └── saliency/
│       ├── mod.rs
│       └── gradient.rs     # Saliency computation
└── examples/
    ├── fetch_data.rs       # Data fetching example
    ├── saliency_trading.rs # Saliency analysis example
    └── backtest.rs         # Backtesting example
```

## Getting Started

### Prerequisites

- Rust 1.70 or later
- Cargo package manager

### Building

```bash
cargo build --release
```

### Running Examples

#### Fetch Market Data
```bash
cargo run --example fetch_data
```

#### Compute Saliency Maps
```bash
cargo run --example saliency_trading
```

#### Run Backtest
```bash
cargo run --example backtest
```

## API Usage

### Creating a Trading Network

```rust
use saliency_maps_trading::models::network::TradingNetwork;

let network = TradingNetwork::new(
    30,           // sequence length
    5,            // number of features (OHLCV)
    &[64, 32]     // hidden layer sizes
);
```

### Computing Saliency Maps

```rust
use saliency_maps_trading::saliency::gradient::SaliencyComputer;
use ndarray::Array2;

let computer = SaliencyComputer::new(network);
let input = Array2::from_shape_fn((30, 5), |_| 0.5);

// Vanilla gradient
let vanilla = computer.vanilla_gradient(&input);

// Gradient × Input
let grad_input = computer.gradient_x_input(&input);

// Integrated gradients
let integrated = computer.integrated_gradients(&input, 50);

// SmoothGrad
let smooth = computer.smoothgrad(&input, 0.1, 20);
```

### Generating Trading Signals

```rust
use saliency_maps_trading::saliency::gradient::SaliencyTrader;

let trader = SaliencyTrader::new(
    network,
    0.55,          // minimum confidence
    0.3,           // minimum concentration
    vec![3, 4],    // interpretable features (Close, Volume)
);

let (signal, position_size) = trader.generate_signal(&input);
// signal: 1 (long), -1 (short), or 0 (no trade)
```

### Fetching Data from Bybit

```rust
use saliency_maps_trading::api::bybit::{BybitClient, Interval};

let client = BybitClient::new();
let klines = client.get_klines(
    "BTCUSDT",
    Interval::Hour1,
    Some(100),
    None,
    None
)?;
```

## Saliency Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Vanilla Gradient | Basic sensitivity analysis | Quick overview |
| Gradient × Input | Scales by input magnitude | Better attribution |
| Integrated Gradients | Path-based attribution | Theoretical guarantees |
| SmoothGrad | Noise-averaged gradients | Cleaner visualizations |

## Performance Considerations

- The Rust implementation uses numerical gradients, which are slower than autodiff
- For production use, consider using a deep learning framework with automatic differentiation
- The current implementation is optimized for educational purposes

## Dependencies

- `reqwest` - HTTP client
- `serde` - Serialization
- `ndarray` - N-dimensional arrays
- `chrono` - Date/time handling
- `rand` - Random number generation

## License

MIT License
