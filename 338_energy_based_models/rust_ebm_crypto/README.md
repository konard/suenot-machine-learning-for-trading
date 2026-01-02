# Energy-Based Models for Crypto Trading

A Rust implementation of Energy-Based Models (EBMs) for cryptocurrency trading with Bybit exchange data.

## Features

- **Data Module**: Fetch OHLCV data from Bybit API
- **EBM Module**: Multiple EBM implementations (Neural Net, RBM, Score Matching)
- **Features Module**: Technical indicators and feature engineering
- **Strategy Module**: Trading signals, position management, and backtesting

## Quick Start

```bash
# Build the project
cargo build --release

# Fetch data from Bybit
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 1000 --output data.csv

# Train EBM model
cargo run --bin train_ebm -- --input data.csv --epochs 50

# Detect anomalies
cargo run --bin detect_anomalies -- --symbol BTCUSDT --limit 500

# Run backtest
cargo run --bin backtest -- --symbol BTCUSDT --limit 5000

# Real-time monitoring
cargo run --bin realtime_monitor -- --symbol BTCUSDT --interval 1
```

## Binaries

### fetch_data
Fetch OHLCV data from Bybit API.

```bash
cargo run --bin fetch_data -- \
  --symbol BTCUSDT \
  --interval 60 \
  --limit 1000 \
  --output data.csv
```

### train_ebm
Train Energy-Based Model on market data.

```bash
cargo run --bin train_ebm -- \
  --input data.csv \
  --epochs 50 \
  --model ebm \
  --hidden 64,32,16
```

Model types: `ebm`, `rbm`, `score`

### detect_anomalies
Detect anomalies in market data.

```bash
cargo run --bin detect_anomalies -- \
  --symbol BTCUSDT \
  --limit 500 \
  --threshold 2.0 \
  --online
```

### backtest
Backtest EBM trading strategy.

```bash
cargo run --bin backtest -- \
  --symbol BTCUSDT \
  --limit 5000 \
  --capital 10000 \
  --show-trades
```

### realtime_monitor
Real-time market monitoring with EBM.

```bash
cargo run --bin realtime_monitor -- \
  --symbol BTCUSDT \
  --interval 1 \
  --poll-seconds 10
```

## Library Usage

```rust
use rust_ebm_crypto::data::{BybitClient, OhlcvData};
use rust_ebm_crypto::ebm::{EnergyModel, OnlineEnergyEstimator};
use rust_ebm_crypto::features::FeatureEngine;

// Fetch data
let client = BybitClient::public();
let data = client.get_klines("BTCUSDT", "60", 1000, None, None)?;

// Extract features
let engine = FeatureEngine::default();
let features = engine.compute(&data.data);

// Train EBM
let mut model = EnergyModel::new(features.ncols());
model.train(&features, 50);

// Detect anomalies
let anomalies = model.detect_anomalies(&features, 0.9);
```

## Module Structure

```
rust_ebm_crypto/
├── src/
│   ├── lib.rs              # Library entry point
│   ├── data/
│   │   ├── mod.rs          # Data module
│   │   ├── bybit.rs        # Bybit API client
│   │   ├── ohlcv.rs        # OHLCV data structures
│   │   └── normalize.rs    # Data normalization
│   ├── ebm/
│   │   ├── mod.rs          # EBM module
│   │   ├── energy_net.rs   # Neural network energy function
│   │   ├── rbm.rs          # Restricted Boltzmann Machine
│   │   ├── score_matching.rs # Score matching training
│   │   └── online.rs       # Online energy estimation
│   ├── features/
│   │   ├── mod.rs          # Features module
│   │   ├── indicators.rs   # Technical indicators
│   │   └── engine.rs       # Feature engineering
│   ├── strategy/
│   │   ├── mod.rs          # Strategy module
│   │   ├── signals.rs      # Trading signals
│   │   ├── position.rs     # Position management
│   │   └── backtest.rs     # Backtesting engine
│   └── bin/
│       ├── fetch_data.rs   # Data fetching binary
│       ├── train_ebm.rs    # Training binary
│       ├── detect_anomalies.rs # Anomaly detection binary
│       ├── backtest.rs     # Backtesting binary
│       └── realtime_monitor.rs # Real-time monitoring
└── Cargo.toml
```

## Energy-Based Models

### 1. Neural Network EBM
Uses a feedforward neural network to compute energy E(x).
- Low energy = normal/typical market state
- High energy = anomalous/unusual market state

### 2. Restricted Boltzmann Machine (RBM)
Two-layer network with visible and hidden units.
- Uses Contrastive Divergence for training
- Free energy for anomaly scoring

### 3. Score Matching
Alternative training method without MCMC sampling.
- Denoising Score Matching
- Sliced Score Matching for high dimensions

## Trading Strategy

The trading strategy uses energy levels to:

1. **Risk Management**: Reduce position size when energy is high
2. **Regime Detection**: Identify market regimes (Calm, Normal, Elevated, Crisis)
3. **Contrarian Signals**: Enter after energy spikes resolve
4. **Stop Loss**: Exit on extreme energy spikes

### Position Scaling

```
Position Size = Base Size × (1 - normalized_energy / threshold)
```

| Energy Level | Regime | Action |
|--------------|--------|--------|
| < 0 | Calm | Full position |
| 0-1 | Normal | Slight reduction |
| 1-2 | Elevated | Moderate reduction |
| > 2 | Crisis | Minimal/Exit |

## Dependencies

- `reqwest`: HTTP client for Bybit API
- `ndarray`: Numerical arrays for computations
- `serde`: Serialization/deserialization
- `tokio`: Async runtime
- `clap`: Command-line argument parsing

## License

MIT License
