# GLOW Trading - Rust Implementation

High-performance GLOW (Generative Flow with Invertible 1x1 Convolutions) implementation for cryptocurrency trading.

## Features

- Real-time data fetching from Bybit API
- Complete GLOW model with:
  - ActNorm (Activation Normalization)
  - Invertible 1x1 Convolutions
  - Affine Coupling Layers
  - Multi-scale architecture
- Exact log-likelihood computation
- Scenario generation for risk analysis
- Comprehensive backtesting framework
- Performance metrics (Sharpe, Sortino, Max Drawdown, etc.)

## Quick Start

### 1. Fetch Data

```bash
cargo run --example fetch_data
```

This will fetch BTC/USDT hourly data from Bybit and save to `btc_data.csv`.

### 2. Train Model

```bash
cargo run --example train_model
```

This trains a GLOW model on the fetched data and saves to `glow_model.bin`.

### 3. Run Backtest

```bash
cargo run --example backtest
```

This runs a backtest using the trained model and outputs comprehensive metrics.

### 4. Generate Scenarios

```bash
cargo run --example generate_scenarios
```

This generates market scenarios for risk analysis (VaR, CVaR, etc.).

## CLI Usage

```bash
# Fetch 30 days of ETH data
cargo run -- fetch --symbol ETHUSDT --days 30 --output eth_data.csv

# Train model with custom epochs
cargo run -- train --input data.csv --output model.bin --epochs 200

# Run backtest with custom capital
cargo run -- backtest --model model.bin --data data.csv --capital 50000

# Generate trading signal
cargo run -- signal --model model.bin --symbol BTCUSDT

# Generate scenarios
cargo run -- scenarios --model model.bin --num 2000 --temperature 1.0
```

## Project Structure

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Library exports
│   ├── main.rs             # CLI application
│   ├── data/
│   │   ├── mod.rs
│   │   ├── bybit_client.rs # Bybit API client
│   │   ├── features.rs     # Feature engineering
│   │   └── preprocessing.rs # Data normalization
│   ├── model/
│   │   ├── mod.rs
│   │   ├── layers.rs       # ActNorm, Conv1x1, Coupling
│   │   └── glow.rs         # Full GLOW model
│   ├── trading/
│   │   ├── mod.rs
│   │   ├── trader.rs       # GLOW trader
│   │   ├── backtest.rs     # Backtesting framework
│   │   └── metrics.rs      # Performance metrics
│   └── utils/
│       ├── mod.rs
│       ├── config.rs       # Configuration
│       └── checkpoint.rs   # Model saving/loading
└── examples/
    ├── fetch_data.rs       # Data fetching example
    ├── train_model.rs      # Training example
    ├── backtest.rs         # Backtesting example
    └── generate_scenarios.rs # Scenario generation
```

## Dependencies

- `ndarray` - N-dimensional arrays
- `reqwest` - HTTP client for Bybit API
- `tokio` - Async runtime
- `serde` - Serialization
- `chrono` - Time handling
- `indicatif` - Progress bars
- `clap` - CLI parsing
- `bincode` - Binary serialization

## Model Architecture

```
Input Features (16-dim)
        │
        ▼
    ┌───────────────┐
    │   Level 1     │
    │ ┌───────────┐ │
    │ │  ActNorm  │ │ × K steps
    │ │ Conv 1x1  │ │
    │ │  Coupling │ │
    │ └───────────┘ │
    └───────┬───────┘
            │ split
    ┌───────┴───────┐
    │     z_1       │ → Latent (8-dim)
    │     h_1       │
    └───────┬───────┘
            │
    ┌───────────────┐
    │   Level 2     │
    │    K steps    │
    └───────┬───────┘
            │ split
    ┌───────┴───────┐
    │     z_2       │ → Latent (4-dim)
    │     h_2       │
    └───────┬───────┘
            │
    ┌───────────────┐
    │   Level 3     │
    │    K steps    │
    └───────┬───────┘
            │
            ▼
         z_3 (4-dim)

Total Latent: z = concat(z_1, z_2, z_3) = 16-dim
```

## Feature Engineering

The model uses 16 features extracted from OHLCV data:

| Feature | Description |
|---------|-------------|
| return_1 | 1-period return |
| return_5 | 5-period cumulative return |
| return_10 | 10-period cumulative return |
| return_20 | 20-period cumulative return |
| volatility_5 | 5-period volatility |
| volatility_20 | 20-period volatility |
| vol_ratio | Volatility ratio (5/20) |
| momentum_10 | 10-period momentum |
| momentum_20 | 20-period momentum |
| volume_ratio | Volume vs 20-period MA |
| price_position | Price position in range |
| body_ratio | Candle body ratio |
| upper_shadow | Upper shadow ratio |
| lower_shadow | Lower shadow ratio |
| atr_norm | Normalized ATR |
| rsi | RSI-like indicator |

## Trading Strategy

1. **Likelihood-based filtering**: Only trade when log p(x) > threshold
2. **Latent space signal**: Use first latent dimension for direction
3. **Confidence scaling**: Scale position by normalized likelihood
4. **Regime detection**: Cluster latent space for regime identification

## Performance Metrics

- **Total Return**: Cumulative P&L
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Calmar Ratio**: Return / Max Drawdown

## Risk Analysis

The scenario generation feature provides:
- **VaR (Value at Risk)**: 95% and 99% confidence levels
- **CVaR (Conditional VaR)**: Expected shortfall
- **Distribution analysis**: Skewness, kurtosis, percentiles
- **Stress testing**: Temperature-based scenario scaling

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_glow_invertibility
```

## License

MIT

## References

1. [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039) - Kingma & Dhariwal, 2018
2. [Density Estimation Using Real-NVP](https://arxiv.org/abs/1605.08803) - Dinh et al., 2016
