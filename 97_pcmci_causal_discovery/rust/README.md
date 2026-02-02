# PCMCI Causal Discovery for Trading (Rust)

A Rust implementation of the PCMCI (Peter and Clark Momentary Conditional Independence) algorithm
for causal discovery in financial time series. PCMCI identifies causal relationships between
market variables at different time lags, enabling causality-driven trading strategies.

## Overview

PCMCI is a two-phase algorithm for causal discovery in time series:

1. **PC-stable phase (condition selection)**: Uses iterative conditional independence tests to
   identify potential parent variables for each node, removing spurious correlations.
2. **MCI phase (momentary conditional independence)**: Applies the Momentary Conditional
   Independence test to validate causal links, controlling for autocorrelation and indirect effects.

This implementation applies PCMCI to financial data (stocks and crypto) to discover causal
relationships between market variables such as returns, volume, and volatility across different
time lags.

## Features

- **PCMCI Algorithm**: Full implementation with PC-stable phase and MCI testing
- **Independence Tests**: Partial correlation (ParCorr) and Conditional Mutual Information (CMI)
- **Causal Graph**: Directed causal graph with lag structure and significance testing
- **Data Loaders**: Support for CSV (yfinance format) and Bybit API
- **Trading Strategy**: Causality-driven trading signals based on discovered causal links
- **Backtesting**: Full backtesting framework with position management and performance metrics

## Quick Start

### Building

```bash
cd rust
cargo build --release
```

### Running Examples

**Simple backtest with synthetic data:**
```bash
cargo run --example simple_backtest
```

**Live data from Bybit:**
```bash
cargo run --example bybit_live
```

## Usage

### Basic Causal Discovery

```rust
use pcmci_causal_discovery::{
    PCMCICausalDiscovery, PCMCIConfig, IndependenceTest,
};
use pcmci_causal_discovery::data::generate_synthetic_causal_data;

// Define known causal links for synthetic data
let true_links = vec![(0, 1, 1, 0.5), (1, 2, 2, -0.3)];
let data = generate_synthetic_causal_data(3, 500, &true_links);

// Configure PCMCI
let config = PCMCIConfig {
    max_lag: 3,
    significance_level: 0.05,
    test_type: IndependenceTest::ParCorr,
    max_conds_dim: None,
};

// Run causal discovery
let mut pcmci = PCMCICausalDiscovery::new(config);
let graph = pcmci.fit(&data);

// Get significant causal links
let links = graph.get_significant_links(0.05);
for link in &links {
    println!("Var {} -> Var {} (lag {}): strength={:.3}, p={:.4}",
        link.source, link.target, link.lag, link.strength, link.p_value);
}
```

### Loading Data

```rust
use pcmci_causal_discovery::data::{load_csv_data, BybitClient};

// From CSV (yfinance format)
let data = load_csv_data("AAPL.csv", "AAPL")?;

// From Bybit API
let client = BybitClient::new();
let data = client.get_klines("BTCUSDT", "15", 500).await?;
```

### Running a Causal Trading Backtest

```rust
use pcmci_causal_discovery::{
    CausalTradingStrategy, Backtester, BacktestConfig, PCMCIConfig,
};

let backtest_config = BacktestConfig::default();
let pcmci_config = PCMCIConfig::default();

let mut strategy = CausalTradingStrategy::new(pcmci_config, backtest_config.clone());
let backtester = Backtester::new(backtest_config);
let result = backtester.run(&mut strategy, &data);

println!("Total Return: {:.2}%", result.total_return * 100.0);
println!("Sharpe Ratio: {:.2}", result.sharpe_ratio);
println!("Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
```

## Module Structure

```text
rust/
+-- Cargo.toml
+-- README.md
+-- src/
|   +-- lib.rs          # Library entry point
|   +-- model/          # PCMCI causal discovery algorithm
|   |   +-- mod.rs
|   +-- data/           # Data loading and synthetic generation
|   |   +-- mod.rs
|   +-- strategy/       # Trading strategy and backtesting
|       +-- mod.rs
+-- examples/
    +-- simple_backtest.rs
    +-- bybit_live.rs
```

## Configuration Options

### PCMCIConfig

| Field | Default | Description |
|-------|---------|-------------|
| `max_lag` | 3 | Maximum time lag to consider |
| `significance_level` | 0.05 | P-value threshold for significance |
| `test_type` | ParCorr | Independence test type |
| `max_conds_dim` | None | Max conditioning set size |

### BacktestConfig

| Field | Default | Description |
|-------|---------|-------------|
| `initial_capital` | 100,000 | Starting capital |
| `position_size` | 0.1 | Fraction of capital per trade |
| `stop_loss` | 0.02 | Stop loss percentage |
| `take_profit` | 0.04 | Take profit percentage |
| `fee_rate` | 0.001 | Trading fee rate |

## Tests

Run all tests:
```bash
cargo test
```

Run with output:
```bash
cargo test -- --nocapture
```

## Dependencies

- `ndarray`: N-dimensional array operations
- `tokio`: Async runtime for API calls
- `reqwest`: HTTP client for Bybit API
- `serde`: Serialization/deserialization
- `chrono`: Date/time handling
- `rayon`: Parallel computation

## References

- Runge, J. et al. (2019). "Detecting and quantifying causal associations in large nonlinear
  time series datasets." Science Advances, 5(11).
- Runge, J. (2018). "Causal network reconstruction from time series: From theoretical assumptions
  to practical estimation." Chaos, 28(7).

## License

MIT License
