# Crypto ML - Decision Trees & Random Forests in Rust

Machine learning library for cryptocurrency trading using Decision Trees and Random Forests with Bybit market data.

## Features

- **Bybit API Client** - Fetch historical kline data from Bybit exchange
- **Feature Engineering** - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Decision Tree** - Full implementation with visualization and feature importance
- **Random Forest** - Parallel training with OOB score and bagging
- **Backtesting** - Complete backtesting framework with performance metrics

## Project Structure

```
rust_examples/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs              # Main library
    ├── api/                # Bybit API client
    │   ├── mod.rs
    │   ├── client.rs       # HTTP client
    │   └── types.rs        # API types
    ├── data/               # Data structures
    │   ├── mod.rs
    │   ├── candle.rs       # OHLCV candle
    │   └── dataset.rs      # ML dataset
    ├── features/           # Feature engineering
    │   ├── mod.rs
    │   ├── engine.rs       # Feature generator
    │   └── indicators.rs   # Technical indicators
    ├── models/             # ML models
    │   ├── mod.rs
    │   ├── decision_tree.rs
    │   └── random_forest.rs
    ├── backtest/           # Backtesting
    │   ├── mod.rs
    │   ├── engine.rs       # Backtest engine
    │   └── metrics.rs      # Performance metrics
    └── bin/                # Examples
        ├── fetch_data.rs
        ├── train_decision_tree.rs
        ├── train_random_forest.rs
        └── backtest.rs
```

## Quick Start

### Build

```bash
cd rust_examples
cargo build --release
```

### Fetch Data

```bash
# Fetch 30 days of BTCUSDT hourly data
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 1h --days 30

# Save to CSV
cargo run --bin fetch_data -- --symbol ETHUSDT --days 90 --output data.csv
```

### Train Decision Tree

```bash
# Basic training
cargo run --bin train_decision_tree -- --symbol BTCUSDT --days 90

# Classification mode with custom depth
cargo run --bin train_decision_tree -- \
  --symbol BTCUSDT \
  --days 180 \
  --max-depth 8 \
  --classification
```

### Train Random Forest

```bash
# Train with 100 trees
cargo run --bin train_random_forest -- \
  --symbol BTCUSDT \
  --days 180 \
  --trees 100 \
  --classification

# Regression mode
cargo run --bin train_random_forest -- \
  --symbol ETHUSDT \
  --days 365 \
  --trees 50
```

### Backtest Strategy

```bash
# Basic backtest
cargo run --bin backtest -- --symbol BTCUSDT --days 365

# With short selling and custom capital
cargo run --bin backtest -- \
  --symbol BTCUSDT \
  --days 365 \
  --capital 50000 \
  --allow-short \
  --fee 0.1
```

## Library Usage

### Fetch Data

```rust
use crypto_ml::api::{BybitClient, Interval, Symbol};
use chrono::{Duration, Utc};

#[tokio::main]
async fn main() {
    let client = BybitClient::new();
    let symbol = Symbol::btcusdt();

    let end = Utc::now();
    let start = end - Duration::days(30);

    let candles = client
        .get_historical_klines(&symbol, Interval::Hour1, start, end)
        .await
        .unwrap();

    println!("Got {} candles", candles.len());
}
```

### Generate Features

```rust
use crypto_ml::features::FeatureEngine;

let engine = FeatureEngine::new()
    .with_horizon(1);  // Predict 1 period ahead

let dataset = engine.generate(&candles);
println!("Features: {:?}", dataset.feature_names);
```

### Train Decision Tree

```rust
use crypto_ml::models::{DecisionTree, TreeConfig, TaskType};

let config = TreeConfig {
    max_depth: 10,
    min_samples_split: 5,
    task: TaskType::Classification,
    ..Default::default()
};

let mut tree = DecisionTree::new(config);
tree.fit(&train_data);

let predictions = tree.predict(&test_data);
let accuracy = tree.accuracy(&test_data);
```

### Train Random Forest

```rust
use crypto_ml::models::{RandomForest, TaskType};
use crypto_ml::models::random_forest::ForestConfig;

let config = ForestConfig {
    n_trees: 100,
    max_depth: 8,
    task: TaskType::Classification,
    ..Default::default()
};

let mut forest = RandomForest::new(config);
forest.fit(&train_data);

// Get feature importance
for (name, importance) in forest.feature_importance_ranking() {
    println!("{}: {:.4}", name, importance);
}
```

### Run Backtest

```rust
use crypto_ml::backtest::{Backtest, BacktestConfig};

let config = BacktestConfig {
    initial_capital: 10000.0,
    transaction_cost: 0.001,
    allow_short: true,
    ..Default::default()
};

let backtest = Backtest::new(config);
let result = backtest.run(&candles, &predictions);

println!("Total Return: {:.2}%", result.metrics.total_return * 100.0);
println!("Sharpe Ratio: {:.2}", result.metrics.sharpe_ratio);
```

## Technical Indicators

Available features:

| Indicator | Description |
|-----------|-------------|
| SMA | Simple Moving Average |
| EMA | Exponential Moving Average |
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| Bollinger Bands | %B and Bandwidth |
| ATR | Average True Range |
| Stochastic | %K and %D oscillators |
| Momentum | Rate of Change |
| Volatility | Rolling standard deviation |
| Volume Ratio | Current vs average volume |
| VWAP | Volume Weighted Average Price |

## Performance Metrics

Backtesting provides:

- Total Return / Annual Return
- Sharpe Ratio / Sortino Ratio
- Maximum Drawdown / Calmar Ratio
- Win Rate / Profit Factor
- Trade Statistics

## Requirements

- Rust 1.70+
- Internet connection for Bybit API

## License

MIT
