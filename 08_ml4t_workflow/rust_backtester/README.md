# Rust Backtester for Cryptocurrency

A modular cryptocurrency backtesting framework with Bybit integration, implementing the ML4T workflow in Rust.

## Features

- **Bybit API Integration**: Fetch historical candlestick data from Bybit exchange
- **Modular Architecture**: Clean separation of concerns with distinct modules
- **Multiple Strategies**: SMA Crossover, RSI, and ML Signal strategies
- **Event-Driven Backtesting**: Realistic simulation with fees and slippage
- **Vectorized Backtesting**: Fast backtests for strategy comparison
- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown, and more

## Project Structure

```
rust_backtester/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP client implementation
│   │   ├── error.rs        # Error types
│   │   └── response.rs     # API response structures
│   ├── models/             # Data models
│   │   ├── mod.rs
│   │   ├── candle.rs       # OHLCV candlestick
│   │   ├── order.rs        # Trading orders
│   │   ├── position.rs     # Position tracking
│   │   └── timeframe.rs    # Timeframe definitions
│   ├── backtest/           # Backtesting engine
│   │   ├── mod.rs
│   │   ├── engine.rs       # Main backtest logic
│   │   ├── broker.rs       # Simulated broker
│   │   └── result.rs       # Backtest results
│   ├── strategies/         # Trading strategies
│   │   ├── mod.rs
│   │   ├── base.rs         # Strategy trait
│   │   ├── sma_crossover.rs
│   │   ├── rsi_strategy.rs
│   │   └── ml_strategy.rs  # ML signal integration
│   └── utils/              # Utilities
│       ├── mod.rs
│       ├── indicators.rs   # Technical indicators
│       ├── metrics.rs      # Performance metrics
│       └── data.rs         # Data I/O
└── examples/
    ├── fetch_data.rs       # Download market data
    ├── simple_backtest.rs  # Vectorized backtest
    ├── sma_crossover.rs    # Event-driven SMA strategy
    └── ml_signals.rs       # ML signal backtesting
```

## Quick Start

### Prerequisites

- Rust 1.70+ (install from https://rustup.rs)
- Internet connection for Bybit API access

### Build

```bash
cd rust_backtester
cargo build --release
```

### Run Examples

#### 1. Fetch Historical Data

```bash
# Fetch 30 days of BTCUSDT hourly data
cargo run --example fetch_data -- --symbol BTCUSDT --timeframe 1h --days 30

# Save as JSON
cargo run --example fetch_data -- --symbol ETHUSDT --timeframe 4h --days 60 --format json
```

#### 2. Simple Vectorized Backtest

```bash
cargo run --example simple_backtest
```

This runs multiple strategies (Buy & Hold, SMA, Mean Reversion, Momentum) and compares results.

#### 3. SMA Crossover Event-Driven Backtest

```bash
# Default: BTCUSDT, 10/50 SMA
cargo run --example sma_crossover

# Custom parameters
cargo run --example sma_crossover -- --symbol ETHUSDT --fast 20 --slow 100 --days 90
```

#### 4. ML Signals Backtest

```bash
# With mock signals (momentum-based)
cargo run --example ml_signals

# With custom signals file
cargo run --example ml_signals -- --signals-file my_predictions.csv
```

## Usage in Your Code

### Basic Example

```rust
use rust_backtester::{
    api::BybitClient,
    backtest::BacktestEngine,
    models::Timeframe,
    strategies::SmaCrossover,
};
use chrono::{Duration, Utc};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fetch data
    let client = BybitClient::new();
    let candles = client
        .get_historical_klines(
            "BTCUSDT",
            Timeframe::H1,
            Utc::now() - Duration::days(30),
            Utc::now(),
        )
        .await?;

    // Create strategy
    let mut strategy = SmaCrossover::new(10, 50);

    // Run backtest
    let mut engine = BacktestEngine::new();
    let result = engine.run(&mut strategy, &candles);

    // Print results
    result.print_report();

    Ok(())
}
```

### Custom Strategy

```rust
use rust_backtester::{
    models::Candle,
    strategies::{Signal, Strategy},
};

struct MyStrategy {
    // Your strategy state
}

impl Strategy for MyStrategy {
    fn name(&self) -> &str {
        "My Custom Strategy"
    }

    fn on_candle(&mut self, candle: &Candle, historical: &[Candle]) -> Signal {
        // Your logic here
        if candle.close > candle.open {
            Signal::Buy(1.0)
        } else {
            Signal::Hold
        }
    }
}
```

### Using ML Signals

```rust
use rust_backtester::strategies::MlSignalStrategy;
use std::path::Path;

// Load from CSV (timestamp,signal format)
let strategy = MlSignalStrategy::from_csv_file(
    Path::new("predictions.csv"),
    0.5,   // buy threshold
    -0.5,  // sell threshold
)?;

// Or from JSON
let strategy = MlSignalStrategy::from_json_file(
    Path::new("predictions.json"),
    0.5,
    -0.5,
)?;
```

## Available Indicators

- **SMA**: Simple Moving Average
- **EMA**: Exponential Moving Average
- **RSI**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands
- **ATR**: Average True Range

## Performance Metrics

- Total Return
- Annualized Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Win Rate
- Profit Factor
- Deflated Sharpe Ratio (for multiple testing correction)

## Supported Timeframes

| Code | Description |
|------|-------------|
| M1   | 1 minute    |
| M5   | 5 minutes   |
| M15  | 15 minutes  |
| M30  | 30 minutes  |
| H1   | 1 hour      |
| H4   | 4 hours     |
| D1   | 1 day       |
| W1   | 1 week      |

## Integration with Python ML

The typical workflow:

1. **Python**: Train ML model and generate predictions
   ```python
   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier

   # Train model
   model = RandomForestClassifier()
   model.fit(X_train, y_train)

   # Generate signals
   predictions = model.predict_proba(X_test)[:, 1]
   signals = pd.DataFrame({
       'timestamp': timestamps,
       'signal': predictions * 2 - 1  # Convert to -1 to 1
   })
   signals.to_csv('ml_signals.csv', index=False)
   ```

2. **Rust**: Load signals and backtest
   ```bash
   cargo run --example ml_signals -- --signals-file ml_signals.csv
   ```

## License

MIT License - See LICENSE file for details.
