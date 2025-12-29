# Order Flow Imbalance - Rust Implementation

High-performance Rust implementation of Order Flow Imbalance (OFI) analysis for cryptocurrency trading using Bybit exchange data.

## Overview

This library provides tools for analyzing market microstructure and order flow in cryptocurrency markets:

- **Order Flow Imbalance (OFI)** - Measure buy/sell pressure from order book changes
- **VPIN** - Volume-Synchronized Probability of Informed Trading
- **Kyle's Lambda** - Price impact coefficient
- **Feature Engineering** - 50+ microstructure features for ML
- **ML Models** - Gradient boosting for direction prediction
- **Backtesting** - Intraday strategy backtesting framework

## Quick Start

### Installation

```bash
# Clone the repository
cd 30_order_flow_imbalance/rust_examples

# Build the library
cargo build --release

# Run tests
cargo test
```

### Run Examples

```bash
# Fetch order book from Bybit
cargo run --example fetch_orderbook

# Calculate OFI in real-time
cargo run --example calculate_ofi

# Calculate VPIN
cargo run --example calculate_vpin

# Feature engineering demo
cargo run --example feature_engineering

# Train ML model
cargo run --example train_model

# Backtest a strategy
cargo run --example backtest

# Real-time analysis dashboard
cargo run --example realtime_analysis

# Complete trading strategy
cargo run --example trading_strategy
```

## Architecture

```
src/
├── api/                    # Bybit API clients
│   ├── bybit.rs           # REST API client
│   └── websocket.rs       # WebSocket client
│
├── data/                   # Data structures
│   ├── orderbook.rs       # Order book representation
│   ├── trade.rs           # Trade data and statistics
│   └── snapshot.rs        # Market snapshots
│
├── orderflow/              # Order flow metrics
│   ├── ofi.rs             # Order Flow Imbalance
│   ├── vpin.rs            # VPIN calculation
│   ├── kyle.rs            # Kyle's Lambda
│   └── toxicity.rs        # Composite toxicity indicators
│
├── features/               # Feature engineering
│   ├── engine.rs          # Feature extraction engine
│   └── indicators.rs      # Technical indicators
│
├── models/                 # ML models
│   ├── gradient_boosting.rs # Gradient boosting classifier
│   └── linear.rs          # Logistic regression
│
├── metrics/                # Evaluation metrics
│   ├── classification.rs  # ML metrics (accuracy, AUC, etc.)
│   └── trading.rs         # Trading metrics (Sharpe, PF, etc.)
│
├── strategy/               # Trading strategies
│   ├── signal.rs          # Signal generation
│   └── position.rs        # Position management
│
├── backtest/               # Backtesting framework
│   ├── engine.rs          # Backtest engine
│   └── report.rs          # Report generation
│
└── lib.rs                  # Library entry point
```

## API Reference

### Bybit Client

```rust
use order_flow_imbalance::BybitClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = BybitClient::new(); // mainnet
    // let client = BybitClient::testnet(); // testnet

    // Fetch order book (50 levels)
    let orderbook = client.get_orderbook("BTCUSDT", 50).await?;
    println!("Mid price: ${}", orderbook.mid_price().unwrap());
    println!("Spread: {} bps", orderbook.spread_bps().unwrap());

    // Fetch recent trades
    let trades = client.get_trades("BTCUSDT", 100).await?;
    println!("Fetched {} trades", trades.len());

    Ok(())
}
```

### Order Flow Imbalance

```rust
use order_flow_imbalance::orderflow::ofi::OrderFlowCalculator;

let mut calculator = OrderFlowCalculator::new();

// Update with new order book snapshots
if let Some(ofi) = calculator.update(&orderbook) {
    println!("Current OFI: {}", ofi);
}

// Get rolling metrics
println!("1-min OFI: {}", calculator.ofi_1min());
println!("5-min OFI: {}", calculator.ofi_5min());
println!("Cumulative: {}", calculator.cumulative());

// Z-score for signal generation
if let Some(zscore) = calculator.z_score(100) {
    if zscore > 2.0 {
        println!("Strong buy signal!");
    }
}
```

### VPIN Calculator

```rust
use order_flow_imbalance::orderflow::vpin::VpinCalculator;

// 50 BTC per bucket, 50 buckets for rolling VPIN
let mut vpin_calc = VpinCalculator::new(50.0, 50);

for trade in trades {
    if let Some(vpin) = vpin_calc.add_trade(&trade) {
        println!("New VPIN: {}", vpin);

        if vpin > 0.7 {
            println!("High toxicity - informed traders active!");
        }
    }
}
```

### Feature Engineering

```rust
use order_flow_imbalance::features::engine::FeatureEngine;

let mut engine = FeatureEngine::new();

// Feed market data
engine.update_orderbook(&orderbook);
for trade in &trades {
    engine.update_trade(trade);
}

// Extract 50+ features
let features = engine.extract_features(&orderbook);

println!("OFI Z-score: {}", features.get("ofi_zscore").unwrap());
println!("Depth imbalance: {}", features.get("depth_imbalance_l5").unwrap());
```

### ML Model Training

```rust
use order_flow_imbalance::models::gradient_boosting::GradientBoostingModel;

let mut model = GradientBoostingModel::new(0.1);

// Train on labeled data
model.train(&training_data, 100, 6);

// Predict
let probability = model.predict_proba(&features);
let prediction = model.predict(&features);

// Get feature importance
for (name, importance) in model.top_features(10) {
    println!("{}: {:.4}", name, importance);
}
```

### Signal Generation

```rust
use order_flow_imbalance::strategy::signal::{SignalGenerator, SignalConfig};

let config = SignalConfig {
    prob_threshold_long: 0.55,
    prob_threshold_short: 0.45,
    ofi_threshold: 1.5,
    max_spread_bps: 10.0,
    ..Default::default()
};

let mut generator = SignalGenerator::new(config);
generator.set_model(model);

let signal = generator.generate(&features, mid_price, spread_bps);

match signal.signal {
    Signal::Long => println!("BUY @ ${}", mid_price),
    Signal::Short => println!("SELL @ ${}", mid_price),
    Signal::Hold => println!("No action"),
    _ => {}
}
```

### Backtesting

```rust
use order_flow_imbalance::backtest::engine::{BacktestEngine, BacktestConfig};

let config = BacktestConfig {
    initial_capital: 10000.0,
    position_size: 0.1,
    commission_rate: 0.0004,
    slippage_bps: 1.0,
    max_holding_time: 300,
    max_daily_loss: 500.0,
};

let mut engine = BacktestEngine::new(config);

// Run backtest
for orderbook in historical_data {
    engine.process_orderbook(&orderbook);
}

// Get results
let metrics = engine.metrics();
println!("Total trades: {}", metrics.total_trades);
println!("Win rate: {:.1}%", metrics.win_rate() * 100.0);
println!("Sharpe ratio: {:.2}", metrics.sharpe_ratio());
println!("Net P&L: ${:.2}", metrics.net_pnl);
```

## Key Concepts

### Order Flow Imbalance (OFI)

Based on Cont et al. (2014), OFI measures the net order flow pressure:

```
OFI = ΔBid + ΔAsk

where:
  ΔBid = change in bid pressure
  ΔAsk = change in ask pressure

OFI > 0 → buy pressure → price likely to increase
OFI < 0 → sell pressure → price likely to decrease
```

### VPIN (Volume-Synchronized Probability of Informed Trading)

From Easley et al. (2012), VPIN measures the probability of informed trading:

```
VPIN = |Buy Volume - Sell Volume| / Total Volume

VPIN > 0.7 → High toxicity, informed traders active
VPIN < 0.3 → Low toxicity, noise traders dominate
```

### Kyle's Lambda

Measures price impact per unit of order flow:

```
ΔPrice = λ × OrderFlow

High λ → Illiquid market, orders move prices
Low λ → Liquid market, orders absorbed easily
```

## Configuration

### Environment Variables

```bash
# Optional: Bybit API keys (for private endpoints)
export BYBIT_API_KEY=your_api_key
export BYBIT_API_SECRET=your_api_secret

# Optional: Use testnet
export BYBIT_TESTNET=true
```

### Feature Configuration

```rust
use order_flow_imbalance::features::engine::FeatureConfig;

let config = FeatureConfig {
    orderbook_levels: 10,      // Number of order book levels
    vpin_bucket_size: 50.0,    // BTC per VPIN bucket
    vpin_num_buckets: 50,      // Rolling window for VPIN
    history_window: 1000,      // OFI history size
    spread_window: 100,        // Spread z-score window
};
```

## Performance

Designed for low-latency trading applications:

- Feature extraction: < 1ms
- Model prediction: < 1ms
- Order book update: < 0.1ms

Use `--release` for optimized builds:

```bash
cargo build --release
cargo run --release --example trading_strategy
```

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_ofi_calculation
```

## Dependencies

- `tokio` - Async runtime
- `reqwest` - HTTP client
- `serde` - Serialization
- `ndarray` - Numerical arrays
- `chrono` - Date/time handling
- `tokio-tungstenite` - WebSocket support

## References

1. Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events.
2. Easley, D., López de Prado, M., & O'Hara, M. (2012). Flow Toxicity and Liquidity.
3. Kyle, A. S. (1985). Continuous Auctions and Insider Trading.

## License

MIT License

## Disclaimer

This is educational software. Do not use for real trading without thorough testing and understanding of the risks involved. Cryptocurrency trading carries significant risk of loss.
