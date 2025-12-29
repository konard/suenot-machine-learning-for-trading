# Conformal Prediction for Cryptocurrency Trading (Rust)

Rust implementation of conformal prediction methods for cryptocurrency trading using data from **Bybit** exchange.

## Features

- **Bybit API Client** - Fetch OHLCV data for any cryptocurrency pair
- **Feature Engineering** - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.)
- **Split Conformal Prediction** - Guaranteed coverage prediction intervals
- **Adaptive Conformal Inference** - Dynamic adjustment for time series
- **Trading Strategy** - Trade only when confident, size based on uncertainty
- **Comprehensive Metrics** - Coverage, Sharpe ratio, win rate, drawdown

## Project Structure

```
rust_examples/
├── Cargo.toml
├── README.md
├── README.ru.md
├── src/
│   ├── lib.rs                  # Library entry point
│   ├── api/
│   │   ├── mod.rs
│   │   └── bybit.rs            # Bybit API client
│   ├── data/
│   │   ├── mod.rs
│   │   ├── processor.rs        # Data preprocessing utilities
│   │   └── features.rs         # Technical indicator calculations
│   ├── conformal/
│   │   ├── mod.rs
│   │   ├── model.rs            # Simple prediction models
│   │   ├── split.rs            # Split conformal prediction
│   │   └── adaptive.rs         # Adaptive conformal inference
│   ├── strategy/
│   │   ├── mod.rs
│   │   ├── trading.rs          # Trading strategy with intervals
│   │   └── sizing.rs           # Position sizing methods
│   └── metrics/
│       ├── mod.rs
│       ├── coverage.rs         # Coverage and interval metrics
│       └── trading.rs          # Trading performance metrics
└── examples/
    ├── fetch_data.rs           # Data fetching example
    ├── split_conformal.rs      # Split CP example
    ├── adaptive_conformal.rs   # Adaptive CP example
    ├── trading_strategy.rs     # Trading strategy example
    └── full_pipeline.rs        # Complete ML trading pipeline
```

## Installation

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))

```bash
# Install Rust if you haven't
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build

```bash
cd rust_examples
cargo build --release
```

## Usage

### Run Examples

```bash
# Fetch cryptocurrency data from Bybit
cargo run --example fetch_data

# Split conformal prediction for return forecasting
cargo run --example split_conformal

# Adaptive conformal inference for time series
cargo run --example adaptive_conformal

# Trading strategy with prediction intervals
cargo run --example trading_strategy

# Complete ML trading pipeline
cargo run --example full_pipeline
```

### Use as Library

```rust
use conformal_prediction_trading::{
    api::bybit::{BybitClient, Interval},
    data::features::FeatureEngineering,
    conformal::split::SplitConformalPredictor,
    conformal::model::LinearModel,
    strategy::trading::ConformalTradingStrategy,
    metrics::coverage::CoverageMetrics,
};

fn main() -> anyhow::Result<()> {
    // Fetch data from Bybit
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", Interval::Hour4, Some(500), None, None)?;

    // Generate features and targets
    let (features, _) = FeatureEngineering::generate_features(&klines);
    let targets = FeatureEngineering::create_returns(&klines, 1);

    // Split data
    // ... training, calibration, test split ...

    // Train conformal predictor (90% coverage)
    let model = LinearModel::new(true);
    let mut cp = SplitConformalPredictor::new(model, 0.1);
    cp.fit(&x_train, &y_train, &x_calib, &y_calib);

    // Generate predictions with intervals
    let intervals = cp.predict(&x_test);

    // Create trading strategy
    let strategy = ConformalTradingStrategy::new(0.02, 0.005);
    let signals = strategy.generate_signals(&intervals);

    // Evaluate
    let coverage = CoverageMetrics::calculate(&intervals, &y_test, 0.1);
    println!("Coverage: {:.1}%", coverage.coverage * 100.0);

    Ok(())
}
```

## Modules

### API (Bybit Client)

```rust
use conformal_prediction_trading::api::bybit::{BybitClient, Interval};

let client = BybitClient::new();

// Fetch 100 hourly candles
let klines = client.get_klines("BTCUSDT", Interval::Hour1, Some(100), None, None)?;

// Get current ticker
let ticker = client.get_ticker("ETHUSDT")?;
```

### Feature Engineering

Supported technical indicators:
- **Moving Averages**: SMA, EMA
- **Momentum**: RSI, MACD, Momentum, ROC
- **Volatility**: Bollinger Bands, ATR, Rolling Volatility
- **Volume**: Volume Ratio
- **Returns**: Lagged returns, Forward returns

```rust
use conformal_prediction_trading::data::features::FeatureEngineering;

// Generate all features
let (features, names) = FeatureEngineering::generate_features(&klines);

// Individual indicators
let sma_20 = FeatureEngineering::sma(&prices, 20);
let rsi_14 = FeatureEngineering::rsi(&prices, 14);
let (macd, signal, hist) = FeatureEngineering::macd(&prices, 12, 26, 9);
```

### Conformal Prediction

#### Split Conformal Prediction

```rust
use conformal_prediction_trading::conformal::{
    split::SplitConformalPredictor,
    model::LinearModel,
};

// Create predictor with 90% coverage target
let model = LinearModel::new(true);
let mut cp = SplitConformalPredictor::new(model, 0.1);

// Train and calibrate
cp.fit(&x_train, &y_train, &x_calib, &y_calib);

// Predict with intervals
let intervals = cp.predict(&x_test);

// Each interval has: prediction, lower, upper, width
for interval in &intervals {
    println!("Pred: {:.4}, Interval: [{:.4}, {:.4}]",
        interval.prediction, interval.lower, interval.upper);
}
```

#### Adaptive Conformal Inference

```rust
use conformal_prediction_trading::conformal::{
    adaptive::AdaptiveConformalPredictor,
    model::LinearModel,
};

// Create adaptive predictor
let model = LinearModel::new(true);
let mut acp = AdaptiveConformalPredictor::new(model, 0.9, 0.05);

// Initial training
acp.fit(&x_train, &y_train, &x_calib, &y_calib);

// Online prediction with updates
let interval = acp.predict_one(&x_new);
acp.update(y_actual, &interval);  // Adapts alpha based on coverage
```

### Trading Strategy

```rust
use conformal_prediction_trading::strategy::trading::ConformalTradingStrategy;

// Create strategy
let strategy = ConformalTradingStrategy::new(
    0.02,   // Width threshold (2%)
    0.005   // Minimum edge (0.5%)
);

// Generate signals
let signal = strategy.generate_signal(&interval);

if signal.trade {
    println!("Direction: {}, Size: {:.2}",
        signal.direction, signal.size);
} else {
    println!("Skip: {}", signal.skip_reason.unwrap());
}
```

### Position Sizing

```rust
use conformal_prediction_trading::strategy::sizing::{PositionSizer, KellyCriterion};

// Inverse sizing (narrower interval = larger position)
let sizer = PositionSizer::inverse().with_baseline_width(0.02);
let size = sizer.compute_size(interval_width, edge);

// Kelly criterion with conformal intervals
let kelly = KellyCriterion::calculate(prediction, lower, upper, 0.0);
```

### Metrics

```rust
use conformal_prediction_trading::metrics::{
    coverage::CoverageMetrics,
    trading::TradingMetrics,
};

// Coverage metrics
let coverage = CoverageMetrics::calculate(&intervals, &actuals, 0.1);
println!("{}", coverage.report());

// Trading metrics from backtest
let trading = TradingMetrics::calculate(&trade_results);
println!("{}", trading.report());
```

## Key Concepts

### Coverage Guarantee

Split conformal prediction provides:

```
P(Y ∈ [lower, upper]) ≥ 1 - α
```

For α = 0.1, approximately 90% of your intervals will contain the true value.

### Trading Strategy Logic

1. **Only trade when confident**: Skip if interval_width > threshold
2. **Clear direction required**: Skip if interval crosses zero
3. **Size based on confidence**: Narrower interval → larger position

```
if interval_width < threshold AND lower > min_edge:
    LONG with size = f(1/interval_width)
elif interval_width < threshold AND upper < -min_edge:
    SHORT with size = f(1/interval_width)
else:
    NO TRADE
```

## Performance Tips

1. **Use release mode** for 10-100x speedup:
   ```bash
   cargo run --release --example full_pipeline
   ```

2. **Standardize features** before training

3. **Use time series splits** to avoid lookahead bias

4. **Start with wider thresholds** and tune based on results

## Contributing

Contributions are welcome! Please ensure:
- Code passes `cargo clippy`
- Tests pass with `cargo test`
- Format with `cargo fmt`

## License

MIT License - See main repository for details.
