# Risk Hedging with Anomaly Detection (Rust)

A Rust implementation of anomaly detection for cryptocurrency risk hedging, using data from Bybit exchange.

## Features

- **Real-time data fetching** from Bybit API
- **Multiple anomaly detectors**: Z-Score, Isolation Forest, Mahalanobis Distance
- **Ensemble detection** combining multiple methods
- **Automatic hedging recommendations** based on risk levels
- **Portfolio tracking** and risk metrics
- **Backtesting** framework for strategy validation

## Project Structure

```
rust_risk_hedging/
├── src/
│   ├── lib.rs                 # Library entry point
│   ├── data/                  # Data fetching and handling
│   │   ├── mod.rs
│   │   ├── bybit.rs          # Bybit API client
│   │   ├── ohlcv.rs          # OHLCV data structures
│   │   └── normalize.rs      # Data normalization
│   ├── features/             # Feature engineering
│   │   ├── mod.rs
│   │   ├── indicators.rs     # Technical indicators
│   │   └── risk_features.rs  # Risk-specific features
│   ├── anomaly/              # Anomaly detection algorithms
│   │   ├── mod.rs
│   │   ├── zscore.rs         # Z-Score detector
│   │   ├── isolation_forest.rs # Isolation Forest
│   │   ├── mahalanobis.rs    # Mahalanobis Distance
│   │   └── ensemble.rs       # Ensemble detector
│   ├── risk/                 # Risk management
│   │   ├── mod.rs
│   │   ├── hedging.rs        # Hedging strategies
│   │   ├── portfolio.rs      # Portfolio tracking
│   │   └── signals.rs        # Risk signals
│   └── bin/                  # Executable examples
│       ├── fetch_data.rs     # Fetch market data
│       ├── detect_risk.rs    # Detect anomalies
│       ├── hedging_backtest.rs # Backtest strategy
│       └── risk_monitor.rs   # Real-time monitoring
└── Cargo.toml
```

## Installation

```bash
cd rust_risk_hedging
cargo build --release
```

## Usage Examples

### 1. Fetch Market Data

```bash
# Fetch 200 hourly candles for BTCUSDT
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 200

# Fetch data for multiple symbols
cargo run --bin fetch_data -- --multi

# Save to CSV
cargo run --bin fetch_data -- --symbol BTCUSDT --output btc_data.csv
```

### 2. Detect Risk Anomalies

```bash
# Basic detection
cargo run --bin detect_risk -- --symbol BTCUSDT

# Detailed analysis with all features
cargo run --bin detect_risk -- --symbol BTCUSDT --detailed

# Custom portfolio size
cargo run --bin detect_risk -- --symbol BTCUSDT --portfolio 50000
```

### 3. Backtest Hedging Strategy

```bash
# Run backtest
cargo run --bin hedging_backtest -- --symbol BTCUSDT --limit 500

# Custom position size
cargo run --bin hedging_backtest -- --symbol BTCUSDT --position-size 0.3
```

### 4. Real-time Risk Monitor

```bash
# Monitor BTC and ETH
cargo run --bin risk_monitor -- --symbols BTCUSDT,ETHUSDT --interval 60

# Monitor with custom update count
cargo run --bin risk_monitor -- --symbols BTCUSDT --updates 20
```

## Anomaly Detection Methods

### Z-Score Detector

Measures how many standard deviations a value is from the rolling mean.

```rust
use rust_risk_hedging::anomaly::ZScoreDetector;

let detector = ZScoreDetector::new(20, 3.0); // 20-period window, 3 sigma threshold
let scores = detector.detect(&prices);
```

### Isolation Forest

Identifies anomalies based on how easily a point can be "isolated" from others.

```rust
use rust_risk_hedging::anomaly::SimpleIsolationDetector;

let detector = SimpleIsolationDetector::new(50, 10);
let scores = detector.detect(&prices);
```

### Mahalanobis Distance

Statistical distance accounting for correlations between features.

```rust
use rust_risk_hedging::anomaly::RollingMahalanobisDetector;

let detector = RollingMahalanobisDetector::new(30, 3.0);
let scores = detector.detect(&prices);
```

### Ensemble Detector

Combines all methods for robust detection.

```rust
use rust_risk_hedging::anomaly::EnsembleDetector;

let detector = EnsembleDetector::default();
let results = detector.detect_from_ohlcv(&data);

for result in results {
    println!("Score: {:.2}, Level: {:?}", result.score, result.level);
}
```

## Risk Levels

| Level | Score Range | Recommended Action |
|-------|-------------|-------------------|
| Normal | 0% - 50% | Continue trading normally |
| Elevated | 50% - 70% | Monitor closely, reduce leverage |
| High | 70% - 90% | Reduce positions, light hedge (5%) |
| Extreme | 90% - 100% | Exit positions, heavy hedge (15%) |

## Hedging Strategy

```rust
use rust_risk_hedging::risk::HedgingStrategy;

let strategy = HedgingStrategy::default();
let allocation = strategy.decide(anomaly_score, portfolio_value);

println!("Hedge {}% of portfolio", allocation.total_hedge_pct * 100.0);
for (instrument, amount) in allocation.dollar_amounts(portfolio_value) {
    println!("  {:?}: ${:.2}", instrument, amount);
}
```

## Hedging Instruments

- **Stablecoin**: USDT/USDC - instant conversion
- **Short Position**: Profit when price falls
- **Put Options**: High leverage protection
- **Inverse Perpetual**: Alternative hedging

## API Reference

### BybitClient

```rust
use rust_risk_hedging::data::BybitClient;

let client = BybitClient::public();

// Get klines
let data = client.get_klines("BTCUSDT", "60", 200, None, None)?;

// Get ticker
let ticker = client.get_ticker("BTCUSDT")?;
println!("Price: ${:.2}", ticker.last_price);
```

### Risk Features

```rust
use rust_risk_hedging::features::RiskFeatures;

let features = RiskFeatures::from_ohlcv(&data);
println!("Volatility: {:.2}%", features.volatility.last().unwrap_or(&0.0));
println!("Max Drawdown: {:.2}%", features.max_drawdown.last().unwrap_or(&0.0));
```

### Portfolio Tracking

```rust
use rust_risk_hedging::risk::{Portfolio, PortfolioTracker, Position, PositionType};

let mut portfolio = Portfolio::new(100_000.0);
portfolio.add_position(Position::new(
    "BTCUSDT".into(),
    1.0,
    50000.0,
    PositionType::Long,
));

let tracker = PortfolioTracker::new(portfolio);
let metrics = tracker.risk_metrics();
println!("{}", metrics.format());
```

## Dependencies

- `reqwest` - HTTP client for API calls
- `serde` - JSON serialization
- `tokio` - Async runtime
- `ndarray` - Linear algebra
- `chrono` - Date/time handling
- `clap` - Command line parsing
- `colored` - Terminal colors

## Testing

```bash
cargo test
```

## Performance

- Data fetching: ~100-300ms per request
- Anomaly detection: ~1-5ms for 200 candles
- Ensemble detection: ~5-15ms for full analysis

## License

MIT
