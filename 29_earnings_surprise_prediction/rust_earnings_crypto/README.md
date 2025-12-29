# Crypto Event Surprise Prediction

Rust implementation of event surprise prediction for cryptocurrency markets using Bybit exchange data.

## Overview

This project adapts traditional earnings surprise concepts to cryptocurrency markets:

| Traditional Finance | Crypto Equivalent |
|-------------------|-------------------|
| Earnings announcement | Protocol updates, halvings, token burns |
| Analyst estimates | Community expectations, social sentiment |
| EPS surprise | Volume/price deviation from expectations |
| Earnings call | AMAs, community calls |

## Project Structure

```
rust_earnings_crypto/
├── Cargo.toml
├── src/
│   ├── main.rs           # CLI entry point
│   ├── lib.rs            # Library exports
│   ├── api/
│   │   ├── mod.rs
│   │   ├── bybit.rs      # Bybit API client
│   │   └── error.rs      # API error types
│   ├── data/
│   │   ├── mod.rs
│   │   ├── types.rs      # Core data types
│   │   └── processor.rs  # Data processing
│   ├── events/
│   │   ├── mod.rs
│   │   ├── detector.rs   # Event detection
│   │   └── types.rs      # Event types
│   ├── features/
│   │   ├── mod.rs
│   │   ├── technical.rs  # Technical indicators
│   │   ├── volume.rs     # Volume analysis
│   │   └── surprise.rs   # Surprise calculation
│   ├── models/
│   │   ├── mod.rs
│   │   └── predictor.rs  # Simple prediction model
│   └── analysis/
│       ├── mod.rs
│       └── post_event.rs # Post-event drift analysis
└── examples/
    ├── fetch_data.rs         # Fetch market data
    ├── event_analysis.rs     # Analyze crypto events
    ├── surprise_prediction.rs # Predict surprises
    ├── volume_surprise.rs    # Volume-based surprises
    └── full_pipeline.rs      # Complete workflow
```

## Usage

### Fetch Market Data

```bash
cargo run --example fetch_data -- --symbol BTCUSDT --interval 1h --limit 200
```

### Analyze Events

```bash
cargo run --example event_analysis -- --symbol BTCUSDT
```

### Run Surprise Prediction

```bash
cargo run --example surprise_prediction -- --symbol BTCUSDT --lookback 30
```

### Full Pipeline

```bash
cargo run --example full_pipeline -- --symbol BTCUSDT
```

## Key Concepts

### Event Detection

We detect significant crypto events based on:
- **Volume spikes**: >2x average volume
- **Price gaps**: >3% overnight gap
- **Volatility expansion**: >2x average volatility

### Surprise Calculation

```rust
// Price surprise
surprise = (actual_return - expected_return) / expected_volatility

// Volume surprise
vol_surprise = (actual_volume - avg_volume) / std_volume
```

### Post-Event Drift (PEAD analog)

Similar to PEAD in equities, we analyze price drift after crypto events:
- Day 0: Initial reaction
- Day 1-3: Short-term drift
- Day 4-7: Extended drift

## Examples

### 1. Fetch and Display Candles

```rust
use earnings_crypto::api::BybitClient;

#[tokio::main]
async fn main() {
    let client = BybitClient::new();
    let candles = client.get_klines("BTCUSDT", "1h", 100).await.unwrap();

    for candle in candles.iter().take(5) {
        println!("{}: O={:.2} H={:.2} L={:.2} C={:.2}",
            candle.datetime(), candle.open, candle.high, candle.low, candle.close);
    }
}
```

### 2. Detect Volume Events

```rust
use earnings_crypto::events::EventDetector;
use earnings_crypto::data::Candle;

let detector = EventDetector::new(2.0, 0.03);
let events = detector.detect_volume_events(&candles);

for event in events {
    println!("Event at {}: {} (magnitude: {:.2})",
        event.timestamp, event.event_type, event.magnitude);
}
```

### 3. Calculate Surprise Metrics

```rust
use earnings_crypto::features::SurpriseCalculator;

let calculator = SurpriseCalculator::new(20);
let surprises = calculator.calculate(&candles);

for (i, surprise) in surprises.iter().enumerate().take(5) {
    println!("Candle {}: price_surprise={:.2}, vol_surprise={:.2}",
        i, surprise.price_surprise, surprise.volume_surprise);
}
```

## Dependencies

- `tokio` - Async runtime
- `reqwest` - HTTP client for Bybit API
- `serde` - JSON serialization
- `ndarray` - Numerical arrays
- `statrs` - Statistical functions
- `chrono` - DateTime handling

## License

MIT License
