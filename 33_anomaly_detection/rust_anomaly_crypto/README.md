# Rust Anomaly Detection for Cryptocurrency Trading

High-performance anomaly detection library for cryptocurrency trading, with native Bybit exchange integration.

## Features

- **Data Module**: Bybit API client with OHLCV data handling
- **Anomaly Detection**:
  - Z-Score (rolling and global)
  - Modified Z-Score (MAD-based, robust)
  - IQR-based detection
  - Isolation Forest
  - Ensemble methods
  - Online (streaming) detection
- **Feature Engineering**: Technical indicators and anomaly-specific features
- **Trading Strategy**: Signal generation and position management

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rust_anomaly_crypto = { path = "path/to/rust_anomaly_crypto" }
```

## Quick Start

### Fetch Data from Bybit

```bash
cargo run --bin fetch_data -- --symbol BTCUSDT --interval 60 --limit 500
```

### Detect Anomalies

```bash
cargo run --bin detect_anomalies -- --symbol BTCUSDT --method ensemble
```

### Real-time Monitoring

```bash
cargo run --bin realtime_monitor -- --symbol BTCUSDT --poll-seconds 10
```

### Backtest Strategy

```bash
cargo run --bin backtest -- --symbol BTCUSDT --days 7 --contrarian
```

## Library Usage

```rust
use rust_anomaly_crypto::{
    data::{BybitClient, BybitConfig},
    anomaly::{AnomalyDetector, ZScoreDetector, EnsembleDetector},
    features::FeatureEngine,
};

// Fetch data
let client = BybitClient::public();
let data = client.get_klines("BTCUSDT", "60", 500, None, None)?;

// Extract returns
let returns = data.returns();

// Detect anomalies with Z-Score
let detector = ZScoreDetector::new(20, 3.0);
let result = detector.detect(&returns);

println!("Found {} anomalies", result.anomaly_count());

// Use ensemble detection
let mut ensemble = EnsembleDetector::new();
ensemble.fit(&returns);
let result = ensemble.detect(&returns);
```

## Modules

### `data`

- `BybitClient`: HTTP client for Bybit API
- `OHLCVSeries`: Candlestick data container
- Normalization utilities

### `anomaly`

| Detector | Description |
|----------|-------------|
| `ZScoreDetector` | Rolling Z-score based detection |
| `GlobalZScoreDetector` | Global statistics Z-score |
| `MADDetector` | Median Absolute Deviation (robust) |
| `IQRDetector` | Interquartile Range method |
| `IsolationForest` | Tree-based isolation |
| `EnsembleDetector` | Weighted combination |
| `OnlineDetector` | Streaming detection |

### `features`

- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Feature engine for computing anomaly-relevant features

### `strategy`

- Signal generation from anomaly scores
- Position management with risk controls

## Configuration

### Environment Variables

- `RUST_LOG`: Logging level (`debug`, `info`, `warn`, `error`)

### Signal Thresholds

```rust
use rust_anomaly_crypto::strategy::SignalConfig;

let config = SignalConfig {
    reduce_threshold: 0.7,    // Reduce position when score > 0.7
    exit_threshold: 1.5,      // Exit all when score > 1.5
    entry_threshold: 1.0,     // Consider entry after score was > 1.0
    enable_contrarian: true,  // Enable contrarian entries
    ..Default::default()
};
```

## Performance

The Rust implementation offers:
- Fast computation of rolling statistics
- Efficient Isolation Forest with parallel tree building
- Low-latency online detection for real-time use
- Memory-efficient data handling

## Examples

### Custom Anomaly Detection Pipeline

```rust
use rust_anomaly_crypto::{
    data::BybitClient,
    features::{FeatureEngine, FeatureConfig},
    anomaly::{MultivariateDetector, IsolationForest},
};

// Fetch data
let client = BybitClient::public();
let data = client.get_klines("ETHUSDT", "15", 1000, None, None)?;

// Compute features
let engine = FeatureEngine::with_config(FeatureConfig {
    window: 20,
    include_volume: true,
    include_momentum: true,
    include_volatility: true,
    ..Default::default()
});
let features = engine.compute(&data);

// Multivariate detection with Isolation Forest
let mut iforest = IsolationForest::new(100, 0.01);
iforest.fit(&features.valid_data());
let result = iforest.detect(&features.valid_data());

// Analyze top anomalies
for idx in result.anomaly_indices().iter().take(5) {
    println!("Anomaly at index {}, score: {:.4}", idx, result.scores[*idx]);
}
```

### Online Monitoring

```rust
use rust_anomaly_crypto::anomaly::OnlineDetector;

let mut detector = OnlineDetector::new(100, 3.0);

// Process streaming data
for price in price_stream {
    let (score, is_anomaly, z_score) = detector.update(price);

    if is_anomaly {
        println!("ANOMALY DETECTED! Score: {:.2}, Z: {:.2}", score, z_score);
    }
}
```

## License

MIT
