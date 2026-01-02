# EfficientNet Trading - Rust Examples

A modular Rust library for cryptocurrency trading using EfficientNet-based image classification.

## Quick Start

```bash
# Fetch market data
cargo run --example fetch_data

# Generate chart images
cargo run --example generate_images

# Run training demo
cargo run --example train_model

# Real-time predictions (requires WebSocket)
cargo run --example realtime_prediction

# Backtest strategy
cargo run --example backtest
```

## Project Structure

```
rust_examples/
├── Cargo.toml              # Dependencies
├── src/
│   ├── lib.rs              # Library entry point
│   ├── api/                # Bybit API integration
│   │   ├── bybit.rs        # REST API client
│   │   └── websocket.rs    # WebSocket streaming
│   ├── data/               # Data structures
│   │   ├── candle.rs       # OHLCV candlestick
│   │   └── orderbook.rs    # Order book
│   ├── imaging/            # Image generation
│   │   ├── candlestick.rs  # Candlestick charts
│   │   ├── gasf.rs         # GASF/GADF encoding
│   │   ├── orderbook_heatmap.rs
│   │   └── recurrence.rs   # Recurrence plots
│   ├── model/              # EfficientNet model
│   │   ├── efficientnet.rs # Architecture config
│   │   ├── blocks.rs       # Building blocks
│   │   └── inference.rs    # Prediction
│   ├── features/           # Feature extraction
│   ├── strategy/           # Trading strategy
│   │   ├── signal.rs       # Signal generation
│   │   └── position.rs     # Position management
│   ├── backtest/           # Backtesting engine
│   └── utils/              # Utilities
└── examples/               # Example programs
```

## Features

### Data Fetching

```rust
use efficientnet_trading::api::BybitClient;

let client = BybitClient::new();
let candles = client.fetch_klines("BTCUSDT", "5", 100).await?;
let orderbook = client.fetch_orderbook("BTCUSDT", 50).await?;
```

### Image Generation

```rust
use efficientnet_trading::imaging::CandlestickRenderer;

let renderer = CandlestickRenderer::new(224, 224);
let image = renderer.render(&candles);
image.save("chart.png")?;
```

### Prediction

```rust
use efficientnet_trading::model::ModelPredictor;

let predictor = ModelPredictor::new(224);
let result = predictor.predict(&image)?;

match result.signal {
    SignalType::Buy => println!("BUY signal"),
    SignalType::Sell => println!("SELL signal"),
    SignalType::Hold => println!("HOLD"),
}
```

### Backtesting

```rust
use efficientnet_trading::backtest::{BacktestEngine, BacktestConfig};

let config = BacktestConfig::default();
let mut engine = BacktestEngine::new(config);
let result = engine.run(&candles, &signals);

println!("Total Return: {:.2}%", result.metrics.total_return);
println!("Sharpe Ratio: {:.2}", result.metrics.sharpe_ratio);
```

## Dependencies

- `tokio` - Async runtime
- `reqwest` - HTTP client
- `image` - Image processing
- `ndarray` - Numerical computing
- `serde` - Serialization

## Optional Features

Enable GPU support:

```toml
[dependencies]
efficientnet_trading = { path = ".", features = ["cuda"] }
```

## License

MIT
