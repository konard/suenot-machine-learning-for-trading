# BloombergGPT Trading - Rust Implementation

High-performance financial LLM trading toolkit inspired by BloombergGPT.

## Features

- **Sentiment Analysis**: Financial text sentiment analysis using LLM APIs
- **Signal Generation**: Convert sentiment into actionable trading signals
- **Backtesting**: Test LLM-derived trading strategies on historical data
- **Market Data**: Load data from Yahoo Finance and Bybit exchange
- **Production Ready**: Async/await, proper error handling, comprehensive tests

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
bloomberggpt_trading = { path = "." }
tokio = { version = "1.35", features = ["full"] }
```

### Basic Usage

```rust
use bloomberggpt_trading::{SentimentAnalyzer, TradingSignalGenerator};
use bloomberggpt_trading::signals::SignalSource;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create sentiment analyzer (mock for demo)
    let analyzer = SentimentAnalyzer::new_mock();

    // Analyze financial text
    let result = analyzer.analyze(
        "Apple reports record quarterly earnings, beating expectations"
    ).await?;

    println!("Sentiment: {} (score: {:.2})", result.sentiment, result.score);

    // Generate trading signal
    let generator = TradingSignalGenerator::new();
    if let Some(signal) = generator.generate_signal(&result, "AAPL", SignalSource::News)? {
        println!("Signal: {} with strength {:.2}", signal.signal_type, signal.strength);
    }

    Ok(())
}
```

## Running Examples

```bash
# Sentiment analysis demo
cargo run --bin sentiment_demo

# Trading signal generation demo
cargo run --bin signal_demo

# Backtesting demo
cargo run --bin backtest_demo

# Cryptocurrency demo with Bybit API
cargo run --bin crypto_demo
```

## Modules

### Sentiment Analysis (`sentiment.rs`)

```rust
use bloomberggpt_trading::{SentimentAnalyzer, Sentiment};

// Using OpenAI (requires API key)
let analyzer = SentimentAnalyzer::new_openai("your-api-key");

// Using mock (for testing)
let analyzer = SentimentAnalyzer::new_mock();

let result = analyzer.analyze("NVIDIA AI chips see record demand").await?;
assert!(matches!(result.sentiment, Sentiment::Positive | Sentiment::VeryPositive));
```

### Signal Generation (`signals.rs`)

```rust
use bloomberggpt_trading::{TradingSignalGenerator, SignalType};
use bloomberggpt_trading::signals::{SignalConfig, SignalSource};

let config = SignalConfig {
    min_confidence: 0.6,
    signal_duration_hours: 24,
    decay_hours: 12.0,
    ..Default::default()
};

let generator = TradingSignalGenerator::with_config(config);
let signal = generator.generate_signal(&sentiment_result, "AAPL", SignalSource::Earnings)?;

if let Some(s) = signal {
    let position = s.position_size(10000.0);  // Max $10k position
    println!("Recommended position: ${:.2}", position.abs());
}
```

### Backtesting (`backtest.rs`)

```rust
use bloomberggpt_trading::{Backtester, BacktestConfig};
use bloomberggpt_trading::backtest::print_backtest_report;

let config = BacktestConfig {
    initial_capital: 100_000.0,
    max_position_size: 0.1,
    transaction_cost_bps: 10.0,
    short_selling: true,
    ..Default::default()
};

let mut backtester = Backtester::new(config);
let result = backtester.run(&signals, &prices)?;

print_backtest_report(&result);
```

### Market Data (`data.rs`)

```rust
use bloomberggpt_trading::data::{MarketDataLoader, Interval};
use chrono::{Duration, Utc};

// Yahoo Finance (stocks)
let loader = MarketDataLoader::yahoo();
let bars = loader.fetch_ohlcv(
    "AAPL",
    Interval::Day1,
    Utc::now() - Duration::days(30),
    Utc::now()
).await?;

// Bybit (crypto)
let loader = MarketDataLoader::bybit();
let bars = loader.fetch_ohlcv(
    "BTCUSDT",
    Interval::Hour1,
    Utc::now() - Duration::days(7),
    Utc::now()
).await?;
```

### Bybit API (`api.rs`)

```rust
use bloomberggpt_trading::api::BybitClient;

let client = BybitClient::new();

// Get ticker
let ticker = client.get_ticker("BTCUSDT").await?;
println!("BTC price: ${:.2}", ticker.last_price);

// Get orderbook
let book = client.get_orderbook("ETHUSDT", 10).await?;
println!("Best bid: ${:.2}", book.bids[0].price);
```

## Architecture

```
src/
  lib.rs          - Library entry point and re-exports
  error.rs        - Error types and Result alias
  sentiment.rs    - Sentiment analysis module
  signals.rs      - Trading signal generation
  backtest.rs     - Backtesting engine
  data.rs         - Market data loaders
  api.rs          - External API clients
  bin/
    sentiment_demo.rs   - Sentiment analysis example
    signal_demo.rs      - Signal generation example
    backtest_demo.rs    - Backtesting example
    crypto_demo.rs      - Crypto/Bybit example
```

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_sentiment_from_score
```

## Benchmarks

```bash
cargo bench
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: API key for OpenAI sentiment analysis
- `BYBIT_API_KEY`: API key for Bybit authenticated endpoints
- `BYBIT_API_SECRET`: API secret for Bybit

### Using OpenAI for Sentiment

```rust
use std::env;

let api_key = env::var("OPENAI_API_KEY")?;
let analyzer = SentimentAnalyzer::new_openai(&api_key);
```

## Performance

- Async/await for non-blocking I/O
- Connection pooling with reqwest
- Efficient data structures (HashMap, Vec)
- Zero-copy parsing where possible

## License

MIT License
