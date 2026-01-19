# LLM Sentiment Analysis for Trading (Rust)

A Rust library for LLM-based sentiment analysis in cryptocurrency trading, with Bybit exchange integration.

## Features

- **Text Preprocessing**: Unicode normalization, URL/mention removal, tokenization
- **Financial Lexicon**: Domain-specific sentiment scoring with negation handling
- **Crypto Terminology**: Support for crypto slang (HODL, moon, rekt, WAGMI, etc.)
- **Sentiment Aggregation**: Time-weighted, source-weighted sentiment combining
- **Trading Signals**: Convert sentiment to actionable buy/sell/hold signals
- **Backtesting**: Framework for testing sentiment-based strategies
- **Bybit Integration**: Market data fetching from Bybit exchange

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
llm_sentiment = { path = "path/to/rust_llm_sentiment" }
tokio = { version = "1.0", features = ["full"] }
```

## Quick Start

### Basic Sentiment Analysis

```rust
use llm_sentiment::SentimentAnalyzer;

fn main() {
    let analyzer = SentimentAnalyzer::new();

    let result = analyzer.analyze("Bitcoin surges 10% as institutional adoption accelerates!");

    println!("Score: {:.2}", result.score);
    println!("Level: {}", result.level.as_str());
    println!("Confidence: {:.0}%", result.confidence * 100.0);
}
```

### Crypto-Specific Analysis

```rust
use llm_sentiment::CryptoTerminology;

fn main() {
    let crypto = CryptoTerminology::new();

    let result = crypto.analyze("$BTC mooning! Diamond hands! WAGMI!");

    println!("Crypto Score: {:.2}", result.score);
    println!("Terms Found: {:?}", result.found_terms);
}
```

### Generate Trading Signals

```rust
use llm_sentiment::{SentimentAnalyzer, SignalGenerator};

fn main() {
    let analyzer = SentimentAnalyzer::new();
    let signal_gen = SignalGenerator::new()
        .with_thresholds(0.3, -0.3, 0.6)
        .with_min_confidence(0.6);

    let result = analyzer.analyze("Massive whale accumulation detected. Bullish!");
    let signal = signal_gen.generate(&result);

    println!("Signal: {}", signal.signal.as_str());
    println!("Position Size: {:.1}%", signal.position_size * 100.0);
}
```

### Fetch Bybit Data

```rust
use llm_sentiment::{BybitClient, Interval};
use chrono::{Duration, Utc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = BybitClient::new();

    // Get current ticker
    let ticker = client.get_ticker("BTCUSDT").await?;
    println!("BTC Price: ${:.2}", ticker.last_price);

    // Get historical candles
    let end = Utc::now();
    let start = end - Duration::days(7);
    let candles = client.get_klines("BTCUSDT", Interval::OneHour, start, end).await?;
    println!("Fetched {} candles", candles.len());

    Ok(())
}
```

## Examples

Run the examples to see the library in action:

```bash
# Fetch market data
cargo run --example fetch_news

# Analyze sentiment
cargo run --example analyze_sentiment

# Crypto-specific sentiment
cargo run --example crypto_sentiment

# Generate trading signals
cargo run --example trading_signals

# Run backtest
cargo run --example backtest
```

## Module Structure

```
src/
├── lib.rs              # Main library exports
├── api/
│   ├── mod.rs          # API module
│   └── bybit.rs        # Bybit exchange client
├── data/
│   ├── mod.rs          # Data module
│   ├── preprocessing.rs # Text preprocessing
│   └── news.rs         # News article structures
├── sentiment/
│   ├── mod.rs          # Sentiment module
│   ├── analyzer.rs     # Main sentiment analyzer
│   ├── lexicon.rs      # Financial lexicon
│   ├── crypto.rs       # Crypto terminology
│   └── aggregator.rs   # Sentiment aggregation
└── strategy/
    ├── mod.rs          # Strategy module
    ├── signals.rs      # Signal generation
    └── backtest.rs     # Backtesting framework
```

## Configuration

### Signal Generator

```rust
let signal_gen = SignalGenerator::new()
    .with_thresholds(
        0.3,   // Bullish threshold
        -0.3,  // Bearish threshold
        0.6,   // Strong signal threshold
    )
    .with_min_confidence(0.6)
    .with_risk_params(
        0.5,   // Max position size
        0.05,  // Stop loss %
        0.10,  // Take profit %
    );
```

### Backtesting

```rust
let config = BacktestConfig {
    initial_capital: 10_000.0,
    fee_pct: 0.001,       // 0.1%
    slippage_pct: 0.0005, // 0.05%
    max_position_size: 0.5,
    use_stop_loss: true,
    use_take_profit: true,
};

let backtester = Backtester::new(config);
```

## Testing

Run tests:

```bash
cargo test
```

Run tests with output:

```bash
cargo test -- --nocapture
```

## Supported Crypto Slang

### Bullish Terms
- moon, mooning, HODL, diamond hands, bullish, pump, ATH
- FOMO, WAGMI, LFG, buy the dip, accumulating, stacking sats

### Bearish Terms
- rekt, dump, bearish, rug pull, scam, FUD, paper hands
- NGMI, crash, bloodbath, liquidated, dead coin

### Neutral Terms
- DYOR, NFA, DCA, DeFi, DEX, CEX, gas, whale, staking

## API Reference

### SentimentAnalyzer

```rust
impl SentimentAnalyzer {
    fn new() -> Self;
    fn with_weights(self, lexicon: f64, crypto: f64) -> Self;
    fn analyze(&self, text: &str) -> SentimentResult;
    fn analyze_batch(&self, texts: &[&str]) -> Vec<SentimentResult>;
    fn analyze_with_context(&self, text: &str, ctx: &AnalysisContext) -> SentimentResult;
}
```

### SignalGenerator

```rust
impl SignalGenerator {
    fn new() -> Self;
    fn generate(&self, result: &SentimentResult) -> GeneratedSignal;
    fn generate_batch(&self, results: &[SentimentResult]) -> Vec<GeneratedSignal>;
    fn generate_aggregated(&self, results: &[SentimentResult]) -> GeneratedSignal;
}
```

### BybitClient

```rust
impl BybitClient {
    fn new() -> Self;
    async fn get_ticker(&self, symbol: &str) -> Result<Ticker, BybitError>;
    async fn get_klines(&self, symbol: &str, interval: Interval,
                        start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Vec<Candle>, BybitError>;
    async fn get_symbols(&self) -> Result<Vec<String>, BybitError>;
}
```

## License

MIT

## Related

- [Python Implementation](../python/) - Python version with FinBERT and GPT integration
- [Main Chapter](../README.md) - Chapter documentation
