# Bybit Client - Reusable Cryptocurrency Market Data Fetcher

A robust, production-ready Rust library for fetching cryptocurrency market data from Bybit exchange.

## Features

- ✅ Historical candlestick (OHLCV) data with multiple timeframes
- ✅ Real-time ticker information
- ✅ Symbol listing and discovery
- ✅ Built-in rate limiting (120 requests/minute)
- ✅ Automatic pagination for large date ranges
- ✅ Async/await with Tokio
- ✅ Comprehensive error handling
- ✅ Type-safe API with serde

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
bybit_client = { path = "../../../.claude/rust_templates/bybit_client" }
tokio = { version = "1.0", features = ["full"] }
chrono = "0.4"
anyhow = "1.0"
```

## Quick Start

### Fetch Historical Data

```rust
use bybit_client::{BybitClient, Interval};
use chrono::{Utc, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = BybitClient::new();

    // Fetch last 7 days of 1-hour candles
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(7);

    let candles = client
        .get_klines("BTCUSDT", Interval::OneHour, start_time, end_time)
        .await?;

    println!("Fetched {} candles", candles.len());

    for candle in candles.iter().take(5) {
        println!(
            "Time: {}, Open: {}, High: {}, Low: {}, Close: {}, Volume: {}",
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        );
    }

    Ok(())
}
```

### Get Current Market Ticker

```rust
use bybit_client::BybitClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = BybitClient::new();

    let ticker = client.get_ticker("BTCUSDT").await?;

    println!("Symbol: {}", ticker.symbol);
    println!("Last Price: ${}", ticker.last_price);
    println!("24h Change: {}%", ticker.price_change_percent_24h);
    println!("24h Volume: {}", ticker.volume_24h);

    Ok(())
}
```

### List Available Symbols

```rust
use bybit_client::BybitClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = BybitClient::new();

    let symbols = client.get_symbols().await?;

    println!("Found {} trading pairs", symbols.len());

    // Print USDT pairs
    for symbol in symbols.iter().filter(|s| s.ends_with("USDT")).take(10) {
        println!("  - {}", symbol);
    }

    Ok(())
}
```

## Supported Timeframes

```rust
use bybit_client::Interval;

Interval::OneMinute       // 1m
Interval::ThreeMinutes    // 3m
Interval::FiveMinutes     // 5m
Interval::FifteenMinutes  // 15m
Interval::ThirtyMinutes   // 30m
Interval::OneHour         // 1h
Interval::TwoHours        // 2h
Interval::FourHours       // 4h
Interval::SixHours        // 6h
Interval::TwelveHours     // 12h
Interval::OneDay          // 1D
Interval::OneWeek         // 1W
Interval::OneMonth        // 1M
```

## Data Structures

### Candle

```rust
pub struct Candle {
    pub timestamp: i64,    // Unix timestamp in milliseconds
    pub open: f64,         // Opening price
    pub high: f64,         // Highest price
    pub low: f64,          // Lowest price
    pub close: f64,        // Closing price
    pub volume: f64,       // Trading volume
    pub turnover: f64,     // Volume * Price
}
```

### Ticker

```rust
pub struct Ticker {
    pub symbol: String,
    pub last_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub volume_24h: f64,
    pub price_change_percent_24h: f64,
}
```

## Rate Limiting

The client automatically handles rate limiting:
- **Default**: 120 requests per minute
- **Automatic delays**: Waits between requests if limit approached
- **Thread-safe**: Safe to use across async tasks

## Error Handling

All methods return `Result<T, anyhow::Error>` for easy error propagation:

```rust
use bybit_client::BybitClient;

#[tokio::main]
async fn main() {
    let client = BybitClient::new();

    match client.get_ticker("INVALID_SYMBOL").await {
        Ok(ticker) => println!("Price: {}", ticker.last_price),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Advanced Usage

### Fetch Large Date Ranges

The client automatically paginates requests:

```rust
use bybit_client::{BybitClient, Interval};
use chrono::{Utc, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = BybitClient::new();

    // Fetch 3 months of data (will make multiple paginated requests)
    let end_time = Utc::now();
    let start_time = end_time - Duration::days(90);

    let candles = client
        .get_klines("ETHUSDT", Interval::OneHour, start_time, end_time)
        .await?;

    println!("Fetched {} candles over 90 days", candles.len());

    Ok(())
}
```

### Save Data to CSV

```rust
use bybit_client::{BybitClient, Interval};
use chrono::{Utc, Duration};
use csv::Writer;
use std::fs::File;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = BybitClient::new();

    let end_time = Utc::now();
    let start_time = end_time - Duration::days(30);

    let candles = client
        .get_klines("BTCUSDT", Interval::OneDay, start_time, end_time)
        .await?;

    let mut wtr = Writer::from_writer(File::create("btc_data.csv")?);

    wtr.write_record(&["timestamp", "open", "high", "low", "close", "volume"])?;

    for candle in candles {
        wtr.write_record(&[
            candle.timestamp.to_string(),
            candle.open.to_string(),
            candle.high.to_string(),
            candle.low.to_string(),
            candle.close.to_string(),
            candle.volume.to_string(),
        ])?;
    }

    wtr.flush()?;
    println!("Data saved to btc_data.csv");

    Ok(())
}
```

## Testing

Run tests (requires network access):

```bash
cargo test -- --ignored
```

Run without network tests:

```bash
cargo test
```

## API Documentation

This client uses Bybit's V5 REST API:
- Base URL: `https://api.bybit.com/v5`
- Documentation: https://bybit-exchange.github.io/docs/v5/intro

## License

MIT

## Contributing

This is a reusable module for the "Machine Learning for Trading" book examples.
Feel free to extend functionality as needed for your trading strategies!

## Common Use Cases

### 1. Backtest Data Collection

```rust
// Collect data for backtesting a trading strategy
let data = client
    .get_klines("BTCUSDT", Interval::FifteenMinutes, start, end)
    .await?;
```

### 2. Real-Time Monitoring

```rust
// Monitor current price
let ticker = client.get_ticker("ETHUSDT").await?;
if ticker.price_change_percent_24h > 5.0 {
    println!("ETH is up {}% in 24h!", ticker.price_change_percent_24h);
}
```

### 3. Multi-Asset Analysis

```rust
// Fetch data for multiple cryptocurrencies
let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
for symbol in symbols {
    let candles = client
        .get_klines(symbol, Interval::OneHour, start, end)
        .await?;
    println!("{}: {} candles fetched", symbol, candles.len());
}
```

## Notes

- All timestamps are in milliseconds
- Prices are in quote currency (USDT for BTCUSDT)
- Volume is in base currency (BTC for BTCUSDT)
- Data is returned sorted by timestamp (oldest first)
- Maximum 1000 candles per API request (handled automatically)
