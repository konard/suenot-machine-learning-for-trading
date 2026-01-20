# LLM Portfolio Construction - Rust Implementation

This module provides Rust implementations for LLM-based portfolio construction, supporting both cryptocurrency (Bybit) and stock market (Yahoo Finance) data.

## Features

- **Bybit Client**: Fetch cryptocurrency OHLCV data from Bybit exchange
- **Stock Client**: Fetch stock market data from Yahoo Finance
- **LLM Portfolio Engine**: Analyze assets and generate portfolios using LLM
- **Portfolio Optimization**: Mean-variance and risk parity allocation strategies
- **Async/Await**: Full async support with Tokio runtime

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
llm_portfolio = { path = "path/to/rust_llm_portfolio" }
tokio = { version = "1.0", features = ["full"] }
```

## Quick Start

### Cryptocurrency Portfolio

```rust
use llm_portfolio::{
    data::bybit::BybitClient,
    llm::engine::LLMPortfolioEngine,
    portfolio::{Asset, AssetClass},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fetch crypto data
    let client = BybitClient::new();
    let btc_data = client.get_ohlcv("BTCUSDT", "D", 30).await?;

    // Define assets
    let assets = vec![
        Asset {
            symbol: "BTCUSDT".to_string(),
            name: "Bitcoin".to_string(),
            asset_class: AssetClass::Crypto,
            current_price: 65000.0,
        },
        Asset {
            symbol: "ETHUSDT".to_string(),
            name: "Ethereum".to_string(),
            asset_class: AssetClass::Crypto,
            current_price: 3200.0,
        },
    ];

    // Analyze with LLM (mock mode)
    let engine = LLMPortfolioEngine::new(None);
    let scores = engine.analyze_assets_mock(&assets, &HashMap::new(), &[]);
    let portfolio = engine.generate_portfolio(&scores, 0.05);

    println!("Portfolio weights: {:?}", portfolio.weights);
    Ok(())
}
```

### Stock Portfolio

```rust
use llm_portfolio::{
    data::stock::{StockClient, StockPortfolioDataFetcher},
    llm::engine::LLMPortfolioEngine,
    portfolio::{Asset, AssetClass},
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fetch stock data
    let client = StockClient::new();
    let fetcher = StockPortfolioDataFetcher::new(client);
    let data = fetcher.fetch_portfolio_data(&["AAPL", "MSFT", "GOOGL"], "1y").await?;

    // Define assets
    let assets = vec![
        Asset {
            symbol: "AAPL".to_string(),
            name: "Apple Inc".to_string(),
            asset_class: AssetClass::Equity,
            current_price: 185.0,
        },
        Asset {
            symbol: "MSFT".to_string(),
            name: "Microsoft".to_string(),
            asset_class: AssetClass::Equity,
            current_price: 420.0,
        },
    ];

    // Analyze and generate portfolio
    let engine = LLMPortfolioEngine::new(None);
    let scores = engine.analyze_assets_mock(&assets, &HashMap::new(), &[]);
    let portfolio = engine.generate_portfolio(&scores, 0.05);

    println!("Portfolio: {:?}", portfolio);
    Ok(())
}
```

## Running Examples

```bash
# Cryptocurrency portfolio
cargo run --example crypto_portfolio

# Stock portfolio
cargo run --example stock_portfolio
```

## Module Structure

```
src/
├── lib.rs              # Library entry point
├── data/
│   ├── mod.rs          # Data module exports
│   ├── bybit.rs        # Bybit exchange client
│   └── stock.rs        # Stock market client
├── portfolio/
│   ├── mod.rs          # Portfolio module exports
│   ├── types.rs        # Asset, Portfolio, Score types
│   └── optimizer.rs    # Optimization algorithms
└── llm/
    ├── mod.rs          # LLM module exports
    └── engine.rs       # LLM portfolio engine
```

## LLM Integration

To use real LLM analysis (not mock), set your API key:

```rust
use std::env;

let api_key = env::var("OPENAI_API_KEY").ok();
let engine = LLMPortfolioEngine::new(api_key);

// Or use a custom endpoint (for local models)
let engine = LLMPortfolioEngine::with_base_url(
    Some("your-api-key".to_string()),
    "http://localhost:8080/v1".to_string(),
    "local-model".to_string(),
);
```

## Asset Scoring

The LLM analyzes each asset across four dimensions:

| Score | Description |
|-------|-------------|
| **Fundamental** | Company/project fundamentals, financials, technology |
| **Momentum** | Price trends and technical indicators |
| **Sentiment** | News, social media, and market sentiment |
| **Risk** | Volatility, regulatory risks, etc. (higher = riskier) |

Each score ranges from 1.0 to 10.0, with an overall score calculated as a weighted combination.

## Portfolio Generation Strategies

### Score-Weighted Allocation

Assets are weighted proportionally to their overall scores:

```rust
let portfolio = engine.generate_portfolio(&scores, 0.05); // 5% min weight
```

### Mean-Variance Optimization

```rust
use llm_portfolio::portfolio::optimizer::MeanVarianceOptimizer;

let optimizer = MeanVarianceOptimizer::new(target_return);
let result = optimizer.optimize(&returns_matrix, &expected_returns)?;
```

## Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture
```

## Dependencies

- `reqwest`: HTTP client for API calls
- `serde`: Serialization/deserialization
- `tokio`: Async runtime
- `ndarray`: Numerical arrays for optimization
- `chrono`: Date/time handling

## License

MIT
