# LLM Risk Assessment - Rust Implementation

A high-performance Rust implementation for LLM-based risk assessment in trading.

## Features

- **Bybit Integration** - Fetch real-time and historical cryptocurrency data
- **LLM Risk Scoring** - Multi-dimensional risk assessment using LLMs
- **Trading Strategies** - Risk-based signal generation and position sizing
- **Backtesting** - Comprehensive strategy validation framework
- **CLI Interface** - Easy-to-use command-line tool

## Project Structure

```
rust_llm_risk/
├── src/
│   ├── lib.rs           # Library exports
│   ├── main.rs          # CLI application
│   ├── data/            # Data fetching module
│   │   ├── mod.rs
│   │   ├── ohlcv.rs     # OHLCV data structures
│   │   ├── bybit_client.rs  # Bybit API client
│   │   └── news.rs      # News data structures
│   ├── llm/             # LLM integration module
│   │   ├── mod.rs
│   │   ├── client.rs    # LLM API client
│   │   └── provider.rs  # Provider configurations
│   ├── risk/            # Risk assessment module
│   │   ├── mod.rs
│   │   ├── score.rs     # Risk score structures
│   │   └── assessor.rs  # LLM-based risk assessor
│   ├── strategy/        # Trading strategy module
│   │   ├── mod.rs
│   │   ├── signal.rs    # Trading signals
│   │   ├── trader.rs    # Risk-based trader
│   │   └── backtest.rs  # Backtesting framework
│   └── utils/           # Utility module
│       ├── mod.rs
│       └── config.rs    # Configuration management
├── examples/
│   ├── fetch_data.rs    # Data fetching example
│   ├── assess_risk.rs   # Risk assessment example
│   └── backtest.rs      # Backtesting example
└── data/                # Local data storage
```

## Quick Start

### Build

```bash
cargo build --release
```

### Generate Configuration

```bash
cargo run -- config --output config.toml
```

Edit `config.toml` to configure your LLM provider (OpenAI, Anthropic, or local).

### Fetch Market Data

```bash
# Fetch BTCUSDT and ETHUSDT data for 7 days
cargo run -- fetch --symbols BTCUSDT,ETHUSDT --days 7 --interval 60

# Fetch with hourly candles
cargo run -- fetch -s BTCUSDT -d 30 -i 60
```

### Assess Risk

```bash
# Analyze text for risk
cargo run -- assess --text "Bitcoin faces regulatory scrutiny as SEC announces investigation" --symbol BTCUSDT
```

### Run Backtest

```bash
# Backtest on BTCUSDT with 30 days of data
cargo run -- backtest --symbol BTCUSDT --days 30 --capital 10000
```

## Examples

### Fetch Data Example

```bash
cargo run --example fetch_data
```

Demonstrates fetching market data from Bybit and performing basic analysis.

### Risk Assessment Example

```bash
# Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
cargo run --example assess_risk
```

Shows how to use the LLM-based risk assessor.

### Backtesting Example

```bash
cargo run --example backtest
```

Runs a complete backtest with risk-based trading signals.

## Configuration

Create a `config.toml` file:

```toml
[llm]
provider = "openai"  # or "anthropic", "local"
model = "gpt-4"
temperature = 0.3
max_tokens = 1000
# api_key = "your-key"  # Or use environment variable

[trading]
risk_threshold_long = 4.0
risk_threshold_short = 7.0
stop_loss = 0.02
take_profit = 0.04
max_position_size = 1.0

[data]
symbols = ["BTCUSDT", "ETHUSDT"]
default_interval = "60"
```

## API Reference

### BybitClient

```rust
use llm_risk_assessment::data::BybitClient;

let client = BybitClient::new();

// Fetch klines
let klines = client.fetch_klines("BTCUSDT", "60", 100).await?;

// Fetch historical data
let historical = client.fetch_historical_klines("BTCUSDT", "D", 30).await?;

// Fetch multiple symbols
let multi_data = client.fetch_multi_symbol(&["BTCUSDT", "ETHUSDT"], "60", 7).await?;
```

### RiskAssessor

```rust
use llm_risk_assessment::llm::{LLMClient, LLMProvider};
use llm_risk_assessment::risk::RiskAssessor;

let provider = LLMProvider::OpenAI {
    model: "gpt-4".to_string(),
    api_key: "your-key".to_string(),
};

let llm_client = LLMClient::new(provider)
    .with_temperature(0.3)
    .with_max_tokens(1000);

let assessor = RiskAssessor::new(llm_client);

let score = assessor.assess("Bitcoin faces regulatory scrutiny...").await?;
println!("Risk: {}", score.overall());
```

### RiskBasedTrader

```rust
use llm_risk_assessment::strategy::RiskBasedTrader;

let mut trader = RiskBasedTrader::new()
    .with_risk_threshold_long(4.0)
    .with_risk_threshold_short(7.0)
    .with_stop_loss(0.02)
    .with_take_profit(0.04);

let signal = trader.generate_signal(&risk_score, current_price);
```

### Backtester

```rust
use llm_risk_assessment::strategy::{Backtester, BacktestConfig};

let config = BacktestConfig {
    initial_capital: 10000.0,
    commission_rate: 0.001,
    slippage: 0.0005,
    ..Default::default()
};

let backtester = Backtester::with_config(config);
let result = backtester.run(&price_data, &risk_scores, &mut trader);

result.print_summary();
```

## Risk Dimensions

The system evaluates six risk dimensions:

| Dimension | Description |
|-----------|-------------|
| Market Risk | Exposure to market volatility |
| Credit Risk | Counterparty default probability |
| Liquidity Risk | Ability to trade without impact |
| Operational Risk | System/process failures |
| Regulatory Risk | Legal/compliance exposure |
| Sentiment Risk | Market perception shifts |

Each dimension is scored from 1 (lowest risk) to 10 (highest risk).

## Performance Metrics

The backtester calculates:

- **Total Return** - Percentage gain/loss
- **Sharpe Ratio** - Risk-adjusted return
- **Sortino Ratio** - Downside risk-adjusted return
- **Maximum Drawdown** - Largest peak-to-trough decline
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross profit / Gross loss

## Dependencies

- `reqwest` - HTTP client for API calls
- `serde` / `serde_json` - Serialization
- `tokio` - Async runtime
- `chrono` - Date/time handling
- `ndarray` - Numerical arrays
- `clap` - CLI argument parsing
- `tracing` - Logging
- `anyhow` - Error handling

## License

MIT License - See LICENSE file for details.
