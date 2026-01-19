# Rust CoT Trading Library

Chain-of-Thought Trading implementation in Rust.

## Features

- **Transparent Reasoning**: Every trading decision includes a full reasoning chain
- **Multi-Step Analysis**: 6-step signal generation with trend, momentum, volume analysis
- **Risk Management**: Position sizing with CoT-based risk assessment
- **Audit Trails**: Complete backtesting with trade-by-trade reasoning logs
- **Multiple Data Sources**: Support for Yahoo Finance and Bybit

## Quick Start

```rust
use cot_trading::{SignalGenerator, PositionSizer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create signal generator with mock analyzer
    let generator = SignalGenerator::new_mock();

    // Generate trading signal
    let signal = generator.generate(
        "AAPL",
        150.0,  // current price
        45.0,   // RSI
        0.5,    // MACD
        0.3,    // MACD signal
        148.0,  // SMA 20
        145.0,  // SMA 50
        1.2,    // volume ratio
        2.5,    // ATR
    ).await?;

    println!("Signal: {:?}", signal.signal_type);
    println!("Reasoning: {:?}", signal.reasoning_chain);

    Ok(())
}
```

## Building

```bash
cargo build --release
```

## Running Examples

```bash
# Basic CoT analysis demo
cargo run --bin cot_analysis_demo

# Signal generation demo
cargo run --bin signal_demo

# Backtest demo
cargo run --bin backtest_demo

# Cryptocurrency demo
cargo run --bin crypto_demo
```

## Testing

```bash
cargo test
```

## License

MIT
