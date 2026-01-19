# Multi-Agent LLM Trading (Rust)

High-performance multi-agent trading system implementation in Rust.

## Features

- **Type-safe Agent Framework**: Strongly typed agents with compile-time guarantees
- **Async Communication**: Non-blocking agent communication using Tokio
- **Bull vs Bear Debates**: Adversarial debate mechanism for better decisions
- **Backtesting Engine**: Fast backtesting with parallel execution
- **LLM Integration Ready**: Prepared for OpenAI/Anthropic API integration

## Building

```bash
cd rust_multi_agent_trading
cargo build --release
```

## Running Examples

```bash
# Agent demo
cargo run --bin agent_demo

# Debate demo
cargo run --bin debate_demo

# Backtest demo
cargo run --bin backtest_demo
```

## Architecture

```
src/
├── lib.rs          # Library exports
├── agents.rs       # Agent trait and implementations
├── communication.rs # Message passing and debates
├── backtest.rs     # Backtesting engine
├── data.rs         # Market data handling
├── error.rs        # Error types
└── bin/            # Demo executables
```

## Usage

```rust
use multi_agent_trading::{
    agents::{TechnicalAgent, BullAgent, BearAgent, TraderAgent},
    communication::Debate,
    data::MarketData,
};

// Create agents
let tech = TechnicalAgent::new("Tech-1");
let bull = BullAgent::new("Bull-1");
let bear = BearAgent::new("Bear-1");

// Run debate
let debate = Debate::new(bull, bear, 3);
let result = debate.conduct("AAPL", &market_data).await;

// Get final decision
let trader = TraderAgent::new("Trader-1");
let decision = trader.aggregate(&[tech_analysis, debate_result]);
```

## Performance

Rust implementation offers:
- 10-100x faster backtesting than Python
- Type-safe agent interactions
- Zero-cost abstractions
- Parallel analysis execution

## License

MIT
