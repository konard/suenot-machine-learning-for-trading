# Attention Rollout Trading - Rust Implementation

High-performance Rust implementation of Attention Rollout for interpretable trading models.

## Features

- **Attention Rollout Computation** - Recursive combination of attention matrices across transformer layers
- **Multi-head Fusion** - Mean, Max, Min strategies for combining attention heads
- **Trading Regime Detection** - Identify momentum vs mean-reversion attention patterns
- **Bybit API Integration** - Fetch real-time cryptocurrency market data
- **Backtesting Metrics** - Sharpe ratio, Sortino ratio, Maximum drawdown

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
attention_rollout_trading = { path = "." }
```

### Basic Usage

```rust
use attention_rollout_trading::{AttentionRollout, AttentionRolloutConfig, HeadFusion};
use ndarray::Array3;

// Create attention rollout engine
let config = AttentionRolloutConfig {
    head_fusion: HeadFusion::Mean,
    discard_ratio: 0.1,
    add_residual: true,
    residual_weight: 0.5,
};
let rollout = AttentionRollout::with_config(config);

// Compute rollout from attention matrices
// attentions: Vec<Array3<f64>> with shape (n_heads, seq_len, seq_len)
let rollout_matrix = rollout.compute_rollout(&attentions, 0)?;

// Get input attribution for the last position
let attribution = rollout.get_input_attribution(&attentions, seq_len - 1)?;
```

### Trading Regime Analysis

```rust
use attention_rollout_trading::{TradingAttentionAnalyzer, AttentionRegime};

let analyzer = TradingAttentionAnalyzer::new();
let analysis = analyzer.analyze(&attentions)?;

match analysis.regime {
    AttentionRegime::Momentum => println!("Model focuses on recent prices"),
    AttentionRegime::MeanReversion => println!("Model uses historical patterns"),
    AttentionRegime::Mixed => println!("Balanced attention distribution"),
}
```

### Fetching Bybit Data

```rust
use attention_rollout_trading::data::fetch_bybit_candles;

let candles = fetch_bybit_candles("BTCUSDT", "D", 100)?;
for candle in &candles {
    println!("Close: {:.2}, Volume: {:.2}", candle.close, candle.volume);
}
```

## Running Examples

```bash
# Full trading example with Bybit data, attention rollout, and backtesting
cargo run --example trading_example
```

## Module Structure

| Module | Description |
|--------|-------------|
| `AttentionRollout` | Core rollout computation engine |
| `TradingAttentionAnalyzer` | Trading-specific attention analysis |
| `data` | Bybit API client for market data |
| `backtest` | Performance metrics (Sharpe, Sortino, Max DD) |

## Running Tests

```bash
cargo test
```

## Important Notes

- This is an **inference-only** implementation - model training should be done in Python (PyTorch)
- The Rust implementation focuses on high-performance computation for production use
- Attention matrices are expected as input (from a trained transformer model)
- For model training, refer to the Python implementation in `../python/`
