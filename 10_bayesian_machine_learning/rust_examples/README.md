# Bayesian Machine Learning for Crypto Trading (Rust)

Rust implementation of Bayesian machine learning techniques for cryptocurrency trading, using real-time data from Bybit exchange.

## Features

- **Real-time data from Bybit** - Fetch OHLCV data for any supported trading pair
- **Modular architecture** - Clean separation between data, inference, and examples
- **MCMC sampling** - Metropolis-Hastings and adaptive MCMC implementations
- **Trading-focused** - Examples specifically designed for crypto trading applications

## Project Structure

```
rust_examples/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs                    # Library root
    ├── data/
    │   ├── mod.rs
    │   ├── bybit.rs              # Bybit API client
    │   └── returns.rs            # Returns calculation utilities
    ├── bayesian/
    │   ├── mod.rs
    │   ├── distributions.rs      # Beta, Normal, Student-t distributions
    │   ├── inference.rs          # MCMC samplers
    │   ├── linear_regression.rs  # Bayesian linear/rolling regression
    │   ├── sharpe.rs             # Bayesian Sharpe ratio
    │   └── volatility.rs         # Stochastic volatility models
    └── bin/
        ├── conjugate_priors.rs   # Price movement probability estimation
        ├── bayesian_sharpe.rs    # Sharpe ratio comparison
        ├── pairs_trading.rs      # Rolling regression for pairs
        └── stochastic_volatility.rs  # Time-varying volatility
```

## Installation

```bash
cd rust_examples
cargo build --release
```

## Usage

### 1. Conjugate Priors: Price Movement Probability

Estimate the probability of price increases using Beta-Binomial conjugate priors:

```bash
# Default: BTCUSDT hourly
cargo run --release --bin conjugate_priors

# Custom symbol and interval
cargo run --release --bin conjugate_priors -- --symbol ETHUSDT --interval 15 --limit 500

# With informative prior (e.g., expecting 60% up probability)
cargo run --release --bin conjugate_priors -- --prior-alpha 6 --prior-beta 4
```

**Output:**
- Sequential Bayesian updates
- Posterior distribution statistics
- Credible intervals
- Probability statements

### 2. Bayesian Sharpe Ratio

Compare risk-adjusted performance of two cryptocurrencies:

```bash
# Compare BTC and ETH
cargo run --release --bin bayesian_sharpe

# Compare any two symbols
cargo run --release --bin bayesian_sharpe -- -1 SOLUSDT -2 AVAXUSDT --samples 10000

# Different timeframes
cargo run --release --bin bayesian_sharpe -- --interval D --limit 365
```

**Output:**
- Full posterior distributions
- Probability of outperformance
- Effect size analysis
- Risk threshold probabilities

### 3. Pairs Trading: Rolling Bayesian Regression

Estimate time-varying hedge ratios for pairs trading:

```bash
# ETH/BTC pair (default)
cargo run --release --bin pairs_trading

# Custom pair
cargo run --release --bin pairs_trading -- -1 SOLUSDT -2 ETHUSDT --window 100

# Daily data
cargo run --release --bin pairs_trading -- --interval D --limit 365
```

**Output:**
- Rolling hedge ratio estimates with uncertainty
- Spread z-score and trading signals
- Regime change detection
- Position sizing suggestions

### 4. Stochastic Volatility Model

Model time-varying volatility in crypto markets:

```bash
# BTC volatility analysis
cargo run --release --bin stochastic_volatility

# Different asset
cargo run --release --bin stochastic_volatility -- --symbol ETHUSDT

# More MCMC samples for better estimates
cargo run --release --bin stochastic_volatility -- --samples 5000
```

**Output:**
- Volatility persistence and mean reversion parameters
- Time-varying volatility estimates with credible intervals
- Current volatility regime classification
- VaR-based risk metrics
- Volatility forecasts

## Supported Symbols

- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- SOLUSDT (Solana)
- XRPUSDT (Ripple)
- DOGEUSDT (Dogecoin)
- ADAUSDT (Cardano)
- AVAXUSDT (Avalanche)
- DOTUSDT (Polkadot)
- LINKUSDT (Chainlink)
- MATICUSDT (Polygon)

## Supported Intervals

| Interval | Description |
|----------|-------------|
| 1 | 1 minute |
| 5 | 5 minutes |
| 15 | 15 minutes |
| 30 | 30 minutes |
| 60 | 1 hour |
| 240 | 4 hours |
| D | 1 day |
| W | 1 week |
| M | 1 month |

## Library Usage

Use the library in your own Rust projects:

```rust
use bayesian_crypto::data::{BybitClient, Symbol, Returns};
use bayesian_crypto::bayesian::distributions::Beta;
use bayesian_crypto::bayesian::sharpe::BayesianSharpe;
use bayesian_crypto::bayesian::inference::MCMCConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fetch data
    let client = BybitClient::new();
    let klines = client.get_klines(Symbol::BTCUSDT, "60", 500).await?;

    // Calculate returns
    let returns = Returns::from_klines(&klines);

    // Bayesian Sharpe ratio
    let estimator = BayesianSharpe::new(8760.0); // Hourly annualization
    let config = MCMCConfig::new(5000).with_warmup(1000);
    let result = estimator.estimate(&returns.values, &config);

    println!("Sharpe Ratio: {:.4} (95% CI: [{:.4}, {:.4}])",
        result.sharpe_mean(),
        result.sharpe_ci(0.95).0,
        result.sharpe_ci(0.95).1
    );

    Ok(())
}
```

## Algorithm Details

### Conjugate Priors
Uses Beta-Binomial conjugacy for exact posterior updates:
- Prior: Beta(α, β)
- Likelihood: Binomial(n, p)
- Posterior: Beta(α + successes, β + failures)

### Bayesian Sharpe Ratio
Uses Student-t likelihood for robustness to outliers:
- Returns ~ StudentT(ν, μ, σ)
- Priors: μ ~ Normal, σ ~ HalfCauchy, ν ~ Exponential
- Inference via Adaptive Metropolis-Hastings MCMC

### Rolling Bayesian Regression
Normal-Inverse-Gamma conjugate prior for linear regression:
- β | σ² ~ Normal(μ₀, σ²V₀)
- σ² ~ InverseGamma(a₀, b₀)
- Provides closed-form posterior updates

### Stochastic Volatility
AR(1) model for log-volatility:
- rₜ = exp(hₜ/2) × εₜ, εₜ ~ N(0,1)
- hₜ = μ + φ(hₜ₋₁ - μ) + σηηₜ
- Inference via particle MCMC

## Performance Tips

1. **Use release mode**: `cargo run --release` is significantly faster
2. **Adjust samples**: Start with fewer samples for exploration, increase for final analysis
3. **Set random seed**: Use `--seed` for reproducible results
4. **Cache data**: Consider saving fetched data for repeated analysis

## License

MIT
