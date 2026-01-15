# Chapter 32: Cross-Asset Momentum — Global Tactical Asset Allocation

## Overview

Cross-Asset Momentum applies momentum strategies across various asset classes: cryptocurrencies, traditional equities, bonds, commodities, and currencies. This approach diversifies alpha sources and reduces correlation with traditional single-asset-class momentum strategies.

<p align="center">
<img src="https://i.imgur.com/XqKxZ8N.png" width="70%">
</p>

## Table of Contents

1. [What is Momentum](#what-is-momentum)
   * [Intuition Behind Momentum](#intuition-behind-momentum)
   * [Time-Series Momentum](#time-series-momentum)
   * [Cross-Sectional Momentum](#cross-sectional-momentum)
2. [Dual Momentum Strategy](#dual-momentum-strategy)
   * [Combining Approaches](#combining-approaches)
   * [Drawdown Protection](#drawdown-protection)
3. [Cryptocurrency Implementation](#cryptocurrency-implementation)
   * [Asset Selection](#asset-selection)
   * [Signal Calculation](#signal-calculation)
   * [Position Management](#position-management)
4. [Code Examples](#code-examples)
   * [Rust Implementation](#rust-implementation)
   * [Python Notebooks](#python-notebooks)
5. [Backtesting](#backtesting)
6. [Resources](#resources)

## What is Momentum

Momentum is the tendency for assets that have performed well in the past to continue rising, and vice versa. It is one of the most persistent anomalies in financial markets, documented in academic literature since 1993.

### Intuition Behind Momentum

Why does momentum work? There are several explanations:

1. **Behavioral Factors:**
   - Investors react slowly to new information
   - Herding effect amplifies trends
   - Confirmation bias — people seek validation of their positions

2. **Structural Factors:**
   - Institutional investors buy/sell gradually
   - Funds follow benchmarks with a lag
   - Rebalancing creates predictable flows

3. **Risk Premium:**
   - Momentum assets may carry hidden risk of sharp reversals
   - Investors receive a premium for this risk

### Time-Series Momentum (TSM)

Time-series momentum compares an asset with itself in the past:

```
TSM Signal = Asset return over period > 0
```

- **Long:** if return over period is positive
- **Cash/Short:** if return over period is negative

Advantages of TSM:
- Simple calculation
- Protection from falling markets (exit to cash)
- Works independently for each asset

```python
def time_series_momentum(prices, lookback=252):
    """
    Calculate time-series momentum

    Args:
        prices: Asset prices
        lookback: Period in days (252 = 1 year)

    Returns:
        signal: 1 (long) or 0 (cash)
    """
    returns = prices.pct_change(lookback)
    signal = (returns > 0).astype(int)
    return signal
```

### Cross-Sectional Momentum (CSM)

Cross-sectional momentum compares assets with each other:

```
CSM Signal = Rank of asset among all assets by return
```

- **Long:** assets in top quartile/decile
- **Short:** assets in bottom quartile/decile

Advantages of CSM:
- Always have positions (market-neutral possible)
- Uses relative strength
- Diversification across assets

```python
def cross_sectional_momentum(returns_df, top_n=3, bottom_n=3):
    """
    Calculate cross-sectional momentum

    Args:
        returns_df: DataFrame with asset returns
        top_n: Number of best assets to buy
        bottom_n: Number of worst assets to short

    Returns:
        signals: DataFrame with signals (-1, 0, 1)
    """
    # Rank assets
    ranks = returns_df.rank(axis=1, ascending=False)

    # Long for top_n, short for bottom_n
    n_assets = returns_df.shape[1]
    signals = pd.DataFrame(0, index=returns_df.index, columns=returns_df.columns)
    signals[ranks <= top_n] = 1
    signals[ranks > n_assets - bottom_n] = -1

    return signals
```

## Dual Momentum Strategy

### Combining Approaches

Dual Momentum, developed by Gary Antonacci, combines both types of momentum:

```
Position = TSM × CSM
```

1. **Step 1 (Absolute Momentum):** Check if asset is better than risk-free rate
2. **Step 2 (Relative Momentum):** Among those passing the filter, select the best

This gives the best of both worlds:
- Protection from falling markets (from TSM)
- Selection of best assets (from CSM)

```python
def dual_momentum(prices_df, risk_free_rate, lookback=252, top_n=3):
    """
    Dual Momentum strategy

    Args:
        prices_df: DataFrame with asset prices
        risk_free_rate: Risk-free rate (annual)
        lookback: Period for momentum calculation
        top_n: Number of assets to buy

    Returns:
        weights: Portfolio weights
    """
    returns = prices_df.pct_change(lookback)

    # Step 1: Absolute momentum filter
    excess_returns = returns - risk_free_rate
    passed_filter = excess_returns > 0

    # Step 2: Relative momentum ranking
    filtered_returns = returns.where(passed_filter, -np.inf)
    ranks = filtered_returns.rank(axis=1, ascending=False)

    # Select top_n assets
    weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    weights[ranks <= top_n] = 1.0 / top_n

    # If all assets filtered out - go to cash
    all_filtered = ~passed_filter.any(axis=1)
    weights.loc[all_filtered] = 0

    return weights
```

### Drawdown Protection

One of the main advantages of Dual Momentum is protection from large drawdowns:

| Event | S&P 500 | Dual Momentum |
|-------|---------|---------------|
| Dot-com crash (2000-02) | -49% | -10% |
| Financial crisis (2008-09) | -57% | -15% |
| COVID crash (2020) | -34% | -12% |
| Crypto winter (2022) | N/A | -25% |

This is achieved by:
1. Exiting to cash with negative absolute momentum
2. Switching to defensive assets
3. Avoiding the worst performers

## Cryptocurrency Implementation

### Asset Selection

For the cryptocurrency market, we use the following asset universe:

```
Cryptocurrencies (Bybit):
├── BTCUSDT  - Bitcoin
├── ETHUSDT  - Ethereum
├── SOLUSDT  - Solana
├── BNBUSDT  - Binance Coin
├── XRPUSDT  - Ripple
├── ADAUSDT  - Cardano
├── AVAXUSDT - Avalanche
├── DOTUSDT  - Polkadot
├── MATICUSDT- Polygon
├── LINKUSDT - Chainlink
└── ATOMUSDT - Cosmos

Stablecoins (risk-free asset):
├── USDT - Tether
└── USDC - USD Coin
```

### Signal Calculation

For cryptocurrencies, shorter periods are used due to high volatility:

```python
# Periods for momentum calculation (in days)
LOOKBACK_PERIODS = {
    'short': 7,      # 1 week
    'medium': 30,    # 1 month
    'long': 90,      # 3 months
}

# Weights for combined signal
PERIOD_WEIGHTS = {
    'short': 0.3,
    'medium': 0.4,
    'long': 0.3,
}

def calculate_crypto_momentum(prices, skip_days=1):
    """
    Calculate momentum for cryptocurrencies

    Args:
        prices: DataFrame with prices
        skip_days: Skip last N days (to avoid mean reversion)

    Returns:
        momentum: Combined momentum signal
    """
    momentum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for name, days in LOOKBACK_PERIODS.items():
        # Skip last days
        shifted = prices.shift(skip_days)
        returns = shifted.pct_change(days)

        momentum += returns * PERIOD_WEIGHTS[name]

    return momentum
```

### Position Management

For the cryptocurrency market, risk management is especially important:

```python
def volatility_adjusted_weights(returns, target_vol=0.30):
    """
    Calculate volatility-adjusted weights

    Crypto has high volatility, so target_vol = 30%
    """
    # Realized volatility over 30 days
    realized_vol = returns.rolling(30).std() * np.sqrt(365)

    # Raw weights inversely proportional to volatility
    raw_weights = target_vol / realized_vol

    # Cap maximum position size
    capped_weights = raw_weights.clip(upper=2.0)  # Max 2x leverage

    return capped_weights

def risk_parity_crypto(returns, max_correlation=0.7):
    """
    Risk parity with correlation adjustment

    Cryptocurrencies are often highly correlated, which is important to consider
    """
    # Covariance matrix
    cov_matrix = returns.rolling(90).cov()

    # Volatility of each asset
    vol = returns.rolling(90).std()

    # Correlation matrix
    corr_matrix = returns.rolling(90).corr()

    # Penalize highly correlated assets
    correlation_penalty = (corr_matrix.mean() / max_correlation).clip(lower=1.0)

    # Inverse volatility with correlation penalty
    inv_vol_weights = 1 / (vol * correlation_penalty)
    weights = inv_vol_weights / inv_vol_weights.sum()

    return weights
```

## Code Examples

### Rust Implementation

The [rust_momentum_crypto](rust_momentum_crypto/) directory contains a modular Rust implementation:

```
rust_momentum_crypto/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library module
│   ├── main.rs             # CLI interface
│   ├── data/
│   │   ├── mod.rs          # Data module
│   │   ├── bybit.rs        # Bybit API client
│   │   └── types.rs        # Data types (OHLCV, etc)
│   ├── momentum/
│   │   ├── mod.rs          # Momentum module
│   │   ├── timeseries.rs   # Time-series momentum
│   │   ├── crosssection.rs # Cross-sectional momentum
│   │   └── dual.rs         # Dual momentum
│   ├── strategy/
│   │   ├── mod.rs          # Strategy module
│   │   ├── signals.rs      # Signal generation
│   │   └── weights.rs      # Weight calculation
│   ├── backtest/
│   │   ├── mod.rs          # Backtest module
│   │   ├── engine.rs       # Backtest engine
│   │   └── metrics.rs      # Performance metrics
│   └── utils/
│       ├── mod.rs          # Utilities
│       └── config.rs       # Configuration
└── examples/
    ├── fetch_prices.rs     # Fetch data from Bybit
    ├── calc_momentum.rs    # Calculate momentum
    ├── run_strategy.rs     # Run strategy
    └── backtest.rs         # Full backtest
```

See [rust_momentum_crypto/README.md](rust_momentum_crypto/README.md) for details.

### Quick Start with Rust

```bash
# Clone and navigate to the project
cd 32_cross_asset_momentum/rust_momentum_crypto

# Fetch price data from Bybit
cargo run --example fetch_prices

# Calculate momentum for all assets
cargo run --example calc_momentum

# Run the full strategy
cargo run --example run_strategy

# Run a complete backtest
cargo run --example backtest
```

### Python Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_crypto_universe.ipynb` | Select cryptocurrencies for strategy |
| 2 | `02_data_collection.ipynb` | Fetch data from Bybit |
| 3 | `03_momentum_signals.ipynb` | Calculate momentum signals |
| 4 | `04_time_series_momentum.ipynb` | Time-series momentum filter |
| 5 | `05_cross_sectional_momentum.ipynb` | Cross-sectional ranking |
| 6 | `06_dual_momentum.ipynb` | Combination of TSM + CSM |
| 7 | `07_volatility_targeting.ipynb` | Volatility targeting |
| 8 | `08_risk_parity_weights.ipynb` | Risk parity allocation |
| 9 | `09_rebalancing.ipynb` | Rebalancing logic |
| 10 | `10_backtesting.ipynb` | Full backtest |
| 11 | `11_regime_analysis.ipynb` | Performance by market regimes |
| 12 | `12_ml_enhancement.ipynb` | ML for rebalancing timing |

## Backtesting

### Key Metrics

- **Returns:** CAGR, Total Return
- **Risk:** Volatility, Maximum Drawdown, VaR
- **Risk-Adjusted:** Sharpe, Sortino, Calmar
- **Momentum-Specific:** Hit Rate, Average Win/Loss, Turnover
- **Comparison:** vs Buy&Hold BTC, vs Equal Weight

### Typical Results for Cryptocurrencies

| Metric | Buy&Hold BTC | Equal Weight | Dual Momentum |
|--------|--------------|--------------|---------------|
| CAGR | 45% | 35% | 55% |
| Volatility | 75% | 60% | 40% |
| Max Drawdown | -85% | -75% | -35% |
| Sharpe Ratio | 0.6 | 0.58 | 1.35 |
| Calmar Ratio | 0.53 | 0.47 | 1.57 |

*Note: Historical results do not guarantee future performance*

### Rebalancing Rules

```
Rebalancing Schedule:
├── Weekly (Sunday 00:00 UTC)
├── Optional: daily during high volatility
└── Account for exchange fees

Rebalancing Bands:
├── Trade only if weight deviation > 10%
├── Significantly reduces turnover
└── Maintains approximate target allocation

Signal Decay:
├── Fresh signal = full weight
├── Aging signal = reduced weight
└── Prevents whipsaws at signal boundary
```

## Resources

### Books

- [Dual Momentum Investing](https://www.amazon.com/Dual-Momentum-Investing-Innovative-Strategy/dp/0071849440) (Gary Antonacci)
- [Quantitative Momentum](https://www.amazon.com/Quantitative-Momentum-Practitioners-Momentum-Based-Selection/dp/111923719X) (Wesley Gray)
- [Expected Returns](https://www.amazon.com/Expected-Returns-Investors-Harvesting-Rewards/dp/1119990726) (Antti Ilmanen)

### Academic Papers

- [Time Series Momentum](https://pages.stern.nyu.edu/~lpedMDL1/papers/TimeSeriesMomentum.pdf) (Moskowitz, Ooi, Pedersen)
- [Value and Momentum Everywhere](https://pages.stern.nyu.edu/~lpedMDL1/papers/ValMomEverywhere.pdf) (Asness, Moskowitz, Pedersen)
- [Momentum in Cryptocurrency Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690508) (Liu, Tsyvinski, Wu)

### Related Chapters

- [Chapter 22: Deep Reinforcement Learning](../22_deep_reinforcement_learning) — RL for trading
- [Chapter 28: Regime Detection with HMM](../28_regime_detection_hmm) — Market regime detection
- [Chapter 36: Crypto DEX Arbitrage](../36_crypto_dex_arbitrage) — Cryptocurrency exchange arbitrage

## Dependencies

### Python

```python
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
requests>=2.28.0
empyrical>=0.5.5   # For performance metrics
pyfolio>=0.9.2     # For tearsheets
```

### Rust

```toml
reqwest = "0.12"      # HTTP client
tokio = "1.0"         # Async runtime
serde = "1.0"         # Serialization
chrono = "0.4"        # Time handling
ndarray = "0.16"      # Arrays
```

## Difficulty Level

Intermediate

**Required knowledge:**
- Momentum factors
- Asset allocation
- Risk parity
- Portfolio construction
- Cryptocurrency markets
