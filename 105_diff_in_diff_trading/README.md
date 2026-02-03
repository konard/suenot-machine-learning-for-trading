# Chapter 105: Difference-in-Differences (DiD) for Trading

## Overview

Difference-in-Differences (DiD) is a quasi-experimental statistical technique used in econometrics to estimate causal effects by comparing changes in outcomes over time between a treatment group and a control group. Originally developed for policy evaluation, DiD has become a powerful tool in financial markets for measuring the impact of events, regulations, and policies on asset prices.

This chapter implements DiD methods for cryptocurrency trading on Bybit and stock market analysis, demonstrating how causal inference techniques can be applied to identify trading opportunities arising from regulatory changes, policy announcements, and market events.

## Table of Contents

1. [Theoretical Foundations](#theoretical-foundations)
2. [DiD Architecture](#did-architecture)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Implementation](#implementation)
5. [Trading Strategy](#trading-strategy)
6. [Results and Metrics](#results-and-metrics)
7. [References](#references)

## Theoretical Foundations

### What is Difference-in-Differences?

DiD is a research design that estimates causal effects by comparing the difference in outcomes over time between units that receive a treatment and units that do not. The key insight is that by taking a "difference of differences," we can control for time-invariant unobserved confounders and common time trends.

```
DiD Estimate = (Y_treated,after - Y_treated,before) - (Y_control,after - Y_control,before)
```

### The Parallel Trends Assumption

The fundamental identifying assumption of DiD is that, in the absence of treatment, the treated and control groups would have followed parallel trends:

```
E[Y(0)_t | D=1] - E[Y(0)_{t-1} | D=1] = E[Y(0)_t | D=0] - E[Y(0)_{t-1} | D=0]
```

Where:
- `Y(0)` is the potential outcome without treatment
- `D` indicates treatment status (1 = treated, 0 = control)
- The subscripts denote time periods

### Why DiD for Trading?

Financial markets offer numerous natural experiments for DiD analysis:

| Event Type | Treatment Group | Control Group | Application |
|---|---|---|---|
| Regulatory changes | Affected assets | Unaffected assets | Measure compliance costs |
| Policy announcements | Target sector | Other sectors | Identify market impact |
| Exchange listings | Listed token | Similar tokens | Estimate listing premium |
| Earnings surprises | Announcing company | Industry peers | Isolate company-specific effect |
| Macroeconomic events | Exposed economies | Unexposed economies | Quantify shock transmission |

### Advantages Over Simple Event Studies

Traditional event studies measure abnormal returns around an event but struggle with:
- Selection bias (which firms are affected is often non-random)
- Confounding events occurring simultaneously
- Market-wide trends unrelated to the event

DiD addresses these by explicitly constructing a counterfactual using control units.

## DiD Architecture

```
Pre-Period Data                    Post-Period Data
     │                                  │
     ▼                                  ▼
┌─────────────────┐              ┌─────────────────┐
│ Treatment Group │              │ Treatment Group │
│ (Affected)      │              │ (Affected)      │
│    Y_T,pre      │              │    Y_T,post     │
└─────────────────┘              └─────────────────┘
     │                                  │
     │  Δ_T = Y_T,post - Y_T,pre        │
     └──────────────┬───────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   DiD = Δ_T - Δ_C
            │  (Causal Effect)
            └───────────────┘
                    ▲
     ┌──────────────┴───────────────────┐
     │  Δ_C = Y_C,post - Y_C,pre        │
┌─────────────────┐              ┌─────────────────┐
│  Control Group  │              │  Control Group  │
│  (Unaffected)   │              │  (Unaffected)   │
│    Y_C,pre      │              │    Y_C,post     │
└─────────────────┘              └─────────────────┘
```

### Components

1. **Treatment Identification**: Determine which assets are affected by an event
2. **Control Selection**: Choose comparable assets unaffected by the event
3. **Pre-Period Analysis**: Establish baseline trends and verify parallel trends
4. **Post-Period Analysis**: Measure outcomes after treatment
5. **Estimation**: Compute DiD estimate with standard errors

### Extensions

1. **Staggered DiD**: When treatment occurs at different times for different units
2. **Triple Differences (DDD)**: Adding a third differencing dimension
3. **Synthetic Control DiD**: Constructing optimal control weights
4. **Continuous DiD**: When treatment intensity varies

## Mathematical Formulation

### Basic Two-Period Model

For a simple two-period, two-group design:

```
Y_it = α + β·Treat_i + γ·Post_t + δ·(Treat_i × Post_t) + ε_it
```

Where:
- `Y_it`: Outcome for unit i at time t
- `Treat_i`: 1 if unit is in treatment group, 0 otherwise
- `Post_t`: 1 if time period is after treatment, 0 otherwise
- `δ`: The DiD estimator (causal effect of treatment)
- `ε_it`: Error term

### Generalized Multi-Period Model

With multiple time periods and two-way fixed effects (TWFE):

```
Y_it = α_i + λ_t + δ·D_it + X_it·β + ε_it
```

Where:
- `α_i`: Unit fixed effects (time-invariant characteristics)
- `λ_t`: Time fixed effects (common trends)
- `D_it`: Treatment indicator (1 if unit i is treated at time t)
- `X_it`: Time-varying covariates
- `δ`: Average treatment effect on the treated (ATT)

### Parallel Trends Test

To validate the parallel trends assumption, estimate:

```
Y_it = α_i + λ_t + Σ_k δ_k·1(t=k)·Treat_i + ε_it
```

Where `k` indexes event time relative to treatment. The pre-treatment coefficients `δ_k` for `k < 0` should be statistically indistinguishable from zero.

### Standard Errors

For panel data with potential serial correlation, cluster standard errors at the unit level:

```
V̂(δ) = (X'X)^{-1} (Σ_i X_i' ε̂_i ε̂_i' X_i) (X'X)^{-1}
```

## Implementation

### Python

The Python implementation uses PyTorch and includes:

- **`python/model.py`**: DiD estimation with regression and matching methods
- **`python/data_loader.py`**: Data loading for stock (yfinance) and crypto (Bybit) markets
- **`python/backtest.py`**: Backtesting engine for DiD-based trading strategies

```python
from python.model import DifferenceInDifferences

model = DifferenceInDifferences(
    treatment_col='treated',
    time_col='post_treatment',
    outcome_col='return',
    covariates=['volume', 'volatility', 'market_cap'],
    cluster_col='asset_id',
    n_bootstrap=1000
)

# Fit the model
results = model.fit(panel_data)

# Generate trading signals
signals = model.generate_signals(
    threshold=0.02,      # Minimum effect size
    confidence=0.95,     # Confidence level
    holding_period=5     # Days to hold position
)
```

### Rust

The Rust implementation provides a production-ready version:

- **`src/model/`**: DiD estimation with efficient matrix operations
- **`src/data/`**: Bybit API client and feature engineering
- **`src/trading/`**: Signal generation and strategy execution
- **`src/backtest/`**: Performance evaluation engine

```bash
# Run basic example
cargo run --example basic_did

# Run trading strategy
cargo run --example trading_strategy

# Run event study analysis
cargo run --example event_study
```

## Trading Strategy

### Signal Generation

The DiD-based trading strategy identifies opportunities from:

1. **Regulatory Events**: New regulations affecting specific sectors
2. **Exchange Listings**: Crypto tokens added to major exchanges
3. **Policy Changes**: Central bank decisions, fiscal announcements
4. **Corporate Events**: Earnings, M&A, management changes

### Entry Rules

- **Long**: DiD estimate > threshold AND statistically significant (p < 0.05)
- **Short**: DiD estimate < -threshold AND statistically significant
- **Flat**: Effect not significant or below threshold

### Position Sizing

Kelly-criterion inspired sizing based on effect magnitude and confidence:

```
position_size = (DiD_estimate / std_error) × (confidence - 0.5) × scale_factor
```

### Risk Management

- Maximum position size: 15% of portfolio per event
- Stop-loss: 2× estimated volatility from control group
- Take-profit: 3× DiD effect estimate
- Maximum concurrent events: 5
- Minimum pre-treatment window: 30 days

### Implementation Details

1. **Event Detection**: Monitor news feeds, regulatory filings, exchange announcements
2. **Control Construction**: Use propensity score matching or synthetic control
3. **Pre-Trend Validation**: Verify parallel trends before trading
4. **Continuous Monitoring**: Update estimates as new data arrives

## Results and Metrics

### Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE / MAE | Prediction accuracy for effect magnitude |
| Coverage Rate | Confidence interval coverage |
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted return |
| Maximum Drawdown | Worst peak-to-trough decline |
| Hit Rate | Percentage of correct directional predictions |
| Profit Factor | Gross profit / Gross loss |

### Comparison with Baselines

The DiD approach is compared against:
- Simple event study (abnormal returns)
- Propensity score matching without DiD
- Buy-and-hold benchmark
- Market-neutral strategy

### Empirical Applications

Example use cases included in this chapter:

1. **Crypto Exchange Listings**: Measuring the price impact when tokens are listed on Binance/Bybit
2. **Stock Regulation**: Impact of SEC enforcement actions on related stocks
3. **Monetary Policy**: Effect of FOMC announcements on sector ETFs
4. **ESG Events**: Market reaction to sustainability ratings changes

## Project Structure

```
105_diff_in_diff_trading/
├── README.md                  # This file
├── README.ru.md               # Russian translation
├── readme.simple.md           # Simplified explanation (English)
├── readme.simple.ru.md        # Simplified explanation (Russian)
├── Cargo.toml                 # Rust project configuration
├── python/
│   ├── __init__.py
│   ├── model.py               # DiD estimation model
│   ├── data_loader.py         # Stock & crypto data loading
│   ├── backtest.py            # Backtesting framework
│   └── requirements.txt       # Python dependencies
├── src/
│   ├── lib.rs                 # Rust library root
│   ├── model/
│   │   ├── mod.rs             # Model module
│   │   └── did.rs             # DiD implementation
│   ├── data/
│   │   ├── mod.rs             # Data module
│   │   ├── bybit.rs           # Bybit API client
│   │   └── features.rs        # Feature engineering
│   ├── trading/
│   │   ├── mod.rs             # Trading module
│   │   ├── signals.rs         # Signal generation
│   │   └── strategy.rs        # Trading strategy
│   └── backtest/
│       ├── mod.rs             # Backtest module
│       └── engine.rs          # Backtesting engine
└── examples/
    ├── basic_did.rs           # Basic DiD example
    ├── event_study.rs         # Event study analysis
    └── trading_strategy.rs    # Full trading strategy
```

## References

1. **Goodman-Bacon, A. (2021)**. Difference-in-Differences with Variation in Treatment Timing. *Journal of Econometrics*, 225(2), 254-277. https://www.sciencedirect.com/science/article/pii/S0304407621001445

2. **Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023)**. What's Trending in Difference-in-Differences? A Synthesis of the Recent Econometrics Literature. *Journal of Econometrics*, 235(2), 2218-2244. https://arxiv.org/abs/2201.01194

3. **de Chaisemartin, C., & D'Haultfœuille, X. (2020)**. Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects. *American Economic Review*, 110(9), 2964-2996.

4. **Callaway, B., & Sant'Anna, P. H. C. (2021)**. Difference-in-Differences with Multiple Time Periods. *Journal of Econometrics*, 225(2), 200-230.

5. **Abadie, A. (2005)**. Semiparametric Difference-in-Differences Estimators. *Review of Economic Studies*, 72(1), 1-19.

6. **Baker, A. C., Larcker, D. F., & Wang, C. C. (2022)**. How Much Should We Trust Staggered Difference-In-Differences Estimates? *Journal of Financial Economics*, 144(2), 370-395.

7. **Sun, L., & Abraham, S. (2021)**. Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects. *Journal of Econometrics*, 225(2), 175-199.

8. **MacKinlay, A. C. (1997)**. Event Studies in Economics and Finance. *Journal of Economic Literature*, 35(1), 13-39.

9. **De Prado, M. L. (2018)**. Advances in Financial Machine Learning. *Wiley*.

## License

MIT
