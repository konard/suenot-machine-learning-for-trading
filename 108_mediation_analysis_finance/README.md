# Chapter 108: Mediation Analysis for Finance

This chapter explores **Mediation Analysis**, a powerful causal inference technique for understanding the mechanisms through which one variable affects another. In financial markets, mediation analysis helps uncover the causal pathways that explain *how* factors influence returns -- whether directly or through intermediate mechanisms like sentiment, liquidity, or volatility.

<p align="center">
<img src="https://i.imgur.com/placeholder_mediation.png" width="70%" alt="Mediation Analysis Pipeline: from identifying treatment, mediator, and outcome variables through estimating direct and indirect effects to understanding causal mechanisms in financial markets">
</p>

## Contents

1. [Introduction to Mediation Analysis](#introduction-to-mediation-analysis)
    * [What is Mediation?](#what-is-mediation)
    * [Why Mediation Analysis for Trading?](#why-mediation-analysis-for-trading)
    * [Direct vs Indirect Effects](#direct-vs-indirect-effects)
2. [Core Concepts](#core-concepts)
    * [The Mediation Framework](#the-mediation-framework)
    * [Total, Direct, and Indirect Effects](#total-direct-and-indirect-effects)
    * [Types of Mediation](#types-of-mediation)
    * [Identification Assumptions](#identification-assumptions)
3. [Mathematical Foundation](#mathematical-foundation)
    * [Baron and Kenny Approach](#baron-and-kenny-approach)
    * [Causal Mediation Analysis](#causal-mediation-analysis)
    * [Potential Outcomes Framework](#potential-outcomes-framework)
    * [Natural Direct and Indirect Effects](#natural-direct-and-indirect-effects)
4. [Financial Applications](#financial-applications)
    * [Factor Return Decomposition](#factor-return-decomposition)
    * [News Sentiment Mediation](#news-sentiment-mediation)
    * [Liquidity as Mediator](#liquidity-as-mediator)
    * [Volatility Transmission Mechanisms](#volatility-transmission-mechanisms)
5. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: Classical Mediation Analysis](#02-classical-mediation-analysis)
    * [03: Causal Mediation with Sensitivity](#03-causal-mediation-with-sensitivity)
    * [04: Trading Strategy from Mediation Insights](#04-trading-strategy-from-mediation-insights)
    * [05: Backtesting](#05-backtesting)
6. [Rust Implementation](#rust-implementation)
7. [Python Implementation](#python-implementation)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to Mediation Analysis

### What is Mediation?

Mediation analysis examines the **mechanism** or **pathway** through which an independent variable (treatment) influences a dependent variable (outcome). Instead of just asking "Does X affect Y?", mediation asks "**How** does X affect Y, and what role does M play in transmitting this effect?"

```
Simple Direct Effect:

    Treatment (X) ─────────────────────► Outcome (Y)

                  "X causes Y directly"


Mediated Effect:

    Treatment (X) ────► Mediator (M) ────► Outcome (Y)
         │                                      ▲
         └──────── direct path ─────────────────┘

                  "X causes M, M causes Y"
                  "X also has a direct effect on Y"
```

In finance, we often want to understand not just that a factor predicts returns, but *how* and *why* it does so. Mediation analysis provides this mechanistic understanding.

### Why Mediation Analysis for Trading?

Traditional factor investing identifies return predictors, but doesn't explain the economic mechanism:

```
Traditional Factor Analysis:

    Earnings Surprise ──?──► Stock Returns

    "Positive earnings surprises predict positive returns"
    But WHY? Through what mechanism?


Mediation Analysis:

    Earnings Surprise ────► Analyst Revisions ────► Stock Returns
           │                                              ▲
           └──────────── sentiment change ────────────────┘

    Now we understand:
    1. Some effect works through analyst behavior (indirect)
    2. Some effect is direct (market reaction to news)
```

**Benefits for trading:**

1. **Robust alpha signals**: Understanding the mechanism helps distinguish robust factors from spurious ones
2. **Strategy timing**: If the mediator channel is blocked, the factor may not work
3. **Risk management**: Direct and indirect effects may have different risk profiles
4. **Novel factors**: Mediators themselves become candidate alpha factors
5. **Regime detection**: Changes in mediation patterns signal regime shifts

### Direct vs Indirect Effects

The key insight of mediation analysis is decomposing the **total effect** into components:

```
TOTAL EFFECT = DIRECT EFFECT + INDIRECT EFFECT

┌──────────────────────────────────────────────────────────────────────┐
│                                                                       │
│   Total Effect of Macro News on Stock Returns                        │
│                                                                       │
│   ┌─────────────────┬─────────────────┬─────────────────────────┐   │
│   │                 │                  │                         │   │
│   │  DIRECT EFFECT  │  INDIRECT EFFECT │  INDIRECT EFFECT        │   │
│   │  (30%)          │  via Volatility  │  via Sector Rotation    │   │
│   │                 │  (45%)           │  (25%)                   │   │
│   │  Market reacts  │  News → Vol ↑    │  News → Fund flows      │   │
│   │  immediately to │  → Risk premia   │  → Sector weights       │   │
│   │  information    │  → Returns       │  → Returns              │   │
│   │                 │                  │                         │   │
│   └─────────────────┴─────────────────┴─────────────────────────┘   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### The Mediation Framework

The classical mediation model involves three variables:

```
Variables:
    X = Treatment/Independent variable (the cause)
    M = Mediator (the mechanism/pathway)
    Y = Outcome/Dependent variable (the effect)

The mediation structure:

         a                    b
    X ─────────► M ─────────────► Y
    │                              ▲
    └──────────── c' ──────────────┘
              (direct effect)

Paths:
    c  = Total effect of X on Y (without M in model)
    c' = Direct effect of X on Y (controlling for M)
    a  = Effect of X on M
    b  = Effect of M on Y (controlling for X)

    Indirect effect = a × b
    Direct effect   = c'
    Total effect    = c = c' + a×b
```

### Total, Direct, and Indirect Effects

**Total Effect (c)**: The overall effect of X on Y, ignoring any mediators

```python
# Total effect regression
Y = τ₀ + c·X + ε

# Example: News sentiment → Stock returns
returns = β₀ + 0.15·sentiment + ε
#                  ↑
#           Total effect = 0.15
```

**Direct Effect (c')**: The effect of X on Y when holding M constant

```python
# Direct effect regression (controlling for mediator)
Y = τ₀ + c'·X + b·M + ε

# Example: News sentiment → Stock returns, controlling for trading volume
returns = β₀ + 0.08·sentiment + 0.12·volume + ε
#                  ↑
#           Direct effect = 0.08
```

**Indirect Effect (a×b)**: The effect of X on Y that operates through M

```python
# Step 1: Effect of X on M
M = α₀ + a·X + ε₁
# volume = β₀ + 0.58·sentiment + ε₁
#                    ↑
#             a = 0.58

# Step 2: Effect of M on Y (from direct effect regression)
# b = 0.12

# Indirect effect = a × b = 0.58 × 0.12 = 0.07

# Verification: Total = Direct + Indirect
# 0.15 ≈ 0.08 + 0.07 ✓
```

### Types of Mediation

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TYPES OF MEDIATION                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. FULL MEDIATION (c' ≈ 0)                                         │
│                                                                      │
│        a                    b                                        │
│     X ───────► M ───────────► Y                                     │
│                                                                      │
│     The entire effect of X on Y goes through M.                     │
│     When we control for M, X has no effect on Y.                    │
│                                                                      │
│     Example: Insider buying → Price pressure → Returns               │
│              (insiders don't have magic, just information → volume) │
│                                                                      │
│  2. PARTIAL MEDIATION (c' ≠ 0 and a×b ≠ 0)                         │
│                                                                      │
│        a                    b                                        │
│     X ───────► M ───────────► Y                                     │
│     │                         ▲                                      │
│     └────────── c' ───────────┘                                     │
│                                                                      │
│     X affects Y both directly AND through M.                         │
│                                                                      │
│     Example: Earnings surprise → Analyst revisions → Returns         │
│                       └──────── direct market reaction ─────→        │
│                                                                      │
│  3. NO MEDIATION (a×b ≈ 0)                                          │
│                                                                      │
│     X ───────────────────────────────────────────► Y                 │
│     │                                                                │
│     └───────► M    (M is not on the causal path)                    │
│                                                                      │
│     Example: Company bankruptcy → Stock returns                      │
│                       └──→ CEO tweets (but tweets don't cause returns)│
│                                                                      │
│  4. SUPPRESSION (opposite signs)                                     │
│                                                                      │
│     Direct and indirect effects have OPPOSITE signs.                │
│     Total effect may be smaller than either component!              │
│                                                                      │
│     Example: Risk factor → Return (positive direct)                 │
│              Risk factor → Volatility → Return (negative indirect)  │
│              Net effect might be small despite strong mechanisms    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Identification Assumptions

For mediation analysis to identify causal effects, we need:

```
CAUSAL MEDIATION ASSUMPTIONS:

1. NO UNMEASURED CONFOUNDING of X → Y

   ┌───────────────────────────────────────────┐
   │  GOOD:            │  BAD:                 │
   │                   │                       │
   │  X ────► Y        │    U (unmeasured)     │
   │  (randomized)     │    ↓         ↓        │
   │                   │    X ───?──► Y        │
   │                   │                       │
   │                   │  Can't identify c'    │
   └───────────────────┴───────────────────────┘

2. NO UNMEASURED CONFOUNDING of M → Y

   ┌───────────────────────────────────────────┐
   │  Must control for:                        │
   │                                           │
   │  X ──► M ──► Y                            │
   │        ↑     ↑                            │
   │        └─ C ─┘  (measured confounders)    │
   │                                           │
   │  Example: When analyzing if volume        │
   │  mediates sentiment→returns, must         │
   │  control for volatility (affects both)    │
   └───────────────────────────────────────────┘

3. NO UNMEASURED CONFOUNDING of X → M

   Same as above but for the X→M relationship.

4. NO EFFECT OF X ON M→Y CONFOUNDERS (Sequential Ignorability)

   ┌───────────────────────────────────────────┐
   │  PROBLEMATIC:                             │
   │                                           │
   │  X ──────► C ──────────┐                 │
   │  │                     │                 │
   │  └──► M ──────► Y ◄────┘                 │
   │                                           │
   │  If X affects confounders of M→Y,        │
   │  identification fails!                    │
   │                                           │
   │  Example: Earnings news (X) affects       │
   │  both sentiment (M) and analyst           │
   │  attention (C), where C also             │
   │  affects the M→Y relationship.           │
   └───────────────────────────────────────────┘
```

## Mathematical Foundation

### Baron and Kenny Approach

The classical (Baron & Kenny, 1986) approach uses a series of regressions:

```
BARON AND KENNY'S FOUR STEPS:

Step 1: Show X predicts Y (establish total effect)
    Y = τ₀ + c·X + ε                    Test: c ≠ 0

Step 2: Show X predicts M
    M = α₀ + a·X + ε₁                   Test: a ≠ 0

Step 3: Show M predicts Y controlling for X
    Y = τ'₀ + c'·X + b·M + ε₂          Test: b ≠ 0

Step 4: Show direct effect is reduced
    Compare c (Step 1) with c' (Step 3)
    Full mediation: c' ≈ 0
    Partial mediation: |c'| < |c| but c' ≠ 0

Statistical test for indirect effect (Sobel test):

    z = (a·b) / √(b²·SE(a)² + a²·SE(b)²)

    Where SE() is the standard error from regressions.

    Under H₀: a·b = 0, z ~ N(0,1)
```

**Limitations of Baron-Kenny:**
- Sobel test has low power
- Assumes no confounding
- Assumes linear relationships
- Doesn't provide confidence intervals for indirect effect

### Causal Mediation Analysis

Modern causal mediation analysis uses the **potential outcomes framework** (Imai, Keele, & Tingley, 2010):

```
POTENTIAL OUTCOMES NOTATION:

Y(x, m) = Potential outcome under treatment X=x and mediator M=m
M(x)    = Potential mediator value under treatment X=x

For binary treatment (X ∈ {0,1}):

  Y(1, M(1)) = Outcome when treated, with treated mediator
  Y(0, M(0)) = Outcome when untreated, with untreated mediator

  Y(1, M(0)) = COUNTERFACTUAL: Outcome when treated,
               but with mediator value that WOULD have occurred
               if untreated

  Y(0, M(1)) = COUNTERFACTUAL: Outcome when untreated,
               but with mediator value that WOULD have occurred
               if treated

AVERAGE TREATMENT EFFECT:

    ATE = E[Y(1, M(1)) - Y(0, M(0))]
        = Average total effect of X on Y
```

### Potential Outcomes Framework

The causal decomposition:

```
CAUSAL EFFECT DECOMPOSITION:

Total Effect = E[Y(1, M(1))] - E[Y(0, M(0))]

            = E[Y(1, M(1))] - E[Y(1, M(0))]     ← Indirect effect
            + E[Y(1, M(0))] - E[Y(0, M(0))]     ← Direct effect

Or equivalently:

            = E[Y(1, M(1))] - E[Y(0, M(1))]     ← Direct effect (alt)
            + E[Y(0, M(1))] - E[Y(0, M(0))]     ← Indirect effect (alt)

NOTE: These two decompositions generally differ!
      (unless there's no interaction between X and M)

Graphical representation:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                              │
    │  Y(0,M(0)) ─────────────────────────────────────► Y(1,M(1)) │
    │      │                                                  ▲    │
    │      │    ┌──────── indirect ────────┐                 │    │
    │      │    │                          │                 │    │
    │      └────┼──► Y(0,M(1)) ────────────┼─────────────────┘    │
    │           │        │                  │                      │
    │           │        └── direct ────────┘                      │
    │           │                                                  │
    │      Alternative decomposition                               │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
```

### Natural Direct and Indirect Effects

```
NATURAL DIRECT EFFECT (NDE):

    NDE(x) = E[Y(1, M(x)) - Y(0, M(x))]

    "Effect of X on Y when M is held at its natural value M(x)"

    For x=0: NDE(0) = E[Y(1, M(0)) - Y(0, M(0))]
        "Direct effect when mediator takes its untreated value"

    For x=1: NDE(1) = E[Y(1, M(1)) - Y(0, M(1))]
        "Direct effect when mediator takes its treated value"

NATURAL INDIRECT EFFECT (NIE):

    NIE(x) = E[Y(x, M(1)) - Y(x, M(0))]

    "Effect of shifting M from untreated to treated value,
     holding X constant at x"

    For x=0: NIE(0) = E[Y(0, M(1)) - Y(0, M(0))]
        "Indirect effect in untreated group"

    For x=1: NIE(1) = E[Y(1, M(1)) - Y(1, M(0))]
        "Indirect effect in treated group"

DECOMPOSITION:

    Total Effect = NDE(0) + NIE(1)  = NDE(1) + NIE(0)

    Average natural direct effect:   ANDE = (NDE(0) + NDE(1)) / 2
    Average natural indirect effect: ANIE = (NIE(0) + NIE(1)) / 2
```

**Identification under Sequential Ignorability:**

```
Sequential Ignorability Assumption:

    {Y(x', m), M(x)} ⊥⊥ X | C        (given covariates)
    Y(x', m) ⊥⊥ M | X, C            (given treatment and covariates)

Under this assumption, NDE and NIE are identified by:

    NDE(0) = ∫∫ E[Y | X=1, M=m, C=c] - E[Y | X=0, M=m, C=c]
             × p(m | X=0, C=c) × p(c) dm dc

    NIE(1) = ∫∫ E[Y | X=1, M=m, C=c] × [p(m | X=1, C=c) - p(m | X=0, C=c)]
             × p(c) dm dc

These can be estimated using:
    - Regression-based approaches (linear models)
    - Weighting estimators
    - Simulation-based methods (mediation package in R/Python)
```

## Financial Applications

### Factor Return Decomposition

Understanding how factor returns are generated through various channels:

```
FACTOR RETURN MEDIATION EXAMPLE: Momentum

Question: How does past return (X) predict future return (Y)?
          What role does trading activity (M) play?

    Past Return ────► Trading Volume ────► Future Return
    (momentum)            ↑                     ▲
                          │                     │
                   investor attention           │
                          │                     │
                          └─────── direct momentum ──────┘

Hypotheses:
1. DIRECT EFFECT: Price trends continue due to slow information diffusion
2. INDIRECT EFFECT: High returns → Attention → Volume → Price pressure → Returns

Implications:
- If indirect dominates: Momentum works via behavioral channels
- If direct dominates: Momentum reflects fundamental information

Trading Strategy Insight:
- Monitor volume to gauge which channel is active
- High volume momentum: Behavioral, may reverse
- Low volume momentum: Informational, may persist
```

### News Sentiment Mediation

```
NEWS IMPACT PATHWAY ANALYSIS:

Treatment (X): Earnings announcement surprise
Mediator (M): Analyst sentiment revisions
Outcome (Y): 30-day post-announcement return

Model:
    M = α + a·X + ε₁
    Y = β + c'·X + b·M + ε₂

Results Example:
    a  = 0.45  (earnings surprise → analyst revisions)
    b  = 0.28  (analyst revisions → returns)
    c' = 0.12  (direct effect)

    Indirect = 0.45 × 0.28 = 0.126
    Total    = 0.12 + 0.126 = 0.246

Interpretation:
    ┌────────────────────────────────────────────────────────────┐
    │                                                             │
    │  Earnings Surprise Total Effect: +24.6 bps                 │
    │                                                             │
    │  ├── Direct Effect:   +12.0 bps (49%)                      │
    │  │   └── Immediate market reaction to news                 │
    │  │                                                          │
    │  └── Indirect Effect: +12.6 bps (51%)                      │
    │      └── Mediated through analyst revisions                │
    │                                                             │
    │  Implication: Trading signal should incorporate            │
    │               analyst revision speed as timing indicator    │
    │                                                             │
    └────────────────────────────────────────────────────────────┘
```

### Liquidity as Mediator

```
LIQUIDITY MEDIATION IN PRICE IMPACT:

    Order Flow ────► Liquidity ────► Price Impact
    Imbalance           ↑                 ▲
        │               │                 │
        │        inventory cost           │
        │               │                 │
        └───────── direct impact ─────────┘

Direct Effect: Large orders mechanically move price
Indirect Effect: Orders → Liquidity withdrawal → Larger impact

This matters for execution:
- In high liquidity regime: Direct effect dominates
- In low liquidity regime: Indirect amplification occurs

Trading Rule:
    If current_liquidity < threshold:
        # Indirect effect is strong - split order more
        execution_urgency = LOW
    else:
        # Direct effect dominates - can execute faster
        execution_urgency = HIGH
```

### Volatility Transmission Mechanisms

```
CROSS-ASSET VOLATILITY MEDIATION:

    BTC Volatility ────► Market Sentiment ────► ETH Volatility
          │                     ↑                      ▲
          │              fear index                    │
          │                     │                      │
          └────────── direct correlation ──────────────┘

Research Question:
    Does BTC volatility affect ETH volatility directly,
    or through a sentiment channel?

Empirical Model:
    sentiment_t = α₀ + a·btc_vol_{t-1} + controls + ε₁
    eth_vol_t   = β₀ + c'·btc_vol_{t-1} + b·sentiment_t + controls + ε₂

Results:
    Direct effect (BTC vol → ETH vol): 0.35
    Indirect effect (via sentiment):    0.22
    Total effect:                       0.57

Portfolio Implication:
    - When sentiment is calm: BTC-ETH correlation = 0.35 / total
    - When sentiment is fearful: Full correlation activates

    Hedging Strategy:
    - In calm regime: Partial hedge sufficient
    - In fear regime: Full hedge required
```

## Practical Examples

### 01: Data Preparation

```python
"""
Example 01: Data Preparation for Mediation Analysis
Using stock market data and crypto (Bybit) data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def prepare_stock_data(
    treatment_ticker: str,
    mediator_indicator: str,  # e.g., 'volume', 'volatility'
    outcome_ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Prepare data for mediation analysis with stock data.

    Example: Does SPY affect individual stock returns through VIX?
        X (Treatment): SPY returns
        M (Mediator):  VIX level
        Y (Outcome):   AAPL returns
    """
    # Download data
    treatment_data = yf.download(treatment_ticker, start=start_date, end=end_date)
    outcome_data = yf.download(outcome_ticker, start=start_date, end=end_date)

    # Download mediator (e.g., VIX for volatility)
    if mediator_indicator == 'volatility':
        mediator_data = yf.download('^VIX', start=start_date, end=end_date)
        mediator = mediator_data['Close']
    elif mediator_indicator == 'volume':
        mediator = treatment_data['Volume']

    # Calculate returns
    treatment_returns = treatment_data['Adj Close'].pct_change()
    outcome_returns = outcome_data['Adj Close'].pct_change()

    # Combine into DataFrame
    df = pd.DataFrame({
        'X': treatment_returns.shift(1),  # Lagged treatment
        'M': mediator.shift(1),           # Lagged mediator
        'Y': outcome_returns,              # Current outcome
        'X_M_interaction': treatment_returns.shift(1) * mediator.shift(1)
    }).dropna()

    return df


def prepare_crypto_data_bybit(
    treatment_symbol: str,
    mediator_type: str,  # 'funding_rate', 'open_interest', 'volume'
    outcome_symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1h'
) -> pd.DataFrame:
    """
    Prepare crypto data from Bybit for mediation analysis.

    Example: Does BTC price movement affect altcoin returns through funding rates?
        X (Treatment): BTC returns
        M (Mediator):  Perpetual funding rate
        Y (Outcome):   ETH returns
    """
    from pybit.unified_trading import HTTP

    # Initialize Bybit session (public endpoints don't need API keys)
    session = HTTP(testnet=False)

    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    # Get treatment (BTC) klines
    treatment_klines = session.get_kline(
        category="linear",
        symbol=treatment_symbol,
        interval=interval,
        start=start_ts,
        end=end_ts,
        limit=1000
    )

    # Get outcome klines
    outcome_klines = session.get_kline(
        category="linear",
        symbol=outcome_symbol,
        interval=interval,
        start=start_ts,
        end=end_ts,
        limit=1000
    )

    # Parse klines
    def parse_klines(klines):
        df = pd.DataFrame(klines['result']['list'],
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df = df.set_index('timestamp').sort_index()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    treatment_df = parse_klines(treatment_klines)
    outcome_df = parse_klines(outcome_klines)

    # Get mediator data based on type
    if mediator_type == 'funding_rate':
        # Get funding rate history
        funding = session.get_funding_rate_history(
            category="linear",
            symbol=treatment_symbol,
            startTime=start_ts,
            endTime=end_ts,
            limit=200
        )
        mediator_df = pd.DataFrame(funding['result']['list'])
        mediator_df['timestamp'] = pd.to_datetime(mediator_df['fundingRateTimestamp'].astype(int), unit='ms')
        mediator_df['M'] = mediator_df['fundingRate'].astype(float)
        mediator_df = mediator_df.set_index('timestamp')[['M']].sort_index()

    elif mediator_type == 'open_interest':
        # Get open interest
        oi = session.get_open_interest(
            category="linear",
            symbol=treatment_symbol,
            intervalTime="1h",
            startTime=start_ts,
            endTime=end_ts,
            limit=200
        )
        mediator_df = pd.DataFrame(oi['result']['list'])
        mediator_df['timestamp'] = pd.to_datetime(mediator_df['timestamp'].astype(int), unit='ms')
        mediator_df['M'] = mediator_df['openInterest'].astype(float)
        mediator_df = mediator_df.set_index('timestamp')[['M']].sort_index()

    elif mediator_type == 'volume':
        mediator_df = treatment_df[['volume']].rename(columns={'volume': 'M'})

    # Calculate returns
    treatment_returns = treatment_df['close'].pct_change()
    outcome_returns = outcome_df['close'].pct_change()

    # Combine data
    combined = pd.DataFrame({
        'X': treatment_returns,
        'Y': outcome_returns
    })

    # Merge mediator (may have different frequency)
    combined = combined.join(mediator_df, how='left')
    combined['M'] = combined['M'].ffill()  # Forward fill for lower frequency mediators

    # Lag treatment and mediator
    combined['X'] = combined['X'].shift(1)
    combined['M'] = combined['M'].shift(1)

    return combined.dropna()


# Example usage
if __name__ == "__main__":
    # Stock market example: SPY → VIX → AAPL
    stock_data = prepare_stock_data(
        treatment_ticker='SPY',
        mediator_indicator='volatility',
        outcome_ticker='AAPL',
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    print("Stock data shape:", stock_data.shape)
    print(stock_data.head())

    # Crypto example: BTC → Funding Rate → ETH
    crypto_data = prepare_crypto_data_bybit(
        treatment_symbol='BTCUSDT',
        mediator_type='funding_rate',
        outcome_symbol='ETHUSDT',
        start_date='2024-01-01',
        end_date='2024-06-01'
    )
    print("\nCrypto data shape:", crypto_data.shape)
    print(crypto_data.head())
```

### 02: Classical Mediation Analysis

```python
"""
Example 02: Classical Baron-Kenny Mediation Analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class MediationResults:
    """Results container for mediation analysis"""
    total_effect: float
    total_effect_se: float
    total_effect_pvalue: float

    direct_effect: float
    direct_effect_se: float
    direct_effect_pvalue: float

    indirect_effect: float
    indirect_effect_se: float
    indirect_effect_pvalue: float

    a_path: float  # X → M
    a_path_se: float
    a_path_pvalue: float

    b_path: float  # M → Y (controlling for X)
    b_path_se: float
    b_path_pvalue: float

    proportion_mediated: float
    sobel_z: float
    sobel_pvalue: float

    def __str__(self):
        return f"""
========== MEDIATION ANALYSIS RESULTS ==========

Path Coefficients:
    a (X → M):        {self.a_path:.4f} (SE={self.a_path_se:.4f}, p={self.a_path_pvalue:.4f})
    b (M → Y|X):      {self.b_path:.4f} (SE={self.b_path_se:.4f}, p={self.b_path_pvalue:.4f})

Effect Decomposition:
    Total Effect (c):     {self.total_effect:.4f} (SE={self.total_effect_se:.4f}, p={self.total_effect_pvalue:.4f})
    Direct Effect (c'):   {self.direct_effect:.4f} (SE={self.direct_effect_se:.4f}, p={self.direct_effect_pvalue:.4f})
    Indirect Effect (ab): {self.indirect_effect:.4f} (SE={self.indirect_effect_se:.4f}, p={self.indirect_effect_pvalue:.4f})

Sobel Test:
    Z-statistic: {self.sobel_z:.4f}
    P-value:     {self.sobel_pvalue:.4f}

Proportion Mediated: {self.proportion_mediated:.2%}

Interpretation:
    {"Full Mediation" if abs(self.direct_effect_pvalue) > 0.05 and self.indirect_effect_pvalue < 0.05
     else "Partial Mediation" if self.indirect_effect_pvalue < 0.05 and self.direct_effect_pvalue < 0.05
     else "No Mediation" if self.indirect_effect_pvalue > 0.05
     else "Direct Effect Only"}
"""


def baron_kenny_mediation(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    covariates: Optional[np.ndarray] = None
) -> MediationResults:
    """
    Perform Baron-Kenny mediation analysis.

    Parameters:
    -----------
    X : Treatment variable (n,)
    M : Mediator variable (n,)
    Y : Outcome variable (n,)
    covariates : Optional control variables (n, k)

    Returns:
    --------
    MediationResults object
    """
    n = len(X)

    # Prepare design matrices
    if covariates is not None:
        X_with_const = sm.add_constant(np.column_stack([X, covariates]))
        XM_with_const = sm.add_constant(np.column_stack([X, M, covariates]))
    else:
        X_with_const = sm.add_constant(X)
        XM_with_const = sm.add_constant(np.column_stack([X, M]))

    # Step 1: Total effect (c) - Y ~ X
    model_total = sm.OLS(Y, X_with_const).fit()
    c = model_total.params[1]
    c_se = model_total.bse[1]
    c_pvalue = model_total.pvalues[1]

    # Step 2: a path - M ~ X
    model_a = sm.OLS(M, X_with_const).fit()
    a = model_a.params[1]
    a_se = model_a.bse[1]
    a_pvalue = model_a.pvalues[1]

    # Step 3: Direct effect (c') and b path - Y ~ X + M
    model_direct = sm.OLS(Y, XM_with_const).fit()
    c_prime = model_direct.params[1]
    c_prime_se = model_direct.bse[1]
    c_prime_pvalue = model_direct.pvalues[1]

    b = model_direct.params[2]
    b_se = model_direct.bse[2]
    b_pvalue = model_direct.pvalues[2]

    # Indirect effect
    indirect = a * b

    # Sobel test for indirect effect
    sobel_se = np.sqrt(b**2 * a_se**2 + a**2 * b_se**2)
    sobel_z = indirect / sobel_se
    sobel_pvalue = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    # Proportion mediated
    if c != 0:
        prop_mediated = indirect / c
    else:
        prop_mediated = np.nan

    return MediationResults(
        total_effect=c,
        total_effect_se=c_se,
        total_effect_pvalue=c_pvalue,
        direct_effect=c_prime,
        direct_effect_se=c_prime_se,
        direct_effect_pvalue=c_prime_pvalue,
        indirect_effect=indirect,
        indirect_effect_se=sobel_se,
        indirect_effect_pvalue=sobel_pvalue,
        a_path=a,
        a_path_se=a_se,
        a_path_pvalue=a_pvalue,
        b_path=b,
        b_path_se=b_se,
        b_path_pvalue=b_pvalue,
        proportion_mediated=prop_mediated,
        sobel_z=sobel_z,
        sobel_pvalue=sobel_pvalue
    )


def bootstrap_mediation(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    covariates: Optional[np.ndarray] = None
) -> dict:
    """
    Bootstrap confidence intervals for mediation effects.
    More reliable than Sobel test, especially for small samples.
    """
    n = len(X)
    indirect_effects = np.zeros(n_bootstrap)
    direct_effects = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        M_boot = M[idx]
        Y_boot = Y[idx]
        cov_boot = covariates[idx] if covariates is not None else None

        # Fit models
        if cov_boot is not None:
            X_design = sm.add_constant(np.column_stack([X_boot, cov_boot]))
            XM_design = sm.add_constant(np.column_stack([X_boot, M_boot, cov_boot]))
        else:
            X_design = sm.add_constant(X_boot)
            XM_design = sm.add_constant(np.column_stack([X_boot, M_boot]))

        try:
            # a path
            model_a = sm.OLS(M_boot, X_design).fit()
            a = model_a.params[1]

            # b and c' paths
            model_direct = sm.OLS(Y_boot, XM_design).fit()
            b = model_direct.params[2]
            c_prime = model_direct.params[1]

            indirect_effects[i] = a * b
            direct_effects[i] = c_prime
        except:
            indirect_effects[i] = np.nan
            direct_effects[i] = np.nan

    # Remove failed bootstraps
    indirect_effects = indirect_effects[~np.isnan(indirect_effects)]
    direct_effects = direct_effects[~np.isnan(direct_effects)]

    alpha = 1 - confidence_level

    return {
        'indirect_mean': np.mean(indirect_effects),
        'indirect_ci_lower': np.percentile(indirect_effects, 100 * alpha / 2),
        'indirect_ci_upper': np.percentile(indirect_effects, 100 * (1 - alpha / 2)),
        'indirect_significant': (
            np.percentile(indirect_effects, 100 * alpha / 2) > 0 or
            np.percentile(indirect_effects, 100 * (1 - alpha / 2)) < 0
        ),
        'direct_mean': np.mean(direct_effects),
        'direct_ci_lower': np.percentile(direct_effects, 100 * alpha / 2),
        'direct_ci_upper': np.percentile(direct_effects, 100 * (1 - alpha / 2)),
    }


# Example usage
if __name__ == "__main__":
    # Generate synthetic financial data
    np.random.seed(42)
    n = 500

    # X = Market returns (treatment)
    X = np.random.randn(n) * 0.02

    # M = Trading volume (mediator) - affected by returns
    a_true = 0.5
    M = a_true * X + np.random.randn(n) * 0.01

    # Y = Stock returns (outcome) - affected by both X and M
    c_prime_true = 0.3  # direct effect
    b_true = 0.4        # mediator effect
    Y = c_prime_true * X + b_true * M + np.random.randn(n) * 0.015

    # Run mediation analysis
    results = baron_kenny_mediation(X, M, Y)
    print(results)

    # Bootstrap confidence intervals
    boot_results = bootstrap_mediation(X, M, Y, n_bootstrap=1000)
    print("\nBootstrap 95% CI for Indirect Effect:")
    print(f"  [{boot_results['indirect_ci_lower']:.4f}, {boot_results['indirect_ci_upper']:.4f}]")
    print(f"  Significant: {boot_results['indirect_significant']}")
```

### 03: Causal Mediation with Sensitivity

```python
"""
Example 03: Causal Mediation Analysis with Sensitivity Analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


class CausalMediationAnalysis:
    """
    Implements causal mediation analysis based on Imai, Keele, & Tingley (2010)
    with sensitivity analysis for unmeasured confounding.
    """

    def __init__(
        self,
        X: np.ndarray,
        M: np.ndarray,
        Y: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        treatment_m_interaction: bool = False
    ):
        """
        Initialize causal mediation analysis.

        Parameters:
        -----------
        X : Treatment variable (n,)
        M : Mediator variable (n,)
        Y : Outcome variable (n,)
        covariates : Optional control variables (n, k)
        treatment_m_interaction : Whether to include X*M interaction
        """
        self.X = X
        self.M = M
        self.Y = Y
        self.covariates = covariates
        self.n = len(X)
        self.interaction = treatment_m_interaction

        # Fit models
        self._fit_models()

    def _fit_models(self):
        """Fit mediator and outcome models"""
        # Mediator model: M ~ X + C
        if self.covariates is not None:
            design_m = sm.add_constant(np.column_stack([self.X, self.covariates]))
        else:
            design_m = sm.add_constant(self.X)

        self.model_m = sm.OLS(self.M, design_m).fit()

        # Outcome model: Y ~ X + M + (X*M) + C
        if self.interaction:
            XM_interaction = self.X * self.M
            if self.covariates is not None:
                design_y = sm.add_constant(
                    np.column_stack([self.X, self.M, XM_interaction, self.covariates])
                )
            else:
                design_y = sm.add_constant(
                    np.column_stack([self.X, self.M, XM_interaction])
                )
        else:
            if self.covariates is not None:
                design_y = sm.add_constant(
                    np.column_stack([self.X, self.M, self.covariates])
                )
            else:
                design_y = sm.add_constant(np.column_stack([self.X, self.M]))

        self.model_y = sm.OLS(self.Y, design_y).fit()

    def estimate_effects(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Estimate natural direct and indirect effects using simulation.

        Uses quasi-Bayesian Monte Carlo simulation approach.
        """
        # Get parameter estimates and covariance
        beta_m = self.model_m.params
        cov_m = self.model_m.cov_params()

        beta_y = self.model_y.params
        cov_y = self.model_y.cov_params()

        # Simulate from parameter distributions
        np.random.seed(42)
        beta_m_sim = np.random.multivariate_normal(beta_m, cov_m, n_simulations)
        beta_y_sim = np.random.multivariate_normal(beta_y, cov_y, n_simulations)

        # Storage for effects
        nde_0 = np.zeros(n_simulations)  # NDE at X=0
        nde_1 = np.zeros(n_simulations)  # NDE at X=1
        nie_0 = np.zeros(n_simulations)  # NIE at X=0
        nie_1 = np.zeros(n_simulations)  # NIE at X=1

        # Simulate potential mediators
        sigma_m = np.sqrt(self.model_m.mse_resid)
        sigma_y = np.sqrt(self.model_y.mse_resid)

        for i in range(n_simulations):
            # For each observation, compute effects
            for j in range(min(100, self.n)):  # Subsample for speed
                # Covariates for this observation
                if self.covariates is not None:
                    c_j = self.covariates[j]
                    design_j_m = np.concatenate([[1], [0], c_j])  # X=0
                    design_j_m1 = np.concatenate([[1], [1], c_j])  # X=1
                else:
                    design_j_m = np.array([1, 0])  # X=0
                    design_j_m1 = np.array([1, 1])  # X=1

                # Potential mediators
                M_0 = design_j_m @ beta_m_sim[i] + np.random.randn() * sigma_m
                M_1 = design_j_m1 @ beta_m_sim[i] + np.random.randn() * sigma_m

                # Potential outcomes
                if self.interaction:
                    if self.covariates is not None:
                        # Y ~ intercept + X + M + X*M + C
                        Y_00 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*0 + beta_y_sim[i, 2]*M_0 + beta_y_sim[i, 3]*0*M_0 + c_j @ beta_y_sim[i, 4:]
                        Y_01 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*0 + beta_y_sim[i, 2]*M_1 + beta_y_sim[i, 3]*0*M_1 + c_j @ beta_y_sim[i, 4:]
                        Y_10 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*1 + beta_y_sim[i, 2]*M_0 + beta_y_sim[i, 3]*1*M_0 + c_j @ beta_y_sim[i, 4:]
                        Y_11 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*1 + beta_y_sim[i, 2]*M_1 + beta_y_sim[i, 3]*1*M_1 + c_j @ beta_y_sim[i, 4:]
                    else:
                        Y_00 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*0 + beta_y_sim[i, 2]*M_0 + beta_y_sim[i, 3]*0*M_0
                        Y_01 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*0 + beta_y_sim[i, 2]*M_1 + beta_y_sim[i, 3]*0*M_1
                        Y_10 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*1 + beta_y_sim[i, 2]*M_0 + beta_y_sim[i, 3]*1*M_0
                        Y_11 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*1 + beta_y_sim[i, 2]*M_1 + beta_y_sim[i, 3]*1*M_1
                else:
                    if self.covariates is not None:
                        Y_00 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*0 + beta_y_sim[i, 2]*M_0 + c_j @ beta_y_sim[i, 3:]
                        Y_01 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*0 + beta_y_sim[i, 2]*M_1 + c_j @ beta_y_sim[i, 3:]
                        Y_10 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*1 + beta_y_sim[i, 2]*M_0 + c_j @ beta_y_sim[i, 3:]
                        Y_11 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*1 + beta_y_sim[i, 2]*M_1 + c_j @ beta_y_sim[i, 3:]
                    else:
                        Y_00 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*0 + beta_y_sim[i, 2]*M_0
                        Y_01 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*0 + beta_y_sim[i, 2]*M_1
                        Y_10 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*1 + beta_y_sim[i, 2]*M_0
                        Y_11 = beta_y_sim[i, 0] + beta_y_sim[i, 1]*1 + beta_y_sim[i, 2]*M_1

                # Accumulate effects for this simulation
                nde_0[i] += (Y_10 - Y_00)  # NDE at control mediator
                nde_1[i] += (Y_11 - Y_01)  # NDE at treated mediator
                nie_0[i] += (Y_01 - Y_00)  # NIE at control treatment
                nie_1[i] += (Y_11 - Y_10)  # NIE at treated treatment

            # Average over observations
            nde_0[i] /= min(100, self.n)
            nde_1[i] /= min(100, self.n)
            nie_0[i] /= min(100, self.n)
            nie_1[i] /= min(100, self.n)

        # Compute summary statistics
        alpha = 1 - confidence_level

        results = {
            'ACME': {  # Average Causal Mediation Effect (average of NIEs)
                'estimate': np.mean((nie_0 + nie_1) / 2),
                'ci_lower': np.percentile((nie_0 + nie_1) / 2, 100 * alpha / 2),
                'ci_upper': np.percentile((nie_0 + nie_1) / 2, 100 * (1 - alpha / 2)),
                'pvalue': self._compute_pvalue((nie_0 + nie_1) / 2)
            },
            'ADE': {  # Average Direct Effect
                'estimate': np.mean((nde_0 + nde_1) / 2),
                'ci_lower': np.percentile((nde_0 + nde_1) / 2, 100 * alpha / 2),
                'ci_upper': np.percentile((nde_0 + nde_1) / 2, 100 * (1 - alpha / 2)),
                'pvalue': self._compute_pvalue((nde_0 + nde_1) / 2)
            },
            'Total_Effect': {
                'estimate': np.mean((nde_0 + nde_1) / 2 + (nie_0 + nie_1) / 2),
                'ci_lower': np.percentile((nde_0 + nde_1 + nie_0 + nie_1) / 2, 100 * alpha / 2),
                'ci_upper': np.percentile((nde_0 + nde_1 + nie_0 + nie_1) / 2, 100 * (1 - alpha / 2)),
            },
            'Proportion_Mediated': {
                'estimate': np.mean((nie_0 + nie_1) / (nde_0 + nde_1 + nie_0 + nie_1)),
                'ci_lower': np.percentile((nie_0 + nie_1) / (nde_0 + nde_1 + nie_0 + nie_1 + 1e-10), 100 * alpha / 2),
                'ci_upper': np.percentile((nie_0 + nie_1) / (nde_0 + nde_1 + nie_0 + nie_1 + 1e-10), 100 * (1 - alpha / 2)),
            }
        }

        return results

    def _compute_pvalue(self, samples: np.ndarray) -> float:
        """Compute two-sided p-value from simulation samples"""
        # Proportion of samples with opposite sign to mean
        mean_sign = np.sign(np.mean(samples))
        p = np.mean(np.sign(samples) != mean_sign) * 2
        return min(p, 1.0)

    def sensitivity_analysis(
        self,
        rho_range: np.ndarray = np.linspace(-0.9, 0.9, 19),
        n_simulations: int = 500
    ) -> Dict:
        """
        Sensitivity analysis for unmeasured confounding.

        Assesses how results change if there's correlation between
        mediator and outcome residuals (violation of sequential ignorability).

        Parameters:
        -----------
        rho_range : Correlation values to test
        n_simulations : Simulations per rho value

        Returns:
        --------
        Dictionary with sensitivity analysis results
        """
        results = {
            'rho': rho_range,
            'acme': np.zeros(len(rho_range)),
            'acme_ci_lower': np.zeros(len(rho_range)),
            'acme_ci_upper': np.zeros(len(rho_range))
        }

        sigma_m = np.sqrt(self.model_m.mse_resid)
        sigma_y = np.sqrt(self.model_y.mse_resid)

        beta_m = self.model_m.params
        beta_y = self.model_y.params

        for i, rho in enumerate(rho_range):
            acme_samples = np.zeros(n_simulations)

            for s in range(n_simulations):
                # Generate correlated residuals
                cov_matrix = np.array([
                    [sigma_m**2, rho * sigma_m * sigma_y],
                    [rho * sigma_m * sigma_y, sigma_y**2]
                ])

                effects = []
                for j in range(min(50, self.n)):
                    # Correlated errors
                    eps_m, eps_y = np.random.multivariate_normal([0, 0], cov_matrix)

                    if self.covariates is not None:
                        c_j = self.covariates[j]
                        M_0 = beta_m[0] + beta_m[1]*0 + c_j @ beta_m[2:] + eps_m
                        M_1 = beta_m[0] + beta_m[1]*1 + c_j @ beta_m[2:] + eps_m
                    else:
                        M_0 = beta_m[0] + beta_m[1]*0 + eps_m
                        M_1 = beta_m[0] + beta_m[1]*1 + eps_m

                    if self.interaction:
                        if self.covariates is not None:
                            Y_11 = beta_y[0] + beta_y[1]*1 + beta_y[2]*M_1 + beta_y[3]*1*M_1 + c_j @ beta_y[4:] + eps_y
                            Y_10 = beta_y[0] + beta_y[1]*1 + beta_y[2]*M_0 + beta_y[3]*1*M_0 + c_j @ beta_y[4:] + eps_y
                        else:
                            Y_11 = beta_y[0] + beta_y[1]*1 + beta_y[2]*M_1 + beta_y[3]*1*M_1 + eps_y
                            Y_10 = beta_y[0] + beta_y[1]*1 + beta_y[2]*M_0 + beta_y[3]*1*M_0 + eps_y
                    else:
                        if self.covariates is not None:
                            Y_11 = beta_y[0] + beta_y[1]*1 + beta_y[2]*M_1 + c_j @ beta_y[3:] + eps_y
                            Y_10 = beta_y[0] + beta_y[1]*1 + beta_y[2]*M_0 + c_j @ beta_y[3:] + eps_y
                        else:
                            Y_11 = beta_y[0] + beta_y[1]*1 + beta_y[2]*M_1 + eps_y
                            Y_10 = beta_y[0] + beta_y[1]*1 + beta_y[2]*M_0 + eps_y

                    effects.append(Y_11 - Y_10)

                acme_samples[s] = np.mean(effects)

            results['acme'][i] = np.mean(acme_samples)
            results['acme_ci_lower'][i] = np.percentile(acme_samples, 2.5)
            results['acme_ci_upper'][i] = np.percentile(acme_samples, 97.5)

        # Find rho where ACME crosses zero (breakdown point)
        signs = np.sign(results['acme'])
        sign_changes = np.where(np.diff(signs) != 0)[0]

        if len(sign_changes) > 0:
            results['breakdown_rho'] = rho_range[sign_changes[0]]
        else:
            results['breakdown_rho'] = None

        return results

    def plot_sensitivity(self, sensitivity_results: Dict, save_path: Optional[str] = None):
        """Plot sensitivity analysis results"""
        fig, ax = plt.subplots(figsize=(10, 6))

        rho = sensitivity_results['rho']
        acme = sensitivity_results['acme']
        ci_lower = sensitivity_results['acme_ci_lower']
        ci_upper = sensitivity_results['acme_ci_upper']

        ax.plot(rho, acme, 'b-', linewidth=2, label='ACME')
        ax.fill_between(rho, ci_lower, ci_upper, alpha=0.3, color='blue')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)

        if sensitivity_results['breakdown_rho'] is not None:
            ax.axvline(x=sensitivity_results['breakdown_rho'], color='orange',
                      linestyle='--', linewidth=2,
                      label=f"Breakdown ρ = {sensitivity_results['breakdown_rho']:.2f}")

        ax.set_xlabel('Residual Correlation (ρ)', fontsize=12)
        ax.set_ylabel('Average Causal Mediation Effect (ACME)', fontsize=12)
        ax.set_title('Sensitivity Analysis: Robustness to Unmeasured Confounding', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()
        return fig


# Example usage
if __name__ == "__main__":
    # Generate realistic financial data
    np.random.seed(42)
    n = 1000

    # Covariate: Market volatility regime
    volatility = np.random.uniform(0.1, 0.3, n)

    # X = Earnings surprise (standardized)
    X = np.random.randn(n)

    # M = Analyst revision (mediator) - affected by earnings surprise
    a = 0.4
    M = a * X + 0.3 * volatility + np.random.randn(n) * 0.5

    # Y = Stock return - affected by both direct and indirect paths
    c_prime = 0.2  # Direct effect
    b = 0.35       # Mediator effect
    Y = c_prime * X + b * M - 0.5 * volatility + np.random.randn(n) * 0.4

    # Run causal mediation analysis
    cma = CausalMediationAnalysis(X, M, Y, covariates=volatility.reshape(-1, 1))

    # Estimate effects
    effects = cma.estimate_effects(n_simulations=1000)

    print("="*60)
    print("CAUSAL MEDIATION ANALYSIS RESULTS")
    print("="*60)
    for effect_name, values in effects.items():
        print(f"\n{effect_name}:")
        print(f"  Estimate: {values['estimate']:.4f}")
        print(f"  95% CI: [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]")
        if 'pvalue' in values:
            print(f"  P-value: {values['pvalue']:.4f}")

    # Sensitivity analysis
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    sensitivity = cma.sensitivity_analysis()
    if sensitivity['breakdown_rho'] is not None:
        print(f"\nBreakdown point: ρ = {sensitivity['breakdown_rho']:.2f}")
        print("The indirect effect would be zero if the correlation between")
        print("mediator and outcome residuals equals this value.")
    else:
        print("\nNo breakdown point found in [-0.9, 0.9] range.")
        print("Results are robust to substantial unmeasured confounding.")

    # Plot sensitivity
    cma.plot_sensitivity(sensitivity)
```

### 04: Trading Strategy from Mediation Insights

```python
"""
Example 04: Trading Strategy Based on Mediation Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MediationSignal:
    """Signal based on mediation pathway activation"""
    timestamp: pd.Timestamp
    treatment: str       # Name of treatment variable
    mediator: str        # Name of mediator variable
    target: str          # Target asset

    direct_strength: float       # Strength of direct effect
    indirect_strength: float     # Strength of indirect effect
    mediator_activated: bool     # Is mediator channel active?

    signal: int          # -1, 0, 1 (short, neutral, long)
    confidence: float    # Signal confidence


class MediationTradingStrategy:
    """
    Trading strategy that exploits mediation pathway dynamics.

    Core insight: When the mediator channel is activated, we expect
    the full effect to materialize. When blocked, only direct effect works.
    """

    def __init__(
        self,
        lookback_window: int = 60,
        mediation_threshold: float = 0.3,
        signal_threshold: float = 0.5
    ):
        """
        Initialize strategy.

        Parameters:
        -----------
        lookback_window : Window for estimating mediation parameters
        mediation_threshold : Min proportion mediated to consider pathway active
        signal_threshold : Min signal strength to trade
        """
        self.lookback = lookback_window
        self.med_threshold = mediation_threshold
        self.signal_threshold = signal_threshold

        # History for rolling estimation
        self.history = {
            'X': [],
            'M': [],
            'Y': []
        }

        # Parameter estimates
        self.params = {
            'a': None,  # X → M
            'b': None,  # M → Y | X
            'c_prime': None,  # Direct effect
            'a_se': None,
            'b_se': None
        }

    def update(self, X: float, M: float, Y: float) -> Optional[MediationSignal]:
        """
        Update strategy with new observation and generate signal.

        Parameters:
        -----------
        X : Treatment value (e.g., market return)
        M : Mediator value (e.g., trading volume)
        Y : Outcome value (e.g., stock return)

        Returns:
        --------
        MediationSignal if enough data, None otherwise
        """
        # Update history
        self.history['X'].append(X)
        self.history['M'].append(M)
        self.history['Y'].append(Y)

        # Keep only lookback window
        for key in self.history:
            if len(self.history[key]) > self.lookback:
                self.history[key] = self.history[key][-self.lookback:]

        # Need at least 30 observations
        if len(self.history['X']) < 30:
            return None

        # Estimate mediation parameters
        self._estimate_parameters()

        # Generate signal
        return self._generate_signal(X, M)

    def _estimate_parameters(self):
        """Estimate mediation parameters from history"""
        X = np.array(self.history['X'])
        M = np.array(self.history['M'])
        Y = np.array(self.history['Y'])

        n = len(X)

        # Use lagged values
        X_lag = X[:-1]
        M_lag = M[:-1]
        Y_curr = Y[1:]
        M_curr = M[1:]

        # Standardize
        X_std = (X_lag - X_lag.mean()) / (X_lag.std() + 1e-10)
        M_std = (M_lag - M_lag.mean()) / (M_lag.std() + 1e-10)
        M_curr_std = (M_curr - M_curr.mean()) / (M_curr.std() + 1e-10)
        Y_std = (Y_curr - Y_curr.mean()) / (Y_curr.std() + 1e-10)

        # a path: M ~ X (lagged)
        X_design = np.column_stack([np.ones(len(X_std)), X_std])
        try:
            beta_a = np.linalg.lstsq(X_design, M_curr_std, rcond=None)[0]
            self.params['a'] = beta_a[1]

            # SE estimation
            resid_a = M_curr_std - X_design @ beta_a
            mse_a = np.sum(resid_a**2) / (len(resid_a) - 2)
            var_beta_a = mse_a * np.linalg.inv(X_design.T @ X_design)
            self.params['a_se'] = np.sqrt(var_beta_a[1, 1])
        except:
            self.params['a'] = 0
            self.params['a_se'] = 1

        # b and c' paths: Y ~ X + M
        XM_design = np.column_stack([np.ones(len(X_std)), X_std, M_std])
        try:
            beta_y = np.linalg.lstsq(XM_design, Y_std, rcond=None)[0]
            self.params['c_prime'] = beta_y[1]
            self.params['b'] = beta_y[2]

            # SE estimation
            resid_y = Y_std - XM_design @ beta_y
            mse_y = np.sum(resid_y**2) / (len(resid_y) - 3)
            var_beta_y = mse_y * np.linalg.inv(XM_design.T @ XM_design)
            self.params['b_se'] = np.sqrt(var_beta_y[2, 2])
        except:
            self.params['c_prime'] = 0
            self.params['b'] = 0
            self.params['b_se'] = 1

    def _generate_signal(self, X: float, M: float) -> MediationSignal:
        """Generate trading signal based on mediation analysis"""
        # Get current parameters
        a = self.params['a']
        b = self.params['b']
        c_prime = self.params['c_prime']

        # Calculate effects
        indirect = a * b
        direct = c_prime
        total = indirect + direct

        # Determine if mediator is activated
        # Compare current M with expected M given X
        X_arr = np.array(self.history['X'])
        M_arr = np.array(self.history['M'])

        X_mean, X_std = X_arr.mean(), X_arr.std() + 1e-10
        M_mean, M_std = M_arr.mean(), M_arr.std() + 1e-10

        X_norm = (X - X_mean) / X_std
        M_norm = (M - M_mean) / M_std

        # Expected mediator value given treatment
        M_expected = a * X_norm

        # Is mediator activated (stronger than expected)?
        mediator_activated = abs(M_norm) > abs(M_expected) * 1.2

        # Proportion mediated
        if total != 0:
            prop_mediated = indirect / total
        else:
            prop_mediated = 0

        # Calculate expected effect
        if mediator_activated and abs(prop_mediated) > self.med_threshold:
            # Full effect expected (both direct and indirect)
            expected_effect = (direct + indirect) * X_norm
            confidence = 0.8
        else:
            # Only direct effect expected
            expected_effect = direct * X_norm
            confidence = 0.5

        # Generate signal
        if abs(expected_effect) < self.signal_threshold:
            signal = 0  # Neutral
        elif expected_effect > 0:
            signal = 1  # Long
        else:
            signal = -1  # Short

        return MediationSignal(
            timestamp=pd.Timestamp.now(),
            treatment='X',
            mediator='M',
            target='Y',
            direct_strength=direct,
            indirect_strength=indirect,
            mediator_activated=mediator_activated,
            signal=signal,
            confidence=confidence * abs(expected_effect)
        )


class MediationBacktester:
    """Backtest mediation-based trading strategy"""

    def __init__(
        self,
        strategy: MediationTradingStrategy,
        transaction_cost: float = 0.001
    ):
        self.strategy = strategy
        self.tc = transaction_cost

    def run(
        self,
        data: pd.DataFrame,
        x_col: str = 'X',
        m_col: str = 'M',
        y_col: str = 'Y'
    ) -> pd.DataFrame:
        """
        Run backtest.

        Parameters:
        -----------
        data : DataFrame with columns for X, M, Y
        x_col, m_col, y_col : Column names

        Returns:
        --------
        DataFrame with backtest results
        """
        results = []
        position = 0
        pnl = 0

        for i in range(len(data)):
            row = data.iloc[i]

            # Update strategy and get signal
            signal = self.strategy.update(
                X=row[x_col],
                M=row[m_col],
                Y=row[y_col]
            )

            if signal is None:
                results.append({
                    'timestamp': data.index[i],
                    'signal': 0,
                    'position': 0,
                    'return': 0,
                    'pnl': 0,
                    'cumulative_pnl': 0
                })
                continue

            # Position management
            new_position = signal.signal if signal.confidence > 0.3 else 0

            # Transaction cost
            tc_cost = abs(new_position - position) * self.tc

            # P&L from previous position
            ret = position * row[y_col] - tc_cost
            pnl += ret

            results.append({
                'timestamp': data.index[i],
                'signal': signal.signal,
                'confidence': signal.confidence,
                'direct_effect': signal.direct_strength,
                'indirect_effect': signal.indirect_strength,
                'mediator_active': signal.mediator_activated,
                'position': new_position,
                'return': ret,
                'pnl': ret,
                'cumulative_pnl': pnl
            })

            position = new_position

        return pd.DataFrame(results)

    def analyze_results(self, results: pd.DataFrame) -> Dict:
        """Analyze backtest results"""
        # Filter to trading period
        trading = results[results['position'] != 0]

        if len(trading) == 0:
            return {'error': 'No trades executed'}

        returns = results['return'].dropna()

        metrics = {
            'total_return': results['cumulative_pnl'].iloc[-1],
            'n_trades': len(trading),
            'win_rate': (trading['return'] > 0).mean(),
            'avg_return': returns.mean(),
            'std_return': returns.std(),
            'sharpe_ratio': returns.mean() / (returns.std() + 1e-10) * np.sqrt(252),
            'max_drawdown': self._max_drawdown(results['cumulative_pnl']),
            'calmar_ratio': results['cumulative_pnl'].iloc[-1] / (self._max_drawdown(results['cumulative_pnl']) + 1e-10)
        }

        # Analyze mediation dynamics
        if 'mediator_active' in results.columns:
            active_trades = results[results['mediator_active'] == True]
            inactive_trades = results[results['mediator_active'] == False]

            metrics['return_when_mediated'] = active_trades['return'].mean() if len(active_trades) > 0 else 0
            metrics['return_when_direct_only'] = inactive_trades['return'].mean() if len(inactive_trades) > 0 else 0

        return metrics

    def _max_drawdown(self, cumulative: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = cumulative.cummax()
        drawdown = cumulative - rolling_max
        return abs(drawdown.min())


# Example usage
if __name__ == "__main__":
    # Generate synthetic data mimicking market dynamics
    np.random.seed(42)
    n = 500

    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    # Regime switching
    regime = np.zeros(n)
    regime[100:200] = 1  # High mediation regime
    regime[300:400] = 1  # Another high mediation period

    # Market returns (treatment)
    X = np.random.randn(n) * 0.02

    # Trading volume (mediator) - more responsive in regime 1
    a_base = 0.3
    a_regime = 0.6
    M = (a_base + a_regime * regime) * X + np.random.randn(n) * 0.01

    # Stock returns (outcome)
    c_prime = 0.4  # Direct effect
    b = 0.5        # Mediator effect (stronger in regime 1)
    Y = c_prime * X + (b + 0.3 * regime) * M + np.random.randn(n) * 0.015

    # Create DataFrame
    data = pd.DataFrame({
        'X': X,
        'M': M,
        'Y': Y,
        'regime': regime
    }, index=dates)

    # Initialize and run strategy
    strategy = MediationTradingStrategy(
        lookback_window=60,
        mediation_threshold=0.25,
        signal_threshold=0.3
    )

    backtester = MediationBacktester(strategy, transaction_cost=0.001)
    results = backtester.run(data)

    # Analyze
    metrics = backtester.analyze_results(results)

    print("="*60)
    print("MEDIATION TRADING STRATEGY BACKTEST RESULTS")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Plot results
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Cumulative P&L
    axes[0].plot(results['timestamp'], results['cumulative_pnl'], label='Strategy')
    axes[0].fill_between(results['timestamp'], 0, data['regime'].values * 0.1,
                         alpha=0.2, color='green', label='High Mediation Regime')
    axes[0].set_ylabel('Cumulative P&L')
    axes[0].set_title('Strategy Performance vs Mediation Regimes')
    axes[0].legend()

    # Effect decomposition over time
    axes[1].plot(results['timestamp'], results['direct_effect'].rolling(20).mean(),
                 label='Direct Effect (20-day MA)')
    axes[1].plot(results['timestamp'], results['indirect_effect'].rolling(20).mean(),
                 label='Indirect Effect (20-day MA)')
    axes[1].set_ylabel('Effect Strength')
    axes[1].set_title('Dynamic Effect Decomposition')
    axes[1].legend()

    # Position and confidence
    axes[2].plot(results['timestamp'], results['position'], label='Position', alpha=0.7)
    axes[2].fill_between(results['timestamp'], 0, results['confidence'], alpha=0.3, label='Confidence')
    axes[2].set_ylabel('Position / Confidence')
    axes[2].set_title('Trading Signals')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
```

### 05: Backtesting

```python
"""
Example 05: Comprehensive Backtesting Framework for Mediation Strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


@dataclass
class BacktestConfig:
    """Configuration for backtest"""
    initial_capital: float = 100000
    transaction_cost_bps: float = 10  # 10 basis points
    slippage_bps: float = 5
    max_position_size: float = 0.2  # Max 20% of capital per position
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'


class MediationBacktestEngine:
    """
    Comprehensive backtesting engine for mediation-based strategies.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        strategy_params: Dict
    ) -> Dict:
        """
        Run complete backtest.

        Parameters:
        -----------
        data : DataFrame with OHLCV and factor data
        strategy_func : Function that generates signals
        strategy_params : Parameters for strategy

        Returns:
        --------
        Dictionary with results and analytics
        """
        results = {
            'dates': [],
            'signals': [],
            'positions': [],
            'returns': [],
            'portfolio_value': [],
            'drawdown': [],
            'mediation_metrics': []
        }

        capital = self.config.initial_capital
        position = 0
        peak_value = capital

        # Initialize strategy state
        state = {'lookback_data': []}

        for i in range(len(data)):
            current_date = data.index[i]
            current_row = data.iloc[i]

            # Update state with new data
            state['lookback_data'].append(current_row)
            if len(state['lookback_data']) > strategy_params.get('lookback', 60):
                state['lookback_data'] = state['lookback_data'][-strategy_params.get('lookback', 60):]

            # Generate signal
            signal, mediation_info = strategy_func(state, strategy_params)

            # Calculate position size with risk management
            target_position = self._calculate_position(signal, capital, current_row)

            # Transaction costs
            trade_size = abs(target_position - position)
            tc = trade_size * (self.config.transaction_cost_bps + self.config.slippage_bps) / 10000

            # Calculate return
            if i > 0:
                price_return = current_row.get('return', 0)
                pnl = position * price_return - tc
            else:
                pnl = 0

            # Update capital
            capital += pnl

            # Track drawdown
            peak_value = max(peak_value, capital)
            drawdown = (peak_value - capital) / peak_value

            # Store results
            results['dates'].append(current_date)
            results['signals'].append(signal)
            results['positions'].append(target_position)
            results['returns'].append(pnl / (capital - pnl + 1e-10))
            results['portfolio_value'].append(capital)
            results['drawdown'].append(drawdown)
            results['mediation_metrics'].append(mediation_info)

            # Update position
            position = target_position

        # Convert to DataFrame
        results_df = pd.DataFrame({
            'date': results['dates'],
            'signal': results['signals'],
            'position': results['positions'],
            'return': results['returns'],
            'portfolio_value': results['portfolio_value'],
            'drawdown': results['drawdown']
        })
        results_df.set_index('date', inplace=True)

        # Add mediation metrics
        mediation_df = pd.DataFrame(results['mediation_metrics'], index=results['dates'])
        results_df = pd.concat([results_df, mediation_df], axis=1)

        # Calculate analytics
        analytics = self._calculate_analytics(results_df)

        return {
            'results': results_df,
            'analytics': analytics
        }

    def _calculate_position(self, signal: float, capital: float, row: pd.Series) -> float:
        """Calculate position size with risk management"""
        max_position = capital * self.config.max_position_size
        return signal * max_position

    def _calculate_analytics(self, results: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance analytics"""
        returns = results['return']
        portfolio_values = results['portfolio_value']

        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        n_days = len(returns)

        # Annualized metrics
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        annual_vol = returns.std() * np.sqrt(252)

        # Risk metrics
        sharpe = annual_return / (annual_vol + 1e-10)
        max_dd = results['drawdown'].max()
        calmar = annual_return / (max_dd + 1e-10)

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-10
        sortino = annual_return / downside_vol

        # Win rate
        winning_days = (returns > 0).sum()
        total_trading_days = (returns != 0).sum()
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / (gross_loss + 1e-10)

        # Mediation-specific analytics
        mediation_analytics = self._analyze_mediation_dynamics(results)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_trading_days': total_trading_days,
            **mediation_analytics
        }

    def _analyze_mediation_dynamics(self, results: pd.DataFrame) -> Dict:
        """Analyze how mediation dynamics affect strategy performance"""
        analytics = {}

        if 'proportion_mediated' in results.columns:
            # Split by mediation strength
            high_mediation = results[results['proportion_mediated'] > 0.5]
            low_mediation = results[results['proportion_mediated'] <= 0.5]

            analytics['return_high_mediation'] = high_mediation['return'].mean() * 252
            analytics['return_low_mediation'] = low_mediation['return'].mean() * 252
            analytics['sharpe_high_mediation'] = (
                high_mediation['return'].mean() / (high_mediation['return'].std() + 1e-10) * np.sqrt(252)
            )
            analytics['sharpe_low_mediation'] = (
                low_mediation['return'].mean() / (low_mediation['return'].std() + 1e-10) * np.sqrt(252)
            )

        if 'mediator_activated' in results.columns:
            activated = results[results['mediator_activated'] == True]
            not_activated = results[results['mediator_activated'] == False]

            analytics['return_mediator_active'] = activated['return'].mean() * 252 if len(activated) > 0 else 0
            analytics['return_mediator_inactive'] = not_activated['return'].mean() * 252 if len(not_activated) > 0 else 0

        return analytics


def mediation_momentum_strategy(state: Dict, params: Dict) -> Tuple[float, Dict]:
    """
    Example strategy: Mediation-aware momentum.

    Idea: Use momentum signal but adjust based on mediation channel activation.
    """
    lookback_data = state['lookback_data']

    if len(lookback_data) < params.get('lookback', 60):
        return 0, {'proportion_mediated': 0, 'mediator_activated': False}

    # Convert to arrays
    df = pd.DataFrame(lookback_data)

    X = df['market_return'].values if 'market_return' in df.columns else df.get('X', np.zeros(len(df))).values
    M = df['volume'].values if 'volume' in df.columns else df.get('M', np.zeros(len(df))).values
    Y = df['return'].values if 'return' in df.columns else df.get('Y', np.zeros(len(df))).values

    # Estimate mediation parameters (simplified)
    X_lag = X[:-1]
    M_curr = M[1:]
    Y_curr = Y[1:]

    # Correlations as proxies for paths
    corr_xm = np.corrcoef(X_lag, M_curr)[0, 1] if len(X_lag) > 2 else 0
    corr_my = np.corrcoef(M_curr, Y_curr)[0, 1] if len(M_curr) > 2 else 0
    corr_xy = np.corrcoef(X_lag, Y_curr)[0, 1] if len(X_lag) > 2 else 0

    # Indirect effect proxy
    indirect = corr_xm * corr_my

    # Direct effect proxy (controlled for M, simplified)
    direct = corr_xy - indirect

    # Proportion mediated
    total = abs(direct) + abs(indirect)
    prop_mediated = abs(indirect) / total if total > 0.01 else 0

    # Is mediator currently activated?
    recent_M = M[-5:].mean()
    historical_M = M[:-5].mean()
    mediator_activated = abs(recent_M) > abs(historical_M) * 1.2

    # Momentum signal
    momentum = Y[-20:].mean() - Y[-60:-20].mean() if len(Y) >= 60 else 0

    # Adjust signal based on mediation
    if mediator_activated and prop_mediated > params.get('mediation_threshold', 0.3):
        # Expect full effect - stronger signal
        signal = np.sign(momentum) * min(abs(momentum) * 2, 1.0)
    else:
        # Expect only direct effect - weaker signal
        signal = np.sign(momentum) * min(abs(momentum), 0.5)

    mediation_info = {
        'proportion_mediated': prop_mediated,
        'mediator_activated': mediator_activated,
        'direct_effect': direct,
        'indirect_effect': indirect,
        'momentum': momentum
    }

    return signal, mediation_info


# Example usage
if __name__ == "__main__":
    # Generate realistic market data
    np.random.seed(42)
    n = 750  # 3 years of daily data

    dates = pd.date_range('2022-01-01', periods=n, freq='B')

    # Regime-switching parameters
    regimes = np.zeros(n)
    regimes[100:200] = 1  # Bull with high mediation
    regimes[400:500] = 1  # Another bull period

    # Market returns
    market_return = np.random.randn(n) * 0.01 + 0.0002  # Slight positive drift
    market_return += regimes * 0.002  # Higher returns in bull regime

    # Volume (mediator)
    volume = 1e6 + np.random.randn(n) * 1e5
    volume += market_return * 1e7  # Volume responds to returns
    volume += regimes * 5e5  # Higher volume in bull regime

    # Stock return (target)
    noise = np.random.randn(n) * 0.015
    stock_return = 0.5 * market_return + 0.3 * (volume - volume.mean()) / volume.std() * 0.01 + noise
    stock_return += regimes * 0.003  # Extra return in bull regime

    # Create DataFrame
    data = pd.DataFrame({
        'market_return': market_return,
        'X': market_return,
        'volume': volume,
        'M': volume,
        'return': stock_return,
        'Y': stock_return
    }, index=dates)

    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        transaction_cost_bps=10,
        slippage_bps=5,
        max_position_size=0.3
    )

    engine = MediationBacktestEngine(config)

    results = engine.run_backtest(
        data=data,
        strategy_func=mediation_momentum_strategy,
        strategy_params={
            'lookback': 60,
            'mediation_threshold': 0.25
        }
    )

    # Print analytics
    print("="*70)
    print("MEDIATION MOMENTUM STRATEGY - BACKTEST RESULTS")
    print("="*70)

    for key, value in results['analytics'].items():
        if isinstance(value, float):
            if 'return' in key or 'ratio' in key or 'drawdown' in key:
                print(f"{key:30s}: {value:10.2%}")
            else:
                print(f"{key:30s}: {value:10.4f}")
        else:
            print(f"{key:30s}: {value}")

    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Portfolio value
    axes[0].plot(results['results']['portfolio_value'], linewidth=1.5)
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].set_title('Strategy Equity Curve')
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    axes[1].fill_between(results['results'].index, 0, -results['results']['drawdown'],
                         color='red', alpha=0.5)
    axes[1].set_ylabel('Drawdown')
    axes[1].set_title('Underwater Plot')
    axes[1].grid(True, alpha=0.3)

    # Mediation dynamics
    axes[2].plot(results['results']['proportion_mediated'].rolling(20).mean(),
                 label='Proportion Mediated (20d MA)', color='blue')
    axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    axes[2].set_ylabel('Proportion Mediated')
    axes[2].set_title('Mediation Strength Over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Rolling Sharpe
    rolling_return = results['results']['return'].rolling(60).mean() * 252
    rolling_vol = results['results']['return'].rolling(60).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / (rolling_vol + 1e-10)
    axes[3].plot(rolling_sharpe, label='60-day Rolling Sharpe')
    axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Sharpe Ratio')
    axes[3].set_title('Rolling Risk-Adjusted Performance')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mediation_backtest_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nBacktest complete. Results saved to mediation_backtest_results.png")
```

## Rust Implementation

For high-performance production systems, we provide a Rust implementation:

```rust
//! Mediation Analysis for Financial Trading
//!
//! High-performance Rust implementation of causal mediation analysis
//! for algorithmic trading systems.

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve;
use std::f64;

/// Results from mediation analysis
#[derive(Debug, Clone)]
pub struct MediationResults {
    /// Total effect of X on Y
    pub total_effect: f64,
    pub total_effect_se: f64,

    /// Direct effect (X → Y controlling for M)
    pub direct_effect: f64,
    pub direct_effect_se: f64,

    /// Indirect effect (X → M → Y)
    pub indirect_effect: f64,
    pub indirect_effect_se: f64,

    /// Path coefficients
    pub a_path: f64,  // X → M
    pub b_path: f64,  // M → Y | X

    /// Statistical tests
    pub sobel_z: f64,
    pub sobel_pvalue: f64,

    /// Proportion of effect mediated
    pub proportion_mediated: f64,
}

/// Mediation analyzer for financial data
pub struct MediationAnalyzer {
    /// Lookback window for rolling estimation
    lookback: usize,

    /// History buffers
    x_history: Vec<f64>,
    m_history: Vec<f64>,
    y_history: Vec<f64>,
}

impl MediationAnalyzer {
    /// Create new analyzer
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback,
            x_history: Vec::with_capacity(lookback),
            m_history: Vec::with_capacity(lookback),
            y_history: Vec::with_capacity(lookback),
        }
    }

    /// Update with new observation
    pub fn update(&mut self, x: f64, m: f64, y: f64) {
        self.x_history.push(x);
        self.m_history.push(m);
        self.y_history.push(y);

        // Keep only lookback window
        if self.x_history.len() > self.lookback {
            self.x_history.remove(0);
            self.m_history.remove(0);
            self.y_history.remove(0);
        }
    }

    /// Run mediation analysis on current data
    pub fn analyze(&self) -> Option<MediationResults> {
        if self.x_history.len() < 30 {
            return None;
        }

        let n = self.x_history.len();

        // Convert to arrays
        let x = Array1::from_vec(self.x_history.clone());
        let m = Array1::from_vec(self.m_history.clone());
        let y = Array1::from_vec(self.y_history.clone());

        // Standardize
        let x_std = standardize(&x);
        let m_std = standardize(&m);
        let y_std = standardize(&y);

        // Step 1: Total effect (Y ~ X)
        let (c, c_se) = simple_regression(&x_std, &y_std);

        // Step 2: a path (M ~ X)
        let (a, a_se) = simple_regression(&x_std, &m_std);

        // Step 3: Direct effect and b path (Y ~ X + M)
        let (c_prime, b, c_prime_se, b_se) = multiple_regression(&x_std, &m_std, &y_std);

        // Indirect effect
        let indirect = a * b;

        // Sobel test
        let sobel_se = (b * b * a_se * a_se + a * a * b_se * b_se).sqrt();
        let sobel_z = indirect / sobel_se;
        let sobel_pvalue = 2.0 * (1.0 - normal_cdf(sobel_z.abs()));

        // Proportion mediated
        let prop_mediated = if c.abs() > 1e-10 {
            indirect / c
        } else {
            0.0
        };

        Some(MediationResults {
            total_effect: c,
            total_effect_se: c_se,
            direct_effect: c_prime,
            direct_effect_se: c_prime_se,
            indirect_effect: indirect,
            indirect_effect_se: sobel_se,
            a_path: a,
            b_path: b,
            sobel_z,
            sobel_pvalue,
            proportion_mediated: prop_mediated,
        })
    }

    /// Generate trading signal based on mediation analysis
    pub fn generate_signal(&self, current_x: f64, current_m: f64) -> Option<f64> {
        let results = self.analyze()?;

        // Standardize current values
        let x_mean: f64 = self.x_history.iter().sum::<f64>() / self.x_history.len() as f64;
        let x_std_dev = std_dev(&self.x_history);
        let x_norm = (current_x - x_mean) / (x_std_dev + 1e-10);

        let m_mean: f64 = self.m_history.iter().sum::<f64>() / self.m_history.len() as f64;
        let m_std_dev = std_dev(&self.m_history);
        let m_norm = (current_m - m_mean) / (m_std_dev + 1e-10);

        // Expected mediator value
        let m_expected = results.a_path * x_norm;

        // Is mediator activated?
        let mediator_activated = m_norm.abs() > m_expected.abs() * 1.2;

        // Calculate expected effect
        let expected_effect = if mediator_activated && results.proportion_mediated.abs() > 0.3 {
            // Full effect
            (results.direct_effect + results.indirect_effect) * x_norm
        } else {
            // Only direct effect
            results.direct_effect * x_norm
        };

        // Signal: bounded between -1 and 1
        Some(expected_effect.max(-1.0).min(1.0))
    }
}

/// Standardize array to zero mean and unit variance
fn standardize(arr: &Array1<f64>) -> Array1<f64> {
    let mean = arr.mean().unwrap_or(0.0);
    let std = arr.std(0.0);
    if std < 1e-10 {
        return arr.clone();
    }
    (arr - mean) / std
}

/// Simple linear regression: y = b0 + b1*x
fn simple_regression(x: &Array1<f64>, y: &Array1<f64>) -> (f64, f64) {
    let n = x.len() as f64;

    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;

    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        ss_xy += x_diff * y_diff;
        ss_xx += x_diff * x_diff;
    }

    if ss_xx < 1e-10 {
        return (0.0, f64::MAX);
    }

    let b1 = ss_xy / ss_xx;
    let b0 = y_mean - b1 * x_mean;

    // Residual standard error
    let mut ss_res = 0.0;
    for i in 0..x.len() {
        let pred = b0 + b1 * x[i];
        let resid = y[i] - pred;
        ss_res += resid * resid;
    }

    let mse = ss_res / (n - 2.0);
    let se_b1 = (mse / ss_xx).sqrt();

    (b1, se_b1)
}

/// Multiple regression: y = b0 + b1*x + b2*m
fn multiple_regression(
    x: &Array1<f64>,
    m: &Array1<f64>,
    y: &Array1<f64>,
) -> (f64, f64, f64, f64) {
    let n = x.len();

    // Design matrix [1, x, m]
    let mut design = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        design[[i, 0]] = 1.0;
        design[[i, 1]] = x[i];
        design[[i, 2]] = m[i];
    }

    // OLS: (X'X)^-1 X'y
    let xt = design.t();
    let xtx = xt.dot(&design);
    let xty = xt.dot(y);

    // Solve using LU decomposition (simplified)
    let beta = match solve_linear_system(&xtx, &xty) {
        Some(b) => b,
        None => return (0.0, 0.0, f64::MAX, f64::MAX),
    };

    let b1 = beta[1];  // Coefficient for x (direct effect)
    let b2 = beta[2];  // Coefficient for m (b path)

    // Calculate standard errors
    let y_pred = design.dot(&beta);
    let residuals = y - &y_pred;
    let mse = residuals.dot(&residuals) / (n as f64 - 3.0);

    // Variance-covariance matrix
    let xtx_inv = match invert_matrix(&xtx) {
        Some(inv) => inv,
        None => return (b1, b2, f64::MAX, f64::MAX),
    };

    let se_b1 = (mse * xtx_inv[[1, 1]]).sqrt();
    let se_b2 = (mse * xtx_inv[[2, 2]]).sqrt();

    (b1, b2, se_b1, se_b2)
}

/// Standard deviation
fn std_dev(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean: f64 = data.iter().sum::<f64>() / n;
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Solve linear system Ax = b (3x3 for our case)
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    // Simple 3x3 Cramer's rule for small matrices
    if a.nrows() != 3 || a.ncols() != 3 || b.len() != 3 {
        return None;
    }

    let det = determinant_3x3(a);
    if det.abs() < 1e-10 {
        return None;
    }

    let mut result = Array1::<f64>::zeros(3);

    for i in 0..3 {
        let mut a_i = a.clone();
        for j in 0..3 {
            a_i[[j, i]] = b[j];
        }
        result[i] = determinant_3x3(&a_i) / det;
    }

    Some(result)
}

/// 3x3 matrix determinant
fn determinant_3x3(a: &Array2<f64>) -> f64 {
    a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
        - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
        + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]])
}

/// 3x3 matrix inverse
fn invert_matrix(a: &Array2<f64>) -> Option<Array2<f64>> {
    let det = determinant_3x3(a);
    if det.abs() < 1e-10 {
        return None;
    }

    let mut inv = Array2::<f64>::zeros((3, 3));

    // Adjugate matrix / determinant
    inv[[0, 0]] = (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]]) / det;
    inv[[0, 1]] = (a[[0, 2]] * a[[2, 1]] - a[[0, 1]] * a[[2, 2]]) / det;
    inv[[0, 2]] = (a[[0, 1]] * a[[1, 2]] - a[[0, 2]] * a[[1, 1]]) / det;
    inv[[1, 0]] = (a[[1, 2]] * a[[2, 0]] - a[[1, 0]] * a[[2, 2]]) / det;
    inv[[1, 1]] = (a[[0, 0]] * a[[2, 2]] - a[[0, 2]] * a[[2, 0]]) / det;
    inv[[1, 2]] = (a[[0, 2]] * a[[1, 0]] - a[[0, 0]] * a[[1, 2]]) / det;
    inv[[2, 0]] = (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]) / det;
    inv[[2, 1]] = (a[[0, 1]] * a[[2, 0]] - a[[0, 0]] * a[[2, 1]]) / det;
    inv[[2, 2]] = (a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]) / det;

    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mediation_analysis() {
        let mut analyzer = MediationAnalyzer::new(100);

        // Generate synthetic data
        let mut rng = rand::thread_rng();

        for i in 0..100 {
            let x = rand::random::<f64>() * 2.0 - 1.0;  // U(-1, 1)
            let m = 0.5 * x + (rand::random::<f64>() - 0.5) * 0.3;
            let y = 0.3 * x + 0.4 * m + (rand::random::<f64>() - 0.5) * 0.2;

            analyzer.update(x, m, y);
        }

        let results = analyzer.analyze().expect("Should have results");

        // Check that effects are in reasonable range
        assert!(results.total_effect.abs() < 2.0);
        assert!(results.direct_effect.abs() < 2.0);
        assert!(results.indirect_effect.abs() < 2.0);

        // Total ≈ Direct + Indirect
        let total_check = (results.total_effect - results.direct_effect - results.indirect_effect).abs();
        assert!(total_check < 0.2);
    }

    #[test]
    fn test_signal_generation() {
        let mut analyzer = MediationAnalyzer::new(60);

        // Fill with data
        for i in 0..60 {
            let x = (i as f64 - 30.0) / 30.0;
            let m = 0.5 * x + (rand::random::<f64>() - 0.5) * 0.2;
            let y = 0.3 * x + 0.4 * m + (rand::random::<f64>() - 0.5) * 0.15;

            analyzer.update(x, m, y);
        }

        // Test signal generation
        let signal = analyzer.generate_signal(0.5, 0.3);
        assert!(signal.is_some());
        let s = signal.unwrap();
        assert!(s >= -1.0 && s <= 1.0);
    }
}
```

## Python Implementation

Complete Python module for mediation analysis:

```python
"""
Mediation Analysis Module for Financial Trading

This module provides comprehensive tools for mediation analysis
in financial applications, including:
- Classical Baron-Kenny mediation
- Causal mediation analysis with sensitivity
- Bootstrap confidence intervals
- Trading signal generation
"""

from .model import (
    MediationAnalyzer,
    CausalMediationAnalysis,
    baron_kenny_mediation,
    bootstrap_mediation,
    MediationResults
)

from .data import (
    prepare_stock_data,
    prepare_crypto_data_bybit
)

from .strategy import (
    MediationTradingStrategy,
    MediationSignal,
    MediationBacktester
)

__version__ = "0.1.0"
__all__ = [
    "MediationAnalyzer",
    "CausalMediationAnalysis",
    "baron_kenny_mediation",
    "bootstrap_mediation",
    "MediationResults",
    "prepare_stock_data",
    "prepare_crypto_data_bybit",
    "MediationTradingStrategy",
    "MediationSignal",
    "MediationBacktester"
]
```

## Best Practices

### When to Use Mediation Analysis in Trading

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MEDIATION ANALYSIS DECISION GUIDE                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ✅ USE MEDIATION WHEN:                                              │
│                                                                       │
│  1. You have a clear causal hypothesis                               │
│     "News → Sentiment → Returns"                                     │
│     "Order flow → Liquidity → Price impact"                          │
│                                                                       │
│  2. The mediator is measurable and tradeable                         │
│     Volume, volatility, sentiment scores, funding rates              │
│                                                                       │
│  3. You want to understand WHY a factor works                        │
│     Not just that momentum predicts returns, but HOW                 │
│                                                                       │
│  4. You need to time your strategies                                 │
│     When is the mediation channel active vs blocked?                 │
│                                                                       │
│  ❌ AVOID MEDIATION WHEN:                                            │
│                                                                       │
│  1. You only care about prediction, not mechanism                    │
│     Pure ML approaches may be more effective                         │
│                                                                       │
│  2. The potential mediators are not observable                       │
│     "Private information" is not a measurable mediator              │
│                                                                       │
│  3. You cannot defend the identification assumptions                 │
│     If confounding is severe and unmeasurable                       │
│                                                                       │
│  4. Sample sizes are very small                                      │
│     Need 200+ observations for reliable mediation                   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Common Pitfalls

1. **Ignoring measurement error**: Noisy mediators attenuate indirect effects
2. **Assuming linearity**: Financial relationships are often nonlinear
3. **Ignoring time dynamics**: Mediation effects can be lagged and time-varying
4. **Post-treatment confounding**: Variables affected by treatment cannot be controlled
5. **Over-interpreting small effects**: Statistical significance ≠ economic significance

### Recommended Workflow

```
1. HYPOTHESIS DEVELOPMENT
   └── What mechanism do you think explains the effect?
   └── Is the mediator measurable and reliable?

2. DATA PREPARATION
   └── Ensure proper temporal ordering (X precedes M precedes Y)
   └── Handle missing data appropriately
   └── Standardize if comparing effect sizes

3. EXPLORATORY ANALYSIS
   └── Check correlations: X-M, M-Y, X-Y
   └── Plot time series to understand dynamics
   └── Look for regime changes

4. MEDIATION ESTIMATION
   └── Run Baron-Kenny as baseline
   └── Use bootstrap for confidence intervals
   └── Run causal mediation if assumptions hold

5. SENSITIVITY ANALYSIS
   └── Test robustness to unmeasured confounding
   └── Find breakdown point for conclusions

6. TRADING STRATEGY DEVELOPMENT
   └── Design signals based on mediation insights
   └── Backtest with realistic transaction costs
   └── Analyze performance by mediation regime

7. VALIDATION
   └── Out-of-sample testing
   └── Paper trading before live deployment
```

## Resources

### Key Papers

1. **Baron, R. M., & Kenny, D. A. (1986)**. "The moderator-mediator variable distinction in social psychological research." *Journal of Personality and Social Psychology*, 51(6), 1173-1182.
   - The foundational paper on mediation analysis

2. **Imai, K., Keele, L., & Tingley, D. (2010)**. "A general approach to causal mediation analysis." *Psychological Methods*, 15(4), 309-334.
   - Modern causal mediation framework with sensitivity analysis

3. **VanderWeele, T. J. (2015)**. *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press.
   - Comprehensive textbook on causal mediation

4. **Pearl, J. (2014)**. "Interpretation and identification of causal mediation." *Psychological Methods*, 19(4), 459-481.
   - Causal interpretation of direct and indirect effects

### Software

- **Python**: `statsmodels`, `mediation` package (port of R mediation)
- **R**: `mediation` package by Imai, Keele, & Tingley
- **Rust**: Custom implementation in this chapter

### Further Reading

- Chapter 96: Granger Causality Trading
- Chapter 97: PCMCI Causal Discovery
- Chapter 105: Difference-in-Differences Trading
- Chapter 107: Propensity Score Trading

---

*This chapter is part of the Machine Learning for Trading series. For questions or contributions, please open an issue on GitHub.*
