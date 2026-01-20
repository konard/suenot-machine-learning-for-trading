# Chapter 75: LLM Factor Discovery for Trading

## Overview

LLM Factor Discovery represents a paradigm shift in quantitative trading research. Traditional alpha factor mining relies on manual formula design, domain expertise, and exhaustive search algorithms. Large Language Models (LLMs) can now automate and accelerate this process by generating, evaluating, and refining trading factors through natural language understanding and code generation capabilities.

This chapter explores how to leverage LLMs like GPT-4, Claude, and specialized financial models (FinGPT) to discover new alpha factors for both traditional equity markets and cryptocurrency trading on platforms like Bybit.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Alpha Factor Basics](#alpha-factor-basics)
4. [LLM-Based Factor Discovery](#llm-based-factor-discovery)
5. [Alpha-GPT Architecture](#alpha-gpt-architecture)
6. [Factor Generation Pipeline](#factor-generation-pipeline)
7. [Factor Evaluation and Backtesting](#factor-evaluation-and-backtesting)
8. [Application to Cryptocurrency Trading](#application-to-cryptocurrency-trading)
9. [Implementation Strategy](#implementation-strategy)
10. [Risk Management](#risk-management)
11. [Performance Metrics](#performance-metrics)
12. [References](#references)

---

## Introduction

### What is Alpha Factor Discovery?

Alpha factors are quantitative signals that predict future asset returns beyond market risk. They are the building blocks of systematic trading strategies:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Alpha Factor Discovery Evolution                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional Approach:              LLM Approach:                       │
│   ┌──────────────────┐              ┌──────────────────┐                │
│   │ Domain Expert    │              │ Natural Language │                │
│   │      ↓           │              │ Research Input   │                │
│   │ Manual Formula   │              │      ↓           │                │
│   │ Design           │              │ LLM Generation   │                │
│   │      ↓           │              │      ↓           │                │
│   │ Backtest         │              │ Auto Validation  │                │
│   │      ↓           │              │      ↓           │                │
│   │ Iteration        │              │ Iterative        │                │
│   │ (weeks/months)   │              │ Refinement       │                │
│   └──────────────────┘              │ (hours/days)     │                │
│                                     └──────────────────┘                │
│                                                                          │
│   Examples of Alpha Factors:                                            │
│   • Momentum: rank(returns(20d))                                        │
│   • Mean Reversion: -zscore(close, 10)                                  │
│   • Volume Profile: ts_rank(volume, 20) * sign(returns(5d))            │
│   • Sentiment: sentiment_score * news_volume                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why LLMs for Factor Discovery?

| Aspect | Traditional Methods | LLM-Based Discovery |
|--------|---------------------|---------------------|
| Ideation | Manual brainstorming | Automated generation |
| Domain knowledge | Required | Can leverage training data |
| Formula syntax | Must be learned | Natural language → code |
| Iteration speed | Days/weeks | Minutes/hours |
| Exploration space | Limited by human capacity | Exponentially larger |
| Bias | Strong researcher bias | Can be more diverse |
| Interpretability | Variable | Can explain reasoning |

## Theoretical Foundation

### Factor Model Framework

Modern factor investing is based on the assumption that returns can be decomposed:

$$R_i = \alpha_i + \sum_{k=1}^{K} \beta_{ik} F_k + \epsilon_i$$

Where:
- $R_i$ is the return of asset $i$
- $\alpha_i$ is the asset-specific alpha (what we want to find!)
- $\beta_{ik}$ is the exposure to factor $k$
- $F_k$ is the return of factor $k$
- $\epsilon_i$ is the idiosyncratic error

### LLM as Factor Generator

LLMs can be viewed as functions that map research ideas to factor formulas:

$$f_{LLM}: \text{Research Idea} \rightarrow \text{Factor Expression}$$

The key insight is that LLMs have learned patterns from vast amounts of financial literature, research papers, and code repositories. They can:

1. **Synthesize knowledge** from multiple domains
2. **Generate novel combinations** of existing factors
3. **Translate intuition** into mathematical expressions
4. **Iterate rapidly** based on feedback

### Information Content of Factors

A good alpha factor should have:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Factor Quality Criteria                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Predictive Power (IC: Information Coefficient)                      │
│     ┌──────────────────────────────────────────────────────┐            │
│     │ Correlation between factor values and future returns  │            │
│     │ IC = corr(factor_t, returns_{t+1})                   │            │
│     │ Target: |IC| > 0.02 (2% correlation is significant)  │            │
│     └──────────────────────────────────────────────────────┘            │
│                                                                          │
│  2. Stability (IC_IR: Information Ratio)                                │
│     ┌──────────────────────────────────────────────────────┐            │
│     │ Consistency of predictive power over time            │            │
│     │ IC_IR = mean(IC) / std(IC)                           │            │
│     │ Target: IC_IR > 0.3                                  │            │
│     └──────────────────────────────────────────────────────┘            │
│                                                                          │
│  3. Uniqueness (Low Correlation with Existing Factors)                  │
│     ┌──────────────────────────────────────────────────────┐            │
│     │ Factor should capture NEW information                 │            │
│     │ corr(new_factor, existing_factors) < 0.7             │            │
│     └──────────────────────────────────────────────────────┘            │
│                                                                          │
│  4. Turnover (Trading Feasibility)                                      │
│     ┌──────────────────────────────────────────────────────┐            │
│     │ How often positions need to change                    │            │
│     │ Lower turnover = lower trading costs                  │            │
│     │ Target: Turnover < 50% per period                    │            │
│     └──────────────────────────────────────────────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Alpha Factor Basics

### Factor Expression Language

Factors are typically expressed as mathematical formulas operating on price and volume data:

```
Basic Operations:
├── returns(d)      - d-day returns: close / close.shift(d) - 1
├── rank(x)         - Cross-sectional rank (0 to 1)
├── zscore(x, d)    - Rolling z-score: (x - mean(x,d)) / std(x,d)
├── ts_rank(x, d)   - Time-series rank over d periods
├── ts_sum(x, d)    - Rolling sum over d periods
├── ts_mean(x, d)   - Rolling mean over d periods
├── ts_std(x, d)    - Rolling standard deviation
├── ts_max(x, d)    - Rolling maximum
├── ts_min(x, d)    - Rolling minimum
├── delay(x, d)     - Lagged value
├── delta(x, d)     - Change: x - delay(x, d)
├── correlation(x, y, d) - Rolling correlation
└── covariance(x, y, d)  - Rolling covariance

Market Data Variables:
├── open            - Opening price
├── high            - High price
├── low             - Low price
├── close           - Closing price
├── volume          - Trading volume
├── vwap            - Volume-weighted average price
└── amount          - Trading amount (price * volume)
```

### Example Factors

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Classic Alpha Factor Examples                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Price Momentum (12-month momentum, skip last month)                 │
│     formula: rank(returns(252) - returns(21))                           │
│     logic: Assets with positive momentum tend to continue               │
│                                                                          │
│  2. Short-Term Reversal                                                 │
│     formula: -rank(returns(5))                                          │
│     logic: Short-term losers tend to bounce back                        │
│                                                                          │
│  3. Volume-Price Divergence                                             │
│     formula: -correlation(close, volume, 10)                            │
│     logic: Price up on low volume = weakness                            │
│                                                                          │
│  4. Volatility-Adjusted Momentum                                        │
│     formula: returns(20) / ts_std(returns(1), 20)                       │
│     logic: Momentum normalized by risk                                  │
│                                                                          │
│  5. Smart Money Flow                                                    │
│     formula: ts_sum(volume * sign(close - open), 10) / ts_sum(volume,10)│
│     logic: Track if volume flows with price direction                   │
│                                                                          │
│  6. Liquidity Reversal                                                  │
│     formula: -correlation(returns(1), volume, 20) * ts_rank(volume, 20) │
│     logic: High-volume reversals are more significant                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## LLM-Based Factor Discovery

### The Alpha-GPT Paradigm

Alpha-GPT introduced a revolutionary approach to factor mining through Human-AI interaction:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Alpha-GPT System Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                     ┌─────────────────────────────────────┐             │
│                     │        USER INPUT                    │             │
│                     │  "I want a momentum factor that      │             │
│                     │   accounts for volume confirmation"  │             │
│                     └─────────────────┬───────────────────┘             │
│                                       │                                  │
│                                       ↓                                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    KNOWLEDGE COMPILATION                           │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │ │
│  │  │ External Memory │  │ Similar Examples │  │ Domain Rules    │   │ │
│  │  │ (Factor Library)│  │ (Few-shot)       │  │ (Constraints)   │   │ │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │ │
│  │           └──────────────────┬─┴────────────────────┘            │ │
│  │                              │                                    │ │
│  │                              ↓                                    │ │
│  │                    ┌─────────────────────┐                       │ │
│  │                    │   SYSTEM PROMPT     │                       │ │
│  │                    │   + User Request    │                       │ │
│  │                    └──────────┬──────────┘                       │ │
│  └───────────────────────────────┼───────────────────────────────────┘ │
│                                  │                                      │
│                                  ↓                                      │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                         LLM ENGINE                                 │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │ Generated Factor Expression:                                  │ │ │
│  │  │ rank(ts_sum(volume * sign(returns(1)), 5)) *                 │ │ │
│  │  │ rank(returns(10))                                             │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────┼───────────────────────────────────┘ │
│                                  │                                      │
│                                  ↓                                      │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      ALPHA SEARCH                                  │ │
│  │  • Syntax validation                                               │ │
│  │  • Backtest execution                                              │ │
│  │  • IC/IR calculation                                               │ │
│  │  • Turnover analysis                                               │ │
│  └───────────────────────────────┼───────────────────────────────────┘ │
│                                  │                                      │
│                                  ↓                                      │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                   RESULTS & INTERPRETATION                         │ │
│  │  • IC: 0.035, IC_IR: 0.42                                         │ │
│  │  • Annual Return: 12.3%, Sharpe: 1.8                              │ │
│  │  • "This factor combines volume-confirmed momentum..."            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Alpha-GPT 2.0: Multi-Agent Architecture

The evolution to Alpha-GPT 2.0 introduced a multi-agent system:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Alpha-GPT 2.0 Multi-Agent Pipeline                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LAYER 1: ALPHA MINING AGENT                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Tools: Factor Generator, Expression Validator, Syntax Checker   │   │
│  │ Task: Generate candidate alpha expressions from research ideas   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  LAYER 2: ALPHA MODELING AGENT                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Tools: ML Model Trainer, Feature Combiner, Ensemble Builder      │   │
│  │ Task: Combine factors into predictive models                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  LAYER 3: ALPHA ANALYSIS AGENT                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Tools: Backtester, Risk Analyzer, Performance Reporter           │   │
│  │ Task: Evaluate and report on factor/model performance            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Human-in-the-Loop: User can intervene at any layer with feedback      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Chain-of-Alpha: Latest Advances (2025)

The Chain-of-Alpha framework introduces a dual-chain architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Chain-of-Alpha Framework                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 FACTOR GENERATION CHAIN                          │   │
│  │                                                                   │   │
│  │  Market Data → LLM Analysis → Candidate Factors → Validation    │   │
│  │       ↑                                               │          │   │
│  │       └───────────────────────────────────────────────┘          │   │
│  │                      Iteration with feedback                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↕                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 FACTOR OPTIMIZATION CHAIN                        │   │
│  │                                                                   │   │
│  │  Backtest Results → Performance Analysis → Refinement Prompts   │   │
│  │       ↑                                               │          │   │
│  │       └───────────────────────────────────────────────┘          │   │
│  │                  Optimization knowledge base                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Key Innovation: Both chains work synergistically, sharing knowledge   │
│  about what works and what doesn't for continuous improvement          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Alpha-GPT Architecture

### System Components

```rust
/// Core components of an LLM-based factor discovery system
#[derive(Debug, Clone)]
pub struct FactorDiscoverySystem {
    /// LLM client for factor generation
    pub llm_client: LlmClient,

    /// Factor expression parser and validator
    pub expression_parser: ExpressionParser,

    /// Backtesting engine
    pub backtester: Backtester,

    /// Factor library for similarity search
    pub factor_library: FactorLibrary,

    /// Configuration
    pub config: DiscoveryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Minimum IC threshold for factor acceptance
    pub min_ic: f64,

    /// Minimum IC_IR threshold
    pub min_ic_ir: f64,

    /// Maximum correlation with existing factors
    pub max_correlation: f64,

    /// Maximum turnover allowed
    pub max_turnover: f64,

    /// Number of factors to generate per iteration
    pub batch_size: usize,

    /// Maximum iterations for refinement
    pub max_iterations: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            min_ic: 0.02,
            min_ic_ir: 0.3,
            max_correlation: 0.7,
            max_turnover: 0.5,
            batch_size: 10,
            max_iterations: 5,
        }
    }
}
```

### Prompt Engineering for Factor Generation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Factor Generation Prompt Template                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SYSTEM PROMPT:                                                         │
│  """                                                                    │
│  You are a quantitative researcher specializing in alpha factor        │
│  discovery. Your task is to generate trading factors that predict      │
│  future asset returns.                                                  │
│                                                                          │
│  Available operators:                                                    │
│  - rank(x): Cross-sectional rank                                        │
│  - zscore(x, d): Rolling z-score                                        │
│  - ts_rank(x, d): Time-series rank                                      │
│  - returns(d): d-day returns                                            │
│  - ts_sum(x, d), ts_mean(x, d), ts_std(x, d): Rolling statistics       │
│  - correlation(x, y, d), covariance(x, y, d): Rolling correlations     │
│  - delta(x, d): Change over d periods                                   │
│  - delay(x, d): Lagged value                                            │
│                                                                          │
│  Available data:                                                        │
│  - open, high, low, close, volume, vwap, amount                        │
│                                                                          │
│  Requirements:                                                           │
│  - Generate valid factor expressions                                    │
│  - Explain the economic intuition                                       │
│  - Consider transaction costs (prefer lower turnover)                   │
│  """                                                                    │
│                                                                          │
│  USER PROMPT:                                                           │
│  """                                                                    │
│  Generate alpha factors based on this research idea: {research_idea}    │
│                                                                          │
│  Recent performance of similar factors:                                  │
│  {similar_factors_performance}                                           │
│                                                                          │
│  Please output:                                                          │
│  1. Factor expression                                                   │
│  2. Economic rationale                                                  │
│  3. Expected behavior (momentum/reversal/etc.)                          │
│  4. Potential risks and limitations                                      │
│  """                                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Factor Generation Pipeline

### End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Factor Discovery Pipeline                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 1: IDEATION                                                      │
│  ───────────────────────────────────────────────────────────────────    │
│  Input: Research papers, market anomalies, intuitions                   │
│  Output: List of factor hypotheses                                       │
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ Academic Papers │ →  │ LLM Summarizer  │ →  │ Factor Ideas    │     │
│  │ Market News     │    │ & Translator    │    │ in NL           │     │
│  │ Anomaly Reports │    └─────────────────┘    └─────────────────┘     │
│  └─────────────────┘                                                    │
│                                                                          │
│  PHASE 2: GENERATION                                                    │
│  ───────────────────────────────────────────────────────────────────    │
│  Input: Factor ideas                                                    │
│  Output: Candidate factor expressions                                   │
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ Factor Ideas    │ →  │ LLM Generator   │ →  │ Expression      │     │
│  │ + Examples      │    │ (with RAG)      │    │ Candidates      │     │
│  │ + Constraints   │    └─────────────────┘    └─────────────────┘     │
│  └─────────────────┘                                                    │
│                                                                          │
│  PHASE 3: VALIDATION                                                    │
│  ───────────────────────────────────────────────────────────────────    │
│  Input: Factor expressions                                               │
│  Output: Valid, computable factors                                      │
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ Expression      │ →  │ Parser &        │ →  │ Valid Factors   │     │
│  │ Candidates      │    │ Validator       │    │                 │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│                                                                          │
│  PHASE 4: EVALUATION                                                    │
│  ───────────────────────────────────────────────────────────────────    │
│  Input: Valid factors                                                   │
│  Output: Performance metrics, rankings                                  │
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ Valid Factors   │ →  │ Backtesting     │ →  │ IC, IR, Sharpe  │     │
│  │ + Market Data   │    │ Engine          │    │ Turnover        │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│                                                                          │
│  PHASE 5: REFINEMENT                                                    │
│  ───────────────────────────────────────────────────────────────────    │
│  Input: Performance feedback                                             │
│  Output: Improved factors                                                │
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │ Low-performing  │ →  │ LLM Refiner     │ →  │ Improved        │     │
│  │ Factors         │    │ (with feedback) │    │ Expressions     │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│                                                                          │
│  Iterate until convergence or max_iterations reached                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Factor Expression Parser

```rust
/// Factor expression parser using a simple DSL
#[derive(Debug, Clone)]
pub enum FactorExpr {
    /// Raw market data variables
    Variable(String),  // "close", "volume", etc.

    /// Constant value
    Constant(f64),

    /// Binary operations
    Add(Box<FactorExpr>, Box<FactorExpr>),
    Sub(Box<FactorExpr>, Box<FactorExpr>),
    Mul(Box<FactorExpr>, Box<FactorExpr>),
    Div(Box<FactorExpr>, Box<FactorExpr>),

    /// Unary operations
    Neg(Box<FactorExpr>),
    Abs(Box<FactorExpr>),
    Sign(Box<FactorExpr>),
    Log(Box<FactorExpr>),

    /// Cross-sectional operations
    Rank(Box<FactorExpr>),
    ZScore(Box<FactorExpr>, usize),  // (expr, window)

    /// Time-series operations
    Returns(usize),                   // window
    TsRank(Box<FactorExpr>, usize),
    TsSum(Box<FactorExpr>, usize),
    TsMean(Box<FactorExpr>, usize),
    TsStd(Box<FactorExpr>, usize),
    TsMax(Box<FactorExpr>, usize),
    TsMin(Box<FactorExpr>, usize),
    Delay(Box<FactorExpr>, usize),
    Delta(Box<FactorExpr>, usize),

    /// Correlation operations
    Correlation(Box<FactorExpr>, Box<FactorExpr>, usize),
    Covariance(Box<FactorExpr>, Box<FactorExpr>, usize),
}

impl FactorExpr {
    /// Parse a factor expression from string
    pub fn parse(expr: &str) -> Result<Self, ParseError> {
        // Implementation uses recursive descent parser
        // or pest/nom parser combinators
        todo!()
    }

    /// Evaluate the factor on market data
    pub fn evaluate(&self, data: &MarketData) -> Result<Vec<f64>, EvalError> {
        match self {
            Self::Variable(name) => data.get_column(name),
            Self::Constant(val) => Ok(vec![*val; data.len()]),
            Self::Add(a, b) => {
                let a_vals = a.evaluate(data)?;
                let b_vals = b.evaluate(data)?;
                Ok(a_vals.iter().zip(b_vals.iter())
                    .map(|(x, y)| x + y).collect())
            }
            // ... other cases
            _ => todo!()
        }
    }
}
```

## Factor Evaluation and Backtesting

### Information Coefficient Calculation

```rust
/// Calculate Information Coefficient (IC) between factor and returns
pub fn calculate_ic(
    factor_values: &[f64],
    forward_returns: &[f64],
) -> f64 {
    // Use Spearman rank correlation
    let n = factor_values.len();

    // Rank both series
    let factor_ranks = rank_values(factor_values);
    let return_ranks = rank_values(forward_returns);

    // Calculate Spearman correlation
    let mean_f: f64 = factor_ranks.iter().sum::<f64>() / n as f64;
    let mean_r: f64 = return_ranks.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_f = 0.0;
    let mut var_r = 0.0;

    for i in 0..n {
        let df = factor_ranks[i] - mean_f;
        let dr = return_ranks[i] - mean_r;
        cov += df * dr;
        var_f += df * df;
        var_r += dr * dr;
    }

    if var_f == 0.0 || var_r == 0.0 {
        return 0.0;
    }

    cov / (var_f.sqrt() * var_r.sqrt())
}

/// Calculate IC statistics over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ICStats {
    /// Mean IC
    pub mean_ic: f64,

    /// IC standard deviation
    pub ic_std: f64,

    /// Information Ratio (IC_IR = mean_ic / ic_std)
    pub ic_ir: f64,

    /// Percentage of positive IC periods
    pub positive_ic_ratio: f64,

    /// T-statistic for IC significance
    pub t_stat: f64,
}
```

### Backtesting Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Factor Backtesting Framework                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: Factor Computation                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ For each date t in backtest period:                               │   │
│  │   factor_values[t] = compute_factor(market_data[..t])            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  STEP 2: Portfolio Construction                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Long-Short Portfolio:                                             │   │
│  │   long_assets  = top_decile(factor_values)                       │   │
│  │   short_assets = bottom_decile(factor_values)                    │   │
│  │   weights = equal_weight within each group                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  STEP 3: Return Attribution                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ portfolio_return[t] = Σ(weight[i] * return[i])                   │   │
│  │ long_return[t]      = mean(returns of long assets)               │   │
│  │ short_return[t]     = mean(returns of short assets)              │   │
│  │ spread_return[t]    = long_return[t] - short_return[t]           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  STEP 4: Performance Metrics                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ IC, IC_IR, Sharpe, Sortino, Max Drawdown, Turnover, etc.        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Application to Cryptocurrency Trading

### Bybit Integration for Factor-Based Trading

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Crypto Factor Trading Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                     DATA INGESTION                                  │ │
│  │  Bybit API ──→ ┐                                                   │ │
│  │  Binance   ──→ │──→ OHLCV Data ──→ Feature Engineering           │ │
│  │  On-chain  ──→ ┘                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                           │                                              │
│                           ↓                                              │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                     FACTOR COMPUTATION                              │ │
│  │                                                                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │ │
│  │  │ Momentum    │  │ Reversal    │  │ Volume-Based│               │ │
│  │  │ Factors     │  │ Factors     │  │ Factors     │               │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │ │
│  │         │                │                │                        │ │
│  │         └────────────────┴────────────────┘                        │ │
│  │                          │                                          │ │
│  │                          ↓                                          │ │
│  │                ┌─────────────────┐                                 │ │
│  │                │ Factor Combiner │                                 │ │
│  │                │ (ML or Linear)  │                                 │ │
│  │                └────────┬────────┘                                 │ │
│  └─────────────────────────┼─────────────────────────────────────────┘ │
│                            │                                            │
│                            ↓                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                     SIGNAL GENERATION                               │ │
│  │                                                                     │ │
│  │  Combined Score ──→ Position Sizing ──→ Risk Limits               │ │
│  │                                              │                      │ │
│  │                                              ↓                      │ │
│  │                                    ┌─────────────────┐             │ │
│  │                                    │ Trade Signals   │             │ │
│  │                                    │ BTC: +0.3 (long)│             │ │
│  │                                    │ ETH: -0.2 (short│             │ │
│  │                                    │ SOL: +0.5 (long)│             │ │
│  │                                    └────────┬────────┘             │ │
│  └─────────────────────────────────────────────┼─────────────────────┘ │
│                                                │                        │
│                                                ↓                        │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                     EXECUTION                                       │ │
│  │                                                                     │ │
│  │  Trade Signals ──→ Order Routing ──→ Bybit API ──→ Confirmation  │ │
│  │                                                                     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Crypto-Specific Factors

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Cryptocurrency Factor Examples                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Funding Rate Momentum                                               │
│     formula: ts_mean(funding_rate, 8) * rank(returns(24h))             │
│     logic: Persistent funding indicates directional conviction          │
│                                                                          │
│  2. Open Interest Divergence                                            │
│     formula: delta(open_interest, 4h) / ts_std(open_interest, 24)      │
│     logic: OI expansion can signal trend continuation                   │
│                                                                          │
│  3. Liquidation Asymmetry                                               │
│     formula: (long_liquidations - short_liquidations) /                │
│              (long_liquidations + short_liquidations)                   │
│     logic: Lopsided liquidations can signal reversal                    │
│                                                                          │
│  4. Whale Accumulation                                                  │
│     formula: ts_sum(exchange_outflow, 7d) / ts_mean(volume, 30d)       │
│     logic: Outflows from exchanges suggest accumulation                 │
│                                                                          │
│  5. Cross-Exchange Spread                                               │
│     formula: zscore(price_binance - price_bybit, 24)                   │
│     logic: Arbitrage opportunities signal mispricing                    │
│                                                                          │
│  6. Social Momentum                                                     │
│     formula: rank(delta(social_volume, 1d)) *                          │
│              sign(sentiment_score)                                      │
│     logic: Rising positive attention can lead price                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Real-Time Factor Computation

```rust
/// Real-time factor computation for crypto trading
pub struct RealtimeFactorEngine {
    /// Factor definitions
    factors: Vec<CompiledFactor>,

    /// Rolling data buffers
    data_buffer: RollingBuffer,

    /// Current factor values
    current_values: HashMap<String, f64>,

    /// Update frequency
    update_interval: Duration,
}

impl RealtimeFactorEngine {
    /// Update factors with new market data
    pub async fn update(&mut self, tick: &MarketTick) -> Result<FactorUpdate, EngineError> {
        // Add new data to buffer
        self.data_buffer.push(tick);

        // Recompute factors that need updating
        let mut updates = HashMap::new();
        for factor in &self.factors {
            if factor.needs_update(&self.data_buffer) {
                let new_value = factor.compute(&self.data_buffer)?;
                let old_value = self.current_values.get(&factor.name);

                if Some(&new_value) != old_value {
                    updates.insert(factor.name.clone(), FactorChange {
                        old: old_value.copied(),
                        new: new_value,
                        timestamp: tick.timestamp,
                    });
                    self.current_values.insert(factor.name.clone(), new_value);
                }
            }
        }

        Ok(FactorUpdate {
            timestamp: tick.timestamp,
            changes: updates,
        })
    }

    /// Get current signal for trading
    pub fn get_combined_signal(&self) -> TradingSignal {
        // Combine factor values using trained weights
        let mut score = 0.0;
        for (name, value) in &self.current_values {
            if let Some(weight) = self.factor_weights.get(name) {
                score += value * weight;
            }
        }

        TradingSignal {
            score,
            direction: if score > 0.3 { Direction::Long }
                      else if score < -0.3 { Direction::Short }
                      else { Direction::Neutral },
            confidence: score.abs().min(1.0),
        }
    }
}
```

## Implementation Strategy

### Module Architecture

```
75_llm_factor_discovery/
├── Cargo.toml
├── README.md
├── README.ru.md
├── readme.simple.md
├── readme.simple.ru.md
├── src/
│   ├── lib.rs                    # Library root
│   ├── discovery/
│   │   ├── mod.rs               # Discovery module
│   │   ├── llm_client.rs        # LLM API client
│   │   ├── prompt_builder.rs    # Prompt engineering
│   │   ├── generator.rs         # Factor generation
│   │   └── refiner.rs           # Factor refinement
│   ├── parser/
│   │   ├── mod.rs               # Parser module
│   │   ├── expression.rs        # Factor expression AST
│   │   ├── lexer.rs             # Tokenizer
│   │   └── validator.rs         # Expression validation
│   ├── evaluation/
│   │   ├── mod.rs               # Evaluation module
│   │   ├── ic.rs                # IC calculation
│   │   ├── backtester.rs        # Backtesting engine
│   │   └── metrics.rs           # Performance metrics
│   ├── data/
│   │   ├── mod.rs               # Data module
│   │   ├── bybit.rs             # Bybit client
│   │   ├── types.rs             # Data types
│   │   └── buffer.rs            # Rolling buffers
│   ├── strategy/
│   │   ├── mod.rs               # Strategy module
│   │   ├── signals.rs           # Signal generation
│   │   ├── combiner.rs          # Factor combination
│   │   └── execution.rs         # Order execution
│   └── utils/
│       ├── mod.rs               # Utilities
│       └── config.rs            # Configuration
├── examples/
│   ├── basic_discovery.rs       # Basic factor discovery
│   ├── backtest_factors.rs      # Backtesting demo
│   └── live_trading.rs          # Live trading demo
├── experiments/
│   └── test_factors.rs          # Factor experiments
└── tests/
    └── integration.rs           # Integration tests
```

### Key Design Principles

1. **Modular LLM Support**: Easy switching between OpenAI, Anthropic, local models
2. **Expression Validation**: Strict parsing prevents invalid factors
3. **Efficient Backtesting**: Vectorized operations for fast evaluation
4. **Caching**: Store successful factors to avoid regeneration
5. **Incremental Updates**: Real-time factor computation for live trading

## Risk Management

### Factor-Specific Risks

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Factor Discovery Risk Management                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Overfitting Risk                                                    │
│     ├── LLM may generate factors that fit historical data too well    │
│     ├── Complex expressions are more prone to overfitting              │
│     └── Mitigation:                                                    │
│         → Out-of-sample testing (walk-forward validation)              │
│         → Complexity penalties (prefer simpler expressions)            │
│         → Multiple testing correction (Bonferroni, BH)                 │
│                                                                         │
│  2. Data Snooping Risk                                                  │
│     ├── Testing many factors increases false discovery rate            │
│     ├── LLM training data may include known factors                    │
│     └── Mitigation:                                                    │
│         → Track number of factors tested                               │
│         → Apply statistical corrections                                │
│         → Use holdout periods never seen during development            │
│                                                                         │
│  3. Regime Change Risk                                                  │
│     ├── Factors may stop working when market regime changes            │
│     ├── Crypto markets have frequent regime shifts                     │
│     └── Mitigation:                                                    │
│         → Monitor factor decay (rolling IC)                            │
│         → Diversify across factor types                                │
│         → Implement circuit breakers                                   │
│                                                                         │
│  4. LLM Hallucination Risk                                              │
│     ├── LLM may generate nonsensical expressions                       │
│     ├── May claim factors work without evidence                        │
│     └── Mitigation:                                                    │
│         → Strict expression validation                                 │
│         → Always verify with actual backtests                          │
│         → Require economic rationale                                   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Position Sizing

```
Position Sizing for Factor-Based Strategies:

1. Factor Confidence Scaling:
   position_size = base_size × abs(factor_signal) × confidence_score

2. IC-Based Scaling:
   position_size = base_size × (rolling_IC / target_IC)
   - Increase position when factor is working
   - Decrease when IC drops

3. Volatility Adjustment:
   position_size = target_vol / realized_vol × base_size

4. Maximum Constraints:
   - Max single position: 10% of portfolio (crypto)
   - Max factor exposure: 30% per factor type
   - Max total leverage: 2x (conservative)
```

## Performance Metrics

### Factor Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| IC (Information Coefficient) | Rank correlation with returns | > 0.02 |
| IC_IR (IC Information Ratio) | IC / std(IC) | > 0.3 |
| Sharpe Ratio | Risk-adjusted return | > 1.5 |
| Sortino Ratio | Downside risk-adjusted return | > 2.0 |
| Maximum Drawdown | Largest peak-to-trough decline | < 20% |
| Turnover | Position change rate | < 50%/month |
| T-statistic | Statistical significance of IC | > 2.0 |
| Decay Time | Half-life of factor predictiveness | > 5 days |

### Discovery System Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Hit Rate | Factors passing validation | > 30% |
| Novel Factor Rate | Unique factors (low correlation) | > 20% |
| Generation Latency | Time to generate factor batch | < 60s |
| Backtest Throughput | Factors backtested per hour | > 100 |
| Refinement Improvement | IC gain after iteration | > 20% |

## References

1. **Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment**
   - URL: https://arxiv.org/abs/2308.00016
   - Year: 2023

2. **Alpha-GPT 2.0: Human-in-the-Loop AI for Quantitative Investment**
   - URL: https://arxiv.org/abs/2402.09746
   - Year: 2024

3. **Chain-of-Alpha: Unleashing the Power of Large Language Models for Alpha Mining**
   - URL: https://arxiv.org/abs/2508.06312
   - Year: 2025

4. **From Deep Learning to LLMs: A Survey of AI in Quantitative Investment**
   - URL: https://arxiv.org/abs/2503.21422
   - Year: 2025

5. **FinGPT: Open-Source Financial Large Language Models**
   - Yang, H., et al. (2023)
   - URL: https://arxiv.org/abs/2306.06031

6. **A Hybrid Approach to Formulaic Alpha Discovery with Large Language Model Assistance**
   - URL: https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-41061-5
   - Year: 2025

7. **Large Language Models in Equity Markets: Applications, Techniques, and Insights**
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12421730/
   - Year: 2025

8. **Machine Learning for Factor Investing**
   - Coqueret, G., & Guida, T. (2020). Chapman & Hall/CRC

9. **Quantitative Equity Portfolio Management**
   - Chincarini, L., & Kim, D. (2006). McGraw-Hill

---

## Next Steps

- [View Simple Explanation](readme.simple.md) - Beginner-friendly version
- [Russian Version](README.ru.md) - Русская версия
- [Run Examples](examples/) - Working Rust code

---

*Chapter 75 of Machine Learning for Trading*
