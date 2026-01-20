# Chapter 79: LLM Backtesting Assistant

## Overview

LLM Backtesting Assistant represents a new paradigm in quantitative trading analysis where Large Language Models (LLMs) serve as intelligent assistants for analyzing backtesting results, identifying strategy weaknesses, and suggesting improvements. Instead of manually interpreting complex performance metrics, traders can leverage LLMs to provide natural language insights, explanations, and actionable recommendations.

This chapter explores how to build an LLM-powered backtesting assistant that can analyze trading strategy performance across both traditional equity markets and cryptocurrency trading on platforms like Bybit.

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Backtesting Fundamentals](#backtesting-fundamentals)
4. [LLM-Assisted Analysis](#llm-assisted-analysis)
5. [System Architecture](#system-architecture)
6. [Performance Metrics Analysis](#performance-metrics-analysis)
7. [Strategy Improvement Pipeline](#strategy-improvement-pipeline)
8. [Application to Cryptocurrency Trading](#application-to-cryptocurrency-trading)
9. [Implementation Strategy](#implementation-strategy)
10. [Risk Assessment](#risk-assessment)
11. [Code Examples](#code-examples)
12. [References](#references)

---

## Introduction

### What is an LLM Backtesting Assistant?

An LLM Backtesting Assistant is an AI-powered tool that analyzes trading strategy backtest results and provides human-readable insights:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              LLM Backtesting Assistant Overview                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional Analysis:             LLM-Assisted Analysis:              │
│   ┌──────────────────┐              ┌──────────────────┐                │
│   │ Raw Metrics      │              │ Raw Metrics      │                │
│   │      ↓           │              │      ↓           │                │
│   │ Manual Review    │              │ LLM Processing   │                │
│   │      ↓           │              │      ↓           │                │
│   │ Expert           │              │ Natural Language │                │
│   │ Interpretation   │              │ Insights         │                │
│   │      ↓           │              │      ↓           │                │
│   │ Slow Iteration   │              │ Actionable       │                │
│   │ (days/weeks)     │              │ Recommendations  │                │
│   └──────────────────┘              │ (minutes)        │                │
│                                     └──────────────────┘                │
│                                                                          │
│   Key Capabilities:                                                      │
│   • Interpret complex performance metrics                               │
│   • Identify strategy weaknesses and failure modes                      │
│   • Suggest parameter optimizations                                     │
│   • Generate natural language reports                                   │
│   • Compare multiple strategy variants                                  │
│   • Detect regime-specific performance issues                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Use LLMs for Backtesting Analysis?

| Aspect | Traditional Analysis | LLM-Assisted Analysis |
|--------|----------------------|----------------------|
| Metric interpretation | Manual, requires expertise | Automated explanations |
| Report generation | Time-consuming | Instant natural language |
| Pattern detection | Limited by human capacity | Comprehensive analysis |
| Actionable insights | Subjective | Structured recommendations |
| Scalability | One strategy at a time | Multiple strategies in parallel |
| Learning curve | Years of experience needed | Accessible to beginners |
| Consistency | Variable quality | Consistent framework |

## Theoretical Foundation

### Backtesting Framework

Backtesting is the process of evaluating a trading strategy using historical data:

$$P\&L_{strategy} = \sum_{t=1}^{T} signal_t \cdot returns_{t+1}$$

Where:
- $signal_t$ is the trading signal at time $t$ (position size and direction)
- $returns_{t+1}$ is the subsequent return
- $T$ is the total number of periods

### Key Performance Metrics

The LLM assistant analyzes multiple metrics to form a comprehensive view:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Core Performance Metrics                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Risk-Adjusted Returns                                               │
│     ┌──────────────────────────────────────────────────────────────┐   │
│     │ Sharpe Ratio = (R_p - R_f) / σ_p                             │   │
│     │ Sortino Ratio = (R_p - R_f) / σ_downside                     │   │
│     │ Calmar Ratio = Annual Return / Max Drawdown                   │   │
│     │ Target: Sharpe > 1.0, Sortino > 1.5                          │   │
│     └──────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  2. Drawdown Analysis                                                   │
│     ┌──────────────────────────────────────────────────────────────┐   │
│     │ Maximum Drawdown = max(peak_value - trough_value) / peak     │   │
│     │ Average Drawdown = mean of all drawdown periods              │   │
│     │ Drawdown Duration = time to recover from drawdown            │   │
│     │ Target: Max DD < 20%, Avg Recovery < 30 days                 │   │
│     └──────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  3. Trade Statistics                                                    │
│     ┌──────────────────────────────────────────────────────────────┐   │
│     │ Win Rate = Winning Trades / Total Trades                     │   │
│     │ Profit Factor = Gross Profits / Gross Losses                 │   │
│     │ Average Win/Loss Ratio = Avg Win Size / Avg Loss Size        │   │
│     │ Target: Win Rate > 50%, Profit Factor > 1.5                  │   │
│     └──────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  4. Market Regime Performance                                           │
│     ┌──────────────────────────────────────────────────────────────┐   │
│     │ Bull Market Returns, Bear Market Returns                     │   │
│     │ High Volatility Performance, Low Volatility Performance      │   │
│     │ Correlation with Market Benchmark                            │   │
│     │ Target: Positive returns across regimes, low beta            │   │
│     └──────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### LLM as Analysis Engine

LLMs can be viewed as functions that map metrics to insights:

$$f_{LLM}: \text{Backtest Metrics} \rightarrow \text{Natural Language Analysis}$$

The key capabilities include:
1. **Contextual understanding** of what metrics mean for strategy health
2. **Pattern recognition** across multiple performance dimensions
3. **Comparative analysis** against benchmarks and best practices
4. **Causal reasoning** about why certain patterns occur
5. **Recommendation generation** based on analysis

## Backtesting Fundamentals

### Strategy Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Strategy Development Lifecycle                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │ IDEATION│ →  │ CODING  │ →  │BACKTEST │ →  │ANALYSIS │             │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘             │
│       ↑                                             │                   │
│       │                                             │                   │
│       │         ┌─────────────────────────┐         │                   │
│       │         │   LLM ASSISTANT         │         │                   │
│       └─────────│ • Analyzes results      │←────────┘                   │
│                 │ • Suggests improvements  │                            │
│                 │ • Generates reports      │                            │
│                 └─────────────────────────┘                             │
│                                                                          │
│  Iteration continues until:                                             │
│  • Risk-adjusted returns meet targets                                   │
│  • Strategy passes robustness checks                                    │
│  • Out-of-sample performance validates                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Common Backtesting Pitfalls

The LLM assistant helps identify these issues:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Backtesting Pitfalls Detection                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. LOOK-AHEAD BIAS                                                     │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ Using future information that wouldn't be available           │  │
│     │ LLM Check: "Are signals generated using only past data?"      │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  2. SURVIVORSHIP BIAS                                                   │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ Testing only on assets that survived to present               │  │
│     │ LLM Check: "Does the universe include delisted securities?"   │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  3. OVERFITTING                                                         │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ Optimizing too many parameters on limited data                │  │
│     │ LLM Check: "Parameter count vs data points ratio?"            │  │
│     │ LLM Check: "In-sample vs out-of-sample performance gap?"      │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  4. TRANSACTION COSTS                                                   │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ Underestimating real-world trading costs                      │  │
│     │ LLM Check: "Are slippage and commissions realistic?"          │  │
│     │ LLM Check: "What is turnover vs expected alpha?"              │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  5. DATA QUALITY                                                        │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ Using incorrect or adjusted historical data                   │  │
│     │ LLM Check: "Are corporate actions handled correctly?"         │  │
│     │ LLM Check: "Is data frequency appropriate for strategy?"      │  │
│     └───────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## LLM-Assisted Analysis

### Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM Analysis Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: Backtest Results                                                │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ {                                                                │   │
│  │   "sharpe_ratio": 1.45,                                         │   │
│  │   "max_drawdown": -0.18,                                        │   │
│  │   "total_return": 0.42,                                         │   │
│  │   "win_rate": 0.52,                                             │   │
│  │   "profit_factor": 1.67,                                        │   │
│  │   "trades": [...]                                               │   │
│  │ }                                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  STEP 1: CONTEXT BUILDING                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • Extract strategy type and market context                       │   │
│  │ • Identify relevant benchmark comparisons                        │   │
│  │ • Load historical regime data                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  STEP 2: METRIC ANALYSIS                                               │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • LLM interprets each metric in context                          │   │
│  │ • Identifies strengths and weaknesses                            │   │
│  │ • Flags potential concerns                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  STEP 3: PATTERN DETECTION                                             │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • Analyze trade distribution patterns                            │   │
│  │ • Detect regime-specific behavior                                │   │
│  │ • Identify clustering of losses                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  STEP 4: RECOMMENDATION GENERATION                                     │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ • Suggest parameter adjustments                                  │   │
│  │ • Recommend risk management improvements                         │   │
│  │ • Propose strategy modifications                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  OUTPUT: Analysis Report                                               │
│  ─────────────────────────────────────────────────────────────────────  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ "Your momentum strategy shows strong risk-adjusted returns       │   │
│  │  (Sharpe 1.45) but exhibits concentrated losses during          │   │
│  │  high-volatility regimes. Consider:                              │   │
│  │  1. Adding a volatility filter to reduce position size...       │   │
│  │  2. Implementing dynamic stop-losses based on ATR..."           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Prompt Engineering for Backtest Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Analysis Prompt Template                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SYSTEM PROMPT:                                                         │
│  """                                                                    │
│  You are an expert quantitative analyst specializing in trading         │
│  strategy evaluation. Your task is to analyze backtest results and     │
│  provide actionable insights.                                           │
│                                                                          │
│  When analyzing results:                                                │
│  1. Consider the strategy type and its expected behavior                │
│  2. Compare metrics against industry benchmarks                         │
│  3. Identify potential risks and failure modes                          │
│  4. Suggest specific, implementable improvements                        │
│                                                                          │
│  Benchmark reference values:                                            │
│  - Sharpe Ratio: > 1.0 good, > 2.0 excellent                           │
│  - Max Drawdown: < 15% conservative, < 25% moderate                    │
│  - Win Rate: context-dependent (50%+ for freq trading)                 │
│  - Profit Factor: > 1.5 good, > 2.0 excellent                          │
│                                                                          │
│  Be specific about:                                                     │
│  - What the numbers mean for this strategy type                        │
│  - Which aspects need improvement                                       │
│  - Concrete steps to address weaknesses                                 │
│  """                                                                    │
│                                                                          │
│  USER PROMPT:                                                           │
│  """                                                                    │
│  Analyze the following backtest results for a {strategy_type} strategy │
│  trading {asset_class} over {time_period}:                              │
│                                                                          │
│  Performance Metrics:                                                   │
│  {metrics_json}                                                         │
│                                                                          │
│  Trade Statistics:                                                      │
│  {trade_stats_json}                                                     │
│                                                                          │
│  Market Conditions:                                                     │
│  {market_context}                                                       │
│                                                                          │
│  Please provide:                                                        │
│  1. Overall assessment (1-2 sentences)                                  │
│  2. Key strengths (bullet points)                                       │
│  3. Areas of concern (bullet points)                                    │
│  4. Specific recommendations (numbered list)                            │
│  5. Risk assessment and position sizing guidance                        │
│  """                                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## System Architecture

### Core Components

```rust
/// Core components of an LLM Backtesting Assistant
#[derive(Debug, Clone)]
pub struct BacktestingAssistant {
    /// LLM client for analysis generation
    pub llm_client: LlmClient,

    /// Metrics calculator for processing raw results
    pub metrics_calculator: MetricsCalculator,

    /// Report generator for formatted output
    pub report_generator: ReportGenerator,

    /// Configuration
    pub config: AssistantConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantConfig {
    /// Minimum data points for reliable analysis
    pub min_trades: usize,

    /// Benchmark Sharpe ratio for comparison
    pub benchmark_sharpe: f64,

    /// Risk-free rate for calculations
    pub risk_free_rate: f64,

    /// Analysis verbosity level
    pub verbosity: VerbosityLevel,

    /// Include visualizations in reports
    pub include_charts: bool,
}

impl Default for AssistantConfig {
    fn default() -> Self {
        Self {
            min_trades: 30,
            benchmark_sharpe: 1.0,
            risk_free_rate: 0.05,
            verbosity: VerbosityLevel::Detailed,
            include_charts: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerbosityLevel {
    Brief,      // One-paragraph summary
    Standard,   // Key metrics and recommendations
    Detailed,   // Full analysis with explanations
    Expert,     // Technical deep-dive
}
```

### Data Structures

```rust
/// Backtest results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResults {
    /// Strategy identifier
    pub strategy_id: String,

    /// Strategy type description
    pub strategy_type: String,

    /// Asset class traded
    pub asset_class: AssetClass,

    /// Backtest period
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,

    /// Performance metrics
    pub metrics: PerformanceMetrics,

    /// Individual trades
    pub trades: Vec<Trade>,

    /// Equity curve
    pub equity_curve: Vec<EquityPoint>,

    /// Market regime data
    pub regime_performance: Option<RegimePerformance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return over period
    pub total_return: f64,

    /// Annualized return
    pub annual_return: f64,

    /// Sharpe ratio
    pub sharpe_ratio: f64,

    /// Sortino ratio
    pub sortino_ratio: f64,

    /// Calmar ratio
    pub calmar_ratio: f64,

    /// Maximum drawdown
    pub max_drawdown: f64,

    /// Win rate
    pub win_rate: f64,

    /// Profit factor
    pub profit_factor: f64,

    /// Total number of trades
    pub total_trades: usize,

    /// Average trade duration
    pub avg_trade_duration: Duration,

    /// Turnover rate
    pub turnover: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade identifier
    pub id: String,

    /// Entry timestamp
    pub entry_time: DateTime<Utc>,

    /// Exit timestamp
    pub exit_time: DateTime<Utc>,

    /// Asset symbol
    pub symbol: String,

    /// Trade direction
    pub direction: TradeDirection,

    /// Entry price
    pub entry_price: f64,

    /// Exit price
    pub exit_price: f64,

    /// Position size
    pub quantity: f64,

    /// Trade P&L
    pub pnl: f64,

    /// Return percentage
    pub return_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeDirection {
    Long,
    Short,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetClass {
    Equity,
    Cryptocurrency,
    Forex,
    Futures,
    Options,
}
```

### Analysis Report Structure

```rust
/// LLM-generated analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    /// Report identifier
    pub report_id: String,

    /// Generation timestamp
    pub generated_at: DateTime<Utc>,

    /// Strategy being analyzed
    pub strategy_id: String,

    /// Overall assessment
    pub summary: String,

    /// Performance grade (A-F)
    pub grade: PerformanceGrade,

    /// Key strengths identified
    pub strengths: Vec<String>,

    /// Areas of concern
    pub concerns: Vec<String>,

    /// Specific recommendations
    pub recommendations: Vec<Recommendation>,

    /// Risk assessment
    pub risk_assessment: RiskAssessment,

    /// Detailed metric explanations
    pub metric_explanations: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Priority level
    pub priority: Priority,

    /// Category of recommendation
    pub category: RecommendationCategory,

    /// Description
    pub description: String,

    /// Expected impact
    pub expected_impact: String,

    /// Implementation steps
    pub implementation_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    RiskManagement,
    EntrySignal,
    ExitSignal,
    PositionSizing,
    AssetSelection,
    MarketTiming,
    CostReduction,
}
```

## Performance Metrics Analysis

### Metric Interpretation Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Metric Interpretation Guide                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  SHARPE RATIO INTERPRETATION                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Value      │ Assessment  │ LLM Response Example                 │   │
│  │────────────┼─────────────┼──────────────────────────────────────│   │
│  │ < 0.5      │ Poor        │ "Strategy underperforms risk-free   │   │
│  │            │             │  rate on risk-adjusted basis"       │   │
│  │ 0.5 - 1.0  │ Acceptable  │ "Moderate risk-adjusted returns,    │   │
│  │            │             │  room for improvement"               │   │
│  │ 1.0 - 2.0  │ Good        │ "Strong risk-adjusted performance,  │   │
│  │            │             │  suitable for allocation"            │   │
│  │ 2.0 - 3.0  │ Excellent   │ "Exceptional performance, verify    │   │
│  │            │             │  for overfitting"                    │   │
│  │ > 3.0      │ Suspicious  │ "Unusually high - check for         │   │
│  │            │             │  look-ahead bias or data issues"    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  DRAWDOWN ANALYSIS                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Max DD     │ Assessment  │ Suitable For                         │   │
│  │────────────┼─────────────┼──────────────────────────────────────│   │
│  │ < 10%      │ Conservative│ Pension funds, risk-averse capital  │   │
│  │ 10% - 20%  │ Moderate    │ Standard institutional allocation   │   │
│  │ 20% - 30%  │ Aggressive  │ Hedge funds, active traders         │   │
│  │ > 30%      │ High Risk   │ Small allocation, strong conviction │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  WIN RATE + PROFIT FACTOR MATRIX                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              │ PF < 1.2   │ PF 1.2-1.8  │ PF > 1.8             │   │
│  │──────────────┼────────────┼─────────────┼──────────────────────│   │
│  │ WR < 40%     │ Failing    │ Trend-      │ Few big winners      │   │
│  │              │            │ following   │ (valid strategy)     │   │
│  │ WR 40-60%    │ Marginal   │ Balanced    │ Strong edge          │   │
│  │              │            │ strategy    │                      │   │
│  │ WR > 60%     │ Cost       │ Good mean   │ Excellent            │   │
│  │              │ issues     │ reversion   │ (verify scalping)    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Regime-Based Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Market Regime Analysis                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  REGIME DETECTION                                                       │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Bull Market:   20-day return > 0 AND volatility < historical median   │
│  Bear Market:   20-day return < 0 AND volatility > historical median   │
│  High Vol:      20-day volatility > 75th percentile                    │
│  Low Vol:       20-day volatility < 25th percentile                    │
│  Trending:      ADX > 25                                                │
│  Ranging:       ADX < 20                                                │
│                                                                          │
│  PERFORMANCE BREAKDOWN                                                  │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Regime          │ % Time │ Strategy    │ Assessment            │   │
│  │                 │        │ Returns     │                        │   │
│  │─────────────────┼────────┼─────────────┼────────────────────────│   │
│  │ Bull + Low Vol  │  35%   │ +18%        │ Capturing uptrend ✓   │   │
│  │ Bull + High Vol │  15%   │ +5%         │ Whipsawed in rallies  │   │
│  │ Bear + Low Vol  │  20%   │ -2%         │ Acceptable defense    │   │
│  │ Bear + High Vol │  30%   │ -15%        │ PROBLEM AREA ⚠        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  LLM INSIGHT:                                                           │
│  "Your strategy performs well in calm markets but loses significantly  │
│   during volatile bear markets. This 30% of time accounts for 85% of  │
│   your total losses. Consider implementing:                             │
│   1. VIX-based position scaling                                         │
│   2. Tighter stops during high-vol regimes                              │
│   3. Reduced leverage when ADX < 20 in downtrends"                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Strategy Improvement Pipeline

### Iterative Refinement Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM-Guided Strategy Improvement                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ITERATION 1: Initial Analysis                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Backtest v1.0 → LLM Analysis → Recommendations                  │   │
│  │                                                                   │   │
│  │ Findings:                                                         │   │
│  │ • Sharpe 0.8 (below target of 1.0)                               │   │
│  │ • Max DD 28% (too high for moderate risk)                        │   │
│  │ • Win rate 45% with PF 1.3 (marginal)                           │   │
│  │                                                                   │   │
│  │ LLM Recommendation:                                               │   │
│  │ "Primary issue: No position sizing based on volatility.          │   │
│  │  Implement ATR-based position sizing to reduce drawdowns."       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  ITERATION 2: Position Sizing Added                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Backtest v1.1 → LLM Analysis → Recommendations                  │   │
│  │                                                                   │   │
│  │ Improvements:                                                     │   │
│  │ • Sharpe 1.1 (↑37%)                                              │   │
│  │ • Max DD 19% (↓32%)                                              │   │
│  │ • Win rate 47% with PF 1.4                                       │   │
│  │                                                                   │   │
│  │ LLM Recommendation:                                               │   │
│  │ "Good progress. Next: Entry timing shows clustering of           │   │
│  │  losses on Mondays. Consider avoiding positions over weekends   │   │
│  │  or reducing Monday exposure."                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ↓                                          │
│  ITERATION 3: Temporal Filter Added                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Backtest v1.2 → LLM Analysis → Recommendations                  │   │
│  │                                                                   │   │
│  │ Improvements:                                                     │   │
│  │ • Sharpe 1.35 (↑23%)                                             │   │
│  │ • Max DD 15% (↓21%)                                              │   │
│  │ • Win rate 52% with PF 1.6                                       │   │
│  │                                                                   │   │
│  │ LLM Assessment:                                                   │   │
│  │ "Strategy now meets institutional quality standards. Ready       │   │
│  │  for out-of-sample validation. Final suggestion: Add            │   │
│  │  correlation monitor to avoid crowded trades."                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Common Improvement Recommendations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM Recommendation Categories                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  RISK MANAGEMENT                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Issue: Large drawdowns                                           │   │
│  │ Recommendations:                                                  │   │
│  │ • Implement ATR-based stop losses                                │   │
│  │ • Add portfolio-level VaR limits                                  │   │
│  │ • Use volatility-scaled position sizing                          │   │
│  │ • Consider correlation-aware position limits                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ENTRY OPTIMIZATION                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Issue: Low win rate                                              │   │
│  │ Recommendations:                                                  │   │
│  │ • Add confirmation filters (volume, momentum alignment)          │   │
│  │ • Implement regime detection for conditional entry               │   │
│  │ • Test alternative entry timing (open vs close)                  │   │
│  │ • Consider scaling into positions                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  EXIT OPTIMIZATION                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Issue: Giving back profits                                       │   │
│  │ Recommendations:                                                  │   │
│  │ • Implement trailing stops                                       │   │
│  │ • Add profit targets based on ATR multiples                      │   │
│  │ • Test time-based exits for mean reversion                       │   │
│  │ • Consider partial profit taking                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  COST REDUCTION                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Issue: High turnover eating into profits                         │   │
│  │ Recommendations:                                                  │   │
│  │ • Add hysteresis to signals (entry != exit threshold)           │   │
│  │ • Increase rebalancing intervals                                 │   │
│  │ • Use limit orders instead of market orders                      │   │
│  │ • Filter out low-conviction signals                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Application to Cryptocurrency Trading

### Crypto-Specific Considerations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Cryptocurrency Backtesting                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  UNIQUE CHARACTERISTICS                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  1. 24/7 Trading                                                        │
│     • No market close effects                                           │
│     • Weekend trading patterns differ                                   │
│     • Consider time-of-day effects                                      │
│                                                                          │
│  2. Higher Volatility                                                   │
│     • Typical daily vol: 3-8% (vs 1% equities)                         │
│     • Drawdowns can exceed 50% quickly                                  │
│     • Risk metrics need recalibration                                   │
│                                                                          │
│  3. Exchange-Specific Issues                                            │
│     • Different fee structures (maker/taker)                            │
│     • Funding rates for perpetuals                                      │
│     • Liquidation mechanics                                             │
│                                                                          │
│  4. Market Structure                                                    │
│     • Higher correlation across assets                                  │
│     • Dominance of BTC influences altcoins                              │
│     • Faster regime changes                                             │
│                                                                          │
│  LLM ADAPTATION:                                                        │
│  "For your crypto momentum strategy on Bybit:                           │
│                                                                          │
│  ⚠ Adjusted Benchmarks:                                                 │
│  • Sharpe > 1.5 (vs 1.0 for equities) due to higher vol                │
│  • Max DD < 40% acceptable for crypto                                   │
│  • Win rate less important given larger moves                           │
│                                                                          │
│  ⚠ Specific Concerns:                                                   │
│  • Your 2x leverage + 30% DD could trigger liquidation                  │
│  • Funding rates are eating 0.8% monthly - factor this in              │
│  • Slippage assumption of 0.1% may be optimistic for altcoins"         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Bybit Integration

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Bybit Market Data for Backtesting                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DATA SOURCES                                                           │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Spot Markets:                                                          │
│  • BTCUSDT, ETHUSDT, and 100+ trading pairs                            │
│  • 1m, 5m, 15m, 1h, 4h, 1d candles                                     │
│  • Order book snapshots                                                  │
│                                                                          │
│  Perpetual Futures:                                                     │
│  • Linear perpetuals (USDT-margined)                                    │
│  • Funding rate history                                                  │
│  • Open interest data                                                    │
│                                                                          │
│  API ENDPOINTS                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  Historical Klines:                                                     │
│  GET /v5/market/kline                                                   │
│  Parameters: category, symbol, interval, start, end, limit             │
│                                                                          │
│  Funding Rate History:                                                   │
│  GET /v5/market/funding/history                                          │
│  Parameters: category, symbol, startTime, endTime, limit               │
│                                                                          │
│  Ticker Info:                                                           │
│  GET /v5/market/tickers                                                 │
│  Parameters: category, symbol                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation Strategy

### Python Implementation

```python
"""
LLM Backtesting Assistant - Python Implementation
Analyzes trading strategy backtest results using LLMs
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import requests


class AssetClass(Enum):
    """Asset class types"""
    EQUITY = "equity"
    CRYPTOCURRENCY = "cryptocurrency"
    FOREX = "forex"
    FUTURES = "futures"


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Individual trade record"""
    id: str
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float


@dataclass
class PerformanceMetrics:
    """Strategy performance metrics"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration_hours: float
    turnover: float


@dataclass
class BacktestResults:
    """Complete backtest results"""
    strategy_id: str
    strategy_type: str
    asset_class: AssetClass
    start_date: datetime
    end_date: datetime
    metrics: PerformanceMetrics
    trades: List[Trade]
    equity_curve: pd.Series
    regime_performance: Optional[Dict[str, float]] = None


@dataclass
class Recommendation:
    """Strategy improvement recommendation"""
    priority: str  # critical, high, medium, low
    category: str
    description: str
    expected_impact: str
    implementation_steps: List[str]


@dataclass
class AnalysisReport:
    """LLM-generated analysis report"""
    report_id: str
    generated_at: datetime
    strategy_id: str
    summary: str
    grade: str  # A, B, C, D, F
    strengths: List[str]
    concerns: List[str]
    recommendations: List[Recommendation]
    metric_explanations: Dict[str, str]


class MetricsCalculator:
    """Calculate performance metrics from trade data"""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        periods_per_year: int = 252
    ) -> PerformanceMetrics:
        """Calculate all performance metrics"""
        if len(trades) == 0:
            return self._empty_metrics()

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Total and annual return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        n_years = len(equity_curve) / periods_per_year
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / periods_per_year
        sharpe_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std() \
            if returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001
        sortino_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / downside_std \
            if downside_std > 0 else 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        avg_duration = np.mean(durations) if durations else 0

        # Turnover (simplified - based on number of trades)
        turnover = len(trades) / max(1, (equity_curve.index[-1] - equity_curve.index[0]).days / 365)

        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_duration_hours=avg_duration,
            turnover=turnover
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for no trades"""
        return PerformanceMetrics(
            total_return=0, annual_return=0, sharpe_ratio=0,
            sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
            win_rate=0, profit_factor=0, total_trades=0,
            avg_trade_duration_hours=0, turnover=0
        )


class LlmClient:
    """Client for LLM API calls"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def analyze(self, system_prompt: str, user_prompt: str) -> str:
        """Send analysis request to LLM"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }

        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]


class BacktestingAssistant:
    """LLM-powered backtesting analysis assistant"""

    SYSTEM_PROMPT = """You are an expert quantitative analyst specializing in trading
strategy evaluation. Your task is to analyze backtest results and provide actionable insights.

When analyzing results:
1. Consider the strategy type and its expected behavior
2. Compare metrics against industry benchmarks
3. Identify potential risks and failure modes
4. Suggest specific, implementable improvements

Benchmark reference values:
- Sharpe Ratio: > 1.0 good, > 2.0 excellent
- Max Drawdown: < 15% conservative, < 25% moderate
- Win Rate: context-dependent (50%+ for frequency trading)
- Profit Factor: > 1.5 good, > 2.0 excellent

For cryptocurrency strategies, adjust expectations:
- Sharpe > 1.5 is good (due to higher volatility)
- Max Drawdown < 40% is acceptable
- Consider funding rates and liquidation risks

Provide your analysis in a structured format with:
1. Overall assessment (2-3 sentences)
2. Performance grade (A/B/C/D/F)
3. Key strengths (bullet points)
4. Areas of concern (bullet points)
5. Specific recommendations with priority levels
6. Brief explanation of key metrics"""

    def __init__(self, llm_client: LlmClient, config: Optional[Dict] = None):
        self.llm_client = llm_client
        self.metrics_calculator = MetricsCalculator()
        self.config = config or {
            "min_trades": 30,
            "benchmark_sharpe": 1.0
        }

    def analyze(self, results: BacktestResults) -> AnalysisReport:
        """Analyze backtest results and generate report"""
        # Prepare the user prompt with metrics
        user_prompt = self._build_user_prompt(results)

        # Get LLM analysis
        llm_response = self.llm_client.analyze(self.SYSTEM_PROMPT, user_prompt)

        # Parse response into structured report
        return self._parse_response(results, llm_response)

    def _build_user_prompt(self, results: BacktestResults) -> str:
        """Build the user prompt with backtest data"""
        metrics = results.metrics

        prompt = f"""Analyze the following backtest results for a {results.strategy_type} strategy
trading {results.asset_class.value} from {results.start_date.date()} to {results.end_date.date()}:

PERFORMANCE METRICS:
- Total Return: {metrics.total_return:.2%}
- Annual Return: {metrics.annual_return:.2%}
- Sharpe Ratio: {metrics.sharpe_ratio:.2f}
- Sortino Ratio: {metrics.sortino_ratio:.2f}
- Calmar Ratio: {metrics.calmar_ratio:.2f}
- Maximum Drawdown: {metrics.max_drawdown:.2%}
- Win Rate: {metrics.win_rate:.2%}
- Profit Factor: {metrics.profit_factor:.2f}
- Total Trades: {metrics.total_trades}
- Average Trade Duration: {metrics.avg_trade_duration_hours:.1f} hours
- Annual Turnover: {metrics.turnover:.1f}x

"""

        if results.regime_performance:
            prompt += "REGIME PERFORMANCE:\n"
            for regime, perf in results.regime_performance.items():
                prompt += f"- {regime}: {perf:.2%}\n"
            prompt += "\n"

        prompt += """Please provide your analysis with:
1. Overall assessment
2. Performance grade (A/B/C/D/F)
3. Key strengths
4. Areas of concern
5. Specific recommendations with priority (critical/high/medium/low)
6. Brief metric explanations"""

        return prompt

    def _parse_response(self, results: BacktestResults, llm_response: str) -> AnalysisReport:
        """Parse LLM response into structured report"""
        # Simple parsing - in production, use structured output
        lines = llm_response.split('\n')

        # Extract grade (look for A, B, C, D, or F pattern)
        grade = "C"  # default
        for line in lines:
            if "grade" in line.lower():
                for g in ["A", "B", "C", "D", "F"]:
                    if g in line:
                        grade = g
                        break

        # Create basic report structure
        return AnalysisReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            strategy_id=results.strategy_id,
            summary=llm_response[:500] + "..." if len(llm_response) > 500 else llm_response,
            grade=grade,
            strengths=self._extract_section(llm_response, "strength"),
            concerns=self._extract_section(llm_response, "concern"),
            recommendations=self._extract_recommendations(llm_response),
            metric_explanations=self._extract_explanations(results.metrics, llm_response)
        )

    def _extract_section(self, text: str, keyword: str) -> List[str]:
        """Extract bullet points from a section"""
        items = []
        in_section = False
        for line in text.split('\n'):
            if keyword in line.lower():
                in_section = True
                continue
            if in_section and line.strip().startswith(('-', '*', '•')):
                items.append(line.strip().lstrip('-*• '))
            elif in_section and line.strip() and not line.strip().startswith(('-', '*', '•')):
                if any(x in line.lower() for x in ['recommendation', 'concern', 'strength', 'metric']):
                    in_section = False
        return items[:5]  # Limit to 5 items

    def _extract_recommendations(self, text: str) -> List[Recommendation]:
        """Extract recommendations from response"""
        recommendations = []
        items = self._extract_section(text, "recommendation")

        for item in items:
            priority = "medium"
            for p in ["critical", "high", "medium", "low"]:
                if p in item.lower():
                    priority = p
                    break

            recommendations.append(Recommendation(
                priority=priority,
                category="general",
                description=item,
                expected_impact="Improvement in risk-adjusted returns",
                implementation_steps=["Implement the suggested change", "Backtest the modification", "Compare results"]
            ))

        return recommendations

    def _extract_explanations(self, metrics: PerformanceMetrics, text: str) -> Dict[str, str]:
        """Extract metric explanations"""
        explanations = {}

        metric_names = ["sharpe", "sortino", "drawdown", "win rate", "profit factor"]
        for name in metric_names:
            for line in text.split('\n'):
                if name in line.lower():
                    explanations[name] = line.strip()
                    break

        return explanations


class BybitDataFetcher:
    """Fetch historical data from Bybit for backtesting"""

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        category: str = "spot"
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Bybit"""
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        all_data = []
        current_start = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)

        while current_start < end_ts:
            params = {
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "start": current_start,
                "end": end_ts,
                "limit": 1000
            }

            response = self.session.get(endpoint, params=params)
            data = response.json()

            if data["retCode"] != 0:
                raise Exception(f"Bybit API error: {data['retMsg']}")

            klines = data["result"]["list"]
            if not klines:
                break

            all_data.extend(klines)

            # Move to next batch
            last_ts = int(klines[-1][0])
            if last_ts >= current_start:
                current_start = last_ts + 1
            else:
                break

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.set_index("timestamp").sort_index()
        df = df.astype(float)

        return df

    def get_funding_history(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Fetch funding rate history for perpetuals"""
        endpoint = f"{self.BASE_URL}/v5/market/funding/history"

        params = {
            "category": "linear",
            "symbol": symbol,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": 200
        }

        response = self.session.get(endpoint, params=params)
        data = response.json()

        if data["retCode"] != 0:
            raise Exception(f"Bybit API error: {data['retMsg']}")

        records = data["result"]["list"]
        df = pd.DataFrame(records)
        df["fundingRateTimestamp"] = pd.to_datetime(
            df["fundingRateTimestamp"].astype(int), unit="ms"
        )
        df = df.set_index("fundingRateTimestamp").sort_index()

        return df


def example_usage():
    """Example usage of the backtesting assistant"""

    # Create sample backtest results
    np.random.seed(42)

    # Generate sample trades
    trades = []
    for i in range(100):
        entry_time = datetime(2024, 1, 1) + timedelta(days=i*2)
        exit_time = entry_time + timedelta(hours=np.random.randint(4, 72))
        pnl = np.random.normal(50, 200)

        trades.append(Trade(
            id=f"trade_{i}",
            entry_time=entry_time,
            exit_time=exit_time,
            symbol="BTCUSDT",
            direction=TradeDirection.LONG if np.random.random() > 0.5 else TradeDirection.SHORT,
            entry_price=50000 + np.random.normal(0, 1000),
            exit_price=50000 + np.random.normal(0, 1000) + pnl,
            quantity=0.1,
            pnl=pnl,
            return_pct=pnl / 5000
        ))

    # Generate equity curve
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    equity = 100000 * (1 + pd.Series(np.random.normal(0.0003, 0.02, len(dates))).cumsum())
    equity_curve = pd.Series(equity.values, index=dates)

    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(trades, equity_curve)

    # Create results object
    results = BacktestResults(
        strategy_id="momentum_btc_v1",
        strategy_type="momentum following",
        asset_class=AssetClass.CRYPTOCURRENCY,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve,
        regime_performance={
            "bull_market": 0.25,
            "bear_market": -0.08,
            "high_volatility": 0.15,
            "low_volatility": 0.10
        }
    )

    print("=" * 60)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Strategy: {results.strategy_id}")
    print(f"Period: {results.start_date.date()} to {results.end_date.date()}")
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Total Trades: {metrics.total_trades}")

    print("\n" + "=" * 60)
    print("To use with LLM analysis, initialize BacktestingAssistant")
    print("with your API key and call analyze(results)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = example_usage()
```

### Rust Implementation

The Rust implementation provides a high-performance backtesting assistant suitable for production environments.

**See the `src/` directory for the full Rust implementation including:**
- `src/lib.rs` - Main library exports
- `src/analysis/` - LLM analysis components
- `src/backtesting/` - Backtesting engine
- `src/data/` - Data fetching (including Bybit integration)
- `src/metrics/` - Performance metrics calculation
- `src/reports/` - Report generation

## Risk Assessment

### Strategy Risk Checklist

The LLM assistant evaluates strategies against these risk factors:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Risk Assessment Framework                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MARKET RISK                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ☐ Strategy tested across multiple market regimes               │   │
│  │ ☐ Correlation with market benchmark understood                  │   │
│  │ ☐ Tail risk (extreme loss) scenarios analyzed                  │   │
│  │ ☐ Liquidity assumptions validated                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  MODEL RISK                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ☐ Out-of-sample validation performed                            │   │
│  │ ☐ Parameter sensitivity analyzed                                │   │
│  │ ☐ Overfitting indicators checked                                │   │
│  │ ☐ Data snooping bias mitigated                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  OPERATIONAL RISK                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ☐ Execution assumptions realistic                               │   │
│  │ ☐ System downtime impact assessed                               │   │
│  │ ☐ Data feed reliability considered                              │   │
│  │ ☐ Position sizing within exchange limits                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  CRYPTO-SPECIFIC RISK                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ ☐ Liquidation scenarios modeled                                 │   │
│  │ ☐ Funding rate costs included                                   │   │
│  │ ☐ Exchange counterparty risk acknowledged                       │   │
│  │ ☐ Network congestion effects considered                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Position Sizing Guidance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Position Sizing Recommendations                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Based on strategy characteristics, the LLM recommends:                 │
│                                                                          │
│  SHARPE 0.5-1.0 (Acceptable):                                           │
│  └─→ Max position: 5-10% of portfolio per trade                        │
│  └─→ Max portfolio risk: 15-20%                                         │
│  └─→ Suggested leverage: 1x                                             │
│                                                                          │
│  SHARPE 1.0-2.0 (Good):                                                 │
│  └─→ Max position: 10-15% of portfolio per trade                       │
│  └─→ Max portfolio risk: 25-30%                                         │
│  └─→ Suggested leverage: 1-1.5x                                         │
│                                                                          │
│  SHARPE 2.0+ (Excellent):                                               │
│  └─→ Max position: 15-20% of portfolio per trade                       │
│  └─→ Max portfolio risk: 30-40%                                         │
│  └─→ Suggested leverage: 1.5-2x                                         │
│                                                                          │
│  ⚠ Always verify with out-of-sample testing before scaling up          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Code Examples

### Complete Analysis Workflow

```python
"""
Complete workflow example: Fetch data, backtest, and analyze
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def run_complete_analysis():
    """Run a complete backtesting analysis workflow"""

    # 1. Fetch historical data
    print("Step 1: Fetching data from Bybit...")
    fetcher = BybitDataFetcher()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Fetch BTCUSDT spot data
    btc_data = fetcher.get_klines(
        symbol="BTCUSDT",
        interval="60",  # 1 hour
        start_time=start_date,
        end_time=end_date,
        category="spot"
    )
    print(f"  Loaded {len(btc_data)} candles")

    # 2. Implement simple momentum strategy
    print("\nStep 2: Running backtest...")
    trades, equity_curve = run_momentum_backtest(btc_data)
    print(f"  Generated {len(trades)} trades")

    # 3. Calculate metrics
    print("\nStep 3: Calculating metrics...")
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(trades, equity_curve, periods_per_year=8760)

    # 4. Create results object
    results = BacktestResults(
        strategy_id="btc_momentum_hourly",
        strategy_type="momentum following with moving average crossover",
        asset_class=AssetClass.CRYPTOCURRENCY,
        start_date=start_date,
        end_date=end_date,
        metrics=metrics,
        trades=trades,
        equity_curve=equity_curve
    )

    # 5. Analyze with LLM (if API key available)
    print("\nStep 4: Generating analysis report...")
    print_metrics_summary(metrics)

    return results


def run_momentum_backtest(data: pd.DataFrame) -> tuple:
    """Run a simple momentum strategy backtest"""
    # Calculate moving averages
    data["sma_fast"] = data["close"].rolling(window=20).mean()
    data["sma_slow"] = data["close"].rolling(window=50).mean()

    # Generate signals
    data["signal"] = 0
    data.loc[data["sma_fast"] > data["sma_slow"], "signal"] = 1
    data.loc[data["sma_fast"] < data["sma_slow"], "signal"] = -1

    # Simulate trades
    trades = []
    position = 0
    entry_price = 0
    entry_time = None
    initial_capital = 100000

    equity = [initial_capital]
    capital = initial_capital

    for i, (timestamp, row) in enumerate(data.iterrows()):
        if pd.isna(row["signal"]):
            equity.append(capital)
            continue

        current_signal = row["signal"]

        # Entry
        if position == 0 and current_signal != 0:
            position = current_signal
            entry_price = row["close"]
            entry_time = timestamp

        # Exit on signal change
        elif position != 0 and current_signal != position:
            exit_price = row["close"]
            pnl = position * (exit_price - entry_price) / entry_price * capital * 0.1
            capital += pnl

            trades.append(Trade(
                id=f"trade_{len(trades)}",
                entry_time=entry_time,
                exit_time=timestamp,
                symbol="BTCUSDT",
                direction=TradeDirection.LONG if position == 1 else TradeDirection.SHORT,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=capital * 0.1 / entry_price,
                pnl=pnl,
                return_pct=pnl / (capital * 0.1)
            ))

            position = current_signal
            entry_price = row["close"] if current_signal != 0 else 0
            entry_time = timestamp if current_signal != 0 else None

        equity.append(capital)

    # Create equity curve
    equity_curve = pd.Series(equity[:len(data)], index=data.index[:len(equity)])

    return trades, equity_curve


def print_metrics_summary(metrics: PerformanceMetrics):
    """Print a formatted metrics summary"""
    print("\n" + "=" * 50)
    print("BACKTEST METRICS SUMMARY")
    print("=" * 50)
    print(f"Total Return:       {metrics.total_return:>10.2%}")
    print(f"Annual Return:      {metrics.annual_return:>10.2%}")
    print(f"Sharpe Ratio:       {metrics.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:      {metrics.sortino_ratio:>10.2f}")
    print(f"Calmar Ratio:       {metrics.calmar_ratio:>10.2f}")
    print(f"Max Drawdown:       {metrics.max_drawdown:>10.2%}")
    print(f"Win Rate:           {metrics.win_rate:>10.2%}")
    print(f"Profit Factor:      {metrics.profit_factor:>10.2f}")
    print(f"Total Trades:       {metrics.total_trades:>10}")
    print("=" * 50)


# Stock market example using yfinance
def fetch_stock_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch stock data using yfinance"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data.columns = [c.lower() for c in data.columns]
        return data
    except ImportError:
        print("yfinance not installed. Install with: pip install yfinance")
        return pd.DataFrame()
```

## References

### Academic Papers

1. **QuantGPT: Code Generation with LLMs for Quantitative Finance**
   - URL: https://arxiv.org/abs/2311.04862
   - Year: 2023
   - Key insight: LLMs can generate valid trading code from natural language descriptions

2. **Alpha-GPT: Human-AI Interactive Alpha Mining**
   - URL: https://arxiv.org/abs/2308.00016
   - Year: 2023
   - Key insight: Combining human intuition with LLM generation improves factor quality

3. **Chain-of-Alpha: Dual-Chain Framework for Factor Mining**
   - URL: https://arxiv.org/abs/2401.00246
   - Year: 2024
   - Key insight: Iterative refinement with feedback loops enhances factor discovery

### Libraries and Tools

- **Backtrader**: Python backtesting framework - https://www.backtrader.com/
- **Zipline**: Event-driven backtesting - https://zipline.ml4trading.io/
- **VectorBT**: High-performance vectorized backtesting - https://vectorbt.dev/
- **QuantConnect**: Cloud-based algorithmic trading - https://www.quantconnect.com/

### Data Sources

- **Bybit API**: Cryptocurrency market data - https://bybit-exchange.github.io/docs/
- **Yahoo Finance**: Stock market data via yfinance
- **Alpha Vantage**: Free financial data API
- **Polygon.io**: Real-time and historical market data

---

## Summary

LLM Backtesting Assistants represent a significant advancement in quantitative trading analysis. By leveraging the natural language understanding and generation capabilities of large language models, traders can:

1. **Accelerate analysis**: Get instant insights instead of spending hours reviewing metrics
2. **Improve accessibility**: Make quantitative analysis accessible to non-experts
3. **Enhance consistency**: Apply systematic analysis frameworks across all strategies
4. **Enable iteration**: Rapidly test and refine strategies based on AI recommendations

The combination of traditional backtesting rigor with LLM-powered interpretation creates a powerful feedback loop for strategy development and improvement.

---

*This chapter is part of the Machine Learning for Trading series. For questions or contributions, please open an issue on GitHub.*
