# Chapter 74: LLM Portfolio Construction

## Overview

Large Language Models (LLMs) can revolutionize portfolio construction by analyzing diverse data sources—news, earnings reports, market commentary, and fundamental data—to generate intelligent asset allocation recommendations. This chapter explores how to use LLMs to build and manage investment portfolios, combining natural language understanding with quantitative optimization techniques.

## Trading Strategy

**Core Concept:** LLMs process financial documents, news sentiment, and market data to generate portfolio weights, asset recommendations, and rebalancing signals.

**Entry Signals:**
- Long allocation: Positive sentiment + favorable fundamentals identified by LLM
- Increased weight: LLM identifies undervalued assets with growth catalysts
- Reduced weight: LLM detects deteriorating fundamentals or negative sentiment

**Edge:** LLMs can synthesize vast amounts of unstructured data (earnings calls, news, SEC filings) into actionable portfolio recommendations faster than human analysts, identifying subtle patterns and cross-asset relationships.

## Technical Specification

### Key Components

1. **Data Ingestion Pipeline** - Collect market data, news, and fundamental data
2. **LLM Analysis Engine** - Generate asset assessments and recommendations
3. **Portfolio Optimizer** - Convert LLM insights into optimal weights
4. **Risk Management** - Constraint-based portfolio construction
5. **Backtesting Framework** - Validate strategy performance

### Architecture

```
                    ┌─────────────────────┐
                    │   Data Sources      │
                    │ (News, Filings,     │
                    │  Market Data)       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Processor    │
                    │ (Summarization,     │
                    │  Feature Extract)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LLM Portfolio     │
                    │      Engine         │
                    │ (Analysis + Recs)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Portfolio Optimizer│
                    │ (Mean-Variance,     │
                    │  Risk Parity)       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Execution Layer   │
                    │ (Orders + Rebalance)│
                    └─────────────────────┘
```

### Data Requirements

```
Market Data:
├── OHLCV price data (Bybit for crypto, Yahoo for stocks)
├── Trading volume and liquidity metrics
├── Volatility and correlation data
└── Benchmark indices

Fundamental Data:
├── Earnings reports and guidance
├── SEC filings (10-K, 10-Q, 8-K)
├── Analyst estimates and revisions
└── Financial ratios and metrics

Alternative Data:
├── News articles and headlines
├── Social media sentiment
├── Earnings call transcripts
└── Macroeconomic indicators
```

### LLM Portfolio Approaches

The LLM can be used in several ways for portfolio construction:

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Direct Allocation** | LLM outputs portfolio weights directly | Simple, interpretable |
| **Scoring + Optimization** | LLM scores assets, optimizer sets weights | Combines LLM insight with math |
| **Multi-Agent Ensemble** | Multiple LLM personas vote on allocation | Robust, diverse perspectives |
| **RAG-Enhanced** | LLM retrieves relevant data before deciding | Access to real-time information |

### Prompt Engineering for Portfolio Construction

```python
PORTFOLIO_CONSTRUCTION_PROMPT = """
You are a quantitative portfolio manager. Analyze the following assets and market conditions.

Assets to consider:
{asset_list}

Recent market data:
{market_data}

News and sentiment:
{news_summary}

Current portfolio:
{current_portfolio}

Based on this information, provide:

1. Asset Scores (1-10 scale):
   - Fundamental Score: Quality of financials and business
   - Momentum Score: Price trend and technical indicators
   - Sentiment Score: News and social sentiment
   - Risk Score: Volatility and downside risk

2. Recommended Portfolio Weights (must sum to 100%):
   - For each asset, provide target weight and reasoning

3. Rebalancing Actions:
   - What trades to execute
   - Priority order of trades
   - Risk considerations

4. Confidence Level: (low/medium/high)

Output as JSON format.
"""
```

### Key Metrics

**Portfolio Performance:**
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted)
- Maximum Drawdown
- Calmar Ratio
- Information Ratio vs benchmark

**LLM Quality Metrics:**
- Recommendation accuracy
- Ranking correlation (Spearman)
- Hit rate on direction predictions
- Turnover efficiency

### Dependencies

```python
# Python dependencies
openai>=1.0.0           # OpenAI API client
anthropic>=0.5.0        # Claude API client
transformers>=4.30.0    # HuggingFace models
torch>=2.0.0            # PyTorch
pandas>=2.0.0           # Data manipulation
numpy>=1.24.0           # Numerical computing
yfinance>=0.2.0         # Stock data
scipy>=1.10.0           # Optimization
cvxpy>=1.4.0            # Convex optimization
requests>=2.28.0        # HTTP client
```

```rust
// Rust dependencies
reqwest = "0.12"        // HTTP client
serde = "1.0"           // Serialization
tokio = "1.0"           // Async runtime
ndarray = "0.16"        // Arrays
polars = "0.46"         // DataFrames
```

## Python Implementation

### Portfolio Data Structures

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np

class AssetClass(Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    BOND = "bond"
    COMMODITY = "commodity"

@dataclass
class Asset:
    """Represents a tradeable asset."""
    symbol: str
    name: str
    asset_class: AssetClass
    current_price: float
    market_cap: Optional[float] = None

@dataclass
class AssetScore:
    """LLM-generated scores for an asset."""
    symbol: str
    fundamental_score: float  # 1-10
    momentum_score: float     # 1-10
    sentiment_score: float    # 1-10
    risk_score: float         # 1-10 (higher = more risk)
    overall_score: float      # Weighted combination
    reasoning: str            # LLM explanation
    confidence: str           # low/medium/high

    @property
    def composite_score(self) -> float:
        """Calculate weighted composite score."""
        # Higher is better, so invert risk score
        weights = {
            'fundamental': 0.30,
            'momentum': 0.25,
            'sentiment': 0.25,
            'risk': 0.20
        }
        return (
            weights['fundamental'] * self.fundamental_score +
            weights['momentum'] * self.momentum_score +
            weights['sentiment'] * self.sentiment_score +
            weights['risk'] * (10 - self.risk_score)  # Invert risk
        )

@dataclass
class Portfolio:
    """Represents a portfolio allocation."""
    weights: Dict[str, float]  # symbol -> weight
    cash_weight: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        # Normalize weights to sum to 1
        total = sum(self.weights.values()) + self.cash_weight
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
            self.cash_weight = self.cash_weight / total

    def get_weight(self, symbol: str) -> float:
        return self.weights.get(symbol, 0.0)

    def to_dict(self) -> Dict:
        return {
            "weights": self.weights,
            "cash_weight": self.cash_weight,
            "timestamp": self.timestamp
        }
```

### LLM Portfolio Engine

```python
import json
from typing import List, Dict, Tuple
import openai

class LLMPortfolioEngine:
    """LLM-based portfolio construction engine."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def analyze_assets(
        self,
        assets: List[Asset],
        market_data: Dict,
        news_data: List[str]
    ) -> List[AssetScore]:
        """Analyze assets and generate scores using LLM."""

        # Prepare asset information
        asset_info = self._format_assets(assets)
        market_summary = self._format_market_data(market_data)
        news_summary = self._format_news(news_data)

        prompt = self._build_analysis_prompt(asset_info, market_summary, news_summary)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a quantitative analyst specializing in portfolio construction."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return self._parse_scores(result)

    def generate_portfolio(
        self,
        scores: List[AssetScore],
        constraints: Dict = None
    ) -> Portfolio:
        """Generate portfolio weights from asset scores."""

        if constraints is None:
            constraints = {
                "max_weight": 0.30,
                "min_weight": 0.02,
                "max_assets": 10,
                "min_score": 5.0
            }

        # Filter assets by minimum score
        valid_scores = [s for s in scores if s.composite_score >= constraints["min_score"]]

        # Sort by composite score
        valid_scores.sort(key=lambda x: x.composite_score, reverse=True)

        # Take top N assets
        selected = valid_scores[:constraints["max_assets"]]

        # Calculate weights proportional to scores
        total_score = sum(s.composite_score for s in selected)

        weights = {}
        for score in selected:
            raw_weight = score.composite_score / total_score
            # Apply constraints
            weight = max(constraints["min_weight"],
                        min(constraints["max_weight"], raw_weight))
            weights[score.symbol] = weight

        # Normalize
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        return Portfolio(weights=weights)

    def _build_analysis_prompt(
        self,
        assets: str,
        market: str,
        news: str
    ) -> str:
        return f"""Analyze the following assets for portfolio construction.

ASSETS:
{assets}

MARKET CONDITIONS:
{market}

RECENT NEWS:
{news}

For each asset, provide scores (1-10) and analysis:
- fundamental_score: Quality of business and financials
- momentum_score: Price trend strength
- sentiment_score: News and market sentiment
- risk_score: Volatility and downside risk (10 = highest risk)
- reasoning: Brief explanation
- confidence: low/medium/high

Return JSON with "scores" array containing objects for each asset."""

    def _format_assets(self, assets: List[Asset]) -> str:
        lines = []
        for a in assets:
            lines.append(f"- {a.symbol}: {a.name} ({a.asset_class.value}), Price: ${a.current_price:.2f}")
        return "\n".join(lines)

    def _format_market_data(self, data: Dict) -> str:
        lines = []
        for symbol, info in data.items():
            lines.append(f"- {symbol}: Return 7d: {info.get('return_7d', 0):.1%}, Volatility: {info.get('volatility', 0):.1%}")
        return "\n".join(lines)

    def _format_news(self, news: List[str]) -> str:
        return "\n".join([f"- {n}" for n in news[:10]])

    def _parse_scores(self, data: Dict) -> List[AssetScore]:
        scores = []
        for item in data.get("scores", []):
            scores.append(AssetScore(
                symbol=item.get("symbol", ""),
                fundamental_score=float(item.get("fundamental_score", 5)),
                momentum_score=float(item.get("momentum_score", 5)),
                sentiment_score=float(item.get("sentiment_score", 5)),
                risk_score=float(item.get("risk_score", 5)),
                overall_score=float(item.get("overall_score", 5)),
                reasoning=item.get("reasoning", ""),
                confidence=item.get("confidence", "medium")
            ))
        return scores
```

### Mean-Variance Optimizer

```python
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional

class MeanVarianceOptimizer:
    """Mean-variance portfolio optimization with LLM score integration."""

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        target_volatility: Optional[float] = None
    ):
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        llm_scores: Optional[np.ndarray] = None,
        constraints: Dict = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optimize portfolio weights.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            llm_scores: Optional LLM composite scores to blend
            constraints: Portfolio constraints

        Returns:
            Tuple of (weights, metrics)
        """
        n_assets = len(expected_returns)

        if constraints is None:
            constraints = {
                "max_weight": 0.30,
                "min_weight": 0.0,
                "long_only": True
            }

        # Blend LLM scores with expected returns if provided
        if llm_scores is not None:
            # Normalize LLM scores to be on similar scale as returns
            normalized_scores = (llm_scores - llm_scores.mean()) / llm_scores.std()
            blend_weight = 0.3  # Weight given to LLM scores
            adjusted_returns = (
                (1 - blend_weight) * expected_returns +
                blend_weight * normalized_scores * 0.01  # Scale factor
            )
        else:
            adjusted_returns = expected_returns

        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets

        # Objective: maximize Sharpe ratio (minimize negative Sharpe)
        def neg_sharpe(weights):
            port_return = np.dot(weights, adjusted_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            return -(port_return - self.risk_free_rate) / port_vol

        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        if constraints["long_only"]:
            bounds = [(constraints["min_weight"], constraints["max_weight"])
                     for _ in range(n_assets)]
        else:
            bounds = [(-constraints["max_weight"], constraints["max_weight"])
                     for _ in range(n_assets)]

        # Optimize
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        weights = result.x

        # Calculate metrics
        port_return = np.dot(weights, adjusted_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol

        metrics = {
            "expected_return": port_return,
            "volatility": port_vol,
            "sharpe_ratio": sharpe,
            "optimization_success": result.success
        }

        return weights, metrics

    def risk_parity(
        self,
        covariance_matrix: np.ndarray,
        risk_budget: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Risk parity portfolio allocation.

        Each asset contributes equally to portfolio risk.
        """
        n_assets = covariance_matrix.shape[0]

        if risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets

        def risk_contribution_error(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol
            target_contrib = risk_budget * port_vol
            return np.sum((risk_contrib - target_contrib) ** 2)

        x0 = np.ones(n_assets) / n_assets
        bounds = [(0.01, 0.5) for _ in range(n_assets)]
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        result = minimize(
            risk_contribution_error,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )

        weights = result.x
        port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

        metrics = {
            "volatility": port_vol,
            "optimization_success": result.success
        }

        return weights, metrics
```

### Backtesting Framework

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class BacktestResult:
    """Results from portfolio backtest."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    num_trades: int
    portfolio_values: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"""
Portfolio Backtest Results
==========================
Total Return:      {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Volatility:        {self.volatility:.2%}
Sharpe Ratio:      {self.sharpe_ratio:.2f}
Sortino Ratio:     {self.sortino_ratio:.2f}
Max Drawdown:      {self.max_drawdown:.2%}
Calmar Ratio:      {self.calmar_ratio:.2f}
Win Rate:          {self.win_rate:.2%}
Number of Trades:  {self.num_trades}
"""

class PortfolioBacktester:
    """Backtest LLM-based portfolio strategies."""

    def __init__(
        self,
        initial_capital: float = 100000,
        rebalance_frequency: str = "weekly",  # daily, weekly, monthly
        transaction_cost: float = 0.001,  # 0.1%
        slippage: float = 0.0005  # 0.05%
    ):
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def run(
        self,
        price_data: pd.DataFrame,
        portfolio_weights: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """
        Run backtest with given portfolio weights.

        Args:
            price_data: DataFrame with asset prices (columns = symbols)
            portfolio_weights: Dict mapping dates to weight DataFrames
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResult with performance metrics
        """
        # Filter data
        mask = (price_data.index >= start_date) & (price_data.index <= end_date)
        prices = price_data.loc[mask].copy()

        # Initialize
        capital = self.initial_capital
        portfolio_values = [capital]
        dates = [prices.index[0]]
        current_weights = {}
        num_trades = 0

        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(prices.index)

        for i in range(1, len(prices)):
            date = prices.index[i]
            prev_date = prices.index[i-1]

            # Calculate returns
            daily_returns = (prices.iloc[i] / prices.iloc[i-1]) - 1

            # Check for rebalance
            if date in rebalance_dates and str(date) in portfolio_weights:
                new_weights = portfolio_weights[str(date)]

                # Calculate turnover and costs
                turnover = self._calculate_turnover(current_weights, new_weights)
                cost = turnover * (self.transaction_cost + self.slippage)
                capital *= (1 - cost)
                num_trades += sum(1 for s in new_weights if new_weights.get(s, 0) != current_weights.get(s, 0))

                current_weights = new_weights

            # Calculate portfolio return
            port_return = sum(
                current_weights.get(symbol, 0) * daily_returns.get(symbol, 0)
                for symbol in current_weights
            )

            capital *= (1 + port_return)
            portfolio_values.append(capital)
            dates.append(date)

        # Calculate metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()

        total_return = (capital / self.initial_capital) - 1
        trading_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino = annualized_return / downside_std

        # Max drawdown
        cumulative = pd.Series(portfolio_values)
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Calmar
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            win_rate=win_rate,
            num_trades=num_trades,
            portfolio_values=portfolio_values,
            dates=[str(d) for d in dates]
        )

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> set:
        """Get rebalance dates based on frequency."""
        if self.rebalance_frequency == "daily":
            return set(dates)
        elif self.rebalance_frequency == "weekly":
            # Rebalance on Mondays
            return set(dates[dates.dayofweek == 0])
        elif self.rebalance_frequency == "monthly":
            # Rebalance on first trading day of month
            return set(dates.to_series().groupby(dates.to_period('M')).first())
        return set()

    def _calculate_turnover(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        """Calculate portfolio turnover."""
        all_symbols = set(old_weights.keys()) | set(new_weights.keys())
        turnover = sum(
            abs(new_weights.get(s, 0) - old_weights.get(s, 0))
            for s in all_symbols
        ) / 2
        return turnover
```

## Rust Implementation

See the `rust_llm_portfolio/` directory for the complete Rust implementation, which includes:

- **Data fetching** from Bybit and Yahoo Finance
- **LLM API integration** (OpenAI compatible)
- **Portfolio optimization** algorithms
- **Backtesting** framework
- **Performance metrics** calculation

### Quick Start (Rust)

```bash
cd rust_llm_portfolio

# Build the project
cargo build --release

# Fetch market data
cargo run --example fetch_data

# Run portfolio analysis
cargo run --example analyze_portfolio -- --symbols BTCUSDT,ETHUSDT,SOLUSDT

# Backtest strategy
cargo run --example backtest -- --start 2024-01-01 --end 2024-06-01
```

## Expected Outcomes

1. **LLM Analysis Pipeline** - End-to-end system for asset scoring
2. **Portfolio Construction** - Optimized weights based on LLM insights
3. **Risk Management** - Constraint-based portfolio construction
4. **Backtesting Results** - Historical performance validation
5. **Rebalancing Strategy** - Dynamic portfolio adjustment rules

## Use Cases

### Cryptocurrency Portfolio

```python
# Example: Build crypto portfolio with LLM
assets = [
    Asset("BTCUSDT", "Bitcoin", AssetClass.CRYPTO, 65000),
    Asset("ETHUSDT", "Ethereum", AssetClass.CRYPTO, 3200),
    Asset("SOLUSDT", "Solana", AssetClass.CRYPTO, 140),
    Asset("BNBUSDT", "Binance Coin", AssetClass.CRYPTO, 580),
]

# Get LLM scores
scores = engine.analyze_assets(assets, market_data, news)

# Generate portfolio
portfolio = engine.generate_portfolio(scores, constraints={
    "max_weight": 0.40,  # Max 40% in single asset
    "min_weight": 0.05,  # Min 5% allocation
    "max_assets": 5
})
```

### Stock Portfolio

```python
# Example: Build diversified stock portfolio
assets = [
    Asset("AAPL", "Apple Inc", AssetClass.EQUITY, 185),
    Asset("MSFT", "Microsoft", AssetClass.EQUITY, 420),
    Asset("GOOGL", "Alphabet", AssetClass.EQUITY, 175),
    Asset("NVDA", "NVIDIA", AssetClass.EQUITY, 880),
    Asset("AMZN", "Amazon", AssetClass.EQUITY, 185),
]

# Analyze with sector constraints
scores = engine.analyze_assets(assets, market_data, news)
portfolio = engine.generate_portfolio(scores, constraints={
    "max_weight": 0.25,
    "min_weight": 0.05,
    "sector_limits": {"tech": 0.60}  # Max 60% in tech
})
```

### Multi-Agent Ensemble

```python
# Example: Use multiple LLM personas for robust allocation
personas = [
    "value_investor",     # Focus on fundamentals
    "momentum_trader",    # Focus on trends
    "risk_manager",       # Focus on downside
    "contrarian"          # Opposite of consensus
]

ensemble_weights = {}
for persona in personas:
    scores = engine.analyze_with_persona(assets, persona)
    weights = engine.generate_portfolio(scores)
    ensemble_weights[persona] = weights

# Aggregate: average weights across personas
final_weights = aggregate_portfolios(ensemble_weights)
```

## Best Practices

1. **Prompt Engineering** - Test prompts for consistent, actionable output
2. **Score Calibration** - Validate LLM scores against historical outcomes
3. **Constraint Setting** - Use reasonable position limits and diversification
4. **Regular Validation** - Backtest frequently with out-of-sample data
5. **Human Oversight** - Review LLM recommendations before execution
6. **Cost Management** - Cache LLM responses to reduce API costs
7. **Fallback Logic** - Have rules-based backup if LLM fails

## References

- [Large Language Models in Equity Markets](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full) - Comprehensive survey of LLM applications in stock investing
- [LLM Agents for Investment Management](https://dl.acm.org/doi/10.1145/3768292.3770387) - Review of agent-based approaches
- [FolioLLM: Portfolio Construction with LLMs](https://web.stanford.edu/class/cs224n/final-reports/256938687.pdf) - Stanford research on ETF allocation
- [Persona-Based LLM Ensembles](https://arxiv.org/html/2411.19515v1) - University of Tokyo research on ensemble methods
- [From Text to Returns](https://arxiv.org/abs/2512.05907) - Mutual fund optimization with LLMs
- [BloombergGPT](https://arxiv.org/abs/2303.17564) - Large language model for finance
- [FinGPT](https://arxiv.org/abs/2306.06031) - Open-source financial LLM

## Difficulty Level

Expert

Required knowledge: LLM prompting, portfolio optimization, quantitative finance, API integration, backtesting methodology
