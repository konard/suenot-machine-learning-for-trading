# Chapter 64: Multi-Agent LLM Trading — Collaborative AI Systems for Financial Markets

This chapter explores **Multi-Agent Large Language Model (LLM) Trading Systems**, a cutting-edge approach that combines the reasoning capabilities of LLMs with multi-agent collaboration to simulate how professional trading firms operate. We examine how specialized AI agents can work together to analyze markets, debate strategies, and execute trades.

<p align="center">
<img src="https://i.imgur.com/KqXmF8Z.png" width="75%">
</p>

## Contents

1. [Introduction](#introduction)
    * [Why Multi-Agent Systems for Trading?](#why-multi-agent-systems-for-trading)
    * [Key Innovations](#key-innovations)
    * [Comparison with Traditional Approaches](#comparison-with-traditional-approaches)
2. [Multi-Agent Architecture](#multi-agent-architecture)
    * [Agent Types and Roles](#agent-types-and-roles)
    * [Communication Patterns](#communication-patterns)
    * [Decision-Making Process](#decision-making-process)
3. [Core Frameworks](#core-frameworks)
    * [TradingAgents Framework](#tradingagents-framework)
    * [FinCon Framework](#fincon-framework)
    * [Framework Comparison](#framework-comparison)
4. [Trading Applications](#trading-applications)
    * [Stock Trading](#stock-trading)
    * [Cryptocurrency Trading](#cryptocurrency-trading)
    * [Portfolio Management](#portfolio-management)
5. [Practical Examples](#practical-examples)
    * [01: Building a Multi-Agent Trading System](#01-building-a-multi-agent-trading-system)
    * [02: Agent Communication and Debate](#02-agent-communication-and-debate)
    * [03: Risk Management Integration](#03-risk-management-integration)
    * [04: Backtesting Multi-Agent Strategies](#04-backtesting-multi-agent-strategies)
6. [Rust Implementation](#rust-implementation)
7. [Python Implementation](#python-implementation)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction

Multi-Agent LLM Trading represents a paradigm shift from single-model approaches to collaborative AI systems that mimic how professional trading firms operate. Instead of relying on a single model to make all decisions, these systems employ specialized agents that analyze different aspects of the market, debate their findings, and collectively arrive at trading decisions.

### Why Multi-Agent Systems for Trading?

Traditional single-agent trading systems face several limitations:

```
SINGLE-AGENT LIMITATIONS:
┌──────────────────────────────────────────────────────────────────────────┐
│  1. INFORMATION OVERLOAD                                                  │
│     Single model must process fundamentals, technicals, sentiment, news   │
│     Result: Diluted expertise, inconsistent analysis depth                │
├──────────────────────────────────────────────────────────────────────────┤
│  2. NO DEBATE OR VALIDATION                                               │
│     Decisions made without adversarial review                             │
│     Result: Confirmation bias, overconfident predictions                  │
├──────────────────────────────────────────────────────────────────────────┤
│  3. STATIC RISK ASSESSMENT                                                │
│     Risk checked at point-in-time, not continuously                       │
│     Result: Delayed response to changing market conditions                │
├──────────────────────────────────────────────────────────────────────────┤
│  4. LACK OF SPECIALIZATION                                                │
│     Generalist model vs. domain experts                                   │
│     Result: Shallow analysis across all dimensions                        │
└──────────────────────────────────────────────────────────────────────────┘

MULTI-AGENT SOLUTION:
┌──────────────────────────────────────────────────────────────────────────┐
│  SPECIALIZED AGENTS work together:                                        │
│  • Fundamental Analyst: Deep financial statement analysis                 │
│  • Technical Analyst: Chart patterns, indicators, price action           │
│  • Sentiment Analyst: Social media, news sentiment                       │
│  • Risk Manager: Continuous exposure monitoring                          │
│  • Trader: Synthesizes insights, executes decisions                      │
│                                                                           │
│  Result: Comprehensive analysis with checks and balances                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Innovations

1. **Role Specialization**
   - Each agent has a specific expertise area
   - Agents are prompted and equipped with tools relevant to their role
   - Prevents information overload and ensures deep analysis

2. **Adversarial Debate**
   - Bull and Bear researchers argue opposing viewpoints
   - Forces consideration of both opportunities and risks
   - Reduces confirmation bias in trading decisions

3. **Hierarchical Decision-Making**
   - Analysts provide data and insights
   - Researchers synthesize and debate
   - Traders propose actions
   - Risk managers approve or reject

4. **Verbal Reinforcement Learning**
   - Agents develop investment beliefs through self-critique
   - Knowledge propagates selectively across the system
   - Continuous improvement from experience

### Comparison with Traditional Approaches

| Aspect | Single Model | Traditional Quant | Multi-Agent LLM |
|--------|-------------|-------------------|-----------------|
| Analysis depth | Shallow-broad | Deep-narrow | Deep-broad |
| Adaptability | Limited | Rule-based | High |
| Reasoning transparency | Low | Medium | High |
| Risk assessment | Static | Quantitative | Dynamic + Qualitative |
| Market regime handling | Poor | Manual rules | Automatic adaptation |
| Unstructured data | Good | Poor | Excellent |
| Latency | Medium | Very low | High |
| Cost | Low | Medium | High |

## Multi-Agent Architecture

### Agent Types and Roles

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT TRADING SYSTEM ARCHITECTURE                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           ANALYST TEAM                                   │ │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌───────────┐ │ │
│  │  │ Fundamentals   │ │   Technical    │ │   Sentiment    │ │   News    │ │ │
│  │  │   Analyst      │ │   Analyst      │ │   Analyst      │ │  Analyst  │ │ │
│  │  │ ─────────────  │ │ ─────────────  │ │ ─────────────  │ │ ────────  │ │ │
│  │  │ • P/E, EPS     │ │ • RSI, MACD    │ │ • Social media │ │ • Macro   │ │ │
│  │  │ • Revenue      │ │ • Patterns     │ │ • Forums       │ │ • Events  │ │ │
│  │  │ • Cash flow    │ │ • Support/Res  │ │ • Sentiment    │ │ • Policy  │ │ │
│  │  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘ └─────┬─────┘ │ │
│  └──────────┼──────────────────┼──────────────────┼────────────────┼───────┘ │
│             │                  │                  │                │         │
│             └──────────────────┴─────────┬────────┴────────────────┘         │
│                                          │                                    │
│                                          ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         RESEARCH TEAM                                    │ │
│  │  ┌──────────────────────────────┐ ┌──────────────────────────────┐      │ │
│  │  │      Bull Researcher         │ │      Bear Researcher         │      │ │
│  │  │ ───────────────────────────  │ │ ───────────────────────────  │      │ │
│  │  │ • Identifies opportunities   │ │ • Identifies risks           │      │ │
│  │  │ • Argues for long positions  │ │ • Argues for caution/shorts  │      │ │
│  │  │ • Growth catalysts           │ │ • Downside scenarios         │      │ │
│  │  └──────────────┬───────────────┘ └───────────────┬──────────────┘      │ │
│  │                 │         DEBATE                  │                      │ │
│  │                 └─────────────┬───────────────────┘                      │ │
│  └───────────────────────────────┼──────────────────────────────────────────┘ │
│                                  │                                            │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    TRADING & RISK MANAGEMENT                             │ │
│  │  ┌──────────────────────────────┐ ┌──────────────────────────────┐      │ │
│  │  │         Trader               │ │      Risk Manager            │      │ │
│  │  │ ───────────────────────────  │ │ ───────────────────────────  │      │ │
│  │  │ • Synthesizes all insights   │ │ • Portfolio exposure         │      │ │
│  │  │ • Determines position size   │ │ • Market volatility          │      │ │
│  │  │ • Entry/exit timing          │ │ • Correlation analysis       │      │ │
│  │  │ • Risk profile (aggr/cons)   │ │ • Stop-loss enforcement      │      │ │
│  │  └──────────────┬───────────────┘ └───────────────┬──────────────┘      │ │
│  │                 │                                 │                      │ │
│  │                 └─────────────┬───────────────────┘                      │ │
│  └───────────────────────────────┼──────────────────────────────────────────┘ │
│                                  │                                            │
│                                  ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       PORTFOLIO MANAGER                                  │ │
│  │  • Final approval authority                                              │ │
│  │  • Executes approved trades                                              │ │
│  │  • Updates portfolio state                                               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Communication Patterns

Multi-agent systems use several communication patterns:

```python
# Communication Pattern Types

# 1. SEQUENTIAL (Pipeline)
# Each agent passes output to the next
flow = fundamentals -> technicals -> sentiment -> trader

# 2. PARALLEL (Concurrent analysis)
# Analysts work simultaneously
analysts = [fundamentals, technicals, sentiment, news]
results = await asyncio.gather(*[a.analyze() for a in analysts])

# 3. HIERARCHICAL (Manager-Worker)
# Manager coordinates, workers execute
manager.assign(task, worker)
worker.report(result, manager)

# 4. DEBATE (Adversarial)
# Agents argue opposing views
debate = bull.argue() vs bear.argue()
resolution = moderator.resolve(debate)

# 5. BROADCAST (Shared state)
# All agents see shared market state
state.update(new_data)
all_agents.observe(state)
```

### Decision-Making Process

```
TRADING DECISION WORKFLOW
═══════════════════════════════════════════════════════════════════════════════

Step 1: DATA COLLECTION
────────────────────────
Market data, news, social media, filings → Shared State

Step 2: PARALLEL ANALYSIS
─────────────────────────
┌─────────────────────────────────────────────────────────────────────────────┐
│ Fundamentals    │ Technical       │ Sentiment       │ News               │
│ "P/E is 18x,    │ "RSI at 65,    │ "Twitter buzz   │ "Fed meeting       │
│  below avg"     │  MACD bullish"  │  +15% positive" │  tomorrow"         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 3: RESEARCH SYNTHESIS
──────────────────────────
┌───────────────────────────────────┬─────────────────────────────────────────┐
│ Bull Researcher                   │ Bear Researcher                         │
│ "Strong fundamentals + positive   │ "Fed uncertainty could cause            │
│  momentum = BUY opportunity"      │  volatility. Wait for clarity."         │
└─────────────────────────────┬─────┴──────────────────┬──────────────────────┘
                              │      DEBATE            │
                              └──────────┬─────────────┘
                                         │
                                         ▼
Step 4: TRADING DECISION
────────────────────────
Trader: "Given bullish fundamentals but macro uncertainty:
         Recommend: 50% position size, scale in after Fed"

Step 5: RISK REVIEW
───────────────────
Risk Manager: "Current portfolio 60% equities.
              Adding position keeps us within limits.
              Approved with 5% stop-loss."

Step 6: EXECUTION
─────────────────
Portfolio Manager: Execute BUY 500 shares @ market
                   Set stop-loss @ -5%
```

## Core Frameworks

### TradingAgents Framework

[TradingAgents](https://github.com/TauricResearch/TradingAgents) is an open-source framework that simulates professional trading firms through collaborative LLM agents.

**Key Components:**

```python
# TradingAgents structure (simplified)
from tradingagents import TradingAgentsGraph

class TradingAgentsGraph:
    """
    Main orchestrator for multi-agent trading system.

    Agents follow ReAct (Reasoning + Acting) prompting framework,
    synergizing reasoning and acting in context-appropriate ways.
    """

    def __init__(self, config):
        # Analysis team - data interpretation
        self.fundamentals_analyst = FundamentalsAgent(config)
        self.technical_analyst = TechnicalAgent(config)
        self.sentiment_analyst = SentimentAgent(config)
        self.news_analyst = NewsAgent(config)

        # Research team - critical assessment
        self.bull_researcher = BullResearcher(config)
        self.bear_researcher = BearResearcher(config)

        # Trading team - decision making
        self.trader = TraderAgent(config)
        self.risk_manager = RiskManager(config)
        self.portfolio_manager = PortfolioManager(config)

    def analyze(self, symbol: str, date: str) -> TradingDecision:
        """Run full analysis pipeline for a symbol."""
        # 1. Parallel analysis
        fundamentals = self.fundamentals_analyst.analyze(symbol)
        technicals = self.technical_analyst.analyze(symbol)
        sentiment = self.sentiment_analyst.analyze(symbol)
        news = self.news_analyst.analyze(symbol)

        # 2. Research synthesis with debate
        bull_case = self.bull_researcher.assess(
            fundamentals, technicals, sentiment, news
        )
        bear_case = self.bear_researcher.assess(
            fundamentals, technicals, sentiment, news
        )

        # 3. Trading decision
        trade_recommendation = self.trader.decide(
            bull_case, bear_case, self.portfolio_state
        )

        # 4. Risk review
        approved_trade = self.risk_manager.review(
            trade_recommendation, self.portfolio_state
        )

        return approved_trade
```

**Model Selection Strategy:**

TradingAgents uses different models for different cognitive demands:

| Task Type | Model | Reasoning Depth |
|-----------|-------|-----------------|
| Data retrieval, summarization | gpt-4o-mini | Fast thinking |
| Analysis, report generation | gpt-4o | Moderate depth |
| Decision-making, debate | o1-preview | Deep thinking |

### FinCon Framework

[FinCon](https://arxiv.org/abs/2407.06567) introduces a manager-analyst hierarchy with verbal reinforcement learning.

**Key Innovations:**

```python
# FinCon conceptual structure

class FinConSystem:
    """
    FinCon: Conceptual Verbal Reinforcement for Financial Agents

    Key innovation: Agents develop investment beliefs through self-critique,
    which then guide future behavior (verbal reinforcement learning).
    """

    def __init__(self):
        self.manager = ManagerAgent()
        self.analysts = [
            FundamentalAnalyst(),
            TechnicalAnalyst(),
            MacroAnalyst()
        ]
        self.investment_beliefs = InvestmentBeliefStore()

    def execute_cycle(self, market_state):
        # 1. Manager assigns tasks to analysts
        tasks = self.manager.decompose_analysis(market_state)

        for analyst, task in zip(self.analysts, tasks):
            analyst.execute(task)

        # 2. Analysts report to manager
        reports = [a.generate_report() for a in self.analysts]

        # 3. Manager makes decision guided by beliefs
        decision = self.manager.decide(
            reports,
            self.investment_beliefs.get_relevant()
        )

        # 4. Execute and observe outcome
        outcome = self.execute_trade(decision)

        # 5. Self-critique and belief update (verbal RL)
        critique = self.manager.self_critique(decision, outcome)
        new_beliefs = self.extract_beliefs(critique)
        self.investment_beliefs.update(new_beliefs)

        return decision, outcome

class InvestmentBeliefStore:
    """
    Stores learned investment beliefs that guide future decisions.

    Example beliefs:
    - "In high-VIX environments, reduce position sizes by 30%"
    - "Earnings beats with weak guidance often lead to selloffs"
    - "RSI divergence is more reliable in trending markets"
    """

    def __init__(self):
        self.beliefs = []

    def update(self, new_beliefs):
        """Add or refine investment beliefs."""
        for belief in new_beliefs:
            if self._is_novel(belief):
                self.beliefs.append(belief)
            else:
                self._refine_existing(belief)

    def get_relevant(self, context=None):
        """Retrieve beliefs relevant to current context."""
        if context is None:
            return self.beliefs
        return [b for b in self.beliefs if self._matches_context(b, context)]
```

### Framework Comparison

| Feature | TradingAgents | FinCon |
|---------|---------------|--------|
| Agent hierarchy | Flat with roles | Manager-analyst |
| Debate mechanism | Bull vs Bear | Manager synthesis |
| Learning | None (stateless) | Verbal RL (belief update) |
| Communication | Shared state | Hierarchical messaging |
| Open source | Yes | Paper only |
| Primary use case | Daily trading | Portfolio management |

## Trading Applications

### Stock Trading

```python
# Example: Multi-agent stock trading analysis

class StockTradingSystem:
    """
    Multi-agent system for stock trading decisions.
    """

    def analyze_stock(self, symbol: str) -> Dict:
        """
        Run comprehensive multi-agent analysis on a stock.

        Args:
            symbol: Stock ticker (e.g., "AAPL")

        Returns:
            Analysis report with trading recommendation
        """
        # Gather data
        price_data = self.data_provider.get_ohlcv(symbol)
        fundamentals = self.data_provider.get_fundamentals(symbol)
        news = self.data_provider.get_news(symbol)
        social = self.data_provider.get_social_sentiment(symbol)

        # Agent analyses
        fundamental_report = self.fundamentals_agent.analyze(
            symbol, fundamentals
        )
        # Key metrics: P/E, P/B, ROE, revenue growth, margins

        technical_report = self.technical_agent.analyze(
            symbol, price_data
        )
        # Indicators: RSI, MACD, Bollinger, support/resistance

        sentiment_report = self.sentiment_agent.analyze(
            symbol, news, social
        )
        # Sentiment scores from news and social media

        # Research debate
        bull_thesis = self.bull_researcher.build_case(
            fundamental_report, technical_report, sentiment_report
        )

        bear_thesis = self.bear_researcher.build_case(
            fundamental_report, technical_report, sentiment_report
        )

        # Trading decision
        decision = self.trader.synthesize(
            bull_thesis, bear_thesis,
            current_position=self.portfolio.get_position(symbol)
        )

        return {
            "symbol": symbol,
            "recommendation": decision.action,  # BUY/SELL/HOLD
            "confidence": decision.confidence,
            "position_size": decision.size,
            "entry_price": decision.entry,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "reasoning": decision.reasoning,
            "bull_thesis": bull_thesis.summary,
            "bear_thesis": bear_thesis.summary
        }
```

### Cryptocurrency Trading

Multi-agent systems are particularly suited for crypto markets due to 24/7 operation and high sentiment influence:

```python
# Crypto-specific multi-agent configuration

class CryptoTradingAgents:
    """
    Multi-agent system optimized for cryptocurrency trading.

    Crypto-specific considerations:
    - 24/7 market (no market hours)
    - Higher volatility requires faster response
    - Social sentiment is major price driver
    - On-chain data provides unique insights
    """

    def __init__(self):
        # Standard agents
        self.technical_agent = CryptoTechnicalAgent()
        self.sentiment_agent = CryptoSentimentAgent()

        # Crypto-specific agents
        self.onchain_agent = OnChainAnalyst()  # Whale movements, exchange flows
        self.defi_agent = DeFiAnalyst()        # TVL, yield analysis
        self.social_agent = CryptoSocialAgent() # Twitter, Discord, Telegram

        # Research
        self.bull_researcher = CryptoBullResearcher()
        self.bear_researcher = CryptoBearResearcher()

        # Execution
        self.trader = CryptoTrader()
        self.risk_manager = CryptoRiskManager()

    def analyze_crypto(self, symbol: str) -> Dict:
        """
        Analyze cryptocurrency with crypto-specific insights.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
        """
        # Technical analysis
        technical = self.technical_agent.analyze(symbol)

        # Sentiment from multiple sources
        social_sentiment = self.social_agent.analyze(symbol)
        # Twitter mentions, Discord activity, Telegram groups

        # On-chain analytics
        onchain = self.onchain_agent.analyze(symbol)
        # Exchange inflows/outflows, whale wallet movements
        # Active addresses, transaction volume

        # DeFi metrics (if applicable)
        if self._is_defi_token(symbol):
            defi_metrics = self.defi_agent.analyze(symbol)
            # TVL, protocol revenue, yield comparisons

        # Synthesize with debate
        bull_case = self.bull_researcher.assess({
            "technical": technical,
            "sentiment": social_sentiment,
            "onchain": onchain
        })

        bear_case = self.bear_researcher.assess({
            "technical": technical,
            "sentiment": social_sentiment,
            "onchain": onchain
        })

        return self.trader.decide(bull_case, bear_case)
```

### Portfolio Management

```python
# Multi-agent portfolio management

class MultiAgentPortfolioManager:
    """
    Multi-agent system for portfolio-level decisions.

    Handles:
    - Asset allocation
    - Rebalancing
    - Risk management across positions
    """

    def __init__(self, assets: List[str]):
        self.assets = assets

        # Per-asset analysis agents
        self.asset_analysts = {
            asset: AssetAnalystTeam(asset)
            for asset in assets
        }

        # Portfolio-level agents
        self.allocation_agent = AllocationAgent()
        self.correlation_agent = CorrelationAgent()
        self.risk_agent = PortfolioRiskAgent()
        self.rebalancing_agent = RebalancingAgent()

    def optimize_portfolio(
        self,
        current_holdings: Dict[str, float],
        target_return: float,
        max_volatility: float
    ) -> Dict[str, float]:
        """
        Optimize portfolio allocation using multi-agent analysis.

        Args:
            current_holdings: Current portfolio weights
            target_return: Target annualized return
            max_volatility: Maximum acceptable volatility

        Returns:
            Optimized portfolio weights
        """
        # 1. Analyze each asset
        asset_views = {}
        for asset in self.assets:
            analysis = self.asset_analysts[asset].analyze()
            asset_views[asset] = {
                "expected_return": analysis.expected_return,
                "confidence": analysis.confidence,
                "risks": analysis.risks
            }

        # 2. Analyze correlations
        correlations = self.correlation_agent.analyze(self.assets)
        # Identifies diversification opportunities

        # 3. Propose allocation
        proposed_weights = self.allocation_agent.optimize(
            asset_views,
            correlations,
            target_return,
            max_volatility
        )

        # 4. Risk review
        risk_assessment = self.risk_agent.assess(
            proposed_weights,
            correlations,
            max_volatility
        )

        if not risk_assessment.approved:
            # Iterate with risk constraints
            proposed_weights = self.allocation_agent.optimize(
                asset_views,
                correlations,
                target_return,
                max_volatility * 0.9  # Tighten constraint
            )

        # 5. Determine rebalancing trades
        trades = self.rebalancing_agent.calculate_trades(
            current_holdings,
            proposed_weights
        )

        return proposed_weights, trades
```

## Practical Examples

### 01: Building a Multi-Agent Trading System

```python
# python/01_multi_agent_system.py

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json

class AgentRole(Enum):
    FUNDAMENTALS = "fundamentals"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    NEWS = "news"
    BULL_RESEARCHER = "bull_researcher"
    BEAR_RESEARCHER = "bear_researcher"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"

@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: AgentRole
    content: str
    data: Dict = field(default_factory=dict)
    confidence: float = 0.0

@dataclass
class TradingDecision:
    """Final trading decision from the system."""
    action: str  # BUY, SELL, HOLD
    symbol: str
    size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    reasoning: str = ""

class BaseAgent:
    """Base class for all trading agents."""

    def __init__(self, role: AgentRole, llm_client):
        self.role = role
        self.llm = llm_client
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Get role-specific system prompt."""
        prompts = {
            AgentRole.FUNDAMENTALS: """You are a Fundamental Analyst specializing in financial statement analysis.
Your job is to analyze company fundamentals: revenue, earnings, margins, P/E ratios, debt levels, and growth prospects.
Provide clear, data-driven analysis with specific numbers.""",

            AgentRole.TECHNICAL: """You are a Technical Analyst specializing in price action and chart patterns.
Your job is to analyze technical indicators: RSI, MACD, Bollinger Bands, support/resistance levels, and chart patterns.
Identify key price levels and momentum signals.""",

            AgentRole.SENTIMENT: """You are a Sentiment Analyst specializing in market sentiment and social media.
Your job is to analyze news sentiment, social media buzz, and market psychology.
Quantify sentiment and identify potential sentiment-driven moves.""",

            AgentRole.NEWS: """You are a News Analyst specializing in macroeconomic events and news impact.
Your job is to analyze news events, earnings reports, policy changes, and their market implications.
Focus on material events that could move stock prices.""",

            AgentRole.BULL_RESEARCHER: """You are a Bull Researcher advocating for bullish positions.
Your job is to build the case for why a stock should go UP.
Highlight growth catalysts, positive trends, and upside potential.
Be persuasive but grounded in facts.""",

            AgentRole.BEAR_RESEARCHER: """You are a Bear Researcher advocating for caution or bearish positions.
Your job is to build the case for why a stock could go DOWN or why caution is warranted.
Highlight risks, negative trends, and downside scenarios.
Be persuasive but grounded in facts.""",

            AgentRole.TRADER: """You are a Professional Trader making final trading decisions.
Your job is to synthesize bull and bear cases and make actionable trading decisions.
Consider position sizing, entry points, stop losses, and take profits.
Be decisive and justify your decisions.""",

            AgentRole.RISK_MANAGER: """You are a Risk Manager overseeing portfolio risk.
Your job is to review trading decisions for risk compliance.
Consider position concentration, volatility, correlation, and max drawdown.
Approve, reject, or modify trading proposals."""
        }
        return prompts.get(self.role, "You are a trading assistant.")

    async def analyze(self, context: Dict) -> AgentMessage:
        """Run analysis and return message."""
        raise NotImplementedError

class FundamentalsAgent(BaseAgent):
    """Agent specialized in fundamental analysis."""

    def __init__(self, llm_client):
        super().__init__(AgentRole.FUNDAMENTALS, llm_client)

    async def analyze(self, symbol: str, fundamentals: Dict) -> AgentMessage:
        """Analyze fundamental data for a stock."""
        prompt = f"""Analyze the following fundamental data for {symbol}:

{json.dumps(fundamentals, indent=2)}

Provide:
1. Key financial metrics assessment (P/E, P/B, margins, growth)
2. Financial health evaluation
3. Valuation assessment (overvalued/undervalued/fair)
4. Key risks from fundamentals
5. Overall fundamental score (1-10)
"""
        response = await self.llm.generate(self.system_prompt, prompt)

        return AgentMessage(
            sender=self.role,
            content=response,
            data={"symbol": symbol, "fundamentals": fundamentals},
            confidence=0.8  # Could be extracted from LLM response
        )

class TechnicalAgent(BaseAgent):
    """Agent specialized in technical analysis."""

    def __init__(self, llm_client):
        super().__init__(AgentRole.TECHNICAL, llm_client)

    async def analyze(self, symbol: str, price_data: Dict) -> AgentMessage:
        """Analyze technical indicators for a stock."""
        prompt = f"""Analyze the following technical data for {symbol}:

Price: ${price_data.get('close', 'N/A')}
RSI (14): {price_data.get('rsi', 'N/A')}
MACD: {price_data.get('macd', 'N/A')}
Signal: {price_data.get('macd_signal', 'N/A')}
52-week High: ${price_data.get('high_52w', 'N/A')}
52-week Low: ${price_data.get('low_52w', 'N/A')}
Volume vs Avg: {price_data.get('volume_ratio', 'N/A')}x

Provide:
1. Momentum assessment (RSI, MACD)
2. Trend direction and strength
3. Key support and resistance levels
4. Chart pattern identification (if any)
5. Overall technical score (1-10) and bias (bullish/bearish/neutral)
"""
        response = await self.llm.generate(self.system_prompt, prompt)

        return AgentMessage(
            sender=self.role,
            content=response,
            data={"symbol": symbol, "price_data": price_data},
            confidence=0.75
        )

class ResearcherAgent(BaseAgent):
    """Agent that builds bull or bear case."""

    def __init__(self, llm_client, is_bull: bool):
        role = AgentRole.BULL_RESEARCHER if is_bull else AgentRole.BEAR_RESEARCHER
        super().__init__(role, llm_client)
        self.is_bull = is_bull

    async def build_case(
        self,
        symbol: str,
        analyst_reports: List[AgentMessage]
    ) -> AgentMessage:
        """Build bull or bear case from analyst reports."""
        reports_text = "\n\n".join([
            f"=== {r.sender.value.upper()} ANALYSIS ===\n{r.content}"
            for r in analyst_reports
        ])

        stance = "BULLISH (long)" if self.is_bull else "BEARISH (short/avoid)"

        prompt = f"""Based on the following analyst reports for {symbol}, build a {stance} case:

{reports_text}

Your task:
1. Synthesize the key points supporting a {stance} stance
2. Identify the strongest arguments for your position
3. Address potential counterarguments
4. Provide specific price targets or scenarios
5. Rate the strength of your case (1-10)
"""
        response = await self.llm.generate(self.system_prompt, prompt)

        return AgentMessage(
            sender=self.role,
            content=response,
            data={"symbol": symbol},
            confidence=0.7
        )

class TraderAgent(BaseAgent):
    """Agent that makes final trading decisions."""

    def __init__(self, llm_client):
        super().__init__(AgentRole.TRADER, llm_client)

    async def decide(
        self,
        symbol: str,
        bull_case: AgentMessage,
        bear_case: AgentMessage,
        current_position: float,
        portfolio_value: float
    ) -> TradingDecision:
        """Make trading decision based on bull and bear cases."""
        prompt = f"""Make a trading decision for {symbol}.

=== BULL CASE ===
{bull_case.content}

=== BEAR CASE ===
{bear_case.content}

Current position: {current_position} shares
Portfolio value: ${portfolio_value:,.2f}
Max position size: 10% of portfolio

Provide your decision in this exact format:
ACTION: [BUY/SELL/HOLD]
SIZE: [number of shares or 0]
ENTRY: [target entry price or current]
STOP_LOSS: [stop loss price]
TAKE_PROFIT: [take profit price]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"""
        response = await self.llm.generate(self.system_prompt, prompt)

        # Parse response (in production, use structured output)
        decision = self._parse_decision(response, symbol)
        return decision

    def _parse_decision(self, response: str, symbol: str) -> TradingDecision:
        """Parse LLM response into TradingDecision."""
        # Simple parsing - in production use structured output or better parsing
        lines = response.strip().split('\n')
        decision = TradingDecision(action="HOLD", symbol=symbol, size=0)

        for line in lines:
            if line.startswith("ACTION:"):
                decision.action = line.split(":")[1].strip()
            elif line.startswith("SIZE:"):
                try:
                    decision.size = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("CONFIDENCE:"):
                try:
                    decision.confidence = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("REASONING:"):
                decision.reasoning = line.split(":", 1)[1].strip()

        return decision

class MultiAgentTradingSystem:
    """
    Complete multi-agent trading system.

    Orchestrates multiple specialized agents to analyze markets
    and make trading decisions.
    """

    def __init__(self, llm_client, data_provider):
        self.llm = llm_client
        self.data = data_provider

        # Initialize agents
        self.fundamentals_agent = FundamentalsAgent(llm_client)
        self.technical_agent = TechnicalAgent(llm_client)
        self.bull_researcher = ResearcherAgent(llm_client, is_bull=True)
        self.bear_researcher = ResearcherAgent(llm_client, is_bull=False)
        self.trader = TraderAgent(llm_client)

        # Portfolio state
        self.portfolio = {"cash": 100000, "positions": {}}

    async def analyze_and_trade(self, symbol: str) -> TradingDecision:
        """
        Run full analysis pipeline and make trading decision.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            TradingDecision with action, size, and reasoning
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING {symbol}")
        print('='*60)

        # 1. Gather data
        fundamentals = await self.data.get_fundamentals(symbol)
        price_data = await self.data.get_technical_data(symbol)

        # 2. Run parallel analysis
        print("\n--- Running Analyst Team ---")
        fundamental_report, technical_report = await asyncio.gather(
            self.fundamentals_agent.analyze(symbol, fundamentals),
            self.technical_agent.analyze(symbol, price_data)
        )

        print(f"Fundamentals: {len(fundamental_report.content)} chars")
        print(f"Technical: {len(technical_report.content)} chars")

        # 3. Research synthesis
        print("\n--- Running Research Team ---")
        analyst_reports = [fundamental_report, technical_report]

        bull_case, bear_case = await asyncio.gather(
            self.bull_researcher.build_case(symbol, analyst_reports),
            self.bear_researcher.build_case(symbol, analyst_reports)
        )

        print(f"Bull case: {len(bull_case.content)} chars")
        print(f"Bear case: {len(bear_case.content)} chars")

        # 4. Trading decision
        print("\n--- Making Trading Decision ---")
        current_position = self.portfolio["positions"].get(symbol, 0)
        portfolio_value = self._calculate_portfolio_value()

        decision = await self.trader.decide(
            symbol,
            bull_case,
            bear_case,
            current_position,
            portfolio_value
        )

        print(f"Decision: {decision.action} {decision.size} shares")
        print(f"Confidence: {decision.confidence:.1%}")
        print(f"Reasoning: {decision.reasoning}")

        return decision

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        # Simplified - in production, get current prices
        return self.portfolio["cash"] + sum(
            shares * 100  # Assume $100/share for simplicity
            for shares in self.portfolio["positions"].values()
        )


# Example usage
async def main():
    from mock_llm import MockLLMClient
    from mock_data import MockDataProvider

    # Initialize with mock clients for demo
    llm = MockLLMClient()
    data = MockDataProvider()

    system = MultiAgentTradingSystem(llm, data)

    # Analyze and trade
    decision = await system.analyze_and_trade("AAPL")

    print(f"\n{'='*60}")
    print("FINAL DECISION")
    print('='*60)
    print(f"Symbol: {decision.symbol}")
    print(f"Action: {decision.action}")
    print(f"Size: {decision.size}")
    print(f"Confidence: {decision.confidence:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 02: Agent Communication and Debate

```python
# python/02_agent_debate.py

import asyncio
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class DebateRole(Enum):
    BULL = "bull"
    BEAR = "bear"
    MODERATOR = "moderator"

@dataclass
class DebateArgument:
    """Single argument in a debate."""
    role: DebateRole
    round: int
    argument: str
    evidence: List[str]
    strength: float  # 0-1

@dataclass
class DebateOutcome:
    """Outcome of a debate session."""
    winner: DebateRole
    margin: float  # How decisive the win
    key_points: List[str]
    consensus_view: str
    recommended_action: str

class DebateAgent:
    """Agent that participates in debates."""

    def __init__(self, role: DebateRole, llm_client):
        self.role = role
        self.llm = llm_client
        self.arguments_made = []

    async def make_argument(
        self,
        context: Dict,
        opponent_argument: str = None,
        round_num: int = 1
    ) -> DebateArgument:
        """Make an argument in the debate."""

        if self.role == DebateRole.BULL:
            stance = "bullish (the stock will go UP)"
            goal = "convince that this is a good buying opportunity"
        else:
            stance = "bearish (the stock will go DOWN or stay flat)"
            goal = "convince that caution is warranted"

        if opponent_argument:
            prompt = f"""You are arguing the {stance} case.

Previous opponent argument:
{opponent_argument}

Market context:
{context}

Your previous arguments:
{self.arguments_made}

Round {round_num}: Provide a counter-argument that:
1. Directly addresses the opponent's strongest point
2. Introduces new supporting evidence
3. Strengthens your overall position

Format:
ARGUMENT: [Your main point]
EVIDENCE: [Bullet points of supporting evidence]
REBUTTAL: [Direct response to opponent]
"""
        else:
            prompt = f"""You are arguing the {stance} case.

Market context:
{context}

Round {round_num}: Make your opening argument that:
1. States your thesis clearly
2. Provides supporting evidence
3. Anticipates counterarguments

Format:
ARGUMENT: [Your main point]
EVIDENCE: [Bullet points of supporting evidence]
"""

        response = await self.llm.generate(
            f"You are a {self.role.value} researcher. Your goal is to {goal}.",
            prompt
        )

        argument = DebateArgument(
            role=self.role,
            round=round_num,
            argument=response,
            evidence=[],  # Would parse from response
            strength=0.7  # Would calculate
        )

        self.arguments_made.append(argument)
        return argument

class DebateModerator:
    """Moderates debates between bull and bear researchers."""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def moderate_debate(
        self,
        bull: DebateAgent,
        bear: DebateAgent,
        context: Dict,
        rounds: int = 3
    ) -> DebateOutcome:
        """
        Run a multi-round debate between bull and bear.

        Args:
            bull: Bull researcher agent
            bear: Bear researcher agent
            context: Market data and analysis context
            rounds: Number of debate rounds

        Returns:
            DebateOutcome with winner and key insights
        """
        debate_log = []

        # Opening arguments
        bull_opening = await bull.make_argument(context, round_num=1)
        bear_opening = await bear.make_argument(context, round_num=1)

        debate_log.extend([bull_opening, bear_opening])

        # Subsequent rounds
        for round_num in range(2, rounds + 1):
            # Bull responds to bear
            bull_arg = await bull.make_argument(
                context,
                opponent_argument=debate_log[-1].argument,
                round_num=round_num
            )

            # Bear responds to bull
            bear_arg = await bear.make_argument(
                context,
                opponent_argument=bull_arg.argument,
                round_num=round_num
            )

            debate_log.extend([bull_arg, bear_arg])

        # Judge the debate
        outcome = await self._judge_debate(debate_log, context)

        return outcome

    async def _judge_debate(
        self,
        debate_log: List[DebateArgument],
        context: Dict
    ) -> DebateOutcome:
        """Judge the debate and determine outcome."""

        debate_text = "\n\n".join([
            f"=== {arg.role.value.upper()} (Round {arg.round}) ===\n{arg.argument}"
            for arg in debate_log
        ])

        prompt = f"""As an impartial judge, evaluate this debate about a stock:

{debate_text}

Evaluate based on:
1. Strength of evidence provided
2. Quality of rebuttals
3. Overall persuasiveness
4. Factual accuracy

Provide your judgment:
WINNER: [BULL/BEAR/TIE]
MARGIN: [DECISIVE/MODERATE/SLIGHT/TIE]
KEY_BULL_POINTS: [Bullet points]
KEY_BEAR_POINTS: [Bullet points]
CONSENSUS: [Balanced view incorporating both perspectives]
RECOMMENDATION: [BUY/SELL/HOLD based on the debate]
"""

        response = await self.llm.generate(
            "You are an impartial debate judge with financial expertise.",
            prompt
        )

        # Parse response (simplified)
        outcome = DebateOutcome(
            winner=DebateRole.BULL,  # Would parse from response
            margin=0.6,
            key_points=["Strong fundamentals", "Technical overbought"],
            consensus_view="Fundamentally sound but technically extended",
            recommended_action="HOLD"
        )

        return outcome


class DebateDrivenTrader:
    """
    Trader that uses debate outcomes to make decisions.
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.moderator = DebateModerator(llm_client)

    async def analyze_with_debate(
        self,
        symbol: str,
        market_data: Dict
    ) -> Dict:
        """
        Analyze a stock using debate-based reasoning.

        The debate process helps surface both bullish and bearish
        perspectives, leading to more balanced decisions.
        """
        # Create debate agents
        bull = DebateAgent(DebateRole.BULL, self.llm)
        bear = DebateAgent(DebateRole.BEAR, self.llm)

        # Prepare context
        context = {
            "symbol": symbol,
            "price": market_data.get("price"),
            "pe_ratio": market_data.get("pe_ratio"),
            "revenue_growth": market_data.get("revenue_growth"),
            "rsi": market_data.get("rsi"),
            "recent_news": market_data.get("news", [])
        }

        # Run debate
        print(f"Starting debate for {symbol}...")
        outcome = await self.moderator.moderate_debate(
            bull, bear, context, rounds=3
        )

        print(f"\nDebate Winner: {outcome.winner.value}")
        print(f"Margin: {outcome.margin:.1%}")
        print(f"Recommendation: {outcome.recommended_action}")
        print(f"\nConsensus: {outcome.consensus_view}")

        return {
            "symbol": symbol,
            "debate_outcome": outcome,
            "recommendation": outcome.recommended_action,
            "confidence": outcome.margin
        }


# Example usage
async def main():
    from mock_llm import MockLLMClient

    llm = MockLLMClient()
    trader = DebateDrivenTrader(llm)

    market_data = {
        "price": 175.50,
        "pe_ratio": 28.5,
        "revenue_growth": 0.08,
        "rsi": 65,
        "news": [
            "Company beats Q3 earnings expectations",
            "Analyst downgrades due to valuation concerns"
        ]
    }

    result = await trader.analyze_with_debate("AAPL", market_data)
    print(f"\nFinal recommendation: {result['recommendation']}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 03: Risk Management Integration

```python
# python/03_risk_management.py

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class RiskDecision(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

@dataclass
class RiskAssessment:
    """Risk assessment for a proposed trade."""
    decision: RiskDecision
    original_trade: Dict
    modified_trade: Optional[Dict]
    risk_score: float  # 0-1, higher = riskier
    concerns: List[str]
    recommendations: List[str]

@dataclass
class PortfolioRisk:
    """Current portfolio risk metrics."""
    total_exposure: float
    concentration_risk: float  # Largest position %
    sector_exposure: Dict[str, float]
    correlation_risk: float
    volatility: float
    var_95: float  # Value at Risk 95%
    max_drawdown: float

class RiskManagerAgent:
    """
    Risk management agent that reviews and modifies trades.

    Enforces:
    - Position limits
    - Concentration limits
    - Sector exposure limits
    - Portfolio volatility targets
    - Drawdown limits
    """

    def __init__(self, llm_client, config: Dict):
        self.llm = llm_client

        # Risk parameters
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.max_sector_pct = config.get("max_sector_pct", 0.25)
        self.max_correlation = config.get("max_correlation", 0.8)
        self.max_portfolio_volatility = config.get("max_volatility", 0.20)
        self.max_drawdown = config.get("max_drawdown", 0.15)

    def calculate_portfolio_risk(
        self,
        positions: Dict[str, Dict],
        prices: Dict[str, float]
    ) -> PortfolioRisk:
        """
        Calculate current portfolio risk metrics.

        Args:
            positions: Dict of symbol -> {shares, avg_cost, sector}
            prices: Current prices

        Returns:
            PortfolioRisk with all metrics
        """
        total_value = sum(
            pos["shares"] * prices.get(symbol, pos["avg_cost"])
            for symbol, pos in positions.items()
        )

        # Position concentrations
        position_pcts = {
            symbol: (pos["shares"] * prices.get(symbol, pos["avg_cost"])) / total_value
            for symbol, pos in positions.items()
        }

        # Sector exposures
        sector_exposure = {}
        for symbol, pos in positions.items():
            sector = pos.get("sector", "unknown")
            pct = position_pcts[symbol]
            sector_exposure[sector] = sector_exposure.get(sector, 0) + pct

        return PortfolioRisk(
            total_exposure=total_value,
            concentration_risk=max(position_pcts.values()) if position_pcts else 0,
            sector_exposure=sector_exposure,
            correlation_risk=0.5,  # Would calculate from returns
            volatility=0.15,  # Would calculate from returns
            var_95=total_value * 0.05,  # Simplified
            max_drawdown=0.08  # Would track historically
        )

    async def review_trade(
        self,
        proposed_trade: Dict,
        portfolio_risk: PortfolioRisk,
        portfolio_value: float
    ) -> RiskAssessment:
        """
        Review a proposed trade against risk limits.

        Args:
            proposed_trade: Trade proposal from trader agent
            portfolio_risk: Current portfolio risk metrics
            portfolio_value: Total portfolio value

        Returns:
            RiskAssessment with decision and any modifications
        """
        concerns = []
        recommendations = []

        symbol = proposed_trade["symbol"]
        action = proposed_trade["action"]
        size = proposed_trade["size"]
        price = proposed_trade.get("price", 100)  # Estimate if not provided

        trade_value = size * price
        trade_pct = trade_value / portfolio_value

        # Check position limit
        if trade_pct > self.max_position_pct:
            concerns.append(
                f"Position size {trade_pct:.1%} exceeds limit {self.max_position_pct:.1%}"
            )
            recommendations.append(
                f"Reduce position to {self.max_position_pct:.1%} max"
            )

        # Check concentration after trade
        new_concentration = portfolio_risk.concentration_risk + trade_pct
        if new_concentration > self.max_position_pct:
            concerns.append(
                f"Trade would increase concentration to {new_concentration:.1%}"
            )

        # Check if adding to largest position
        # Would check against current positions

        # Check sector exposure
        sector = proposed_trade.get("sector", "unknown")
        current_sector_exposure = portfolio_risk.sector_exposure.get(sector, 0)
        new_sector_exposure = current_sector_exposure + trade_pct

        if new_sector_exposure > self.max_sector_pct:
            concerns.append(
                f"Sector {sector} exposure would be {new_sector_exposure:.1%}, "
                f"exceeds limit {self.max_sector_pct:.1%}"
            )
            recommendations.append(f"Reduce position or diversify sectors")

        # Use LLM for qualitative risk assessment
        llm_assessment = await self._llm_risk_check(
            proposed_trade, portfolio_risk, concerns
        )

        # Determine decision
        if len(concerns) == 0:
            decision = RiskDecision.APPROVED
            modified_trade = None
            risk_score = 0.3
        elif len(concerns) <= 2 and "Reduce" in str(recommendations):
            decision = RiskDecision.MODIFIED
            modified_trade = self._modify_trade(proposed_trade, recommendations)
            risk_score = 0.6
        else:
            decision = RiskDecision.REJECTED
            modified_trade = None
            risk_score = 0.9

        return RiskAssessment(
            decision=decision,
            original_trade=proposed_trade,
            modified_trade=modified_trade,
            risk_score=risk_score,
            concerns=concerns + llm_assessment.get("additional_concerns", []),
            recommendations=recommendations + llm_assessment.get("recommendations", [])
        )

    async def _llm_risk_check(
        self,
        trade: Dict,
        portfolio_risk: PortfolioRisk,
        existing_concerns: List[str]
    ) -> Dict:
        """Use LLM for qualitative risk assessment."""

        prompt = f"""Review this proposed trade from a risk management perspective:

Trade:
- Symbol: {trade['symbol']}
- Action: {trade['action']}
- Size: {trade['size']} shares
- Confidence: {trade.get('confidence', 'N/A')}

Current Portfolio Risk:
- Total Exposure: ${portfolio_risk.total_exposure:,.0f}
- Concentration: {portfolio_risk.concentration_risk:.1%}
- Volatility: {portfolio_risk.volatility:.1%}
- Max Drawdown: {portfolio_risk.max_drawdown:.1%}

Existing Concerns: {existing_concerns}

Identify any additional risk concerns not captured above:
1. Market timing risks
2. Liquidity concerns
3. Event risks (earnings, Fed, etc.)
4. Correlation with existing positions
5. Any other relevant risks

Format:
ADDITIONAL_CONCERNS: [List any new concerns]
RECOMMENDATIONS: [List actionable recommendations]
RISK_LEVEL: [LOW/MEDIUM/HIGH]
"""

        response = await self.llm.generate(
            "You are a senior risk manager at a hedge fund.",
            prompt
        )

        # Parse response (simplified)
        return {
            "additional_concerns": [],
            "recommendations": [],
            "risk_level": "MEDIUM"
        }

    def _modify_trade(
        self,
        original_trade: Dict,
        recommendations: List[str]
    ) -> Dict:
        """Modify trade based on risk recommendations."""
        modified = original_trade.copy()

        # Reduce size if recommended
        for rec in recommendations:
            if "Reduce" in rec and "%" in rec:
                # Extract target percentage and adjust
                modified["size"] = int(original_trade["size"] * 0.5)

        return modified


class RiskAwareTrader:
    """
    Trader that integrates risk management into decisions.
    """

    def __init__(self, llm_client, risk_config: Dict):
        self.llm = llm_client
        self.risk_manager = RiskManagerAgent(llm_client, risk_config)
        self.portfolio = {
            "cash": 100000,
            "positions": {},
            "value": 100000
        }

    async def execute_with_risk_check(
        self,
        proposed_trade: Dict
    ) -> Dict:
        """
        Execute trade with risk management review.

        Args:
            proposed_trade: Trade proposal from trading system

        Returns:
            Execution result including risk assessment
        """
        # Calculate current portfolio risk
        portfolio_risk = self.risk_manager.calculate_portfolio_risk(
            self.portfolio["positions"],
            {}  # Would pass current prices
        )

        print(f"\n--- Risk Review ---")
        print(f"Current portfolio volatility: {portfolio_risk.volatility:.1%}")
        print(f"Current concentration: {portfolio_risk.concentration_risk:.1%}")

        # Review trade
        assessment = await self.risk_manager.review_trade(
            proposed_trade,
            portfolio_risk,
            self.portfolio["value"]
        )

        print(f"\nRisk Decision: {assessment.decision.value}")
        print(f"Risk Score: {assessment.risk_score:.1%}")

        if assessment.concerns:
            print(f"Concerns: {assessment.concerns}")

        if assessment.recommendations:
            print(f"Recommendations: {assessment.recommendations}")

        # Execute based on decision
        if assessment.decision == RiskDecision.APPROVED:
            trade_to_execute = proposed_trade
            print(f"\nExecuting original trade...")
        elif assessment.decision == RiskDecision.MODIFIED:
            trade_to_execute = assessment.modified_trade
            print(f"\nExecuting modified trade...")
            print(f"Original size: {proposed_trade['size']}")
            print(f"Modified size: {trade_to_execute['size']}")
        else:
            trade_to_execute = None
            print(f"\nTrade rejected by risk management")

        return {
            "proposed_trade": proposed_trade,
            "assessment": assessment,
            "executed_trade": trade_to_execute
        }


# Example usage
async def main():
    from mock_llm import MockLLMClient

    llm = MockLLMClient()

    risk_config = {
        "max_position_pct": 0.10,
        "max_sector_pct": 0.25,
        "max_volatility": 0.20,
        "max_drawdown": 0.15
    }

    trader = RiskAwareTrader(llm, risk_config)

    # Test trade that exceeds limits
    proposed_trade = {
        "symbol": "AAPL",
        "action": "BUY",
        "size": 200,
        "price": 175,
        "sector": "technology",
        "confidence": 0.8
    }

    result = await trader.execute_with_risk_check(proposed_trade)

    print(f"\n--- Final Result ---")
    print(f"Original trade: {result['proposed_trade']}")
    print(f"Risk decision: {result['assessment'].decision.value}")
    if result['executed_trade']:
        print(f"Executed: {result['executed_trade']}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 04: Backtesting Multi-Agent Strategies

```python
# python/04_backtest.py

import asyncio
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta

@dataclass
class BacktestConfig:
    """Configuration for multi-agent backtesting."""
    initial_capital: float = 100000
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    trading_frequency: str = "daily"  # daily, weekly
    transaction_cost_bps: float = 10
    slippage_bps: float = 5
    max_position_pct: float = 0.10

@dataclass
class BacktestTrade:
    """Record of a single trade during backtest."""
    timestamp: datetime
    symbol: str
    action: str
    size: float
    price: float
    value: float
    agent_confidence: float
    reasoning: str

@dataclass
class BacktestMetrics:
    """Performance metrics from backtest."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float

class MultiAgentBacktester:
    """
    Backtester for multi-agent trading systems.

    Simulates how the multi-agent system would have performed
    on historical data, respecting the information available
    at each point in time (no look-ahead bias).
    """

    def __init__(
        self,
        trading_system,
        config: BacktestConfig
    ):
        self.system = trading_system
        self.config = config

        # State tracking
        self.portfolio_history = []
        self.trades = []
        self.agent_decisions = []

    async def run_backtest(
        self,
        symbols: List[str],
        price_data: pd.DataFrame,
        fundamental_data: Dict[str, pd.DataFrame]
    ) -> BacktestMetrics:
        """
        Run backtest on historical data.

        Args:
            symbols: List of symbols to trade
            price_data: Historical OHLCV data
            fundamental_data: Historical fundamental data per symbol

        Returns:
            BacktestMetrics with performance statistics
        """
        # Initialize portfolio
        portfolio = {
            "cash": self.config.initial_capital,
            "positions": {s: 0 for s in symbols},
            "value": self.config.initial_capital
        }

        # Get trading dates
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        trading_dates = price_data.loc[start:end].index

        if self.config.trading_frequency == "weekly":
            # Trade only on Mondays
            trading_dates = [d for d in trading_dates if d.weekday() == 0]

        print(f"Running backtest: {len(trading_dates)} trading days")
        print(f"Symbols: {symbols}")

        # Run through each trading day
        for i, date in enumerate(trading_dates):
            if i % 20 == 0:
                print(f"Processing {date.date()}... Portfolio: ${portfolio['value']:,.0f}")

            # Get data available at this point (no look-ahead)
            available_prices = price_data.loc[:date]

            # Run multi-agent analysis for each symbol
            for symbol in symbols:
                try:
                    # Prepare historical context
                    context = self._prepare_context(
                        symbol, date, available_prices, fundamental_data
                    )

                    # Get trading decision from multi-agent system
                    decision = await self.system.analyze(symbol, context)

                    self.agent_decisions.append({
                        "date": date,
                        "symbol": symbol,
                        "decision": decision
                    })

                    # Execute trade if signaled
                    if decision.action != "HOLD":
                        trade = self._execute_trade(
                            portfolio,
                            decision,
                            price_data.loc[date, (symbol, "close")],
                            date
                        )
                        if trade:
                            self.trades.append(trade)

                except Exception as e:
                    print(f"Error analyzing {symbol} on {date}: {e}")

            # Update portfolio value
            portfolio["value"] = self._calculate_portfolio_value(
                portfolio,
                {s: price_data.loc[date, (s, "close")] for s in symbols}
            )

            self.portfolio_history.append({
                "date": date,
                "value": portfolio["value"],
                "cash": portfolio["cash"],
                "positions": portfolio["positions"].copy()
            })

        # Calculate metrics
        metrics = self._calculate_metrics()

        return metrics

    def _prepare_context(
        self,
        symbol: str,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        fundamentals: Dict
    ) -> Dict:
        """Prepare analysis context from historical data."""
        # Get recent price history
        lookback = 60  # days
        recent_prices = prices.loc[:date].tail(lookback)

        if symbol not in recent_prices.columns.get_level_values(0):
            return {}

        close = recent_prices[(symbol, "close")]
        volume = recent_prices[(symbol, "volume")]

        # Calculate technical indicators
        rsi = self._calculate_rsi(close, 14)
        macd, signal = self._calculate_macd(close)
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20

        # Get fundamentals if available
        fund_data = {}
        if symbol in fundamentals:
            fund_df = fundamentals[symbol]
            # Get most recent data before current date
            available = fund_df[fund_df.index <= date]
            if not available.empty:
                fund_data = available.iloc[-1].to_dict()

        return {
            "symbol": symbol,
            "date": date,
            "price": close.iloc[-1],
            "price_change_1d": close.pct_change().iloc[-1],
            "price_change_5d": close.pct_change(5).iloc[-1] if len(close) >= 5 else 0,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": signal,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "volume_ratio": volume.iloc[-1] / volume.mean() if volume.mean() > 0 else 1,
            "fundamentals": fund_data
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()

        return macd.iloc[-1], signal_line.iloc[-1]

    def _execute_trade(
        self,
        portfolio: Dict,
        decision,
        price: float,
        date: datetime
    ) -> Optional[BacktestTrade]:
        """Execute a trade and update portfolio."""
        symbol = decision.symbol
        action = decision.action

        # Calculate position size
        max_value = portfolio["value"] * self.config.max_position_pct

        # Apply transaction costs
        cost_factor = 1 + (
            self.config.transaction_cost_bps +
            self.config.slippage_bps
        ) / 10000

        if action == "BUY":
            # Buy up to max position
            buy_value = min(decision.size * price, max_value, portfolio["cash"])
            if buy_value < 100:  # Min trade size
                return None

            shares = int(buy_value / (price * cost_factor))
            cost = shares * price * cost_factor

            portfolio["cash"] -= cost
            portfolio["positions"][symbol] += shares

            return BacktestTrade(
                timestamp=date,
                symbol=symbol,
                action="BUY",
                size=shares,
                price=price,
                value=-cost,
                agent_confidence=decision.confidence,
                reasoning=decision.reasoning
            )

        elif action == "SELL":
            # Sell current position
            shares = min(decision.size, portfolio["positions"][symbol])
            if shares <= 0:
                return None

            proceeds = shares * price / cost_factor

            portfolio["cash"] += proceeds
            portfolio["positions"][symbol] -= shares

            return BacktestTrade(
                timestamp=date,
                symbol=symbol,
                action="SELL",
                size=shares,
                price=price,
                value=proceeds,
                agent_confidence=decision.confidence,
                reasoning=decision.reasoning
            )

        return None

    def _calculate_portfolio_value(
        self,
        portfolio: Dict,
        prices: Dict[str, float]
    ) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            portfolio["positions"][s] * prices.get(s, 0)
            for s in portfolio["positions"]
        )
        return portfolio["cash"] + position_value

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate backtest performance metrics."""
        if not self.portfolio_history:
            return BacktestMetrics(
                total_return=0, annualized_return=0, volatility=0,
                sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
                win_rate=0, profit_factor=0, num_trades=0, avg_trade_return=0
            )

        # Portfolio returns
        values = pd.Series(
            [p["value"] for p in self.portfolio_history],
            index=[p["date"] for p in self.portfolio_history]
        )
        returns = values.pct_change().dropna()

        # Basic metrics
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        trading_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = annualized_return / downside_vol if downside_vol > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Trade statistics
        if self.trades:
            trade_returns = [t.value for t in self.trades]
            winning_trades = sum(1 for r in trade_returns if r > 0)
            win_rate = winning_trades / len(self.trades)

            gross_profit = sum(r for r in trade_returns if r > 0)
            gross_loss = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            avg_trade_return = np.mean(trade_returns)
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_return = 0

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(self.trades),
            avg_trade_return=avg_trade_return
        )

    def generate_report(self) -> str:
        """Generate backtest report."""
        metrics = self._calculate_metrics()

        report = f"""
================================================================================
                        MULTI-AGENT BACKTEST REPORT
================================================================================

CONFIGURATION
-------------
Initial Capital: ${self.config.initial_capital:,.0f}
Period: {self.config.start_date} to {self.config.end_date}
Trading Frequency: {self.config.trading_frequency}
Transaction Costs: {self.config.transaction_cost_bps} bps
Max Position Size: {self.config.max_position_pct:.0%}

PERFORMANCE METRICS
-------------------
Total Return:       {metrics.total_return:+.2%}
Annualized Return:  {metrics.annualized_return:+.2%}
Volatility:         {metrics.volatility:.2%}
Sharpe Ratio:       {metrics.sharpe_ratio:.2f}
Sortino Ratio:      {metrics.sortino_ratio:.2f}
Max Drawdown:       {metrics.max_drawdown:.2%}

TRADING STATISTICS
------------------
Number of Trades:   {metrics.num_trades}
Win Rate:           {metrics.win_rate:.1%}
Profit Factor:      {metrics.profit_factor:.2f}
Avg Trade Return:   ${metrics.avg_trade_return:,.0f}

FINAL PORTFOLIO VALUE
---------------------
${self.portfolio_history[-1]['value']:,.0f} (from ${self.config.initial_capital:,.0f})

================================================================================
"""
        return report


# Example usage
async def main():
    from mock_llm import MockLLMClient
    from mock_trading_system import MockMultiAgentSystem

    # Initialize system
    llm = MockLLMClient()
    system = MockMultiAgentSystem(llm)

    config = BacktestConfig(
        initial_capital=100000,
        start_date="2024-01-01",
        end_date="2024-06-30",
        trading_frequency="weekly"
    )

    backtester = MultiAgentBacktester(system, config)

    # Generate mock data
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")

    symbols = ["AAPL", "MSFT", "GOOGL"]
    price_data = pd.DataFrame()

    np.random.seed(42)
    for symbol in symbols:
        base_price = {"AAPL": 150, "MSFT": 350, "GOOGL": 140}[symbol]
        returns = np.random.randn(len(dates)) * 0.02
        prices = base_price * (1 + returns).cumprod()

        price_data[(symbol, "close")] = prices
        price_data[(symbol, "volume")] = np.random.randint(1000000, 10000000, len(dates))

    price_data.index = dates
    price_data.columns = pd.MultiIndex.from_tuples(price_data.columns)

    # Run backtest
    metrics = await backtester.run_backtest(
        symbols,
        price_data,
        fundamental_data={}
    )

    # Print report
    print(backtester.generate_report())


if __name__ == "__main__":
    asyncio.run(main())
```

## Rust Implementation

The Rust implementation provides high-performance multi-agent trading capabilities, optimized for production use with cryptocurrency exchanges like Bybit.

```
rust_multi_agent_trading/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                 # Main library exports
│   ├── agents/                # Agent implementations
│   │   ├── mod.rs
│   │   ├── base.rs           # Base agent traits
│   │   ├── fundamentals.rs   # Fundamentals analyst
│   │   ├── technical.rs      # Technical analyst
│   │   ├── sentiment.rs      # Sentiment analyst
│   │   ├── researcher.rs     # Bull/bear researchers
│   │   ├── trader.rs         # Trading agent
│   │   └── risk.rs           # Risk manager
│   ├── communication/         # Agent communication
│   │   ├── mod.rs
│   │   ├── message.rs        # Message types
│   │   ├── channel.rs        # Communication channels
│   │   └── debate.rs         # Debate mechanism
│   ├── llm/                   # LLM integration
│   │   ├── mod.rs
│   │   ├── client.rs         # API client
│   │   └── prompts.rs        # Prompt templates
│   ├── data/                  # Data providers
│   │   ├── mod.rs
│   │   ├── bybit.rs          # Bybit API
│   │   └── yahoo.rs          # Yahoo Finance
│   ├── strategy/              # Trading strategy
│   │   ├── mod.rs
│   │   ├── decision.rs       # Decision making
│   │   └── execution.rs      # Trade execution
│   └── backtest/              # Backtesting
│       ├── mod.rs
│       └── engine.rs
└── examples/
    ├── simple_analysis.rs
    ├── debate_example.rs
    └── backtest.rs
```

See [rust_multi_agent_trading](rust_multi_agent_trading/) for complete Rust implementation.

### Quick Start (Rust)

```bash
cd rust_multi_agent_trading

# Run simple analysis
cargo run --example simple_analysis -- --symbol BTCUSDT

# Run debate example
cargo run --example debate_example -- --symbol ETHUSDT

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-06-30
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── base.py               # Base agent class
│   ├── fundamentals.py       # Fundamentals analyst
│   ├── technical.py          # Technical analyst
│   ├── sentiment.py          # Sentiment analyst
│   ├── researcher.py         # Bull/bear researchers
│   ├── trader.py             # Trading agent
│   └── risk_manager.py       # Risk manager
├── communication/
│   ├── __init__.py
│   ├── message.py            # Message types
│   └── debate.py             # Debate mechanism
├── system.py                  # Main orchestrator
├── backtest.py                # Backtesting engine
├── data_loader.py             # Data loading utilities
├── requirements.txt           # Dependencies
└── examples/
    ├── 01_simple_analysis.py
    ├── 02_debate_trading.py
    ├── 03_risk_managed.py
    └── 04_full_backtest.py
```

### Quick Start (Python)

```bash
cd python

# Install dependencies
pip install -r requirements.txt

# Run simple analysis
python examples/01_simple_analysis.py --symbol AAPL

# Run debate-based trading
python examples/02_debate_trading.py --symbol BTCUSDT

# Run full backtest
python examples/04_full_backtest.py --capital 100000
```

## Best Practices

### When to Use Multi-Agent LLM Trading

**Ideal use cases:**
- Medium-frequency trading (daily, weekly decisions)
- Complex market analysis requiring multiple perspectives
- Event-driven trading where reasoning is important
- Research and idea generation
- Risk monitoring and compliance

**Not ideal for:**
- High-frequency trading (latency too high)
- Simple momentum strategies (over-engineered)
- Cost-sensitive applications (LLM API costs)

### Agent Design Guidelines

1. **Clear Role Separation**
   ```python
   # Good: Single responsibility per agent
   fundamentals_agent = FundamentalsAgent()  # Only fundamentals
   technical_agent = TechnicalAgent()        # Only technicals

   # Bad: One agent doing everything
   super_agent = AllInOneAgent()  # Diluted expertise
   ```

2. **Appropriate Model Selection**
   ```python
   # Fast tasks -> smaller models
   data_retrieval_model = "gpt-4o-mini"

   # Complex reasoning -> larger models
   decision_making_model = "o1-preview"
   ```

3. **Structured Communication**
   ```python
   # Always use typed messages
   @dataclass
   class AnalystReport:
       symbol: str
       assessment: str
       confidence: float
       key_points: List[str]
   ```

### Risk Management Integration

Always integrate risk management as a gatekeeper:

```python
# Never execute without risk review
decision = trader.decide(...)
assessment = risk_manager.review(decision)

if assessment.approved:
    execute(decision)
else:
    log_rejection(assessment.concerns)
```

### Common Pitfalls

1. **Over-complexity**: Start simple, add agents as needed
2. **Ignoring latency**: Multi-agent adds latency; not suitable for HFT
3. **Hallucination propagation**: Validate agent outputs
4. **Cost explosion**: Monitor and limit API calls
5. **Confirmation bias**: Ensure adversarial (bull/bear) debate

## Resources

### Papers

- [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138) — Multi-agent trading system (2024)
- [FinCon: A Synthesized LLM Multi-Agent System](https://arxiv.org/abs/2407.06567) — Verbal reinforcement learning (2024)
- [Agent Market Arena: Live Multi-Market Trading Benchmark](https://arxiv.org/abs/2510.11695) — LLM agent evaluation (2025)
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) — Multi-agent framework
- [LLM Multi-Agent Systems: Challenges and Open Problems](https://arxiv.org/abs/2402.03578) — Survey paper

### Open-Source Implementations

| Framework | Description | Link |
|-----------|-------------|------|
| TradingAgents | Multi-agent trading | [GitHub](https://github.com/TauricResearch/TradingAgents) |
| AutoGen | Microsoft's agent framework | [GitHub](https://github.com/microsoft/autogen) |
| CrewAI | Multi-agent orchestration | [GitHub](https://github.com/joaomdmoura/crewAI) |
| LangGraph | Agent workflows | [GitHub](https://github.com/langchain-ai/langgraph) |

### Related Chapters

- [Chapter 35: Multi-Agent RL](../35_multi_agent_rl) — RL-based multi-agent systems
- [Chapter 61: FinGPT Financial LLM](../61_fingpt_financial_llm) — Financial LLMs
- [Chapter 62: BloombergGPT Trading](../62_bloomberggpt_trading) — Domain-specific LLMs
- [Chapter 67: LLM Sentiment Analysis](../67_llm_sentiment_analysis) — Sentiment with LLMs
- [Chapter 74: LLM Portfolio Construction](../74_llm_portfolio_construction) — Portfolio management

---

## Difficulty Level

**Expert**

Prerequisites:
- Understanding of LLMs and prompt engineering
- Multi-agent systems and coordination
- Financial markets and trading strategies
- Async programming (Python) or Rust
- Risk management principles
