# Chapter 71: Prompt Engineering for Trading — LLM Optimization Techniques

This chapter explores **Prompt Engineering** techniques specifically designed for financial trading applications. We examine how to craft effective prompts that maximize the accuracy and reliability of Large Language Models (LLMs) for sentiment analysis, trading signal generation, market analysis, and automated trading strategies.

<p align="center">
<img src="https://i.imgur.com/8ZK9xPr.png" width="70%">
</p>

## Contents

1. [Introduction to Prompt Engineering](#introduction-to-prompt-engineering)
    * [What is Prompt Engineering?](#what-is-prompt-engineering)
    * [Why Prompt Engineering Matters for Trading](#why-prompt-engineering-matters-for-trading)
    * [Key Techniques Overview](#key-techniques-overview)
2. [Prompt Engineering Fundamentals](#prompt-engineering-fundamentals)
    * [Zero-Shot Prompting](#zero-shot-prompting)
    * [Few-Shot Prompting](#few-shot-prompting)
    * [Chain-of-Thought Prompting](#chain-of-thought-prompting)
    * [Role-Based Prompting](#role-based-prompting)
3. [Financial Prompt Templates](#financial-prompt-templates)
    * [Sentiment Analysis Prompts](#sentiment-analysis-prompts)
    * [Trading Signal Prompts](#trading-signal-prompts)
    * [Risk Assessment Prompts](#risk-assessment-prompts)
    * [Market Analysis Prompts](#market-analysis-prompts)
4. [Advanced Techniques](#advanced-techniques)
    * [Self-Consistency](#self-consistency)
    * [ReAct (Reasoning + Acting)](#react-reasoning--acting)
    * [Tree-of-Thought](#tree-of-thought)
    * [Prompt Chaining](#prompt-chaining)
5. [Practical Examples](#practical-examples)
    * [01: Financial Sentiment Analysis](#01-financial-sentiment-analysis)
    * [02: Trading Signal Generation](#02-trading-signal-generation)
    * [03: Market Regime Detection](#03-market-regime-detection)
    * [04: Backtesting with LLM Signals](#04-backtesting-with-llm-signals)
6. [Rust Implementation](#rust-implementation)
7. [Python Implementation](#python-implementation)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to Prompt Engineering

### What is Prompt Engineering?

Prompt engineering is the art and science of crafting input instructions (prompts) to guide Large Language Models toward producing desired outputs. In trading applications, well-designed prompts can dramatically improve the quality of LLM-generated insights.

```
PROMPT ENGINEERING WORKFLOW:
┌──────────────────────────────────────────────────────────────────┐
│  RAW INPUT                                                        │
│  "Apple stock went up 5% today"                                  │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  ENGINEERED PROMPT                                                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ You are a financial analyst specializing in tech stocks.    │  │
│  │ Analyze the following news and provide:                     │  │
│  │ 1. Sentiment (POSITIVE/NEGATIVE/NEUTRAL)                    │  │
│  │ 2. Confidence score (0-100)                                 │  │
│  │ 3. Trading recommendation                                   │  │
│  │ 4. Key factors influencing your analysis                    │  │
│  │                                                             │  │
│  │ News: "Apple stock went up 5% today"                        │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  STRUCTURED OUTPUT                                                │
│  {                                                                │
│    "sentiment": "POSITIVE",                                       │
│    "confidence": 85,                                              │
│    "recommendation": "HOLD - wait for confirmation",              │
│    "factors": ["Price momentum", "No catalyst mentioned"]         │
│  }                                                                │
└──────────────────────────────────────────────────────────────────┘
```

### Why Prompt Engineering Matters for Trading

Financial trading presents unique challenges for LLMs:

```
TRADING-SPECIFIC CHALLENGES:
┌──────────────────────────────────────────────────────────────────┐
│  1. PRECISION REQUIREMENTS                                        │
│     Trading decisions require exact numbers, not vague estimates  │
│     Bad: "The stock might go up"                                  │
│     Good: "85% probability of 2-3% increase within 24 hours"      │
├──────────────────────────────────────────────────────────────────┤
│  2. TEMPORAL SENSITIVITY                                          │
│     Financial context changes rapidly                             │
│     - "Rates are high" means different things in 2000 vs 2024    │
│     - "The market crashed" needs specific date context           │
├──────────────────────────────────────────────────────────────────┤
│  3. DOMAIN JARGON                                                 │
│     Financial terminology requires precise understanding          │
│     - "Bullish divergence" vs "Bearish divergence"               │
│     - "P/E expansion" vs "Multiple compression"                   │
├──────────────────────────────────────────────────────────────────┤
│  4. RISK OF HALLUCINATION                                         │
│     LLMs may generate plausible but incorrect financial data     │
│     - Invented stock prices or earnings numbers                   │
│     - Fabricated analyst recommendations                          │
├──────────────────────────────────────────────────────────────────┤
│  5. ACTIONABLE OUTPUT                                             │
│     Trading needs clear, executable recommendations               │
│     - Entry/exit points, position sizes                          │
│     - Stop-loss and take-profit levels                           │
└──────────────────────────────────────────────────────────────────┘
```

### Key Techniques Overview

| Technique | Best For | Complexity |
|-----------|----------|------------|
| Zero-Shot | Simple classification | Low |
| Few-Shot | Consistent output format | Medium |
| Chain-of-Thought | Complex reasoning | Medium |
| Role-Based | Domain expertise | Low |
| Self-Consistency | High-stakes decisions | High |
| ReAct | Multi-step analysis | High |
| Tree-of-Thought | Strategic planning | Very High |

## Prompt Engineering Fundamentals

### Zero-Shot Prompting

Zero-shot prompting provides instructions without examples. It relies on the LLM's pre-trained knowledge.

```python
# Example: Zero-shot sentiment analysis
zero_shot_prompt = """
Analyze the sentiment of the following financial news.
Respond with only: POSITIVE, NEGATIVE, or NEUTRAL

News: "{news_text}"

Sentiment:
"""

# Usage
news = "Tesla reports record quarterly deliveries, beating analyst expectations by 15%"
# Expected output: POSITIVE
```

**Advantages:**
- Simple to implement
- No example data needed
- Fast to iterate

**Limitations:**
- Less control over output format
- May be inconsistent
- Struggles with complex tasks

### Few-Shot Prompting

Few-shot prompting includes examples to guide the LLM toward desired output format and behavior.

```python
# Example: Few-shot sentiment analysis for trading
few_shot_prompt = """
Analyze financial news sentiment and provide a trading signal.

Examples:

News: "Apple beats earnings expectations with strong iPhone sales"
Analysis: {
  "sentiment": "POSITIVE",
  "confidence": 92,
  "signal": "BUY",
  "reasoning": "Earnings beat typically leads to 2-5% price increase"
}

News: "FDA rejects drug application from Pfizer"
Analysis: {
  "sentiment": "NEGATIVE",
  "confidence": 88,
  "signal": "SELL",
  "reasoning": "Drug rejection significantly impacts pipeline value"
}

News: "Microsoft announces stock split effective next month"
Analysis: {
  "sentiment": "NEUTRAL",
  "confidence": 75,
  "signal": "HOLD",
  "reasoning": "Stock splits are mechanical, no fundamental change"
}

Now analyze this news:
News: "{news_text}"
Analysis:
"""
```

**Best Practices for Few-Shot:**
1. Use 3-5 diverse examples
2. Include edge cases
3. Maintain consistent format
4. Cover all output categories

### Chain-of-Thought Prompting

Chain-of-Thought (CoT) prompting encourages step-by-step reasoning, which is crucial for complex financial analysis.

```python
# Example: Chain-of-Thought for trading decision
cot_prompt = """
Analyze the following market situation step by step to determine the optimal trading action.

Market Data:
- Asset: {symbol}
- Current Price: ${current_price}
- 24h Change: {change_24h}%
- Volume: {volume} (vs 20-day avg: {volume_ratio}x)
- RSI(14): {rsi}
- News: "{news_headline}"

Think through this step by step:

Step 1: Assess the price action
- What does the 24h change tell us about momentum?
- How does this compare to typical daily moves?

Step 2: Evaluate volume
- Is volume confirming the price move?
- High volume on up moves = bullish confirmation

Step 3: Check technical indicators
- RSI > 70 suggests overbought
- RSI < 30 suggests oversold

Step 4: Incorporate news sentiment
- How might the news impact future price?
- Is the move already priced in?

Step 5: Synthesize and decide
- Combine all factors
- Determine confidence level
- Provide specific recommendation

Final Analysis:
"""
```

**CoT Benefits for Trading:**
- Forces systematic analysis
- Exposes reasoning for review
- Reduces impulsive signals
- Enables backtesting of logic

### Role-Based Prompting

Role-based prompting assigns the LLM a specific persona to leverage domain expertise.

```python
# Example: Role-based prompts for different trading styles
role_prompts = {
    "quant_analyst": """
You are a quantitative analyst at a top hedge fund with 15 years of experience.
Your analysis style:
- Data-driven, skeptical of narratives
- Focus on statistical significance
- Risk-adjusted returns over absolute returns
- Always consider position sizing and correlation
""",

    "technical_trader": """
You are a professional technical analyst specializing in crypto markets.
Your analysis style:
- Price action and volume are primary
- Support/resistance levels are key
- Trend following with momentum confirmation
- Strict stop-loss discipline
""",

    "fundamental_analyst": """
You are a fundamental analyst focused on long-term value investing.
Your analysis style:
- Earnings quality and sustainability
- Competitive moat analysis
- Management track record
- Margin of safety is essential
""",

    "risk_manager": """
You are a senior risk manager at an institutional trading desk.
Your analysis style:
- Maximum drawdown is the primary concern
- Correlation and concentration risk
- Tail risk and black swan events
- Position limits and portfolio VaR
"""
}

def create_role_prompt(role: str, task: str, data: str) -> str:
    return f"""
{role_prompts[role]}

Task: {task}

Data:
{data}

Provide your analysis:
"""
```

## Financial Prompt Templates

### Sentiment Analysis Prompts

```python
# Template: Comprehensive sentiment analysis
SENTIMENT_TEMPLATE = """
You are analyzing financial news for trading signals.

NEWS ARTICLE:
"{article_text}"

Analyze this article and provide:

1. OVERALL SENTIMENT
   - Classification: [BULLISH | BEARISH | NEUTRAL]
   - Confidence: [0-100]%
   - Reasoning: [Brief explanation]

2. ENTITY-SPECIFIC SENTIMENT
   For each mentioned company/asset:
   - Entity: [Name/Ticker]
   - Sentiment: [BULLISH | BEARISH | NEUTRAL]
   - Impact Level: [HIGH | MEDIUM | LOW]

3. TEMPORAL ANALYSIS
   - Short-term impact (1-5 days): [Description]
   - Medium-term impact (1-4 weeks): [Description]
   - Already priced in: [YES | NO | PARTIALLY]

4. TRADING IMPLICATIONS
   - Primary signal: [BUY | SELL | HOLD]
   - Suggested entry: [Immediate | Wait for pullback | Wait for confirmation]
   - Risk level: [HIGH | MEDIUM | LOW]

Respond in JSON format.
"""
```

### Trading Signal Prompts

```python
# Template: Trading signal generation
SIGNAL_TEMPLATE = """
You are a systematic trading signal generator.

MARKET CONTEXT:
- Asset: {symbol}
- Timeframe: {timeframe}
- Current Price: ${price}
- Market Regime: {regime}

TECHNICAL DATA:
- Trend (50 SMA): {trend_direction}
- Momentum (RSI): {rsi}
- Volatility (ATR%): {atr_percent}%
- Volume Trend: {volume_trend}

FUNDAMENTAL DATA:
- P/E Ratio: {pe_ratio}
- Earnings Surprise (Last): {earnings_surprise}%
- Analyst Consensus: {analyst_rating}

SENTIMENT DATA:
- News Sentiment (24h): {news_sentiment}
- Social Sentiment: {social_sentiment}
- Institutional Flow: {inst_flow}

Generate a trading signal:

1. SIGNAL
   - Direction: [LONG | SHORT | FLAT]
   - Strength: [STRONG | MODERATE | WEAK]
   - Confidence: [0-100]%

2. ENTRY
   - Type: [MARKET | LIMIT]
   - Price (if LIMIT): $___
   - Size (% of portfolio): ___

3. EXIT PLAN
   - Take Profit 1: $___ (___% of position)
   - Take Profit 2: $___ (___% of position)
   - Stop Loss: $___
   - Trailing Stop: [YES/NO] at ___% ATR

4. RATIONALE
   [2-3 sentence explanation of the signal]

5. RISK ASSESSMENT
   - Max Loss: $___
   - Risk/Reward Ratio: ___
   - Key Risks: [List main risks]

Output as JSON.
"""
```

### Risk Assessment Prompts

```python
# Template: Portfolio risk assessment
RISK_TEMPLATE = """
You are a risk management specialist evaluating portfolio risk.

CURRENT PORTFOLIO:
{portfolio_positions}

MARKET CONDITIONS:
- VIX Level: {vix}
- Market Trend: {market_trend}
- Correlation Regime: {correlation_regime}
- Key Upcoming Events: {upcoming_events}

Perform comprehensive risk assessment:

1. POSITION-LEVEL RISK
   For each position, assess:
   - Concentration risk (% of portfolio)
   - Liquidity risk (days to exit)
   - Specific risk factors

2. PORTFOLIO-LEVEL RISK
   - Total market exposure (beta)
   - Sector concentration
   - Geographic concentration
   - Factor exposures

3. SCENARIO ANALYSIS
   Estimate portfolio impact under:
   - Market crash (-20%): ___% portfolio loss
   - Sector rotation: ___% portfolio impact
   - Interest rate shock (+100bps): ___% impact
   - Black swan event: Worst-case loss

4. RECOMMENDATIONS
   - Positions to reduce: [List]
   - Hedges to consider: [List]
   - Diversification gaps: [List]
   - Priority actions: [Ranked list]

5. RISK METRICS
   - Portfolio VaR (95%, 1-day): $___
   - Expected Shortfall: $___
   - Maximum Drawdown (historical): ____%
   - Sharpe Ratio: ___

Provide detailed JSON output.
"""
```

### Market Analysis Prompts

```python
# Template: Market regime detection
REGIME_TEMPLATE = """
You are analyzing current market conditions to identify the market regime.

MARKET DATA:
- S&P 500: {sp500_price} ({sp500_change}% MTD)
- VIX: {vix}
- 10Y Treasury: {treasury_10y}%
- Dollar Index (DXY): {dxy}
- Crude Oil: ${oil}
- Gold: ${gold}

BREADTH INDICATORS:
- Advance/Decline: {adv_dec_ratio}
- % Above 200 SMA: {pct_above_200sma}%
- New Highs - New Lows: {high_low_diff}

INTERMARKET SIGNALS:
- Stock/Bond Correlation: {stock_bond_corr}
- Dollar/Equity Correlation: {dollar_equity_corr}

Classify the current market regime:

1. REGIME CLASSIFICATION
   Select one:
   - RISK_ON_TRENDING: Strong uptrend, low volatility
   - RISK_ON_VOLATILE: Uptrend with high volatility
   - RISK_OFF_TRENDING: Strong downtrend, rising volatility
   - RISK_OFF_PANIC: Crash mode, extreme fear
   - RANGING: Sideways, low conviction
   - TRANSITIONAL: Regime change in progress

2. CONFIDENCE: [0-100]%

3. KEY DRIVERS
   [List 3-5 factors driving current regime]

4. HISTORICAL ANALOG
   [Most similar historical period and what happened next]

5. REGIME OUTLOOK
   - Expected duration: [Days/Weeks/Months]
   - Likely next regime: [Regime name]
   - Transition triggers: [What to watch]

6. STRATEGY IMPLICATIONS
   - Recommended strategies: [List]
   - Strategies to avoid: [List]
   - Position sizing adjustment: [Increase/Decrease/Maintain]

Output as JSON.
"""
```

## Advanced Techniques

### Self-Consistency

Self-consistency improves reliability by generating multiple responses and aggregating them.

```python
# Example: Self-consistency for trading signals
import json
from typing import List, Dict
import asyncio

async def self_consistent_analysis(
    prompt: str,
    llm_client,
    num_samples: int = 5,
    temperature: float = 0.7
) -> Dict:
    """
    Generate multiple analyses and aggregate for consistency.

    Args:
        prompt: The analysis prompt
        llm_client: LLM API client
        num_samples: Number of independent samples
        temperature: Sampling temperature (higher = more diverse)

    Returns:
        Aggregated analysis with confidence
    """
    # Generate multiple samples
    tasks = [
        llm_client.complete(prompt, temperature=temperature)
        for _ in range(num_samples)
    ]
    responses = await asyncio.gather(*tasks)

    # Parse responses
    analyses = []
    for response in responses:
        try:
            analysis = json.loads(response)
            analyses.append(analysis)
        except json.JSONDecodeError:
            continue

    if not analyses:
        return {"error": "No valid responses"}

    # Aggregate results
    signals = [a.get("signal", "HOLD") for a in analyses]
    confidences = [a.get("confidence", 50) for a in analyses]

    # Majority vote for signal
    signal_counts = {}
    for signal in signals:
        signal_counts[signal] = signal_counts.get(signal, 0) + 1

    majority_signal = max(signal_counts, key=signal_counts.get)
    agreement_ratio = signal_counts[majority_signal] / len(signals)

    # Adjust confidence based on agreement
    avg_confidence = sum(confidences) / len(confidences)
    final_confidence = avg_confidence * agreement_ratio

    return {
        "signal": majority_signal,
        "confidence": final_confidence,
        "agreement_ratio": agreement_ratio,
        "sample_count": len(analyses),
        "individual_signals": signals,
        "reasoning_samples": [a.get("reasoning", "") for a in analyses[:3]]
    }
```

### ReAct (Reasoning + Acting)

ReAct combines reasoning with actions, enabling multi-step analysis with tool use.

```python
# Example: ReAct pattern for market research
REACT_TEMPLATE = """
You are a trading analyst who can research and analyze markets.

Available tools:
- get_price(symbol): Get current price and daily change
- get_news(symbol, days): Get recent news headlines
- get_technicals(symbol): Get technical indicators
- get_fundamentals(symbol): Get fundamental metrics
- calculate_position_size(symbol, risk_pct): Calculate position size

Task: {task}

Use the following format:

Thought: [Your reasoning about what to do next]
Action: [Tool name and parameters]
Observation: [Result from the tool]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to provide my final answer
Final Answer: [Your complete analysis and recommendation]

Begin!

Thought:
"""

# Example implementation
class ReActTrader:
    def __init__(self, llm_client, tools: Dict):
        self.llm = llm_client
        self.tools = tools

    async def analyze(self, task: str, max_steps: int = 10) -> str:
        prompt = REACT_TEMPLATE.format(task=task)

        for step in range(max_steps):
            response = await self.llm.complete(prompt)

            if "Final Answer:" in response:
                return response.split("Final Answer:")[-1].strip()

            # Parse action
            if "Action:" in response:
                action_line = response.split("Action:")[-1].split("\n")[0]
                tool_name, params = self._parse_action(action_line)

                # Execute tool
                if tool_name in self.tools:
                    result = await self.tools[tool_name](**params)
                    prompt += f"{response}\nObservation: {result}\nThought:"
                else:
                    prompt += f"{response}\nObservation: Tool not found\nThought:"
            else:
                prompt += f"{response}\nThought:"

        return "Analysis incomplete - max steps reached"

    def _parse_action(self, action_line: str):
        # Parse tool name and parameters
        # e.g., "get_price(AAPL)" -> ("get_price", {"symbol": "AAPL"})
        # Implementation depends on your tool format
        pass
```

### Tree-of-Thought

Tree-of-Thought explores multiple reasoning paths for complex strategic decisions.

```python
# Example: Tree-of-Thought for portfolio allocation
TREE_OF_THOUGHT_TEMPLATE = """
You are optimizing a portfolio allocation decision.

SCENARIO:
{scenario_description}

CONSTRAINTS:
- Max position size: {max_position}%
- Max sector concentration: {max_sector}%
- Target volatility: {target_vol}%
- Rebalancing frequency: {rebalance_freq}

Explore THREE different allocation strategies:

STRATEGY A: Conservative
- Philosophy: [Describe approach]
- Allocation: [Detail positions]
- Expected return: [Estimate]
- Expected risk: [Estimate]
- Key assumption: [What must be true for this to work]

STRATEGY B: Moderate
- Philosophy: [Describe approach]
- Allocation: [Detail positions]
- Expected return: [Estimate]
- Expected risk: [Estimate]
- Key assumption: [What must be true for this to work]

STRATEGY C: Aggressive
- Philosophy: [Describe approach]
- Allocation: [Detail positions]
- Expected return: [Estimate]
- Expected risk: [Estimate]
- Key assumption: [What must be true for this to work]

EVALUATION:
Compare strategies on:
1. Risk-adjusted return (Sharpe ratio)
2. Downside protection
3. Robustness to assumption violations
4. Implementation complexity

RECOMMENDATION:
[Select best strategy with detailed justification]

HYBRID OPTION:
[Propose a combination taking best elements from each]
"""
```

### Prompt Chaining

Prompt chaining breaks complex analysis into sequential specialized prompts.

```python
# Example: Prompt chain for trading decision
class TradingPromptChain:
    """
    Multi-step prompt chain for comprehensive trading analysis.

    Chain: Data Analysis → Risk Assessment → Signal Generation → Execution Plan
    """

    PROMPTS = {
        "data_analysis": """
Analyze the following market data for {symbol}:

Price Data: {price_data}
Volume Data: {volume_data}
Technical Indicators: {technicals}

Provide:
1. Trend analysis (short/medium/long term)
2. Key support/resistance levels
3. Volume analysis
4. Anomalies or notable patterns

Output as JSON with keys: trend, levels, volume, patterns
""",

        "risk_assessment": """
Based on the following data analysis:
{data_analysis_result}

Current portfolio context:
{portfolio_context}

Assess the risk of taking a position in {symbol}:
1. Position-specific risks
2. Portfolio impact
3. Market timing risk
4. Maximum recommended position size

Output as JSON with keys: risks, portfolio_impact, timing_risk, max_size
""",

        "signal_generation": """
Given:
- Data analysis: {data_analysis_result}
- Risk assessment: {risk_assessment_result}
- Market regime: {market_regime}

Generate a trading signal for {symbol}:
1. Signal direction (LONG/SHORT/FLAT)
2. Confidence level (0-100)
3. Entry conditions
4. Supporting factors
5. Contradicting factors

Output as JSON with keys: signal, confidence, entry, pros, cons
""",

        "execution_plan": """
Trading signal: {signal_result}
Risk parameters: {risk_assessment_result}
Current price: {current_price}

Create detailed execution plan:
1. Entry method (market/limit/scaled)
2. Position size calculation
3. Stop loss placement
4. Take profit targets
5. Position management rules
6. Exit triggers

Output as JSON with keys: entry, size, stop_loss, take_profit, management, exit_triggers
"""
    }

    async def execute(self, symbol: str, market_data: Dict, llm_client) -> Dict:
        """Execute the full prompt chain."""
        results = {}

        # Step 1: Data Analysis
        prompt1 = self.PROMPTS["data_analysis"].format(
            symbol=symbol,
            price_data=market_data["price"],
            volume_data=market_data["volume"],
            technicals=market_data["technicals"]
        )
        results["data_analysis"] = await llm_client.complete(prompt1)

        # Step 2: Risk Assessment
        prompt2 = self.PROMPTS["risk_assessment"].format(
            data_analysis_result=results["data_analysis"],
            portfolio_context=market_data["portfolio"],
            symbol=symbol
        )
        results["risk_assessment"] = await llm_client.complete(prompt2)

        # Step 3: Signal Generation
        prompt3 = self.PROMPTS["signal_generation"].format(
            data_analysis_result=results["data_analysis"],
            risk_assessment_result=results["risk_assessment"],
            market_regime=market_data["regime"],
            symbol=symbol
        )
        results["signal"] = await llm_client.complete(prompt3)

        # Step 4: Execution Plan
        prompt4 = self.PROMPTS["execution_plan"].format(
            signal_result=results["signal"],
            risk_assessment_result=results["risk_assessment"],
            current_price=market_data["price"]["current"]
        )
        results["execution_plan"] = await llm_client.complete(prompt4)

        return results
```

## Practical Examples

### 01: Financial Sentiment Analysis

```python
# python/01_sentiment_analysis.py

import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class Sentiment(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment: Sentiment
    confidence: float
    entities: Dict[str, Sentiment]
    reasoning: str
    trading_signal: str

class FinancialSentimentAnalyzer:
    """
    Prompt engineering-based financial sentiment analyzer.

    Uses carefully crafted prompts to extract trading-relevant
    sentiment from financial text.
    """

    # Few-shot examples for consistent output
    EXAMPLES = [
        {
            "text": "Apple Inc. reported quarterly earnings that exceeded Wall Street expectations, with revenue growing 8% year-over-year. CEO Tim Cook highlighted strong iPhone demand in emerging markets.",
            "analysis": {
                "sentiment": "BULLISH",
                "confidence": 88,
                "entities": {"Apple Inc.": "BULLISH", "iPhone": "BULLISH"},
                "reasoning": "Earnings beat with strong growth indicates positive momentum. CEO commentary is optimistic.",
                "trading_signal": "BUY on pullback"
            }
        },
        {
            "text": "The Federal Reserve signaled it may keep interest rates higher for longer than markets expected, citing persistent inflation concerns.",
            "analysis": {
                "sentiment": "BEARISH",
                "confidence": 75,
                "entities": {"Federal Reserve": "BEARISH", "interest rates": "BEARISH"},
                "reasoning": "Hawkish Fed stance typically pressures equity valuations, especially growth stocks.",
                "trading_signal": "REDUCE equity exposure"
            }
        },
        {
            "text": "Tesla announced a 2% price cut on Model 3 vehicles in China, matching competitor discounts.",
            "analysis": {
                "sentiment": "NEUTRAL",
                "confidence": 65,
                "entities": {"Tesla": "NEUTRAL", "Model 3": "NEUTRAL"},
                "reasoning": "Price cuts maintain competitiveness but signal margin pressure. Market likely expected this.",
                "trading_signal": "HOLD - wait for delivery data"
            }
        }
    ]

    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_template = self._build_prompt_template()

    def _build_prompt_template(self) -> str:
        """Build the few-shot prompt template."""
        examples_text = ""
        for i, ex in enumerate(self.EXAMPLES, 1):
            examples_text += f"""
Example {i}:
Text: "{ex['text']}"
Analysis: {json.dumps(ex['analysis'], indent=2)}

"""

        return f"""You are an expert financial analyst specializing in sentiment analysis for trading.
Analyze financial text and extract actionable trading insights.

{examples_text}

Now analyze the following text:
Text: "{{text}}"

Provide your analysis in the exact JSON format shown in the examples.
Only respond with valid JSON, no additional text.

Analysis:"""

    async def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of financial text.

        Args:
            text: Financial news, report, or social media post

        Returns:
            SentimentResult with sentiment classification and trading signal
        """
        prompt = self.prompt_template.format(text=text)

        response = await self.llm.complete(
            prompt,
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=500
        )

        try:
            analysis = json.loads(response.strip())
            return SentimentResult(
                sentiment=Sentiment(analysis["sentiment"]),
                confidence=analysis["confidence"],
                entities={k: Sentiment(v) for k, v in analysis["entities"].items()},
                reasoning=analysis["reasoning"],
                trading_signal=analysis["trading_signal"]
            )
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback for parsing errors
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0,
                entities={},
                reasoning=f"Parsing error: {e}",
                trading_signal="HOLD - analysis failed"
            )

    async def analyze_batch(
        self,
        texts: List[str],
        aggregate: bool = True
    ) -> Dict:
        """
        Analyze multiple texts and optionally aggregate sentiment.

        Args:
            texts: List of financial texts
            aggregate: Whether to aggregate into overall sentiment

        Returns:
            Individual results and optional aggregate
        """
        results = []
        for text in texts:
            result = await self.analyze(text)
            results.append(result)

        if not aggregate:
            return {"individual": results}

        # Aggregate sentiment with confidence weighting
        sentiment_scores = {
            Sentiment.BULLISH: 1,
            Sentiment.NEUTRAL: 0,
            Sentiment.BEARISH: -1
        }

        weighted_sum = sum(
            sentiment_scores[r.sentiment] * r.confidence
            for r in results
        )
        total_confidence = sum(r.confidence for r in results)

        if total_confidence > 0:
            avg_score = weighted_sum / total_confidence
            if avg_score > 0.3:
                aggregate_sentiment = Sentiment.BULLISH
            elif avg_score < -0.3:
                aggregate_sentiment = Sentiment.BEARISH
            else:
                aggregate_sentiment = Sentiment.NEUTRAL
        else:
            aggregate_sentiment = Sentiment.NEUTRAL
            avg_score = 0

        return {
            "individual": results,
            "aggregate": {
                "sentiment": aggregate_sentiment,
                "score": avg_score,
                "confidence": total_confidence / len(results) if results else 0,
                "sample_count": len(results)
            }
        }


# Example usage
async def main():
    # Mock LLM client for demonstration
    class MockLLM:
        async def complete(self, prompt, **kwargs):
            # In production, this calls actual LLM API
            return json.dumps({
                "sentiment": "BULLISH",
                "confidence": 82,
                "entities": {"NVIDIA": "BULLISH"},
                "reasoning": "Strong earnings beat with positive AI demand outlook.",
                "trading_signal": "BUY"
            })

    analyzer = FinancialSentimentAnalyzer(MockLLM())

    news = """
    NVIDIA reported Q3 earnings that crushed expectations, with data center
    revenue up 279% year-over-year. CEO Jensen Huang said AI demand remains
    'incredible' and raised guidance for next quarter.
    """

    result = await analyzer.analyze(news)

    print(f"Sentiment: {result.sentiment.value}")
    print(f"Confidence: {result.confidence}%")
    print(f"Reasoning: {result.reasoning}")
    print(f"Signal: {result.trading_signal}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 02: Trading Signal Generation

```python
# python/02_signal_generator.py

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

class SignalStrength(Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"

@dataclass
class TradingSignal:
    """Complete trading signal with entry/exit plan."""
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    entry_price: Optional[float]
    stop_loss: float
    take_profit: List[float]
    position_size_pct: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

class PromptBasedSignalGenerator:
    """
    Generate trading signals using engineered prompts.

    Combines multiple prompt techniques:
    - Role-based (quant analyst persona)
    - Chain-of-thought (systematic reasoning)
    - Few-shot (output format consistency)
    """

    SYSTEM_PROMPT = """You are a senior quantitative analyst at a systematic trading fund.
Your role is to analyze market data and generate precise, actionable trading signals.

Your analysis style:
- Data-driven and objective
- Always consider risk first
- Provide specific price levels, not vague directions
- Acknowledge uncertainty when confidence is low
- Never recommend trading without a stop-loss"""

    SIGNAL_PROMPT = """
MARKET DATA FOR {symbol}:

Current Price: ${current_price}
24h Change: {change_24h:+.2f}%
7d Change: {change_7d:+.2f}%

Technical Indicators:
- RSI(14): {rsi:.1f}
- MACD: {macd_signal} (histogram: {macd_hist:+.4f})
- 20 SMA: ${sma_20:.2f} (price {sma_20_position})
- 50 SMA: ${sma_50:.2f} (price {sma_50_position})
- 200 SMA: ${sma_200:.2f} (price {sma_200_position})
- Bollinger Bands: ${bb_lower:.2f} - ${bb_upper:.2f}
- ATR(14): ${atr:.2f} ({atr_pct:.2f}% of price)

Volume Analysis:
- Current Volume: {volume:,.0f}
- 20-day Avg Volume: {avg_volume:,.0f}
- Volume Ratio: {volume_ratio:.2f}x

Recent News Sentiment: {news_sentiment}

Market Context:
- Market Regime: {market_regime}
- Sector Performance: {sector_perf}
- Correlation to SPY: {spy_corr:.2f}

Think step by step:

1. TREND ANALYSIS
   - What is the primary trend direction?
   - Are the moving averages aligned?
   - Any divergences between price and indicators?

2. MOMENTUM ASSESSMENT
   - Is momentum confirming or diverging from price?
   - RSI overbought (>70) or oversold (<30)?
   - MACD signal strength?

3. VOLUME CONFIRMATION
   - Is volume supporting the move?
   - Any unusual volume patterns?

4. RISK PARAMETERS
   - Where is the logical stop-loss level?
   - What is the risk/reward ratio?
   - How does volatility affect position sizing?

5. FINAL SIGNAL
   Based on the above analysis, provide your trading signal.

Output your signal in this exact JSON format:
{{
    "direction": "LONG" | "SHORT" | "FLAT",
    "strength": "STRONG" | "MODERATE" | "WEAK",
    "confidence": <0-100>,
    "entry": {{
        "type": "MARKET" | "LIMIT",
        "price": <price if LIMIT, null if MARKET>
    }},
    "stop_loss": <price>,
    "take_profit": [<price1>, <price2>],
    "position_size_pct": <0-100>,
    "reasoning": "<2-3 sentence summary>"
}}

Your analysis and signal:
"""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def generate_signal(
        self,
        symbol: str,
        market_data: Dict
    ) -> TradingSignal:
        """
        Generate trading signal from market data.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "AAPL")
            market_data: Dict containing price, technicals, etc.

        Returns:
            TradingSignal with complete entry/exit plan
        """
        # Calculate derived metrics
        current_price = market_data["price"]["current"]
        sma_20 = market_data["technicals"]["sma_20"]
        sma_50 = market_data["technicals"]["sma_50"]
        sma_200 = market_data["technicals"]["sma_200"]

        prompt = self.SIGNAL_PROMPT.format(
            symbol=symbol,
            current_price=current_price,
            change_24h=market_data["price"]["change_24h"],
            change_7d=market_data["price"]["change_7d"],
            rsi=market_data["technicals"]["rsi"],
            macd_signal="BULLISH" if market_data["technicals"]["macd"] > 0 else "BEARISH",
            macd_hist=market_data["technicals"]["macd_hist"],
            sma_20=sma_20,
            sma_20_position="above" if current_price > sma_20 else "below",
            sma_50=sma_50,
            sma_50_position="above" if current_price > sma_50 else "below",
            sma_200=sma_200,
            sma_200_position="above" if current_price > sma_200 else "below",
            bb_lower=market_data["technicals"]["bb_lower"],
            bb_upper=market_data["technicals"]["bb_upper"],
            atr=market_data["technicals"]["atr"],
            atr_pct=(market_data["technicals"]["atr"] / current_price) * 100,
            volume=market_data["volume"]["current"],
            avg_volume=market_data["volume"]["average"],
            volume_ratio=market_data["volume"]["current"] / market_data["volume"]["average"],
            news_sentiment=market_data.get("sentiment", "NEUTRAL"),
            market_regime=market_data.get("regime", "UNKNOWN"),
            sector_perf=market_data.get("sector", "N/A"),
            spy_corr=market_data.get("spy_correlation", 0.5)
        )

        # Combine system prompt with signal prompt
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"

        response = await self.llm.complete(
            full_prompt,
            temperature=0.2,
            max_tokens=1000
        )

        # Extract JSON from response
        signal_data = self._parse_signal_response(response)

        return TradingSignal(
            symbol=symbol,
            direction=SignalDirection(signal_data["direction"]),
            strength=SignalStrength(signal_data["strength"]),
            confidence=signal_data["confidence"],
            entry_price=signal_data["entry"].get("price"),
            stop_loss=signal_data["stop_loss"],
            take_profit=signal_data["take_profit"],
            position_size_pct=signal_data["position_size_pct"],
            reasoning=signal_data["reasoning"]
        )

    def _parse_signal_response(self, response: str) -> Dict:
        """Extract JSON signal from LLM response."""
        # Find JSON block in response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")

        json_str = response[start_idx:end_idx]
        return json.loads(json_str)


# Example usage and data fetching
async def main():
    # Mock market data (in production, fetch from API)
    market_data = {
        "price": {
            "current": 45250.00,
            "change_24h": 2.3,
            "change_7d": -1.5
        },
        "technicals": {
            "rsi": 58.5,
            "macd": 120.5,
            "macd_hist": 45.2,
            "sma_20": 44800.0,
            "sma_50": 43500.0,
            "sma_200": 41000.0,
            "bb_lower": 43200.0,
            "bb_upper": 47000.0,
            "atr": 1200.0
        },
        "volume": {
            "current": 25000000000,
            "average": 20000000000
        },
        "sentiment": "BULLISH",
        "regime": "RISK_ON_TRENDING",
        "sector": "Crypto +3.5%",
        "spy_correlation": 0.65
    }

    # Mock LLM
    class MockLLM:
        async def complete(self, prompt, **kwargs):
            return json.dumps({
                "direction": "LONG",
                "strength": "MODERATE",
                "confidence": 72,
                "entry": {"type": "LIMIT", "price": 45000},
                "stop_loss": 43500,
                "take_profit": [47000, 49000],
                "position_size_pct": 5,
                "reasoning": "Bullish trend with price above all major SMAs. RSI neutral with room to run. Volume confirms momentum."
            })

    generator = PromptBasedSignalGenerator(MockLLM())
    signal = await generator.generate_signal("BTCUSDT", market_data)

    print(f"Symbol: {signal.symbol}")
    print(f"Direction: {signal.direction.value} ({signal.strength.value})")
    print(f"Confidence: {signal.confidence}%")
    print(f"Entry: ${signal.entry_price:,.2f}" if signal.entry_price else "Market order")
    print(f"Stop Loss: ${signal.stop_loss:,.2f}")
    print(f"Take Profit: {[f'${tp:,.2f}' for tp in signal.take_profit]}")
    print(f"Position Size: {signal.position_size_pct}%")
    print(f"Reasoning: {signal.reasoning}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 03: Market Regime Detection

```python
# python/03_regime_detection.py

import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class MarketRegime(Enum):
    RISK_ON_TRENDING = "RISK_ON_TRENDING"  # Strong uptrend, low volatility
    RISK_ON_VOLATILE = "RISK_ON_VOLATILE"  # Uptrend with high volatility
    RISK_OFF_TRENDING = "RISK_OFF_TRENDING"  # Strong downtrend
    RISK_OFF_PANIC = "RISK_OFF_PANIC"  # Crash mode
    RANGING = "RANGING"  # Sideways consolidation
    TRANSITIONAL = "TRANSITIONAL"  # Regime change in progress

@dataclass
class RegimeAnalysis:
    """Market regime analysis result."""
    regime: MarketRegime
    confidence: float
    key_drivers: List[str]
    recommended_strategies: List[str]
    avoid_strategies: List[str]
    position_sizing_adj: str
    next_regime_probability: Dict[str, float]
    reasoning: str

class MarketRegimeDetector:
    """
    Detect market regimes using LLM analysis.

    Combines quantitative indicators with LLM reasoning
    for robust regime classification.
    """

    REGIME_PROMPT = """You are a market regime analyst at a macro hedge fund.
Your task is to classify the current market regime and recommend appropriate strategies.

CURRENT MARKET DATA:

Equity Markets:
- S&P 500: {sp500_price} ({sp500_change:+.2f}% MTD, {sp500_ytd:+.2f}% YTD)
- NASDAQ: {nasdaq_price} ({nasdaq_change:+.2f}% MTD)
- Russell 2000: {russell_price} ({russell_change:+.2f}% MTD)

Volatility:
- VIX: {vix} (20-day avg: {vix_avg})
- VIX Term Structure: {vix_term_structure}
- Realized Vol (20d): {realized_vol}%

Fixed Income:
- 10Y Treasury Yield: {treasury_10y}%
- 2Y Treasury Yield: {treasury_2y}%
- Yield Curve (10Y-2Y): {yield_curve:+.0f}bps
- Credit Spreads (IG): {credit_spread}bps

Currencies & Commodities:
- Dollar Index (DXY): {dxy} ({dxy_change:+.2f}% MTD)
- Gold: ${gold} ({gold_change:+.2f}% MTD)
- Crude Oil: ${oil} ({oil_change:+.2f}% MTD)

Market Breadth:
- NYSE Advance/Decline: {adv_dec_ratio}
- % Stocks Above 200 SMA: {pct_above_200}%
- New 52-Week Highs - Lows: {high_low_diff}

Flows & Sentiment:
- Fund Flows (Weekly): {fund_flows}
- Put/Call Ratio: {put_call}
- AAII Bull/Bear Spread: {aaii_spread:+.1f}%

Analyze step by step:

1. TREND ASSESSMENT
   - Is the market in an uptrend, downtrend, or ranging?
   - Are major indices confirming each other?

2. VOLATILITY ASSESSMENT
   - Is volatility elevated, normal, or suppressed?
   - Is the VIX term structure in contango or backwardation?

3. RISK APPETITE
   - Are credit spreads widening or tightening?
   - Is there rotation into/out of safe havens?

4. BREADTH & PARTICIPATION
   - Is the rally/selloff broad-based or narrow?
   - Are small caps confirming large cap moves?

5. REGIME CLASSIFICATION
   Based on the above, classify the regime:
   - RISK_ON_TRENDING: Strong uptrend, low vol, broad participation
   - RISK_ON_VOLATILE: Uptrend with high volatility, corrections normal
   - RISK_OFF_TRENDING: Sustained downtrend, rising vol, defensive rotation
   - RISK_OFF_PANIC: Crash mode, extreme vol, indiscriminate selling
   - RANGING: Sideways consolidation, low conviction
   - TRANSITIONAL: Signs of regime change, mixed signals

Output in JSON format:
{{
    "regime": "<REGIME_TYPE>",
    "confidence": <0-100>,
    "key_drivers": ["<driver1>", "<driver2>", "<driver3>"],
    "recommended_strategies": ["<strategy1>", "<strategy2>"],
    "avoid_strategies": ["<strategy1>", "<strategy2>"],
    "position_sizing": "INCREASE" | "MAINTAIN" | "REDUCE",
    "next_regime_probabilities": {{
        "<regime1>": <probability>,
        "<regime2>": <probability>
    }},
    "reasoning": "<2-3 sentence summary>"
}}

Your analysis:
"""

    def __init__(self, llm_client):
        self.llm = llm_client

    async def detect_regime(self, market_data: Dict) -> RegimeAnalysis:
        """
        Detect current market regime.

        Args:
            market_data: Comprehensive market data dict

        Returns:
            RegimeAnalysis with classification and recommendations
        """
        prompt = self.REGIME_PROMPT.format(**market_data)

        response = await self.llm.complete(
            prompt,
            temperature=0.3,
            max_tokens=1000
        )

        result = self._parse_response(response)

        return RegimeAnalysis(
            regime=MarketRegime(result["regime"]),
            confidence=result["confidence"],
            key_drivers=result["key_drivers"],
            recommended_strategies=result["recommended_strategies"],
            avoid_strategies=result["avoid_strategies"],
            position_sizing_adj=result["position_sizing"],
            next_regime_probability=result.get("next_regime_probabilities", {}),
            reasoning=result["reasoning"]
        )

    def _parse_response(self, response: str) -> Dict:
        """Extract JSON from LLM response."""
        start = response.find('{')
        end = response.rfind('}') + 1
        return json.loads(response[start:end])


# Regime-based strategy selector
class RegimeStrategySelector:
    """Select and adjust strategies based on detected regime."""

    REGIME_STRATEGIES = {
        MarketRegime.RISK_ON_TRENDING: {
            "preferred": ["trend_following", "momentum", "buy_dips"],
            "position_multiplier": 1.2,
            "stop_loss_multiplier": 1.5,  # Wider stops in trending markets
            "sectors": ["technology", "consumer_discretionary", "industrials"]
        },
        MarketRegime.RISK_ON_VOLATILE: {
            "preferred": ["mean_reversion", "volatility_selling", "range_trading"],
            "position_multiplier": 0.8,
            "stop_loss_multiplier": 2.0,  # Account for volatility
            "sectors": ["technology", "healthcare"]
        },
        MarketRegime.RISK_OFF_TRENDING: {
            "preferred": ["short_selling", "put_buying", "defensive_rotation"],
            "position_multiplier": 0.6,
            "stop_loss_multiplier": 1.0,
            "sectors": ["utilities", "consumer_staples", "healthcare"]
        },
        MarketRegime.RISK_OFF_PANIC: {
            "preferred": ["cash", "tail_hedges", "volatility_buying"],
            "position_multiplier": 0.3,
            "stop_loss_multiplier": 0.5,  # Tight stops, quick exits
            "sectors": ["cash", "treasuries", "gold"]
        },
        MarketRegime.RANGING: {
            "preferred": ["mean_reversion", "premium_selling", "pairs_trading"],
            "position_multiplier": 0.7,
            "stop_loss_multiplier": 1.2,
            "sectors": ["high_dividend", "value"]
        },
        MarketRegime.TRANSITIONAL: {
            "preferred": ["reduce_exposure", "straddles", "diversification"],
            "position_multiplier": 0.5,
            "stop_loss_multiplier": 0.8,
            "sectors": ["balanced"]
        }
    }

    def get_strategy_params(self, regime: MarketRegime) -> Dict:
        """Get strategy parameters for given regime."""
        return self.REGIME_STRATEGIES.get(
            regime,
            self.REGIME_STRATEGIES[MarketRegime.RANGING]
        )


# Example usage
async def main():
    # Sample market data
    market_data = {
        "sp500_price": 4785.50,
        "sp500_change": 2.5,
        "sp500_ytd": 8.2,
        "nasdaq_price": 15050.25,
        "nasdaq_change": 3.1,
        "russell_price": 2025.30,
        "russell_change": 1.8,
        "vix": 14.5,
        "vix_avg": 16.2,
        "vix_term_structure": "Contango",
        "realized_vol": 12.3,
        "treasury_10y": 4.25,
        "treasury_2y": 4.65,
        "yield_curve": -40,
        "credit_spread": 110,
        "dxy": 104.2,
        "dxy_change": 0.5,
        "gold": 2045,
        "gold_change": 1.2,
        "oil": 78.50,
        "oil_change": -2.1,
        "adv_dec_ratio": 1.8,
        "pct_above_200": 65,
        "high_low_diff": 150,
        "fund_flows": "$15B inflow",
        "put_call": 0.85,
        "aaii_spread": 12.5
    }

    # Mock LLM
    class MockLLM:
        async def complete(self, prompt, **kwargs):
            return json.dumps({
                "regime": "RISK_ON_TRENDING",
                "confidence": 78,
                "key_drivers": [
                    "Strong equity momentum across indices",
                    "Low VIX with contango term structure",
                    "Positive fund flows"
                ],
                "recommended_strategies": [
                    "Trend following on major indices",
                    "Buy-the-dip in quality tech"
                ],
                "avoid_strategies": [
                    "Short selling",
                    "Volatility buying"
                ],
                "position_sizing": "INCREASE",
                "next_regime_probabilities": {
                    "RISK_ON_TRENDING": 0.65,
                    "RISK_ON_VOLATILE": 0.25,
                    "TRANSITIONAL": 0.10
                },
                "reasoning": "Market shows classic risk-on characteristics with broad participation, low volatility, and positive sentiment. Yield curve inversion is a concern but not yet impacting equities."
            })

    detector = MarketRegimeDetector(MockLLM())
    regime_analysis = await detector.detect_regime(market_data)

    print(f"Detected Regime: {regime_analysis.regime.value}")
    print(f"Confidence: {regime_analysis.confidence}%")
    print(f"\nKey Drivers:")
    for driver in regime_analysis.key_drivers:
        print(f"  - {driver}")
    print(f"\nRecommended Strategies: {regime_analysis.recommended_strategies}")
    print(f"Avoid: {regime_analysis.avoid_strategies}")
    print(f"Position Sizing: {regime_analysis.position_sizing_adj}")
    print(f"\nReasoning: {regime_analysis.reasoning}")

    # Get strategy parameters
    selector = RegimeStrategySelector()
    params = selector.get_strategy_params(regime_analysis.regime)
    print(f"\nStrategy Parameters:")
    print(f"  Position Multiplier: {params['position_multiplier']}x")
    print(f"  Stop Loss Multiplier: {params['stop_loss_multiplier']}x")
    print(f"  Preferred Sectors: {params['sectors']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 04: Backtesting with LLM Signals

```python
# python/04_backtest.py

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

@dataclass
class BacktestConfig:
    """Configuration for LLM signal backtesting."""
    initial_capital: float = 100000
    max_position_pct: float = 10.0  # Max 10% per position
    transaction_cost_bps: float = 10  # 10 basis points
    slippage_bps: float = 5
    confidence_threshold: float = 60  # Min confidence to trade
    signal_decay_hours: float = 24
    max_concurrent_positions: int = 5

@dataclass
class BacktestResult:
    """Results from backtesting."""
    returns: pd.Series
    equity_curve: pd.Series
    trades: List[Dict]
    metrics: Dict[str, float]
    signal_stats: Dict[str, any]

class LLMSignalBacktester:
    """
    Backtest trading signals generated from LLM analysis.

    Handles the unique characteristics of LLM-derived signals:
    - Variable confidence levels
    - Reasoning-based position sizing
    - Signal decay over time
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResult:
        """
        Run backtest on LLM-generated signals.

        Args:
            signals: DataFrame with columns [timestamp, symbol, direction,
                     confidence, stop_loss, take_profit, reasoning]
            prices: DataFrame with OHLCV data, multi-indexed by (timestamp, symbol)
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResult with performance metrics
        """
        # Filter to date range
        if start_date:
            signals = signals[signals['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            signals = signals[signals['timestamp'] <= pd.to_datetime(end_date)]

        # Sort signals by timestamp
        signals = signals.sort_values('timestamp')

        # Initialize portfolio
        capital = self.config.initial_capital
        positions: Dict[str, Dict] = {}
        trades: List[Dict] = []
        equity_curve = []

        # Get all unique timestamps from prices
        all_dates = prices.index.get_level_values('timestamp').unique().sort_values()

        for date in all_dates:
            # Check for exit conditions on existing positions
            for symbol, pos in list(positions.items()):
                if symbol in prices.xs(date, level='timestamp').index:
                    current_price = prices.loc[(date, symbol), 'close']

                    # Check stop loss
                    if pos['direction'] == 'LONG' and current_price <= pos['stop_loss']:
                        capital, trade = self._close_position(
                            positions, symbol, current_price, date,
                            'STOP_LOSS', capital
                        )
                        trades.append(trade)
                    elif pos['direction'] == 'SHORT' and current_price >= pos['stop_loss']:
                        capital, trade = self._close_position(
                            positions, symbol, current_price, date,
                            'STOP_LOSS', capital
                        )
                        trades.append(trade)

                    # Check take profit
                    elif pos['direction'] == 'LONG' and current_price >= pos['take_profit'][0]:
                        capital, trade = self._close_position(
                            positions, symbol, current_price, date,
                            'TAKE_PROFIT', capital
                        )
                        trades.append(trade)
                    elif pos['direction'] == 'SHORT' and current_price <= pos['take_profit'][0]:
                        capital, trade = self._close_position(
                            positions, symbol, current_price, date,
                            'TAKE_PROFIT', capital
                        )
                        trades.append(trade)

                    # Check signal decay
                    hours_since_entry = (date - pos['entry_time']).total_seconds() / 3600
                    if hours_since_entry > self.config.signal_decay_hours:
                        capital, trade = self._close_position(
                            positions, symbol, current_price, date,
                            'SIGNAL_DECAY', capital
                        )
                        trades.append(trade)

            # Check for new signals on this date
            day_signals = signals[signals['timestamp'].dt.date == date.date()]

            for _, signal in day_signals.iterrows():
                # Skip if below confidence threshold
                if signal['confidence'] < self.config.confidence_threshold:
                    continue

                # Skip if already have position in symbol
                if signal['symbol'] in positions:
                    continue

                # Skip if at max positions
                if len(positions) >= self.config.max_concurrent_positions:
                    continue

                # Skip FLAT signals
                if signal['direction'] == 'FLAT':
                    continue

                # Get current price
                if signal['symbol'] not in prices.xs(date, level='timestamp').index:
                    continue

                current_price = prices.loc[(date, signal['symbol']), 'close']

                # Calculate position size based on confidence
                confidence_factor = signal['confidence'] / 100
                base_size = self.config.max_position_pct / 100
                position_size_pct = base_size * confidence_factor
                position_value = capital * position_size_pct

                # Apply transaction costs
                cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
                position_value *= (1 - cost_bps / 10000)

                # Calculate shares
                shares = position_value / current_price

                # Open position
                positions[signal['symbol']] = {
                    'direction': signal['direction'],
                    'shares': shares if signal['direction'] == 'LONG' else -shares,
                    'entry_price': current_price,
                    'entry_time': date,
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'] if isinstance(signal['take_profit'], list) else [signal['take_profit']],
                    'confidence': signal['confidence'],
                    'reasoning': signal.get('reasoning', '')
                }

                # Deduct from capital
                capital -= position_value

                trades.append({
                    'timestamp': date,
                    'symbol': signal['symbol'],
                    'action': 'OPEN',
                    'direction': signal['direction'],
                    'shares': shares,
                    'price': current_price,
                    'value': position_value,
                    'confidence': signal['confidence'],
                    'reasoning': signal.get('reasoning', '')
                })

            # Calculate portfolio value
            portfolio_value = capital
            for symbol, pos in positions.items():
                if symbol in prices.xs(date, level='timestamp').index:
                    current_price = prices.loc[(date, symbol), 'close']
                    position_value = abs(pos['shares']) * current_price
                    if pos['direction'] == 'LONG':
                        pnl = (current_price - pos['entry_price']) * abs(pos['shares'])
                    else:
                        pnl = (pos['entry_price'] - current_price) * abs(pos['shares'])
                    portfolio_value += position_value + pnl

            equity_curve.append({
                'timestamp': date,
                'portfolio_value': portfolio_value,
                'num_positions': len(positions)
            })

        # Create results
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        returns = equity_df['portfolio_value'].pct_change().dropna()

        metrics = self._calculate_metrics(returns, trades, equity_df)
        signal_stats = self._calculate_signal_stats(trades)

        return BacktestResult(
            returns=returns,
            equity_curve=equity_df['portfolio_value'],
            trades=trades,
            metrics=metrics,
            signal_stats=signal_stats
        )

    def _close_position(
        self,
        positions: Dict,
        symbol: str,
        price: float,
        timestamp,
        reason: str,
        capital: float
    ) -> tuple:
        """Close a position and return updated capital and trade record."""
        pos = positions.pop(symbol)

        # Calculate PnL
        if pos['direction'] == 'LONG':
            pnl = (price - pos['entry_price']) * abs(pos['shares'])
        else:
            pnl = (pos['entry_price'] - price) * abs(pos['shares'])

        # Apply exit costs
        exit_value = abs(pos['shares']) * price
        cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        exit_value *= (1 - cost_bps / 10000)

        capital += exit_value + pnl

        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'CLOSE',
            'reason': reason,
            'direction': pos['direction'],
            'shares': abs(pos['shares']),
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'pnl_pct': (price / pos['entry_price'] - 1) * 100 * (1 if pos['direction'] == 'LONG' else -1),
            'hold_time_hours': (timestamp - pos['entry_time']).total_seconds() / 3600,
            'confidence': pos['confidence']
        }

        return capital, trade

    def _calculate_metrics(
        self,
        returns: pd.Series,
        trades: List[Dict],
        equity_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate backtest performance metrics."""
        if returns.empty:
            return {}

        ann_factor = 252

        total_return = (equity_df['portfolio_value'].iloc[-1] /
                       self.config.initial_capital - 1)

        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ann_factor)
        sharpe = ann_return / volatility if volatility > 0 else 0

        # Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Trade statistics
        closed_trades = [t for t in trades if t['action'] == 'CLOSE']
        if closed_trades:
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(closed_trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            losing_trades = [t for t in closed_trades if t['pnl'] <= 0]
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win * len(winning_trades) /
                              (avg_loss * len(losing_trades))) if losing_trades and avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0

        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(closed_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

    def _calculate_signal_stats(self, trades: List[Dict]) -> Dict:
        """Calculate statistics about signal quality."""
        closed_trades = [t for t in trades if t['action'] == 'CLOSE']

        if not closed_trades:
            return {}

        # Performance by confidence bucket
        confidence_buckets = {
            'high': [t for t in closed_trades if t['confidence'] >= 80],
            'medium': [t for t in closed_trades if 60 <= t['confidence'] < 80],
            'low': [t for t in closed_trades if t['confidence'] < 60]
        }

        bucket_stats = {}
        for bucket, trades_in_bucket in confidence_buckets.items():
            if trades_in_bucket:
                bucket_stats[bucket] = {
                    'count': len(trades_in_bucket),
                    'win_rate': len([t for t in trades_in_bucket if t['pnl'] > 0]) / len(trades_in_bucket),
                    'avg_pnl_pct': np.mean([t['pnl_pct'] for t in trades_in_bucket])
                }

        # Performance by exit reason
        reason_stats = {}
        for reason in ['STOP_LOSS', 'TAKE_PROFIT', 'SIGNAL_DECAY']:
            reason_trades = [t for t in closed_trades if t.get('reason') == reason]
            if reason_trades:
                reason_stats[reason] = {
                    'count': len(reason_trades),
                    'avg_pnl_pct': np.mean([t['pnl_pct'] for t in reason_trades])
                }

        return {
            'by_confidence': bucket_stats,
            'by_exit_reason': reason_stats,
            'avg_hold_time_hours': np.mean([t['hold_time_hours'] for t in closed_trades])
        }


# Example usage
def run_example_backtest():
    """Run example backtest with synthetic data."""
    np.random.seed(42)

    # Generate synthetic price data
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    price_data = []
    for symbol in symbols:
        base_price = {'AAPL': 150, 'MSFT': 350, 'GOOGL': 140}[symbol]
        prices = base_price * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
        for i, date in enumerate(dates):
            price_data.append({
                'timestamp': date,
                'symbol': symbol,
                'close': prices[i],
                'volume': np.random.randint(1000000, 10000000)
            })

    prices = pd.DataFrame(price_data).set_index(['timestamp', 'symbol'])

    # Generate synthetic LLM signals
    signal_data = []
    signal_dates = np.random.choice(dates, size=30, replace=False)
    for date in sorted(signal_dates):
        symbol = np.random.choice(symbols)
        direction = np.random.choice(['LONG', 'SHORT'])
        confidence = np.random.uniform(50, 95)

        current_price = prices.loc[(date, symbol), 'close']
        if direction == 'LONG':
            stop_loss = current_price * 0.95
            take_profit = [current_price * 1.08, current_price * 1.15]
        else:
            stop_loss = current_price * 1.05
            take_profit = [current_price * 0.92, current_price * 0.85]

        signal_data.append({
            'timestamp': date,
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reasoning': f"LLM analysis suggested {direction} based on momentum"
        })

    signals = pd.DataFrame(signal_data)

    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        max_position_pct=10,
        confidence_threshold=60
    )

    backtester = LLMSignalBacktester(config)
    result = backtester.run_backtest(signals, prices)

    print("=" * 60)
    print("LLM Signal Backtest Results")
    print("=" * 60)
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {result.metrics['total_return']:.2%}")
    print(f"  Annualized Return: {result.metrics['annualized_return']:.2%}")
    print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"  Win Rate: {result.metrics['win_rate']:.2%}")
    print(f"  Profit Factor: {result.metrics['profit_factor']:.2f}")
    print(f"  Number of Trades: {result.metrics['num_trades']}")

    print(f"\nSignal Statistics:")
    print(f"  Avg Hold Time: {result.signal_stats.get('avg_hold_time_hours', 0):.1f} hours")

    if 'by_confidence' in result.signal_stats:
        print(f"\n  Performance by Confidence:")
        for bucket, stats in result.signal_stats['by_confidence'].items():
            print(f"    {bucket.capitalize()}: {stats['count']} trades, "
                  f"{stats['win_rate']:.1%} win rate, {stats['avg_pnl_pct']:.2f}% avg PnL")

    return result


if __name__ == "__main__":
    run_example_backtest()
```

## Rust Implementation

See [rust/](rust/) for complete Rust implementation with async prompt handling.

```
rust/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs              # Main library exports
    ├── prompts.rs          # Prompt templates for all trading tasks
    ├── llm_client.rs       # Async HTTP client for LLM APIs
    ├── sentiment.rs        # Sentiment analysis module
    ├── signals.rs          # Trading signal generation
    ├── regime.rs           # Market regime detection
    ├── backtest.rs         # Backtesting engine with metrics
    ├── data.rs             # Mock data loader with indicators
    └── error.rs            # Error types
```

### Quick Start (Rust)

```bash
cd rust

# Build the library
cargo build

# Run tests
cargo test

# Use the library in your code
# See rust/README.md for usage examples
```

## Python Implementation

See [python/](python/) for complete Python implementation.

```
python/
├── __init__.py
├── sentiment_analysis.py   # Sentiment analyzer
├── signal_generator.py     # Signal generator
├── regime_detection.py     # Regime detector
├── backtest.py             # Backtesting engine
├── prompts/                # Prompt templates
│   ├── __init__.py
│   ├── sentiment.py
│   ├── signals.py
│   └── regime.py
├── llm_client.py           # LLM API client
├── data_loader.py          # Market data utilities
├── requirements.txt        # Dependencies
└── examples/
    ├── 01_sentiment_demo.py
    ├── 02_signal_generation.py
    ├── 03_regime_detection.py
    └── 04_full_backtest.py
```

### Quick Start (Python)

```bash
cd python

# Install dependencies
pip install -r requirements.txt

# Run sentiment analysis
python examples/01_sentiment_demo.py

# Generate signals
python examples/02_signal_generation.py --symbol BTCUSDT

# Run backtest
python examples/04_full_backtest.py --capital 100000
```

## Best Practices

### Prompt Design Guidelines

1. **Be Specific About Output Format**
   ```python
   # Bad: "Analyze this news"
   # Good: "Analyze and respond in JSON with keys: sentiment, confidence, reasoning"
   ```

2. **Include Domain Context**
   ```python
   # Bad: "Is this positive or negative?"
   # Good: "As a financial analyst, classify the trading sentiment..."
   ```

3. **Use Consistent Examples**
   ```python
   # Use 3-5 diverse examples covering edge cases
   # Maintain identical format across all examples
   ```

4. **Constrain Numerical Outputs**
   ```python
   # Bad: "Rate your confidence"
   # Good: "Confidence (integer 0-100):"
   ```

5. **Request Reasoning**
   ```python
   # Always ask for reasoning to verify logic
   # Enables post-hoc analysis and improvement
   ```

### Common Pitfalls to Avoid

1. **Vague Instructions**
   - LLMs interpret ambiguity inconsistently
   - Be explicit about every expected output

2. **No Error Handling**
   - LLMs can output malformed JSON or unexpected values
   - Always validate and provide fallbacks

3. **Over-reliance on Single Response**
   - Use self-consistency for high-stakes decisions
   - Multiple samples reduce variance

4. **Ignoring Context Window**
   - Long prompts may truncate important information
   - Summarize data when necessary

5. **Static Prompts**
   - Market conditions change; prompts should evolve
   - A/B test prompt variations

### Performance Optimization

```python
# Tips for production prompt engineering

1. **Cache Embeddings**
   # Reuse embeddings for similar queries

2. **Batch Requests**
   # Process multiple analyses in single API call when possible

3. **Async Processing**
   # Use async/await for parallel signal generation

4. **Prompt Compression**
   # Remove unnecessary tokens while preserving meaning

5. **Temperature Tuning**
   # Lower (0.1-0.3) for consistent signals
   # Higher (0.5-0.7) for diverse analysis
```

## Resources

### Papers

- [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234) — Comprehensive survey on prompting techniques (2023)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) — Original CoT paper from Google (2022)
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — Self-consistency technique (2023)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — ReAct framework (2023)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) — ToT for deliberate problem solving (2023)
- [FinGPT: Financial LLMs](https://arxiv.org/abs/2306.06031) — Open-source financial LLM (2023)

### Tools & Libraries

| Tool | Description | Link |
|------|-------------|------|
| LangChain | LLM application framework | [langchain.com](https://langchain.com) |
| OpenAI API | GPT-4 and embedding models | [platform.openai.com](https://platform.openai.com) |
| Anthropic API | Claude models | [anthropic.com](https://anthropic.com) |
| Ollama | Run LLMs locally | [ollama.ai](https://ollama.ai) |
| PromptFlow | Prompt engineering toolkit | [Azure ML](https://azure.microsoft.com) |

### Related Chapters

- [Chapter 61: FinGPT Financial LLM](../61_fingpt_financial_llm) — Open-source financial LLM
- [Chapter 66: Chain-of-Thought for Trading](../66_chain_of_thought_trading) — CoT deep dive
- [Chapter 67: LLM Sentiment Analysis](../67_llm_sentiment_analysis) — Sentiment analysis
- [Chapter 62: BloombergGPT](../62_bloomberggpt_trading) — Domain-specific LLM
- [Chapter 65: RAG for Trading](../65_rag_for_trading) — Retrieval-augmented generation

---

## Difficulty Level

**Intermediate to Advanced**

Prerequisites:
- Understanding of LLMs and transformer architecture
- Python programming experience
- Basic trading knowledge (signals, backtesting)
- Familiarity with API integration

## References

1. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.
2. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models."
3. Wang, X., et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning."
4. Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models."
5. Yao, S., et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models."
6. Yang, H., et al. (2023). "FinGPT: Open-Source Financial Large Language Models."
