//! Prompt templates for trading analysis.

/// Sentiment analysis prompts.
pub mod sentiment {
    /// Zero-shot sentiment analysis prompt.
    pub const ZERO_SHOT: &str = r#"Analyze the financial sentiment of this news:

"{text}"

Respond in JSON format:
{{
    "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
    "confidence": 0-100,
    "market_impact": "BULLISH" | "BEARISH" | "NEUTRAL",
    "time_horizon": "SHORT_TERM" | "MEDIUM_TERM" | "LONG_TERM",
    "key_factors": ["factor1", "factor2"],
    "reasoning": "brief explanation"
}}"#;

    /// Few-shot sentiment analysis prompt.
    pub const FEW_SHOT: &str = r#"Analyze financial news sentiment. Here are examples:

Example 1:
News: "Company XYZ beats quarterly earnings by 15%, raises full-year guidance"
Analysis: {{"sentiment": "POSITIVE", "confidence": 90, "market_impact": "BULLISH"}}

Example 2:
News: "CEO announces unexpected resignation amid accounting investigation"
Analysis: {{"sentiment": "NEGATIVE", "confidence": 95, "market_impact": "BEARISH"}}

Example 3:
News: "Company reports results in line with analyst expectations"
Analysis: {{"sentiment": "NEUTRAL", "confidence": 70, "market_impact": "NEUTRAL"}}

Now analyze:
News: "{text}"
Symbol: {symbol}

Respond in JSON format with sentiment, confidence (0-100), market_impact, time_horizon, key_factors, and reasoning."#;

    /// Chain-of-thought sentiment analysis prompt.
    pub const CHAIN_OF_THOUGHT: &str = r#"Analyze this financial news step by step:

News: "{text}"
Symbol: {symbol}

Think through this systematically:

STEP 1 - Identify Key Information:
What are the main facts in this news?

STEP 2 - Assess Impact Direction:
Is this likely positive, negative, or neutral for the stock?

STEP 3 - Evaluate Magnitude:
How significant is this news? Minor, moderate, or major?

STEP 4 - Consider Time Horizon:
Is this a short-term catalyst or long-term fundamental change?

STEP 5 - Final Assessment:
Based on your analysis, provide your conclusion.

After your reasoning, respond in JSON:
{{
    "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
    "confidence": 0-100,
    "market_impact": "BULLISH" | "BEARISH" | "NEUTRAL",
    "time_horizon": "SHORT_TERM" | "MEDIUM_TERM" | "LONG_TERM",
    "key_factors": ["factor1", "factor2"],
    "reasoning": "your step-by-step reasoning summary"
}}"#;
}

/// Trading signal generation prompts.
pub mod signals {
    /// Basic trading signal prompt.
    pub const BASIC_SIGNAL: &str = r#"You are a quantitative trading analyst. Generate a trading signal based on:

Symbol: {symbol}
Current Price: ${current_price}
Technical Indicators:
- SMA(20): ${sma_20}
- SMA(50): ${sma_50}
- RSI(14): {rsi}
- MACD: {macd}
- Support: ${support}
- Resistance: ${resistance}

Trend: {trend}

Provide a trading signal in JSON format:
{{
    "direction": "BUY" | "SELL" | "HOLD",
    "confidence": 0-100,
    "entry_price": number,
    "stop_loss": number,
    "take_profit": number,
    "position_size_pct": 0-10,
    "timeframe": "1h" | "4h" | "1d",
    "reasoning": "explanation",
    "key_factors": ["factor1", "factor2"]
}}"#;

    /// News-driven signal prompt.
    pub const NEWS_DRIVEN: &str = r#"Generate a trading signal based on this news event:

Headline: "{headline}"
Symbol: {symbol}
Current Price: ${current_price}

Consider:
1. How significant is this news?
2. What's the likely market reaction?
3. What timeframe is appropriate?
4. Where are logical stop-loss and take-profit levels?

Respond in JSON:
{{
    "direction": "BUY" | "SELL" | "HOLD",
    "confidence": 0-100,
    "entry_price": number,
    "stop_loss": number,
    "take_profit": number,
    "position_size_pct": 0-10,
    "timeframe": "1h" | "4h" | "1d" | "1w",
    "reasoning": "explanation",
    "catalyst_strength": "HIGH" | "MEDIUM" | "LOW"
}}"#;

    /// Multi-timeframe analysis prompt.
    pub const MULTI_TIMEFRAME: &str = r#"Analyze multiple timeframes for {symbol}:

Short-Term ({short_tf}):
- Trend: {short_trend}
- RSI: {short_rsi}
- Momentum: {short_momentum}

Medium-Term ({medium_tf}):
- Trend: {medium_trend}
- RSI: {medium_rsi}
- Momentum: {medium_momentum}

Long-Term ({long_tf}):
- Trend: {long_trend}
- RSI: {long_rsi}
- Momentum: {long_momentum}

Current Price: ${current_price}
Key Levels: Support ${support}, Resistance ${resistance}

Generate a signal considering timeframe alignment:
{{
    "direction": "BUY" | "SELL" | "HOLD",
    "confidence": 0-100,
    "entry_price": number,
    "stop_loss": number,
    "take_profit": number,
    "recommended_timeframe": "1h" | "4h" | "1d",
    "timeframe_alignment": {{"short": "BULLISH|BEARISH|NEUTRAL", "medium": "...", "long": "..."}},
    "reasoning": "explanation"
}}"#;
}

/// Market regime detection prompts.
pub mod regime {
    /// Comprehensive regime analysis prompt.
    pub const COMPREHENSIVE: &str = r#"You are a macro strategist analyzing current market conditions.

MARKET DATA:
Equities:
- S&P 500: {sp500_price} ({sp500_change}% change)
- NASDAQ: {nasdaq_price} ({nasdaq_change}% change)
- Russell 2000: {russell_price} ({russell_change}% change)

Volatility:
- VIX: {vix} (avg: {vix_avg})
- VIX Term Structure: {vix_term_structure}

Rates & Credit:
- 10Y Treasury: {treasury_10y}%
- 2Y Treasury: {treasury_2y}%
- Credit Spread: {credit_spread}bps

Macro:
- DXY: {dxy}
- Gold: ${gold}
- Oil: ${oil}

Breadth:
- Advance/Decline: {adv_dec_ratio}
- % Above 200 SMA: {pct_above_200}%

Classify the current market regime:
{{
    "regime": "RISK_ON_TRENDING" | "RISK_ON_VOLATILE" | "RISK_OFF_TRENDING" | "RISK_OFF_PANIC" | "RANGING" | "TRANSITIONAL",
    "confidence": 0-100,
    "key_drivers": ["driver1", "driver2", "driver3"],
    "recommended_strategies": ["strategy1", "strategy2"],
    "avoid_strategies": ["strategy1"],
    "position_sizing": "INCREASE" | "MAINTAIN" | "REDUCE",
    "next_regime_probabilities": {{"RISK_ON_TRENDING": 0.X, ...}},
    "reasoning": "explanation"
}}"#;

    /// Crypto market regime prompt.
    pub const CRYPTO: &str = r#"Analyze cryptocurrency market regime:

Bitcoin:
- Price: ${btc_price}
- 24h Change: {btc_change}%
- Dominance: {btc_dominance}%
- Funding Rate: {funding_rate}%

Market Metrics:
- Total Market Cap: ${total_mcap}
- Alt Season Index: {alt_season}
- Fear & Greed: {fear_greed}

On-Chain:
- Exchange Flows: {exchange_flows}
- Whale Activity: {whale_activity}

Classify the crypto regime:
{{
    "regime": "BULL_RUN" | "ACCUMULATION" | "DISTRIBUTION" | "BEAR_MARKET" | "ALTSEASON" | "BTC_DOMINANCE",
    "confidence": 0-100,
    "btc_outlook": "BULLISH" | "NEUTRAL" | "BEARISH",
    "altcoin_outlook": "BULLISH" | "NEUTRAL" | "BEARISH",
    "recommended_allocation": {{"btc_pct": X, "altcoin_pct": Y, "stablecoin_pct": Z}},
    "reasoning": "explanation"
}}"#;
}
