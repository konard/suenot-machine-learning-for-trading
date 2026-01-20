"""
Market Regime Detection Prompt Templates

Templates for identifying market regimes and recommending
appropriate trading strategies.
"""

REGIME_PROMPTS = {
    "basic_regime": """
Classify the current market regime based on the following data:

MARKET DATA:
- S&P 500: {sp500_price} ({sp500_change:+.2f}% MTD)
- VIX: {vix}
- 10Y Treasury: {treasury_10y}%

REGIME OPTIONS:
- RISK_ON_TRENDING: Strong uptrend, low volatility
- RISK_ON_VOLATILE: Uptrend with high volatility
- RISK_OFF_TRENDING: Strong downtrend
- RISK_OFF_PANIC: Crash mode, extreme fear
- RANGING: Sideways consolidation
- TRANSITIONAL: Regime change in progress

Response (JSON):
{{
    "regime": "REGIME_NAME",
    "confidence": 0-100,
    "reasoning": "brief explanation"
}}
""",

    "comprehensive_regime": """
You are a market regime analyst at a macro hedge fund.
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

1. TREND ASSESSMENT: Is the market trending up, down, or ranging?
2. VOLATILITY ASSESSMENT: Is volatility elevated, normal, or suppressed?
3. RISK APPETITE: Are investors seeking or avoiding risk?
4. BREADTH & PARTICIPATION: Is the move broad-based or narrow?

Classify the regime and provide recommendations:

{{
    "regime": "RISK_ON_TRENDING" | "RISK_ON_VOLATILE" | "RISK_OFF_TRENDING" | "RISK_OFF_PANIC" | "RANGING" | "TRANSITIONAL",
    "confidence": 0-100,
    "key_drivers": ["driver1", "driver2", "driver3"],
    "recommended_strategies": ["strategy1", "strategy2"],
    "avoid_strategies": ["strategy1", "strategy2"],
    "position_sizing": "INCREASE" | "MAINTAIN" | "REDUCE",
    "next_regime_probabilities": {{
        "regime1": probability,
        "regime2": probability
    }},
    "reasoning": "2-3 sentence summary"
}}
""",

    "crypto_regime": """
You are analyzing the cryptocurrency market regime.

CRYPTO MARKET DATA:
- BTC Price: ${btc_price} ({btc_change:+.2f}% 24h)
- BTC Dominance: {btc_dominance}%
- Total Market Cap: ${total_mcap}B
- Fear & Greed Index: {fear_greed}

CORRELATION DATA:
- BTC/SPY Correlation (30d): {btc_spy_corr}
- ETH/BTC Ratio: {eth_btc_ratio}
- Altcoin Season Index: {altcoin_index}

ON-CHAIN METRICS:
- Exchange Inflows: {exchange_inflows}
- Whale Activity: {whale_activity}
- Funding Rates: {funding_rates}

Classify the crypto market regime:

{{
    "regime": "BULL_RUN" | "ACCUMULATION" | "DISTRIBUTION" | "BEAR_MARKET" | "ALTSEASON" | "BTC_DOMINANCE",
    "confidence": 0-100,
    "btc_outlook": "BULLISH" | "BEARISH" | "NEUTRAL",
    "altcoin_outlook": "OUTPERFORM" | "UNDERPERFORM" | "NEUTRAL",
    "recommended_allocation": {{
        "btc_pct": 0-100,
        "eth_pct": 0-100,
        "altcoin_pct": 0-100,
        "stablecoin_pct": 0-100
    }},
    "key_levels": {{
        "btc_support": price,
        "btc_resistance": price
    }},
    "reasoning": "explanation"
}}
"""
}
