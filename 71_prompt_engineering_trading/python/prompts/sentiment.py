"""
Sentiment Analysis Prompt Templates

Templates for financial sentiment analysis with various techniques:
- Zero-shot
- Few-shot
- Chain-of-thought
"""

SENTIMENT_PROMPTS = {
    "zero_shot": """
Analyze the sentiment of the following financial news.
Respond with ONLY valid JSON in this exact format:
{{"sentiment": "POSITIVE" or "NEGATIVE" or "NEUTRAL", "confidence": 0-100}}

News: "{text}"

JSON Response:
""",

    "few_shot": """
You are analyzing financial news sentiment for trading signals.

Examples:

News: "Apple Inc. reported quarterly earnings that exceeded Wall Street expectations, with revenue growing 8% year-over-year."
Analysis: {{"sentiment": "POSITIVE", "confidence": 88, "reasoning": "Earnings beat with strong growth indicates positive momentum."}}

News: "The Federal Reserve signaled it may keep interest rates higher for longer than markets expected."
Analysis: {{"sentiment": "NEGATIVE", "confidence": 75, "reasoning": "Hawkish Fed stance pressures equity valuations."}}

News: "Tesla announced a 2% price cut on Model 3 vehicles in China, matching competitor discounts."
Analysis: {{"sentiment": "NEUTRAL", "confidence": 65, "reasoning": "Price cuts maintain competitiveness but signal margin pressure."}}

Now analyze this news:
News: "{text}"

Provide your analysis in the exact JSON format shown above.
Analysis:
""",

    "chain_of_thought": """
You are a financial analyst. Analyze the sentiment of the following news step by step.

NEWS: "{text}"

Think through this systematically:

Step 1: Identify the key event or announcement
- What happened?
- Who is involved?

Step 2: Assess immediate market implications
- How might this affect stock price?
- Is this expected or surprising news?

Step 3: Consider secondary effects
- What related companies or sectors might be affected?
- Are there any potential risks mentioned?

Step 4: Determine overall sentiment
- Weigh positive vs negative factors
- Assess confidence level

Final Analysis (respond in JSON format):
{{"sentiment": "POSITIVE/NEGATIVE/NEUTRAL", "confidence": 0-100, "reasoning": "your reasoning"}}
""",

    "role_based_analyst": """
You are a senior financial analyst at a top investment bank with 15 years of experience.
Your analysis style is:
- Data-driven and objective
- Always considers multiple perspectives
- Provides actionable insights
- Quantifies confidence levels

TASK: Analyze the sentiment of this financial news for trading purposes.

NEWS: "{text}"

Provide your analysis in JSON format:
{{
    "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
    "confidence": 0-100,
    "short_term_impact": "description of 1-5 day impact",
    "entities": {{"entity_name": "sentiment"}},
    "reasoning": "your professional analysis"
}}
""",

    "aspect_based": """
Analyze the sentiment of the following financial news toward SPECIFIC entities mentioned.

NEWS: "{text}"

ENTITIES TO ANALYZE: {entities}

For each entity, provide:
1. Sentiment (POSITIVE/NEGATIVE/NEUTRAL)
2. Confidence (0-100)
3. Specific reasoning

Response format (JSON):
{{
    "overall_sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
    "overall_confidence": 0-100,
    "entity_sentiments": {{
        "entity_name": {{
            "sentiment": "...",
            "confidence": 0-100,
            "reasoning": "..."
        }}
    }}
}}
"""
}
