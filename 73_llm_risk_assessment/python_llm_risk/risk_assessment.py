"""
LLM Risk Assessment Module

This module provides classes for risk assessment using Large Language Models.
It supports multiple LLM providers (OpenAI, Anthropic) and generates
multi-dimensional risk scores for trading decisions.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification."""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    SEVERE = 5

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Convert numeric score to risk level."""
        if score <= 2:
            return cls.VERY_LOW
        elif score <= 4:
            return cls.LOW
        elif score <= 6:
            return cls.MODERATE
        elif score <= 8:
            return cls.HIGH
        return cls.SEVERE


@dataclass
class RiskScore:
    """
    Multi-dimensional risk score from LLM analysis.

    Each dimension is scored from 1 (lowest risk) to 10 (highest risk).
    """
    market_risk: float = 5.0
    credit_risk: float = 5.0
    liquidity_risk: float = 5.0
    operational_risk: float = 5.0
    regulatory_risk: float = 5.0
    sentiment_risk: float = 5.0
    confidence: str = "medium"
    direction: str = "stable"
    reasoning: str = ""

    def overall(self) -> float:
        """Calculate weighted average risk score."""
        weights = {
            "market_risk": 0.25,
            "credit_risk": 0.15,
            "liquidity_risk": 0.15,
            "operational_risk": 0.15,
            "regulatory_risk": 0.15,
            "sentiment_risk": 0.15,
        }

        total = (
            self.market_risk * weights["market_risk"] +
            self.credit_risk * weights["credit_risk"] +
            self.liquidity_risk * weights["liquidity_risk"] +
            self.operational_risk * weights["operational_risk"] +
            self.regulatory_risk * weights["regulatory_risk"] +
            self.sentiment_risk * weights["sentiment_risk"]
        )

        return round(total, 2)

    @property
    def risk_level(self) -> RiskLevel:
        """Get overall risk level."""
        return RiskLevel.from_score(self.overall())

    def position_multiplier(self) -> float:
        """
        Calculate position size multiplier based on risk.

        Lower risk = larger position (inverse relationship).
        Returns value between 0.1 and 1.0.
        """
        score = self.overall()

        # Confidence adjustment
        confidence_factor = {
            "high": 1.0,
            "medium": 0.85,
            "low": 0.7
        }.get(self.confidence, 0.7)

        # Base multiplier: inverse of risk
        base = max(0.1, 1.0 - (score / 10))

        return round(base * confidence_factor, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_risk": self.market_risk,
            "credit_risk": self.credit_risk,
            "liquidity_risk": self.liquidity_risk,
            "operational_risk": self.operational_risk,
            "regulatory_risk": self.regulatory_risk,
            "sentiment_risk": self.sentiment_risk,
            "overall_score": self.overall(),
            "risk_level": self.risk_level.name,
            "confidence": self.confidence,
            "direction": self.direction,
            "position_multiplier": self.position_multiplier(),
            "reasoning": self.reasoning,
        }

    def __str__(self) -> str:
        """Format risk score for display."""
        return f"""
Risk Assessment
===============
Market Risk:      {self.market_risk:.1f}/10
Credit Risk:      {self.credit_risk:.1f}/10
Liquidity Risk:   {self.liquidity_risk:.1f}/10
Operational Risk: {self.operational_risk:.1f}/10
Regulatory Risk:  {self.regulatory_risk:.1f}/10
Sentiment Risk:   {self.sentiment_risk:.1f}/10
---------------
Overall Score:    {self.overall():.1f}/10 ({self.risk_level.name})
Confidence:       {self.confidence}
Direction:        {self.direction}

Reasoning: {self.reasoning}
"""


# Risk assessment prompt template
RISK_ASSESSMENT_PROMPT = """You are a financial risk analyst. Analyze the following text and provide a comprehensive risk assessment.

Text to analyze:
{text}

{context}

Evaluate the following risk dimensions on a scale of 1-10 (1=lowest risk, 10=highest risk):

1. Market Risk: Exposure to market volatility and systematic risk
2. Credit Risk: Counterparty or default risk indicators
3. Liquidity Risk: Trading and market depth concerns
4. Operational Risk: Business execution and management risks
5. Regulatory Risk: Legal and compliance exposure
6. Sentiment Risk: Market perception and reputation

For each dimension, identify key factors from the text.

Also determine:
- Confidence: "low", "medium", or "high" based on text clarity
- Direction: "increasing", "stable", or "decreasing" risk trend
- Reasoning: Brief explanation of the assessment

Respond in JSON format only:
{{
    "market_risk": <number>,
    "credit_risk": <number>,
    "liquidity_risk": <number>,
    "operational_risk": <number>,
    "regulatory_risk": <number>,
    "sentiment_risk": <number>,
    "confidence": "<string>",
    "direction": "<string>",
    "reasoning": "<string>"
}}"""


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMRiskAssessor:
    """
    LLM-based risk assessment engine.

    Supports multiple LLM providers for analyzing text and generating
    multi-dimensional risk scores.
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ):
        """
        Initialize the risk assessor.

        Args:
            provider: LLM provider to use
            model: Model name (provider-specific)
            api_key: API key (or use environment variable)
            base_url: Base URL for local providers
            temperature: Model temperature (lower = more deterministic)
            max_tokens: Maximum response tokens
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

        # Set default models
        if model is None:
            model = {
                LLMProvider.OPENAI: "gpt-4",
                LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229",
                LLMProvider.LOCAL: "llama2",
            }.get(provider, "gpt-4")
        self.model = model

        # Get API key
        if api_key is None:
            api_key = self._get_api_key_from_env()
        self.api_key = api_key

        # Initialize client
        self.client = self._init_client()

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        env_vars = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        }

        var_name = env_vars.get(self.provider)
        if var_name:
            return os.environ.get(var_name)
        return None

    def _init_client(self) -> Any:
        """Initialize the appropriate LLM client."""
        if self.provider == LLMProvider.OPENAI:
            try:
                import openai
                return openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

        elif self.provider == LLMProvider.ANTHROPIC:
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")

        elif self.provider == LLMProvider.LOCAL:
            # For local providers, we'll use requests
            return None

        raise ValueError(f"Unsupported provider: {self.provider}")

    def assess(self, text: str) -> RiskScore:
        """
        Analyze text and return risk scores.

        Args:
            text: Text to analyze

        Returns:
            RiskScore with multi-dimensional assessment
        """
        return self.assess_with_context(text, None, None)

    def assess_with_context(
        self,
        text: str,
        symbol: Optional[str] = None,
        market_data: Optional[Dict] = None,
    ) -> RiskScore:
        """
        Analyze text with additional context.

        Args:
            text: Text to analyze
            symbol: Trading symbol for context
            market_data: Additional market data context

        Returns:
            RiskScore with multi-dimensional assessment
        """
        # Build context string
        context_parts = []
        if symbol:
            context_parts.append(f"Symbol: {symbol}")
        if market_data:
            context_parts.append(f"Market Data: {json.dumps(market_data)}")

        context = "\n".join(context_parts) if context_parts else ""

        # Build prompt
        prompt = RISK_ASSESSMENT_PROMPT.format(text=text, context=context)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        return self._parse_response(response)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return response text."""
        if self.provider == LLMProvider.OPENAI:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial risk analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content

        elif self.provider == LLMProvider.ANTHROPIC:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            return response.content[0].text

        elif self.provider == LLMProvider.LOCAL:
            import requests

            base_url = self.base_url or "http://localhost:11434"
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["response"]

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _parse_response(self, response: str) -> RiskScore:
        """Parse LLM response into RiskScore."""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Handle markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)

            return RiskScore(
                market_risk=float(data.get("market_risk", 5)),
                credit_risk=float(data.get("credit_risk", 5)),
                liquidity_risk=float(data.get("liquidity_risk", 5)),
                operational_risk=float(data.get("operational_risk", 5)),
                regulatory_risk=float(data.get("regulatory_risk", 5)),
                sentiment_risk=float(data.get("sentiment_risk", 5)),
                confidence=data.get("confidence", "medium"),
                direction=data.get("direction", "stable"),
                reasoning=data.get("reasoning", ""),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Return default risk score
            return RiskScore(reasoning=f"Parse error: {str(e)}")

    def assess_multiple(self, texts: List[str]) -> List[RiskScore]:
        """
        Assess multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of RiskScore objects
        """
        return [self.assess(text) for text in texts]


# Example usage and testing
if __name__ == "__main__":
    # Demo with mock response (no API key needed)
    print("LLM Risk Assessment Demo")
    print("=" * 50)

    # Create sample risk score
    sample_score = RiskScore(
        market_risk=6.5,
        credit_risk=3.0,
        liquidity_risk=4.5,
        operational_risk=7.0,
        regulatory_risk=8.0,
        sentiment_risk=5.5,
        confidence="high",
        direction="increasing",
        reasoning="Regulatory concerns are elevated due to recent SEC investigation announcements. Market volatility is moderate but operational risks are high due to management changes."
    )

    print(sample_score)
    print(f"\nPosition Multiplier: {sample_score.position_multiplier()}")
    print(f"\nAs Dictionary:")
    print(json.dumps(sample_score.to_dict(), indent=2))
