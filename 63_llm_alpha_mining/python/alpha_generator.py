"""
Alpha Factor Generator using LLMs

This module provides tools for generating alpha factors using Large Language Models.
It supports both local LLMs and API-based models (OpenAI, Anthropic).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import re


@dataclass
class AlphaFactor:
    """
    Represents a generated alpha factor.

    Attributes:
        name: Human-readable name
        expression: Mathematical expression or code
        description: Natural language description
        source: Data sources used
        generated_at: Generation timestamp
        confidence: Model's confidence (0-1)
        metadata: Additional metadata
    """
    name: str
    expression: str
    description: str
    source: str = "llm"
    generated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "expression": self.expression,
            "description": self.description,
            "source": self.source,
            "generated_at": self.generated_at.isoformat(),
            "confidence": self.confidence,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlphaFactor":
        """Create from dictionary."""
        data = data.copy()
        if "generated_at" in data:
            data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        return cls(**data)


class AlphaExpressionParser:
    """
    Parse and validate alpha factor expressions.

    Supports common financial operations:
    - ts_mean(x, n): Rolling mean over n periods
    - ts_std(x, n): Rolling standard deviation
    - ts_rank(x, n): Rolling rank
    - ts_max(x, n): Rolling maximum
    - ts_min(x, n): Rolling minimum
    - ts_corr(x, y, n): Rolling correlation
    - ts_delta(x, n): x - x.shift(n)
    - rank(x): Cross-sectional rank
    - log(x): Natural logarithm
    - abs(x): Absolute value
    - sign(x): Sign function
    """

    SAFE_FUNCTIONS = {
        "ts_mean": lambda x, n: x.rolling(n).mean(),
        "ts_std": lambda x, n: x.rolling(n).std(),
        "ts_rank": lambda x, n: x.rolling(n).apply(lambda arr: pd.Series(arr).rank().iloc[-1] / len(arr), raw=False),
        "ts_max": lambda x, n: x.rolling(n).max(),
        "ts_min": lambda x, n: x.rolling(n).min(),
        "ts_sum": lambda x, n: x.rolling(n).sum(),
        "ts_delta": lambda x, n: x - x.shift(n),
        "ts_delay": lambda x, n: x.shift(n),
        "ts_corr": lambda x, y, n: x.rolling(n).corr(y),
        "ts_cov": lambda x, y, n: x.rolling(n).cov(y),
        "rank": lambda x: x.rank(pct=True),
        "log": lambda x: np.log(x.clip(lower=1e-10)),
        "abs": lambda x: np.abs(x),
        "sign": lambda x: np.sign(x),
        "sqrt": lambda x: np.sqrt(x.clip(lower=0)),
        "power": lambda x, n: np.power(x, n),
    }

    SAFE_VARIABLES = {
        "open", "high", "low", "close", "volume", "turnover",
        "returns", "log_returns", "vwap"
    }

    def __init__(self):
        self._compiled_cache = {}

    def validate(self, expression: str) -> bool:
        """
        Validate an alpha expression for safety and correctness.

        Args:
            expression: The alpha expression to validate

        Returns:
            True if valid, False otherwise
        """
        # Check for dangerous operations
        dangerous_patterns = [
            r"import\s+",
            r"exec\(",
            r"eval\(",
            r"__",
            r"open\(",
            r"file\(",
            r"os\.",
            r"sys\.",
            r"subprocess",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                return False

        # Check that only safe functions are used
        func_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        funcs_used = set(re.findall(func_pattern, expression))

        for func in funcs_used:
            if func not in self.SAFE_FUNCTIONS and func not in ["max", "min"]:
                return False

        return True

    def evaluate(
        self,
        expression: str,
        data: pd.DataFrame,
        additional_vars: Optional[Dict[str, pd.Series]] = None
    ) -> pd.Series:
        """
        Evaluate an alpha expression on data.

        Args:
            expression: The alpha expression
            data: OHLCV DataFrame
            additional_vars: Additional variables to include

        Returns:
            Series of alpha factor values
        """
        if not self.validate(expression):
            raise ValueError(f"Invalid or unsafe expression: {expression}")

        # Build evaluation context
        context = {}

        # Add data columns
        for col in data.columns:
            context[col.lower()] = data[col]

        # Add computed variables
        if "close" in data.columns:
            context["returns"] = data["close"].pct_change()
            context["log_returns"] = np.log(data["close"] / data["close"].shift(1))

        if all(col in data.columns for col in ["high", "low", "close", "volume"]):
            context["vwap"] = (
                (data["high"] + data["low"] + data["close"]) / 3 * data["volume"]
            ).cumsum() / data["volume"].cumsum()

        # Add additional variables
        if additional_vars:
            context.update(additional_vars)

        # Add safe functions
        context.update(self.SAFE_FUNCTIONS)
        context["max"] = max
        context["min"] = min
        context["np"] = np
        context["pd"] = pd

        try:
            result = eval(expression, {"__builtins__": {}}, context)
            if isinstance(result, pd.Series):
                return result
            elif isinstance(result, np.ndarray):
                return pd.Series(result, index=data.index)
            else:
                raise ValueError(f"Expression returned {type(result)}, expected Series")
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {e}")


class AlphaGenerator:
    """
    Generate alpha factors using Large Language Models.

    This class provides methods to prompt LLMs to generate trading signals
    based on market data and domain knowledge.

    Examples:
        >>> generator = AlphaGenerator()
        >>> factors = generator.generate_from_description(
        ...     "momentum reversal strategy",
        ...     data=btc_data
        ... )
    """

    # Prompt templates for different generation modes
    PROMPTS = {
        "basic": """Generate a quantitative alpha factor for {market_type} trading.

Context:
- Asset: {symbol}
- Timeframe: {timeframe}
- Available data: {available_data}

Requirements:
1. The factor should predict future returns
2. Use only the available data columns
3. Express as a mathematical formula using these operators:
   - ts_mean(x, n): Rolling mean over n periods
   - ts_std(x, n): Rolling standard deviation
   - ts_delta(x, n): Difference from n periods ago
   - ts_rank(x, n): Rolling percentile rank
   - rank(x): Cross-sectional rank
   - log(x), abs(x), sign(x): Standard math functions

Generate 3 different alpha factors in JSON format:
[
  {{"name": "factor_name", "expression": "formula", "description": "explanation"}}
]
""",

        "momentum": """Generate momentum-based alpha factors for cryptocurrency trading.

The data includes OHLCV (open, high, low, close, volume) data for {symbol}.

Consider these momentum concepts:
- Price momentum (trend following)
- Volume momentum
- Momentum divergence
- Mean reversion opportunities

Generate 3 momentum factors as JSON:
[
  {{"name": "...", "expression": "...", "description": "..."}}
]
""",

        "volatility": """Generate volatility-based alpha factors.

Consider:
- Volatility clustering
- Volatility breakouts
- ATR-based signals
- Volatility regime changes

Generate 3 volatility factors as JSON for {symbol}:
[
  {{"name": "...", "expression": "...", "description": "..."}}
]
""",

        "crypto_specific": """Generate cryptocurrency-specific alpha factors.

For {symbol}, consider crypto-unique features:
- High volatility regime
- 24/7 trading patterns
- Volume spikes
- Funding rate effects (for perpetuals)
- Whale activity indicators

Available data: {available_data}

Generate 3 crypto-specific factors as JSON:
[
  {{"name": "...", "expression": "...", "description": "..."}}
]
"""
    }

    def __init__(
        self,
        model: str = "mock",
        api_key: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Initialize the alpha generator.

        Args:
            model: Model to use ("mock", "openai", "anthropic", "local")
            api_key: API key for cloud models
            temperature: Generation temperature
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.parser = AlphaExpressionParser()
        self._client = None

    def _ensure_client(self):
        """Initialize the appropriate client."""
        if self._client is not None:
            return

        if self.model == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

        elif self.model == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        if self.model == "mock":
            return self._mock_response(prompt)

        self._ensure_client()

        if self.model == "openai":
            response = self._client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content

        elif self.model == "anthropic":
            response = self._client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _mock_response(self, prompt: str) -> str:
        """Generate mock responses for testing."""
        # Determine which type of factors to generate based on prompt
        if "momentum" in prompt.lower():
            factors = [
                {
                    "name": "momentum_5d",
                    "expression": "ts_delta(close, 5) / ts_delay(close, 5)",
                    "description": "5-day price momentum"
                },
                {
                    "name": "volume_momentum",
                    "expression": "ts_mean(volume, 5) / ts_mean(volume, 20) - 1",
                    "description": "Volume momentum relative to 20-day average"
                },
                {
                    "name": "momentum_reversal",
                    "expression": "-1 * ts_delta(close, 1) * sign(ts_delta(close, 5))",
                    "description": "Short-term reversal in momentum direction"
                }
            ]
        elif "volatility" in prompt.lower():
            factors = [
                {
                    "name": "volatility_breakout",
                    "expression": "(close - ts_mean(close, 20)) / ts_std(close, 20)",
                    "description": "Normalized deviation from 20-day mean"
                },
                {
                    "name": "volatility_ratio",
                    "expression": "ts_std(close, 5) / ts_std(close, 20)",
                    "description": "Short vs long-term volatility ratio"
                },
                {
                    "name": "range_expansion",
                    "expression": "(high - low) / ts_mean(high - low, 20)",
                    "description": "Daily range relative to average"
                }
            ]
        elif "crypto" in prompt.lower():
            factors = [
                {
                    "name": "crypto_momentum",
                    "expression": "rank(ts_delta(close, 24)) * rank(volume)",
                    "description": "Combined price and volume momentum"
                },
                {
                    "name": "volume_spike",
                    "expression": "sign(volume / ts_mean(volume, 48) - 2)",
                    "description": "Detect significant volume spikes"
                },
                {
                    "name": "overnight_gap",
                    "expression": "(open - ts_delay(close, 1)) / ts_delay(close, 1)",
                    "description": "Overnight price gap (in 24/7 markets)"
                }
            ]
        else:
            factors = [
                {
                    "name": "mean_reversion",
                    "expression": "-1 * (close - ts_mean(close, 20)) / ts_std(close, 20)",
                    "description": "Mean reversion signal based on z-score"
                },
                {
                    "name": "trend_strength",
                    "expression": "ts_mean(sign(ts_delta(close, 1)), 10)",
                    "description": "Trend consistency over 10 periods"
                },
                {
                    "name": "volume_price",
                    "expression": "ts_corr(close, volume, 20)",
                    "description": "Price-volume correlation"
                }
            ]

        return json.dumps(factors, indent=2)

    def generate(
        self,
        data: pd.DataFrame,
        prompt_type: str = "basic",
        symbol: str = "UNKNOWN",
        market_type: str = "crypto",
        n_factors: int = 3,
        validate: bool = True
    ) -> List[AlphaFactor]:
        """
        Generate alpha factors using LLM.

        Args:
            data: OHLCV DataFrame
            prompt_type: Type of prompt ("basic", "momentum", "volatility", "crypto_specific")
            symbol: Asset symbol
            market_type: Market type ("stock", "crypto")
            n_factors: Number of factors to generate
            validate: Whether to validate expressions

        Returns:
            List of AlphaFactor objects
        """
        # Get prompt template
        template = self.PROMPTS.get(prompt_type, self.PROMPTS["basic"])

        # Format prompt
        prompt = template.format(
            symbol=symbol,
            market_type=market_type,
            timeframe="hourly" if len(data) > 1000 else "daily",
            available_data=", ".join(data.columns.tolist())
        )

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        factors = self._parse_response(response, validate=validate)

        return factors

    def _parse_response(
        self,
        response: str,
        validate: bool = True
    ) -> List[AlphaFactor]:
        """Parse LLM response into AlphaFactor objects."""
        # Try to extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            raise ValueError("Could not find JSON array in response")

        try:
            factors_data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        factors = []
        for factor_dict in factors_data:
            if validate:
                if not self.parser.validate(factor_dict.get("expression", "")):
                    continue  # Skip invalid expressions

            factor = AlphaFactor(
                name=factor_dict.get("name", "unnamed"),
                expression=factor_dict.get("expression", ""),
                description=factor_dict.get("description", ""),
                source="llm",
                confidence=factor_dict.get("confidence", 0.5)
            )
            factors.append(factor)

        return factors

    def generate_from_description(
        self,
        description: str,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> List[AlphaFactor]:
        """
        Generate factors from a natural language description.

        Args:
            description: Natural language description of desired strategy
            data: OHLCV DataFrame
            symbol: Asset symbol

        Returns:
            List of AlphaFactor objects
        """
        prompt = f"""Based on this strategy description, generate alpha factors:

Strategy: {description}

Available data columns: {', '.join(data.columns.tolist())}
Symbol: {symbol}

Generate 3 alpha factors as JSON:
[
  {{"name": "...", "expression": "...", "description": "..."}}
]

Use these operators: ts_mean, ts_std, ts_delta, ts_rank, ts_delay, rank, log, abs, sign
"""

        response = self._call_llm(prompt)
        return self._parse_response(response)

    def evaluate_factor(
        self,
        factor: AlphaFactor,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Evaluate an alpha factor on data.

        Args:
            factor: The AlphaFactor to evaluate
            data: OHLCV DataFrame

        Returns:
            Series of factor values
        """
        return self.parser.evaluate(factor.expression, data)


# Predefined alpha factors library
PREDEFINED_FACTORS = [
    AlphaFactor(
        name="momentum_5d",
        expression="ts_delta(close, 5) / ts_delay(close, 5)",
        description="5-day price momentum (percentage change)",
        source="predefined",
        confidence=0.7
    ),
    AlphaFactor(
        name="mean_reversion_20d",
        expression="-1 * (close - ts_mean(close, 20)) / ts_std(close, 20)",
        description="Mean reversion z-score over 20 periods",
        source="predefined",
        confidence=0.65
    ),
    AlphaFactor(
        name="volume_breakout",
        expression="sign(volume / ts_mean(volume, 20) - 1.5)",
        description="Volume breakout signal (50% above average)",
        source="predefined",
        confidence=0.6
    ),
    AlphaFactor(
        name="volatility_expansion",
        expression="ts_std(close, 5) / ts_std(close, 20) - 1",
        description="Volatility expansion indicator",
        source="predefined",
        confidence=0.55
    ),
    AlphaFactor(
        name="trend_strength",
        expression="ts_mean(sign(ts_delta(close, 1)), 10)",
        description="Trend consistency (average sign of daily returns)",
        source="predefined",
        confidence=0.7
    ),
    AlphaFactor(
        name="price_volume_divergence",
        expression="ts_corr(close, volume, 20) * -1",
        description="Negative price-volume correlation (divergence)",
        source="predefined",
        confidence=0.5
    ),
]


if __name__ == "__main__":
    from data_loader import generate_synthetic_data, calculate_features

    print("LLM Alpha Mining - Alpha Generator Demo")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Loading Data")
    print("-" * 40)
    data = generate_synthetic_data(["BTCUSDT"], days=180)
    btc_data = data["BTCUSDT"].ohlcv
    print(f"Data shape: {btc_data.shape}")

    # Initialize generator (mock mode for demo)
    print("\n2. Initializing Alpha Generator (Mock Mode)")
    print("-" * 40)
    generator = AlphaGenerator(model="mock")
    parser = AlphaExpressionParser()

    # Generate factors
    print("\n3. Generating Alpha Factors")
    print("-" * 40)

    for prompt_type in ["basic", "momentum", "crypto_specific"]:
        print(f"\n{prompt_type.upper()} factors:")
        factors = generator.generate(
            btc_data,
            prompt_type=prompt_type,
            symbol="BTCUSDT"
        )
        for factor in factors:
            print(f"  - {factor.name}: {factor.description}")
            print(f"    Expression: {factor.expression}")

    # Evaluate predefined factors
    print("\n4. Evaluating Predefined Factors")
    print("-" * 40)

    for factor in PREDEFINED_FACTORS[:3]:
        try:
            values = parser.evaluate(factor.expression, btc_data)
            print(f"\n{factor.name}:")
            print(f"  Expression: {factor.expression}")
            print(f"  Mean: {values.mean():.4f}")
            print(f"  Std: {values.std():.4f}")
            print(f"  Non-null: {values.notna().sum()}/{len(values)}")
        except Exception as e:
            print(f"  Error: {e}")

    # Generate from description
    print("\n5. Generate from Natural Language")
    print("-" * 40)

    factors = generator.generate_from_description(
        "A contrarian strategy that buys when prices drop sharply but volume is low",
        btc_data,
        symbol="BTCUSDT"
    )

    for factor in factors:
        print(f"  - {factor.name}: {factor.description}")

    print("\n" + "=" * 60)
    print("Demo complete!")
