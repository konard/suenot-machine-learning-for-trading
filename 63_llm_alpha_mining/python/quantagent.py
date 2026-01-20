"""
QuantAgent - Self-Improving Alpha Mining Agent

This module implements a self-improving agent architecture inspired by the
QuantAgent paper (arXiv:2402.03755). The agent uses two feedback loops:
1. Inner loop: Iterative reasoning with tool use
2. Outer loop: Learning from experience via knowledge base

The agent continuously generates, evaluates, and refines alpha factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib


@dataclass
class Experience:
    """
    A single learning experience from alpha mining.

    Attributes:
        factor_expression: The alpha expression
        factor_name: Human-readable name
        metrics: Evaluation metrics
        market_condition: Market state when evaluated
        success: Whether factor met quality threshold
        timestamp: When experience was recorded
        notes: Additional observations
    """
    factor_expression: str
    factor_name: str
    metrics: Dict[str, float]
    market_condition: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "factor_expression": self.factor_expression,
            "factor_name": self.factor_name,
            "metrics": self.metrics,
            "market_condition": self.market_condition,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class KnowledgeBase:
    """
    Knowledge base for storing and retrieving alpha mining experiences.

    The knowledge base enables:
    - Storage of successful/failed alpha factors
    - Retrieval of relevant past experiences
    - Learning from mistakes
    - Pattern recognition across market conditions

    Examples:
        >>> kb = KnowledgeBase()
        >>> kb.add_experience(experience)
        >>> relevant = kb.query("momentum", market_condition="volatile")
    """

    def __init__(self, max_experiences: int = 10000):
        """
        Initialize knowledge base.

        Args:
            max_experiences: Maximum experiences to store
        """
        self.max_experiences = max_experiences
        self.experiences: List[Experience] = []
        self.successful_patterns: Dict[str, int] = {}
        self.failed_patterns: Dict[str, int] = {}
        self._expression_hashes: set = set()

    def add_experience(self, experience: Experience) -> bool:
        """
        Add a new experience to the knowledge base.

        Args:
            experience: The experience to add

        Returns:
            True if added (not duplicate), False otherwise
        """
        # Check for duplicates
        expr_hash = self._hash_expression(experience.factor_expression)
        if expr_hash in self._expression_hashes:
            return False

        self._expression_hashes.add(expr_hash)
        self.experiences.append(experience)

        # Update pattern counters
        patterns = self._extract_patterns(experience.factor_expression)
        counter = self.successful_patterns if experience.success else self.failed_patterns

        for pattern in patterns:
            counter[pattern] = counter.get(pattern, 0) + 1

        # Trim if needed
        if len(self.experiences) > self.max_experiences:
            # Remove oldest non-successful experiences first
            self.experiences.sort(key=lambda x: (x.success, x.timestamp))
            removed = self.experiences.pop(0)
            self._expression_hashes.discard(
                self._hash_expression(removed.factor_expression)
            )

        return True

    def _hash_expression(self, expression: str) -> str:
        """Create hash of expression for deduplication."""
        normalized = expression.lower().replace(" ", "")
        return hashlib.md5(normalized.encode()).hexdigest()

    def _extract_patterns(self, expression: str) -> List[str]:
        """Extract patterns from expression for learning."""
        patterns = []

        # Extract function calls
        import re
        funcs = re.findall(r'(ts_\w+|rank|log|abs|sign)', expression)
        patterns.extend(funcs)

        # Extract variable names
        vars_used = re.findall(r'\b(close|open|high|low|volume)\b', expression)
        patterns.extend(vars_used)

        # Extract operations
        if "ts_mean" in expression and "ts_std" in expression:
            patterns.append("zscore_pattern")
        if "ts_delta" in expression:
            patterns.append("momentum_pattern")
        if "-1 *" in expression or "* -1" in expression:
            patterns.append("reversal_pattern")

        return patterns

    def query(
        self,
        keyword: Optional[str] = None,
        market_condition: Optional[str] = None,
        success_only: bool = False,
        limit: int = 10
    ) -> List[Experience]:
        """
        Query the knowledge base for relevant experiences.

        Args:
            keyword: Search keyword
            market_condition: Filter by market condition
            success_only: Only return successful experiences
            limit: Maximum results

        Returns:
            List of matching experiences
        """
        results = self.experiences.copy()

        if success_only:
            results = [e for e in results if e.success]

        if market_condition:
            results = [e for e in results if e.market_condition == market_condition]

        if keyword:
            keyword_lower = keyword.lower()
            results = [
                e for e in results
                if keyword_lower in e.factor_expression.lower()
                or keyword_lower in e.factor_name.lower()
                or keyword_lower in e.notes.lower()
            ]

        # Sort by quality (IC * success)
        results.sort(
            key=lambda e: abs(e.metrics.get("ic", 0)) * (2 if e.success else 1),
            reverse=True
        )

        return results[:limit]

    def get_best_patterns(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most successful patterns.

        Returns patterns with highest success rate.
        """
        pattern_scores = {}

        for pattern in set(self.successful_patterns.keys()) | set(self.failed_patterns.keys()):
            success = self.successful_patterns.get(pattern, 0)
            fail = self.failed_patterns.get(pattern, 0)
            total = success + fail

            if total >= 3:  # Minimum samples
                score = success / total
                pattern_scores[pattern] = score

        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_patterns[:n]

    def get_avoid_patterns(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get patterns to avoid (low success rate).
        """
        pattern_scores = {}

        for pattern in set(self.successful_patterns.keys()) | set(self.failed_patterns.keys()):
            success = self.successful_patterns.get(pattern, 0)
            fail = self.failed_patterns.get(pattern, 0)
            total = success + fail

            if total >= 3:
                score = fail / total
                pattern_scores[pattern] = score

        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_patterns[:n]

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of knowledge base."""
        if not self.experiences:
            return {"total": 0}

        successful = [e for e in self.experiences if e.success]
        return {
            "total": len(self.experiences),
            "successful": len(successful),
            "success_rate": len(successful) / len(self.experiences),
            "unique_patterns": len(self.successful_patterns) + len(self.failed_patterns),
            "best_patterns": self.get_best_patterns(5),
            "avoid_patterns": self.get_avoid_patterns(3)
        }

    def to_json(self) -> str:
        """Export knowledge base to JSON."""
        data = {
            "experiences": [e.to_dict() for e in self.experiences],
            "successful_patterns": self.successful_patterns,
            "failed_patterns": self.failed_patterns
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeBase":
        """Import knowledge base from JSON."""
        data = json.loads(json_str)
        kb = cls()

        for exp_dict in data.get("experiences", []):
            experience = Experience.from_dict(exp_dict)
            kb.add_experience(experience)

        return kb


class QuantAgent:
    """
    Self-improving alpha mining agent.

    QuantAgent implements the two-loop architecture:
    - Inner Loop: Generate -> Evaluate -> Refine
    - Outer Loop: Learn from experience, update knowledge base

    The agent improves over time by learning which patterns work
    in different market conditions.

    Examples:
        >>> agent = QuantAgent()
        >>> factors = agent.mine(btc_data, n_iterations=10)
        >>> agent.kb.summary()  # View learning progress
    """

    # Market condition classifiers
    MARKET_CONDITIONS = {
        "bullish": lambda df: df["close"].iloc[-20:].mean() > df["close"].iloc[-60:-20].mean() * 1.05,
        "bearish": lambda df: df["close"].iloc[-20:].mean() < df["close"].iloc[-60:-20].mean() * 0.95,
        "volatile": lambda df: df["close"].pct_change().iloc[-20:].std() > df["close"].pct_change().std() * 1.5,
        "quiet": lambda df: df["close"].pct_change().iloc[-20:].std() < df["close"].pct_change().std() * 0.5,
    }

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        alpha_generator=None,
        factor_evaluator=None,
        quality_threshold: float = 40.0,
        model: str = "mock"
    ):
        """
        Initialize QuantAgent.

        Args:
            knowledge_base: Existing knowledge base (or create new)
            alpha_generator: Alpha factor generator
            factor_evaluator: Factor evaluator
            quality_threshold: Minimum quality score for success
            model: LLM model to use
        """
        self.kb = knowledge_base or KnowledgeBase()
        self.quality_threshold = quality_threshold
        self.model = model

        # Lazy load components
        self._generator = alpha_generator
        self._evaluator = factor_evaluator
        self._parser = None

    def _ensure_components(self):
        """Initialize components if needed."""
        if self._generator is None:
            from .alpha_generator import AlphaGenerator
            self._generator = AlphaGenerator(model=self.model)

        if self._evaluator is None:
            from .factor_evaluator import FactorEvaluator
            self._evaluator = FactorEvaluator()

        if self._parser is None:
            from .alpha_generator import AlphaExpressionParser
            self._parser = AlphaExpressionParser()

    def classify_market(self, data: pd.DataFrame) -> str:
        """
        Classify current market condition.

        Args:
            data: OHLCV DataFrame

        Returns:
            Market condition string
        """
        conditions = []

        for name, classifier in self.MARKET_CONDITIONS.items():
            try:
                if classifier(data):
                    conditions.append(name)
            except Exception:
                pass

        if not conditions:
            return "neutral"

        return "_".join(sorted(conditions))

    def _create_context_prompt(self, market_condition: str) -> str:
        """Create context from knowledge base for LLM prompt."""
        # Get relevant past experiences
        successful = self.kb.query(success_only=True, market_condition=market_condition, limit=3)
        best_patterns = self.kb.get_best_patterns(5)
        avoid_patterns = self.kb.get_avoid_patterns(3)

        context_parts = []

        if successful:
            context_parts.append("Successful factors in similar conditions:")
            for exp in successful:
                context_parts.append(f"  - {exp.factor_name}: {exp.factor_expression}")
                context_parts.append(f"    IC: {exp.metrics.get('ic', 0):.4f}")

        if best_patterns:
            context_parts.append("\nPatterns that work well:")
            for pattern, score in best_patterns:
                context_parts.append(f"  - {pattern}: {score:.1%} success rate")

        if avoid_patterns:
            context_parts.append("\nPatterns to avoid:")
            for pattern, score in avoid_patterns:
                context_parts.append(f"  - {pattern}: {score:.1%} failure rate")

        return "\n".join(context_parts)

    def mine(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        n_iterations: int = 10,
        factors_per_iteration: int = 3,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Mine for alpha factors with self-improvement.

        This implements the two-loop architecture:
        - Outer loop: iterations
        - Inner loop: generate, evaluate, learn

        Args:
            data: OHLCV DataFrame
            symbol: Asset symbol
            n_iterations: Number of mining iterations
            factors_per_iteration: Factors to generate per iteration
            verbose: Print progress

        Returns:
            List of successful factors with metrics
        """
        self._ensure_components()

        successful_factors = []
        market_condition = self.classify_market(data)

        # Calculate forward returns once
        returns = data["close"].pct_change().shift(-1)

        if verbose:
            print(f"Starting QuantAgent mining for {symbol}")
            print(f"Market condition: {market_condition}")
            print(f"Iterations: {n_iterations}")
            print("-" * 40)

        for iteration in range(n_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}/{n_iterations}")

            # Generate context from knowledge base
            context = self._create_context_prompt(market_condition)

            # Determine prompt type based on market
            if "volatile" in market_condition:
                prompt_type = "volatility"
            elif market_condition in ["bullish", "bearish"]:
                prompt_type = "momentum"
            else:
                prompt_type = "basic"

            # Generate factors (inner loop - generate)
            try:
                factors = self._generator.generate(
                    data,
                    prompt_type=prompt_type,
                    symbol=symbol,
                    n_factors=factors_per_iteration
                )
            except Exception as e:
                if verbose:
                    print(f"  Generation error: {e}")
                continue

            # Evaluate each factor (inner loop - evaluate)
            for factor in factors:
                try:
                    # Calculate factor values
                    factor_values = self._parser.evaluate(factor.expression, data)

                    # Evaluate performance
                    metrics = self._evaluator.evaluate(factor_values, returns)
                    quality = metrics.quality_score()

                    success = quality >= self.quality_threshold

                    if verbose:
                        status = "SUCCESS" if success else "fail"
                        print(f"  [{status}] {factor.name}: IC={metrics.ic:.4f}, Quality={quality:.1f}")

                    # Create experience
                    experience = Experience(
                        factor_expression=factor.expression,
                        factor_name=factor.name,
                        metrics=metrics.to_dict(),
                        market_condition=market_condition,
                        success=success,
                        notes=factor.description
                    )

                    # Add to knowledge base (outer loop - learn)
                    self.kb.add_experience(experience)

                    if success:
                        successful_factors.append({
                            "factor": factor,
                            "metrics": metrics,
                            "iteration": iteration + 1
                        })

                except Exception as e:
                    if verbose:
                        print(f"  [error] {factor.name}: {e}")

                    # Still learn from failures
                    experience = Experience(
                        factor_expression=factor.expression,
                        factor_name=factor.name,
                        metrics={"error": str(e)},
                        market_condition=market_condition,
                        success=False,
                        notes=f"Evaluation error: {e}"
                    )
                    self.kb.add_experience(experience)

        if verbose:
            print("\n" + "-" * 40)
            print(f"Mining complete!")
            print(f"Successful factors: {len(successful_factors)}")
            summary = self.kb.summary()
            print(f"Knowledge base: {summary['total']} experiences, "
                  f"{summary['success_rate']:.1%} success rate")

        return successful_factors

    def refine_factor(
        self,
        factor_expression: str,
        data: pd.DataFrame,
        n_variations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Refine an existing factor by generating variations.

        Args:
            factor_expression: Base factor to refine
            data: OHLCV DataFrame
            n_variations: Number of variations to try

        Returns:
            List of improved factor variations
        """
        self._ensure_components()

        prompt = f"""Improve this alpha factor by creating variations:

Original: {factor_expression}

Generate {n_variations} improved versions with different:
- Lookback periods
- Normalization methods
- Combination approaches

Return as JSON array:
[{{"name": "...", "expression": "...", "description": "..."}}]
"""
        response = self._generator._call_llm(prompt)
        factors = self._generator._parse_response(response)

        # Evaluate all variations
        returns = data["close"].pct_change().shift(-1)
        results = []

        for factor in factors:
            try:
                values = self._parser.evaluate(factor.expression, data)
                metrics = self._evaluator.evaluate(values, returns)

                results.append({
                    "factor": factor,
                    "metrics": metrics,
                    "improvement": metrics.quality_score()
                })
            except Exception:
                continue

        # Sort by quality
        results.sort(key=lambda x: x["improvement"], reverse=True)

        return results

    def get_recommendations(
        self,
        data: pd.DataFrame,
        n_recommendations: int = 5
    ) -> List[Experience]:
        """
        Get factor recommendations based on current market.

        Args:
            data: Current market data
            n_recommendations: Number of recommendations

        Returns:
            List of recommended past experiences
        """
        market_condition = self.classify_market(data)

        # Query knowledge base for successful factors in similar conditions
        recommendations = self.kb.query(
            market_condition=market_condition,
            success_only=True,
            limit=n_recommendations
        )

        return recommendations


if __name__ == "__main__":
    from data_loader import generate_synthetic_data

    print("LLM Alpha Mining - QuantAgent Demo")
    print("=" * 60)

    # Generate data
    print("\n1. Loading Data")
    print("-" * 40)
    data = generate_synthetic_data(["BTCUSDT"], days=300)
    btc_data = data["BTCUSDT"].ohlcv
    print(f"Data shape: {btc_data.shape}")

    # Initialize agent
    print("\n2. Initializing QuantAgent")
    print("-" * 40)
    agent = QuantAgent(
        quality_threshold=30.0,  # Lower threshold for demo
        model="mock"
    )

    # Classify market
    market = agent.classify_market(btc_data)
    print(f"Current market condition: {market}")

    # Run mining
    print("\n3. Mining for Alpha Factors")
    print("-" * 40)
    successful = agent.mine(
        btc_data,
        symbol="BTCUSDT",
        n_iterations=5,
        factors_per_iteration=3,
        verbose=True
    )

    # Show results
    print("\n4. Results Summary")
    print("-" * 40)
    print(f"\nTotal successful factors: {len(successful)}")

    if successful:
        print("\nTop factors found:")
        for i, result in enumerate(successful[:5], 1):
            factor = result["factor"]
            metrics = result["metrics"]
            print(f"\n  {i}. {factor.name}")
            print(f"     Expression: {factor.expression}")
            print(f"     IC: {metrics.ic:.4f}")
            print(f"     Sharpe: {metrics.sharpe_ratio:.2f}")
            print(f"     Quality: {metrics.quality_score():.1f}")

    # Knowledge base summary
    print("\n5. Knowledge Base Summary")
    print("-" * 40)
    summary = agent.kb.summary()
    print(f"Total experiences: {summary['total']}")
    print(f"Success rate: {summary['success_rate']:.1%}")

    if summary.get("best_patterns"):
        print("\nBest patterns:")
        for pattern, score in summary["best_patterns"][:5]:
            print(f"  - {pattern}: {score:.1%}")

    # Get recommendations
    print("\n6. Factor Recommendations for Current Market")
    print("-" * 40)
    recommendations = agent.get_recommendations(btc_data, n_recommendations=3)

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  {i}. {rec.factor_name}")
            print(f"     Expression: {rec.factor_expression}")
            print(f"     Historical IC: {rec.metrics.get('ic', 0):.4f}")
    else:
        print("  No recommendations available yet (need more experience)")

    print("\n" + "=" * 60)
    print("Demo complete!")
