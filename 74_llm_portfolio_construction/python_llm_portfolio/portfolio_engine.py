"""
LLM Portfolio Construction Engine

This module provides the core portfolio construction engine
that uses LLM analysis for asset scoring and allocation.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class enumeration."""
    EQUITY = "equity"
    CRYPTO = "crypto"
    BOND = "bond"
    COMMODITY = "commodity"
    CASH = "cash"


@dataclass
class Asset:
    """Represents a tradeable asset."""
    symbol: str
    name: str
    asset_class: AssetClass
    current_price: float
    market_cap: Optional[float] = None
    sector: Optional[str] = None


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
    metadata: Dict = field(default_factory=dict)

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
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def __str__(self) -> str:
        lines = ["Portfolio Allocation:"]
        for symbol, weight in sorted(self.weights.items(), key=lambda x: -x[1]):
            lines.append(f"  {symbol}: {weight:.2%}")
        if self.cash_weight > 0:
            lines.append(f"  CASH: {self.cash_weight:.2%}")
        return "\n".join(lines)


class LLMPortfolioEngine:
    """
    LLM-based portfolio construction engine.

    Uses LLM analysis to score assets and generate portfolio allocations.
    Can work with any OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None
    ):
        """
        Initialize the portfolio engine.

        Args:
            api_key: API key for LLM service
            model: Model name to use
            base_url: Custom base URL for API
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = openai.OpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client

    def analyze_assets(
        self,
        assets: List[Asset],
        market_data: Dict[str, Dict],
        news_data: List[str]
    ) -> List[AssetScore]:
        """
        Analyze assets and generate scores using LLM.

        Args:
            assets: List of assets to analyze
            market_data: Dict with market metrics per symbol
            news_data: List of recent news headlines

        Returns:
            List of AssetScore objects
        """
        client = self._get_client()

        # Prepare prompts
        asset_info = self._format_assets(assets)
        market_summary = self._format_market_data(market_data)
        news_summary = self._format_news(news_data)

        prompt = self._build_analysis_prompt(asset_info, market_summary, news_summary)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a quantitative analyst specializing in portfolio construction. Provide analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return self._parse_scores(result, assets)

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Return default scores if LLM fails
            return self._generate_default_scores(assets)

    def analyze_assets_mock(
        self,
        assets: List[Asset],
        market_data: Dict[str, Dict],
        news_data: List[str]
    ) -> List[AssetScore]:
        """
        Generate mock asset scores for testing without API.

        Args:
            assets: List of assets to analyze
            market_data: Dict with market metrics per symbol
            news_data: List of recent news headlines

        Returns:
            List of AssetScore objects
        """
        scores = []
        for asset in assets:
            # Generate semi-random but consistent scores based on symbol
            np.random.seed(hash(asset.symbol) % 2**32)

            # Use market data if available
            data = market_data.get(asset.symbol, {})
            momentum_boost = 1 if data.get("return_7d", 0) > 0 else -1

            fundamental = np.clip(np.random.normal(6.5, 1.5), 1, 10)
            momentum = np.clip(np.random.normal(6, 1.5) + momentum_boost, 1, 10)
            sentiment = np.clip(np.random.normal(6, 1.5), 1, 10)
            risk = np.clip(np.random.normal(5, 1.5), 1, 10)

            scores.append(AssetScore(
                symbol=asset.symbol,
                fundamental_score=round(fundamental, 1),
                momentum_score=round(momentum, 1),
                sentiment_score=round(sentiment, 1),
                risk_score=round(risk, 1),
                overall_score=round((fundamental + momentum + sentiment + (10-risk)) / 4, 1),
                reasoning=f"Mock analysis for {asset.name}",
                confidence="medium"
            ))

        return scores

    def generate_portfolio(
        self,
        scores: List[AssetScore],
        constraints: Optional[Dict] = None
    ) -> Portfolio:
        """
        Generate portfolio weights from asset scores.

        Args:
            scores: List of asset scores
            constraints: Portfolio constraints dict

        Returns:
            Portfolio object with weights
        """
        if constraints is None:
            constraints = {
                "max_weight": 0.30,
                "min_weight": 0.02,
                "max_assets": 10,
                "min_score": 4.0
            }

        # Filter assets by minimum score
        valid_scores = [s for s in scores if s.composite_score >= constraints.get("min_score", 4.0)]

        if not valid_scores:
            logger.warning("No assets meet minimum score. Using all assets.")
            valid_scores = scores

        # Sort by composite score
        valid_scores.sort(key=lambda x: x.composite_score, reverse=True)

        # Take top N assets
        max_assets = constraints.get("max_assets", 10)
        selected = valid_scores[:max_assets]

        # Calculate weights proportional to scores
        total_score = sum(s.composite_score for s in selected)

        if total_score == 0:
            # Equal weight if all scores are 0
            weights = {s.symbol: 1/len(selected) for s in selected}
        else:
            weights = {}
            for score in selected:
                raw_weight = score.composite_score / total_score
                # Apply constraints
                max_w = constraints.get("max_weight", 0.30)
                min_w = constraints.get("min_weight", 0.02)
                weight = max(min_w, min(max_w, raw_weight))
                weights[score.symbol] = weight

        # Normalize
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        return Portfolio(
            weights=weights,
            metadata={
                "method": "llm_score_weighted",
                "scores": {s.symbol: s.composite_score for s in selected}
            }
        )

    def _build_analysis_prompt(
        self,
        assets: str,
        market: str,
        news: str
    ) -> str:
        return f"""Analyze the following assets for portfolio construction.

ASSETS TO ANALYZE:
{assets}

RECENT MARKET DATA:
{market}

RECENT NEWS:
{news}

For each asset, provide scores (1-10) where 10 is best:
- fundamental_score: Quality of business/project fundamentals
- momentum_score: Price trend strength and technical outlook
- sentiment_score: Market sentiment and news tone
- risk_score: Risk level (10 = highest risk, 1 = lowest risk)
- overall_score: Your overall assessment
- reasoning: Brief 1-2 sentence explanation
- confidence: low/medium/high

Return a JSON object with a "scores" array containing an object for each asset with fields:
symbol, fundamental_score, momentum_score, sentiment_score, risk_score, overall_score, reasoning, confidence"""

    def _format_assets(self, assets: List[Asset]) -> str:
        lines = []
        for a in assets:
            info = f"- {a.symbol}: {a.name} ({a.asset_class.value}), Price: ${a.current_price:.2f}"
            if a.market_cap:
                info += f", Market Cap: ${a.market_cap:,.0f}"
            if a.sector:
                info += f", Sector: {a.sector}"
            lines.append(info)
        return "\n".join(lines)

    def _format_market_data(self, data: Dict[str, Dict]) -> str:
        if not data:
            return "No market data available"
        lines = []
        for symbol, info in data.items():
            ret_7d = info.get('return_7d', 0)
            vol = info.get('volatility', 0)
            lines.append(f"- {symbol}: 7-day return: {ret_7d:.1%}, Volatility: {vol:.1%}")
        return "\n".join(lines)

    def _format_news(self, news: List[str]) -> str:
        if not news:
            return "No recent news available"
        return "\n".join([f"- {n}" for n in news[:10]])

    def _parse_scores(self, data: Dict, assets: List[Asset]) -> List[AssetScore]:
        scores = []
        asset_map = {a.symbol: a for a in assets}

        for item in data.get("scores", []):
            symbol = item.get("symbol", "")
            if symbol in asset_map:
                scores.append(AssetScore(
                    symbol=symbol,
                    fundamental_score=float(item.get("fundamental_score", 5)),
                    momentum_score=float(item.get("momentum_score", 5)),
                    sentiment_score=float(item.get("sentiment_score", 5)),
                    risk_score=float(item.get("risk_score", 5)),
                    overall_score=float(item.get("overall_score", 5)),
                    reasoning=item.get("reasoning", ""),
                    confidence=item.get("confidence", "medium")
                ))

        return scores

    def _generate_default_scores(self, assets: List[Asset]) -> List[AssetScore]:
        """Generate default scores when LLM is unavailable."""
        return [
            AssetScore(
                symbol=a.symbol,
                fundamental_score=5.0,
                momentum_score=5.0,
                sentiment_score=5.0,
                risk_score=5.0,
                overall_score=5.0,
                reasoning="Default score (LLM unavailable)",
                confidence="low"
            )
            for a in assets
        ]


class MeanVarianceOptimizer:
    """
    Mean-variance portfolio optimization with LLM score integration.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        target_volatility: Optional[float] = None
    ):
        """
        Initialize optimizer.

        Args:
            risk_free_rate: Annual risk-free rate
            target_volatility: Optional target volatility constraint
        """
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        symbols: List[str],
        llm_scores: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None
    ) -> Tuple[Portfolio, Dict]:
        """
        Optimize portfolio weights.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            symbols: List of asset symbols
            llm_scores: Optional LLM composite scores to blend
            constraints: Portfolio constraints

        Returns:
            Tuple of (Portfolio, metrics dict)
        """
        n_assets = len(expected_returns)

        if constraints is None:
            constraints = {
                "max_weight": 0.30,
                "min_weight": 0.0,
                "long_only": True
            }

        # Blend LLM scores with expected returns if provided
        if llm_scores is not None and len(llm_scores) == n_assets:
            # Normalize LLM scores
            if np.std(llm_scores) > 0:
                normalized_scores = (llm_scores - np.mean(llm_scores)) / np.std(llm_scores)
            else:
                normalized_scores = np.zeros(n_assets)

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
            if port_vol == 0:
                return 1e10
            return -(port_return - self.risk_free_rate) / port_vol

        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]

        # Bounds
        max_w = constraints.get("max_weight", 0.30)
        min_w = constraints.get("min_weight", 0.0)

        if constraints.get("long_only", True):
            bounds = [(min_w, max_w) for _ in range(n_assets)]
        else:
            bounds = [(-max_w, max_w) for _ in range(n_assets)]

        # Optimize
        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )

        weights_array = result.x

        # Calculate metrics
        port_return = np.dot(weights_array, adjusted_returns)
        port_vol = np.sqrt(np.dot(weights_array.T, np.dot(covariance_matrix, weights_array)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Create portfolio
        weights_dict = {symbols[i]: max(0, weights_array[i]) for i in range(n_assets)}
        portfolio = Portfolio(
            weights=weights_dict,
            metadata={
                "method": "mean_variance",
                "optimization_success": result.success
            }
        )

        metrics = {
            "expected_return": port_return,
            "volatility": port_vol,
            "sharpe_ratio": sharpe,
            "optimization_success": result.success
        }

        return portfolio, metrics

    def risk_parity(
        self,
        covariance_matrix: np.ndarray,
        symbols: List[str],
        risk_budget: Optional[np.ndarray] = None
    ) -> Tuple[Portfolio, Dict]:
        """
        Risk parity portfolio allocation.

        Args:
            covariance_matrix: Covariance matrix of returns
            symbols: List of asset symbols
            risk_budget: Optional risk budget per asset

        Returns:
            Tuple of (Portfolio, metrics dict)
        """
        n_assets = covariance_matrix.shape[0]

        if risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets

        def risk_contribution_error(weights):
            weights = np.maximum(weights, 1e-10)  # Avoid division by zero
            port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            if port_vol == 0:
                return 1e10
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
            constraints=cons,
            options={'maxiter': 1000}
        )

        weights_array = result.x
        port_vol = np.sqrt(np.dot(weights_array.T, np.dot(covariance_matrix, weights_array)))

        # Create portfolio
        weights_dict = {symbols[i]: max(0, weights_array[i]) for i in range(n_assets)}
        portfolio = Portfolio(
            weights=weights_dict,
            metadata={
                "method": "risk_parity",
                "optimization_success": result.success
            }
        )

        metrics = {
            "volatility": port_vol,
            "optimization_success": result.success
        }

        return portfolio, metrics


# Example usage
if __name__ == "__main__":
    print("LLM Portfolio Engine Demo")
    print("=" * 50)

    # Create sample assets
    assets = [
        Asset("BTCUSDT", "Bitcoin", AssetClass.CRYPTO, 65000, market_cap=1.2e12),
        Asset("ETHUSDT", "Ethereum", AssetClass.CRYPTO, 3200, market_cap=380e9),
        Asset("SOLUSDT", "Solana", AssetClass.CRYPTO, 140, market_cap=60e9),
        Asset("BNBUSDT", "BNB", AssetClass.CRYPTO, 580, market_cap=85e9),
    ]

    # Sample market data
    market_data = {
        "BTCUSDT": {"return_7d": 0.05, "volatility": 0.50},
        "ETHUSDT": {"return_7d": 0.03, "volatility": 0.55},
        "SOLUSDT": {"return_7d": -0.02, "volatility": 0.70},
        "BNBUSDT": {"return_7d": 0.01, "volatility": 0.45},
    }

    # Sample news
    news = [
        "Bitcoin hits new all-time high on institutional demand",
        "Ethereum upgrade successful, gas fees reduced",
        "Solana network experiences brief outage",
        "BNB chain sees increased DeFi activity",
    ]

    # Initialize engine (mock mode for demo)
    engine = LLMPortfolioEngine()

    # Analyze assets using mock (no API key needed)
    print("\nAnalyzing assets...")
    scores = engine.analyze_assets_mock(assets, market_data, news)

    print("\nAsset Scores:")
    for score in scores:
        print(f"  {score.symbol}:")
        print(f"    Fundamental: {score.fundamental_score:.1f}")
        print(f"    Momentum: {score.momentum_score:.1f}")
        print(f"    Sentiment: {score.sentiment_score:.1f}")
        print(f"    Risk: {score.risk_score:.1f}")
        print(f"    Composite: {score.composite_score:.2f}")

    # Generate portfolio
    print("\nGenerating portfolio...")
    portfolio = engine.generate_portfolio(scores, constraints={
        "max_weight": 0.35,
        "min_weight": 0.10,
        "max_assets": 4,
        "min_score": 4.0
    })

    print(f"\n{portfolio}")

    # Mean-variance optimization example
    print("\n" + "=" * 50)
    print("Mean-Variance Optimization Demo")

    # Sample data
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    expected_returns = np.array([0.30, 0.25, 0.20, 0.15])  # Annual
    cov_matrix = np.array([
        [0.25, 0.15, 0.10, 0.08],
        [0.15, 0.30, 0.12, 0.10],
        [0.10, 0.12, 0.45, 0.15],
        [0.08, 0.10, 0.15, 0.20],
    ])

    optimizer = MeanVarianceOptimizer(risk_free_rate=0.04)

    # With LLM scores
    llm_scores = np.array([s.composite_score for s in scores])
    mv_portfolio, mv_metrics = optimizer.optimize(
        expected_returns, cov_matrix, symbols,
        llm_scores=llm_scores,
        constraints={"max_weight": 0.40, "min_weight": 0.05}
    )

    print(f"\n{mv_portfolio}")
    print(f"\nExpected Return: {mv_metrics['expected_return']:.2%}")
    print(f"Volatility: {mv_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {mv_metrics['sharpe_ratio']:.2f}")
