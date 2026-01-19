"""
Market Regime Detection using Prompt Engineering

This module provides market regime classification capabilities
using LLM analysis combined with quantitative indicators.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio

from .prompts.regime import REGIME_PROMPTS


class MarketRegime(Enum):
    """Market regime classification."""
    RISK_ON_TRENDING = "RISK_ON_TRENDING"  # Strong uptrend, low volatility
    RISK_ON_VOLATILE = "RISK_ON_VOLATILE"  # Uptrend with high volatility
    RISK_OFF_TRENDING = "RISK_OFF_TRENDING"  # Strong downtrend
    RISK_OFF_PANIC = "RISK_OFF_PANIC"  # Crash mode, extreme fear
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
    raw_response: Optional[str] = None


class MarketRegimeDetector:
    """
    Detect market regimes using LLM analysis.

    Combines quantitative indicators with LLM reasoning
    for robust regime classification.

    Example:
        >>> detector = MarketRegimeDetector(llm_client)
        >>> analysis = await detector.detect_regime(market_data)
        >>> print(f"Current regime: {analysis.regime.value}")
    """

    # Strategy recommendations for each regime
    REGIME_STRATEGIES = {
        MarketRegime.RISK_ON_TRENDING: {
            "preferred": ["trend_following", "momentum", "buy_dips"],
            "avoid": ["short_selling", "volatility_buying"],
            "position_multiplier": 1.2,
            "sectors": ["technology", "consumer_discretionary", "industrials"]
        },
        MarketRegime.RISK_ON_VOLATILE: {
            "preferred": ["mean_reversion", "volatility_selling", "range_trading"],
            "avoid": ["trend_following", "leveraged_positions"],
            "position_multiplier": 0.8,
            "sectors": ["technology", "healthcare"]
        },
        MarketRegime.RISK_OFF_TRENDING: {
            "preferred": ["short_selling", "put_buying", "defensive_rotation"],
            "avoid": ["buy_dips", "momentum_long"],
            "position_multiplier": 0.6,
            "sectors": ["utilities", "consumer_staples", "healthcare"]
        },
        MarketRegime.RISK_OFF_PANIC: {
            "preferred": ["cash", "tail_hedges", "volatility_buying"],
            "avoid": ["any_long_exposure", "leverage"],
            "position_multiplier": 0.3,
            "sectors": ["cash", "treasuries", "gold"]
        },
        MarketRegime.RANGING: {
            "preferred": ["mean_reversion", "premium_selling", "pairs_trading"],
            "avoid": ["trend_following", "breakout_trading"],
            "position_multiplier": 0.7,
            "sectors": ["high_dividend", "value"]
        },
        MarketRegime.TRANSITIONAL: {
            "preferred": ["reduce_exposure", "straddles", "diversification"],
            "avoid": ["concentrated_positions", "directional_bets"],
            "position_multiplier": 0.5,
            "sectors": ["balanced"]
        }
    }

    def __init__(
        self,
        llm_client,
        prompt_type: str = "comprehensive_regime",
        temperature: float = 0.3
    ):
        """
        Initialize regime detector.

        Args:
            llm_client: LLM client instance
            prompt_type: Type of prompt to use
            temperature: LLM temperature
        """
        self.llm = llm_client
        self.prompt_type = prompt_type
        self.temperature = temperature

        if prompt_type not in REGIME_PROMPTS:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        self.prompt_template = REGIME_PROMPTS[prompt_type]

    async def detect_regime(self, market_data: Dict[str, Any]) -> RegimeAnalysis:
        """
        Detect current market regime.

        Args:
            market_data: Comprehensive market data dict

        Returns:
            RegimeAnalysis with classification and recommendations
        """
        # Format prompt with market data
        prompt = self.prompt_template.format(**market_data)

        response = await self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=1000
        )

        return self._parse_response(response)

    async def detect_crypto_regime(self, crypto_data: Dict[str, Any]) -> RegimeAnalysis:
        """
        Detect cryptocurrency market regime.

        Args:
            crypto_data: Crypto-specific market data

        Returns:
            RegimeAnalysis for crypto markets
        """
        prompt = REGIME_PROMPTS["crypto_regime"].format(**crypto_data)

        response = await self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=800
        )

        return self._parse_crypto_response(response)

    async def detect_with_confidence(
        self,
        market_data: Dict[str, Any],
        num_samples: int = 3
    ) -> RegimeAnalysis:
        """
        Detect regime with confidence validation using multiple samples.

        Args:
            market_data: Market data dict
            num_samples: Number of samples for consistency

        Returns:
            RegimeAnalysis with validated confidence
        """
        prompt = self.prompt_template.format(**market_data)

        # Generate multiple samples
        tasks = [
            self.llm.complete(prompt, temperature=0.5, max_tokens=1000)
            for _ in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse all responses
        analyses = []
        for response in responses:
            if isinstance(response, str):
                try:
                    analysis = self._parse_response(response)
                    analyses.append(analysis)
                except Exception:
                    continue

        if not analyses:
            return RegimeAnalysis(
                regime=MarketRegime.TRANSITIONAL,
                confidence=0,
                key_drivers=["Unable to determine"],
                recommended_strategies=["reduce_exposure"],
                avoid_strategies=["concentrated_positions"],
                position_sizing_adj="REDUCE",
                next_regime_probability={},
                reasoning="Failed to generate consistent regime analysis"
            )

        # Majority vote for regime
        regime_counts = {}
        for a in analyses:
            regime_counts[a.regime] = regime_counts.get(a.regime, 0) + 1

        majority_regime = max(regime_counts, key=regime_counts.get)
        agreement_ratio = regime_counts[majority_regime] / len(analyses)

        # Use first matching analysis as base
        base_analysis = next(a for a in analyses if a.regime == majority_regime)

        return RegimeAnalysis(
            regime=majority_regime,
            confidence=base_analysis.confidence * agreement_ratio,
            key_drivers=base_analysis.key_drivers,
            recommended_strategies=base_analysis.recommended_strategies,
            avoid_strategies=base_analysis.avoid_strategies,
            position_sizing_adj=base_analysis.position_sizing_adj,
            next_regime_probability=base_analysis.next_regime_probability,
            reasoning=f"[Agreement: {agreement_ratio:.0%}] {base_analysis.reasoning}"
        )

    def get_strategy_params(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get strategy parameters for given regime.

        Args:
            regime: Market regime

        Returns:
            Strategy parameters dict
        """
        return self.REGIME_STRATEGIES.get(
            regime,
            self.REGIME_STRATEGIES[MarketRegime.RANGING]
        )

    def _parse_response(self, response: str) -> RegimeAnalysis:
        """Parse LLM response into RegimeAnalysis."""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found")

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            # Map regime string to enum
            regime_str = data.get("regime", "TRANSITIONAL")
            regime = MarketRegime(regime_str)

            # Get default strategies if not provided
            default_strategies = self.REGIME_STRATEGIES.get(regime, {})

            return RegimeAnalysis(
                regime=regime,
                confidence=float(data.get("confidence", 50)),
                key_drivers=data.get("key_drivers", []),
                recommended_strategies=data.get(
                    "recommended_strategies",
                    default_strategies.get("preferred", [])
                ),
                avoid_strategies=data.get(
                    "avoid_strategies",
                    default_strategies.get("avoid", [])
                ),
                position_sizing_adj=data.get("position_sizing", "MAINTAIN"),
                next_regime_probability=data.get("next_regime_probabilities", {}),
                reasoning=data.get("reasoning", ""),
                raw_response=response
            )

        except Exception as e:
            return RegimeAnalysis(
                regime=MarketRegime.TRANSITIONAL,
                confidence=0,
                key_drivers=[f"Parse error: {e}"],
                recommended_strategies=["reduce_exposure"],
                avoid_strategies=["concentrated_positions"],
                position_sizing_adj="REDUCE",
                next_regime_probability={},
                reasoning=f"Failed to parse: {e}",
                raw_response=response
            )

    def _parse_crypto_response(self, response: str) -> RegimeAnalysis:
        """Parse crypto regime response."""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            data = json.loads(response[start_idx:end_idx])

            # Map crypto-specific regimes to standard regimes
            crypto_regime_map = {
                "BULL_RUN": MarketRegime.RISK_ON_TRENDING,
                "ACCUMULATION": MarketRegime.RANGING,
                "DISTRIBUTION": MarketRegime.TRANSITIONAL,
                "BEAR_MARKET": MarketRegime.RISK_OFF_TRENDING,
                "ALTSEASON": MarketRegime.RISK_ON_VOLATILE,
                "BTC_DOMINANCE": MarketRegime.RISK_ON_TRENDING
            }

            crypto_regime = data.get("regime", "ACCUMULATION")
            regime = crypto_regime_map.get(crypto_regime, MarketRegime.RANGING)

            # Build recommendations from allocation
            allocation = data.get("recommended_allocation", {})
            recommendations = []
            if allocation.get("btc_pct", 0) > 50:
                recommendations.append("btc_focus")
            if allocation.get("altcoin_pct", 0) > 30:
                recommendations.append("altcoin_rotation")
            if allocation.get("stablecoin_pct", 0) > 30:
                recommendations.append("defensive_cash")

            return RegimeAnalysis(
                regime=regime,
                confidence=float(data.get("confidence", 50)),
                key_drivers=[
                    f"BTC Outlook: {data.get('btc_outlook', 'NEUTRAL')}",
                    f"Altcoin Outlook: {data.get('altcoin_outlook', 'NEUTRAL')}"
                ],
                recommended_strategies=recommendations or ["balanced_allocation"],
                avoid_strategies=["high_leverage"] if allocation.get("stablecoin_pct", 0) > 20 else [],
                position_sizing_adj="MAINTAIN",
                next_regime_probability={},
                reasoning=data.get("reasoning", ""),
                raw_response=response
            )

        except Exception as e:
            return RegimeAnalysis(
                regime=MarketRegime.TRANSITIONAL,
                confidence=0,
                key_drivers=[f"Parse error: {e}"],
                recommended_strategies=["reduce_exposure"],
                avoid_strategies=["leverage"],
                position_sizing_adj="REDUCE",
                next_regime_probability={},
                reasoning=f"Failed to parse: {e}",
                raw_response=response
            )


# Example usage
async def main():
    """Demo of regime detection."""
    from .llm_client import MockLLMClient

    mock_responses = {
        "regime": json.dumps({
            "regime": "RISK_ON_TRENDING",
            "confidence": 78,
            "key_drivers": [
                "Strong equity momentum across indices",
                "Low VIX with contango term structure",
                "Positive fund flows"
            ],
            "recommended_strategies": ["trend_following", "buy_dips"],
            "avoid_strategies": ["short_selling", "volatility_buying"],
            "position_sizing": "INCREASE",
            "next_regime_probabilities": {
                "RISK_ON_TRENDING": 0.65,
                "RISK_ON_VOLATILE": 0.25,
                "TRANSITIONAL": 0.10
            },
            "reasoning": "Market shows classic risk-on characteristics with broad participation and positive sentiment."
        })
    }

    client = MockLLMClient(responses=mock_responses)
    detector = MarketRegimeDetector(client)

    market_data = {
        "sp500_price": 4785.50, "sp500_change": 2.5, "sp500_ytd": 8.2,
        "nasdaq_price": 15050.25, "nasdaq_change": 3.1,
        "russell_price": 2025.30, "russell_change": 1.8,
        "vix": 14.5, "vix_avg": 16.2, "vix_term_structure": "Contango",
        "realized_vol": 12.3,
        "treasury_10y": 4.25, "treasury_2y": 4.65, "yield_curve": -40,
        "credit_spread": 110,
        "dxy": 104.2, "dxy_change": 0.5,
        "gold": 2045, "gold_change": 1.2,
        "oil": 78.50, "oil_change": -2.1,
        "adv_dec_ratio": 1.8, "pct_above_200": 65, "high_low_diff": 150,
        "fund_flows": "$15B inflow", "put_call": 0.85, "aaii_spread": 12.5
    }

    analysis = await detector.detect_regime(market_data)

    print("Market Regime Detection Demo")
    print("=" * 50)
    print(f"Detected Regime: {analysis.regime.value}")
    print(f"Confidence: {analysis.confidence}%")
    print(f"\nKey Drivers:")
    for driver in analysis.key_drivers:
        print(f"  - {driver}")
    print(f"\nRecommended Strategies: {analysis.recommended_strategies}")
    print(f"Avoid: {analysis.avoid_strategies}")
    print(f"Position Sizing: {analysis.position_sizing_adj}")
    print(f"\nReasoning: {analysis.reasoning}")

    # Get strategy parameters
    params = detector.get_strategy_params(analysis.regime)
    print(f"\nStrategy Parameters:")
    print(f"  Position Multiplier: {params['position_multiplier']}x")
    print(f"  Preferred Sectors: {params['sectors']}")


if __name__ == "__main__":
    asyncio.run(main())
