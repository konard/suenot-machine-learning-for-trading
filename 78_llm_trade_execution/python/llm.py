"""
LLM adapter for intelligent execution decisions.

This module provides:
- LLM integration for execution optimization
- Market state analysis
- Execution context management
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import aiohttp

from .data import OrderBook
from .execution import ParentOrder, Side


class ExecutionAction(Enum):
    """Execution actions recommended by LLM."""
    EXECUTE = "Execute"
    WAIT = "Wait"
    CONTINUE = "Continue"
    ACCELERATE = "Accelerate"
    DECELERATE = "Decelerate"
    PAUSE = "Pause"
    CANCEL = "Cancel"


@dataclass
class LlmConfig:
    """LLM configuration."""
    api_url: str = "https://api.openai.com/v1/chat/completions"
    api_key: str = ""
    model: str = "gpt-4"
    max_tokens: int = 512
    temperature: float = 0.1
    timeout: int = 30000

    @classmethod
    def openai(cls, api_key: str) -> "LlmConfig":
        """Create config for OpenAI."""
        return cls(api_key=api_key)

    @classmethod
    def anthropic(cls, api_key: str) -> "LlmConfig":
        """Create config for Anthropic Claude."""
        return cls(
            api_url="https://api.anthropic.com/v1/messages",
            api_key=api_key,
            model="claude-3-sonnet-20240229",
        )

    @classmethod
    def local(cls, url: str, model: str) -> "LlmConfig":
        """Create config for local LLM."""
        return cls(api_url=url, model=model, api_key="")


@dataclass
class LlmDecision:
    """LLM execution decision."""
    action: ExecutionAction = ExecutionAction.CONTINUE
    quantity_fraction: float = 0.0
    aggressiveness: float = 0.0
    confidence: float = 0.5
    reasoning: str = "Default decision"
    urgency_adjustment: float = 0.0


@dataclass
class MarketState:
    """Market state for LLM analysis."""
    mid_price: float = 0.0
    spread_bps: float = 0.0
    imbalance: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    price_change_pct: float = 0.0
    volatility: float = 0.0
    volume_ratio: float = 1.0
    funding_rate: Optional[float] = None
    oi_change_pct: Optional[float] = None

    @classmethod
    def from_orderbook(cls, orderbook: OrderBook) -> "MarketState":
        """Create from order book."""
        return cls(
            mid_price=orderbook.mid_price() or 0.0,
            spread_bps=orderbook.spread_bps() or 0.0,
            imbalance=orderbook.imbalance(10),
            bid_depth=orderbook.bid_depth(10),
            ask_depth=orderbook.ask_depth(10),
        )


@dataclass
class ExecutionContext:
    """Execution context for LLM."""
    side: Side
    total_quantity: float
    filled_quantity: float
    remaining_time: int
    target_participation: float
    actual_participation: float
    vwap_slippage_bps: float
    is_bps: float
    urgency: float


class LlmAdapter:
    """LLM adapter for execution decisions."""

    def __init__(self, config: LlmConfig):
        self.config = config

    @classmethod
    def openai(cls, api_key: str) -> "LlmAdapter":
        """Create adapter for OpenAI."""
        return cls(LlmConfig.openai(api_key))

    @classmethod
    def anthropic(cls, api_key: str) -> "LlmAdapter":
        """Create adapter for Anthropic."""
        return cls(LlmConfig.anthropic(api_key))

    def _build_prompt(
        self, market: MarketState, context: ExecutionContext
    ) -> str:
        """Build the execution prompt."""
        fill_pct = (context.filled_quantity / context.total_quantity) * 100 if context.total_quantity > 0 else 0

        funding_str = f"- Funding Rate: {market.funding_rate * 100:.4f}%" if market.funding_rate else ""
        oi_str = f"- OI Change: {market.oi_change_pct:.2f}%" if market.oi_change_pct else ""

        return f"""You are an expert algorithmic trading execution optimizer. Analyze the current market conditions and execution progress to recommend the next action.

## Market State
- Mid Price: {market.mid_price:.2f}
- Spread: {market.spread_bps:.2f} bps
- Order Book Imbalance: {market.imbalance:.2f} (positive = more bids)
- Bid Depth (10 levels): {market.bid_depth:.4f}
- Ask Depth (10 levels): {market.ask_depth:.4f}
- Recent Price Change: {market.price_change_pct:.2f}%
- Volatility: {market.volatility:.2f}%
- Volume Ratio (vs avg): {market.volume_ratio:.2f}x
{funding_str}
{oi_str}

## Execution Context
- Side: {context.side.value}
- Total Quantity: {context.total_quantity:.4f}
- Filled: {context.filled_quantity:.4f} ({fill_pct:.1f}%)
- Remaining Time: {context.remaining_time} seconds
- Target Participation: {context.target_participation * 100:.1f}%
- Actual Participation: {context.actual_participation * 100:.1f}%
- VWAP Slippage: {context.vwap_slippage_bps:.2f} bps
- Implementation Shortfall: {context.is_bps:.2f} bps
- Urgency: {context.urgency:.2f}

## Task
Recommend the optimal execution action. Consider:
1. Market impact vs opportunity cost tradeoff
2. Current vs target progress
3. Market conditions (spread, depth, volatility)
4. Time pressure

Respond in JSON format:
{{
    "action": "Execute|Wait|Continue|Accelerate|Decelerate|Pause|Cancel",
    "quantity_fraction": <0.0 to 1.0>,
    "aggressiveness": <-1.0 to 1.0>,
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>",
    "urgency_adjustment": <-0.2 to 0.2>
}}"""

    def _parse_response(self, response: str) -> LlmDecision:
        """Parse LLM response into decision."""
        # Find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")

        json_str = response[start:end]
        data = json.loads(json_str)

        action_str = data.get("action", "Continue")
        action = ExecutionAction(action_str)

        return LlmDecision(
            action=action,
            quantity_fraction=float(data.get("quantity_fraction", 0.0)),
            aggressiveness=float(data.get("aggressiveness", 0.0)),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            urgency_adjustment=float(data.get("urgency_adjustment", 0.0)),
        )

    async def get_decision(
        self, market: MarketState, context: ExecutionContext
    ) -> LlmDecision:
        """Get execution decision from LLM."""
        prompt = self._build_prompt(market, context)

        request_body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert algorithmic trading execution optimizer. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

            async with session.post(
                self.config.api_url,
                json=request_body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout / 1000),
            ) as response:
                if response.status == 429:
                    raise Exception("Rate limit exceeded")

                if not response.ok:
                    error_text = await response.text()
                    raise Exception(f"API error: {error_text}")

                data = await response.json()

                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                return self._parse_response(content)

    def get_heuristic_decision(
        self, market: MarketState, context: ExecutionContext
    ) -> LlmDecision:
        """Get a simple heuristic decision (fallback)."""
        remaining_fraction = 1.0 - (
            context.filled_quantity / context.total_quantity
            if context.total_quantity > 0
            else 0
        )
        time_fraction = context.remaining_time / 3600.0  # Normalize to 1 hour

        # Determine if behind or ahead of schedule
        expected_fill = 1.0 - time_fraction
        actual_fill = (
            context.filled_quantity / context.total_quantity
            if context.total_quantity > 0
            else 0
        )
        progress_diff = actual_fill - expected_fill

        # Adjust based on market conditions
        if market.spread_bps < 5.0:
            spread_factor = 1.2
        elif market.spread_bps > 20.0:
            spread_factor = 0.8
        else:
            spread_factor = 1.0

        # Consider order book imbalance
        if context.side == Side.BUY:
            if market.imbalance > 0.3:
                imbalance_factor = 0.8
            elif market.imbalance < -0.3:
                imbalance_factor = 1.2
            else:
                imbalance_factor = 1.0
        else:
            if market.imbalance < -0.3:
                imbalance_factor = 0.8
            elif market.imbalance > 0.3:
                imbalance_factor = 1.2
            else:
                imbalance_factor = 1.0

        # Calculate quantity fraction
        remaining_slices = max(1.0, context.remaining_time / 60.0)
        base_fraction = remaining_fraction / remaining_slices
        quantity_fraction = min(0.25, max(0.01, base_fraction * spread_factor * imbalance_factor))

        # Determine action
        if progress_diff < -0.1:
            action = ExecutionAction.ACCELERATE
            aggressiveness = 0.3
        elif progress_diff > 0.1:
            action = ExecutionAction.DECELERATE
            aggressiveness = -0.3
        elif market.spread_bps > 30.0:
            action = ExecutionAction.WAIT
            aggressiveness = -0.5
        else:
            action = ExecutionAction.EXECUTE
            aggressiveness = 0.0

        return LlmDecision(
            action=action,
            quantity_fraction=quantity_fraction,
            aggressiveness=aggressiveness,
            confidence=0.7,
            reasoning=f"Heuristic: progress_diff={progress_diff:.2f}, spread={market.spread_bps:.1f}bps, imbalance={market.imbalance:.2f}",
            urgency_adjustment=max(-0.1, min(0.1, progress_diff)),
        )


def build_execution_context(
    order: ParentOrder,
    target_participation: float,
    actual_participation: float,
    vwap: float,
    market_vwap: float,
) -> ExecutionContext:
    """Build execution context from parent order."""
    vwap_slippage_bps = 0.0
    if market_vwap > 0 and vwap > 0:
        vwap_slippage_bps = ((vwap - market_vwap) / market_vwap * 10000.0) * order.side.sign()

    is_bps = 0.0
    if order.arrival_price and order.average_price and order.arrival_price > 0:
        is_bps = (
            (order.average_price - order.arrival_price) / order.arrival_price * 10000.0
        ) * order.side.sign()

    return ExecutionContext(
        side=order.side,
        total_quantity=order.total_quantity,
        filled_quantity=order.filled_quantity,
        remaining_time=order.remaining_time(),
        target_participation=target_participation,
        actual_participation=actual_participation,
        vwap_slippage_bps=vwap_slippage_bps,
        is_bps=is_bps,
        urgency=order.urgency,
    )
