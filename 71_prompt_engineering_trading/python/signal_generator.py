"""
Trading Signal Generation using Prompt Engineering

This module provides trading signal generation capabilities
using various prompt engineering techniques.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import asyncio

from .prompts.signals import SIGNAL_PROMPTS


class SignalDirection(Enum):
    """Trading signal direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class SignalStrength(Enum):
    """Trading signal strength."""
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
    entry_type: str  # "MARKET" or "LIMIT"
    stop_loss: float
    take_profit: List[float]
    position_size_pct: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[str] = None


class PromptBasedSignalGenerator:
    """
    Generate trading signals using engineered prompts.

    Combines multiple prompt techniques:
    - Role-based (quant analyst persona)
    - Chain-of-thought (systematic reasoning)
    - Few-shot (output format consistency)

    Example:
        >>> generator = PromptBasedSignalGenerator(llm_client)
        >>> signal = await generator.generate_signal("BTCUSDT", market_data)
        >>> print(f"{signal.direction.value} with {signal.confidence}% confidence")
    """

    def __init__(
        self,
        llm_client,
        prompt_type: str = "comprehensive_signal",
        temperature: float = 0.2
    ):
        """
        Initialize signal generator.

        Args:
            llm_client: LLM client instance
            prompt_type: Type of prompt to use
            temperature: LLM temperature (lower = more consistent)
        """
        self.llm = llm_client
        self.prompt_type = prompt_type
        self.temperature = temperature

        if prompt_type not in SIGNAL_PROMPTS:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        self.prompt_template = SIGNAL_PROMPTS[prompt_type]

    async def generate_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> TradingSignal:
        """
        Generate trading signal from market data.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "AAPL")
            market_data: Dict containing price, technicals, volume, etc.

        Returns:
            TradingSignal with complete entry/exit plan
        """
        # Prepare prompt data
        prompt_data = self._prepare_prompt_data(symbol, market_data)

        prompt = self.prompt_template.format(**prompt_data)

        response = await self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=1000
        )

        return self._parse_signal_response(symbol, response, market_data)

    async def generate_news_signal(
        self,
        symbol: str,
        news_text: str,
        current_price: float,
        pre_news_price: float,
        source: str = "unknown",
        category: str = "general"
    ) -> TradingSignal:
        """
        Generate trading signal from breaking news.

        Args:
            symbol: Trading symbol
            news_text: The news content
            current_price: Current market price
            pre_news_price: Price before news
            source: News source
            category: News category

        Returns:
            TradingSignal based on news analysis
        """
        prompt = SIGNAL_PROMPTS["news_driven"].format(
            symbol=symbol,
            current_price=current_price,
            pre_news_price=pre_news_price,
            news_text=news_text,
            source=source,
            timestamp=datetime.now().isoformat(),
            category=category
        )

        response = await self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=800
        )

        return self._parse_news_signal_response(symbol, response, current_price)

    async def generate_multi_timeframe_signal(
        self,
        symbol: str,
        timeframe_data: Dict[str, Dict],
        current_price: float,
        sentiment: str = "NEUTRAL"
    ) -> TradingSignal:
        """
        Generate signal using multi-timeframe analysis.

        Args:
            symbol: Trading symbol
            timeframe_data: Dict with keys "short", "medium", "long"
            current_price: Current price
            sentiment: Overall market sentiment

        Returns:
            TradingSignal with timeframe alignment analysis
        """
        prompt = SIGNAL_PROMPTS["multi_timeframe"].format(
            symbol=symbol,
            short_trend=timeframe_data["short"]["trend"],
            short_momentum=timeframe_data["short"]["momentum"],
            short_level=timeframe_data["short"]["level"],
            medium_trend=timeframe_data["medium"]["trend"],
            medium_momentum=timeframe_data["medium"]["momentum"],
            medium_level=timeframe_data["medium"]["level"],
            long_trend=timeframe_data["long"]["trend"],
            long_momentum=timeframe_data["long"]["momentum"],
            long_level=timeframe_data["long"]["level"],
            current_price=current_price,
            sentiment=sentiment
        )

        response = await self.llm.complete(
            prompt,
            temperature=self.temperature,
            max_tokens=800
        )

        return self._parse_mtf_signal_response(symbol, response, current_price)

    async def self_consistent_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        num_samples: int = 5
    ) -> TradingSignal:
        """
        Generate signal using self-consistency for higher reliability.

        Args:
            symbol: Trading symbol
            market_data: Market data dict
            num_samples: Number of samples for consistency check

        Returns:
            TradingSignal with consensus-based confidence
        """
        prompt_data = self._prepare_prompt_data(symbol, market_data)
        prompt = self.prompt_template.format(**prompt_data)

        # Generate multiple samples
        tasks = [
            self.llm.complete(prompt, temperature=0.7, max_tokens=1000)
            for _ in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse all responses
        signals = []
        for response in responses:
            if isinstance(response, str):
                try:
                    signal = self._parse_signal_response(symbol, response, market_data)
                    signals.append(signal)
                except Exception:
                    continue

        if not signals:
            return TradingSignal(
                symbol=symbol,
                direction=SignalDirection.FLAT,
                strength=SignalStrength.WEAK,
                confidence=0,
                entry_price=None,
                entry_type="MARKET",
                stop_loss=0,
                take_profit=[],
                position_size_pct=0,
                reasoning="Failed to generate consistent signals"
            )

        # Majority vote for direction
        direction_counts = {}
        for s in signals:
            direction_counts[s.direction] = direction_counts.get(s.direction, 0) + 1

        majority_direction = max(direction_counts, key=direction_counts.get)
        agreement_ratio = direction_counts[majority_direction] / len(signals)

        # Average metrics from agreeing signals
        agreeing_signals = [s for s in signals if s.direction == majority_direction]

        avg_confidence = sum(s.confidence for s in agreeing_signals) / len(agreeing_signals)
        avg_stop_loss = sum(s.stop_loss for s in agreeing_signals) / len(agreeing_signals)
        avg_position_size = sum(s.position_size_pct for s in agreeing_signals) / len(agreeing_signals)

        # Collect all take profit levels and average
        all_tps = []
        for s in agreeing_signals:
            all_tps.extend(s.take_profit)
        avg_take_profit = sorted(set(all_tps))[:2] if all_tps else []

        return TradingSignal(
            symbol=symbol,
            direction=majority_direction,
            strength=self._calculate_strength(avg_confidence * agreement_ratio),
            confidence=avg_confidence * agreement_ratio,
            entry_price=agreeing_signals[0].entry_price,
            entry_type=agreeing_signals[0].entry_type,
            stop_loss=avg_stop_loss,
            take_profit=avg_take_profit,
            position_size_pct=avg_position_size * agreement_ratio,
            reasoning=f"[Consensus: {agreement_ratio:.0%}] {agreeing_signals[0].reasoning}"
        )

    def _prepare_prompt_data(self, symbol: str, market_data: Dict) -> Dict:
        """Prepare data for prompt formatting."""
        price = market_data.get("price", {})
        technicals = market_data.get("technicals", {})
        volume = market_data.get("volume", {})

        current_price = price.get("current", 0)
        sma_20 = technicals.get("sma_20", current_price)
        sma_50 = technicals.get("sma_50", current_price)
        sma_200 = technicals.get("sma_200", current_price)
        atr = technicals.get("atr", current_price * 0.02)

        return {
            "symbol": symbol,
            "current_price": current_price,
            "change_24h": price.get("change_24h", 0),
            "change_7d": price.get("change_7d", 0),
            "change_30d": price.get("change_30d", 0),
            "rsi": technicals.get("rsi", 50),
            "macd_signal": "BULLISH" if technicals.get("macd", 0) > 0 else "BEARISH",
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "atr": atr,
            "atr_pct": (atr / current_price * 100) if current_price > 0 else 0,
            "volume": volume.get("current", 0),
            "avg_volume": volume.get("average", 1),
            "volume_ratio": volume.get("current", 0) / max(volume.get("average", 1), 1),
            "news_sentiment": market_data.get("sentiment", "NEUTRAL"),
            "market_regime": market_data.get("regime", "UNKNOWN")
        }

    def _parse_signal_response(
        self,
        symbol: str,
        response: str,
        market_data: Dict
    ) -> TradingSignal:
        """Parse signal from LLM response."""
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found")

            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)

            # Parse entry info
            entry = data.get("entry", {})
            if isinstance(entry, dict):
                entry_type = entry.get("type", "MARKET")
                entry_price = entry.get("price")
            else:
                entry_type = "MARKET"
                entry_price = None

            # Parse take profit
            take_profit = data.get("take_profit", [])
            if not isinstance(take_profit, list):
                take_profit = [take_profit] if take_profit else []

            return TradingSignal(
                symbol=symbol,
                direction=SignalDirection(data.get("direction", "FLAT")),
                strength=SignalStrength(data.get("strength", "MODERATE")),
                confidence=float(data.get("confidence", 50)),
                entry_price=float(entry_price) if entry_price else None,
                entry_type=entry_type,
                stop_loss=float(data.get("stop_loss", 0)),
                take_profit=[float(tp) for tp in take_profit],
                position_size_pct=float(data.get("position_size_pct", 5)),
                reasoning=data.get("reasoning", ""),
                raw_response=response
            )

        except Exception as e:
            # Fallback signal
            current_price = market_data.get("price", {}).get("current", 0)
            return TradingSignal(
                symbol=symbol,
                direction=SignalDirection.FLAT,
                strength=SignalStrength.WEAK,
                confidence=0,
                entry_price=None,
                entry_type="MARKET",
                stop_loss=current_price * 0.95 if current_price else 0,
                take_profit=[],
                position_size_pct=0,
                reasoning=f"Parse error: {e}",
                raw_response=response
            )

    def _parse_news_signal_response(
        self,
        symbol: str,
        response: str,
        current_price: float
    ) -> TradingSignal:
        """Parse news-driven signal response."""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            data = json.loads(response[start_idx:end_idx])

            signal_data = data.get("signal", {})

            return TradingSignal(
                symbol=symbol,
                direction=SignalDirection(signal_data.get("direction", "FLAT")),
                strength=self._calculate_strength(signal_data.get("confidence", 50)),
                confidence=float(signal_data.get("confidence", 50)),
                entry_price=current_price,
                entry_type="MARKET" if signal_data.get("urgency") == "IMMEDIATE" else "LIMIT",
                stop_loss=current_price * (1 - signal_data.get("stop_loss_pct", 5) / 100),
                take_profit=[float(data.get("price_target", current_price * 1.05))],
                position_size_pct=float(signal_data.get("max_position_pct", 5)),
                reasoning=data.get("reasoning", ""),
                raw_response=response
            )

        except Exception as e:
            return TradingSignal(
                symbol=symbol,
                direction=SignalDirection.FLAT,
                strength=SignalStrength.WEAK,
                confidence=0,
                entry_price=None,
                entry_type="MARKET",
                stop_loss=current_price * 0.95,
                take_profit=[],
                position_size_pct=0,
                reasoning=f"Parse error: {e}",
                raw_response=response
            )

    def _parse_mtf_signal_response(
        self,
        symbol: str,
        response: str,
        current_price: float
    ) -> TradingSignal:
        """Parse multi-timeframe signal response."""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            data = json.loads(response[start_idx:end_idx])

            signal_data = data.get("signal", {})
            alignment = data.get("timeframe_alignment", "MIXED")

            # Adjust confidence based on alignment
            base_confidence = float(signal_data.get("confidence", 50))
            if alignment == "ALIGNED":
                confidence = base_confidence
            elif alignment == "MIXED":
                confidence = base_confidence * 0.7
            else:  # CONFLICTING
                confidence = base_confidence * 0.4

            return TradingSignal(
                symbol=symbol,
                direction=SignalDirection(signal_data.get("direction", "FLAT")),
                strength=self._calculate_strength(confidence),
                confidence=confidence,
                entry_price=signal_data.get("entry"),
                entry_type="LIMIT" if signal_data.get("entry") else "MARKET",
                stop_loss=float(signal_data.get("stop_loss", current_price * 0.95)),
                take_profit=[float(signal_data.get("take_profit", current_price * 1.05))],
                position_size_pct=5 * (confidence / 100),
                reasoning=data.get("reasoning", ""),
                raw_response=response
            )

        except Exception as e:
            return TradingSignal(
                symbol=symbol,
                direction=SignalDirection.FLAT,
                strength=SignalStrength.WEAK,
                confidence=0,
                entry_price=None,
                entry_type="MARKET",
                stop_loss=current_price * 0.95,
                take_profit=[],
                position_size_pct=0,
                reasoning=f"Parse error: {e}",
                raw_response=response
            )

    def _calculate_strength(self, confidence: float) -> SignalStrength:
        """Calculate signal strength from confidence."""
        if confidence >= 75:
            return SignalStrength.STRONG
        elif confidence >= 50:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK


# Example usage
async def main():
    """Demo of signal generation."""
    from .llm_client import MockLLMClient

    mock_responses = {
        "btc": json.dumps({
            "direction": "LONG",
            "strength": "MODERATE",
            "confidence": 72,
            "entry": {"type": "LIMIT", "price": 45000},
            "stop_loss": 43500,
            "take_profit": [47000, 49000],
            "position_size_pct": 5,
            "reasoning": "Bullish trend with price above all SMAs. RSI neutral with room to run."
        })
    }

    client = MockLLMClient(responses=mock_responses)
    generator = PromptBasedSignalGenerator(client)

    market_data = {
        "price": {"current": 45250, "change_24h": 2.3, "change_7d": -1.5, "change_30d": 5.0},
        "technicals": {
            "rsi": 58.5, "macd": 120.5, "macd_hist": 45.2,
            "sma_20": 44800, "sma_50": 43500, "sma_200": 41000,
            "bb_lower": 43200, "bb_upper": 47000, "atr": 1200
        },
        "volume": {"current": 25000000000, "average": 20000000000},
        "sentiment": "BULLISH",
        "regime": "RISK_ON_TRENDING"
    }

    signal = await generator.generate_signal("BTCUSDT", market_data)

    print("Trading Signal Generation Demo")
    print("=" * 50)
    print(f"Symbol: {signal.symbol}")
    print(f"Direction: {signal.direction.value} ({signal.strength.value})")
    print(f"Confidence: {signal.confidence}%")
    print(f"Entry: ${signal.entry_price:,.2f}" if signal.entry_price else "Market order")
    print(f"Stop Loss: ${signal.stop_loss:,.2f}")
    print(f"Take Profit: {[f'${tp:,.2f}' for tp in signal.take_profit]}")
    print(f"Position Size: {signal.position_size_pct}%")
    print(f"Reasoning: {signal.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
