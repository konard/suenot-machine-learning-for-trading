"""
Multi-Step Signal Generator

This module provides multi-step signal generation with explicit
reasoning chains, making trading decisions transparent and auditable.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Signal(Enum):
    """Trading signal enum with numeric values."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class CoTSignal:
    """Trading signal with chain-of-thought reasoning."""
    symbol: str
    signal: Signal
    confidence: float
    reasoning_chain: List[str]
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'signal': self.signal.name,
            'signal_value': self.signal.value,
            'confidence': self.confidence,
            'reasoning_chain': self.reasoning_chain,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size_pct': self.position_size_pct,
            'timestamp': self.timestamp.isoformat()
        }


class MultiStepSignalGenerator:
    """
    Generate trading signals using multi-step reasoning.

    Each signal is produced through a chain of analytical steps,
    providing full transparency into the decision process.

    Attributes:
        risk_tolerance: Maximum risk per trade as fraction
        min_confidence: Minimum confidence required for signals

    Example:
        >>> generator = MultiStepSignalGenerator()
        >>> price_data = {'open': 100, 'high': 105, 'low': 98, 'close': 103}
        >>> indicators = {'rsi': 45, 'macd': 0.5, 'macd_signal': 0.3}
        >>> signal = generator.generate_signal("AAPL", price_data, indicators)
        >>> print(signal.signal, signal.confidence)
    """

    def __init__(
        self,
        risk_tolerance: float = 0.02,
        min_confidence: float = 0.6
    ):
        """
        Initialize signal generator.

        Args:
            risk_tolerance: Risk per trade (default 2%)
            min_confidence: Minimum confidence for signals
        """
        self.risk_tolerance = risk_tolerance
        self.min_confidence = min_confidence
        self.reasoning_chain: List[str] = []

    def _add_reasoning(self, step: str):
        """Add a step to the reasoning chain."""
        self.reasoning_chain.append(step)

    def generate_signal(
        self,
        symbol: str,
        price_data: Dict,
        indicators: Dict,
        fundamentals: Optional[Dict] = None,
        sentiment: Optional[Dict] = None
    ) -> CoTSignal:
        """
        Generate a trading signal with full reasoning chain.

        Args:
            symbol: Trading symbol
            price_data: Dict with open, high, low, close, volume
            indicators: Dict with technical indicators
            fundamentals: Optional fundamental data
            sentiment: Optional sentiment scores

        Returns:
            CoTSignal with reasoning chain
        """
        self.reasoning_chain = []

        # Step 1: Trend Analysis
        trend_score, trend_reasoning = self._analyze_trend(price_data, indicators)
        self._add_reasoning(f"STEP 1 - TREND: {trend_reasoning}")

        # Step 2: Momentum Analysis
        momentum_score, momentum_reasoning = self._analyze_momentum(indicators)
        self._add_reasoning(f"STEP 2 - MOMENTUM: {momentum_reasoning}")

        # Step 3: Volume Analysis
        volume_score, volume_reasoning = self._analyze_volume(price_data)
        self._add_reasoning(f"STEP 3 - VOLUME: {volume_reasoning}")

        # Step 4: Sentiment Analysis
        if sentiment:
            sentiment_score, sentiment_reasoning = self._analyze_sentiment(sentiment)
            self._add_reasoning(f"STEP 4 - SENTIMENT: {sentiment_reasoning}")
        else:
            sentiment_score = 0
            self._add_reasoning("STEP 4 - SENTIMENT: No sentiment data available, neutral assumption.")

        # Step 5: Risk/Reward Calculation
        rr_score, rr_reasoning, levels = self._calculate_risk_reward(
            price_data, indicators
        )
        self._add_reasoning(f"STEP 5 - RISK/REWARD: {rr_reasoning}")

        # Step 6: Aggregate and Decide
        final_signal, confidence, decision_reasoning = self._aggregate_signals(
            trend_score, momentum_score, volume_score, sentiment_score, rr_score
        )
        self._add_reasoning(f"STEP 6 - DECISION: {decision_reasoning}")

        # Calculate position size
        position_size = self._calculate_position_size(
            confidence,
            levels.get('entry', price_data.get('close', 100)),
            levels.get('stop_loss', price_data.get('close', 100) * 0.95)
        )

        return CoTSignal(
            symbol=symbol,
            signal=final_signal,
            confidence=confidence,
            reasoning_chain=self.reasoning_chain.copy(),
            entry_price=levels.get('entry'),
            stop_loss=levels.get('stop_loss'),
            take_profit=levels.get('take_profit'),
            position_size_pct=position_size
        )

    def _analyze_trend(
        self,
        price_data: Dict,
        indicators: Dict
    ) -> Tuple[float, str]:
        """
        Analyze price trend.

        Args:
            price_data: OHLCV data
            indicators: Technical indicators

        Returns:
            Tuple of (score from -1 to 1, reasoning string)
        """
        close = price_data.get('close', 100)
        sma_20 = indicators.get('sma_20', close)
        sma_50 = indicators.get('sma_50', close)
        sma_200 = indicators.get('sma_200', close)

        score = 0.0
        reasons = []

        # Price vs SMAs
        if close > sma_20:
            score += 0.2
            reasons.append("Price above SMA(20)")
        else:
            score -= 0.2
            reasons.append("Price below SMA(20)")

        if close > sma_50:
            score += 0.3
            reasons.append("Price above SMA(50)")
        else:
            score -= 0.3
            reasons.append("Price below SMA(50)")

        if close > sma_200:
            score += 0.3
            reasons.append("Price above SMA(200) - bullish long-term")
        else:
            score -= 0.3
            reasons.append("Price below SMA(200) - bearish long-term")

        # SMA alignment
        if sma_20 > sma_50 > sma_200:
            score += 0.2
            reasons.append("SMAs aligned bullishly (20>50>200)")
        elif sma_20 < sma_50 < sma_200:
            score -= 0.2
            reasons.append("SMAs aligned bearishly (20<50<200)")

        score = np.clip(score, -1, 1)
        reasoning = f"Trend score: {score:.2f}. " + "; ".join(reasons)

        return score, reasoning

    def _analyze_momentum(self, indicators: Dict) -> Tuple[float, str]:
        """
        Analyze momentum indicators.

        Args:
            indicators: Technical indicators dict

        Returns:
            Tuple of (score from -1 to 1, reasoning string)
        """
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)

        score = 0.0
        reasons = []

        # RSI analysis
        if rsi < 30:
            score += 0.4
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 45:
            score += 0.2
            reasons.append(f"RSI approaching oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 0.4
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 55:
            score -= 0.2
            reasons.append(f"RSI approaching overbought ({rsi:.1f})")
        else:
            reasons.append(f"RSI neutral ({rsi:.1f})")

        # MACD analysis
        if macd > macd_signal:
            score += 0.3
            reasons.append("MACD above signal (bullish)")
        else:
            score -= 0.3
            reasons.append("MACD below signal (bearish)")

        if macd > 0:
            score += 0.15
            reasons.append("MACD positive")
        else:
            score -= 0.15
            reasons.append("MACD negative")

        score = np.clip(score, -1, 1)
        reasoning = f"Momentum score: {score:.2f}. " + "; ".join(reasons)

        return score, reasoning

    def _analyze_volume(self, price_data: Dict) -> Tuple[float, str]:
        """
        Analyze volume patterns.

        Args:
            price_data: OHLCV data

        Returns:
            Tuple of (score from -1 to 1, reasoning string)
        """
        volume = price_data.get('volume', 0)
        avg_volume = price_data.get('avg_volume', volume if volume > 0 else 1)
        close = price_data.get('close', 100)
        prev_close = price_data.get('prev_close', close)

        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        price_change = (close - prev_close) / prev_close if prev_close > 0 else 0

        score = 0.0
        reasons = []

        # Volume relative to average
        if volume_ratio > 1.5:
            if price_change > 0:
                score += 0.4
                reasons.append(f"High volume ({volume_ratio:.1f}x avg) on up day - strong buying")
            else:
                score -= 0.4
                reasons.append(f"High volume ({volume_ratio:.1f}x avg) on down day - strong selling")
        elif volume_ratio > 1.0:
            if price_change > 0:
                score += 0.2
                reasons.append(f"Above-average volume ({volume_ratio:.1f}x) confirms up move")
            else:
                score -= 0.2
                reasons.append(f"Above-average volume ({volume_ratio:.1f}x) confirms down move")
        else:
            reasons.append(f"Below-average volume ({volume_ratio:.1f}x) - weak conviction")

        score = np.clip(score, -1, 1)
        reasoning = f"Volume score: {score:.2f}. " + "; ".join(reasons)

        return score, reasoning

    def _analyze_sentiment(self, sentiment: Dict) -> Tuple[float, str]:
        """
        Analyze sentiment data.

        Args:
            sentiment: Sentiment scores dict

        Returns:
            Tuple of (score from -1 to 1, reasoning string)
        """
        news_sentiment = sentiment.get('news', 0)
        social_sentiment = sentiment.get('social', 0)
        analyst_rating = sentiment.get('analyst', 0)

        # Weight sentiment sources
        score = (
            news_sentiment * 0.4 +
            social_sentiment * 0.3 +
            analyst_rating * 0.3
        )

        reasons = []
        if news_sentiment > 0.3:
            reasons.append("Positive news sentiment")
        elif news_sentiment < -0.3:
            reasons.append("Negative news sentiment")

        if social_sentiment > 0.3:
            reasons.append("Bullish social media")
        elif social_sentiment < -0.3:
            reasons.append("Bearish social media")

        if analyst_rating > 0.3:
            reasons.append("Positive analyst outlook")
        elif analyst_rating < -0.3:
            reasons.append("Negative analyst outlook")

        if not reasons:
            reasons.append("Mixed/neutral sentiment")

        score = np.clip(score, -1, 1)
        reasoning = f"Sentiment score: {score:.2f}. " + "; ".join(reasons)

        return score, reasoning

    def _calculate_risk_reward(
        self,
        price_data: Dict,
        indicators: Dict
    ) -> Tuple[float, str, Dict]:
        """
        Calculate risk/reward and key levels.

        Args:
            price_data: OHLCV data
            indicators: Technical indicators

        Returns:
            Tuple of (score, reasoning, levels dict)
        """
        close = price_data.get('close', 100)
        high = price_data.get('high', close)
        low = price_data.get('low', close)
        atr = indicators.get('atr', close * 0.02)

        # Support and resistance estimation
        support = low - (0.5 * atr)
        resistance = high + (0.5 * atr)

        # Entry, stop, target levels
        entry = close
        stop_loss = support - (0.5 * atr)
        take_profit = resistance + atr

        # Risk/reward ratio
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr_ratio = reward / risk if risk > 0 else 0

        # Score based on R/R ratio
        if rr_ratio >= 3:
            score = 0.8
            assessment = "Excellent"
        elif rr_ratio >= 2:
            score = 0.5
            assessment = "Good"
        elif rr_ratio >= 1.5:
            score = 0.2
            assessment = "Acceptable"
        elif rr_ratio >= 1:
            score = 0
            assessment = "Marginal"
        else:
            score = -0.5
            assessment = "Poor"

        reasoning = (
            f"R/R ratio: {rr_ratio:.2f} ({assessment}). "
            f"Entry: ${entry:.2f}, Stop: ${stop_loss:.2f}, Target: ${take_profit:.2f}. "
            f"Risk: ${risk:.2f}, Reward: ${reward:.2f}"
        )

        levels = {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'support': support,
            'resistance': resistance
        }

        return score, reasoning, levels

    def _aggregate_signals(
        self,
        trend: float,
        momentum: float,
        volume: float,
        sentiment: float,
        risk_reward: float
    ) -> Tuple[Signal, float, str]:
        """
        Aggregate all scores into final signal.

        Args:
            trend: Trend score (-1 to 1)
            momentum: Momentum score (-1 to 1)
            volume: Volume score (-1 to 1)
            sentiment: Sentiment score (-1 to 1)
            risk_reward: Risk/reward score (-1 to 1)

        Returns:
            Tuple of (Signal enum, confidence, reasoning)
        """
        # Weight the factors
        weights = {
            'trend': 0.25,
            'momentum': 0.25,
            'volume': 0.15,
            'sentiment': 0.15,
            'risk_reward': 0.20
        }

        weighted_score = (
            trend * weights['trend'] +
            momentum * weights['momentum'] +
            volume * weights['volume'] +
            sentiment * weights['sentiment'] +
            risk_reward * weights['risk_reward']
        )

        # Calculate confidence based on signal alignment
        scores = [trend, momentum, volume, sentiment, risk_reward]
        score_variance = np.var(scores)
        alignment_factor = 1 - min(score_variance, 0.5)
        confidence = 0.5 + (abs(weighted_score) * 0.4 * alignment_factor)
        confidence = min(confidence, 0.95)

        # Determine signal
        if weighted_score > 0.5:
            signal = Signal.STRONG_BUY
        elif weighted_score > 0.2:
            signal = Signal.BUY
        elif weighted_score < -0.5:
            signal = Signal.STRONG_SELL
        elif weighted_score < -0.2:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        reasoning = (
            f"Weighted score: {weighted_score:.3f}. "
            f"Factors: Trend({trend:.2f}), Momentum({momentum:.2f}), "
            f"Volume({volume:.2f}), Sentiment({sentiment:.2f}), R/R({risk_reward:.2f}). "
            f"Signal: {signal.name} with {confidence:.1%} confidence."
        )

        return signal, confidence, reasoning

    def _calculate_position_size(
        self,
        confidence: float,
        entry: float,
        stop_loss: float
    ) -> float:
        """
        Calculate position size as percentage of portfolio.

        Args:
            confidence: Signal confidence
            entry: Entry price
            stop_loss: Stop loss price

        Returns:
            Position size as fraction of portfolio
        """
        if confidence < self.min_confidence:
            return 0.0

        risk_per_share = abs(entry - stop_loss)
        if risk_per_share == 0:
            return 0.0

        # Scale position by confidence
        confidence_factor = (confidence - self.min_confidence) / (1 - self.min_confidence)
        base_risk = self.risk_tolerance * confidence_factor

        return base_risk


if __name__ == "__main__":
    print("Multi-Step Signal Generation Demo")
    print("=" * 50)

    generator = MultiStepSignalGenerator()

    # Sample market data
    price_data = {
        'open': 42500,
        'high': 43800,
        'low': 42200,
        'close': 43250,
        'prev_close': 42800,
        'volume': 25000000000,
        'avg_volume': 20000000000
    }

    indicators = {
        'rsi': 55,
        'macd': 250,
        'macd_signal': 180,
        'sma_20': 42800,
        'sma_50': 41500,
        'sma_200': 38000,
        'atr': 800
    }

    sentiment = {
        'news': 0.35,
        'social': 0.25,
        'analyst': 0.40
    }

    # Generate signal
    signal = generator.generate_signal(
        "BTCUSDT",
        price_data,
        indicators,
        sentiment=sentiment
    )

    print(f"\nSignal for: {signal.symbol}")
    print(f"Generated at: {signal.timestamp}")
    print("\n" + "=" * 50)
    print("REASONING CHAIN:")
    print("=" * 50)

    for step in signal.reasoning_chain:
        print(f"\n{step}")

    print("\n" + "=" * 50)
    print("FINAL SIGNAL:")
    print("=" * 50)
    print(f"Signal: {signal.signal.name}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"Entry: ${signal.entry_price:,.2f}")
    print(f"Stop Loss: ${signal.stop_loss:,.2f}")
    print(f"Take Profit: ${signal.take_profit:,.2f}")
    print(f"Position Size: {signal.position_size_pct:.2%} of portfolio")
