"""
Multi-Agent Trading System - Agent Definitions

This module provides the core agent classes for building multi-agent LLM trading systems.
Each agent specializes in a specific type of analysis (technical, fundamental, sentiment, etc.)
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

    @property
    def value_numeric(self) -> float:
        """Convert signal to numeric value for aggregation."""
        mapping = {
            Signal.STRONG_BUY: 1.0,
            Signal.BUY: 0.5,
            Signal.NEUTRAL: 0.0,
            Signal.SELL: -0.5,
            Signal.STRONG_SELL: -1.0,
        }
        return mapping[self]


@dataclass
class Analysis:
    """Container for agent analysis results."""
    agent_name: str
    agent_type: str
    symbol: str
    signal: Signal
    confidence: float  # 0.0 to 1.0
    reasoning: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "symbol": self.symbol,
            "signal": self.signal.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseAgent(ABC):
    """
    Base class for all trading agents.

    Each agent processes market data and produces an analysis with:
    - A trading signal (BUY/SELL/NEUTRAL)
    - A confidence score (0.0 to 1.0)
    - Reasoning explaining the decision
    """

    def __init__(self, name: str, llm_client: Optional[Any] = None):
        """
        Initialize agent.

        Args:
            name: Unique agent name
            llm_client: Optional LLM client for natural language reasoning
        """
        self.name = name
        self.llm_client = llm_client
        self._history: List[Analysis] = []

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the type of this agent."""
        pass

    @abstractmethod
    def analyze(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Analysis:
        """
        Perform analysis on the given data.

        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame with optional indicators
            context: Additional context (news, sentiment, etc.)

        Returns:
            Analysis result
        """
        pass

    def _create_analysis(
        self,
        symbol: str,
        signal: Signal,
        confidence: float,
        reasoning: str,
        metrics: Optional[Dict] = None
    ) -> Analysis:
        """Helper to create Analysis object."""
        analysis = Analysis(
            agent_name=self.name,
            agent_type=self.agent_type,
            symbol=symbol,
            signal=signal,
            confidence=min(max(confidence, 0.0), 1.0),
            reasoning=reasoning,
            metrics=metrics or {}
        )
        self._history.append(analysis)
        return analysis

    def get_history(self, limit: Optional[int] = None) -> List[Analysis]:
        """Get analysis history."""
        if limit:
            return self._history[-limit:]
        return self._history.copy()


class TechnicalAgent(BaseAgent):
    """
    Technical Analysis Agent.

    Analyzes price charts, indicators, and patterns to generate trading signals.
    """

    @property
    def agent_type(self) -> str:
        return "technical"

    def analyze(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Analysis:
        """
        Analyze technical indicators.

        Looks at: RSI, MACD, Moving Averages, Bollinger Bands, Volume
        """
        if len(data) < 50:
            return self._create_analysis(
                symbol, Signal.NEUTRAL, 0.3,
                "Insufficient data for technical analysis",
                {"error": "Need at least 50 data points"}
            )

        # Calculate indicators if not present
        close = data["close"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Moving Averages
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        current_price = close.iloc[-1]

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        current_macd = macd.iloc[-1]
        current_macd_signal = macd_signal.iloc[-1]
        macd_histogram = current_macd - current_macd_signal

        # Volume analysis
        volume = data["volume"]
        volume_sma = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1.0

        # Scoring system
        score = 0
        reasons = []

        # RSI analysis
        if current_rsi < 30:
            score += 2
            reasons.append(f"RSI oversold ({current_rsi:.1f})")
        elif current_rsi < 40:
            score += 1
            reasons.append(f"RSI approaching oversold ({current_rsi:.1f})")
        elif current_rsi > 70:
            score -= 2
            reasons.append(f"RSI overbought ({current_rsi:.1f})")
        elif current_rsi > 60:
            score -= 1
            reasons.append(f"RSI approaching overbought ({current_rsi:.1f})")

        # Moving average analysis
        if current_price > sma_20 > sma_50:
            score += 2
            reasons.append("Price above rising MAs (bullish trend)")
        elif current_price > sma_20:
            score += 1
            reasons.append("Price above SMA20")
        elif current_price < sma_20 < sma_50:
            score -= 2
            reasons.append("Price below falling MAs (bearish trend)")
        elif current_price < sma_20:
            score -= 1
            reasons.append("Price below SMA20")

        # MACD analysis
        if macd_histogram > 0 and current_macd > 0:
            score += 1
            reasons.append("MACD bullish with positive histogram")
        elif macd_histogram < 0 and current_macd < 0:
            score -= 1
            reasons.append("MACD bearish with negative histogram")

        # Volume confirmation
        if volume_ratio > 1.5:
            reasons.append(f"High volume ({volume_ratio:.1f}x average)")
        elif volume_ratio < 0.5:
            reasons.append(f"Low volume ({volume_ratio:.1f}x average)")

        # Determine signal
        if score >= 3:
            signal = Signal.STRONG_BUY
        elif score >= 1:
            signal = Signal.BUY
        elif score <= -3:
            signal = Signal.STRONG_SELL
        elif score <= -1:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        confidence = min(abs(score) / 5 + 0.4, 0.95)

        metrics = {
            "rsi": round(current_rsi, 2),
            "price_vs_sma20": round((current_price / sma_20 - 1) * 100, 2),
            "price_vs_sma50": round((current_price / sma_50 - 1) * 100, 2),
            "macd": round(current_macd, 4),
            "macd_signal": round(current_macd_signal, 4),
            "volume_ratio": round(volume_ratio, 2),
            "score": score,
        }

        return self._create_analysis(
            symbol, signal, confidence,
            "; ".join(reasons) if reasons else "Mixed technical signals",
            metrics
        )


class FundamentalsAgent(BaseAgent):
    """
    Fundamental Analysis Agent.

    Analyzes company financials, valuation metrics, and business fundamentals.
    For crypto, analyzes on-chain metrics and tokenomics.
    """

    @property
    def agent_type(self) -> str:
        return "fundamental"

    def analyze(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Analysis:
        """
        Analyze fundamental data.

        For stocks: P/E ratio, revenue growth, margins
        For crypto: TVL, active addresses, developer activity
        """
        context = context or {}
        fundamentals = context.get("fundamentals", {})

        if not fundamentals:
            # Simulate fundamental analysis based on price trends
            close = data["close"]
            returns_30d = (close.iloc[-1] / close.iloc[-30] - 1) if len(close) >= 30 else 0
            volatility = close.pct_change().std() * np.sqrt(252)

            # Basic valuation proxy
            if returns_30d > 0.15:
                signal = Signal.BUY
                reasoning = f"Strong momentum ({returns_30d:.1%} over 30 days) suggests fundamental strength"
            elif returns_30d < -0.15:
                signal = Signal.SELL
                reasoning = f"Weak momentum ({returns_30d:.1%} over 30 days) may indicate fundamental concerns"
            else:
                signal = Signal.NEUTRAL
                reasoning = "Waiting for clearer fundamental signals"

            return self._create_analysis(
                symbol, signal, 0.5, reasoning,
                {"returns_30d": round(returns_30d, 4), "volatility": round(volatility, 4)}
            )

        # Use provided fundamental data
        pe_ratio = fundamentals.get("pe_ratio")
        revenue_growth = fundamentals.get("revenue_growth")
        profit_margin = fundamentals.get("profit_margin")
        debt_to_equity = fundamentals.get("debt_to_equity")

        score = 0
        reasons = []

        if pe_ratio is not None:
            if pe_ratio < 15:
                score += 2
                reasons.append(f"Attractively valued (P/E: {pe_ratio:.1f})")
            elif pe_ratio < 25:
                score += 1
                reasons.append(f"Fairly valued (P/E: {pe_ratio:.1f})")
            elif pe_ratio > 40:
                score -= 1
                reasons.append(f"Expensive valuation (P/E: {pe_ratio:.1f})")

        if revenue_growth is not None:
            if revenue_growth > 0.20:
                score += 2
                reasons.append(f"Strong revenue growth ({revenue_growth:.0%})")
            elif revenue_growth > 0.10:
                score += 1
                reasons.append(f"Solid revenue growth ({revenue_growth:.0%})")
            elif revenue_growth < 0:
                score -= 1
                reasons.append(f"Declining revenue ({revenue_growth:.0%})")

        if profit_margin is not None:
            if profit_margin > 0.20:
                score += 1
                reasons.append(f"Excellent margins ({profit_margin:.0%})")
            elif profit_margin < 0:
                score -= 1
                reasons.append(f"Unprofitable ({profit_margin:.0%} margin)")

        # Determine signal
        if score >= 3:
            signal = Signal.STRONG_BUY
        elif score >= 1:
            signal = Signal.BUY
        elif score <= -2:
            signal = Signal.STRONG_SELL
        elif score <= -1:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        confidence = min(abs(score) / 4 + 0.5, 0.9)

        return self._create_analysis(
            symbol, signal, confidence,
            "; ".join(reasons) if reasons else "Insufficient fundamental data",
            {"score": score, **fundamentals}
        )


class SentimentAgent(BaseAgent):
    """
    Sentiment Analysis Agent.

    Analyzes market sentiment from social media, forums, and other sources.
    """

    @property
    def agent_type(self) -> str:
        return "sentiment"

    def analyze(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Analysis:
        """
        Analyze market sentiment.

        Uses: Social media mentions, Fear & Greed index, put/call ratios
        """
        context = context or {}
        sentiment_data = context.get("sentiment", {})

        if not sentiment_data:
            # Simulate sentiment based on recent price action
            close = data["close"]
            returns_5d = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
            volatility = close.pct_change().tail(20).std()

            # Price momentum affects sentiment
            if returns_5d > 0.05:
                sentiment_score = 0.7 + min(returns_5d, 0.1)
                signal = Signal.BUY
                reasoning = "Bullish sentiment implied by strong recent performance"
            elif returns_5d < -0.05:
                sentiment_score = 0.3 - min(abs(returns_5d), 0.1)
                signal = Signal.SELL
                reasoning = "Bearish sentiment implied by weak recent performance"
            else:
                sentiment_score = 0.5
                signal = Signal.NEUTRAL
                reasoning = "Neutral sentiment, market in wait-and-see mode"

            return self._create_analysis(
                symbol, signal, 0.6, reasoning,
                {"implied_sentiment": round(sentiment_score, 2)}
            )

        # Use provided sentiment data
        social_sentiment = sentiment_data.get("social_score", 0.5)  # 0-1 scale
        fear_greed = sentiment_data.get("fear_greed", 50)  # 0-100 scale
        mentions_change = sentiment_data.get("mentions_change", 0)  # % change

        reasons = []

        # Social sentiment analysis
        if social_sentiment > 0.7:
            reasons.append(f"Very bullish social sentiment ({social_sentiment:.0%})")
        elif social_sentiment > 0.55:
            reasons.append(f"Positive social sentiment ({social_sentiment:.0%})")
        elif social_sentiment < 0.3:
            reasons.append(f"Very bearish social sentiment ({social_sentiment:.0%})")
        elif social_sentiment < 0.45:
            reasons.append(f"Negative social sentiment ({social_sentiment:.0%})")

        # Fear & Greed analysis
        if fear_greed > 75:
            reasons.append(f"Extreme greed ({fear_greed}) - potential reversal risk")
        elif fear_greed > 55:
            reasons.append(f"Greed zone ({fear_greed})")
        elif fear_greed < 25:
            reasons.append(f"Extreme fear ({fear_greed}) - potential buying opportunity")
        elif fear_greed < 45:
            reasons.append(f"Fear zone ({fear_greed})")

        # Determine signal
        combined_score = (social_sentiment * 100 + fear_greed) / 2

        if combined_score > 70:
            signal = Signal.BUY
        elif combined_score > 55:
            signal = Signal.BUY if social_sentiment > 0.6 else Signal.NEUTRAL
        elif combined_score < 30:
            signal = Signal.SELL
        elif combined_score < 45:
            signal = Signal.SELL if social_sentiment < 0.4 else Signal.NEUTRAL
        else:
            signal = Signal.NEUTRAL

        # High fear can be contrarian buy signal
        if fear_greed < 20:
            signal = Signal.BUY
            reasons.append("Contrarian buy: extreme fear often precedes rebounds")

        confidence = 0.5 + abs(combined_score - 50) / 100

        return self._create_analysis(
            symbol, signal, confidence,
            "; ".join(reasons) if reasons else "Mixed sentiment signals",
            {"social_sentiment": social_sentiment, "fear_greed": fear_greed}
        )


class NewsAgent(BaseAgent):
    """
    News Analysis Agent.

    Analyzes news headlines and articles for trading signals.
    """

    @property
    def agent_type(self) -> str:
        return "news"

    def analyze(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Analysis:
        """
        Analyze recent news.

        Evaluates: Headlines sentiment, news volume, event impact
        """
        context = context or {}
        news_data = context.get("news", [])

        if not news_data:
            return self._create_analysis(
                symbol, Signal.NEUTRAL, 0.3,
                "No news data available for analysis",
                {"news_count": 0}
            )

        # Analyze news headlines
        positive_keywords = ["upgrade", "beat", "growth", "profit", "surge", "rally", "bullish", "positive"]
        negative_keywords = ["downgrade", "miss", "decline", "loss", "crash", "bearish", "negative", "warning"]

        positive_count = 0
        negative_count = 0
        headlines = []

        for news_item in news_data:
            headline = news_item.get("headline", "").lower()
            headlines.append(news_item.get("headline", ""))

            if any(word in headline for word in positive_keywords):
                positive_count += 1
            if any(word in headline for word in negative_keywords):
                negative_count += 1

        total_news = len(news_data)
        sentiment_ratio = (positive_count - negative_count) / max(total_news, 1)

        reasons = []
        reasons.append(f"Analyzed {total_news} recent news items")

        if positive_count > negative_count:
            reasons.append(f"More positive ({positive_count}) than negative ({negative_count}) headlines")
        elif negative_count > positive_count:
            reasons.append(f"More negative ({negative_count}) than positive ({positive_count}) headlines")

        # Determine signal
        if sentiment_ratio > 0.3:
            signal = Signal.STRONG_BUY
        elif sentiment_ratio > 0.1:
            signal = Signal.BUY
        elif sentiment_ratio < -0.3:
            signal = Signal.STRONG_SELL
        elif sentiment_ratio < -0.1:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        confidence = min(abs(sentiment_ratio) + 0.4, 0.85)

        return self._create_analysis(
            symbol, signal, confidence,
            "; ".join(reasons),
            {
                "news_count": total_news,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "sentiment_ratio": round(sentiment_ratio, 2),
                "recent_headlines": headlines[:5]
            }
        )


class BullAgent(BaseAgent):
    """
    Bull Researcher Agent.

    Always looks for reasons to be optimistic and buy.
    Provides counterbalance to bear agent in debates.
    """

    @property
    def agent_type(self) -> str:
        return "bull_researcher"

    def analyze(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Analysis:
        """
        Find bullish arguments.

        Focuses on: Growth potential, positive catalysts, technical breakouts
        """
        close = data["close"]
        volume = data["volume"]

        reasons = []
        confidence_boost = 0

        # Look for bullish signals
        # 1. Recent performance
        returns_30d = (close.iloc[-1] / close.iloc[-30] - 1) if len(close) >= 30 else 0
        returns_7d = (close.iloc[-1] / close.iloc[-7] - 1) if len(close) >= 7 else 0

        if returns_30d > 0:
            reasons.append(f"Price up {returns_30d:.1%} over 30 days - positive momentum")
            confidence_boost += 0.1
        else:
            reasons.append(f"Down {returns_30d:.1%} over 30 days - potential mean reversion opportunity")

        # 2. Volume analysis
        recent_volume = volume.tail(5).mean()
        avg_volume = volume.tail(30).mean()
        if recent_volume > avg_volume * 1.2:
            reasons.append("Increasing volume suggests growing interest")
            confidence_boost += 0.05

        # 3. Support levels
        low_52w = close.tail(252).min() if len(close) >= 252 else close.min()
        if close.iloc[-1] > low_52w * 1.1:
            reasons.append("Trading well above 52-week lows - strong support")
            confidence_boost += 0.05

        # 4. Always find something positive
        if returns_7d < 0:
            reasons.append("Recent pullback provides attractive entry point")
        else:
            reasons.append("Short-term momentum is positive")

        reasons.append("Long-term fundamentals remain intact")

        # Bull is always at least slightly bullish
        signal = Signal.BUY if confidence_boost < 0.1 else Signal.STRONG_BUY
        confidence = min(0.7 + confidence_boost, 0.95)

        return self._create_analysis(
            symbol, signal, confidence,
            " | ".join(reasons),
            {"returns_30d": round(returns_30d, 4), "returns_7d": round(returns_7d, 4)}
        )


class BearAgent(BaseAgent):
    """
    Bear Researcher Agent.

    Always looks for reasons to be cautious and sell.
    Provides counterbalance to bull agent in debates.
    """

    @property
    def agent_type(self) -> str:
        return "bear_researcher"

    def analyze(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Analysis:
        """
        Find bearish arguments.

        Focuses on: Risk factors, overvaluation, technical resistance
        """
        close = data["close"]
        volume = data["volume"]

        reasons = []
        confidence_boost = 0

        # Look for bearish signals
        # 1. Valuation concerns
        returns_30d = (close.iloc[-1] / close.iloc[-30] - 1) if len(close) >= 30 else 0

        if returns_30d > 0.2:
            reasons.append(f"Up {returns_30d:.1%} in 30 days - potentially overextended")
            confidence_boost += 0.1
        else:
            reasons.append("Recent weakness may continue")

        # 2. Volatility risk
        volatility = close.pct_change().tail(30).std() * np.sqrt(252)
        if volatility > 0.3:
            reasons.append(f"High volatility ({volatility:.0%} annualized) indicates risk")
            confidence_boost += 0.05

        # 3. Resistance levels
        high_52w = close.tail(252).max() if len(close) >= 252 else close.max()
        if close.iloc[-1] > high_52w * 0.95:
            reasons.append("Near 52-week highs - resistance expected")
            confidence_boost += 0.05

        # 4. Volume analysis
        recent_volume = volume.tail(5).mean()
        avg_volume = volume.tail(30).mean()
        if recent_volume < avg_volume * 0.8:
            reasons.append("Declining volume suggests waning interest")
            confidence_boost += 0.05

        # 5. Always find something negative
        reasons.append("Macro environment remains uncertain")
        reasons.append("Risk-reward may not favor entry at current levels")

        # Bear is always at least slightly bearish
        signal = Signal.SELL if confidence_boost < 0.1 else Signal.STRONG_SELL
        confidence = min(0.7 + confidence_boost, 0.95)

        return self._create_analysis(
            symbol, signal, confidence,
            " | ".join(reasons),
            {"volatility": round(volatility, 4), "pct_from_high": round(close.iloc[-1] / high_52w - 1, 4)}
        )


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent.

    Evaluates portfolio risk and sets position size limits.
    Acts as a gatekeeper before trade execution.
    """

    def __init__(
        self,
        name: str,
        max_position_pct: float = 0.05,
        max_drawdown: float = 0.15,
        llm_client: Optional[Any] = None
    ):
        super().__init__(name, llm_client)
        self.max_position_pct = max_position_pct
        self.max_drawdown = max_drawdown

    @property
    def agent_type(self) -> str:
        return "risk_manager"

    def analyze(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Analysis:
        """
        Assess risk and recommend position sizing.

        Evaluates: Volatility, drawdown potential, portfolio concentration
        """
        context = context or {}
        portfolio = context.get("portfolio", {})

        close = data["close"]

        # Calculate risk metrics
        returns = close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)

        # Value at Risk (95%)
        var_95 = returns.quantile(0.05)

        # Maximum drawdown in recent period
        rolling_max = close.rolling(window=len(close), min_periods=1).max()
        drawdown = (close / rolling_max - 1)
        max_drawdown = drawdown.min()

        # Current drawdown
        current_drawdown = close.iloc[-1] / close.max() - 1

        reasons = []
        risk_score = 0  # Higher = more risky

        # Volatility assessment
        if volatility > 0.5:
            reasons.append(f"Very high volatility ({volatility:.0%}) - reduce position size")
            risk_score += 2
        elif volatility > 0.3:
            reasons.append(f"High volatility ({volatility:.0%}) - caution advised")
            risk_score += 1
        else:
            reasons.append(f"Acceptable volatility ({volatility:.0%})")

        # Drawdown assessment
        if current_drawdown < -0.2:
            reasons.append(f"Asset in significant drawdown ({current_drawdown:.0%})")
            risk_score += 1
        elif max_drawdown < -0.3:
            reasons.append(f"History of large drawdowns ({max_drawdown:.0%})")
            risk_score += 1

        # Position size recommendation
        if risk_score >= 3:
            recommended_size = self.max_position_pct * 0.25
            signal = Signal.STRONG_SELL  # Don't trade
        elif risk_score >= 2:
            recommended_size = self.max_position_pct * 0.5
            signal = Signal.SELL  # Reduce
        elif risk_score >= 1:
            recommended_size = self.max_position_pct * 0.75
            signal = Signal.NEUTRAL  # Caution
        else:
            recommended_size = self.max_position_pct
            signal = Signal.BUY  # Proceed

        reasons.append(f"Recommended position size: {recommended_size:.1%} of portfolio")

        # Stop loss recommendation
        atr = (data["high"] - data["low"]).rolling(14).mean().iloc[-1]
        stop_loss_distance = atr * 2
        stop_loss_price = close.iloc[-1] - stop_loss_distance
        reasons.append(f"Suggested stop-loss: ${stop_loss_price:.2f} ({stop_loss_distance / close.iloc[-1]:.1%} below)")

        confidence = 0.85  # Risk assessment is usually confident

        return self._create_analysis(
            symbol, signal, confidence,
            "; ".join(reasons),
            {
                "volatility": round(volatility, 4),
                "var_95": round(var_95, 4),
                "max_drawdown": round(max_drawdown, 4),
                "current_drawdown": round(current_drawdown, 4),
                "risk_score": risk_score,
                "recommended_position_pct": round(recommended_size, 4),
                "stop_loss_price": round(stop_loss_price, 2)
            }
        )


class TraderAgent(BaseAgent):
    """
    Trader Agent.

    Aggregates inputs from all other agents and makes final trading decisions.
    """

    def __init__(
        self,
        name: str,
        llm_client: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(name, llm_client)
        self.weights = weights or {
            "technical": 0.25,
            "fundamental": 0.20,
            "sentiment": 0.15,
            "news": 0.15,
            "bull_researcher": 0.10,
            "bear_researcher": 0.10,
            "risk_manager": 0.05,
        }

    @property
    def agent_type(self) -> str:
        return "trader"

    def analyze(self, symbol: str, data: pd.DataFrame, context: Optional[Dict] = None) -> Analysis:
        """
        Aggregate all agent analyses and make trading decision.
        """
        context = context or {}
        analyses = context.get("analyses", [])

        if not analyses:
            return self._create_analysis(
                symbol, Signal.NEUTRAL, 0.3,
                "No agent analyses available for aggregation",
                {}
            )

        # Aggregate signals
        weighted_score = 0
        total_weight = 0
        agent_contributions = {}

        for analysis in analyses:
            agent_type = analysis.agent_type
            weight = self.weights.get(agent_type, 0.1)

            signal_value = analysis.signal.value_numeric
            contribution = signal_value * weight * analysis.confidence

            weighted_score += contribution
            total_weight += weight

            agent_contributions[analysis.agent_name] = {
                "signal": analysis.signal.value,
                "confidence": analysis.confidence,
                "contribution": round(contribution, 3)
            }

        # Normalize score to -1 to 1 range
        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0

        # Determine final signal
        if normalized_score > 0.5:
            signal = Signal.STRONG_BUY
        elif normalized_score > 0.2:
            signal = Signal.BUY
        elif normalized_score < -0.5:
            signal = Signal.STRONG_SELL
        elif normalized_score < -0.2:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        # Build reasoning
        bullish_agents = [a.agent_name for a in analyses if a.signal in [Signal.BUY, Signal.STRONG_BUY]]
        bearish_agents = [a.agent_name for a in analyses if a.signal in [Signal.SELL, Signal.STRONG_SELL]]

        reasons = []
        if bullish_agents:
            reasons.append(f"Bullish signals from: {', '.join(bullish_agents)}")
        if bearish_agents:
            reasons.append(f"Bearish signals from: {', '.join(bearish_agents)}")

        reasons.append(f"Weighted score: {normalized_score:.2f}")

        confidence = 0.5 + abs(normalized_score) * 0.4

        return self._create_analysis(
            symbol, signal, confidence,
            "; ".join(reasons),
            {
                "weighted_score": round(normalized_score, 3),
                "agent_contributions": agent_contributions,
                "bullish_count": len(bullish_agents),
                "bearish_count": len(bearish_agents)
            }
        )


if __name__ == "__main__":
    print("Agent Demo\n" + "=" * 50)

    # Create mock data
    import pandas as pd
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")
    close = 100 * (1 + np.random.randn(252) * 0.02).cumprod()

    data = pd.DataFrame({
        "open": close * (1 + np.random.randn(252) * 0.005),
        "high": close * (1 + abs(np.random.randn(252) * 0.01)),
        "low": close * (1 - abs(np.random.randn(252) * 0.01)),
        "close": close,
        "volume": np.random.randint(1e6, 1e8, 252)
    }, index=dates)

    # Create and test agents
    agents = [
        TechnicalAgent("Tech-1"),
        FundamentalsAgent("Fundamentals-1"),
        SentimentAgent("Sentiment-1"),
        NewsAgent("News-1"),
        BullAgent("Bull-1"),
        BearAgent("Bear-1"),
        RiskManagerAgent("Risk-1", max_position_pct=0.05),
    ]

    analyses = []
    for agent in agents:
        analysis = agent.analyze("DEMO", data)
        analyses.append(analysis)
        print(f"\n{agent.name} ({agent.agent_type}):")
        print(f"  Signal: {analysis.signal.value}")
        print(f"  Confidence: {analysis.confidence:.0%}")
        print(f"  Reasoning: {analysis.reasoning[:100]}...")

    # Test trader aggregation
    trader = TraderAgent("Trader-1")
    final = trader.analyze("DEMO", data, {"analyses": analyses})

    print(f"\n{'='*50}")
    print(f"FINAL DECISION ({trader.name}):")
    print(f"  Signal: {final.signal.value}")
    print(f"  Confidence: {final.confidence:.0%}")
    print(f"  Reasoning: {final.reasoning}")
