"""
Trading signal generation based on anomaly detection.

This module converts anomaly detection results into actionable trading signals.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

import numpy as np
import pandas as pd

try:
    from .detector import AnomalyResult, AnomalyType
except ImportError:
    from detector import AnomalyResult, AnomalyType

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    REDUCE_POSITION = "reduce_position"
    INCREASE_POSITION = "increase_position"


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    signal_type: SignalType
    confidence: float  # 0-1
    strength: float  # 0-1, how strong the signal is
    reason: str
    anomaly_result: Optional[AnomalyResult]
    timestamp: Optional[pd.Timestamp] = None
    price: Optional[float] = None
    suggested_position_size: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "reason": self.reason,
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "price": self.price,
            "suggested_position_size": self.suggested_position_size,
            "anomaly": self.anomaly_result.to_dict() if self.anomaly_result else None,
        }


class AnomalySignalGenerator:
    """
    Generate trading signals from anomaly detection results.

    Strategies:
    1. Contrarian: Trade against anomalies (mean reversion)
    2. Momentum: Trade with anomalies (trend following)
    3. Risk-based: Reduce exposure on anomalies
    4. Selective: Only trade specific anomaly types
    """

    def __init__(
        self,
        strategy: str = "contrarian",
        min_anomaly_score: float = 0.6,
        min_confidence: float = 0.5,
        position_sizing: str = "fixed",
        base_position_size: float = 1.0,
    ):
        """
        Initialize signal generator.

        Args:
            strategy: Trading strategy ("contrarian", "momentum", "risk", "selective")
            min_anomaly_score: Minimum anomaly score to generate signal
            min_confidence: Minimum confidence to generate signal
            position_sizing: "fixed", "score_based", or "kelly"
            base_position_size: Base position size (1.0 = 100%)
        """
        self.strategy = strategy
        self.min_anomaly_score = min_anomaly_score
        self.min_confidence = min_confidence
        self.position_sizing = position_sizing
        self.base_position_size = base_position_size

        # Track recent signals for filtering
        self._recent_signals: List[TradingSignal] = []

    def _compute_position_size(
        self,
        anomaly_result: AnomalyResult,
        current_price: Optional[float] = None,
    ) -> float:
        """Compute position size based on anomaly characteristics."""
        if self.position_sizing == "fixed":
            return self.base_position_size

        elif self.position_sizing == "score_based":
            # Higher anomaly score = smaller position (more risk)
            return self.base_position_size * (1 - anomaly_result.score * 0.5)

        elif self.position_sizing == "kelly":
            # Simplified Kelly criterion based on confidence
            win_prob = anomaly_result.confidence
            # Assume 1:1 risk/reward for simplicity
            kelly_fraction = 2 * win_prob - 1
            return self.base_position_size * max(0, kelly_fraction)

        return self.base_position_size

    def generate_contrarian_signal(
        self,
        anomaly_result: AnomalyResult,
        market_data: Optional[pd.Series] = None,
    ) -> TradingSignal:
        """
        Generate contrarian signal (trade against anomaly).

        Logic: If price spiked up anomalously, expect reversion (sell).
               If price crashed anomalously, expect reversion (buy).
        """
        if not anomaly_result.is_anomaly:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=1 - anomaly_result.score,
                strength=0.0,
                reason="No anomaly detected",
                anomaly_result=anomaly_result,
            )

        # Determine direction based on anomaly type and price movement
        signal_type = SignalType.HOLD
        reason = ""

        if market_data is not None and "returns" in market_data.index:
            returns = market_data["returns"]
            if pd.notna(returns):
                if anomaly_result.anomaly_type == AnomalyType.PRICE_SPIKE:
                    if returns > 0:
                        signal_type = SignalType.SELL
                        reason = f"Price spike up {returns*100:.2f}% - expect reversion"
                    else:
                        signal_type = SignalType.BUY
                        reason = f"Price spike down {returns*100:.2f}% - expect reversion"

                elif anomaly_result.anomaly_type == AnomalyType.FLASH_CRASH:
                    signal_type = SignalType.BUY
                    reason = "Flash crash detected - expect reversion"

                elif anomaly_result.anomaly_type == AnomalyType.PUMP_AND_DUMP:
                    signal_type = SignalType.SELL
                    reason = "Pump and dump pattern - expect dump phase"

        position_size = self._compute_position_size(anomaly_result)

        return TradingSignal(
            signal_type=signal_type,
            confidence=anomaly_result.confidence,
            strength=anomaly_result.score,
            reason=reason or f"Contrarian signal on {anomaly_result.anomaly_type.value}",
            anomaly_result=anomaly_result,
            suggested_position_size=position_size,
        )

    def generate_momentum_signal(
        self,
        anomaly_result: AnomalyResult,
        market_data: Optional[pd.Series] = None,
    ) -> TradingSignal:
        """
        Generate momentum signal (trade with anomaly).

        Logic: If price is breaking out anomalously, follow the trend.
        """
        if not anomaly_result.is_anomaly:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=1 - anomaly_result.score,
                strength=0.0,
                reason="No anomaly detected",
                anomaly_result=anomaly_result,
            )

        signal_type = SignalType.HOLD
        reason = ""

        if market_data is not None and "returns" in market_data.index:
            returns = market_data["returns"]
            if pd.notna(returns):
                # Only follow momentum for volume anomalies (could indicate breakout)
                if anomaly_result.anomaly_type == AnomalyType.VOLUME_ANOMALY:
                    if returns > 0:
                        signal_type = SignalType.BUY
                        reason = "High volume breakout - momentum buy"
                    else:
                        signal_type = SignalType.SELL
                        reason = "High volume breakdown - momentum sell"

                elif anomaly_result.anomaly_type == AnomalyType.PATTERN_BREAK:
                    if returns > 0:
                        signal_type = SignalType.BUY
                        reason = "Pattern breakout - momentum buy"
                    else:
                        signal_type = SignalType.SELL
                        reason = "Pattern breakdown - momentum sell"

        position_size = self._compute_position_size(anomaly_result)

        return TradingSignal(
            signal_type=signal_type,
            confidence=anomaly_result.confidence,
            strength=anomaly_result.score,
            reason=reason or "No momentum signal",
            anomaly_result=anomaly_result,
            suggested_position_size=position_size,
        )

    def generate_risk_signal(
        self,
        anomaly_result: AnomalyResult,
        current_position: float = 0.0,
    ) -> TradingSignal:
        """
        Generate risk management signal.

        Logic: Reduce exposure when anomalies indicate increased risk.
        """
        if not anomaly_result.is_anomaly:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=1 - anomaly_result.score,
                strength=0.0,
                reason="Risk levels normal",
                anomaly_result=anomaly_result,
            )

        # High anomaly score = reduce position
        if anomaly_result.score > 0.8:
            if current_position > 0:
                signal_type = SignalType.EXIT_LONG
                reason = "High risk - exit long position"
            elif current_position < 0:
                signal_type = SignalType.EXIT_SHORT
                reason = "High risk - exit short position"
            else:
                signal_type = SignalType.HOLD
                reason = "High risk - no new positions"
        elif anomaly_result.score > 0.6:
            signal_type = SignalType.REDUCE_POSITION
            reason = f"Elevated risk ({anomaly_result.anomaly_type.value}) - reduce position"
        else:
            signal_type = SignalType.HOLD
            reason = "Moderate anomaly - hold position"

        return TradingSignal(
            signal_type=signal_type,
            confidence=anomaly_result.confidence,
            strength=anomaly_result.score,
            reason=reason,
            anomaly_result=anomaly_result,
            suggested_position_size=self.base_position_size * (1 - anomaly_result.score),
        )

    def generate_signal(
        self,
        anomaly_result: AnomalyResult,
        market_data: Optional[pd.Series] = None,
        current_position: float = 0.0,
    ) -> TradingSignal:
        """
        Generate trading signal based on configured strategy.

        Args:
            anomaly_result: Result from anomaly detector
            market_data: Current market data (optional)
            current_position: Current position size

        Returns:
            TradingSignal
        """
        # Filter by thresholds
        if anomaly_result.score < self.min_anomaly_score:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=1 - anomaly_result.score,
                strength=0.0,
                reason="Anomaly score below threshold",
                anomaly_result=anomaly_result,
            )

        if anomaly_result.confidence < self.min_confidence:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=anomaly_result.confidence,
                strength=0.0,
                reason="Confidence below threshold",
                anomaly_result=anomaly_result,
            )

        # Generate based on strategy
        if self.strategy == "contrarian":
            signal = self.generate_contrarian_signal(anomaly_result, market_data)
        elif self.strategy == "momentum":
            signal = self.generate_momentum_signal(anomaly_result, market_data)
        elif self.strategy == "risk":
            signal = self.generate_risk_signal(anomaly_result, current_position)
        else:
            signal = TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                strength=0.0,
                reason=f"Unknown strategy: {self.strategy}",
                anomaly_result=anomaly_result,
            )

        # Add market data context
        if market_data is not None:
            signal.timestamp = market_data.get("timestamp")
            signal.price = market_data.get("close")

        self._recent_signals.append(signal)
        return signal

    def generate_signals(
        self,
        anomaly_results: List[AnomalyResult],
        market_data: pd.DataFrame,
    ) -> List[TradingSignal]:
        """
        Generate signals for multiple anomaly results.

        Args:
            anomaly_results: List of anomaly detection results
            market_data: DataFrame with market data (aligned with results)

        Returns:
            List of trading signals
        """
        signals = []

        for i, result in enumerate(anomaly_results):
            if i < len(market_data):
                row = market_data.iloc[i]
            else:
                row = None

            signal = self.generate_signal(result, row)
            signals.append(signal)

        return signals

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get summary statistics of recent signals."""
        if not self._recent_signals:
            return {"total_signals": 0}

        signal_types = [s.signal_type.value for s in self._recent_signals]
        anomaly_signals = [s for s in self._recent_signals if s.signal_type != SignalType.HOLD]

        return {
            "total_signals": len(self._recent_signals),
            "anomaly_signals": len(anomaly_signals),
            "signal_distribution": {
                st: signal_types.count(st) for st in set(signal_types)
            },
            "avg_confidence": np.mean([s.confidence for s in self._recent_signals]),
            "avg_strength": np.mean([s.strength for s in self._recent_signals]),
        }


class MultiStrategySignalGenerator:
    """
    Combine signals from multiple strategies.

    Useful for creating robust trading systems that consider
    multiple perspectives on anomalies.
    """

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-strategy generator.

        Args:
            strategies: List of strategy names
            weights: Weights for each strategy
        """
        strategies = strategies or ["contrarian", "momentum", "risk"]
        weights = weights or {s: 1.0/len(strategies) for s in strategies}

        self.generators = {
            name: AnomalySignalGenerator(strategy=name)
            for name in strategies
        }
        self.weights = weights

    def generate_signal(
        self,
        anomaly_result: AnomalyResult,
        market_data: Optional[pd.Series] = None,
    ) -> TradingSignal:
        """
        Generate combined signal from all strategies.

        Uses weighted voting to determine final signal.
        """
        signals = {
            name: gen.generate_signal(anomaly_result, market_data)
            for name, gen in self.generators.items()
        }

        # Weighted voting for signal type
        signal_votes: Dict[SignalType, float] = {}
        for name, signal in signals.items():
            weight = self.weights.get(name, 1.0)
            st = signal.signal_type
            signal_votes[st] = signal_votes.get(st, 0) + weight * signal.strength

        # Get winning signal type
        if signal_votes:
            winning_type = max(signal_votes, key=signal_votes.get)
        else:
            winning_type = SignalType.HOLD

        # Aggregate confidence and strength
        total_weight = sum(self.weights.values())
        avg_confidence = sum(
            self.weights.get(name, 1.0) * s.confidence
            for name, s in signals.items()
        ) / total_weight

        avg_strength = sum(
            self.weights.get(name, 1.0) * s.strength
            for name, s in signals.items()
        ) / total_weight

        # Combine reasons
        reasons = [
            f"{name}: {s.reason}"
            for name, s in signals.items()
            if s.signal_type != SignalType.HOLD
        ]

        return TradingSignal(
            signal_type=winning_type,
            confidence=avg_confidence,
            strength=avg_strength,
            reason=" | ".join(reasons) if reasons else "No strong signals",
            anomaly_result=anomaly_result,
        )


if __name__ == "__main__":
    # Example usage
    from detector import AnomalyResult, AnomalyType

    # Create sample anomaly result
    anomaly = AnomalyResult(
        is_anomaly=True,
        score=0.85,
        anomaly_type=AnomalyType.PRICE_SPIKE,
        confidence=0.9,
        explanation="Unusual price movement detected",
        details={"zscore": 4.2},
    )

    # Create sample market data
    market_data = pd.Series({
        "timestamp": pd.Timestamp.now(),
        "close": 50000.0,
        "returns": 0.05,  # 5% return
        "volume": 1000000,
    })

    # Test contrarian signal
    print("Testing Contrarian Strategy:")
    generator = AnomalySignalGenerator(strategy="contrarian")
    signal = generator.generate_signal(anomaly, market_data)
    print(f"Signal: {signal.signal_type.value}")
    print(f"Reason: {signal.reason}")
    print(f"Confidence: {signal.confidence:.2f}")

    # Test momentum signal
    print("\nTesting Momentum Strategy:")
    generator = AnomalySignalGenerator(strategy="momentum")
    signal = generator.generate_signal(anomaly, market_data)
    print(f"Signal: {signal.signal_type.value}")
    print(f"Reason: {signal.reason}")

    # Test multi-strategy
    print("\nTesting Multi-Strategy:")
    multi_gen = MultiStrategySignalGenerator()
    signal = multi_gen.generate_signal(anomaly, market_data)
    print(f"Signal: {signal.signal_type.value}")
    print(f"Reason: {signal.reason}")
