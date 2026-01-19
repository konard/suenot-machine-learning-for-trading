"""
Risk-Based Trading Strategy Module

This module provides classes for generating trading signals and managing
positions based on LLM risk assessments.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging

import numpy as np

from risk_assessment import RiskScore, RiskLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    EXIT = "exit"


@dataclass
class TradingSignal:
    """Trading signal with position sizing."""
    signal_type: SignalType
    position_size: float
    confidence: float
    risk_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal": self.signal_type.value,
            "position_size": self.position_size,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
        }


@dataclass
class Position:
    """Trading position."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == "long":
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.side == "long":
            return (current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - current_price) / self.entry_price * 100


class RiskBasedTrader:
    """
    Trading strategy based on LLM risk assessment.

    Generates trading signals based on risk scores and manages
    position sizing inversely proportional to risk.
    """

    def __init__(
        self,
        risk_threshold_long: float = 4.0,
        risk_threshold_short: float = 7.0,
        max_position_size: float = 1.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ):
        """
        Initialize trader.

        Args:
            risk_threshold_long: Max risk score to go long
            risk_threshold_short: Min risk score to go short
            max_position_size: Maximum position size (1.0 = 100%)
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.risk_threshold_long = risk_threshold_long
        self.risk_threshold_short = risk_threshold_short
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.risk_history: List[RiskScore] = []
        self.signal_history: List[TradingSignal] = []
        self.current_position: Optional[Position] = None

    def generate_signal(
        self,
        risk_score: RiskScore,
        current_price: float,
    ) -> TradingSignal:
        """
        Generate trading signal from risk assessment.

        Args:
            risk_score: Risk score from LLM assessment
            current_price: Current market price

        Returns:
            TradingSignal with signal type and position size
        """
        self.risk_history.append(risk_score)
        overall = risk_score.overall()

        # Check if we should exit current position
        if self.current_position:
            should_exit, exit_reason = self._check_exit(risk_score, current_price)
            if should_exit:
                signal = TradingSignal(
                    signal_type=SignalType.EXIT,
                    position_size=0.0,
                    confidence=self._confidence_to_float(risk_score.confidence),
                    risk_score=overall,
                    reason=exit_reason,
                )
                self.signal_history.append(signal)
                return signal

        # Calculate position size
        position_size = self._calculate_position_size(risk_score)

        # Generate signal based on risk level
        if overall <= self.risk_threshold_long:
            # Low risk - consider going long
            if risk_score.direction in ["stable", "decreasing"]:
                signal = TradingSignal(
                    signal_type=SignalType.LONG,
                    position_size=position_size,
                    confidence=self._confidence_to_float(risk_score.confidence),
                    risk_score=overall,
                    reason=f"Low risk ({overall:.1f}) with {risk_score.direction} trend",
                )
            else:
                signal = TradingSignal(
                    signal_type=SignalType.NEUTRAL,
                    position_size=0.0,
                    confidence=self._confidence_to_float(risk_score.confidence),
                    risk_score=overall,
                    reason=f"Low risk but increasing trend",
                )

        elif overall >= self.risk_threshold_short:
            # High risk - consider going short
            if risk_score.direction in ["stable", "increasing"]:
                signal = TradingSignal(
                    signal_type=SignalType.SHORT,
                    position_size=position_size * 0.5,  # Smaller short positions
                    confidence=self._confidence_to_float(risk_score.confidence),
                    risk_score=overall,
                    reason=f"High risk ({overall:.1f}) with {risk_score.direction} trend",
                )
            else:
                signal = TradingSignal(
                    signal_type=SignalType.NEUTRAL,
                    position_size=0.0,
                    confidence=self._confidence_to_float(risk_score.confidence),
                    risk_score=overall,
                    reason=f"High risk but decreasing trend",
                )

        else:
            # Medium risk - stay neutral
            signal = TradingSignal(
                signal_type=SignalType.NEUTRAL,
                position_size=0.0,
                confidence=self._confidence_to_float(risk_score.confidence),
                risk_score=overall,
                reason=f"Medium risk ({overall:.1f}) - no clear signal",
            )

        self.signal_history.append(signal)
        return signal

    def _calculate_position_size(self, risk: RiskScore) -> float:
        """Calculate position size based on risk score."""
        # Lower risk = larger position
        risk_factor = 1 - (risk.overall() / 10)

        # Confidence adjustment
        confidence_multiplier = self._confidence_to_float(risk.confidence)

        return self.max_position_size * risk_factor * confidence_multiplier

    def _confidence_to_float(self, confidence: str) -> float:
        """Convert confidence string to float."""
        return {
            "high": 1.0,
            "medium": 0.75,
            "low": 0.5,
        }.get(confidence, 0.5)

    def _check_exit(
        self,
        risk_score: RiskScore,
        current_price: float,
    ) -> Tuple[bool, str]:
        """Check if current position should be exited."""
        if not self.current_position:
            return False, ""

        pos = self.current_position
        overall = risk_score.overall()

        # Check stop loss
        if pos.stop_loss and current_price <= pos.stop_loss:
            return True, "Stop loss triggered"

        # Check take profit
        if pos.take_profit and current_price >= pos.take_profit:
            return True, "Take profit triggered"

        # Check risk reversal
        if pos.side == "long" and overall >= self.risk_threshold_short:
            return True, f"Risk increased to {overall:.1f}"

        if pos.side == "short" and overall <= self.risk_threshold_long:
            return True, f"Risk decreased to {overall:.1f}"

        return False, ""

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
    ) -> Position:
        """Open a new position."""
        stop_loss = price * (1 - self.stop_loss_pct) if side == "long" else price * (1 + self.stop_loss_pct)
        take_profit = price * (1 + self.take_profit_pct) if side == "long" else price * (1 - self.take_profit_pct)

        self.current_position = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            size=size,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        logger.info(f"Opened {side} position: {size} @ {price}")
        return self.current_position

    def close_position(self, price: float) -> float:
        """Close current position and return P&L."""
        if not self.current_position:
            return 0.0

        pnl = self.current_position.unrealized_pnl(price)
        logger.info(f"Closed position @ {price}, P&L: {pnl:.2f}")

        self.current_position = None
        return pnl

    def get_risk_trend(self, window: int = 5) -> str:
        """Analyze recent risk trend."""
        if len(self.risk_history) < window:
            return "insufficient_data"

        recent_scores = [r.overall() for r in self.risk_history[-window:]]
        slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        if slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        return "stable"


@dataclass
class BacktestResult:
    """Backtest performance results."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    equity_curve: List[float]

    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Total Return:    {self.total_return:>10.2f}%")
        print(f"Sharpe Ratio:    {self.sharpe_ratio:>10.2f}")
        print(f"Sortino Ratio:   {self.sortino_ratio:>10.2f}")
        print(f"Max Drawdown:    {self.max_drawdown:>10.2f}%")
        print("-" * 50)
        print(f"Total Trades:    {self.total_trades:>10}")
        print(f"Win Rate:        {self.win_rate:>10.2f}%")
        print(f"Profit Factor:   {self.profit_factor:>10.2f}")
        print(f"Winning Trades:  {self.winning_trades:>10}")
        print(f"Losing Trades:   {self.losing_trades:>10}")
        print(f"Avg Win:         {self.avg_win:>10.2f}")
        print(f"Avg Loss:        {self.avg_loss:>10.2f}")
        print("=" * 50)


class Backtester:
    """
    Backtesting framework for risk-based trading strategies.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,
        slippage: float = 0.0005,
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission_rate: Commission per trade (0.001 = 0.1%)
            slippage: Slippage per trade (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage

    def run(
        self,
        prices: List[float],
        risk_scores: List[RiskScore],
        trader: RiskBasedTrader,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            prices: List of prices
            risk_scores: List of risk scores (aligned with prices)
            trader: RiskBasedTrader instance

        Returns:
            BacktestResult with performance metrics
        """
        if len(prices) != len(risk_scores):
            raise ValueError("Prices and risk scores must have same length")

        capital = self.initial_capital
        position_size = 0.0
        position_side: Optional[str] = None
        entry_price = 0.0

        equity_curve = [capital]
        trades: List[float] = []
        wins: List[float] = []
        losses: List[float] = []

        for i, (price, risk) in enumerate(zip(prices, risk_scores)):
            signal = trader.generate_signal(risk, price)

            # Handle signals
            if signal.signal_type == SignalType.EXIT and position_size > 0:
                # Close position
                pnl = self._calculate_pnl(
                    entry_price, price, position_size, position_side
                )
                capital += pnl
                trades.append(pnl)

                if pnl > 0:
                    wins.append(pnl)
                else:
                    losses.append(abs(pnl))

                position_size = 0.0
                position_side = None

            elif signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                # Close existing position if opposite side
                if position_size > 0 and position_side != signal.signal_type.value:
                    pnl = self._calculate_pnl(
                        entry_price, price, position_size, position_side
                    )
                    capital += pnl
                    trades.append(pnl)

                    if pnl > 0:
                        wins.append(pnl)
                    else:
                        losses.append(abs(pnl))

                    position_size = 0.0

                # Open new position
                if position_size == 0:
                    position_size = capital * signal.position_size
                    position_side = signal.signal_type.value
                    entry_price = price * (1 + self.slippage)

            # Update equity
            if position_size > 0:
                current_pnl = self._calculate_pnl(
                    entry_price, price, position_size, position_side
                )
                equity_curve.append(capital + current_pnl)
            else:
                equity_curve.append(capital)

        # Close final position
        if position_size > 0:
            pnl = self._calculate_pnl(
                entry_price, prices[-1], position_size, position_side
            )
            capital += pnl
            trades.append(pnl)

            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(abs(pnl))

        # Calculate metrics
        return self._calculate_metrics(
            equity_curve, trades, wins, losses
        )

    def _calculate_pnl(
        self,
        entry: float,
        exit: float,
        size: float,
        side: Optional[str],
    ) -> float:
        """Calculate P&L for a trade."""
        if side == "long":
            gross_pnl = (exit - entry) / entry * size
        else:
            gross_pnl = (entry - exit) / entry * size

        commission = size * self.commission_rate * 2  # Entry and exit
        return gross_pnl - commission

    def _calculate_metrics(
        self,
        equity_curve: List[float],
        trades: List[float],
        wins: List[float],
        losses: List[float],
    ) -> BacktestResult:
        """Calculate performance metrics."""
        # Returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital * 100

        # Sharpe ratio (annualized, assuming daily data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio
        downside_returns = [r for r in returns if r < 0]
        if downside_returns and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino = sharpe

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Trade statistics
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=equity_curve,
        )


# Example usage
if __name__ == "__main__":
    import random

    print("Risk-Based Trading Strategy Demo")
    print("=" * 50)

    # Create trader
    trader = RiskBasedTrader(
        risk_threshold_long=4.0,
        risk_threshold_short=7.0,
        max_position_size=0.5,
    )

    # Generate synthetic data
    np.random.seed(42)
    n_periods = 100

    # Synthetic prices (random walk)
    prices = [100.0]
    for _ in range(n_periods - 1):
        change = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + change))

    # Synthetic risk scores
    risk_scores = []
    for i in range(n_periods):
        base = 5.0 + np.sin(i / 10) * 2  # Oscillating base
        noise = np.random.normal(0, 0.5)
        score = max(1, min(10, base + noise))

        risk_scores.append(RiskScore(
            market_risk=score,
            credit_risk=score * 0.8,
            liquidity_risk=score * 0.6,
            operational_risk=score * 0.7,
            regulatory_risk=score * 0.9,
            sentiment_risk=score * 0.85,
            confidence=random.choice(["low", "medium", "high"]),
            direction=random.choice(["increasing", "stable", "decreasing"]),
        ))

    # Generate signals
    print("\nGenerating signals...")
    for i in range(5):
        signal = trader.generate_signal(risk_scores[i], prices[i])
        print(f"  Day {i+1}: {signal.signal_type.value} (risk={signal.risk_score:.1f})")

    # Run backtest
    print("\nRunning backtest...")
    backtester = Backtester(
        initial_capital=10000.0,
        commission_rate=0.001,
        slippage=0.0005,
    )

    # Reset trader for clean backtest
    trader = RiskBasedTrader()
    result = backtester.run(prices, risk_scores, trader)

    result.print_summary()
