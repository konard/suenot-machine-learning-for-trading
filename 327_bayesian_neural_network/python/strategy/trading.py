"""
Uncertainty-aware trading strategy for Bayesian Neural Networks.

Implements trading logic that uses uncertainty estimates for:
- Signal generation
- Position sizing
- Risk management
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import torch


class SignalType(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class Signal:
    """Trading signal with uncertainty information."""
    timestamp: pd.Timestamp
    signal_type: SignalType
    probability: float  # Predicted probability
    uncertainty: float  # Epistemic uncertainty
    confidence: float  # 1 / (1 + uncertainty)
    position_size: float  # Recommended position size [0, 1]
    expected_return: Optional[float] = None
    aleatoric_uncertainty: Optional[float] = None

    def __repr__(self):
        return (
            f"Signal({self.signal_type.value}, "
            f"prob={self.probability:.3f}, "
            f"conf={self.confidence:.3f}, "
            f"size={self.position_size:.3f})"
        )


@dataclass
class Portfolio:
    """Portfolio state."""
    cash: float = 10000.0
    position: float = 0.0  # Positive = long, negative = short
    entry_price: float = 0.0
    total_value: float = 10000.0
    unrealized_pnl: float = 0.0


@dataclass
class Trade:
    """Completed trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float
    signal_type: SignalType
    pnl: float
    pnl_pct: float
    holding_period: int
    entry_confidence: float
    entry_uncertainty: float


@dataclass
class BacktestResult:
    """Backtesting results."""
    trades: List[Trade]
    equity_curve: pd.Series
    returns: pd.Series
    signals: List[Signal]

    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    n_trades: int = 0

    # Uncertainty analysis
    high_conf_win_rate: float = 0.0  # Win rate on high confidence trades
    low_conf_win_rate: float = 0.0  # Win rate on low confidence trades

    def summary(self) -> str:
        """Return summary string."""
        return f"""
Backtest Results
================
Total Return: {self.total_return:.2%}
Sharpe Ratio: {self.sharpe_ratio:.3f}
Sortino Ratio: {self.sortino_ratio:.3f}
Max Drawdown: {self.max_drawdown:.2%}
Win Rate: {self.win_rate:.2%}
Profit Factor: {self.profit_factor:.3f}
Number of Trades: {self.n_trades}
Avg Trade Return: {self.avg_trade_return:.2%}

Uncertainty Analysis:
- High Confidence Win Rate: {self.high_conf_win_rate:.2%}
- Low Confidence Win Rate: {self.low_conf_win_rate:.2%}
"""


class TradingStrategy:
    """
    Uncertainty-aware trading strategy using Bayesian Neural Networks.

    Features:
    - Signal generation based on predicted probabilities
    - Position sizing based on uncertainty/confidence
    - Risk management with uncertainty-adjusted stops
    - Support for both classification and regression outputs

    Args:
        model: Trained Bayesian Neural Network
        prob_threshold: Minimum probability for signal generation
        confidence_threshold: Minimum confidence for trading
        max_position_size: Maximum position size as fraction of portfolio
        use_kelly: Use Kelly Criterion for position sizing
        stop_loss: Stop loss percentage
        take_profit: Take profit percentage
    """

    def __init__(
        self,
        model: torch.nn.Module,
        prob_threshold: float = 0.55,
        confidence_threshold: float = 0.3,
        max_position_size: float = 0.25,
        use_kelly: bool = True,
        stop_loss: float = 0.02,
        take_profit: float = 0.04,
        n_mc_samples: int = 100,
        device: str = 'cpu'
    ):
        self.model = model
        self.prob_threshold = prob_threshold
        self.confidence_threshold = confidence_threshold
        self.max_position_size = max_position_size
        self.use_kelly = use_kelly
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.n_mc_samples = n_mc_samples
        self.device = device

        self.model.to(device)

    def generate_signal(
        self,
        features: np.ndarray,
        timestamp: pd.Timestamp
    ) -> Signal:
        """
        Generate trading signal from features.

        Args:
            features: Feature vector (1, n_features)
            timestamp: Current timestamp

        Returns:
            Signal object with recommendation
        """
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Monte Carlo inference
        self.model.train(False)
        probs_list = []
        return_means = []
        return_vars = []

        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                logits, return_dist = self.model(x, sample=True)
                probs = torch.softmax(logits, dim=-1)
                probs_list.append(probs)

                if return_dist is not None:
                    mu, log_var = return_dist
                    return_means.append(mu)
                    return_vars.append(torch.exp(log_var))

        # Stack and compute statistics
        probs_stack = torch.stack(probs_list, dim=0)  # (n_samples, 1, 3)
        mean_probs = probs_stack.mean(dim=0).squeeze()  # (3,)
        epistemic_unc = probs_stack.var(dim=0).sum(dim=-1).squeeze().item()

        prob_up = mean_probs[0].item()
        prob_neutral = mean_probs[1].item()
        prob_down = mean_probs[2].item()

        # Compute confidence
        confidence = 1.0 / (1.0 + epistemic_unc)

        # Expected return (if heteroscedastic)
        expected_return = None
        aleatoric_unc = None
        if return_means:
            means_stack = torch.stack(return_means, dim=0)
            vars_stack = torch.stack(return_vars, dim=0)
            expected_return = means_stack.mean().item()
            aleatoric_unc = vars_stack.mean().item()

        # Determine signal type
        if prob_up > self.prob_threshold and prob_up > prob_down:
            signal_type = SignalType.LONG
            probability = prob_up
        elif prob_down > self.prob_threshold and prob_down > prob_up:
            signal_type = SignalType.SHORT
            probability = prob_down
        else:
            signal_type = SignalType.NEUTRAL
            probability = prob_neutral

        # Position sizing
        if signal_type == SignalType.NEUTRAL or confidence < self.confidence_threshold:
            position_size = 0.0
        elif self.use_kelly:
            position_size = self._kelly_position_size(
                probability, confidence, epistemic_unc
            )
        else:
            position_size = confidence * self.max_position_size

        return Signal(
            timestamp=timestamp,
            signal_type=signal_type,
            probability=probability,
            uncertainty=epistemic_unc,
            confidence=confidence,
            position_size=position_size,
            expected_return=expected_return,
            aleatoric_uncertainty=aleatoric_unc
        )

    def _kelly_position_size(
        self,
        prob_win: float,
        confidence: float,
        uncertainty: float,
        win_loss_ratio: float = 2.0
    ) -> float:
        """
        Calculate position size using modified Kelly Criterion.

        Kelly: f* = (p * b - q) / b
        where p = prob of win, q = prob of loss, b = win/loss ratio

        Modified for uncertainty: f* = kelly * (1 - uncertainty_penalty)

        Args:
            prob_win: Probability of winning trade
            confidence: Model confidence
            uncertainty: Epistemic uncertainty
            win_loss_ratio: Expected win size / loss size

        Returns:
            Position size as fraction [0, max_position_size]
        """
        b = win_loss_ratio
        q = 1 - prob_win

        # Standard Kelly
        kelly = (prob_win * b - q) / b

        # Uncertainty penalty
        uncertainty_penalty = np.sqrt(uncertainty)  # Sqrt for gradual scaling
        adjusted_kelly = kelly * (1 - uncertainty_penalty) * confidence

        # Constrain to valid range
        position_size = max(0, min(self.max_position_size, adjusted_kelly))

        return position_size

    def backtest(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001
    ) -> BacktestResult:
        """
        Backtest the strategy on historical data.

        Args:
            features: Feature matrix (n_samples, n_features)
            prices: Price array (n_samples,)
            timestamps: Timestamp index
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction

        Returns:
            BacktestResult with performance metrics
        """
        n_samples = len(features)
        portfolio = Portfolio(cash=initial_capital, total_value=initial_capital)

        trades: List[Trade] = []
        signals: List[Signal] = []
        equity_curve = [initial_capital]
        daily_returns = [0.0]

        current_signal: Optional[Signal] = None
        entry_time: Optional[pd.Timestamp] = None
        entry_price: float = 0.0

        for i in range(n_samples):
            current_price = prices[i]
            timestamp = timestamps[i]

            # Update portfolio value
            if portfolio.position != 0:
                portfolio.unrealized_pnl = portfolio.position * (current_price - entry_price)
                portfolio.total_value = portfolio.cash + portfolio.position * current_price

            # Generate signal
            signal = self.generate_signal(features[i:i+1], timestamp)
            signals.append(signal)

            # Check for exit conditions
            if portfolio.position != 0:
                should_exit = False
                exit_reason = ""

                # Take profit
                pnl_pct = portfolio.unrealized_pnl / (abs(portfolio.position) * entry_price)
                if pnl_pct >= self.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"

                # Stop loss
                elif pnl_pct <= -self.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"

                # Signal reversal
                elif signal.signal_type != SignalType.NEUTRAL:
                    is_long = portfolio.position > 0
                    if (is_long and signal.signal_type == SignalType.SHORT) or \
                       (not is_long and signal.signal_type == SignalType.LONG):
                        should_exit = True
                        exit_reason = "signal_reversal"

                if should_exit and entry_time is not None and current_signal is not None:
                    # Close position
                    exit_price = current_price * (1 - transaction_cost * np.sign(portfolio.position))
                    pnl = portfolio.position * (exit_price - entry_price)

                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=timestamp,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=abs(portfolio.position),
                        signal_type=SignalType.LONG if portfolio.position > 0 else SignalType.SHORT,
                        pnl=pnl,
                        pnl_pct=pnl / (abs(portfolio.position) * entry_price),
                        holding_period=(timestamp - entry_time).days if hasattr(timestamp - entry_time, 'days') else 1,
                        entry_confidence=current_signal.confidence,
                        entry_uncertainty=current_signal.uncertainty
                    )
                    trades.append(trade)

                    portfolio.cash += portfolio.position * exit_price
                    portfolio.position = 0
                    portfolio.unrealized_pnl = 0
                    current_signal = None

            # Entry condition
            if portfolio.position == 0 and signal.signal_type != SignalType.NEUTRAL:
                if signal.position_size > 0 and signal.confidence >= self.confidence_threshold:
                    # Calculate position size
                    position_value = portfolio.cash * signal.position_size
                    entry_price = current_price * (1 + transaction_cost)

                    if signal.signal_type == SignalType.LONG:
                        portfolio.position = position_value / entry_price
                    else:
                        portfolio.position = -position_value / entry_price

                    portfolio.cash -= abs(portfolio.position) * entry_price
                    entry_time = timestamp
                    current_signal = signal

            # Update equity curve
            equity_curve.append(portfolio.total_value)
            if i > 0:
                daily_return = (portfolio.total_value - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_return)

        # Create result
        equity_series = pd.Series(equity_curve[1:], index=timestamps)
        returns_series = pd.Series(daily_returns[1:], index=timestamps[1:] if len(timestamps) > 1 else timestamps)

        result = BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            returns=returns_series,
            signals=signals
        )

        # Calculate metrics
        result = self._calculate_metrics(result, initial_capital)

        return result

    def _calculate_metrics(
        self,
        result: BacktestResult,
        initial_capital: float
    ) -> BacktestResult:
        """Calculate performance metrics."""
        # Total return
        final_value = result.equity_curve.iloc[-1] if len(result.equity_curve) > 0 else initial_capital
        result.total_return = (final_value - initial_capital) / initial_capital

        # Sharpe ratio (assuming risk-free rate = 0)
        if len(result.returns) > 0 and result.returns.std() > 0:
            result.sharpe_ratio = result.returns.mean() / result.returns.std() * np.sqrt(252)
        else:
            result.sharpe_ratio = 0.0

        # Sortino ratio
        downside_returns = result.returns[result.returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            result.sortino_ratio = result.returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            result.sortino_ratio = result.sharpe_ratio

        # Maximum drawdown
        cumulative = (1 + result.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        result.max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Trade statistics
        result.n_trades = len(result.trades)
        if result.n_trades > 0:
            wins = [t for t in result.trades if t.pnl > 0]
            losses = [t for t in result.trades if t.pnl <= 0]

            result.win_rate = len(wins) / result.n_trades

            gross_profit = sum(t.pnl for t in wins) if wins else 0
            gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            result.avg_trade_return = np.mean([t.pnl_pct for t in result.trades])

            # Uncertainty analysis
            median_conf = np.median([t.entry_confidence for t in result.trades])
            high_conf_trades = [t for t in result.trades if t.entry_confidence >= median_conf]
            low_conf_trades = [t for t in result.trades if t.entry_confidence < median_conf]

            if high_conf_trades:
                result.high_conf_win_rate = len([t for t in high_conf_trades if t.pnl > 0]) / len(high_conf_trades)
            if low_conf_trades:
                result.low_conf_win_rate = len([t for t in low_conf_trades if t.pnl > 0]) / len(low_conf_trades)

        return result
