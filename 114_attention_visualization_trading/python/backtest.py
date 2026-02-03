"""Backtesting framework for attention-based trading strategies.

Provides tools for:
- Running backtests with attention-based confidence filtering
- Computing trading metrics (Sharpe, Sortino, drawdown, etc.)
- Analyzing strategy performance
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .model import AttentionTransformer
from .visualization import compute_attention_confidence


class Signal(Enum):
    """Trading signal types."""
    HOLD = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Represents a single trade."""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    signal: Signal
    pnl: float
    confidence: float


@dataclass
class TradingMetrics:
    """Container for trading performance metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_duration: float
    avg_confidence: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "num_trades": self.num_trades,
            "avg_trade_duration": self.avg_trade_duration,
            "avg_confidence": self.avg_confidence,
        }


class AttentionTradingStrategy:
    """Trading strategy with attention-based confidence filtering.

    This strategy uses the attention weights to assess prediction confidence
    and only trades when the model is sufficiently focused.

    Args:
        direction_threshold_long: Min direction probability for long signal
        direction_threshold_short: Max direction probability for short signal
        return_threshold: Minimum absolute predicted return
        confidence_threshold: Minimum attention confidence
        position_size: Default position size (fraction of portfolio)
    """

    def __init__(
        self,
        direction_threshold_long: float = 0.6,
        direction_threshold_short: float = 0.4,
        return_threshold: float = 0.001,
        confidence_threshold: float = 0.5,
        position_size: float = 1.0,
    ):
        self.direction_threshold_long = direction_threshold_long
        self.direction_threshold_short = direction_threshold_short
        self.return_threshold = return_threshold
        self.confidence_threshold = confidence_threshold
        self.position_size = position_size

    def generate_signal(
        self,
        pred_return: float,
        pred_direction: float,
        attention_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Signal, float]:
        """Generate trading signal based on predictions and attention.

        Args:
            pred_return: Predicted return
            pred_direction: Direction probability (>0.5 = up)
            attention_weights: Optional attention weights for confidence

        Returns:
            Tuple of (signal, confidence)
        """
        # Compute attention-based confidence
        if attention_weights is not None:
            # Stack all layer attentions
            all_attn = torch.stack([
                attention_weights[k] for k in attention_weights
            ], dim=0)
            # Average over layers
            avg_attn = all_attn.mean(dim=0)
            confidence = compute_attention_confidence(avg_attn).item()
        else:
            confidence = 1.0

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return Signal.HOLD, confidence

        # Generate signal based on predictions
        if (
            pred_direction > self.direction_threshold_long
            and pred_return > self.return_threshold
        ):
            return Signal.LONG, confidence

        elif (
            pred_direction < self.direction_threshold_short
            and pred_return < -self.return_threshold
        ):
            return Signal.SHORT, confidence

        return Signal.HOLD, confidence


class Backtester:
    """Backtesting engine for attention-based trading strategies.

    Args:
        model: Trained AttentionTransformer model
        strategy: Trading strategy instance
        initial_capital: Starting capital
        transaction_cost: Cost per trade as fraction of position
    """

    def __init__(
        self,
        model: AttentionTransformer,
        strategy: AttentionTradingStrategy,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
    ):
        self.model = model
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    @torch.no_grad()
    def run(
        self,
        features: torch.Tensor,
        prices: np.ndarray,
        timestamps: Optional[List] = None,
    ) -> Tuple[TradingMetrics, List[Trade], pd.DataFrame]:
        """Run backtest on historical data.

        Args:
            features: Feature tensor of shape (n_samples, seq_len, n_features)
            prices: Price array of shape (n_samples,)
            timestamps: Optional list of timestamps

        Returns:
            Tuple of (metrics, trades, equity_curve)
        """
        self.model.eval()
        n_samples = len(features)

        # Track state
        equity = self.initial_capital
        equity_curve = [equity]
        position = Signal.HOLD
        entry_price = 0.0
        entry_idx = 0
        entry_confidence = 0.0
        trades = []
        confidences = []

        for i in range(n_samples):
            # Get model predictions
            x = features[i:i+1]  # (1, seq_len, n_features)
            pred_ret, pred_dir, pred_vol, attention = self.model(x)

            # Generate signal
            signal, confidence = self.strategy.generate_signal(
                pred_ret.item(),
                pred_dir.item(),
                attention,
            )
            confidences.append(confidence)

            current_price = prices[i]

            # Handle position changes
            if position == Signal.HOLD and signal != Signal.HOLD:
                # Enter position
                position = signal
                entry_price = current_price
                entry_idx = i
                entry_confidence = confidence
                # Deduct transaction cost
                equity *= (1 - self.transaction_cost)

            elif position != Signal.HOLD and (signal != position or i == n_samples - 1):
                # Exit position
                if position == Signal.LONG:
                    pnl = (current_price - entry_price) / entry_price
                else:  # SHORT
                    pnl = (entry_price - current_price) / entry_price

                # Apply PnL and transaction cost
                equity *= (1 + pnl * self.strategy.position_size)
                equity *= (1 - self.transaction_cost)

                # Record trade
                trades.append(Trade(
                    entry_idx=entry_idx,
                    exit_idx=i,
                    entry_price=entry_price,
                    exit_price=current_price,
                    signal=position,
                    pnl=pnl,
                    confidence=entry_confidence,
                ))

                # Enter new position if signal is not HOLD
                if signal != Signal.HOLD and i < n_samples - 1:
                    position = signal
                    entry_price = current_price
                    entry_idx = i
                    entry_confidence = confidence
                    equity *= (1 - self.transaction_cost)
                else:
                    position = Signal.HOLD

            equity_curve.append(equity)

        # Compute metrics
        metrics = self._compute_metrics(
            equity_curve, trades, confidences, n_samples
        )

        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            "timestamp": timestamps[:len(equity_curve)] if timestamps else range(len(equity_curve)),
            "equity": equity_curve,
        })

        return metrics, trades, equity_df

    def _compute_metrics(
        self,
        equity_curve: List[float],
        trades: List[Trade],
        confidences: List[float],
        n_samples: int,
    ) -> TradingMetrics:
        """Compute trading metrics from backtest results."""
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Total and annualized return
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        # Assume 252 trading days
        annualized_return = (1 + total_return) ** (252 / n_samples) - 1

        # Sharpe ratio (annualized)
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (annualized)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()

        # Trade metrics
        num_trades = len(trades)
        if num_trades > 0:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            win_rate = len(winning_trades) / num_trades

            gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1e-9
            profit_factor = gross_profit / gross_loss

            avg_duration = np.mean([t.exit_idx - t.entry_idx for t in trades])
            avg_confidence = np.mean([t.confidence for t in trades])
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_duration = 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0

        return TradingMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
            avg_trade_duration=avg_duration,
            avg_confidence=avg_confidence,
        )


def compare_strategies(
    model: AttentionTransformer,
    features: torch.Tensor,
    prices: np.ndarray,
    confidence_thresholds: List[float] = [0.0, 0.3, 0.5, 0.7],
) -> pd.DataFrame:
    """Compare strategies with different confidence thresholds.

    Args:
        model: Trained model
        features: Feature tensor
        prices: Price array
        confidence_thresholds: List of thresholds to compare

    Returns:
        DataFrame with comparison results
    """
    results = []

    for threshold in confidence_thresholds:
        strategy = AttentionTradingStrategy(confidence_threshold=threshold)
        backtester = Backtester(model, strategy)
        metrics, _, _ = backtester.run(features, prices)

        result = {"confidence_threshold": threshold}
        result.update(metrics.to_dict())
        results.append(result)

    return pd.DataFrame(results)
