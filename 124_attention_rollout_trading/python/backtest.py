"""
backtest.py - Backtesting framework with attention analysis

This module provides a backtesting framework that incorporates
attention rollout analysis for interpretable trading evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch


@dataclass
class BacktestResult:
    """Results from backtesting a trading strategy."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    returns: np.ndarray
    equity_curve: np.ndarray
    attention_analysis: Optional[Dict] = None


class AttentionBacktester:
    """
    Backtesting framework that incorporates attention rollout analysis.

    This backtester not only evaluates trading performance but also
    analyzes attention patterns to understand model behavior.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        attention_rollout,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001
    ):
        """
        Initialize backtester.

        Args:
            model: Trained trading model
            attention_rollout: AttentionRollout instance
            initial_capital: Starting capital
            transaction_cost: Cost per transaction (fraction)
        """
        self.model = model
        self.attention_rollout = attention_rollout
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run_backtest(
        self,
        X: np.ndarray,
        prices: np.ndarray,
        threshold: float = 0.6
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            X: Feature sequences (n_samples, seq_len, n_features)
            prices: Close prices aligned with X
            threshold: Confidence threshold for trading

        Returns:
            BacktestResult with performance metrics
        """
        self.model.eval()

        n_samples = len(X)
        positions = np.zeros(n_samples)
        returns = np.zeros(n_samples)
        attention_patterns = []

        capital = self.initial_capital
        equity_curve = [capital]
        current_position = 0
        n_trades = 0
        wins = 0

        for i in range(n_samples - 1):
            # Get model prediction
            x_tensor = torch.FloatTensor(X[i:i+1])

            with torch.no_grad():
                logits, _ = self.model(x_tensor)
                probs = torch.softmax(logits, dim=-1).numpy()[0]

            # Get attention rollout
            attribution = self.attention_rollout.get_input_attribution(x_tensor)
            attention_patterns.append(attribution)

            # Determine position
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]

            if confidence > threshold:
                if pred_class == 2:  # Buy signal
                    new_position = 1
                elif pred_class == 0:  # Sell signal
                    new_position = -1
                else:  # Hold
                    new_position = current_position
            else:
                new_position = 0  # No position if low confidence

            # Calculate returns
            price_return = (prices[i+1] - prices[i]) / prices[i]

            # Apply transaction costs on position changes
            if new_position != current_position:
                n_trades += 1
                trade_cost = self.transaction_cost * abs(new_position - current_position)
            else:
                trade_cost = 0

            position_return = current_position * price_return - trade_cost
            returns[i] = position_return

            if position_return > 0:
                wins += 1

            capital *= (1 + position_return)
            equity_curve.append(capital)

            positions[i] = current_position
            current_position = new_position

        # Calculate metrics
        equity_curve = np.array(equity_curve)

        result = BacktestResult(
            total_return=(capital - self.initial_capital) / self.initial_capital,
            sharpe_ratio=self._calculate_sharpe(returns),
            sortino_ratio=self._calculate_sortino(returns),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            win_rate=wins / max(n_trades, 1),
            n_trades=n_trades,
            returns=returns,
            equity_curve=equity_curve,
            attention_analysis=self._analyze_attention_patterns(
                attention_patterns, returns
            )
        )

        return result

    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_sortino(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate annualized Sortino ratio."""
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / downside.std()

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())

    def _analyze_attention_patterns(
        self,
        attention_patterns: List[np.ndarray],
        returns: np.ndarray
    ) -> Dict:
        """
        Analyze attention patterns relative to trading performance.

        Returns insights about which attention patterns correlate
        with profitable vs unprofitable trades.
        """
        if not attention_patterns:
            return {}

        attention_matrix = np.array(attention_patterns)

        # Separate winning and losing trades
        winning_mask = returns[:-1] > 0
        losing_mask = returns[:-1] < 0

        if not winning_mask.any() or not losing_mask.any():
            return {}

        # Average attention patterns
        avg_winning_attention = attention_matrix[winning_mask].mean(axis=0)
        avg_losing_attention = attention_matrix[losing_mask].mean(axis=0)

        # Attention concentration (entropy)
        def entropy(p):
            p = p + 1e-10
            return -np.sum(p * np.log(p))

        winning_concentration = np.mean([
            entropy(att) for att in attention_matrix[winning_mask]
        ])
        losing_concentration = np.mean([
            entropy(att) for att in attention_matrix[losing_mask]
        ])

        return {
            "avg_winning_attention": avg_winning_attention.tolist(),
            "avg_losing_attention": avg_losing_attention.tolist(),
            "winning_attention_entropy": winning_concentration,
            "losing_attention_entropy": losing_concentration,
            "recent_bias_winning": float(avg_winning_attention[-5:].sum()),
            "recent_bias_losing": float(avg_losing_attention[-5:].sum())
        }


def print_backtest_report(result: BacktestResult) -> None:
    """Print formatted backtest report."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return:     {result.total_return*100:>10.2f}%")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:    {result.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:     {result.max_drawdown*100:>10.2f}%")
    print(f"Win Rate:         {result.win_rate*100:>10.2f}%")
    print(f"Number of Trades: {result.n_trades:>10d}")
    print("="*60)

    if result.attention_analysis:
        print("\nATTENTION ANALYSIS")
        print("-"*60)
        if "winning_attention_entropy" in result.attention_analysis:
            print(f"Winning trades attention entropy: "
                  f"{result.attention_analysis['winning_attention_entropy']:.4f}")
            print(f"Losing trades attention entropy:  "
                  f"{result.attention_analysis['losing_attention_entropy']:.4f}")
        if "recent_bias_winning" in result.attention_analysis:
            print(f"Recent bias (winning): "
                  f"{result.attention_analysis['recent_bias_winning']:.4f}")
            print(f"Recent bias (losing):  "
                  f"{result.attention_analysis['recent_bias_losing']:.4f}")
    print()
