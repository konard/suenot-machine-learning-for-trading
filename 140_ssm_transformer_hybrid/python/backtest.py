"""
Backtesting framework for SSM-Transformer Hybrid trading model.

Evaluates model predictions as trading signals and computes
performance metrics: Sharpe ratio, Sortino ratio, max drawdown,
win rate, and profit factor.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    pnl: float = 0.0

    def __post_init__(self):
        self.pnl = self.direction * (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class BacktestResults:
    """Container for backtest results."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    n_trades: int = 0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "calmar_ratio": self.calmar_ratio,
            "n_trades": self.n_trades,
        }


class SSMHybridBacktester:
    """
    Backtesting engine for SSM-Transformer Hybrid model.

    Strategy:
    - Model predicts direction probability, volatility, and return magnitude
    - Position is taken when direction probability exceeds threshold
    - Position size is scaled by confidence and inverse volatility
    - Positions are closed on signal reversal
    """

    def __init__(
        self,
        model,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        position_size: float = 0.1,
        direction_threshold: float = 0.55,
        max_position: float = 1.0,
    ):
        self.model = model
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        self.direction_threshold = direction_threshold
        self.max_position = max_position

    @torch.no_grad()
    def run(self, data_loader) -> Dict[str, float]:
        """
        Run backtest on data from a DataLoader.

        Args:
            data_loader: DataLoader yielding (features, targets) batches

        Returns:
            Dictionary of performance metrics
        """
        self.model.eval()
        all_preds = {"direction": [], "volatility": [], "return_mag": []}
        all_targets = {"direction": [], "volatility": [], "return_mag": []}

        for features, targets in data_loader:
            outputs = self.model(features)
            for key in all_preds:
                all_preds[key].append(outputs[key].cpu().numpy())
                all_targets[key].append(targets[key].cpu().numpy())

        # Concatenate
        for key in all_preds:
            all_preds[key] = np.concatenate(all_preds[key])
            all_targets[key] = np.concatenate(all_targets[key])

        return self._simulate_trading(all_preds, all_targets)

    def run_from_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        actual_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Run backtest from pre-computed predictions.

        Args:
            predictions: Dict with direction, volatility, return_mag arrays
            actual_returns: Actual returns for each prediction

        Returns:
            Dictionary of performance metrics
        """
        targets = {
            "direction": (actual_returns > 0).astype(float),
            "volatility": np.abs(actual_returns),
            "return_mag": actual_returns,
        }
        return self._simulate_trading(predictions, targets)

    def _simulate_trading(
        self,
        preds: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Simulate trading and compute metrics."""
        n = len(preds["direction"])
        actual_returns = targets["return_mag"]

        # Generate position signals
        positions = np.zeros(n)
        for i in range(n):
            dir_prob = preds["direction"][i]
            vol = max(preds["volatility"][i], 1e-6)

            if dir_prob > self.direction_threshold:
                confidence = (dir_prob - 0.5) * 2
                positions[i] = min(
                    confidence * self.position_size / vol,
                    self.max_position,
                )
            elif dir_prob < (1 - self.direction_threshold):
                confidence = (0.5 - dir_prob) * 2
                positions[i] = -min(
                    confidence * self.position_size / vol,
                    self.max_position,
                )

        # Compute returns with transaction costs
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = position_changes * self.transaction_cost
        strategy_returns = positions * actual_returns - costs

        # Equity curve
        equity = self.initial_capital * np.cumprod(1 + strategy_returns)
        equity_list = [self.initial_capital] + equity.tolist()

        # Compute metrics
        results = BacktestResults()
        results.equity_curve = equity_list
        results.total_return = (equity[-1] / self.initial_capital - 1) if len(equity) > 0 else 0.0

        # Sharpe ratio (annualized, assuming hourly data → 8760 hours/year)
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 1e-10:
            results.sharpe_ratio = (
                np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(8760)
            )
        else:
            results.sharpe_ratio = 0.0

        # Sortino ratio
        downside = strategy_returns[strategy_returns < 0]
        if len(downside) > 0 and np.std(downside) > 1e-10:
            results.sortino_ratio = (
                np.mean(strategy_returns) / np.std(downside) * np.sqrt(8760)
            )
        else:
            results.sortino_ratio = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + 1e-10)
        results.max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # Win rate and profit factor
        winning = strategy_returns[strategy_returns > 0]
        losing = strategy_returns[strategy_returns < 0]
        active_trades = strategy_returns[positions != 0]
        results.n_trades = int(np.sum(position_changes > 0))
        if len(active_trades) > 0:
            results.win_rate = len(winning) / max(len(active_trades), 1)
        else:
            results.win_rate = 0.0

        gross_profit = np.sum(winning) if len(winning) > 0 else 0.0
        gross_loss = -np.sum(losing) if len(losing) > 0 else 0.0
        results.profit_factor = (
            gross_profit / max(gross_loss, 1e-10) if gross_loss > 1e-10 else 0.0
        )

        # Calmar ratio
        if abs(results.max_drawdown) > 1e-10:
            annualized_return = results.total_return  # simplified
            results.calmar_ratio = annualized_return / abs(results.max_drawdown)
        else:
            results.calmar_ratio = 0.0

        return results.to_dict()
