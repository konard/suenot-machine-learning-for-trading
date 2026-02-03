"""
Backtesting framework for SHAP-enhanced trading strategies.

This module provides a comprehensive backtesting engine for evaluating
trading strategies that incorporate SHAP-based model interpretability.

Features:
- Standard performance metrics (Sharpe, Sortino, Max Drawdown)
- SHAP-enhanced signal filtering
- Position sizing based on prediction confidence
- Transaction cost modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    returns: np.ndarray
    positions: np.ndarray
    equity_curve: np.ndarray
    metrics: Dict[str, float]
    trades: List[Dict]

    def summary(self) -> str:
        """Generate summary string of results."""
        lines = ["=" * 50, "Backtest Results", "=" * 50]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 1.0  # Maximum position as fraction of capital
    min_confidence: float = 0.5  # Minimum prediction confidence
    use_shap_filter: bool = True  # Use SHAP-based filtering
    shap_stability_threshold: float = 0.3  # SHAP stability threshold


class BacktestEngine:
    """
    Backtesting engine for trading strategies.

    Supports both standard signal-based strategies and SHAP-enhanced
    strategies with confidence filtering.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine.

        Args:
            config: Backtesting configuration.
        """
        self.config = config or BacktestConfig()

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        shap_stability: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run backtest on price data with trading signals.

        Args:
            prices: Array of closing prices.
            signals: Array of trading signals (probability of positive return).
            shap_stability: Optional array of SHAP stability scores (0-1).

        Returns:
            BacktestResult with performance metrics.
        """
        n = len(prices)
        if len(signals) != n:
            raise ValueError("prices and signals must have same length")

        # Initialize arrays
        positions = np.zeros(n)
        returns = np.zeros(n)
        equity = np.zeros(n)
        equity[0] = self.config.initial_capital
        trades = []

        # Calculate returns
        price_returns = np.zeros(n)
        price_returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

        current_position = 0.0

        for i in range(1, n):
            # Determine target position based on signal
            target_position = self._compute_target_position(
                signals[i - 1],
                shap_stability[i - 1] if shap_stability is not None else None,
            )

            # Check if position changes (trade)
            if target_position != current_position:
                # Apply transaction costs
                trade_cost = abs(target_position - current_position) * (
                    self.config.transaction_cost + self.config.slippage
                )

                trades.append({
                    "index": i,
                    "price": prices[i],
                    "old_position": current_position,
                    "new_position": target_position,
                    "cost": trade_cost,
                })

                current_position = target_position
            else:
                trade_cost = 0.0

            positions[i] = current_position

            # Calculate strategy return
            strategy_return = current_position * price_returns[i] - trade_cost
            returns[i] = strategy_return

            # Update equity
            equity[i] = equity[i - 1] * (1 + strategy_return)

        # Compute performance metrics
        metrics = self._compute_metrics(returns, equity, trades)

        return BacktestResult(
            returns=returns,
            positions=positions,
            equity_curve=equity,
            metrics=metrics,
            trades=trades,
        )

    def _compute_target_position(
        self,
        signal: float,
        shap_stability: Optional[float] = None,
    ) -> float:
        """
        Compute target position from signal.

        Args:
            signal: Trading signal (0-1 probability).
            shap_stability: SHAP stability score (0-1).

        Returns:
            Target position (-1 to 1).
        """
        # Convert probability to position (-1 = short, 0 = flat, 1 = long)
        if signal > 0.5 + (self.config.min_confidence - 0.5):
            base_position = (signal - 0.5) * 2  # Scale to 0-1
        elif signal < 0.5 - (self.config.min_confidence - 0.5):
            base_position = (signal - 0.5) * 2  # Scale to -1-0
        else:
            base_position = 0.0

        # Apply SHAP stability filter
        if self.config.use_shap_filter and shap_stability is not None:
            if shap_stability < self.config.shap_stability_threshold:
                # Reduce position when explanations are unstable
                base_position *= shap_stability / self.config.shap_stability_threshold

        # Clip to max position size
        position = np.clip(
            base_position,
            -self.config.max_position_size,
            self.config.max_position_size,
        )

        return position

    def _compute_metrics(
        self,
        returns: np.ndarray,
        equity: np.ndarray,
        trades: List[Dict],
    ) -> Dict[str, float]:
        """
        Compute performance metrics.

        Args:
            returns: Array of strategy returns.
            equity: Equity curve.
            trades: List of trade records.

        Returns:
            Dictionary of performance metrics.
        """
        # Remove zero returns (no position)
        active_returns = returns[returns != 0]

        # Annualization factor (assume hourly data)
        annual_factor = np.sqrt(365 * 24)

        # Sharpe Ratio
        if len(active_returns) > 0 and np.std(active_returns) > 0:
            sharpe = np.mean(active_returns) / np.std(active_returns) * annual_factor
        else:
            sharpe = 0.0

        # Sortino Ratio (downside deviation)
        downside_returns = active_returns[active_returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(active_returns) / np.std(downside_returns) * annual_factor
        else:
            sortino = sharpe

        # Maximum Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = np.min(drawdown)

        # Calmar Ratio
        total_return = (equity[-1] - equity[0]) / equity[0]
        if max_drawdown != 0:
            calmar = total_return / abs(max_drawdown)
        else:
            calmar = 0.0

        # Win Rate
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        win_rate = len(winning_trades) / max(len(trades), 1)

        # Profit Factor
        gross_profit = sum(max(t.get("pnl", 0), 0) for t in trades)
        gross_loss = abs(sum(min(t.get("pnl", 0), 0) for t in trades))
        profit_factor = gross_profit / max(gross_loss, 1e-10)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": len(trades),
            "final_equity": equity[-1],
        }


def compute_shap_stability(
    shap_values_history: List[np.ndarray],
    window: int = 10,
) -> np.ndarray:
    """
    Compute SHAP stability score based on historical explanations.

    Stability is measured as the cosine similarity between consecutive
    SHAP value vectors. High stability suggests consistent model behavior.

    Args:
        shap_values_history: List of SHAP value arrays over time.
        window: Rolling window for stability calculation.

    Returns:
        Array of stability scores (0-1).
    """
    n = len(shap_values_history)
    stability = np.ones(n)

    for i in range(window, n):
        # Compute average cosine similarity over window
        similarities = []
        for j in range(i - window + 1, i):
            v1 = shap_values_history[j]
            v2 = shap_values_history[j + 1]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(v1, v2) / (norm1 * norm2)
                similarities.append((sim + 1) / 2)  # Scale to 0-1

        stability[i] = np.mean(similarities) if similarities else 1.0

    return stability


def demo():
    """Run a demonstration of the backtesting framework."""
    from data_loader import BybitDataLoader
    from shap_model import TradingSHAPModel

    # Load data
    loader = BybitDataLoader()
    data = loader.fetch_klines("BTCUSDT", interval="60", limit=500)

    print(f"\nData source: {data.source}")
    print(f"Data shape: {data.df.shape}")

    # Train model
    model = TradingSHAPModel(n_estimators=50, max_depth=4)
    model.fit(data.df)

    # Generate signals
    features = model.compute_features(data.df)
    valid_idx = features[model.FEATURE_NAMES].dropna().index

    prices = data.df.loc[valid_idx, "close"].values
    signals = model.predict(data.df)

    # Align lengths
    min_len = min(len(prices), len(signals))
    prices = prices[-min_len:]
    signals = signals[-min_len:]

    # Run backtest without SHAP filter
    config_no_shap = BacktestConfig(use_shap_filter=False)
    engine_no_shap = BacktestEngine(config_no_shap)
    result_no_shap = engine_no_shap.run(prices, signals)

    print("\n" + "=" * 50)
    print("Backtest Results (No SHAP Filter)")
    print("=" * 50)
    for k, v in result_no_shap.metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Generate SHAP stability scores
    # (In practice, compute from actual SHAP values over time)
    np.random.seed(42)
    shap_stability = np.clip(
        0.7 + 0.2 * np.random.randn(len(signals)),
        0.1, 1.0
    )

    # Run backtest with SHAP filter
    config_shap = BacktestConfig(use_shap_filter=True, shap_stability_threshold=0.5)
    engine_shap = BacktestEngine(config_shap)
    result_shap = engine_shap.run(prices, signals, shap_stability)

    print("\n" + "=" * 50)
    print("Backtest Results (With SHAP Filter)")
    print("=" * 50)
    for k, v in result_shap.metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Compare results
    print("\n" + "=" * 50)
    print("Comparison")
    print("=" * 50)
    improvement = (
        (result_shap.metrics["sharpe_ratio"] - result_no_shap.metrics["sharpe_ratio"])
        / max(abs(result_no_shap.metrics["sharpe_ratio"]), 0.01)
        * 100
    )
    print(f"  Sharpe improvement: {improvement:.1f}%")


if __name__ == "__main__":
    demo()
