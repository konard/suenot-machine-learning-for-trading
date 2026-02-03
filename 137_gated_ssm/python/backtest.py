"""
Backtesting framework for Gated SSM trading strategies.

Provides:
- GatedSSMBacktester: Signal generation and backtest execution
- walk_forward_backtest: Walk-forward validation with periodic retraining
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Callable
import logging

from model import GatedSSMModel
from data_loader import TradingDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GatedSSMBacktester:
    """Backtesting framework for Gated SSM trading strategy."""

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.55,
        transaction_cost: float = 0.001,
    ):
        self.model = model
        self.threshold = threshold
        self.transaction_cost = transaction_cost

    @torch.no_grad()
    def generate_signals(self, dataset: TradingDataset) -> np.ndarray:
        """Generate trading signals from model predictions."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        signals = []

        for features, _ in loader:
            logits = self.model(features).squeeze(-1)
            probs = torch.sigmoid(logits)
            signal = torch.where(
                probs > self.threshold,
                torch.tensor(1.0),
                torch.where(
                    probs < 1 - self.threshold,
                    torch.tensor(-1.0),
                    torch.tensor(0.0),
                ),
            )
            signals.append(signal.numpy())

        return np.concatenate(signals)

    def run(self, prices: pd.Series, signals: np.ndarray) -> Dict[str, float]:
        """
        Run backtest and compute performance metrics.

        Args:
            prices: Close price series
            signals: Array of trading signals (-1, 0, 1)

        Returns:
            Dictionary of performance metrics
        """
        returns = prices.pct_change().dropna().values
        # Align signals to last N returns
        n = min(len(signals), len(returns))
        returns = returns[-n:]
        signals = signals[-n:]

        # Strategy returns with transaction costs
        positions = signals
        position_changes = np.diff(positions, prepend=0)
        costs = np.abs(position_changes) * self.transaction_cost
        strategy_returns = positions * returns - costs

        # Total return
        total_return = float(np.exp(np.sum(np.log1p(strategy_returns))) - 1)

        # Annualized return
        n_periods = len(strategy_returns)
        if n_periods > 0 and total_return > -1:
            annual_return = float((1 + total_return) ** (252 / n_periods) - 1)
        else:
            annual_return = 0.0

        # Annualized volatility
        annual_vol = float(np.std(strategy_returns) * np.sqrt(252))

        # Sharpe ratio
        sharpe = annual_return / (annual_vol + 1e-10)

        # Maximum drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        max_drawdown = float(np.min(drawdown))

        # Sortino ratio
        downside = strategy_returns[strategy_returns < 0]
        downside_vol = (
            float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else 1e-10
        )
        sortino = annual_return / (downside_vol + 1e-10)

        # Win rate
        active_returns = strategy_returns[strategy_returns != 0]
        wins = np.sum(active_returns > 0)
        win_rate = float(wins / (len(active_returns) + 1e-10))

        # Number of trades
        num_trades = int(np.sum(np.abs(position_changes) > 0))

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": num_trades,
        }


def train_model(
    model: nn.Module,
    dataset: TradingDataset,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> nn.Module:
    """Train the Gated SSM model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for features, targets in loader:
            pred = model(features).squeeze(-1)
            loss = criterion(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


def walk_forward_backtest(
    data: pd.DataFrame,
    model_fn: Callable,
    window: int = 60,
    train_period: int = 500,
    test_period: int = 60,
    retrain_every: int = 60,
) -> pd.DataFrame:
    """
    Walk-forward backtesting with periodic model retraining.

    Args:
        data: Price DataFrame with 'close' and 'volume' columns
        model_fn: Callable that returns a fresh GatedSSMModel
        window: Lookback window for features
        train_period: Number of samples for training
        test_period: Number of samples for testing
        retrain_every: Retrain frequency in samples

    Returns:
        DataFrame with per-period performance metrics
    """
    results = []
    total_len = len(data)
    start = window + 20  # After feature warm-up

    for i in range(start + train_period, total_len - test_period, retrain_every):
        # Train window
        train_data = data.iloc[i - train_period - start : i]
        train_dataset = TradingDataset(train_data, window=window)

        if len(train_dataset) < 10:
            continue

        # Test window
        test_data = data.iloc[i - start : i + test_period]
        test_dataset = TradingDataset(test_data, window=window)

        if len(test_dataset) < 1:
            continue

        # Train model
        model = model_fn()
        model = train_model(model, train_dataset, epochs=30)

        # Evaluate
        backtester = GatedSSMBacktester(model)
        signals = backtester.generate_signals(test_dataset)
        metrics = backtester.run(test_data["close"], signals)
        metrics["period_start"] = data.index[i] if hasattr(data.index, "__getitem__") else i
        results.append(metrics)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 800
    dates = pd.date_range("2020-01-01", periods=n)
    prices = pd.DataFrame(
        {
            "close": 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=dates,
    )

    # Split data
    train_data = prices.iloc[:600]
    test_data = prices.iloc[500:]

    # Create datasets
    train_dataset = TradingDataset(train_data, window=60)
    test_dataset = TradingDataset(test_data, window=60)

    # Build model
    model = GatedSSMModel(
        input_size=4,
        d_model=64,
        d_state=32,
        n_layers=3,
        expand=2,
        dropout=0.1,
    )

    # Train
    logger.info("Training Gated SSM model...")
    model = train_model(model, train_dataset, epochs=30)

    # Backtest
    logger.info("Running backtest...")
    backtester = GatedSSMBacktester(model, threshold=0.55)
    signals = backtester.generate_signals(test_dataset)
    metrics = backtester.run(test_data["close"], signals)

    logger.info("Backtest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
