"""
Backtesting framework for meta-gradient optimization trading strategies.

Provides tools to evaluate trading performance using adaptive
hyperparameter optimization.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Optional

from meta_gradient_optimizer import (
    MetaGradientOptimizer,
    MetaGradientTradingModel,
    OnlineMetaGradientTrader,
)
from data_loader import create_trading_features, create_meta_gradient_tasks


class MetaGradientBacktester:
    """
    Backtesting framework for meta-gradient-based trading strategies.
    """

    def __init__(
        self,
        meta_optimizer: MetaGradientOptimizer,
        adaptation_window: int = 30,
        validation_window: int = 10,
        prediction_threshold: float = 0.001,
        transaction_cost: float = 0.001,
    ):
        self.meta_opt = meta_optimizer
        self.adaptation_window = adaptation_window
        self.validation_window = validation_window
        self.threshold = prediction_threshold
        self.transaction_cost = transaction_cost

    def backtest(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        initial_capital: float = 10000.0,
    ) -> pd.DataFrame:
        """Run backtest on historical data."""
        results = []
        capital = initial_capital
        position = 0

        feature_cols = list(features.columns)
        total_window = self.adaptation_window + self.validation_window

        for i in range(total_window, len(features) - 1):
            window_data = features.iloc[i - total_window:i]
            window_returns = prices.pct_change().iloc[i - total_window + 1:i + 1]

            train_f = torch.FloatTensor(
                window_data.iloc[:self.adaptation_window][feature_cols].values
            )
            train_t = torch.FloatTensor(
                window_returns.iloc[:self.adaptation_window].values
            ).unsqueeze(1)
            val_f = torch.FloatTensor(
                window_data.iloc[self.adaptation_window:][feature_cols].values
            )
            val_t = torch.FloatTensor(
                window_returns.iloc[self.adaptation_window:].values
            ).unsqueeze(1)

            # Meta-gradient update
            task = [((train_f, train_t), (val_f, val_t))]
            self.meta_opt.meta_train_step(task)

            # Predict
            current_features = torch.FloatTensor(
                features.iloc[i][feature_cols].values
            ).unsqueeze(0)

            prediction = self.meta_opt.adapt_and_predict(
                train_f, train_t, current_features
            ).item()

            # Trading logic
            new_position = 0
            if prediction > self.threshold:
                new_position = 1
            elif prediction < -self.threshold:
                new_position = -1

            if new_position != position:
                capital *= (1 - self.transaction_cost)

            actual_return = prices.iloc[i + 1] / prices.iloc[i] - 1
            position_return = position * actual_return
            capital *= (1 + position_return)

            results.append({
                'date': features.index[i],
                'price': prices.iloc[i],
                'prediction': prediction,
                'actual_return': actual_return,
                'position': position,
                'position_return': position_return,
                'capital': capital,
            })

            position = new_position

        return pd.DataFrame(results)


def calculate_metrics(results: pd.DataFrame) -> dict:
    """Calculate trading performance metrics."""
    returns = results['position_return']

    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1

    # Annualized metrics
    ann_return = (1 + total_return) ** (252 / len(results)) - 1
    ann_volatility = returns.std() * np.sqrt(252)

    # Risk-adjusted metrics
    sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
    downside_returns = returns[returns < 0]
    sortino_ratio = np.sqrt(252) * returns.mean() / (downside_returns.std() + 1e-10)

    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Win rate and profit factor
    wins = (returns > 0).sum()
    losses_count = (returns < 0).sum()
    win_rate = wins / (wins + losses_count) if (wins + losses_count) > 0 else 0

    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / (gross_losses + 1e-10)

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': len(results[results['position'] != 0]),
    }


if __name__ == "__main__":
    print("Meta-Gradient Optimization Backtester")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=300, freq='D')
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.randn(300) * 0.015)),
        index=dates,
        name='Close',
    )

    features = create_trading_features(prices)

    # Create model and optimizer
    model = MetaGradientTradingModel(input_size=11, hidden_size=32, output_size=1)
    meta_opt = MetaGradientOptimizer(
        model=model,
        inner_lr_init=0.01,
        meta_lr=0.001,
        inner_steps=3,
        learn_lr=True,
        learn_loss=False,
    )

    # Run backtest
    backtester = MetaGradientBacktester(
        meta_optimizer=meta_opt,
        adaptation_window=30,
        validation_window=10,
        prediction_threshold=0.001,
        transaction_cost=0.001,
    )

    print("Running backtest...")
    results = backtester.backtest(prices, features)
    metrics = calculate_metrics(results)

    print("\nBacktest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
