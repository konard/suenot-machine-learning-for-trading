"""
Backtesting framework for multi-task learning trading strategies.

Provides:
- MTLBacktester: Backtest engine using multi-task model predictions
- calculate_metrics: Performance metric computation
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

from .mtl_model import MultiTaskTradingModel

logger = logging.getLogger(__name__)


class MTLBacktester:
    """
    Backtesting framework for multi-task learning trading strategies.

    Uses both return prediction and direction confidence from the MTL model
    to generate trading signals, with configurable thresholds and costs.
    """

    def __init__(
        self,
        model: MultiTaskTradingModel,
        prediction_threshold: float = 0.001,
        transaction_cost: float = 0.001,
    ):
        """
        Args:
            model: Trained MultiTaskTradingModel
            prediction_threshold: Minimum signal magnitude to trade
            transaction_cost: Cost per trade as a fraction
        """
        self.model = model
        self.threshold = prediction_threshold
        self.transaction_cost = transaction_cost

    def backtest(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        initial_capital: float = 10000.0,
    ) -> pd.DataFrame:
        """
        Run backtest on historical data.

        Args:
            prices: Price series aligned with features
            features: Feature DataFrame
            initial_capital: Starting capital

        Returns:
            DataFrame with columns: date, price, signal, position,
            position_return, capital
        """
        self.model.eval()
        results = []
        capital = initial_capital
        position = 0

        feature_tensor = torch.FloatTensor(features.values)

        with torch.no_grad():
            predictions = self.model(feature_tensor)

        pred_return = predictions['return'].numpy().flatten()
        pred_direction = predictions['direction'].numpy().flatten()

        for i in range(len(features) - 1):
            # Combine return prediction with direction confidence
            signal = pred_return[i] * (2 * pred_direction[i] - 1)

            if signal > self.threshold:
                new_position = 1
            elif signal < -self.threshold:
                new_position = -1
            else:
                new_position = 0

            # Apply transaction cost on position change
            if new_position != position:
                capital *= (1 - self.transaction_cost)

            # Calculate realized return
            actual_return = prices.iloc[i + 1] / prices.iloc[i] - 1
            position_return = position * actual_return
            capital *= (1 + position_return)

            results.append({
                'date': features.index[i],
                'price': prices.iloc[i],
                'signal': signal,
                'position': position,
                'position_return': position_return,
                'capital': capital,
            })

            position = new_position

        return pd.DataFrame(results)


def calculate_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate trading performance metrics.

    Args:
        results: DataFrame from MTLBacktester.backtest()

    Returns:
        Dict with performance metrics
    """
    returns = results['position_return']

    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1
    ann_return = (1 + total_return) ** (252 / max(len(results), 1)) - 1
    ann_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)

    downside = returns[returns < 0]
    sortino_ratio = np.sqrt(252) * returns.mean() / (downside.std() + 1e-10)

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    max_drawdown = (cumulative / rolling_max - 1).min()

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
        'num_trades': int((results['position'].diff().abs() > 0).sum()),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo with synthetic data
    np.random.seed(42)
    n = 300
    dates = pd.date_range('2023-01-01', periods=n, freq='h')
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.randn(n) * 0.005)),
        index=dates,
    )

    from .data_loader import create_mtl_features, create_mtl_targets

    features = create_mtl_features(prices)
    targets = create_mtl_targets(prices)
    aligned = features.join(targets, how='inner').dropna()
    feat_df = aligned[features.columns]
    prices_aligned = prices.loc[feat_df.index]

    model = MultiTaskTradingModel(
        input_size=feat_df.shape[1],
        shared_hidden=[64, 32],
    )

    backtester = MTLBacktester(model, prediction_threshold=0.0005)
    results = backtester.backtest(prices_aligned, feat_df)
    metrics = calculate_metrics(results)

    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
