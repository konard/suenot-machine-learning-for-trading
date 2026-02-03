"""
Backtesting framework with DeepLIFT explanations.

This module provides:
- DeepLIFTBacktester: Backtesting with explanation logging
- calculate_metrics: Performance metric calculation
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from .deeplift_trader import DeepLIFT, Attribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    results_df: pd.DataFrame
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]


class DeepLIFTBacktester:
    """
    Backtesting framework with DeepLIFT explanations.

    Logs which features contributed to each trading decision,
    enabling analysis of strategy behavior over time.
    """

    def __init__(
        self,
        model: nn.Module,
        explainer: DeepLIFT,
        feature_names: List[str],
        prediction_threshold: float = 0.001,
        transaction_cost: float = 0.001
    ):
        """
        Initialize backtester.

        Args:
            model: Trained trading model
            explainer: DeepLIFT explainer instance
            feature_names: Names of input features
            prediction_threshold: Threshold for trading signals
            transaction_cost: Cost per transaction (as fraction)
        """
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
        self.threshold = prediction_threshold
        self.transaction_cost = transaction_cost

    def backtest(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_capital: float = 10000.0,
        log_explanations: bool = True
    ) -> BacktestResult:
        """
        Run backtest with explanation logging.

        Args:
            prices: Price array (should align with features)
            features: Feature array
            initial_capital: Starting capital
            log_explanations: Whether to compute and log explanations

        Returns:
            BacktestResult with DataFrame, metrics, and feature importance
        """
        results = []
        capital = initial_capital
        position = 0  # -1 (short), 0 (neutral), 1 (long)

        # Track feature importance over time
        importance_sum = np.zeros(len(self.feature_names))
        n_attributions = 0

        self.model.eval()

        # Ensure prices and features are aligned
        n = min(len(prices), len(features))

        for i in range(n):
            input_tensor = torch.FloatTensor(features[i:i+1])

            with torch.no_grad():
                prediction = self.model(input_tensor).item()

            # Get explanation if requested
            top_features = []
            if log_explanations:
                attribution = self.explainer.attribute(input_tensor, self.feature_names)
                top_features = attribution.top_features(3)
                importance_sum += np.abs(attribution.scores)
                n_attributions += 1

            # Trading logic
            if prediction > self.threshold:
                new_position = 1  # Long
            elif prediction < -self.threshold:
                new_position = -1  # Short
            else:
                new_position = 0  # Neutral

            # Transaction costs
            if new_position != position and i > 0:
                capital *= (1 - self.transaction_cost)

            # Calculate returns
            if i < n - 1:
                actual_return = prices[i+1] / prices[i] - 1
                position_return = position * actual_return
                capital *= (1 + position_return)
            else:
                actual_return = 0
                position_return = 0

            result = {
                'index': i,
                'price': prices[i],
                'prediction': prediction,
                'position': position,
                'new_position': new_position,
                'actual_return': actual_return,
                'position_return': position_return,
                'capital': capital,
            }

            # Add top feature information
            for j, (feat_name, feat_score) in enumerate(top_features[:3]):
                result[f'top_feature_{j+1}'] = feat_name
                result[f'top_score_{j+1}'] = feat_score

            results.append(result)
            position = new_position

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Calculate metrics
        metrics = calculate_metrics(results_df)

        # Average feature importance
        feature_importance = {}
        if n_attributions > 0:
            avg_importance = importance_sum / n_attributions
            feature_importance = dict(zip(self.feature_names, avg_importance))

        return BacktestResult(
            results_df=results_df,
            metrics=metrics,
            feature_importance=feature_importance
        )

    def analyze_regime_attributions(
        self,
        results: BacktestResult,
        regime_column: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze feature importance by market regime.

        Args:
            results: BacktestResult from backtest
            regime_column: Column indicating regime (if None, uses returns sign)

        Returns:
            Dictionary mapping regime to feature importance
        """
        df = results.results_df.copy()

        # Create regime if not provided
        if regime_column is None:
            # Use rolling return direction as regime
            df['regime'] = np.where(
                df['actual_return'].rolling(10).mean() > 0,
                'bullish',
                'bearish'
            )
        else:
            df['regime'] = df[regime_column]

        regime_importance = {}

        for regime in df['regime'].dropna().unique():
            regime_df = df[df['regime'] == regime]
            regime_importance[regime] = {}

            # Collect scores for this regime
            for feat in self.feature_names:
                scores = []
                for j in range(1, 4):
                    mask = regime_df[f'top_feature_{j}'] == feat
                    if mask.any():
                        scores.extend(regime_df.loc[mask, f'top_score_{j}'].tolist())

                if scores:
                    regime_importance[regime][feat] = np.mean(np.abs(scores))
                else:
                    regime_importance[regime][feat] = 0.0

        return regime_importance


def calculate_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate trading performance metrics.

    Args:
        results: DataFrame with backtest results

    Returns:
        Dictionary of performance metrics
    """
    returns = results['position_return']

    # Handle edge cases
    if len(returns) == 0 or returns.std() == 0:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'annualized_volatility': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'num_trades': 0,
        }

    # Basic metrics
    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1

    # Annualized metrics (assuming hourly data, ~8760 hours/year)
    periods_per_year = 8760
    ann_return = (1 + total_return) ** (periods_per_year / len(results)) - 1
    ann_volatility = returns.std() * np.sqrt(periods_per_year)

    # Risk-adjusted metrics
    sharpe_ratio = np.sqrt(periods_per_year) * returns.mean() / (returns.std() + 1e-10)

    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        sortino_ratio = np.sqrt(periods_per_year) * returns.mean() / (downside_returns.std() + 1e-10)
    else:
        sortino_ratio = np.inf

    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Win rate
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    # Profit factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / (gross_losses + 1e-10)

    # Number of trades (position changes)
    num_trades = (results['position'] != results['position'].shift()).sum()

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': num_trades,
    }


def print_backtest_report(result: BacktestResult):
    """Print a formatted backtest report."""
    print("\n" + "=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)

    print("\nPerformance Metrics:")
    print("-" * 40)
    metrics = result.metrics
    print(f"  Total Return:        {metrics['total_return']*100:>10.2f}%")
    print(f"  Annualized Return:   {metrics['annualized_return']*100:>10.2f}%")
    print(f"  Annualized Vol:      {metrics['annualized_volatility']*100:>10.2f}%")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.3f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.3f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']*100:>10.2f}%")
    print(f"  Win Rate:            {metrics['win_rate']*100:>10.2f}%")
    print(f"  Profit Factor:       {metrics['profit_factor']:>10.3f}")
    print(f"  Number of Trades:    {metrics['num_trades']:>10.0f}")

    if result.feature_importance:
        print("\nFeature Importance (Average):")
        print("-" * 40)
        sorted_importance = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for name, score in sorted_importance:
            print(f"  {name:<20} {score:.6f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    import torch.nn as nn
    from .deeplift_trader import TradingModelWithDeepLIFT, DeepLIFT
    from .data_loader import SimulatedDataGenerator, FeatureGenerator

    print("Running backtest example...")

    # Generate simulated data
    klines = SimulatedDataGenerator.generate_regime_changing_klines(500)
    feature_gen = FeatureGenerator(window=20)
    features = feature_gen.compute_features(klines)
    prices = np.array([k.close for k in klines])[-len(features):]

    feature_names = [
        "returns_1d", "returns_5d", "returns_10d", "sma_ratio", "ema_ratio",
        "volatility", "momentum", "rsi", "macd", "bb_position", "volume_sma_ratio"
    ]

    # Create and train a simple model
    model = TradingModelWithDeepLIFT(input_size=11, hidden_size=32)

    # Quick training
    X = torch.FloatTensor(features[:-5])
    y = torch.FloatTensor(
        (prices[5:] - prices[:-5]) / prices[:-5]
    )[-len(X):].unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for _ in range(50):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    # Create explainer and backtester
    reference = torch.FloatTensor(np.mean(features, axis=0, keepdims=True))
    explainer = DeepLIFT(model, reference=reference)
    backtester = DeepLIFTBacktester(
        model=model,
        explainer=explainer,
        feature_names=feature_names,
        prediction_threshold=0.001
    )

    # Run backtest
    result = backtester.backtest(prices, features)

    # Print report
    print_backtest_report(result)
