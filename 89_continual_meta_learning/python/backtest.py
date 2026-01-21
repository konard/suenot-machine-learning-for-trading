"""
Backtesting Framework for Continual Meta-Learning Trading

This module provides a backtesting framework specifically designed for
continual meta-learning trading strategies.

Features:
- Online adaptation during backtesting
- Regime-aware performance tracking
- Forgetting evaluation
- Risk-adjusted metrics
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from continual_meta_learner import ContinualMetaLearner, TradingModel
from data_loader import create_trading_features, detect_market_regime


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    adaptation_window: int = 20
    adaptation_steps: int = 5
    prediction_threshold: float = 0.001
    retraining_frequency: int = 20
    transaction_cost: float = 0.001  # 0.1% per trade
    max_position: float = 1.0  # Maximum position size


@dataclass
class TradeResult:
    """Result of a single trade."""
    date: pd.Timestamp
    price: float
    prediction: float
    actual_return: float
    position: int
    position_return: float
    capital: float
    regime: str


class CMLBacktester:
    """
    Backtesting framework for Continual Meta-Learning strategies.

    This backtester:
    1. Adapts the model to recent data at each step
    2. Makes predictions for the next period
    3. Periodically retrains with continual learning
    4. Tracks performance by regime
    """

    def __init__(
        self,
        cml: ContinualMetaLearner,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtester.

        Args:
            cml: ContinualMetaLearner instance
            config: Backtest configuration
        """
        self.cml = cml
        self.config = config or BacktestConfig()
        self.results: List[TradeResult] = []

    def backtest(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Run backtest with continual meta-learning.

        The model adapts quickly to new data while retaining
        knowledge of past market regimes.

        Args:
            prices: Price series
            features: Feature DataFrame
            regimes: Regime labels

        Returns:
            DataFrame with backtest results
        """
        self.results = []
        capital = self.config.initial_capital
        position = 0

        feature_cols = list(features.columns)

        for i in range(self.config.adaptation_window, len(features) - 1):
            # Get adaptation data
            adapt_start = i - self.config.adaptation_window
            adapt_features = torch.FloatTensor(
                features.iloc[adapt_start:i][feature_cols].values
            )
            adapt_returns = torch.FloatTensor(
                prices.pct_change().iloc[adapt_start + 1:i + 1].values
            ).unsqueeze(1)

            # Adapt model to recent data
            adapted = self.cml.adapt(
                (adapt_features[:-1], adapt_returns[:-1]),
                adaptation_steps=self.config.adaptation_steps
            )

            # Make prediction
            current_features = torch.FloatTensor(
                features.iloc[i][feature_cols].values
            ).unsqueeze(0)

            with torch.no_grad():
                prediction = adapted(current_features).item()

            # Trading logic
            if prediction > self.config.prediction_threshold:
                new_position = 1  # Long
            elif prediction < -self.config.prediction_threshold:
                new_position = -1  # Short
            else:
                new_position = 0  # Neutral

            # Calculate returns
            actual_return = prices.iloc[i + 1] / prices.iloc[i] - 1
            position_return = position * actual_return

            # Apply transaction costs if position changed
            if new_position != position:
                position_return -= self.config.transaction_cost * abs(new_position - position)

            capital *= (1 + position_return)

            # Get regime
            regime = regimes.iloc[i] if not pd.isna(regimes.iloc[i]) else 'unknown'

            # Periodic retraining with continual learning
            if i % self.config.retraining_frequency == 0:
                support = (adapt_features[:-1], adapt_returns[:-1])
                query = (adapt_features[-5:], adapt_returns[-5:])
                self.cml.meta_train_step((support, query), regime=regime)

            # Store result
            self.results.append(TradeResult(
                date=features.index[i],
                price=prices.iloc[i],
                prediction=prediction,
                actual_return=actual_return,
                position=position,
                position_return=position_return,
                capital=capital,
                regime=regime
            ))

            position = new_position

        return self._create_results_dataframe()

    def _create_results_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([
            {
                'date': r.date,
                'price': r.price,
                'prediction': r.prediction,
                'actual_return': r.actual_return,
                'position': r.position,
                'position_return': r.position_return,
                'capital': r.capital,
                'regime': r.regime
            }
            for r in self.results
        ])


def calculate_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive trading performance metrics.

    Args:
        results: Backtest results DataFrame

    Returns:
        Dictionary with performance metrics
    """
    returns = results['position_return']

    # Basic metrics
    total_return = (results['capital'].iloc[-1] / results['capital'].iloc[0]) - 1

    # Annualized return (assuming daily data)
    n_periods = len(returns)
    annualized_return = (1 + total_return) ** (252 / n_periods) - 1

    # Risk-adjusted metrics
    std = returns.std()
    sharpe_ratio = np.sqrt(252) * returns.mean() / (std + 1e-8)

    downside_std = returns[returns < 0].std()
    sortino_ratio = np.sqrt(252) * returns.mean() / (downside_std + 1e-8)

    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Calculate average drawdown duration
    in_drawdown = drawdowns < 0
    drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
    drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)

    # Trade statistics
    trades = results[results['position'] != 0]
    n_trades = len(trades)
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    # Average win/loss
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'num_trades': n_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'volatility': std * np.sqrt(252)
    }


def calculate_regime_metrics(results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance metrics by market regime.

    Args:
        results: Backtest results DataFrame

    Returns:
        Dictionary with metrics per regime
    """
    regime_metrics = {}

    for regime in results['regime'].unique():
        regime_data = results[results['regime'] == regime]
        if len(regime_data) > 5:  # Need minimum data
            regime_returns = regime_data['position_return']

            regime_metrics[regime] = {
                'return': regime_returns.sum(),
                'sharpe': np.sqrt(252) * regime_returns.mean() / (regime_returns.std() + 1e-8),
                'win_rate': (regime_returns > 0).mean(),
                'num_periods': len(regime_data)
            }

    return regime_metrics


def evaluate_forgetting(
    cml: ContinualMetaLearner,
    test_tasks_by_regime: Dict[str, Tuple],
    adaptation_steps: int = 5
) -> Dict[str, float]:
    """
    Evaluate forgetting by testing on held-out tasks from each regime.

    Args:
        cml: ContinualMetaLearner instance
        test_tasks_by_regime: Dict of {regime: (support_data, query_data)}
        adaptation_steps: Steps for adaptation

    Returns:
        Dictionary with forgetting metrics per regime
    """
    results = {}

    for regime, (support, query) in test_tasks_by_regime.items():
        adapted = cml.adapt(support, adaptation_steps=adaptation_steps)

        with torch.no_grad():
            features, labels = query
            predictions = adapted(features)
            loss = torch.nn.MSELoss()(predictions, labels).item()

        results[regime] = loss

    # Overall forgetting score (higher is worse)
    forgetting_score = np.std(list(results.values()))

    results['forgetting_score'] = forgetting_score
    results['mean_loss'] = np.mean(list(results.values())[:-1])

    return results


def compare_strategies(
    results_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compare multiple trading strategies.

    Args:
        results_dict: Dict of {strategy_name: results_dataframe}

    Returns:
        DataFrame comparing strategy metrics
    """
    comparison = []

    for name, results in results_dict.items():
        metrics = calculate_metrics(results)
        metrics['strategy'] = name
        comparison.append(metrics)

    return pd.DataFrame(comparison).set_index('strategy')


def print_backtest_report(
    results: pd.DataFrame,
    strategy_name: str = "CML Strategy"
):
    """
    Print a comprehensive backtest report.

    Args:
        results: Backtest results DataFrame
        strategy_name: Name of the strategy
    """
    metrics = calculate_metrics(results)
    regime_metrics = calculate_regime_metrics(results)

    print(f"\n{'='*60}")
    print(f"BACKTEST REPORT: {strategy_name}")
    print(f"{'='*60}")

    print(f"\nPERFORMANCE SUMMARY")
    print(f"-" * 40)
    print(f"Total Return:      {metrics['total_return']*100:>10.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:>10.2f}%")
    print(f"Volatility (Ann.): {metrics['volatility']*100:>10.2f}%")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
    print(f"Sortino Ratio:     {metrics['sortino_ratio']:>10.2f}")
    print(f"Calmar Ratio:      {metrics['calmar_ratio']:>10.2f}")
    print(f"Max Drawdown:      {metrics['max_drawdown']*100:>10.2f}%")

    print(f"\nTRADE STATISTICS")
    print(f"-" * 40)
    print(f"Number of Trades:  {metrics['num_trades']:>10}")
    print(f"Win Rate:          {metrics['win_rate']*100:>10.2f}%")
    print(f"Profit Factor:     {metrics['profit_factor']:>10.2f}")
    print(f"Average Win:       {metrics['avg_win']*100:>10.4f}%")
    print(f"Average Loss:      {metrics['avg_loss']*100:>10.4f}%")

    print(f"\nREGIME PERFORMANCE")
    print(f"-" * 40)
    for regime, rmetrics in regime_metrics.items():
        print(f"\n  {regime.upper()}")
        print(f"    Periods: {rmetrics['num_periods']}")
        print(f"    Return:  {rmetrics['return']*100:.2f}%")
        print(f"    Sharpe:  {rmetrics['sharpe']:.2f}")
        print(f"    Win %:   {rmetrics['win_rate']*100:.1f}%")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    print("CML Backtesting Framework")
    print("=" * 50)

    # Create model and CML
    model = TradingModel(input_size=8, hidden_size=64, output_size=1)
    cml = ContinualMetaLearner(
        model=model,
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=5,
        memory_size=100,
        ewc_lambda=0.4
    )

    # Generate synthetic data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')

    # Simulate price series with regime changes
    prices_list = [100]
    regimes_list = []

    for i in range(1, 500):
        # Determine regime
        if i < 100:
            regime = 'bull'
            drift = 0.001
            vol = 0.02
        elif i < 200:
            regime = 'high_vol'
            drift = 0.0
            vol = 0.05
        elif i < 300:
            regime = 'bear'
            drift = -0.001
            vol = 0.025
        elif i < 400:
            regime = 'low_vol'
            drift = 0.0002
            vol = 0.008
        else:
            regime = 'bull'
            drift = 0.0008
            vol = 0.015

        ret = drift + vol * np.random.randn()
        prices_list.append(prices_list[-1] * (1 + ret))
        regimes_list.append(regime)

    prices = pd.Series(prices_list, index=dates, name='close')
    regimes = pd.Series(['unknown'] + regimes_list, index=dates)

    # Create features
    features = create_trading_features(prices)

    # Align data
    common_idx = features.index.intersection(regimes.index)
    features = features.loc[common_idx]
    prices = prices.loc[common_idx]
    regimes = regimes.loc[common_idx]

    print(f"\nData summary:")
    print(f"  Periods: {len(prices)}")
    print(f"  Features: {list(features.columns)}")
    print(f"  Regimes: {regimes.value_counts().to_dict()}")

    # Run backtest
    print("\nRunning backtest...")
    config = BacktestConfig(
        initial_capital=10000,
        adaptation_window=20,
        adaptation_steps=5,
        prediction_threshold=0.001,
        retraining_frequency=20,
        transaction_cost=0.001
    )

    backtester = CMLBacktester(cml, config)
    results = backtester.backtest(prices, features, regimes)

    # Print report
    print_backtest_report(results, "Continual Meta-Learning Strategy")

    # Show final capital
    print(f"Final Capital: ${results['capital'].iloc[-1]:.2f}")
    print(f"(Started with: ${config.initial_capital:.2f})")

    print("\nDone!")
