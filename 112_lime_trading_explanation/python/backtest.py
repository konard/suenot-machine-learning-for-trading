"""
Backtesting Framework with LIME Explainability

Provides backtesting functionality that integrates LIME explanations
for evaluating and filtering trading signals.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .lime_explainer import LimeExplainer, LimeExplanation


@dataclass
class Trade:
    """
    Represents a single trade.

    Attributes:
        entry_date: Date/time of entry
        exit_date: Date/time of exit
        direction: 1 for long, -1 for short
        entry_price: Entry price
        exit_price: Exit price
        return_pct: Return in percentage
        explanation: LIME explanation for the trade signal
    """
    entry_date: datetime
    exit_date: Optional[datetime] = None
    direction: int = 1
    entry_price: float = 0.0
    exit_price: float = 0.0
    return_pct: float = 0.0
    explanation: Optional[LimeExplanation] = None

    def close(self, exit_date: datetime, exit_price: float):
        """Close the trade and calculate return."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.return_pct = self.direction * (exit_price / self.entry_price - 1) * 100


@dataclass
class BacktestResult:
    """
    Results from a backtest run.

    Attributes:
        trades: List of executed trades
        equity_curve: Series of portfolio value over time
        metrics: Dictionary of performance metrics
        explanations: List of all LIME explanations
    """
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    metrics: Dict[str, float] = field(default_factory=dict)
    explanations: List[LimeExplanation] = field(default_factory=list)

    def get_winning_trades(self) -> List[Trade]:
        """Get all winning trades."""
        return [t for t in self.trades if t.return_pct > 0]

    def get_losing_trades(self) -> List[Trade]:
        """Get all losing trades."""
        return [t for t in self.trades if t.return_pct <= 0]


class Backtester:
    """
    Backtesting engine with LIME explainability support.

    Features:
    - Simple long-only or long-short strategies
    - LIME-based signal filtering
    - Performance metrics calculation
    - Trade logging with explanations
    """

    def __init__(
        self,
        model,
        explainer: Optional[LimeExplainer] = None,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        use_lime_filter: bool = False,
        min_lime_score: float = 0.5,
        position_size: float = 1.0
    ):
        """
        Initialize the backtester.

        Args:
            model: Trading model with predict and predict_proba methods
            explainer: LimeExplainer instance (optional)
            initial_capital: Starting capital
            transaction_cost: Cost per trade (as fraction of trade value)
            use_lime_filter: Whether to filter trades based on LIME explanations
            min_lime_score: Minimum local model R² to accept a trade
            position_size: Fraction of capital to use per trade
        """
        self.model = model
        self.explainer = explainer
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.use_lime_filter = use_lime_filter
        self.min_lime_score = min_lime_score
        self.position_size = position_size

    def _should_trade(
        self,
        prediction: int,
        prediction_proba: float,
        explanation: Optional[LimeExplanation] = None,
        min_proba: float = 0.55
    ) -> bool:
        """
        Determine whether to execute a trade.

        Args:
            prediction: Model prediction (0 or 1)
            prediction_proba: Prediction probability
            explanation: LIME explanation (if available)
            min_proba: Minimum probability to trade

        Returns:
            True if trade should be executed
        """
        if prediction_proba < min_proba:
            return False

        if self.use_lime_filter and explanation is not None:
            if explanation.local_model_score < self.min_lime_score:
                return False

        return True

    def run(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        holding_period: int = 1,
        min_proba: float = 0.55
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            prices: DataFrame with OHLCV data (must have 'close' column)
            features: DataFrame with feature values (aligned with prices)
            holding_period: Number of periods to hold each position
            min_proba: Minimum probability to enter a trade

        Returns:
            BacktestResult with trades, metrics, and explanations
        """
        aligned_prices = prices.loc[features.index]

        equity = self.initial_capital
        equity_history = []
        trades = []
        explanations = []

        position = None
        position_entry_idx = None

        for i in range(len(features) - holding_period):
            current_date = features.index[i]
            current_features = features.iloc[i:i+1]
            current_price = aligned_prices.iloc[i]['close']

            if position is not None:
                if i - position_entry_idx >= holding_period:
                    exit_price = current_price
                    trade_return = position.direction * (exit_price / position.entry_price - 1)
                    trade_return -= self.transaction_cost

                    position.close(current_date, exit_price)
                    position.return_pct = trade_return * 100
                    trades.append(position)

                    equity *= (1 + trade_return * self.position_size)
                    position = None

            if position is None:
                prediction = self.model.predict(current_features)[0]
                proba = self.model.predict_proba(current_features)[0]
                prediction_proba = proba[1]

                explanation = None
                if self.explainer is not None:
                    explanation = self.explainer.explain(current_features.values[0])
                    explanations.append(explanation)

                if self._should_trade(prediction, prediction_proba, explanation, min_proba):
                    direction = 1 if prediction == 1 else -1

                    position = Trade(
                        entry_date=current_date,
                        direction=direction,
                        entry_price=current_price,
                        explanation=explanation
                    )
                    position_entry_idx = i

                    equity *= (1 - self.transaction_cost * self.position_size)

            equity_history.append({'date': current_date, 'equity': equity})

        if position is not None:
            exit_date = features.index[-1]
            exit_price = aligned_prices.iloc[-1]['close']
            position.close(exit_date, exit_price)
            trades.append(position)

        equity_curve = pd.DataFrame(equity_history).set_index('date')['equity']

        metrics = self._calculate_metrics(trades, equity_curve)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            explanations=explanations
        )

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            trades: List of completed trades
            equity_curve: Series of equity values

        Returns:
            Dictionary of metrics
        """
        if len(trades) == 0:
            return {
                'total_return': 0,
                'num_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }

        returns = [t.return_pct / 100 for t in trades]
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r <= 0]

        total_return = (equity_curve.iloc[-1] / self.initial_capital - 1) * 100

        win_rate = len(winning_trades) / len(trades) if trades else 0

        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        profit_factor = avg_win * len(winning_trades) / (avg_loss * len(losing_trades)) \
            if losing_trades and avg_loss > 0 else float('inf')

        daily_returns = equity_curve.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100

        return {
            'total_return': total_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win * 100,
            'avg_loss': avg_loss * 100,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_return_per_trade': np.mean(returns) * 100
        }


def compare_with_without_lime(
    model,
    explainer: LimeExplainer,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    **backtest_kwargs
) -> Tuple[BacktestResult, BacktestResult]:
    """
    Compare backtest results with and without LIME filtering.

    Args:
        model: Trading model
        explainer: LimeExplainer instance
        prices: Price data
        features: Feature data
        **backtest_kwargs: Additional arguments for Backtester

    Returns:
        Tuple of (results_without_lime, results_with_lime)
    """
    bt_without = Backtester(
        model=model,
        explainer=explainer,
        use_lime_filter=False,
        **backtest_kwargs
    )
    results_without = bt_without.run(prices, features)

    bt_with = Backtester(
        model=model,
        explainer=explainer,
        use_lime_filter=True,
        **backtest_kwargs
    )
    results_with = bt_with.run(prices, features)

    return results_without, results_with


def print_backtest_comparison(
    results_without: BacktestResult,
    results_with: BacktestResult
):
    """Print a comparison of backtest results."""
    print("=" * 60)
    print("BACKTEST COMPARISON: Without vs With LIME Filtering")
    print("=" * 60)
    print()

    metrics = ['total_return', 'num_trades', 'win_rate', 'sharpe_ratio', 'max_drawdown']
    labels = ['Total Return (%)', 'Number of Trades', 'Win Rate', 'Sharpe Ratio', 'Max Drawdown (%)']

    print(f"{'Metric':<25} {'Without LIME':>15} {'With LIME':>15} {'Improvement':>15}")
    print("-" * 70)

    for metric, label in zip(metrics, labels):
        val_without = results_without.metrics.get(metric, 0)
        val_with = results_with.metrics.get(metric, 0)

        if metric == 'win_rate':
            fmt = '{:.1%}'
            val_without_fmt = fmt.format(val_without)
            val_with_fmt = fmt.format(val_with)
            improvement = (val_with - val_without) * 100
            imp_fmt = f'{improvement:+.1f}pp'
        elif metric == 'num_trades':
            val_without_fmt = f'{val_without:,.0f}'
            val_with_fmt = f'{val_with:,.0f}'
            improvement = val_with - val_without
            imp_fmt = f'{improvement:+,.0f}'
        else:
            val_without_fmt = f'{val_without:.2f}'
            val_with_fmt = f'{val_with:.2f}'
            if val_without != 0:
                improvement = (val_with / val_without - 1) * 100
                imp_fmt = f'{improvement:+.1f}%'
            else:
                imp_fmt = 'N/A'

        print(f"{label:<25} {val_without_fmt:>15} {val_with_fmt:>15} {imp_fmt:>15}")

    print()
