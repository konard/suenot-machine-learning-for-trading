"""
Backtesting Framework for Distilled Trading Models

This module provides backtesting capabilities to compare performance
between teacher (complex) and student (distilled) trading models.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'LONG' or 'SHORT'
    size: float
    pnl: Optional[float] = None
    model_type: str = 'student'  # 'teacher' or 'student'
    confidence: float = 0.0
    rule_path: Optional[str] = None

    def close(self, exit_time: datetime, exit_price: float) -> float:
        """Close the trade and calculate PnL."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        if self.direction == 'LONG':
            self.pnl = (exit_price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - exit_price) * self.size
        return self.pnl


@dataclass
class BacktestResult:
    """Results from backtesting."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_pnl: float
    profit_factor: float
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series

    def summary(self) -> str:
        """Return a summary of backtest results."""
        return f"""
Backtest Results:
  Total Return:    {self.total_return:,.2%}
  Sharpe Ratio:    {self.sharpe_ratio:.2f}
  Max Drawdown:    {self.max_drawdown:.2%}
  Win Rate:        {self.win_rate:.2%}
  Total Trades:    {self.total_trades}
  Avg Trade PnL:   ${self.avg_trade_pnl:,.2f}
  Profit Factor:   {self.profit_factor:.2f}
"""


@dataclass
class ComparisonResult:
    """Results from comparing teacher and student models."""
    teacher_result: BacktestResult
    student_result: BacktestResult
    agreement_rate: float
    disagreement_pnl: float
    correlation: float
    teacher_only_trades: int
    student_only_trades: int

    def summary(self) -> str:
        """Return a summary of comparison results."""
        return f"""
Model Comparison Results:
{'=' * 50}

TEACHER MODEL (Complex):
{self.teacher_result.summary()}

STUDENT MODEL (Distilled):
{self.student_result.summary()}

COMPARISON METRICS:
  Agreement Rate:     {self.agreement_rate:.2%}
  Disagreement PnL:   ${self.disagreement_pnl:,.2f}
  Return Correlation: {self.correlation:.2f}
  Teacher-only Trades: {self.teacher_only_trades}
  Student-only Trades: {self.student_only_trades}

INTERPRETATION:
  Return Difference: {self.student_result.total_return - self.teacher_result.total_return:+.2%}
  Sharpe Difference: {self.student_result.sharpe_ratio - self.teacher_result.sharpe_ratio:+.2f}
"""


class DistillationBacktester:
    """
    Backtesting framework for comparing distilled models with their teachers.

    This class provides tools to evaluate both the trading performance and
    the behavioral fidelity of distilled models.

    Attributes:
        initial_capital: Starting capital for backtest
        position_size: Fraction of capital per trade (0-1)
        transaction_cost: Cost per trade as fraction of trade value
        slippage: Expected slippage as fraction of price

    Example:
        >>> from backtest import DistillationBacktester
        >>> from model import DistillationModel
        >>>
        >>> backtester = DistillationBacktester(
        ...     initial_capital=100000,
        ...     position_size=0.1
        ... )
        >>>
        >>> comparison = backtester.compare_models(
        ...     teacher=teacher_model,
        ...     student=distiller,
        ...     data=test_data,
        ...     feature_cols=feature_names,
        ...     price_col='close',
        ...     date_col='date'
        ... )
        >>> print(comparison.summary())
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size: float = 0.1,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting capital (default: $100,000)
            position_size: Fraction of capital per trade (default: 10%)
            transaction_cost: Cost per trade (default: 0.1%)
            slippage: Expected slippage (default: 0.05%)
            risk_free_rate: Annual risk-free rate for Sharpe (default: 2%)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate

    def backtest_model(
        self,
        model: Any,
        data: pd.DataFrame,
        feature_cols: List[str],
        price_col: str = 'close',
        date_col: str = 'date',
        model_type: str = 'student',
    ) -> BacktestResult:
        """
        Backtest a single model.

        Args:
            model: Trading model with predict method
            data: DataFrame with features and prices
            feature_cols: List of feature column names
            price_col: Name of price column
            date_col: Name of date column
            model_type: 'teacher' or 'student'

        Returns:
            result: BacktestResult with performance metrics
        """
        # Prepare data
        data = data.copy().reset_index(drop=True)
        features = data[feature_cols].values
        prices = data[price_col].values
        dates = pd.to_datetime(data[date_col]) if date_col in data.columns else pd.date_range(start='2020-01-01', periods=len(data))

        # Get predictions
        predictions = model.predict(features)

        # Get confidence if available
        if hasattr(model, 'predict_proba'):
            try:
                probas = model.predict_proba(features)
                confidences = np.max(probas, axis=1)
            except Exception:
                confidences = np.ones(len(predictions))
        else:
            confidences = np.ones(len(predictions))

        # Run backtest
        equity = self.initial_capital
        equity_history = [equity]
        trades: List[Trade] = []
        position: Optional[Trade] = None

        for i in range(1, len(predictions)):
            current_price = prices[i]
            prev_price = prices[i - 1]
            signal = predictions[i]
            confidence = confidences[i]
            current_date = dates.iloc[i] if hasattr(dates, 'iloc') else dates[i]
            prev_date = dates.iloc[i - 1] if hasattr(dates, 'iloc') else dates[i - 1]

            # Close position if signal changes
            if position is not None:
                should_close = False
                if position.direction == 'LONG' and signal == 0:
                    should_close = True
                elif position.direction == 'SHORT' and signal == 1:
                    should_close = True

                if should_close:
                    # Apply slippage
                    exit_price = current_price * (1 - self.slippage if position.direction == 'LONG' else 1 + self.slippage)
                    pnl = position.close(current_date, exit_price)
                    # Apply transaction cost
                    pnl -= position.size * exit_price * self.transaction_cost
                    equity += pnl
                    trades.append(position)
                    position = None

            # Open new position
            if position is None and signal != 0.5:  # Assuming 0.5 or middle class is HOLD
                direction = 'LONG' if signal == 1 else 'SHORT'
                # Apply slippage
                entry_price = current_price * (1 + self.slippage if direction == 'LONG' else 1 - self.slippage)
                size = (equity * self.position_size) / entry_price
                # Apply transaction cost
                equity -= size * entry_price * self.transaction_cost

                position = Trade(
                    entry_time=current_date,
                    exit_time=None,
                    entry_price=entry_price,
                    exit_price=None,
                    direction=direction,
                    size=size,
                    model_type=model_type,
                    confidence=confidence,
                )

            equity_history.append(equity + (
                (current_price - position.entry_price) * position.size
                if position and position.direction == 'LONG'
                else (position.entry_price - current_price) * position.size
                if position
                else 0
            ))

        # Close any remaining position
        if position is not None:
            exit_price = prices[-1] * (1 - self.slippage if position.direction == 'LONG' else 1 + self.slippage)
            pnl = position.close(dates.iloc[-1] if hasattr(dates, 'iloc') else dates[-1], exit_price)
            equity += pnl
            trades.append(position)

        # Calculate metrics
        equity_curve = pd.Series(equity_history)
        daily_returns = equity_curve.pct_change().dropna()

        total_return = (equity - self.initial_capital) / self.initial_capital
        sharpe_ratio = self._calculate_sharpe(daily_returns)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        win_rate = self._calculate_win_rate(trades)
        avg_trade_pnl = np.mean([t.pnl for t in trades]) if trades else 0
        profit_factor = self._calculate_profit_factor(trades)

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_trade_pnl=avg_trade_pnl,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
        )

    def compare_models(
        self,
        teacher: Any,
        student: Any,
        data: pd.DataFrame,
        feature_cols: List[str],
        price_col: str = 'close',
        date_col: str = 'date',
    ) -> ComparisonResult:
        """
        Compare teacher and student model performance.

        Args:
            teacher: Complex teacher model
            student: Distilled student model
            data: DataFrame with features and prices
            feature_cols: List of feature column names
            price_col: Name of price column
            date_col: Name of date column

        Returns:
            result: ComparisonResult with both model results and comparison metrics
        """
        # Backtest both models
        teacher_result = self.backtest_model(
            teacher, data, feature_cols, price_col, date_col, 'teacher'
        )
        student_result = self.backtest_model(
            student, data, feature_cols, price_col, date_col, 'student'
        )

        # Get predictions for comparison
        features = data[feature_cols].values
        teacher_pred = teacher.predict(features)
        student_pred = student.predict(features)

        # Calculate agreement
        agreement_rate = np.mean(teacher_pred == student_pred)

        # Calculate PnL impact of disagreements
        prices = data[price_col].values
        returns = np.diff(prices) / prices[:-1]

        disagreement_mask = teacher_pred[:-1] != student_pred[:-1]
        if np.any(disagreement_mask):
            # Simplified: calculate return difference when models disagree
            teacher_returns = np.where(teacher_pred[:-1] == 1, returns, -returns)
            student_returns = np.where(student_pred[:-1] == 1, returns, -returns)
            disagreement_pnl = np.sum(
                (student_returns - teacher_returns)[disagreement_mask]
            ) * self.initial_capital
        else:
            disagreement_pnl = 0.0

        # Calculate correlation
        correlation = np.corrcoef(
            teacher_result.daily_returns,
            student_result.daily_returns
        )[0, 1] if len(teacher_result.daily_returns) > 1 else 0.0

        # Count exclusive trades
        teacher_only = np.sum((teacher_pred == 1) & (student_pred == 0))
        student_only = np.sum((student_pred == 1) & (teacher_pred == 0))

        return ComparisonResult(
            teacher_result=teacher_result,
            student_result=student_result,
            agreement_rate=agreement_rate,
            disagreement_pnl=disagreement_pnl,
            correlation=correlation,
            teacher_only_trades=int(teacher_only),
            student_only_trades=int(student_only),
        )

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown.min()

    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.pnl and t.pnl > 0)
        return wins / len(trades)

    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if not trades:
            return 0.0
        gross_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss


def analyze_rule_performance(
    trades: List[Trade],
    rules: List[Any],
) -> pd.DataFrame:
    """
    Analyze trading performance by rule/decision path.

    Args:
        trades: List of Trade objects with rule_path set
        rules: List of DistilledRule objects

    Returns:
        analysis: DataFrame with rule-level performance metrics
    """
    rule_trades: Dict[str, List[Trade]] = {}

    for trade in trades:
        if trade.rule_path:
            if trade.rule_path not in rule_trades:
                rule_trades[trade.rule_path] = []
            rule_trades[trade.rule_path].append(trade)

    analysis_data = []
    for rule_path, path_trades in rule_trades.items():
        pnls = [t.pnl for t in path_trades if t.pnl is not None]
        if pnls:
            analysis_data.append({
                'rule_path': rule_path,
                'n_trades': len(path_trades),
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls),
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
                'avg_confidence': np.mean([t.confidence for t in path_trades]),
            })

    return pd.DataFrame(analysis_data).sort_values('total_pnl', ascending=False)


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Distillation Backtesting Example")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_days = 500

    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_days))

    # Generate features
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'rsi': np.random.uniform(20, 80, n_days),
        'macd': np.random.normal(0, 1, n_days),
        'volume_ratio': np.random.uniform(0.5, 2.0, n_days),
    })

    feature_cols = ['rsi', 'macd', 'volume_ratio']

    # Create mock models (in practice, use real trained models)
    class MockTeacher:
        def predict(self, X):
            return ((X[:, 0] < 40) & (X[:, 1] > 0)).astype(int)

        def predict_proba(self, X):
            pred = self.predict(X)
            return np.column_stack([1 - pred, pred])

    class MockStudent:
        def predict(self, X):
            return (X[:, 0] < 35).astype(int)  # Simplified rule

        def predict_proba(self, X):
            pred = self.predict(X)
            return np.column_stack([1 - pred * 0.7, pred * 0.7])

    teacher = MockTeacher()
    student = MockStudent()

    # Run backtest comparison
    backtester = DistillationBacktester(
        initial_capital=100000,
        position_size=0.1,
    )

    comparison = backtester.compare_models(
        teacher=teacher,
        student=student,
        data=data,
        feature_cols=feature_cols,
    )

    print(comparison.summary())
