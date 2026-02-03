"""
Backtesting Framework for TimeParticle SSM

Provides a comprehensive backtesting framework for evaluating
TimeParticle-based trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'long' or 'short'
    size: float
    confidence: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class BacktestResult:
    """Container for backtest results."""
    trades: List[Trade]
    equity_curve: List[float]
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    scale_analysis: Optional[List[Dict]] = None


class TimeParticleBacktest:
    """
    Backtesting framework for TimeParticle trading strategy.

    Supports long-only and long-short strategies with configurable
    position sizing and risk management.
    """

    def __init__(
        self,
        model: Any,
        initial_capital: float = 100000,
        position_size: float = 1.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        long_only: bool = True,
        confidence_threshold: float = 0.6,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """
        Initialize backtester.

        Args:
            model: Trained TimeParticle model
            initial_capital: Starting capital
            position_size: Position size as fraction of capital
            transaction_cost: Transaction cost per trade (%)
            slippage: Slippage per trade (%)
            long_only: If True, only take long positions
            confidence_threshold: Minimum confidence for trade entry
            stop_loss: Stop loss percentage (e.g., 0.05 for 5%)
            take_profit: Take profit percentage (e.g., 0.10 for 10%)
        """
        self.model = model
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.long_only = long_only
        self.confidence_threshold = confidence_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def run(
        self,
        df: pd.DataFrame,
        features: torch.Tensor,
        warmup_period: int = 0
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with price data (must have 'close' column)
            features: Feature tensor [n_samples, lookback, n_features]
            warmup_period: Number of periods to skip at start

        Returns:
            BacktestResult with performance metrics
        """
        signals = self._generate_signals(features)

        capital = self.initial_capital
        position = 0  # 0 = flat, 1 = long, -1 = short
        entry_price = 0.0
        entry_confidence = 0.0

        trades = []
        equity_curve = [capital]
        scale_analysis_history = []

        prices = df['close'].values

        for i in range(warmup_period, len(signals)):
            if i >= len(prices):
                break

            signal, confidence = signals[i]
            price = prices[i]

            # Check stop loss / take profit if in position
            if position != 0 and self.stop_loss is not None:
                if position == 1:  # Long
                    returns = (price - entry_price) / entry_price
                    if returns <= -self.stop_loss:
                        # Stop loss triggered
                        pnl = self._close_position(position, entry_price, price, capital)
                        trades[-1].exit_time = df.index[i]
                        trades[-1].exit_price = price
                        trades[-1].pnl = pnl
                        trades[-1].pnl_pct = pnl / capital * 100
                        capital += pnl
                        position = 0

                elif position == -1:  # Short
                    returns = (entry_price - price) / entry_price
                    if returns <= -self.stop_loss:
                        pnl = self._close_position(position, entry_price, price, capital)
                        trades[-1].exit_time = df.index[i]
                        trades[-1].exit_price = price
                        trades[-1].pnl = pnl
                        trades[-1].pnl_pct = pnl / capital * 100
                        capital += pnl
                        position = 0

            # Check take profit
            if position != 0 and self.take_profit is not None:
                if position == 1:
                    returns = (price - entry_price) / entry_price
                    if returns >= self.take_profit:
                        pnl = self._close_position(position, entry_price, price, capital)
                        trades[-1].exit_time = df.index[i]
                        trades[-1].exit_price = price
                        trades[-1].pnl = pnl
                        trades[-1].pnl_pct = pnl / capital * 100
                        capital += pnl
                        position = 0

                elif position == -1:
                    returns = (entry_price - price) / entry_price
                    if returns >= self.take_profit:
                        pnl = self._close_position(position, entry_price, price, capital)
                        trades[-1].exit_time = df.index[i]
                        trades[-1].exit_price = price
                        trades[-1].pnl = pnl
                        trades[-1].pnl_pct = pnl / capital * 100
                        capital += pnl
                        position = 0

            # Process signals
            if signal == 'BUY' and confidence > self.confidence_threshold:
                if position == 0:
                    # Open long position
                    position = 1
                    entry_price = price * (1 + self.slippage)
                    entry_confidence = confidence
                    cost = capital * self.position_size * self.transaction_cost
                    capital -= cost

                    trades.append(Trade(
                        entry_time=df.index[i],
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        direction='long',
                        size=capital * self.position_size,
                        confidence=confidence
                    ))

                elif position == -1:
                    # Close short and open long
                    pnl = self._close_position(position, entry_price, price, capital)
                    trades[-1].exit_time = df.index[i]
                    trades[-1].exit_price = price
                    trades[-1].pnl = pnl
                    trades[-1].pnl_pct = pnl / capital * 100
                    capital += pnl

                    position = 1
                    entry_price = price * (1 + self.slippage)
                    entry_confidence = confidence
                    cost = capital * self.position_size * self.transaction_cost
                    capital -= cost

                    trades.append(Trade(
                        entry_time=df.index[i],
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        direction='long',
                        size=capital * self.position_size,
                        confidence=confidence
                    ))

            elif signal == 'SELL' and confidence > self.confidence_threshold:
                if position == 1:
                    # Close long position
                    pnl = self._close_position(position, entry_price, price, capital)
                    trades[-1].exit_time = df.index[i]
                    trades[-1].exit_price = price
                    trades[-1].pnl = pnl
                    trades[-1].pnl_pct = pnl / capital * 100
                    capital += pnl
                    position = 0

                elif position == 0 and not self.long_only:
                    # Open short position
                    position = -1
                    entry_price = price * (1 - self.slippage)
                    entry_confidence = confidence
                    cost = capital * self.position_size * self.transaction_cost
                    capital -= cost

                    trades.append(Trade(
                        entry_time=df.index[i],
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        direction='short',
                        size=capital * self.position_size,
                        confidence=confidence
                    ))

            # Calculate equity
            if position == 1:
                equity = capital + capital * self.position_size * (price - entry_price) / entry_price
            elif position == -1:
                equity = capital + capital * self.position_size * (entry_price - price) / entry_price
            else:
                equity = capital

            equity_curve.append(equity)

        # Close any remaining position
        if position != 0 and len(prices) > 0:
            final_price = prices[-1]
            pnl = self._close_position(position, entry_price, final_price, capital)
            if trades:
                trades[-1].exit_time = df.index[-1]
                trades[-1].exit_price = final_price
                trades[-1].pnl = pnl
                trades[-1].pnl_pct = pnl / capital * 100
            capital += pnl

        equity_curve.append(capital)

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            scale_analysis=scale_analysis_history if scale_analysis_history else None,
            **metrics
        )

    def _generate_signals(self, features: torch.Tensor) -> List[Tuple[str, float]]:
        """Generate trading signals from model."""
        self.model.eval()

        signals = []
        with torch.no_grad():
            for i in range(len(features)):
                feat = features[i:i+1]
                probs = self.model.predict_proba(feat)

                buy_prob = probs[0, 2].item()
                sell_prob = probs[0, 0].item()
                hold_prob = probs[0, 1].item()

                if buy_prob > sell_prob and buy_prob > hold_prob:
                    signals.append(('BUY', buy_prob))
                elif sell_prob > buy_prob and sell_prob > hold_prob:
                    signals.append(('SELL', sell_prob))
                else:
                    signals.append(('HOLD', hold_prob))

        return signals

    def _close_position(
        self,
        position: int,
        entry_price: float,
        exit_price: float,
        capital: float
    ) -> float:
        """Calculate P&L for closing a position."""
        if position == 1:  # Long
            pnl = capital * self.position_size * (exit_price - entry_price) / entry_price
        else:  # Short
            pnl = capital * self.position_size * (entry_price - exit_price) / entry_price

        # Subtract transaction costs
        pnl -= capital * self.position_size * self.transaction_cost

        return pnl

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float]
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        equity = np.array(equity_curve)

        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]

        # Total return
        total_return = (equity[-1] / self.initial_capital - 1) * 100

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (annualized)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and np.std(downside_returns) > 0:
            sortino_ratio = np.sqrt(252) * np.mean(returns) / np.std(downside_returns)
        else:
            sortino_ratio = 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) * 100

        # Trade statistics
        completed_trades = [t for t in trades if t.pnl is not None]
        total_trades = len(completed_trades)

        if total_trades > 0:
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl <= 0]

            win_rate = len(winning_trades) / total_trades * 100

            gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1e-10

            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

            avg_trade_pnl = np.mean([t.pnl for t in completed_trades])
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_pnl = 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'avg_trade_pnl': avg_trade_pnl
        }


def run_walk_forward_backtest(
    model_class: Any,
    model_kwargs: Dict,
    df: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    **backtest_kwargs
) -> List[BacktestResult]:
    """
    Run walk-forward optimization/backtest.

    Args:
        model_class: Model class to instantiate
        model_kwargs: Arguments for model initialization
        df: DataFrame with price data
        features: Feature array
        labels: Label array
        n_splits: Number of walk-forward splits
        train_ratio: Ratio of training data in each split
        **backtest_kwargs: Arguments for TimeParticleBacktest

    Returns:
        List of BacktestResult for each split
    """
    results = []
    n_samples = len(features)
    split_size = n_samples // n_splits

    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = min((i + 2) * split_size, n_samples)

        split_features = features[start_idx:end_idx]
        split_labels = labels[start_idx:end_idx]

        train_size = int(len(split_features) * train_ratio)

        train_features = split_features[:train_size]
        train_labels = split_labels[:train_size]

        test_features = split_features[train_size:]

        # Train model
        model = model_class(**model_kwargs)
        # Note: Actual training code would go here
        # model.fit(train_features, train_labels)

        # Backtest on test period
        test_df = df.iloc[start_idx + train_size:start_idx + len(split_features)]
        test_features_tensor = torch.FloatTensor(test_features)

        backtester = TimeParticleBacktest(model, **backtest_kwargs)
        result = backtester.run(test_df, test_features_tensor)
        results.append(result)

    return results


def print_backtest_summary(result: BacktestResult):
    """Print formatted backtest summary."""
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Return:     {result.total_return:>10.2f}%")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:    {result.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:     {result.max_drawdown:>10.2f}%")
    print("-" * 50)
    print(f"Total Trades:     {result.total_trades:>10}")
    print(f"Win Rate:         {result.win_rate:>10.2f}%")
    print(f"Profit Factor:    {result.profit_factor:>10.2f}")
    print(f"Avg Trade P&L:    ${result.avg_trade_pnl:>9.2f}")
    print("=" * 50)


if __name__ == "__main__":
    print("Testing Backtesting Framework...")

    # Create sample model and data
    from timeparticle_model import TimeParticleTradingModel
    from data_loader import generate_simulated_data
    from features import prepare_multiscale_features, get_feature_columns

    # Generate sample data
    df = generate_simulated_data(500)
    features_df = prepare_multiscale_features(df)
    feature_cols = get_feature_columns(features_df)

    # Create model
    model = TimeParticleTradingModel(
        n_features=len(feature_cols),
        d_hidden=32,
        d_state=16,
        n_particles=4,
        n_layers=2
    )

    # Prepare features
    from data_loader import DataPreprocessor
    preprocessor = DataPreprocessor(lookback=50)
    X, y = preprocessor.prepare_sequences(features_df, feature_cols)
    X_tensor = torch.FloatTensor(X)

    # Run backtest
    backtester = TimeParticleBacktest(
        model,
        initial_capital=100000,
        position_size=0.5,
        transaction_cost=0.001,
        confidence_threshold=0.5,
        long_only=True
    )

    # Align dataframe with features
    test_df = features_df.iloc[50:]  # Skip lookback period
    result = backtester.run(test_df, X_tensor)

    # Print results
    print_backtest_summary(result)

    print("\nAll tests passed!")
