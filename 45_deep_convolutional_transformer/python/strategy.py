"""
Trading Strategy and Backtesting for DCT Model

Provides:
- SignalGenerator: Generate trading signals from predictions
- Backtester: Backtest trading strategies
- Performance metrics calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StrategyConfig:
    """Configuration for trading strategy."""
    initial_capital: float = 100000.0
    position_size: float = 0.1  # 10% of capital per trade
    transaction_cost: float = 0.001  # 0.1% per trade
    confidence_threshold: float = 0.6  # Minimum confidence to trade
    max_positions: int = 1  # Maximum concurrent positions
    stop_loss: Optional[float] = 0.02  # 2% stop loss
    take_profit: Optional[float] = 0.05  # 5% take profit


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: int  # 1 = long, -1 = short
    size: float
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class BacktestResult:
    """Results from backtesting."""
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict[str, float]
    daily_returns: pd.Series


class SignalGenerator:
    """
    Generate trading signals from DCT predictions.

    Converts model predictions to actionable trading signals
    with confidence filtering.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def generate_signals(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate trading signals from predictions.

        Args:
            predictions: Array of predicted classes [n_samples]
            probabilities: Array of class probabilities [n_samples, 3]

        Returns:
            DataFrame with signals and confidence
        """
        n_samples = len(predictions)

        signals = pd.DataFrame({
            'prediction': predictions,
            'confidence': probabilities.max(axis=1),
            'prob_up': probabilities[:, 0],
            'prob_down': probabilities[:, 1],
            'prob_stable': probabilities[:, 2]
        })

        # Generate position signals
        signals['signal'] = 0  # 0 = no position

        # Long signal when predicting UP with high confidence
        long_mask = (
            (signals['prediction'] == 0) &
            (signals['confidence'] >= self.config.confidence_threshold)
        )
        signals.loc[long_mask, 'signal'] = 1

        # Short signal when predicting DOWN with high confidence
        short_mask = (
            (signals['prediction'] == 1) &
            (signals['confidence'] >= self.config.confidence_threshold)
        )
        signals.loc[short_mask, 'signal'] = -1

        return signals


class Backtester:
    """
    Backtesting engine for DCT trading strategy.

    Simulates trading with realistic transaction costs,
    position sizing, and risk management.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.Series,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> BacktestResult:
        """
        Run backtest on signals.

        Args:
            signals: DataFrame with trading signals
            prices: Series of close prices
            dates: Optional datetime index

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        if dates is None:
            dates = pd.RangeIndex(len(prices))

        capital = self.config.initial_capital
        position = 0
        position_size = 0
        entry_price = 0
        entry_idx = 0
        entry_capital = 0  # Track capital at entry for accurate PnL

        trades: List[Trade] = []
        equity = [capital]
        daily_returns = []

        for i in range(len(signals)):
            current_price = prices.iloc[i]
            signal = signals['signal'].iloc[i]

            # Check for exit conditions if in position
            if position != 0:
                returns = (current_price - entry_price) / entry_price * position

                # Stop loss
                if self.config.stop_loss and returns < -self.config.stop_loss:
                    # Exit position - use entry_capital for consistent PnL
                    pnl = returns * position_size * entry_capital
                    # Transaction cost on notional value (position_size * entry_capital)
                    cost = position_size * entry_capital * self.config.transaction_cost
                    capital += pnl - cost

                    trades.append(Trade(
                        entry_date=dates[entry_idx],
                        exit_date=dates[i],
                        entry_price=entry_price,
                        exit_price=current_price,
                        direction=position,
                        size=position_size,
                        pnl=pnl - cost,
                        exit_reason='stop_loss'
                    ))
                    position = 0
                    position_size = 0

                # Take profit
                elif self.config.take_profit and returns > self.config.take_profit:
                    pnl = returns * position_size * entry_capital
                    # Transaction cost on notional value
                    cost = position_size * entry_capital * self.config.transaction_cost
                    capital += pnl - cost

                    trades.append(Trade(
                        entry_date=dates[entry_idx],
                        exit_date=dates[i],
                        entry_price=entry_price,
                        exit_price=current_price,
                        direction=position,
                        size=position_size,
                        pnl=pnl - cost,
                        exit_reason='take_profit'
                    ))
                    position = 0
                    position_size = 0

            # Check for new signals
            if position == 0 and signal != 0:
                # Enter new position
                position = signal
                position_size = self.config.position_size
                entry_price = current_price
                entry_idx = i
                entry_capital = capital  # Track capital at entry

                # Entry cost on notional value (position_size * entry_capital)
                cost = position_size * entry_capital * self.config.transaction_cost
                capital -= cost

            elif position != 0 and signal != position and signal != 0:
                # Signal reversal - exit current and enter new
                returns = (current_price - entry_price) / entry_price * position
                pnl = returns * position_size * entry_capital
                # Transaction cost on notional value
                cost = position_size * entry_capital * self.config.transaction_cost
                capital += pnl - cost

                trades.append(Trade(
                    entry_date=dates[entry_idx],
                    exit_date=dates[i],
                    entry_price=entry_price,
                    exit_price=current_price,
                    direction=position,
                    size=position_size,
                    pnl=pnl - cost,
                    exit_reason='signal_reversal'
                ))

                # Enter new position
                position = signal
                position_size = self.config.position_size
                entry_price = current_price
                entry_idx = i
                entry_capital = capital  # Track capital for new position
                # Entry cost on notional value
                cost = position_size * entry_capital * self.config.transaction_cost
                capital -= cost

            # Calculate current equity
            if position != 0:
                unrealized_pnl = (current_price - entry_price) / entry_price * position * position_size * entry_capital
                equity.append(capital + unrealized_pnl)
            else:
                equity.append(capital)

            # Daily returns
            if len(equity) > 1:
                daily_returns.append((equity[-1] - equity[-2]) / equity[-2])
            else:
                daily_returns.append(0)

        # Close any open position at end
        if position != 0:
            current_price = prices.iloc[-1]
            returns = (current_price - entry_price) / entry_price * position
            pnl = returns * position_size * entry_capital
            # Transaction cost on notional value
            cost = position_size * entry_capital * self.config.transaction_cost
            capital += pnl - cost

            trades.append(Trade(
                entry_date=dates[entry_idx],
                exit_date=dates[-1],
                entry_price=entry_price,
                exit_price=current_price,
                direction=position,
                size=position_size,
                pnl=pnl - cost,
                exit_reason='end_of_backtest'
            ))

        # Equity curve: starts with initial capital, then one value per day
        # Align index properly - first value is initial capital at start
        if len(equity) > len(dates):
            # Trim equity to match dates (drop initial capital point)
            equity_curve = pd.Series(equity[1:], index=dates)
        else:
            equity_curve = pd.Series(equity, index=dates[:len(equity)])
        daily_returns = pd.Series(daily_returns, index=dates[:len(daily_returns)])

        metrics = self._calculate_metrics(trades, equity_curve, daily_returns)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            daily_returns=daily_returns
        )

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        daily_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        initial_capital = self.config.initial_capital
        final_capital = equity_curve.iloc[-1]

        # Total return
        total_return = (final_capital - initial_capital) / initial_capital * 100

        # Trade statistics
        if trades:
            trade_pnls = [t.pnl for t in trades if t.pnl is not None]
            winning_trades = [p for p in trade_pnls if p > 0]
            losing_trades = [p for p in trade_pnls if p < 0]

            win_rate = len(winning_trades) / len(trade_pnls) * 100 if trade_pnls else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Risk metrics
        if len(daily_returns) > 0:
            sharpe_ratio = (
                daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                if daily_returns.std() > 0 else 0
            )
            sortino_ratio = (
                daily_returns.mean() / daily_returns[daily_returns < 0].std() * np.sqrt(252)
                if len(daily_returns[daily_returns < 0]) > 0 else 0
            )
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Maximum drawdown
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min() * 100

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': final_capital
        }


def backtest_dct_strategy(
    model,
    test_data: np.ndarray,
    test_prices: pd.Series,
    config: Optional[StrategyConfig] = None,
    dates: Optional[pd.DatetimeIndex] = None
) -> BacktestResult:
    """
    Backtest DCT model on test data.

    Args:
        model: Trained DCT model
        test_data: Test features [n_samples, seq_len, n_features]
        test_prices: Close prices for test period
        config: Strategy configuration
        dates: Optional datetime index

    Returns:
        BacktestResult with trades and metrics
    """
    import torch

    if config is None:
        config = StrategyConfig()

    # Generate predictions
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(test_data)
        if torch.cuda.is_available():
            X = X.cuda()
            model = model.cuda()

        output = model(X)
        predictions = output['predictions'].cpu().numpy()
        probabilities = output['probabilities'].cpu().numpy()

    # Generate signals
    signal_gen = SignalGenerator(config)
    signals = signal_gen.generate_signals(predictions, probabilities)

    # Run backtest
    backtester = Backtester(config)
    result = backtester.run(signals, test_prices, dates)

    return result


def print_backtest_report(result: BacktestResult):
    """Print a formatted backtest report."""
    print("\n" + "="*60)
    print("BACKTEST REPORT")
    print("="*60)

    m = result.metrics

    print(f"\nPerformance Metrics:")
    print(f"  Total Return:     {m['total_return']:>10.2f}%")
    print(f"  Sharpe Ratio:     {m['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:    {m['sortino_ratio']:>10.2f}")
    print(f"  Max Drawdown:     {m['max_drawdown']:>10.2f}%")
    print(f"  Final Capital:    ${m['final_capital']:>10,.2f}")

    print(f"\nTrade Statistics:")
    print(f"  Number of Trades: {m['num_trades']:>10}")
    print(f"  Win Rate:         {m['win_rate']:>10.2f}%")
    print(f"  Avg Win:          ${m['avg_win']:>10,.2f}")
    print(f"  Avg Loss:         ${m['avg_loss']:>10,.2f}")
    print(f"  Profit Factor:    {m['profit_factor']:>10.2f}")

    print("\n" + "="*60)

    # Trade breakdown by exit reason
    if result.trades:
        reasons = {}
        for trade in result.trades:
            reason = trade.exit_reason or 'unknown'
            if reason not in reasons:
                reasons[reason] = {'count': 0, 'pnl': 0}
            reasons[reason]['count'] += 1
            reasons[reason]['pnl'] += trade.pnl or 0

        print("\nTrades by Exit Reason:")
        for reason, stats in reasons.items():
            print(f"  {reason}: {stats['count']} trades, ${stats['pnl']:,.2f} P&L")


if __name__ == "__main__":
    # Test backtesting with dummy data
    print("Testing backtesting engine...")

    # Create dummy signals
    np.random.seed(42)
    n_samples = 100

    predictions = np.random.randint(0, 3, n_samples)
    probabilities = np.random.dirichlet(np.ones(3), n_samples)

    # Create dummy price series (random walk)
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.02)),
        index=pd.date_range('2024-01-01', periods=n_samples)
    )

    # Generate signals
    config = StrategyConfig(
        initial_capital=100000,
        position_size=0.1,
        confidence_threshold=0.5
    )
    signal_gen = SignalGenerator(config)
    signals = signal_gen.generate_signals(predictions, probabilities)

    print(f"Signals generated: {len(signals)}")
    print(f"Long signals: {(signals['signal'] == 1).sum()}")
    print(f"Short signals: {(signals['signal'] == -1).sum()}")
    print(f"No signal: {(signals['signal'] == 0).sum()}")

    # Run backtest
    backtester = Backtester(config)
    result = backtester.run(signals, prices, prices.index)

    print_backtest_report(result)
