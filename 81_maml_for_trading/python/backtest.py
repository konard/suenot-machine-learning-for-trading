"""
Backtesting framework for MAML trading strategies.

This module provides:
- BacktestEngine: Run historical simulations
- BacktestResults: Store and analyze results
- Performance metrics: Sharpe, Sortino, drawdown, etc.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
import logging

from data_loader import Kline, FeatureGenerator, SimulatedDataGenerator
from maml_trader import MAMLTrainer, TradingModel, TaskData, create_tasks_from_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    direction: int  # 1: long, -1: short
    pnl_pct: float
    pnl_absolute: float


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    threshold: float = 0.001
    adaptation_window: int = 30
    adaptation_steps: int = 5


@dataclass
class BacktestResults:
    """Results from backtesting."""
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    profit_factor: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
=== Backtest Results ===
Capital: ${self.initial_capital:.2f} -> ${self.final_capital:.2f}
Total Return: {self.total_return * 100:.2f}%
Annualized Return: {self.annualized_return * 100:.2f}%
Annualized Volatility: {self.annualized_volatility * 100:.2f}%

Risk Metrics:
  Sharpe Ratio: {self.sharpe_ratio:.3f}
  Sortino Ratio: {self.sortino_ratio:.3f}
  Max Drawdown: {self.max_drawdown * 100:.2f}%

Trading Statistics:
  Total Trades: {self.num_trades}
  Win Rate: {self.win_rate * 100:.1f}%
  Profit Factor: {self.profit_factor:.2f}
"""


class BacktestEngine:
    """
    Backtesting engine for MAML trading strategies.

    Simulates trading using historical data with
    realistic transaction costs and slippage.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.feature_generator = FeatureGenerator(window=20)

    def run(
        self,
        trainer: MAMLTrainer,
        klines: List[Kline],
        verbose: bool = False
    ) -> BacktestResults:
        """
        Run backtest on historical data.

        Args:
            trainer: MAML trainer with meta-learned model
            klines: Historical kline data
            verbose: Print progress if True

        Returns:
            BacktestResults with performance metrics
        """
        # Compute features
        features = self.feature_generator.compute_features(klines)
        if len(features) == 0:
            logger.warning("Insufficient data for feature computation")
            return self._empty_results()

        # Compute returns
        closes = np.array([k.close for k in klines])
        returns = np.diff(closes) / closes[:-1]

        # Align features and returns
        offset = len(closes) - 1 - len(features)
        if offset > 0:
            returns = returns[offset:]

        min_len = min(len(features), len(returns))
        features = features[:min_len]
        returns = returns[:min_len]
        aligned_closes = closes[-(min_len + 1):]

        # Initialize backtest state
        capital = self.config.initial_capital
        position = 0  # 0: flat, 1: long, -1: short
        entry_price = 0.0
        entry_time = 0

        trades: List[Trade] = []
        equity_curve: List[float] = [capital]
        daily_returns: List[float] = []

        # Start after enough data for adaptation
        start_idx = max(self.config.adaptation_window + 10, 50)

        if verbose:
            logger.info(f"Starting backtest with {len(features) - start_idx} trading days")

        for i in range(start_idx, len(features)):
            current_price = aligned_closes[i + 1]
            current_time = klines[-(min_len + 1 - i - 1)].timestamp if i + 1 < len(aligned_closes) else 0

            # Get recent data for adaptation
            adapt_start = max(0, i - self.config.adaptation_window)
            recent_features = features[adapt_start:i]
            recent_returns = returns[adapt_start:i]

            # Skip if insufficient data
            if len(recent_features) < 10:
                continue

            # Adapt model and predict
            adapted_model = trainer.adapt(
                recent_features,
                recent_returns,
                self.config.adaptation_steps
            )
            current_features = features[i:i+1]
            prediction = adapted_model.predict(current_features)[0]

            # Check exit conditions for existing position
            if position != 0:
                pnl_pct = self._compute_pnl_pct(entry_price, current_price, position)

                # Stop loss
                if pnl_pct < -0.02:
                    capital, trade = self._close_position(
                        capital, position, entry_price, entry_time,
                        current_price, current_time
                    )
                    trades.append(trade)
                    position = 0

                # Take profit
                elif pnl_pct > 0.04:
                    capital, trade = self._close_position(
                        capital, position, entry_price, entry_time,
                        current_price, current_time
                    )
                    trades.append(trade)
                    position = 0

            # Generate signal
            if position == 0:
                if prediction > self.config.threshold:
                    # Buy signal
                    position = 1
                    entry_price = current_price * (1 + self.config.slippage)
                    entry_time = current_time
                    capital *= (1 - self.config.transaction_cost)

                elif prediction < -self.config.threshold:
                    # Sell signal
                    position = -1
                    entry_price = current_price * (1 - self.config.slippage)
                    entry_time = current_time
                    capital *= (1 - self.config.transaction_cost)

            # Update equity
            if position != 0:
                pnl_pct = self._compute_pnl_pct(entry_price, current_price, position)
                current_equity = capital * (1 + pnl_pct)
            else:
                current_equity = capital

            # Track daily returns
            if len(equity_curve) > 0:
                daily_return = (current_equity / equity_curve[-1]) - 1
                daily_returns.append(daily_return)

            equity_curve.append(current_equity)

        # Close any remaining position
        if position != 0:
            final_price = aligned_closes[-1]
            final_time = klines[-1].timestamp
            capital, trade = self._close_position(
                capital, position, entry_price, entry_time,
                final_price, final_time
            )
            trades.append(trade)

        # Compute metrics
        return self._compute_results(
            self.config.initial_capital,
            capital,
            trades,
            equity_curve,
            daily_returns
        )

    def _compute_pnl_pct(
        self,
        entry_price: float,
        current_price: float,
        position: int
    ) -> float:
        """Compute P&L percentage."""
        if position > 0:
            return (current_price / entry_price) - 1
        else:
            return 1 - (current_price / entry_price)

    def _close_position(
        self,
        capital: float,
        position: int,
        entry_price: float,
        entry_time: int,
        exit_price: float,
        exit_time: int
    ) -> tuple:
        """Close position and return updated capital and trade record."""
        # Apply slippage
        if position > 0:
            actual_exit = exit_price * (1 - self.config.slippage)
        else:
            actual_exit = exit_price * (1 + self.config.slippage)

        # Compute P&L
        pnl_pct = self._compute_pnl_pct(entry_price, actual_exit, position)
        pnl_absolute = capital * pnl_pct

        # Update capital (apply transaction cost)
        new_capital = capital * (1 + pnl_pct) * (1 - self.config.transaction_cost)

        trade = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=actual_exit,
            direction=position,
            pnl_pct=pnl_pct,
            pnl_absolute=pnl_absolute
        )

        return new_capital, trade

    def _compute_results(
        self,
        initial_capital: float,
        final_capital: float,
        trades: List[Trade],
        equity_curve: List[float],
        daily_returns: List[float]
    ) -> BacktestResults:
        """Compute backtest metrics."""
        total_return = (final_capital / initial_capital) - 1

        # Annualized metrics (assuming hourly data, ~8760 hours/year)
        num_periods = len(equity_curve)
        annualized_return = (1 + total_return) ** (8760 / max(num_periods, 1)) - 1

        if daily_returns:
            volatility = np.std(daily_returns)
            annualized_volatility = volatility * np.sqrt(8760)

            # Sharpe ratio (assuming 0% risk-free rate)
            if annualized_volatility > 0:
                sharpe_ratio = annualized_return / annualized_volatility
            else:
                sharpe_ratio = 0.0

            # Sortino ratio
            downside_returns = [r for r in daily_returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                annualized_downside = downside_std * np.sqrt(8760)
                if annualized_downside > 0:
                    sortino_ratio = annualized_return / annualized_downside
                else:
                    sortino_ratio = float('inf')
            else:
                sortino_ratio = float('inf')
        else:
            annualized_volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        # Max drawdown
        max_drawdown = self._compute_max_drawdown(equity_curve)

        # Trade statistics
        num_trades = len(trades)
        if num_trades > 0:
            winning_trades = [t for t in trades if t.pnl_pct > 0]
            losing_trades = [t for t in trades if t.pnl_pct <= 0]

            win_rate = len(winning_trades) / num_trades

            total_profit = sum(t.pnl_pct for t in winning_trades) if winning_trades else 0
            total_loss = abs(sum(t.pnl_pct for t in losing_trades)) if losing_trades else 0

            if total_loss > 0:
                profit_factor = total_profit / total_loss
            else:
                profit_factor = float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return BacktestResults(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_curve
        )

    def _compute_max_drawdown(self, equity_curve: List[float]) -> float:
        """Compute maximum drawdown."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)

        return max_dd

    def _empty_results(self) -> BacktestResults:
        """Return empty results for insufficient data."""
        return BacktestResults(
            initial_capital=self.config.initial_capital,
            final_capital=self.config.initial_capital,
            total_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            num_trades=0,
            win_rate=0.0,
            profit_factor=0.0
        )


def run_full_backtest_example():
    """Run a complete backtesting example."""
    print("=== MAML Trading Backtest Example ===\n")

    # Phase 1: Generate training data (different market regimes)
    print("Phase 1: Generating training data...")
    feature_gen = FeatureGenerator(window=20)

    regimes = [
        ("Bull Market", 0.015, 0.0003),
        ("Bear Market", 0.02, -0.0003),
        ("Sideways", 0.008, 0.0),
        ("High Volatility", 0.03, 0.0),
    ]

    tasks = []
    for name, vol, trend in regimes:
        klines = SimulatedDataGenerator.generate_trending_klines(400, 50000.0, vol, trend)
        features = feature_gen.compute_features(klines)
        closes = np.array([k.close for k in klines])
        returns = np.diff(closes) / closes[:-1]

        # Align
        offset = len(closes) - 1 - len(features)
        if offset > 0:
            returns = returns[offset:]
        min_len = min(len(features), len(returns))
        features = features[:min_len]
        returns = returns[:min_len]

        # Create tasks
        regime_tasks = create_tasks_from_data(features, returns, num_tasks=3)
        tasks.extend(regime_tasks)
        print(f"  {name}: {len(regime_tasks)} tasks")

    print(f"\nTotal tasks: {len(tasks)}\n")

    # Phase 2: Meta-train
    print("Phase 2: Meta-training MAML...")
    model = TradingModel(11, 64)
    trainer = MAMLTrainer(model, inner_lr=0.01, outer_lr=0.001, inner_steps=5, first_order=True)

    for epoch in range(20):
        loss = trainer.meta_train_step(tasks)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {loss:.6f}")
    print()

    # Phase 3: Backtest
    print("Phase 3: Running backtest on new data...")
    test_klines = SimulatedDataGenerator.generate_regime_changing_klines(1000, 50000.0)
    print(f"Generated {len(test_klines)} test candles\n")

    config = BacktestConfig(
        initial_capital=10000.0,
        transaction_cost=0.001,
        slippage=0.0005,
        threshold=0.001,
        adaptation_window=30,
        adaptation_steps=5
    )

    engine = BacktestEngine(config)
    results = engine.run(trainer, test_klines, verbose=True)

    # Phase 4: Display results
    print(results.summary())

    # Compare with buy-and-hold
    first_price = test_klines[0].close
    last_price = test_klines[-1].close
    buy_hold_return = (last_price / first_price) - 1

    print(f"\n=== Comparison ===")
    print(f"MAML Strategy: {results.total_return * 100:+.2f}%")
    print(f"Buy & Hold:    {buy_hold_return * 100:+.2f}%")

    outperformance = results.total_return - buy_hold_return
    if outperformance > 0:
        print(f"\nMAML outperformed by {outperformance * 100:.2f}%")
    else:
        print(f"\nBuy & Hold outperformed by {-outperformance * 100:.2f}%")


if __name__ == "__main__":
    run_full_backtest_example()
