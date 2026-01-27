"""
Backtesting framework for Meta-RL trading strategies.

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
from meta_rl_trader import MetaRLAgent, TradingEnvironment, PPOMetaTrainer

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
    adaptation_episodes: int = 2
    episode_length: int = 200


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
=== Meta-RL Backtest Results ===
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
    Backtesting engine for Meta-RL trading strategies.

    Runs the Meta-RL agent through historical data,
    allowing adaptation via hidden state across episodes.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.feature_generator = FeatureGenerator(window=20)

    def run(
        self,
        agent: MetaRLAgent,
        klines: List[Kline],
        verbose: bool = False
    ) -> BacktestResults:
        """
        Run backtest on historical data.

        Args:
            agent: Meta-RL agent (already meta-trained)
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

        # Prepare prices aligned with features
        closes = np.array([k.close for k in klines])
        prices = closes[len(closes) - len(features):]

        # Create trading environment
        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_capital=self.config.initial_capital,
            transaction_cost=self.config.transaction_cost,
            max_steps=self.config.episode_length
        )

        # Run adaptation episodes (hidden state accumulates)
        hidden = None
        prev_action, prev_reward, prev_done = 0, 0.0, False

        if verbose:
            logger.info(f"Running {self.config.adaptation_episodes} adaptation episodes...")

        for ep in range(self.config.adaptation_episodes):
            state = env.reset()
            done = False
            while not done:
                action, _, _, hidden = agent.get_action(
                    state, prev_action, prev_reward, prev_done, hidden
                )
                state, reward, done, info = env.step(action)
                prev_action, prev_reward, prev_done = action, reward, done

            if verbose:
                logger.info(f"  Adaptation episode {ep + 1}: capital={info['capital']:.2f}")

        # Evaluation episode
        if verbose:
            logger.info("Running evaluation episode...")

        state = env.reset()
        done = False
        step_returns = []
        equity_curve = [self.config.initial_capital]
        trades: List[Trade] = []
        prev_position = 0
        entry_price = 0.0
        entry_time = 0

        step_idx = 0
        while not done:
            action, _, _, hidden = agent.get_action(
                state, prev_action, prev_reward, prev_done, hidden
            )
            state, reward, done, info = env.step(action)
            step_returns.append(reward)
            equity_curve.append(info['capital'])

            # Track trades
            current_position = info['position']
            current_price = prices[min(env.current_idx, len(prices) - 1)]

            if prev_position == 0 and current_position != 0:
                entry_price = current_price
                entry_time = step_idx
            elif prev_position != 0 and (current_position == 0 or current_position != prev_position):
                if entry_price > 0:
                    if prev_position > 0:
                        pnl_pct = (current_price / entry_price) - 1
                    else:
                        pnl_pct = 1 - (current_price / entry_price)
                    trades.append(Trade(
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=step_idx,
                        exit_price=current_price,
                        direction=prev_position,
                        pnl_pct=pnl_pct,
                        pnl_absolute=pnl_pct * self.config.initial_capital
                    ))

            prev_position = current_position
            prev_action, prev_reward, prev_done = action, reward, done
            step_idx += 1

        # Compute metrics
        return self._compute_results(
            self.config.initial_capital,
            info['capital'],
            trades,
            equity_curve,
            step_returns
        )

    def _compute_results(
        self,
        initial_capital: float,
        final_capital: float,
        trades: List[Trade],
        equity_curve: List[float],
        step_returns: List[float]
    ) -> BacktestResults:
        """Compute backtest metrics."""
        total_return = (final_capital / initial_capital) - 1

        num_periods = len(equity_curve)
        annualized_return = (1 + total_return) ** (8760 / max(num_periods, 1)) - 1

        if step_returns:
            returns_arr = np.array(step_returns)
            volatility = returns_arr.std()
            annualized_volatility = volatility * np.sqrt(8760)

            if annualized_volatility > 0:
                sharpe_ratio = annualized_return / annualized_volatility
            else:
                sharpe_ratio = 0.0

            downside_returns = returns_arr[returns_arr < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                annualized_downside = downside_std * np.sqrt(8760)
                sortino_ratio = annualized_return / annualized_downside if annualized_downside > 0 else float('inf')
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
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
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
    print("=== Meta-RL Trading Backtest Example ===\n")

    # Phase 1: Generate training data
    print("Phase 1: Generating training environments...")
    feature_gen = FeatureGenerator(window=20)

    regimes = [
        ("Bull Market", 0.015, 0.0003),
        ("Bear Market", 0.02, -0.0003),
        ("Sideways", 0.008, 0.0),
        ("High Volatility", 0.03, 0.0),
    ]

    environments = []
    for name, vol, trend in regimes:
        klines = SimulatedDataGenerator.generate_trending_klines(400, 50000.0, vol, trend)
        features = feature_gen.compute_features(klines)
        closes = np.array([k.close for k in klines])
        prices = closes[len(closes) - len(features):]

        env = TradingEnvironment(
            prices=prices,
            features=features,
            initial_capital=10000.0,
            max_steps=100
        )
        environments.append(env)
        print(f"  {name}: {len(features)} steps")

    # Phase 2: Meta-train
    print("\nPhase 2: Meta-training Meta-RL agent...")
    state_dim = environments[0].state_dim
    action_dim = environments[0].action_dim
    agent = MetaRLAgent(state_dim, action_dim, hidden_size=64)
    trainer = PPOMetaTrainer(agent, lr=3e-4, num_episodes_per_trial=2)

    for epoch in range(20):
        metrics = trainer.meta_train_step(environments)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: loss={metrics['total_loss']:.4f}")
    print()

    # Phase 3: Backtest
    print("Phase 3: Running backtest on new data...")
    test_klines = SimulatedDataGenerator.generate_regime_changing_klines(1000, 50000.0)
    print(f"Generated {len(test_klines)} test candles\n")

    config = BacktestConfig(
        initial_capital=10000.0,
        transaction_cost=0.001,
        adaptation_episodes=2,
        episode_length=200
    )

    engine = BacktestEngine(config)
    results = engine.run(agent, test_klines, verbose=True)

    # Phase 4: Display results
    print(results.summary())

    # Compare with buy-and-hold
    first_price = test_klines[0].close
    last_price = test_klines[-1].close
    buy_hold_return = (last_price / first_price) - 1

    print(f"\n=== Comparison ===")
    print(f"Meta-RL Strategy: {results.total_return * 100:+.2f}%")
    print(f"Buy & Hold:       {buy_hold_return * 100:+.2f}%")

    outperformance = results.total_return - buy_hold_return
    if outperformance > 0:
        print(f"\nMeta-RL outperformed by {outperformance * 100:.2f}%")
    else:
        print(f"\nBuy & Hold outperformed by {-outperformance * 100:.2f}%")


if __name__ == "__main__":
    run_full_backtest_example()
