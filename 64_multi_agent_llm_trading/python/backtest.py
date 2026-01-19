"""
Backtesting Module for Multi-Agent LLM Trading

This module provides backtesting capabilities for multi-agent trading strategies.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

from .agents import BaseAgent, Analysis, Signal, TraderAgent
from .communication import RoundTable, Debate, DebateModerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: datetime
    symbol: str
    action: str  # "BUY", "SELL", "CLOSE"
    price: float
    quantity: float
    signal: Signal
    confidence: float
    reasoning: str
    pnl: float = 0.0
    fees: float = 0.0


@dataclass
class Position:
    """Current position in an asset."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price / self.entry_price - 1) * 100


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    trades: List[Trade]
    daily_returns: pd.Series
    equity_curve: pd.Series
    positions_history: List[Dict]

    @property
    def total_return(self) -> float:
        """Total return percentage."""
        return (self.final_capital / self.initial_capital - 1) * 100

    @property
    def num_trades(self) -> int:
        """Number of trades executed."""
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if not self.trades:
            return 0.0
        winning = sum(1 for t in self.trades if t.pnl > 0)
        return winning / len(self.trades) * 100

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio."""
        if self.daily_returns.std() == 0:
            return 0.0
        return (self.daily_returns.mean() / self.daily_returns.std()) * np.sqrt(252)

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino ratio."""
        downside_returns = self.daily_returns[self.daily_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if self.daily_returns.mean() > 0 else 0.0
        return (self.daily_returns.mean() / downside_returns.std()) * np.sqrt(252)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown percentage."""
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve / rolling_max - 1) * 100
        return drawdown.min()

    @property
    def calmar_ratio(self) -> float:
        """Calmar ratio (annualized return / max drawdown)."""
        if self.max_drawdown == 0:
            return 0.0
        annual_return = self.total_return * (252 / len(self.daily_returns)) if len(self.daily_returns) > 0 else 0
        return abs(annual_return / self.max_drawdown)

    def summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            "period": f"{self.start_date.date()} to {self.end_date.date()}",
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_return_pct": round(self.total_return, 2),
            "num_trades": self.num_trades,
            "win_rate_pct": round(self.win_rate, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
        }

    def print_summary(self):
        """Print formatted summary."""
        summary = self.summary()
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 50)


class MultiAgentBacktester:
    """
    Backtester for multi-agent trading strategies.

    Supports various communication patterns:
    - Independent analysis aggregation
    - Round table discussions
    - Bull vs Bear debates
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        trader_agent: Optional[TraderAgent] = None,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.1,
        transaction_cost_pct: float = 0.001,
        use_debate: bool = False,
        debate_rounds: int = 2
    ):
        """
        Initialize backtester.

        Args:
            agents: List of analysis agents
            trader_agent: Optional trader for aggregation (created if not provided)
            initial_capital: Starting capital
            position_size_pct: Position size as percentage of capital
            transaction_cost_pct: Transaction cost percentage
            use_debate: Whether to use Bull vs Bear debate
            debate_rounds: Number of debate rounds if using debate
        """
        self.agents = agents
        self.trader_agent = trader_agent or TraderAgent("Backtest-Trader")
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.transaction_cost_pct = transaction_cost_pct
        self.use_debate = use_debate
        self.debate_rounds = debate_rounds

        # State
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.positions_history: List[Dict] = []

    def reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.positions_history = []

    def _get_equity(self) -> float:
        """Calculate total equity (cash + positions)."""
        position_value = sum(
            p.quantity * p.current_price
            for p in self.positions.values()
        )
        return self.capital + position_value

    def _update_position_prices(self, symbol: str, price: float):
        """Update current price for positions."""
        if symbol in self.positions:
            self.positions[symbol].current_price = price

    def _execute_signal(
        self,
        timestamp: datetime,
        symbol: str,
        price: float,
        signal: Signal,
        confidence: float,
        reasoning: str
    ) -> Optional[Trade]:
        """Execute a trading signal."""
        # Determine action based on signal and current position
        has_position = symbol in self.positions

        if signal in [Signal.STRONG_BUY, Signal.BUY] and not has_position:
            # Open long position
            position_value = self.capital * self.position_size_pct * (confidence if signal == Signal.BUY else 1.0)
            quantity = position_value / price
            cost = position_value * (1 + self.transaction_cost_pct)

            if cost <= self.capital:
                self.capital -= cost
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=timestamp,
                    current_price=price
                )

                trade = Trade(
                    timestamp=timestamp,
                    symbol=symbol,
                    action="BUY",
                    price=price,
                    quantity=quantity,
                    signal=signal,
                    confidence=confidence,
                    reasoning=reasoning,
                    fees=position_value * self.transaction_cost_pct
                )
                self.trades.append(trade)
                return trade

        elif signal in [Signal.STRONG_SELL, Signal.SELL] and has_position:
            # Close long position
            position = self.positions[symbol]
            sale_value = position.quantity * price * (1 - self.transaction_cost_pct)
            pnl = sale_value - (position.quantity * position.entry_price)

            self.capital += sale_value

            trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                action="SELL",
                price=price,
                quantity=position.quantity,
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                pnl=pnl,
                fees=position.quantity * price * self.transaction_cost_pct
            )
            self.trades.append(trade)

            del self.positions[symbol]
            return trade

        return None

    def _analyze_step(
        self,
        symbol: str,
        data: pd.DataFrame,
        context: Optional[Dict] = None
    ) -> Tuple[Signal, float, str]:
        """
        Run multi-agent analysis for a single step.

        Returns:
            Tuple of (signal, confidence, reasoning)
        """
        context = context or {}

        if self.use_debate:
            # Find bull and bear agents
            bull_agent = next((a for a in self.agents if "bull" in a.agent_type.lower()), None)
            bear_agent = next((a for a in self.agents if "bear" in a.agent_type.lower()), None)

            if bull_agent and bear_agent:
                debate = Debate(bull_agent, bear_agent, num_rounds=self.debate_rounds)
                result = debate.conduct(symbol, data, context)

                moderator = DebateModerator()
                conclusion = moderator.evaluate(result)

                return (
                    Signal[conclusion["signal"]],
                    conclusion["confidence"],
                    conclusion["recommendation"]
                )

        # Standard round table approach
        analyses = []
        for agent in self.agents:
            analysis = agent.analyze(symbol, data, context)
            analyses.append(analysis)

        # Aggregate with trader
        trader_context = {**context, "analyses": analyses}
        final_analysis = self.trader_agent.analyze(symbol, data, trader_context)

        return (
            final_analysis.signal,
            final_analysis.confidence,
            final_analysis.reasoning
        )

    def run(
        self,
        symbol: str,
        data: pd.DataFrame,
        lookback: int = 50,
        step: int = 1,
        context: Optional[Dict] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame with datetime index
            lookback: Number of periods for analysis lookback
            step: Rebalancing frequency (1 = every day)
            context: Additional context passed to agents

        Returns:
            BacktestResult with performance metrics
        """
        self.reset()
        context = context or {}

        if len(data) < lookback + 10:
            raise ValueError(f"Insufficient data: need at least {lookback + 10} periods")

        dates = data.index[lookback:]
        daily_returns = []
        prev_equity = self.initial_capital

        for i, date in enumerate(dates):
            if i % step != 0:
                # Just update prices, don't trade
                price = data.loc[date, "close"]
                self._update_position_prices(symbol, price)
                continue

            # Get lookback window
            window_end = data.index.get_loc(date)
            window_start = window_end - lookback
            window_data = data.iloc[window_start:window_end + 1].copy()

            current_price = data.loc[date, "close"]
            self._update_position_prices(symbol, current_price)

            # Run multi-agent analysis
            try:
                signal, confidence, reasoning = self._analyze_step(
                    symbol, window_data, context
                )

                # Execute signal
                self._execute_signal(
                    timestamp=date,
                    symbol=symbol,
                    price=current_price,
                    signal=signal,
                    confidence=confidence,
                    reasoning=reasoning
                )
            except Exception as e:
                logger.warning(f"Analysis failed at {date}: {e}")
                signal = Signal.NEUTRAL
                confidence = 0.0
                reasoning = f"Error: {str(e)}"

            # Record equity
            current_equity = self._get_equity()
            self.equity_history.append((date, current_equity))

            # Calculate daily return
            daily_return = (current_equity / prev_equity - 1) if prev_equity > 0 else 0
            daily_returns.append(daily_return)
            prev_equity = current_equity

            # Record position snapshot
            self.positions_history.append({
                "date": date,
                "equity": current_equity,
                "cash": self.capital,
                "positions": {
                    s: {"quantity": p.quantity, "value": p.quantity * p.current_price}
                    for s, p in self.positions.items()
                }
            })

        # Close any remaining positions at the end
        final_date = dates[-1]
        final_price = data.loc[final_date, "close"]
        if symbol in self.positions:
            self._execute_signal(
                timestamp=final_date,
                symbol=symbol,
                price=final_price,
                signal=Signal.SELL,
                confidence=1.0,
                reasoning="End of backtest - closing position"
            )

        # Build result
        equity_series = pd.Series(
            [e[1] for e in self.equity_history],
            index=[e[0] for e in self.equity_history]
        )
        returns_series = pd.Series(daily_returns, index=dates[:len(daily_returns)])

        return BacktestResult(
            start_date=dates[0].to_pydatetime() if hasattr(dates[0], 'to_pydatetime') else dates[0],
            end_date=dates[-1].to_pydatetime() if hasattr(dates[-1], 'to_pydatetime') else dates[-1],
            initial_capital=self.initial_capital,
            final_capital=self._get_equity(),
            trades=self.trades,
            daily_returns=returns_series,
            equity_curve=equity_series,
            positions_history=self.positions_history
        )


class BenchmarkComparison:
    """Compare backtest results against benchmarks."""

    @staticmethod
    def buy_and_hold(
        data: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> BacktestResult:
        """
        Calculate buy-and-hold benchmark.

        Args:
            data: OHLCV DataFrame
            initial_capital: Starting capital

        Returns:
            BacktestResult for buy-and-hold strategy
        """
        prices = data["close"]
        shares = initial_capital / prices.iloc[0]
        equity_curve = shares * prices
        daily_returns = prices.pct_change().dropna()

        return BacktestResult(
            start_date=data.index[0].to_pydatetime() if hasattr(data.index[0], 'to_pydatetime') else data.index[0],
            end_date=data.index[-1].to_pydatetime() if hasattr(data.index[-1], 'to_pydatetime') else data.index[-1],
            initial_capital=initial_capital,
            final_capital=equity_curve.iloc[-1],
            trades=[Trade(
                timestamp=data.index[0],
                symbol="BENCHMARK",
                action="BUY",
                price=prices.iloc[0],
                quantity=shares,
                signal=Signal.BUY,
                confidence=1.0,
                reasoning="Buy and hold"
            )],
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            positions_history=[]
        )

    @staticmethod
    def compare(
        strategy_result: BacktestResult,
        benchmark_result: BacktestResult
    ) -> Dict:
        """
        Compare strategy against benchmark.

        Returns:
            Comparison metrics
        """
        strategy_summary = strategy_result.summary()
        benchmark_summary = benchmark_result.summary()

        return {
            "strategy": strategy_summary,
            "benchmark": benchmark_summary,
            "excess_return": strategy_summary["total_return_pct"] - benchmark_summary["total_return_pct"],
            "sharpe_difference": strategy_summary["sharpe_ratio"] - benchmark_summary["sharpe_ratio"],
            "drawdown_improvement": benchmark_summary["max_drawdown_pct"] - strategy_summary["max_drawdown_pct"],
        }


if __name__ == "__main__":
    print("Backtest Demo\n" + "=" * 50)

    from .agents import (
        TechnicalAgent, FundamentalsAgent, SentimentAgent,
        BullAgent, BearAgent, RiskManagerAgent, TraderAgent
    )

    # Create mock data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=252, freq="B")

    # Create trending + mean reverting price series
    trend = np.linspace(0, 0.3, 252)
    noise = np.random.randn(252).cumsum() * 0.02
    mean_rev = np.sin(np.linspace(0, 8 * np.pi, 252)) * 0.1
    close = 100 * np.exp(trend + noise + mean_rev)

    data = pd.DataFrame({
        "open": close * (1 + np.random.randn(252) * 0.005),
        "high": close * (1 + abs(np.random.randn(252) * 0.01)),
        "low": close * (1 - abs(np.random.randn(252) * 0.01)),
        "close": close,
        "volume": np.random.randint(1e6, 1e8, 252)
    }, index=dates)

    # Create agents
    agents = [
        TechnicalAgent("Tech-1"),
        FundamentalsAgent("Fund-1"),
        SentimentAgent("Sent-1"),
        BullAgent("Bull-1"),
        BearAgent("Bear-1"),
        RiskManagerAgent("Risk-1"),
    ]
    trader = TraderAgent("Trader-1")

    # Run backtest
    print("\nRunning Multi-Agent Strategy Backtest...")
    backtester = MultiAgentBacktester(
        agents=agents,
        trader_agent=trader,
        initial_capital=100000,
        position_size_pct=0.2,
        transaction_cost_pct=0.001
    )

    result = backtester.run("DEMO", data, lookback=50, step=5)
    result.print_summary()

    print(f"\nNumber of trades: {len(result.trades)}")
    for trade in result.trades[:5]:
        print(f"  {trade.timestamp.date()}: {trade.action} @ ${trade.price:.2f}")

    # Compare to benchmark
    print("\nComparing to Buy & Hold...")
    benchmark = BenchmarkComparison.buy_and_hold(data, initial_capital=100000)
    comparison = BenchmarkComparison.compare(result, benchmark)

    print(f"\nExcess Return: {comparison['excess_return']:.2f}%")
    print(f"Sharpe Difference: {comparison['sharpe_difference']:.2f}")
    print(f"Drawdown Improvement: {comparison['drawdown_improvement']:.2f}%")
