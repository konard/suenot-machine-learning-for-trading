"""
Chain-of-Thought Backtesting Engine

This module provides a backtesting engine that maintains full
audit trails of all reasoning chains for trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class TradeDirection(Enum):
    """Trade direction enum."""
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """A single trade with reasoning."""
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_time is not None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'direction': self.direction.name,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'entry_price': self.entry_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'size': self.size,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'reasoning_chain': self.reasoning_chain
        }


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000
    max_position_pct: float = 0.10
    risk_per_trade_pct: float = 0.02
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04


@dataclass
class BacktestResult:
    """Results from backtesting with full reasoning audit."""
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict[str, float]
    reasoning_audit: List[Dict]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trades': [t.to_dict() for t in self.trades],
            'equity_curve': self.equity_curve.to_dict(),
            'metrics': self.metrics,
            'reasoning_audit': self.reasoning_audit
        }


class CoTBacktester:
    """
    Backtest Chain-of-Thought trading strategies.

    This backtester maintains full audit trails of all reasoning
    chains that led to trading decisions.

    Attributes:
        config: Backtest configuration
        capital: Current capital
        positions: Open positions
        closed_trades: List of closed trades
        equity_history: Equity curve history
        reasoning_audit: Full audit trail

    Example:
        >>> from signal_generator import MultiStepSignalGenerator
        >>> backtester = CoTBacktester()
        >>> generator = MultiStepSignalGenerator()
        >>> result = backtester.run(prices, generator)
        >>> print(result.metrics)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration (uses defaults if not provided)
        """
        self.config = config or BacktestConfig()
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.reasoning_audit: List[Dict] = []

    def run(
        self,
        prices: pd.DataFrame,
        signal_generator,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            prices: DataFrame with OHLCV data, indexed by datetime
            signal_generator: Signal generator with generate_signal method
            start_date: Start of backtest period
            end_date: End of backtest period

        Returns:
            BacktestResult with trades, equity curve, and reasoning audit
        """
        # Reset state
        self.capital = self.config.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = []
        self.reasoning_audit = []

        # Filter date range
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]

        # Iterate through each bar
        for timestamp in prices.index:
            bar = prices.loc[timestamp]

            # Update equity
            equity = self._calculate_equity(bar)
            self.equity_history.append((timestamp, equity))

            # Check for exits on existing positions
            self._check_exits(timestamp, bar)

            # Generate signals and potentially enter new positions
            self._process_signals(timestamp, bar, signal_generator, prices)

        # Close any remaining positions
        if len(prices) > 0:
            final_bar = prices.iloc[-1]
            self._close_all_positions(prices.index[-1], final_bar)

        # Calculate metrics
        equity_series = pd.Series(
            [e[1] for e in self.equity_history],
            index=[e[0] for e in self.equity_history]
        )
        metrics = self._calculate_metrics(equity_series)

        return BacktestResult(
            trades=self.closed_trades,
            equity_curve=equity_series,
            metrics=metrics,
            reasoning_audit=self.reasoning_audit
        )

    def _calculate_equity(self, bar) -> float:
        """Calculate current equity including open positions."""
        equity = self.capital

        for symbol, trade in self.positions.items():
            current_price = bar['close'] if isinstance(bar, pd.Series) else bar
            if trade.direction == TradeDirection.LONG:
                unrealized_pnl = (current_price - trade.entry_price) * trade.size
            else:
                unrealized_pnl = (trade.entry_price - current_price) * trade.size
            equity += unrealized_pnl

        return equity

    def _check_exits(self, timestamp: datetime, bar):
        """Check if any positions should be exited."""
        symbols_to_close = []

        for symbol, trade in self.positions.items():
            current_price = bar['close']
            entry_return = (current_price - trade.entry_price) / trade.entry_price

            if trade.direction == TradeDirection.LONG:
                if entry_return <= -self.config.stop_loss_pct:
                    symbols_to_close.append((symbol, "Stop loss hit", current_price))
                elif entry_return >= self.config.take_profit_pct:
                    symbols_to_close.append((symbol, "Take profit hit", current_price))
            else:  # SHORT
                if entry_return >= self.config.stop_loss_pct:
                    symbols_to_close.append((symbol, "Stop loss hit (short)", current_price))
                elif entry_return <= -self.config.take_profit_pct:
                    symbols_to_close.append((symbol, "Take profit hit (short)", current_price))

        for symbol, reason, price in symbols_to_close:
            self._close_position(symbol, timestamp, price, reason)

    def _process_signals(
        self,
        timestamp: datetime,
        bar,
        signal_generator,
        prices: pd.DataFrame
    ):
        """Process signals and enter positions."""
        # Prepare data for signal generation
        price_data = {
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar.get('volume', 0),
            'prev_close': bar['close'] * 0.99,  # Simplified
            'avg_volume': bar.get('volume', 0)  # Simplified
        }

        # Calculate simple indicators
        close = bar['close']
        indicators = {
            'rsi': 50 + np.random.randn() * 15,
            'macd': np.random.randn() * 100,
            'macd_signal': np.random.randn() * 100,
            'sma_20': close * (1 + np.random.randn() * 0.02),
            'sma_50': close * (1 + np.random.randn() * 0.03),
            'sma_200': close * (1 + np.random.randn() * 0.05),
            'atr': close * 0.02
        }

        # Generate signal
        signal = signal_generator.generate_signal(
            "ASSET",
            price_data,
            indicators
        )

        # Record reasoning
        self.reasoning_audit.append({
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'signal': signal.signal.name,
            'confidence': signal.confidence,
            'reasoning_chain': signal.reasoning_chain
        })

        # Enter position if signal is strong enough and no existing position
        if signal.confidence >= 0.6 and "ASSET" not in self.positions:
            if signal.signal.value > 0:  # BUY or STRONG_BUY
                self._enter_position(
                    "ASSET",
                    TradeDirection.LONG,
                    timestamp,
                    bar['close'],
                    signal.stop_loss,
                    signal.reasoning_chain
                )
            elif signal.signal.value < 0:  # SELL or STRONG_SELL
                self._enter_position(
                    "ASSET",
                    TradeDirection.SHORT,
                    timestamp,
                    bar['close'],
                    signal.stop_loss,
                    signal.reasoning_chain
                )

    def _enter_position(
        self,
        symbol: str,
        direction: TradeDirection,
        timestamp: datetime,
        price: float,
        stop_loss: Optional[float],
        reasoning_chain: List[str]
    ):
        """Enter a new position."""
        # Calculate position size based on risk
        stop_distance = abs(price - stop_loss) if stop_loss else price * self.config.stop_loss_pct
        risk_amount = self.capital * self.config.risk_per_trade_pct
        size = risk_amount / stop_distance if stop_distance > 0 else 0

        # Cap at max position size
        max_size = (self.capital * self.config.max_position_pct) / price
        size = min(size, max_size)

        # Apply slippage
        if direction == TradeDirection.LONG:
            entry_price = price * (1 + self.config.slippage_pct)
        else:
            entry_price = price * (1 - self.config.slippage_pct)

        # Apply commission
        commission = size * entry_price * self.config.commission_pct
        self.capital -= commission

        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_time=timestamp,
            entry_price=entry_price,
            size=size,
            reasoning_chain=reasoning_chain.copy()
        )

        self.positions[symbol] = trade

        self.reasoning_audit.append({
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'action': 'ENTRY',
            'symbol': symbol,
            'direction': direction.name,
            'price': entry_price,
            'size': size,
            'reasoning': reasoning_chain[-1] if reasoning_chain else "No reasoning"
        })

    def _close_position(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        reason: str
    ):
        """Close an existing position."""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]

        # Apply slippage
        if trade.direction == TradeDirection.LONG:
            exit_price = price * (1 - self.config.slippage_pct)
        else:
            exit_price = price * (1 + self.config.slippage_pct)

        trade.exit_time = timestamp
        trade.exit_price = exit_price

        # Calculate PnL
        if trade.direction == TradeDirection.LONG:
            trade.pnl = (exit_price - trade.entry_price) * trade.size
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.size

        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.size) if trade.size > 0 else 0

        # Apply commission
        commission = trade.size * exit_price * self.config.commission_pct
        trade.pnl -= commission

        # Update capital
        self.capital += trade.pnl

        # Add exit reasoning
        trade.reasoning_chain.append(f"EXIT: {reason} at ${exit_price:.2f}")

        self.closed_trades.append(trade)
        del self.positions[symbol]

        self.reasoning_audit.append({
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'action': 'EXIT',
            'symbol': symbol,
            'price': exit_price,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct,
            'reason': reason
        })

    def _close_all_positions(self, timestamp: datetime, bar):
        """Close all remaining positions."""
        for symbol in list(self.positions.keys()):
            self._close_position(symbol, timestamp, bar['close'], "End of backtest")

    def _calculate_metrics(self, equity: pd.Series) -> Dict[str, float]:
        """Calculate backtest performance metrics."""
        if len(equity) < 2:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'final_equity': self.config.initial_capital
            }

        returns = equity.pct_change().dropna()

        # Basic metrics
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1 if equity.iloc[0] > 0 else 0

        # Risk metrics
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Trade statistics
        if self.closed_trades:
            winning_trades = [t for t in self.closed_trades if t.pnl > 0]
            losing_trades = [t for t in self.closed_trades if t.pnl <= 0]

            win_rate = len(winning_trades) / len(self.closed_trades)

            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.closed_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': equity.iloc[-1]
        }


def generate_mock_prices(
    start_date: str = "2024-01-01",
    periods: int = 252,
    initial_price: float = 40000,
    volatility: float = 0.02
) -> pd.DataFrame:
    """
    Generate mock price data for backtesting.

    Args:
        start_date: Start date string
        periods: Number of periods
        initial_price: Starting price
        volatility: Daily volatility

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=periods, freq='D')

    # Generate price path
    returns = np.random.randn(periods) * volatility
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(periods) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(periods) * 0.015)),
        'low': prices * (1 - np.abs(np.random.randn(periods) * 0.015)),
        'close': prices,
        'volume': np.random.randint(1e9, 5e10, periods)
    }, index=dates)

    return df


if __name__ == "__main__":
    from signal_generator import MultiStepSignalGenerator

    print("Chain-of-Thought Backtesting Demo")
    print("=" * 50)

    # Generate mock data
    prices = generate_mock_prices()

    # Initialize components
    signal_generator = MultiStepSignalGenerator()
    backtester = CoTBacktester(BacktestConfig(
        initial_capital=100000,
        max_position_pct=0.10,
        risk_per_trade_pct=0.02
    ))

    # Run backtest
    result = backtester.run(prices, signal_generator)

    # Print results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)

    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {result.metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"  Number of Trades: {result.metrics['num_trades']}")
    print(f"  Win Rate: {result.metrics['win_rate']:.1%}")
    print(f"  Profit Factor: {result.metrics['profit_factor']:.2f}")
    print(f"  Final Equity: ${result.metrics['final_equity']:,.2f}")

    if result.trades:
        print(f"\nSample Trade Reasoning (first 3 trades):")
        for i, trade in enumerate(result.trades[:3], 1):
            print(f"\n  Trade {i}: {trade.symbol}")
            print(f"    Direction: {trade.direction.name}")
            print(f"    Entry: ${trade.entry_price:.2f} at {trade.entry_time}")
            print(f"    Exit: ${trade.exit_price:.2f} at {trade.exit_time}")
            print(f"    PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2%})")
            if trade.reasoning_chain:
                print(f"    Reasoning: {trade.reasoning_chain[-1][:80]}...")

    print(f"\nReasoning Audit: {len(result.reasoning_audit)} entries recorded")
    print("Full audit trail available for compliance and review.")
