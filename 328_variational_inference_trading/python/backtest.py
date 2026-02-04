"""
Backtesting Engine for Variational Inference Trading
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from config import BacktestConfig, StrategyConfig
from strategy import TradingSignal, SignalType, VariationalTradingStrategy
from vae_model import MarketVAE


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "OPEN"  # "OPEN", "CLOSED", "STOPPED"

    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Results from backtesting"""
    # Basic metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float

    # Time series
    equity_curve: pd.Series
    returns: pd.Series
    drawdown: pd.Series

    # Trades
    trades: List[Trade]

    # Additional metrics
    volatility: float
    avg_holding_time: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    def summary(self) -> str:
        """Generate summary string"""
        return f"""
Backtest Results Summary
========================
Total Return:     {self.total_return:.2%}
Sharpe Ratio:     {self.sharpe_ratio:.2f}
Sortino Ratio:    {self.sortino_ratio:.2f}
Max Drawdown:     {self.max_drawdown:.2%}
Calmar Ratio:     {self.calmar_ratio:.2f}

Trade Statistics:
  Total Trades:   {self.total_trades}
  Win Rate:       {self.win_rate:.2%}
  Profit Factor:  {self.profit_factor:.2f}
  Avg Trade PnL:  {self.avg_trade_pnl:.2%}
  Avg Winner:     {self.avg_winner:.2%}
  Avg Loser:      {self.avg_loser:.2%}
"""


class BacktestEngine:
    """
    Backtesting engine for variational trading strategy

    Features:
    - Realistic execution simulation
    - Commission and slippage modeling
    - Position tracking
    - Performance metrics calculation
    """

    def __init__(
        self,
        strategy: VariationalTradingStrategy,
        config: Optional[BacktestConfig] = None
    ):
        self.strategy = strategy
        self.config = config or BacktestConfig()

        # State
        self.cash = self.config.initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> size
        self.equity_history: List[Tuple[datetime, float]] = []
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None

    def reset(self):
        """Reset backtester state"""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.equity_history = []
        self.trades = []
        self.current_trade = None

    def run(
        self,
        features: np.ndarray,
        prices: pd.Series,
        timestamps: pd.DatetimeIndex,
        symbol: str = "BTC/USDT"
    ) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            features: Feature array (n_timesteps, seq_len, n_features)
            prices: Close prices series
            timestamps: Timestamps for each price
            symbol: Trading symbol

        Returns:
            BacktestResult with all metrics and trades
        """
        self.reset()
        n_timesteps = len(prices)

        logger.info(f"Starting backtest with {n_timesteps} timesteps")

        for i in range(len(features)):
            current_time = timestamps[i]
            current_price = prices.iloc[i]

            # Generate signal
            signal = self.strategy.generate_signal(features[i])

            # Update equity
            equity = self._calculate_equity(current_price)
            self.equity_history.append((current_time, equity))

            # Check existing position
            if self.current_trade is not None:
                # Check stop loss / take profit
                closed = self._check_stops(current_price, current_time)
                if not closed:
                    # Check for signal reversal
                    if self._should_close_position(signal):
                        self._close_position(current_price, current_time, "SIGNAL")

            # Open new position if no current trade and signal is actionable
            if self.current_trade is None and signal.signal_type != SignalType.HOLD:
                position_size = self.strategy.calculate_position_size(
                    signal, self._calculate_equity(current_price)
                )
                if position_size > 0:
                    self._open_position(
                        signal, current_price, current_time, symbol, position_size
                    )

        # Close any remaining position at the end
        if self.current_trade is not None:
            final_price = prices.iloc[-1]
            final_time = timestamps[-1]
            self._close_position(final_price, final_time, "END_OF_DATA")

        # Calculate results
        return self._calculate_results()

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity"""
        equity = self.cash

        if self.current_trade is not None:
            # Mark-to-market
            if self.current_trade.side == "LONG":
                unrealized_pnl = (current_price - self.current_trade.entry_price) * self.current_trade.size
            else:  # SHORT
                unrealized_pnl = (self.current_trade.entry_price - current_price) * self.current_trade.size
            equity += unrealized_pnl

        return equity

    def _open_position(
        self,
        signal: TradingSignal,
        price: float,
        timestamp: datetime,
        symbol: str,
        size_pct: float
    ):
        """Open a new position"""
        # Calculate position size in units
        equity = self.cash
        position_value = equity * size_pct

        # Apply commission
        commission = position_value * self.config.commission
        position_value -= commission

        # Apply slippage
        if signal.signal_type == SignalType.LONG:
            adjusted_price = price * (1 + self.config.slippage)
        else:
            adjusted_price = price * (1 - self.config.slippage)

        size = position_value / adjusted_price

        # Calculate stop loss and take profit
        stop_loss, take_profit = self.strategy.get_stop_loss_take_profit(
            signal, adjusted_price
        )

        self.current_trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            symbol=symbol,
            side=signal.signal_type.value,
            entry_price=adjusted_price,
            exit_price=None,
            size=size
        )

        # Store stop levels
        self._stop_loss = stop_loss
        self._take_profit = take_profit

        # Deduct from cash
        self.cash -= position_value + commission

        logger.debug(f"Opened {signal.signal_type.value} @ {adjusted_price:.2f}, size={size:.6f}")

    def _close_position(self, price: float, timestamp: datetime, reason: str):
        """Close current position"""
        if self.current_trade is None:
            return

        # Apply slippage
        if self.current_trade.side == "LONG":
            adjusted_price = price * (1 - self.config.slippage)
        else:
            adjusted_price = price * (1 + self.config.slippage)

        # Calculate PnL
        if self.current_trade.side == "LONG":
            pnl = (adjusted_price - self.current_trade.entry_price) * self.current_trade.size
        else:
            pnl = (self.current_trade.entry_price - adjusted_price) * self.current_trade.size

        # Apply exit commission
        position_value = adjusted_price * self.current_trade.size
        commission = position_value * self.config.commission
        pnl -= commission

        # Update trade
        self.current_trade.exit_time = timestamp
        self.current_trade.exit_price = adjusted_price
        self.current_trade.pnl = pnl
        self.current_trade.pnl_pct = pnl / (self.current_trade.entry_price * self.current_trade.size)
        self.current_trade.status = reason

        # Add to trades list
        self.trades.append(self.current_trade)

        # Update cash
        self.cash += position_value + pnl

        logger.debug(f"Closed {self.current_trade.side} @ {adjusted_price:.2f}, PnL={pnl:.2f}")

        self.current_trade = None

    def _check_stops(self, price: float, timestamp: datetime) -> bool:
        """Check stop loss and take profit"""
        if self.current_trade is None:
            return False

        if self.current_trade.side == "LONG":
            if price <= self._stop_loss:
                self._close_position(self._stop_loss, timestamp, "STOP_LOSS")
                return True
            if price >= self._take_profit:
                self._close_position(self._take_profit, timestamp, "TAKE_PROFIT")
                return True
        else:  # SHORT
            if price >= self._stop_loss:
                self._close_position(self._stop_loss, timestamp, "STOP_LOSS")
                return True
            if price <= self._take_profit:
                self._close_position(self._take_profit, timestamp, "TAKE_PROFIT")
                return True

        return False

    def _should_close_position(self, signal: TradingSignal) -> bool:
        """Check if we should close due to signal change"""
        if self.current_trade is None:
            return False

        if self.current_trade.side == "LONG" and signal.signal_type == SignalType.SHORT:
            return True
        if self.current_trade.side == "SHORT" and signal.signal_type == SignalType.LONG:
            return True

        return False

    def _calculate_results(self) -> BacktestResult:
        """Calculate all backtest metrics"""
        # Build equity curve
        if not self.equity_history:
            raise ValueError("No equity history available")

        timestamps, equities = zip(*self.equity_history)
        equity_curve = pd.Series(equities, index=timestamps)

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Calculate drawdown
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak

        # Basic metrics
        total_return = (equity_curve.iloc[-1] / self.config.initial_capital) - 1
        max_drawdown = drawdown.min()

        # Sharpe ratio (annualized, assuming 1-min data)
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(525600)  # 1-min bars per year
        else:
            sharpe_ratio = 0.0

        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(525600)
        else:
            sortino_ratio = sharpe_ratio

        # Calmar ratio
        if max_drawdown != 0:
            calmar_ratio = total_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0

        # Trade statistics
        if self.trades:
            pnl_list = [t.pnl_pct for t in self.trades]
            winning_trades = [t for t in self.trades if t.is_profitable]
            losing_trades = [t for t in self.trades if not t.is_profitable]

            total_trades = len(self.trades)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            avg_trade_pnl = np.mean(pnl_list) if pnl_list else 0
            avg_winner = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
            avg_loser = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0

            # Consecutive wins/losses
            consecutive = self._calculate_consecutive(self.trades)
        else:
            total_trades = 0
            win_rate = 0
            profit_factor = 0
            avg_trade_pnl = 0
            avg_winner = 0
            avg_loser = 0
            consecutive = (0, 0)
            winning_trades = []
            losing_trades = []

        # Average holding time
        if self.trades:
            holding_times = [
                (t.exit_time - t.entry_time).total_seconds() / 60
                for t in self.trades if t.exit_time is not None
            ]
            avg_holding_time = np.mean(holding_times) if holding_times else 0
        else:
            avg_holding_time = 0

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            equity_curve=equity_curve,
            returns=returns,
            drawdown=drawdown,
            trades=self.trades,
            volatility=returns.std() * np.sqrt(525600) if len(returns) > 0 else 0,
            avg_holding_time=avg_holding_time,
            max_consecutive_wins=consecutive[0],
            max_consecutive_losses=consecutive[1]
        )

    def _calculate_consecutive(self, trades: List[Trade]) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses"""
        if not trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.is_profitable:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses


def run_backtest(
    model: MarketVAE,
    features: np.ndarray,
    prices: pd.Series,
    timestamps: pd.DatetimeIndex,
    strategy_config: Optional[StrategyConfig] = None,
    backtest_config: Optional[BacktestConfig] = None,
    symbol: str = "BTC/USDT"
) -> BacktestResult:
    """
    Convenience function to run backtest

    Args:
        model: Trained VAE model
        features: Feature array
        prices: Price series
        timestamps: Timestamp index
        strategy_config: Strategy configuration
        backtest_config: Backtest configuration
        symbol: Trading symbol

    Returns:
        BacktestResult
    """
    strategy = VariationalTradingStrategy(model, strategy_config)
    engine = BacktestEngine(strategy, backtest_config)
    return engine.run(features, prices, timestamps, symbol)


if __name__ == "__main__":
    # Test backtest with synthetic data
    from vae_model import create_model, ModelConfig
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Create model
    model_config = ModelConfig(
        input_dim=13,
        latent_dim=8,
        hidden_dims=[64, 32]
    )
    model = create_model(model_config, sequence_length=60, device="cpu")

    # Generate synthetic data
    n_timesteps = 500
    seq_len = 60
    n_features = 13

    features = np.random.randn(n_timesteps, seq_len, n_features).astype(np.float32)

    # Generate random walk prices
    returns = np.random.randn(n_timesteps) * 0.001  # 0.1% per step
    prices = pd.Series(1000 * np.exp(np.cumsum(returns)))

    timestamps = pd.date_range(start='2024-01-01', periods=n_timesteps, freq='1min')

    # Run backtest
    result = run_backtest(
        model=model,
        features=features,
        prices=prices,
        timestamps=timestamps,
        symbol="BTC/USDT"
    )

    print(result.summary())
