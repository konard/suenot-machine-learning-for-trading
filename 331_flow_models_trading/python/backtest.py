#!/usr/bin/env python3
"""
Backtesting Module for Flow Models Trading

Provides comprehensive backtesting framework with realistic market simulation,
transaction costs, slippage modeling, and performance analytics.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from flow_model import NormalizingFlow
from data_fetcher import BybitDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for backtesting"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class PositionSide(Enum):
    """Position side"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Order:
    """Order representation"""
    timestamp: pd.Timestamp
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    size: float
    price: Optional[float] = None
    filled: bool = False
    fill_price: Optional[float] = None
    fill_timestamp: Optional[pd.Timestamp] = None


@dataclass
class Position:
    """Position tracking"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    position_size_pct: float = 0.1  # 10% of capital per trade
    max_position_size_pct: float = 0.5  # Max 50% in single position
    transaction_cost_bps: float = 10.0  # 10 bps = 0.1%
    slippage_bps: float = 5.0  # 5 bps slippage
    enable_shorting: bool = True
    risk_free_rate: float = 0.02  # 2% annual
    trading_days_per_year: int = 365  # Crypto trades 24/7


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    # Returns
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_returns: List[float] = field(default_factory=list)

    # Risk
    annual_volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_duration_days: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trading
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration_hours: float = 0.0

    # Final state
    final_equity: float = 0.0
    final_position: str = "FLAT"


class BacktestEngine:
    """
    Advanced backtesting engine for flow-based trading strategies.

    Features:
    - Realistic market simulation with slippage and transaction costs
    - Position tracking with P&L calculation
    - Comprehensive performance metrics
    - Support for long and short positions
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """Reset backtest state"""
        self.capital = self.config.initial_capital
        self.equity_curve: List[float] = []
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict] = []
        self.timestamps: List[pd.Timestamp] = []

    def run(
        self,
        features: torch.Tensor,
        prices: np.ndarray,
        timestamps: pd.DatetimeIndex,
        signal_generator: Callable[[torch.Tensor], Tuple[str, float]],
        symbol: str = "BTC/USDT"
    ) -> BacktestMetrics:
        """
        Run backtest on historical data

        Args:
            features: Feature tensor [N, feature_dim]
            prices: Price array [N]
            timestamps: Timestamp index
            signal_generator: Function that takes features and returns (signal, confidence)
            symbol: Trading symbol

        Returns:
            BacktestMetrics with performance results
        """
        self.reset()
        n_samples = len(features)

        logger.info(f"Starting backtest: {n_samples} samples, {symbol}")
        logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")

        for i in range(n_samples):
            current_features = features[i:i+1]
            current_price = prices[i]
            current_time = timestamps[i]

            # Generate signal
            signal, confidence = signal_generator(current_features)

            # Update unrealized P&L
            self._update_positions(current_price)

            # Execute trading logic
            self._process_signal(signal, confidence, current_price, current_time, symbol)

            # Record equity
            equity = self._calculate_equity(current_price)
            self.equity_curve.append(equity)
            self.timestamps.append(current_time)

        # Close any remaining positions
        if symbol in self.positions:
            self._close_position(symbol, prices[-1], timestamps[-1])

        # Compute metrics
        return self._compute_metrics()

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current total equity"""
        equity = self.capital
        for pos in self.positions.values():
            if pos.side == PositionSide.LONG:
                equity += pos.size * current_price
            elif pos.side == PositionSide.SHORT:
                equity += pos.size * (2 * pos.entry_price - current_price)
        return equity

    def _update_positions(self, current_price: float):
        """Update unrealized P&L for all positions"""
        for pos in self.positions.values():
            if pos.side == PositionSide.LONG:
                pos.unrealized_pnl = pos.size * (current_price - pos.entry_price)
            elif pos.side == PositionSide.SHORT:
                pos.unrealized_pnl = pos.size * (pos.entry_price - current_price)

    def _process_signal(
        self,
        signal: str,
        confidence: float,
        price: float,
        timestamp: pd.Timestamp,
        symbol: str
    ):
        """Process trading signal and execute orders"""

        current_position = self.positions.get(symbol)

        if signal == "REDUCE_EXPOSURE":
            # Close any existing position
            if current_position:
                self._close_position(symbol, price, timestamp)

        elif signal == "LONG":
            if current_position:
                if current_position.side == PositionSide.SHORT:
                    # Close short and go long
                    self._close_position(symbol, price, timestamp)
                    self._open_position(symbol, PositionSide.LONG, price, timestamp, confidence)
                # Already long, do nothing
            else:
                self._open_position(symbol, PositionSide.LONG, price, timestamp, confidence)

        elif signal == "SHORT":
            if not self.config.enable_shorting:
                return

            if current_position:
                if current_position.side == PositionSide.LONG:
                    # Close long and go short
                    self._close_position(symbol, price, timestamp)
                    self._open_position(symbol, PositionSide.SHORT, price, timestamp, confidence)
                # Already short, do nothing
            else:
                self._open_position(symbol, PositionSide.SHORT, price, timestamp, confidence)

        # NEUTRAL or HOLD - do nothing

    def _open_position(
        self,
        symbol: str,
        side: PositionSide,
        price: float,
        timestamp: pd.Timestamp,
        confidence: float
    ):
        """Open a new position"""
        # Apply slippage
        slippage = price * (self.config.slippage_bps / 10000)
        if side == PositionSide.LONG:
            fill_price = price + slippage
        else:
            fill_price = price - slippage

        # Calculate position size based on confidence
        position_value = self.capital * self.config.position_size_pct * confidence
        position_value = min(position_value, self.capital * self.config.max_position_size_pct)
        size = position_value / fill_price

        # Apply transaction cost
        cost = position_value * (self.config.transaction_cost_bps / 10000)
        self.capital -= cost

        # For long, deduct capital; for short, we're borrowing
        if side == PositionSide.LONG:
            self.capital -= position_value

        # Create position
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=fill_price,
            entry_timestamp=timestamp
        )

        logger.debug(f"Opened {side.value} position: {size:.6f} @ ${fill_price:.2f}")

    def _close_position(self, symbol: str, price: float, timestamp: pd.Timestamp):
        """Close an existing position"""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Apply slippage
        slippage = price * (self.config.slippage_bps / 10000)
        if pos.side == PositionSide.LONG:
            fill_price = price - slippage
        else:
            fill_price = price + slippage

        # Calculate P&L
        if pos.side == PositionSide.LONG:
            pnl = pos.size * (fill_price - pos.entry_price)
            proceeds = pos.size * fill_price
        else:
            pnl = pos.size * (pos.entry_price - fill_price)
            proceeds = pos.size * (2 * pos.entry_price - fill_price)

        # Apply transaction cost
        cost = abs(proceeds) * (self.config.transaction_cost_bps / 10000)
        pnl -= cost

        # Update capital
        if pos.side == PositionSide.LONG:
            self.capital += proceeds - cost
        else:
            self.capital += pnl

        # Record trade
        duration = (timestamp - pos.entry_timestamp).total_seconds() / 3600  # hours
        self.trades.append({
            'symbol': symbol,
            'side': pos.side.value,
            'entry_price': pos.entry_price,
            'exit_price': fill_price,
            'size': pos.size,
            'pnl': pnl,
            'duration_hours': duration,
            'entry_time': pos.entry_timestamp,
            'exit_time': timestamp
        })

        logger.debug(f"Closed {pos.side.value} position: PnL=${pnl:.2f}")

        # Remove position
        del self.positions[symbol]

    def _compute_metrics(self) -> BacktestMetrics:
        """Compute comprehensive backtest metrics"""
        metrics = BacktestMetrics()

        if len(self.equity_curve) < 2:
            return metrics

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Basic returns
        metrics.total_return = (equity[-1] - self.config.initial_capital) / self.config.initial_capital

        # Annualization
        n_days = (self.timestamps[-1] - self.timestamps[0]).days or 1
        annual_factor = self.config.trading_days_per_year / n_days

        metrics.annual_return = (1 + metrics.total_return) ** annual_factor - 1
        metrics.annual_volatility = returns.std() * np.sqrt(self.config.trading_days_per_year * 24) if len(returns) > 1 else 0

        # Downside volatility
        negative_returns = returns[returns < 0]
        metrics.downside_volatility = negative_returns.std() * np.sqrt(self.config.trading_days_per_year * 24) if len(negative_returns) > 0 else 0

        # Drawdown analysis
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        metrics.max_drawdown = drawdown.max()
        metrics.avg_drawdown = drawdown[drawdown > 0].mean() if (drawdown > 0).any() else 0

        # Find drawdown duration
        in_drawdown = drawdown > 0
        if in_drawdown.any():
            dd_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
            dd_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]
            if len(dd_starts) > 0 and len(dd_ends) > 0:
                durations = dd_ends[:len(dd_starts)] - dd_starts[:len(dd_ends)]
                metrics.drawdown_duration_days = durations.max() / 24 if len(durations) > 0 else 0

        # Risk-adjusted metrics
        excess_return = metrics.annual_return - self.config.risk_free_rate
        metrics.sharpe_ratio = excess_return / metrics.annual_volatility if metrics.annual_volatility > 0 else 0
        metrics.sortino_ratio = excess_return / metrics.downside_volatility if metrics.downside_volatility > 0 else 0
        metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown if metrics.max_drawdown > 0 else 0

        # Trade statistics
        if self.trades:
            metrics.num_trades = len(self.trades)
            pnls = [t['pnl'] for t in self.trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            metrics.win_rate = len(wins) / len(pnls) if pnls else 0
            metrics.avg_win = np.mean(wins) if wins else 0
            metrics.avg_loss = np.mean(losses) if losses else 0
            metrics.largest_win = max(wins) if wins else 0
            metrics.largest_loss = min(losses) if losses else 0

            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            metrics.avg_trade_duration_hours = np.mean([t['duration_hours'] for t in self.trades])

        # Final state
        metrics.final_equity = equity[-1]
        if self.positions:
            pos = list(self.positions.values())[0]
            metrics.final_position = pos.side.value
        else:
            metrics.final_position = "FLAT"

        return metrics

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'equity': self.equity_curve
        }).set_index('timestamp')

    def get_trades(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)


def print_backtest_report(metrics: BacktestMetrics, title: str = "BACKTEST REPORT"):
    """Print formatted backtest report"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

    print("\n--- RETURNS ---")
    print(f"Total Return:        {metrics.total_return * 100:>10.2f}%")
    print(f"Annual Return:       {metrics.annual_return * 100:>10.2f}%")
    print(f"Final Equity:        ${metrics.final_equity:>10,.2f}")

    print("\n--- RISK ---")
    print(f"Annual Volatility:   {metrics.annual_volatility * 100:>10.2f}%")
    print(f"Max Drawdown:        {metrics.max_drawdown * 100:>10.2f}%")
    print(f"Avg Drawdown:        {metrics.avg_drawdown * 100:>10.2f}%")

    print("\n--- RISK-ADJUSTED ---")
    print(f"Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:       {metrics.sortino_ratio:>10.2f}")
    print(f"Calmar Ratio:        {metrics.calmar_ratio:>10.2f}")

    print("\n--- TRADING ---")
    print(f"Number of Trades:    {metrics.num_trades:>10d}")
    print(f"Win Rate:            {metrics.win_rate * 100:>10.2f}%")
    print(f"Profit Factor:       {metrics.profit_factor:>10.2f}")
    print(f"Avg Win:             ${metrics.avg_win:>10.2f}")
    print(f"Avg Loss:            ${metrics.avg_loss:>10.2f}")
    print(f"Largest Win:         ${metrics.largest_win:>10.2f}")
    print(f"Largest Loss:        ${metrics.largest_loss:>10.2f}")
    print(f"Avg Trade Duration:  {metrics.avg_trade_duration_hours:>10.1f} hours")

    print("\n" + "=" * 60)


def main():
    """Example: Run backtest with flow model"""
    print("=" * 60)
    print("Flow Models Trading - Backtest Example")
    print("=" * 60)

    from flow_model import NormalizingFlow, FlowModelTrainer, AnomalyDetector, RegimeDetector

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic data
    print("\n1. Generating synthetic market data...")
    n_samples = 2000
    dim = 8

    # Synthetic features (mixture of regimes)
    data1 = np.random.multivariate_normal(
        mean=np.zeros(dim) + 0.002,
        cov=np.eye(dim) * 0.02,
        size=n_samples // 2
    )
    data2 = np.random.multivariate_normal(
        mean=np.zeros(dim) - 0.001,
        cov=np.eye(dim) * 0.03,
        size=n_samples // 2
    )
    features = np.vstack([data1, data2]).astype(np.float32)
    np.random.shuffle(features)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Synthetic prices
    returns = np.random.randn(n_samples) * 0.02 + 0.0002
    prices = 50000 * np.cumprod(1 + returns)
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')

    # Split data
    train_size = int(n_samples * 0.7)
    train_features = features_tensor[:train_size]
    test_features = features_tensor[train_size:]
    test_prices = prices[train_size:]
    test_timestamps = timestamps[train_size:]

    print(f"   Train samples: {train_size}")
    print(f"   Test samples: {n_samples - train_size}")

    # Train flow model
    print("\n2. Training flow model...")
    model = NormalizingFlow(input_dim=dim, num_layers=4, hidden_dim=64)
    trainer = FlowModelTrainer(model, learning_rate=1e-3)
    trainer.train(train_features, epochs=30, batch_size=64, patience=5)

    # Setup anomaly detector
    print("\n3. Setting up anomaly detector...")
    anomaly_detector = AnomalyDetector(model, threshold_percentile=5.0)
    anomaly_detector.fit(train_features)

    # Setup regime detector
    print("\n4. Setting up regime detector...")
    regime_detector = RegimeDetector(model, n_regimes=4)
    regime_detector.fit(train_features)

    # Create signal generator function
    def signal_generator(features: torch.Tensor) -> Tuple[str, float]:
        """Generate trading signal from features"""
        model.set_training_mode(False)

        with torch.no_grad():
            # Check for anomaly
            is_anomaly, log_probs = anomaly_detector.detect(features)
            if is_anomaly[0]:
                return "REDUCE_EXPOSURE", 0.9

            # Detect regime
            regimes, distances = regime_detector.detect(features)
            regime = regimes[0]
            confidence = 1.0 / (1.0 + distances[0, regime])

            # Simple strategy: regime 0,1 = bullish, regime 2,3 = bearish
            if regime in [0, 1] and confidence > 0.5:
                return "LONG", confidence
            elif regime in [2, 3] and confidence > 0.5:
                return "SHORT", confidence
            else:
                return "NEUTRAL", 0.5

    # Add helper method to model
    def set_training_mode(self, training: bool):
        if training:
            self.train()
        else:
            self.eval_mode()

    # Use eval() method directly
    model.eval_mode = model.eval

    # Run backtest
    print("\n5. Running backtest...")
    config = BacktestConfig(
        initial_capital=100000.0,
        position_size_pct=0.1,
        transaction_cost_bps=10.0,
        slippage_bps=5.0
    )

    engine = BacktestEngine(config)
    metrics = engine.run(
        features=test_features,
        prices=test_prices,
        timestamps=test_timestamps,
        signal_generator=signal_generator,
        symbol="BTC/USDT"
    )

    # Print report
    print_backtest_report(metrics, "FLOW MODEL BACKTEST RESULTS")

    # Get equity curve
    equity_df = engine.get_equity_curve()
    print(f"\nEquity curve samples:")
    print(equity_df.tail())

    # Get trades
    trades_df = engine.get_trades()
    if not trades_df.empty:
        print(f"\nRecent trades:")
        print(trades_df[['side', 'entry_price', 'exit_price', 'pnl', 'duration_hours']].tail())

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
