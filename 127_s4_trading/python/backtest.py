"""
Backtesting engine for S4 trading strategies.

Provides:
- Configurable backtesting engine
- Performance metrics calculation
- Strategy evaluation tools
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging

from .data_loader import MarketData, generate_synthetic_data
from .s4_model import S4TradingModel, S4Config, create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    trading_fee: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    min_confidence: float = 0.6
    max_position_size: float = 0.1
    stop_loss: float = 0.02
    take_profit: float = 0.04


@dataclass
class BacktestResult:
    """Results from backtesting."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    final_equity: float
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)

    def summary(self) -> str:
        """Return formatted summary of results."""
        return f"""
Backtest Results:
-----------------
Total Return:    {self.total_return * 100:.2f}%
Annual Return:   {self.annual_return * 100:.2f}%
Sharpe Ratio:    {self.sharpe_ratio:.2f}
Sortino Ratio:   {self.sortino_ratio:.2f}
Max Drawdown:    {self.max_drawdown * 100:.2f}%
Calmar Ratio:    {self.calmar_ratio:.2f}
Win Rate:        {self.win_rate * 100:.2f}%
Profit Factor:   {self.profit_factor:.2f}
Total Trades:    {self.total_trades}
Final Equity:    ${self.final_equity:,.2f}
"""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "final_equity": self.final_equity,
        }


class BacktestEngine:
    """
    Backtesting engine for S4 trading strategies.

    Simulates trading based on S4 model signals with realistic
    execution assumptions (fees, slippage).
    """

    def __init__(
        self,
        model: Optional[S4TradingModel] = None,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            model: S4 trading model (created if None)
            config: Backtest configuration (default if None)
        """
        self.model = model or create_model(S4Config())
        self.config = config or BacktestConfig()

    def run(
        self,
        data: MarketData,
        lookback: int = 256,
        verbose: bool = False,
    ) -> BacktestResult:
        """
        Run backtest on market data.

        Args:
            data: Market data for backtesting
            lookback: Number of periods for model input
            verbose: Whether to print trade details

        Returns:
            BacktestResult with performance metrics
        """
        features = data.to_features()
        closes = data.close
        n = len(closes)

        if n < lookback + 2:
            return self._empty_result()

        # Initialize state
        equity = self.config.initial_capital
        equity_curve = [equity]
        position = 0.0
        entry_price = 0.0
        trades = []
        trade_pnls = []

        for i in range(lookback, n):
            current_price = closes[i]
            prev_price = closes[i - 1]

            # Update equity based on position
            if abs(position) > 0.001:
                position_return = (current_price - prev_price) / prev_price
                position_pnl = equity * position * position_return
                equity += position_pnl

            equity_curve.append(equity)

            # Check stop loss / take profit
            if abs(position) > 0.001 and entry_price > 0:
                pnl_pct = (current_price - entry_price) / entry_price * np.sign(position)

                if pnl_pct <= -self.config.stop_loss or pnl_pct >= self.config.take_profit:
                    # Close position
                    trade_cost = equity * abs(position) * (self.config.trading_fee + self.config.slippage)
                    equity -= trade_cost

                    trade_pnl = pnl_pct * abs(position)
                    trade_pnls.append(trade_pnl)

                    trades.append({
                        "type": "CLOSE",
                        "price": current_price,
                        "position": position,
                        "pnl": trade_pnl,
                        "reason": "SL/TP",
                    })

                    if verbose:
                        reason = "Stop Loss" if pnl_pct <= -self.config.stop_loss else "Take Profit"
                        logger.info(f"t={i}: {reason} - Closed position at {current_price:.2f}, PnL: {trade_pnl*100:.2f}%")

                    position = 0.0
                    entry_price = 0.0
                    continue

            # Get model prediction
            input_features = features[i - lookback:i]
            prediction = self.model.predict(input_features)

            signal = prediction["signal"]
            confidence = prediction["confidence"]

            # Calculate target position
            if confidence < self.config.min_confidence:
                target_position = 0.0
            elif signal == "BUY":
                target_position = self.config.max_position_size * (confidence - self.config.min_confidence) / (1 - self.config.min_confidence)
            elif signal == "SELL":
                target_position = -self.config.max_position_size * (confidence - self.config.min_confidence) / (1 - self.config.min_confidence)
            else:
                target_position = 0.0

            # Execute position change
            position_change = target_position - position

            if abs(position_change) > 0.001:
                # Apply trading costs
                trade_cost = equity * abs(position_change) * (self.config.trading_fee + self.config.slippage)
                equity -= trade_cost

                # Record trade
                if abs(position) > 0.001 and np.sign(position_change) != np.sign(position):
                    # Closing trade
                    pnl_pct = (current_price - entry_price) / entry_price * np.sign(position)
                    trade_pnl = pnl_pct * abs(position)
                    trade_pnls.append(trade_pnl)

                    trades.append({
                        "type": "CLOSE",
                        "price": current_price,
                        "position": position,
                        "pnl": trade_pnl,
                        "reason": signal,
                    })

                old_position = position
                position += position_change

                if abs(position) > 0.001 and abs(old_position) < 0.001:
                    # Opening new trade
                    entry_price = current_price
                    trades.append({
                        "type": "OPEN",
                        "signal": signal,
                        "price": current_price,
                        "position": position,
                        "confidence": confidence,
                    })

                    if verbose:
                        logger.info(f"t={i}: {signal} at {current_price:.2f}, confidence={confidence:.2f}, position={position:.4f}")

        return self._calculate_metrics(equity_curve, trade_pnls, trades)

    def _calculate_metrics(
        self,
        equity_curve: List[float],
        trade_pnls: List[float],
        trades: List[Dict],
    ) -> BacktestResult:
        """Calculate performance metrics."""
        initial = self.config.initial_capital
        final_equity = equity_curve[-1] if equity_curve else initial

        # Returns
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]

        # Total return
        total_return = (final_equity - initial) / initial

        # Annualize (assuming hourly data)
        n_periods = len(returns) if len(returns) > 0 else 1
        periods_per_year = 8760.0
        annual_factor = periods_per_year / n_periods
        annual_return = (1 + total_return) ** annual_factor - 1

        # Sharpe ratio
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1) + 1e-8
            sharpe_ratio = mean_return / std_return * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns, ddof=1) + 1e-8
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(periods_per_year)
        else:
            sortino_ratio = sharpe_ratio

        # Maximum drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_drawdown = np.max(drawdown)

        # Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0.001 else 0.0

        # Trade statistics
        trade_pnls_array = np.array(trade_pnls) if trade_pnls else np.array([0.0])
        winning_trades = trade_pnls_array[trade_pnls_array > 0]
        losing_trades = trade_pnls_array[trade_pnls_array < 0]

        win_rate = len(winning_trades) / len(trade_pnls_array) if len(trade_pnls_array) > 0 else 0.0

        gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        gross_loss = np.abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0.001 else (float('inf') if gross_profit > 0 else 0.0)

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            final_equity=final_equity,
            equity_curve=equity_curve,
            trades=trades,
        )

    def _empty_result(self) -> BacktestResult:
        """Return empty result for insufficient data."""
        return BacktestResult(
            total_return=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            final_equity=self.config.initial_capital,
            equity_curve=[self.config.initial_capital],
            trades=[],
        )


def run_backtest_demo():
    """Run a demonstration backtest."""
    print("S4 Trading Backtest Demo")
    print("=" * 50)

    # Generate test data
    print("\n1. Generating synthetic market data...")
    data = generate_synthetic_data("BTCUSDT", "60", n_points=2000, seed=42)
    print(f"   Generated {len(data.df)} data points")

    # Create model and engine
    print("\n2. Creating S4 model and backtest engine...")
    model_config = S4Config(d_model=64, d_state=64, n_layers=4)
    model = create_model(model_config)

    backtest_config = BacktestConfig(
        initial_capital=100_000.0,
        trading_fee=0.001,
        slippage=0.0005,
        min_confidence=0.6,
        max_position_size=0.1,
    )

    engine = BacktestEngine(model=model, config=backtest_config)

    # Run backtest
    print("\n3. Running backtest...")
    result = engine.run(data, lookback=256, verbose=False)

    # Print results
    print("\n4. Results:")
    print(result.summary())

    # Additional analysis
    print("5. Additional Analysis:")
    print(f"   Equity curve length: {len(result.equity_curve)}")
    print(f"   Starting equity: ${result.equity_curve[0]:,.2f}")
    print(f"   Ending equity: ${result.equity_curve[-1]:,.2f}")
    print(f"   Peak equity: ${max(result.equity_curve):,.2f}")
    print(f"   Trough equity: ${min(result.equity_curve):,.2f}")

    print("\n" + "=" * 50)
    print("Backtest demo complete!")

    return result


if __name__ == "__main__":
    run_backtest_demo()
