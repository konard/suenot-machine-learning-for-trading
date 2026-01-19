"""
Trading Strategy and Backtesting for FNet

This module provides:
- FNetTradingStrategy: Signal generation from FNet predictions
- Backtester: Complete backtesting engine with performance metrics
- Risk management utilities
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    LONG = 1
    HOLD = 0
    SHORT = -1


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    position: float  # Positive for long, negative for short
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Complete backtest results."""
    equity_curve: np.ndarray
    positions: np.ndarray
    trades: List[Trade]
    metrics: Dict[str, float]


class FNetTradingStrategy:
    """
    Trading strategy based on FNet predictions.

    Uses model predictions to generate trading signals with
    configurable thresholds and risk management.
    """

    def __init__(
        self,
        model,
        threshold: float = 0.001,
        confidence_threshold: float = 0.5,
        position_size: float = 1.0,
        stop_loss: float = 0.02,
        take_profit: float = 0.04,
        max_holding_period: int = 24
    ):
        """
        Args:
            model: Trained FNet model
            threshold: Minimum prediction magnitude for signal
            confidence_threshold: Minimum confidence for trading
            position_size: Position size as fraction of capital
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            max_holding_period: Maximum bars to hold position
        """
        self.model = model
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_holding_period = max_holding_period
        self.device = next(model.parameters()).device

    def generate_signal(
        self,
        x: torch.Tensor
    ) -> Tuple[SignalType, float, float]:
        """
        Generate trading signal from model prediction.

        Args:
            x: Input features [1, seq_len, n_features]

        Returns:
            signal: Trading signal (LONG, HOLD, SHORT)
            confidence: Prediction confidence [0, 1]
            prediction: Raw prediction value
        """
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            prediction, freq_maps = self.model(x, return_frequencies=True)
            pred_value = prediction.item()

            # Calculate confidence from frequency stability
            confidence = self._calculate_confidence(freq_maps)

        # Generate signal
        if pred_value > self.threshold and confidence > self.confidence_threshold:
            return SignalType.LONG, confidence, pred_value
        elif pred_value < -self.threshold and confidence > self.confidence_threshold:
            return SignalType.SHORT, confidence, pred_value
        else:
            return SignalType.HOLD, confidence, pred_value

    def _calculate_confidence(self, freq_maps: List[torch.Tensor]) -> float:
        """
        Calculate prediction confidence from frequency analysis.

        High confidence = stable, concentrated frequency patterns
        Low confidence = noisy, dispersed patterns
        """
        if not freq_maps:
            return 0.5

        confidences = []
        for freq_map in freq_maps:
            # Get magnitude spectrum
            mag = torch.abs(freq_map)

            # Calculate energy concentration in low frequencies
            total_energy = mag.sum()
            low_freq_energy = mag[:, :mag.shape[1]//4, :mag.shape[2]//4].sum()

            # Ratio of low-frequency energy (indicates smooth, predictable patterns)
            ratio = (low_freq_energy / (total_energy + 1e-8)).item()
            confidences.append(ratio)

        return np.mean(confidences)

    def generate_signals_batch(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate signals for a batch of inputs.

        Args:
            X: Input features [n_samples, seq_len, n_features]

        Returns:
            signals: Array of signals (-1, 0, 1)
            confidences: Array of confidence scores
            predictions: Array of raw predictions
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        signals = []
        confidences = []
        predictions = []

        with torch.no_grad():
            for i in range(len(X)):
                signal, conf, pred = self.generate_signal(X_tensor[i:i+1])
                signals.append(signal.value)
                confidences.append(conf)
                predictions.append(pred)

        return np.array(signals), np.array(confidences), np.array(predictions)


class Backtester:
    """
    Backtesting engine for FNet trading strategies.

    Features:
    - Realistic transaction costs and slippage
    - Stop-loss and take-profit execution
    - Comprehensive performance metrics
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        leverage: float = 1.0
    ):
        """
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
            leverage: Maximum leverage allowed
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.leverage = leverage

    def run(
        self,
        strategy: FNetTradingStrategy,
        X: np.ndarray,
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            strategy: Trading strategy instance
            X: Feature sequences [n_samples, seq_len, n_features]
            prices: Close prices for each sample
            timestamps: Optional timestamps

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        n_samples = len(X)

        # Initialize tracking
        capital = self.initial_capital
        position = 0.0  # Current position size
        entry_price = 0.0
        entry_time = 0
        holding_time = 0

        trades = []
        equity_curve = [capital]
        positions_history = [0]

        for i in range(n_samples):
            current_price = prices[i]
            x_tensor = torch.FloatTensor(X[i:i+1])

            # Get signal
            signal, confidence, pred = strategy.generate_signal(x_tensor)

            # Check exit conditions for existing position
            if position != 0:
                holding_time += 1
                pnl_pct = (current_price / entry_price - 1) * np.sign(position)

                exit_reason = None

                # Stop-loss check
                if pnl_pct <= -strategy.stop_loss:
                    exit_reason = "stop_loss"

                # Take-profit check
                elif pnl_pct >= strategy.take_profit:
                    exit_reason = "take_profit"

                # Max holding period check
                elif holding_time >= strategy.max_holding_period:
                    exit_reason = "max_holding"

                # Signal reversal check
                elif signal != SignalType.HOLD and signal.value != np.sign(position):
                    exit_reason = "signal_reversal"

                if exit_reason:
                    # Close position
                    exit_price = current_price * (1 - np.sign(position) * self.slippage)
                    pnl = capital * abs(position) * pnl_pct
                    cost = abs(capital * position * self.transaction_cost)

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=i,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position=position,
                        pnl=pnl - cost,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason
                    ))

                    capital += pnl - cost
                    position = 0
                    holding_time = 0

            # Check entry conditions (only if not already in position)
            if position == 0 and signal != SignalType.HOLD:
                # Enter new position
                position = signal.value * strategy.position_size
                entry_price = current_price * (1 + signal.value * self.slippage)
                entry_time = i
                holding_time = 0

                # Deduct entry cost
                capital -= abs(capital * position * self.transaction_cost)

            # Calculate current equity
            if position != 0:
                unrealized_pnl = capital * position * (current_price / entry_price - 1)
                equity = capital + unrealized_pnl
            else:
                equity = capital

            equity_curve.append(equity)
            positions_history.append(position)

        # Close final position if open
        if position != 0:
            exit_price = prices[-1] * (1 - np.sign(position) * self.slippage)
            pnl_pct = (exit_price / entry_price - 1) * np.sign(position)
            pnl = capital * abs(position) * pnl_pct
            cost = abs(capital * position * self.transaction_cost)

            trades.append(Trade(
                entry_time=entry_time,
                exit_time=n_samples - 1,
                entry_price=entry_price,
                exit_price=exit_price,
                position=position,
                pnl=pnl - cost,
                pnl_pct=pnl_pct,
                exit_reason="end_of_backtest"
            ))
            capital += pnl - cost

        equity_curve = np.array(equity_curve)
        positions_history = np.array(positions_history)

        # Calculate metrics
        metrics = calculate_metrics(equity_curve, trades, self.initial_capital)

        return BacktestResult(
            equity_curve=equity_curve,
            positions=positions_history,
            trades=trades,
            metrics=metrics
        )


def calculate_metrics(
    equity_curve: np.ndarray,
    trades: List[Trade],
    initial_capital: float,
    periods_per_year: int = 24 * 365  # Hourly data
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: Array of portfolio values over time
        trades: List of executed trades
        initial_capital: Starting capital
        periods_per_year: Number of periods per year (for annualization)

    Returns:
        Dictionary of performance metrics
    """
    # Basic returns
    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    returns = np.diff(equity_curve) / equity_curve[:-1]

    # Sharpe Ratio
    if len(returns) > 1 and returns.std() > 0:
        sharpe = np.sqrt(periods_per_year) * returns.mean() / returns.std()
    else:
        sharpe = 0.0

    # Sortino Ratio (using downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std()
        sortino = np.sqrt(periods_per_year) * returns.mean() / (downside_std + 1e-8)
    else:
        sortino = np.inf if returns.mean() > 0 else 0.0

    # Maximum Drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (cummax - equity_curve) / cummax
    max_drawdown = drawdown.max()

    # Calmar Ratio
    calmar = total_return / (max_drawdown + 1e-8)

    # Trade statistics
    if trades:
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(trades)

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / (gross_loss + 1e-8)

        # Average trade
        avg_trade_pnl = np.mean([t.pnl for t in trades])
        avg_trade_pct = np.mean([t.pnl_pct for t in trades])

        # Average holding period
        avg_holding = np.mean([t.exit_time - t.entry_time for t in trades])

        # Largest win/loss
        largest_win = max([t.pnl for t in trades]) if trades else 0
        largest_loss = min([t.pnl for t in trades]) if trades else 0

    else:
        win_rate = 0
        profit_factor = 0
        avg_trade_pnl = 0
        avg_trade_pct = 0
        avg_holding = 0
        largest_win = 0
        largest_loss = 0

    return {
        "total_return": total_return,
        "annualized_return": (1 + total_return) ** (periods_per_year / len(equity_curve)) - 1,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": len(trades),
        "avg_trade_pnl": avg_trade_pnl,
        "avg_trade_pct": avg_trade_pct,
        "avg_holding_period": avg_holding,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "final_capital": equity_curve[-1],
        "volatility": returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0
    }


def print_backtest_results(result: BacktestResult) -> None:
    """Print formatted backtest results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    # Format metrics
    metrics = result.metrics

    print(f"\nReturns:")
    print(f"  Total Return:        {metrics['total_return']*100:>10.2f}%")
    print(f"  Annualized Return:   {metrics['annualized_return']*100:>10.2f}%")
    print(f"  Final Capital:       ${metrics['final_capital']:>12,.2f}")

    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
    print(f"  Maximum Drawdown:    {metrics['max_drawdown']*100:>10.2f}%")
    print(f"  Volatility (ann.):   {metrics['volatility']*100:>10.2f}%")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")

    print(f"\nTrade Statistics:")
    print(f"  Number of Trades:    {metrics['n_trades']:>10}")
    print(f"  Win Rate:            {metrics['win_rate']*100:>10.2f}%")
    print(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")
    print(f"  Avg Trade PnL:       ${metrics['avg_trade_pnl']:>12,.2f}")
    print(f"  Avg Trade Return:    {metrics['avg_trade_pct']*100:>10.2f}%")
    print(f"  Avg Holding Period:  {metrics['avg_holding_period']:>10.1f} bars")

    print(f"\nBest/Worst:")
    print(f"  Largest Win:         ${metrics['largest_win']:>12,.2f}")
    print(f"  Largest Loss:        ${metrics['largest_loss']:>12,.2f}")

    print("=" * 60 + "\n")


def plot_results(result: BacktestResult, save_path: Optional[str] = None) -> None:
    """
    Plot backtest results.

    Args:
        result: BacktestResult object
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Equity curve
        ax1 = axes[0]
        ax1.plot(result.equity_curve, label='Portfolio Value', color='blue')
        ax1.axhline(y=result.equity_curve[0], color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('FNet Strategy Backtest Results')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        cummax = np.maximum.accumulate(result.equity_curve)
        drawdown = (cummax - result.equity_curve) / cummax * 100
        ax2.fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # Positions
        ax3 = axes[2]
        ax3.plot(result.positions, label='Position', color='green', alpha=0.7)
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Time Step')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("matplotlib not available for plotting")


if __name__ == "__main__":
    # Test backtester with dummy data
    print("Testing backtester...")

    # Create dummy model (mock)
    class MockModel:
        def __init__(self):
            self.device = torch.device('cpu')

        def parameters(self):
            return iter([torch.zeros(1)])

        def eval(self):
            pass

        def __call__(self, x, return_frequencies=False):
            pred = torch.randn(1, 1) * 0.01
            if return_frequencies:
                freq = [torch.randn(1, 10, 10)]
                return pred, freq
            return pred

    model = MockModel()
    strategy = FNetTradingStrategy(
        model=model,
        threshold=0.001,
        position_size=1.0,
        stop_loss=0.02,
        take_profit=0.04
    )

    # Create dummy data
    n_samples = 500
    seq_len = 168
    n_features = 7

    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    prices = 100 * np.cumprod(1 + np.random.randn(n_samples) * 0.01)

    # Run backtest
    backtester = Backtester(initial_capital=100000)
    result = backtester.run(strategy, X, prices)

    # Print results
    print_backtest_results(result)

    print("Backtester test completed!")
