"""
Backtesting Module for Operator Learning Trading Strategies

Implements backtesting for trading strategies based on neural operators:
- DeepONet-based price prediction strategy
- FNO-based regime detection strategy
- Order book operator strategy for crypto (Bybit)
- Yield curve operator strategy for fixed income

Metrics: Sharpe ratio, max drawdown, total return, win rate, etc.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Strategy parameters
    initial_capital: float = 100_000.0
    position_size: float = 0.1  # fraction of capital per trade
    max_positions: int = 1
    transaction_cost_bps: float = 10.0  # basis points per trade

    # Signal thresholds
    buy_threshold: float = 0.005   # predicted return above this -> buy
    sell_threshold: float = -0.005  # predicted return below this -> sell

    # Risk management
    stop_loss_pct: float = 0.02    # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_drawdown_pct: float = 0.15  # halt trading at 15% drawdown

    # Operator model settings
    lookback_window: int = 60
    prediction_horizon: int = 5
    model_type: str = "deeponet"  # 'deeponet' or 'fno'


@dataclass
class TradeRecord:
    """Record of a single trade."""
    entry_time: int
    exit_time: int
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit', 'end'


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_trade_pnl: float
    profit_factor: float
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    trades: List[TradeRecord]
    signals: np.ndarray
    daily_returns: np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
# Core Backtesting Engine
# ═══════════════════════════════════════════════════════════════════════════════

class OperatorBacktester:
    """
    Backtesting engine for operator learning trading strategies.

    The backtester feeds historical data through a trained neural operator,
    generates trading signals from predictions, and simulates execution.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.trades: List[TradeRecord] = []

    def generate_signals_deeponet(
        self,
        model,
        prices: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Generate trading signals using a trained DeepONet model.

        The model predicts the future price function from the recent price
        curve. The signal is derived from the predicted return.

        Args:
            model: Trained DeepONet model
            prices: Full price series (T,)
            device: Torch device

        Returns:
            signals: Array of signals (+1, 0, -1) aligned with prices
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        cfg = self.config
        n = len(prices)
        signals = np.zeros(n)

        model.eval()
        eval_point = torch.tensor([[0.5, 0.5]], dtype=torch.float32).to(device)

        with torch.no_grad():
            for i in range(cfg.lookback_window, n - cfg.prediction_horizon):
                window = prices[i - cfg.lookback_window : i]
                w_min, w_max = window.min(), window.max()
                if w_max - w_min < 1e-10:
                    continue

                norm_window = (window - w_min) / (w_max - w_min)
                branch_input = torch.tensor(
                    norm_window, dtype=torch.float32
                ).unsqueeze(0).to(device)

                # Predict mid-point of future
                pred_norm = model(branch_input, eval_point).item()
                pred_price = pred_norm * (w_max - w_min) + w_min
                current_price = prices[i]
                predicted_return = (pred_price - current_price) / current_price

                if predicted_return > cfg.buy_threshold:
                    signals[i] = 1.0
                elif predicted_return < cfg.sell_threshold:
                    signals[i] = -1.0

        return signals

    def generate_signals_fno(
        self,
        model,
        prices: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Generate trading signals using a trained FNO model.

        The FNO predicts the entire future price curve. The signal is
        based on whether the predicted curve trends up or down.

        Args:
            model: Trained FNO model
            prices: Full price series (T,)
            device: Torch device

        Returns:
            signals: Array of signals aligned with prices
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        cfg = self.config
        n = len(prices)
        signals = np.zeros(n)
        grid_size = 128  # FNO grid resolution

        model.eval()
        with torch.no_grad():
            for i in range(grid_size, n - cfg.prediction_horizon):
                window = prices[i - grid_size : i]
                w_min, w_max = window.min(), window.max()
                if w_max - w_min < 1e-10:
                    continue

                norm_window = (window - w_min) / (w_max - w_min)
                x = torch.tensor(
                    norm_window, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, grid_size)

                pred = model(x).squeeze().cpu().numpy()

                # Predicted trend: compare mean of second half to first half
                mid = len(pred) // 2
                trend = np.mean(pred[mid:]) - np.mean(pred[:mid])

                if trend > cfg.buy_threshold:
                    signals[i] = 1.0
                elif trend < cfg.sell_threshold:
                    signals[i] = -1.0

        return signals

    def generate_signals_simple(
        self,
        prices: np.ndarray,
        predicted_returns: np.ndarray,
    ) -> np.ndarray:
        """
        Generate signals from pre-computed predicted returns (no model needed).

        Args:
            prices: Price series
            predicted_returns: Predicted returns at each time step

        Returns:
            signals: +1 / 0 / -1 signal array
        """
        cfg = self.config
        signals = np.zeros(len(prices))
        n = min(len(prices), len(predicted_returns))

        for i in range(n):
            if predicted_returns[i] > cfg.buy_threshold:
                signals[i] = 1.0
            elif predicted_returns[i] < cfg.sell_threshold:
                signals[i] = -1.0

        return signals

    def run_backtest(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """
        Run backtest simulation given price series and signals.

        Args:
            prices: Price series (T,)
            signals: Trading signals (T,) with values in {-1, 0, +1}
            timestamps: Optional datetime array for logging

        Returns:
            BacktestResult with all metrics and equity curve
        """
        cfg = self.config
        n = len(prices)
        assert len(signals) == n, "Prices and signals must have same length"

        capital = cfg.initial_capital
        position = 0  # current shares held (positive=long, negative=short)
        entry_price = 0.0
        entry_time = 0

        equity = np.zeros(n)
        equity[0] = capital
        self.trades = []

        peak_equity = capital

        for i in range(1, n):
            price = prices[i]
            signal = signals[i]

            # Mark to market
            if position != 0:
                unrealized_pnl = position * (price - entry_price)
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital

            # Check stop loss / take profit
            if position != 0:
                pnl_pct = (price - entry_price) / entry_price * np.sign(position)

                if pnl_pct <= -cfg.stop_loss_pct:
                    # Stop loss hit
                    realized_pnl = position * (price - entry_price)
                    cost = abs(position * price) * cfg.transaction_cost_bps / 10000
                    capital += realized_pnl - cost
                    self.trades.append(TradeRecord(
                        entry_time=entry_time, exit_time=i,
                        direction=int(np.sign(position)),
                        entry_price=entry_price, exit_price=price,
                        pnl=realized_pnl - cost, pnl_pct=pnl_pct,
                        exit_reason="stop_loss",
                    ))
                    position = 0

                elif pnl_pct >= cfg.take_profit_pct:
                    # Take profit hit
                    realized_pnl = position * (price - entry_price)
                    cost = abs(position * price) * cfg.transaction_cost_bps / 10000
                    capital += realized_pnl - cost
                    self.trades.append(TradeRecord(
                        entry_time=entry_time, exit_time=i,
                        direction=int(np.sign(position)),
                        entry_price=entry_price, exit_price=price,
                        pnl=realized_pnl - cost, pnl_pct=pnl_pct,
                        exit_reason="take_profit",
                    ))
                    position = 0

            # Process signals
            if position == 0 and signal != 0:
                # Enter position
                trade_capital = capital * cfg.position_size
                n_shares = trade_capital / price
                position = n_shares * signal
                entry_price = price
                entry_time = i
                cost = abs(position * price) * cfg.transaction_cost_bps / 10000
                capital -= cost

            elif position != 0 and signal == -np.sign(position):
                # Reverse signal: close position
                realized_pnl = position * (price - entry_price)
                cost = abs(position * price) * cfg.transaction_cost_bps / 10000
                capital += realized_pnl - cost
                self.trades.append(TradeRecord(
                    entry_time=entry_time, exit_time=i,
                    direction=int(np.sign(position)),
                    entry_price=entry_price, exit_price=price,
                    pnl=realized_pnl - cost,
                    pnl_pct=(price - entry_price) / entry_price * np.sign(position),
                    exit_reason="signal",
                ))
                position = 0

            # Record equity
            if position != 0:
                equity[i] = capital + position * (price - entry_price)
            else:
                equity[i] = capital

            # Max drawdown circuit breaker
            peak_equity = max(peak_equity, equity[i])
            dd = (peak_equity - equity[i]) / peak_equity
            if dd > cfg.max_drawdown_pct:
                # Close everything and stop
                if position != 0:
                    realized_pnl = position * (price - entry_price)
                    cost = abs(position * price) * cfg.transaction_cost_bps / 10000
                    capital += realized_pnl - cost
                    self.trades.append(TradeRecord(
                        entry_time=entry_time, exit_time=i,
                        direction=int(np.sign(position)),
                        entry_price=entry_price, exit_price=price,
                        pnl=realized_pnl - cost,
                        pnl_pct=(price - entry_price) / entry_price * np.sign(position),
                        exit_reason="max_drawdown",
                    ))
                    position = 0
                    equity[i] = capital
                # Fill remaining equity
                equity[i+1:] = capital
                break

        # Close any remaining position at end
        if position != 0:
            final_price = prices[-1]
            realized_pnl = position * (final_price - entry_price)
            cost = abs(position * final_price) * cfg.transaction_cost_bps / 10000
            capital += realized_pnl - cost
            self.trades.append(TradeRecord(
                entry_time=entry_time, exit_time=n - 1,
                direction=int(np.sign(position)),
                entry_price=entry_price, exit_price=final_price,
                pnl=realized_pnl - cost,
                pnl_pct=(final_price - entry_price) / entry_price * np.sign(position),
                exit_reason="end",
            ))
            equity[-1] = capital

        return self._compute_metrics(equity, signals)

    def _compute_metrics(
        self, equity: np.ndarray, signals: np.ndarray
    ) -> BacktestResult:
        """Compute backtest performance metrics."""

        # Daily returns
        daily_returns = np.diff(equity) / (equity[:-1] + 1e-10)

        # Total return
        total_return = (equity[-1] / equity[0]) - 1.0

        # Annualized return (assume 252 trading days)
        n_days = len(daily_returns)
        if n_days > 0 and total_return > -1:
            annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        else:
            annualized_return = -1.0

        # Sharpe ratio
        if len(daily_returns) > 1 and np.std(daily_returns) > 1e-10:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-10)
        max_dd = np.max(drawdown)

        # Trade statistics
        n_trades = len(self.trades)
        if n_trades > 0:
            pnls = [t.pnl for t in self.trades]
            wins = sum(1 for p in pnls if p > 0)
            win_rate = wins / n_trades
            avg_pnl = np.mean(pnls)

            gross_profit = sum(p for p in pnls if p > 0) or 0
            gross_loss = abs(sum(p for p in pnls if p < 0)) or 1e-10
            profit_factor = gross_profit / gross_loss
        else:
            win_rate = 0.0
            avg_pnl = 0.0
            profit_factor = 0.0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            n_trades=n_trades,
            avg_trade_pnl=avg_pnl,
            profit_factor=profit_factor,
            equity_curve=equity,
            drawdown_curve=drawdown,
            trades=self.trades,
            signals=signals,
            daily_returns=daily_returns,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison Backtest (Operator vs Baselines)
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison_backtest(
    prices: np.ndarray,
    operator_signals: np.ndarray,
    config: Optional[BacktestConfig] = None,
) -> Dict[str, BacktestResult]:
    """
    Compare operator learning strategy against baselines.

    Baselines:
    - Buy and Hold
    - Simple Moving Average Crossover
    - Mean Reversion

    Args:
        prices: Price series
        operator_signals: Signals from operator model
        config: Backtest configuration

    Returns:
        Dict mapping strategy name to BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    bt = OperatorBacktester(config)
    results = {}

    # 1. Operator strategy
    results["Operator Learning"] = bt.run_backtest(prices, operator_signals)

    # 2. Buy and Hold
    buy_hold_signals = np.zeros(len(prices))
    buy_hold_signals[config.lookback_window] = 1.0  # enter once
    results["Buy & Hold"] = bt.run_backtest(prices, buy_hold_signals)

    # 3. SMA Crossover
    sma_fast = pd.Series(prices).rolling(20).mean().values
    sma_slow = pd.Series(prices).rolling(50).mean().values
    sma_signals = np.zeros(len(prices))
    for i in range(50, len(prices)):
        if sma_fast[i] > sma_slow[i] and sma_fast[i - 1] <= sma_slow[i - 1]:
            sma_signals[i] = 1.0
        elif sma_fast[i] < sma_slow[i] and sma_fast[i - 1] >= sma_slow[i - 1]:
            sma_signals[i] = -1.0
    results["SMA Crossover"] = bt.run_backtest(prices, sma_signals)

    # 4. Mean Reversion
    lookback = 20
    mr_signals = np.zeros(len(prices))
    for i in range(lookback, len(prices)):
        window = prices[i - lookback : i]
        mu = np.mean(window)
        sigma = np.std(window) + 1e-10
        z_score = (prices[i] - mu) / sigma
        if z_score < -1.5:
            mr_signals[i] = 1.0  # Buy when oversold
        elif z_score > 1.5:
            mr_signals[i] = -1.0  # Sell when overbought
    results["Mean Reversion"] = bt.run_backtest(prices, mr_signals)

    return results


def print_backtest_results(results: Dict[str, BacktestResult]):
    """Print formatted comparison table of backtest results."""
    header = (
        f"{'Strategy':<22} {'Return':>10} {'Ann.Ret':>10} {'Sharpe':>8} "
        f"{'MaxDD':>8} {'WinRate':>8} {'Trades':>7} {'PF':>8}"
    )
    print("=" * len(header))
    print("BACKTEST COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        print(
            f"{name:<22} "
            f"{res.total_return:>+9.2%} "
            f"{res.annualized_return:>+9.2%} "
            f"{res.sharpe_ratio:>8.2f} "
            f"{res.max_drawdown:>7.2%} "
            f"{res.win_rate:>7.1%} "
            f"{res.n_trades:>7d} "
            f"{res.profit_factor:>8.2f}"
        )

    print("=" * len(header))


# ═══════════════════════════════════════════════════════════════════════════════
# Main demo
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run a demo backtest with synthetic data."""
    print("=" * 60)
    print("Operator Learning Backtest Demo - Chapter 152")
    print("=" * 60)

    np.random.seed(42)

    # Generate synthetic price data (geometric Brownian motion)
    n_days = 500
    dt = 1 / 252
    mu = 0.08
    sigma = 0.2
    s0 = 100.0

    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
    prices = s0 * np.exp(np.cumsum(returns))

    print(f"\nSynthetic price series: {n_days} days, S0={s0}")
    print(f"  Final price: {prices[-1]:.2f}")
    print(f"  Buy-and-hold return: {(prices[-1]/prices[0] - 1)*100:.1f}%")

    # Simulate operator signals (mildly predictive)
    # In practice these would come from a trained model
    lookback = 60
    operator_signals = np.zeros(n_days)
    for i in range(lookback, n_days - 5):
        # Simple momentum as proxy for operator prediction
        recent_return = (prices[i] - prices[i - 20]) / prices[i - 20]
        # Add some noise to simulate model uncertainty
        pred = recent_return * 0.5 + np.random.normal(0, 0.005)
        if pred > 0.005:
            operator_signals[i] = 1.0
        elif pred < -0.005:
            operator_signals[i] = -1.0

    # Run comparison
    config = BacktestConfig(
        initial_capital=100_000,
        position_size=0.2,
        transaction_cost_bps=10,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
    )

    results = run_comparison_backtest(prices, operator_signals, config)
    print()
    print_backtest_results(results)

    # Detailed trade analysis for operator strategy
    op_result = results["Operator Learning"]
    print(f"\nOperator Strategy Trade Analysis:")
    print(f"  Total trades: {op_result.n_trades}")

    if op_result.n_trades > 0:
        exit_reasons = {}
        for t in op_result.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        print(f"  Exit reasons: {exit_reasons}")

        pnls = [t.pnl for t in op_result.trades]
        print(f"  Best trade PnL:  ${max(pnls):>10,.2f}")
        print(f"  Worst trade PnL: ${min(pnls):>10,.2f}")
        print(f"  Avg trade PnL:   ${np.mean(pnls):>10,.2f}")

    # Visualize
    try:
        from .visualize import plot_crypto_operator_signals

        predictions = np.roll(prices, -1)  # Placeholder
        predictions[-1] = prices[-1]
        plot_crypto_operator_signals(
            prices, predictions, operator_signals,
            title="Operator Learning Strategy - Backtest Demo",
        )
    except Exception:
        print("\n(Visualization skipped - run with GUI for plots)")


if __name__ == "__main__":
    main()
