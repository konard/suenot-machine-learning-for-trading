"""
Backtest Module for Physics-Constrained GAN Trading Strategy

Implements a Monte Carlo simulation-based trading strategy that:
    1. Detects the current market regime from recent data
    2. Generates conditional synthetic forward paths using the GAN
    3. Computes expected return/risk from synthetic distribution
    4. Sizes positions based on Kelly criterion and confidence

Strategy variants:
    - GAN-MC: Pure Monte Carlo forward simulation
    - GAN-Regime: Regime-conditioned generation with adaptive sizing
    - GAN-Ensemble: Multiple regime scenarios weighted by likelihood

Usage:
    python backtest.py --model checkpoints/physics_gan_final.pt --symbol BTCUSDT
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class BacktestConfig:
    """Configuration for GAN-based backtest."""

    # Strategy parameters
    n_simulations: int = 500           # Monte Carlo paths per decision
    forward_horizon: int = 20          # Forward simulation horizon
    lookback_window: int = 60          # Window for regime detection
    signal_threshold: float = 0.3      # Sharpe threshold for entry
    max_position: float = 1.0          # Maximum position size (fraction of capital)
    use_kelly: bool = True             # Use Kelly criterion for sizing

    # Risk management
    max_drawdown_limit: float = 0.15   # Stop trading if drawdown exceeds this
    position_decay: float = 0.95       # Decay factor for position (gradual exit)
    stop_loss: float = 0.05            # Stop loss per trade
    take_profit: float = 0.10          # Take profit per trade

    # Costs
    transaction_cost: float = 0.001    # Per-trade cost (0.1%)
    slippage: float = 0.0005           # Estimated slippage

    # Capital
    initial_capital: float = 100000.0

    # Rebalance frequency
    rebalance_every: int = 1           # Rebalance every N periods


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0

    # Time series
    equity_curve: List[float] = field(default_factory=list)
    positions: List[float] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "=" * 50,
            "Backtest Results",
            "=" * 50,
            f"  Total Return:      {self.total_return:>10.2%}",
            f"  Annual Return:     {self.annualized_return:>10.2%}",
            f"  Sharpe Ratio:      {self.sharpe_ratio:>10.3f}",
            f"  Sortino Ratio:     {self.sortino_ratio:>10.3f}",
            f"  Max Drawdown:      {self.max_drawdown:>10.2%}",
            f"  Calmar Ratio:      {self.calmar_ratio:>10.3f}",
            f"  Win Rate:          {self.win_rate:>10.2%}",
            f"  Profit Factor:     {self.profit_factor:>10.3f}",
            f"  Number of Trades:  {self.n_trades:>10d}",
            "=" * 50,
        ]
        return "\n".join(lines)


def detect_regime(
    returns: np.ndarray,
    window: int = 60,
) -> Tuple[int, int]:
    """
    Detect current market regime and volatility level from recent returns.

    Returns:
        Tuple of (regime_index, vol_level_index)
        Regime: 0=bull, 1=bear, 2=sideways, 3=crisis
        Vol: 0=low, 1=medium, 2=high, 3=extreme
    """
    if len(returns) < window:
        return 2, 1  # Default: sideways, medium vol

    recent = returns[-window:]
    cum_return = np.sum(recent)
    volatility = np.std(recent)
    mean_abs_return = np.mean(np.abs(recent))

    # Regime detection (simple heuristic)
    if volatility > 0.05:  # High vol regime
        if cum_return < -0.1:
            regime = 3  # crisis
        else:
            regime = 3  # crisis (high vol)
    elif cum_return > 0.05:
        regime = 0  # bull
    elif cum_return < -0.05:
        regime = 1  # bear
    else:
        regime = 2  # sideways

    # Volatility level
    if volatility < 0.01:
        vol_level = 0
    elif volatility < 0.02:
        vol_level = 1
    elif volatility < 0.04:
        vol_level = 2
    else:
        vol_level = 3

    return regime, vol_level


def generate_forward_paths(
    gan,
    regime: int,
    vol_level: int,
    n_paths: int = 500,
    horizon: int = 20,
) -> np.ndarray:
    """
    Generate forward-looking synthetic paths conditioned on current regime.

    Args:
        gan: PhysicsConstrainedGAN instance
        regime: Current regime index
        vol_level: Current volatility level index
        n_paths: Number of Monte Carlo paths
        horizon: Forward horizon (uses first `horizon` steps from generated paths)

    Returns:
        Array of shape [n_paths, horizon] with synthetic returns
    """
    regime_map = {0: "bull", 1: "bear", 2: "sideways", 3: "crisis"}
    vol_map = {0: "low", 1: "medium", 2: "high", 3: "extreme"}

    paths = gan.generate(
        n_paths=n_paths,
        condition={
            "regime": regime_map.get(regime, "sideways"),
            "volatility": vol_map.get(vol_level, "medium"),
        },
    )

    # Use only first `horizon` steps
    if paths.shape[1] > horizon:
        paths = paths[:, :horizon]

    return paths


def compute_signal(
    forward_paths: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Compute trading signal from Monte Carlo forward paths.

    Args:
        forward_paths: Synthetic forward returns [n_paths, horizon]
        threshold: Sharpe ratio threshold for signal

    Returns:
        Tuple of (signal, position_size, details)
        signal: 'LONG', 'SHORT', or 'NEUTRAL'
        position_size: 0.0 to 1.0
        details: Dict with diagnostic values
    """
    # Cumulative returns at horizon end
    cumulative = np.sum(forward_paths, axis=1)

    expected_return = np.mean(cumulative)
    risk = np.std(cumulative)

    if risk < 1e-8:
        return "NEUTRAL", 0.0, {"sharpe": 0.0, "expected_return": 0.0, "risk": 0.0}

    sharpe = expected_return / risk

    # Win probability
    p_win = np.mean(cumulative > 0)

    # Average win / average loss
    wins = cumulative[cumulative > 0]
    losses = cumulative[cumulative <= 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = np.abs(np.mean(losses)) if len(losses) > 0 else 1e-8

    # Kelly fraction
    if avg_loss > 0:
        b = avg_win / avg_loss
        kelly = max(0, (p_win * b - (1 - p_win)) / b)
    else:
        kelly = 0.0

    kelly = min(kelly, 1.0)  # Cap at 100%

    details = {
        "sharpe": float(sharpe),
        "expected_return": float(expected_return),
        "risk": float(risk),
        "p_win": float(p_win),
        "kelly": float(kelly),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "var_95": float(np.percentile(cumulative, 5)),
        "cvar_95": float(np.mean(cumulative[cumulative <= np.percentile(cumulative, 5)])) if len(cumulative[cumulative <= np.percentile(cumulative, 5)]) > 0 else 0.0,
    }

    if sharpe > threshold:
        signal = "LONG"
        position_size = min(kelly * 0.5, 1.0)  # Half-Kelly for safety
    elif sharpe < -threshold:
        signal = "SHORT"
        position_size = min(kelly * 0.5, 1.0)
    else:
        signal = "NEUTRAL"
        position_size = 0.0

    return signal, position_size, details


def run_gan_backtest(
    gan,
    test_returns: np.ndarray,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """
    Run a full backtest of the GAN-based trading strategy.

    Args:
        gan: Trained PhysicsConstrainedGAN instance
        test_returns: Array of test period returns [n_periods]
        config: Backtest configuration

    Returns:
        BacktestResult with performance metrics and time series
    """
    if config is None:
        config = BacktestConfig()

    n_periods = len(test_returns)
    equity = config.initial_capital
    peak_equity = equity
    position = 0.0

    # Track results
    equity_curve = [equity]
    positions = [0.0]
    signals = ["NEUTRAL"]
    period_returns = [0.0]
    trade_pnls = []

    entry_price_approx = 0.0
    in_trade = False

    all_returns_so_far = []

    for t in range(n_periods):
        current_return = test_returns[t]
        all_returns_so_far.append(current_return)

        # Portfolio return for this period
        portfolio_return = position * current_return

        # Transaction costs on position changes (computed below)
        cost = 0.0

        # Check stop loss / take profit
        if in_trade and position != 0:
            cum_since_entry = sum(all_returns_so_far[max(0, t - config.forward_horizon):t + 1])
            if position > 0 and cum_since_entry < -config.stop_loss:
                cost += abs(position) * (config.transaction_cost + config.slippage)
                position = 0.0
                in_trade = False
            elif position < 0 and cum_since_entry > config.stop_loss:
                cost += abs(position) * (config.transaction_cost + config.slippage)
                position = 0.0
                in_trade = False
            elif abs(cum_since_entry) > config.take_profit:
                cost += abs(position) * (config.transaction_cost + config.slippage)
                trade_pnls.append(cum_since_entry * np.sign(position))
                position = 0.0
                in_trade = False

        # Rebalance decision
        if t % config.rebalance_every == 0 and t >= config.lookback_window:
            recent_returns = np.array(all_returns_so_far[-config.lookback_window:])
            regime, vol_level = detect_regime(recent_returns)

            # Generate forward paths
            forward_paths = generate_forward_paths(
                gan, regime, vol_level,
                n_paths=config.n_simulations,
                horizon=config.forward_horizon,
            )

            signal, size, details = compute_signal(forward_paths, config.signal_threshold)

            # Compute new target position
            if signal == "LONG":
                new_position = min(size, config.max_position)
            elif signal == "SHORT":
                new_position = -min(size, config.max_position)
            else:
                new_position = position * config.position_decay

            # Transaction cost for position change
            position_change = abs(new_position - position)
            cost += position_change * (config.transaction_cost + config.slippage)

            if new_position != 0 and not in_trade:
                in_trade = True
                entry_price_approx = t

            position = new_position

            signals.append(signal)
        else:
            signals.append(signals[-1] if signals else "NEUTRAL")

        # Update equity
        net_return = portfolio_return - cost
        equity *= (1 + net_return)
        equity_curve.append(equity)
        positions.append(position)
        period_returns.append(net_return)

        # Track drawdown
        peak_equity = max(peak_equity, equity)

        # Check max drawdown limit
        current_dd = (peak_equity - equity) / peak_equity
        if current_dd > config.max_drawdown_limit:
            # Reduce position
            cost_exit = abs(position) * (config.transaction_cost + config.slippage)
            equity *= (1 - cost_exit / equity) if equity > 0 else 1
            position = 0.0
            in_trade = False

    # Compute metrics
    equity_arr = np.array(equity_curve)
    returns_arr = np.array(period_returns[1:])

    total_return = (equity_arr[-1] / equity_arr[0]) - 1
    n_years = n_periods / 252  # Approximate trading days
    if n_years > 0:
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        annualized_return = 0.0

    # Sharpe ratio (annualized)
    if len(returns_arr) > 1 and np.std(returns_arr) > 0:
        sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino ratio
    downside = returns_arr[returns_arr < 0]
    if len(downside) > 1:
        downside_std = np.std(downside)
        sortino = np.mean(returns_arr) / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (running_max - equity_arr) / running_max
    max_dd = float(np.max(drawdowns))

    # Calmar ratio
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0

    # Win rate
    trades = returns_arr[returns_arr != 0]
    n_trades = len(trades)
    win_rate = np.mean(trades > 0) if n_trades > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(trades[trades > 0]) if np.any(trades > 0) else 0.0
    gross_loss = abs(np.sum(trades[trades < 0])) if np.any(trades < 0) else 1e-8
    profit_factor = gross_profit / gross_loss

    result = BacktestResult(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown=float(max_dd),
        calmar_ratio=float(calmar),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        n_trades=n_trades,
        equity_curve=equity_curve,
        positions=positions,
        signals=signals[:len(equity_curve)],
        drawdown_curve=drawdowns.tolist(),
        returns=period_returns,
    )

    return result


def run_comparison_backtest(
    gan,
    test_returns: np.ndarray,
    config: Optional[BacktestConfig] = None,
) -> Dict[str, BacktestResult]:
    """
    Run comparison between GAN-based strategy and baselines.

    Baselines:
        - Buy and Hold
        - Random (coin flip)
        - Momentum (simple moving average crossover)

    Returns:
        Dict mapping strategy name to BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    results = {}

    # 1. GAN-MC strategy
    print("Running GAN-MC strategy...")
    results["GAN-MC"] = run_gan_backtest(gan, test_returns, config)

    # 2. Buy and Hold
    print("Running Buy & Hold...")
    equity = config.initial_capital
    equity_curve = [equity]
    for r in test_returns:
        equity *= (1 + r)
        equity_curve.append(equity)
    bh_returns = np.diff(np.array(equity_curve)) / np.array(equity_curve[:-1])
    running_max = np.maximum.accumulate(np.array(equity_curve))
    drawdowns = (running_max - np.array(equity_curve)) / running_max

    results["Buy&Hold"] = BacktestResult(
        total_return=float((equity_curve[-1] / equity_curve[0]) - 1),
        sharpe_ratio=float(np.mean(bh_returns) / np.std(bh_returns) * np.sqrt(252)) if np.std(bh_returns) > 0 else 0.0,
        max_drawdown=float(np.max(drawdowns)),
        equity_curve=equity_curve,
        n_trades=1,
    )

    # 3. Simple Momentum
    print("Running Momentum strategy...")
    fast_window = 10
    slow_window = 30
    equity = config.initial_capital
    equity_curve = [equity]
    position = 0.0

    for t in range(len(test_returns)):
        portfolio_return = position * test_returns[t]
        cost = 0.0

        if t >= slow_window:
            recent = test_returns[t - slow_window:t]
            fast_ma = np.mean(recent[-fast_window:])
            slow_ma = np.mean(recent)

            new_pos = 1.0 if fast_ma > slow_ma else -0.5 if fast_ma < slow_ma else 0.0
            cost = abs(new_pos - position) * config.transaction_cost
            position = new_pos

        equity *= (1 + portfolio_return - cost)
        equity_curve.append(equity)

    mom_returns = np.diff(np.array(equity_curve)) / np.array(equity_curve[:-1])
    running_max = np.maximum.accumulate(np.array(equity_curve))
    drawdowns = (running_max - np.array(equity_curve)) / running_max

    results["Momentum"] = BacktestResult(
        total_return=float((equity_curve[-1] / equity_curve[0]) - 1),
        sharpe_ratio=float(np.mean(mom_returns) / np.std(mom_returns) * np.sqrt(252)) if np.std(mom_returns) > 0 else 0.0,
        max_drawdown=float(np.max(drawdowns)),
        equity_curve=equity_curve,
    )

    return results


def print_comparison(results: Dict[str, BacktestResult]):
    """Print a comparison table of backtest results."""
    print("\n" + "=" * 80)
    print("Strategy Comparison")
    print("=" * 80)
    header = f"{'Strategy':<15} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Sortino':>10} {'Trades':>8}"
    print(header)
    print("-" * 80)

    for name, result in results.items():
        print(
            f"{name:<15} "
            f"{result.total_return:>10.2%} "
            f"{result.sharpe_ratio:>10.3f} "
            f"{result.max_drawdown:>10.2%} "
            f"{result.sortino_ratio:>10.3f} "
            f"{result.n_trades:>8d}"
        )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Backtest Physics-Constrained GAN Strategy")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained GAN model checkpoint")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Symbol for test data")
    parser.add_argument("--n-simulations", type=int, default=500,
                        help="Monte Carlo simulations per decision")
    parser.add_argument("--initial-capital", type=float, default=100000.0,
                        help="Initial capital")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Use synthetic test data")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison with baselines")

    args = parser.parse_args()

    from model import PhysicsConstrainedGAN, GANConfig
    from data_loader import generate_synthetic_data, BybitDataLoader

    # Load or create GAN
    config = GANConfig(seq_len=256, device="cpu")
    gan = PhysicsConstrainedGAN(config)

    if args.model and os.path.exists(args.model):
        gan.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print("No trained model provided. Using untrained model for demo.")
        print("(Results will not be meaningful without training)")

    # Load test data
    if args.use_synthetic:
        data = generate_synthetic_data(seq_len=500, n_samples=1)
        test_returns = data["windows"][0]
    else:
        try:
            loader = BybitDataLoader(symbol=args.symbol)
            df = loader.fetch_extended(n_candles=1000)
            if not df.empty:
                prices = df["close"].values
                test_returns = np.diff(np.log(prices))
            else:
                print("Failed to fetch data. Using synthetic.")
                data = generate_synthetic_data(seq_len=500, n_samples=1)
                test_returns = data["windows"][0]
        except Exception as e:
            print(f"Data fetch error: {e}. Using synthetic.")
            data = generate_synthetic_data(seq_len=500, n_samples=1)
            test_returns = data["windows"][0]

    print(f"Test data: {len(test_returns)} periods")

    bt_config = BacktestConfig(
        n_simulations=args.n_simulations,
        initial_capital=args.initial_capital,
    )

    if args.compare:
        results = run_comparison_backtest(gan, test_returns, bt_config)
        print_comparison(results)
    else:
        result = run_gan_backtest(gan, test_returns, bt_config)
        print(result.summary())


if __name__ == "__main__":
    main()
