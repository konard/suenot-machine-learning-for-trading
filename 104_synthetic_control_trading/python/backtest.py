"""
Backtesting framework for Synthetic Control Method trading strategies.

Provides:
- Event-driven backtesting
- Performance metrics calculation
- Risk management utilities
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from python.synthetic_control import SyntheticControlModel, SCMResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a single trade."""
    entry_date: datetime
    exit_date: datetime
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    return_pct: float
    car_at_entry: float
    holding_days: int


@dataclass
class BacktestResults:
    """Results from backtesting."""
    trades: List[TradeResult]
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_car: float
    car_tstat: float
    n_events: int
    n_trades: int
    equity_curve: np.ndarray


class SCMEventBacktester:
    """
    Backtester for SCM-based event trading strategies.

    This backtester:
    1. Processes a list of events
    2. Constructs synthetic controls for each treated unit
    3. Generates trading signals based on cumulative abnormal returns
    4. Executes trades with risk management
    5. Calculates performance metrics

    Example:
        >>> backtester = SCMEventBacktester(initial_capital=100_000)
        >>> strategy = {"entry_threshold": 0.02, "exit_days": 5}
        >>> results = backtester.run(events_df, strategy)
        >>> print(f"Sharpe: {results.sharpe_ratio:.3f}")
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        position_size: float = 0.1,
        max_positions: int = 5,
    ):
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction of trade value
            position_size: Position size as fraction of capital
            max_positions: Maximum concurrent positions
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        self.max_positions = max_positions

    def run(
        self,
        events: List[Dict],
        strategy: Dict,
        data_loader_fn: Optional[Callable] = None,
    ) -> BacktestResults:
        """
        Run backtest on a list of events.

        Args:
            events: List of event dictionaries with keys:
                - treated_symbol: Symbol of treated unit
                - donor_symbols: List of donor symbols
                - event_date: Date of event (YYYY-MM-DD)
                - treated_pre: Pre-treatment data (optional if data_loader_fn provided)
                - donors_pre: Donor pre-treatment data
                - treated_post: Post-treatment data
                - donors_post: Donor post-treatment data
            strategy: Strategy configuration with keys:
                - entry_threshold: CAR threshold for entry (default: 0.02)
                - exit_days: Days to hold position (default: 5)
                - stop_loss: Stop loss as fraction (default: -0.05)
                - take_profit: Take profit as fraction (default: 0.10)
                - direction: "long", "short", or "both" (default: "long")
            data_loader_fn: Optional function to load data for events

        Returns:
            BacktestResults with performance metrics and trades
        """
        # Parse strategy parameters
        entry_threshold = strategy.get("entry_threshold", 0.02)
        exit_days = strategy.get("exit_days", 5)
        stop_loss = strategy.get("stop_loss", -0.05)
        take_profit = strategy.get("take_profit", 0.10)
        direction = strategy.get("direction", "long")

        trades = []
        cars = []
        capital = self.initial_capital
        equity_curve = [capital]

        for event in events:
            try:
                # Get data
                if data_loader_fn is not None:
                    event_data = data_loader_fn(event)
                else:
                    event_data = event

                treated_pre = np.asarray(event_data.get("treated_pre", []))
                donors_pre = np.asarray(event_data.get("donors_pre", []))
                treated_post = np.asarray(event_data.get("treated_post", []))
                donors_post = np.asarray(event_data.get("donors_post", []))

                if len(treated_pre) < 20 or len(treated_post) < 3:
                    logger.warning(f"Insufficient data for event: {event.get('treated_symbol')}")
                    continue

                # Fit SCM
                model = SyntheticControlModel()
                model.fit(treated_pre, donors_pre)

                # Check pre-treatment fit quality
                if model.pre_treatment_rmse_ > 0.05:
                    logger.debug(f"Poor pre-treatment fit, skipping: RMSE={model.pre_treatment_rmse_:.4f}")
                    continue

                # Estimate effects
                effects = model.estimate_effects(treated_post, donors_post)
                car = effects["car"]
                cars.append(car)

                # Check entry signal
                should_enter = False
                trade_direction = 0

                if direction in ["long", "both"] and car > entry_threshold:
                    should_enter = True
                    trade_direction = 1
                elif direction in ["short", "both"] and car < -entry_threshold:
                    should_enter = True
                    trade_direction = -1

                if not should_enter:
                    continue

                # Execute trade
                trade = self._execute_trade(
                    event=event,
                    effects=effects,
                    direction=trade_direction,
                    exit_days=exit_days,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    capital=capital,
                )

                if trade is not None:
                    trades.append(trade)
                    capital += trade.pnl
                    equity_curve.append(capital)

            except Exception as e:
                logger.warning(f"Error processing event: {e}")
                continue

        # Calculate performance metrics
        results = self._calculate_metrics(
            trades=trades,
            cars=cars,
            equity_curve=np.array(equity_curve),
            n_events=len(events),
        )

        return results

    def _execute_trade(
        self,
        event: Dict,
        effects: Dict,
        direction: int,
        exit_days: int,
        stop_loss: float,
        take_profit: float,
        capital: float,
    ) -> Optional[TradeResult]:
        """
        Execute a single trade based on SCM signal.

        Args:
            event: Event dictionary
            effects: SCM effects dictionary
            direction: Trade direction (1=long, -1=short)
            exit_days: Days to hold
            stop_loss: Stop loss level
            take_profit: Take profit level
            capital: Current capital

        Returns:
            TradeResult or None if trade cannot be executed
        """
        treated_post = effects["treated_post"]
        car = effects["car"]

        if len(treated_post) < exit_days:
            exit_days = len(treated_post) - 1

        if exit_days < 1:
            return None

        # Entry at first post-treatment bar
        entry_price = 100.0  # Normalized, using returns
        position_value = capital * self.position_size

        # Calculate exit
        cumulative_return = 0.0
        exit_day = exit_days

        for i in range(1, min(exit_days + 1, len(treated_post))):
            daily_return = treated_post[i] if i < len(treated_post) else 0
            cumulative_return += daily_return * direction

            # Check stop loss / take profit
            if cumulative_return <= stop_loss:
                exit_day = i
                break
            if cumulative_return >= take_profit:
                exit_day = i
                break

        # Calculate PnL
        gross_return = cumulative_return
        net_return = gross_return - 2 * self.transaction_cost  # Entry + exit costs
        pnl = position_value * net_return

        return TradeResult(
            entry_date=pd.to_datetime(event.get("event_date", datetime.now())),
            exit_date=pd.to_datetime(event.get("event_date", datetime.now())) + pd.Timedelta(days=exit_day),
            symbol=event.get("treated_symbol", "UNKNOWN"),
            direction=direction,
            entry_price=entry_price,
            exit_price=entry_price * (1 + cumulative_return * direction),
            position_size=position_value,
            pnl=pnl,
            return_pct=net_return,
            car_at_entry=car,
            holding_days=exit_day,
        )

    def _calculate_metrics(
        self,
        trades: List[TradeResult],
        cars: List[float],
        equity_curve: np.ndarray,
        n_events: int,
    ) -> BacktestResults:
        """
        Calculate backtest performance metrics.

        Args:
            trades: List of executed trades
            cars: List of CARs for all events
            equity_curve: Equity curve array
            n_events: Total number of events

        Returns:
            BacktestResults with all metrics
        """
        if len(trades) == 0:
            return BacktestResults(
                trades=[],
                total_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_car=np.mean(cars) if cars else 0.0,
                car_tstat=0.0,
                n_events=n_events,
                n_trades=0,
                equity_curve=equity_curve,
            )

        # Extract returns
        returns = np.array([t.return_pct for t in trades])
        pnls = np.array([t.pnl for t in trades])

        # Total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        # Sharpe ratio (assuming 252 trading days)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 / np.mean([t.holding_days for t in trades]))
        else:
            sharpe_ratio = 0.0

        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and np.std(negative_returns) > 0:
            sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252 / np.mean([t.holding_days for t in trades]))
        else:
            sortino_ratio = sharpe_ratio

        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)

        # Win rate
        win_rate = np.mean(returns > 0)

        # Profit factor
        gross_profit = np.sum(pnls[pnls > 0])
        gross_loss = np.abs(np.sum(pnls[pnls < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # CAR statistics
        avg_car = np.mean(cars) if cars else 0.0
        if len(cars) > 1:
            car_tstat = avg_car / (np.std(cars) / np.sqrt(len(cars)))
        else:
            car_tstat = 0.0

        return BacktestResults(
            trades=trades,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_car=avg_car,
            car_tstat=car_tstat,
            n_events=n_events,
            n_trades=len(trades),
            equity_curve=equity_curve,
        )


def generate_sample_events(
    n_events: int = 50,
    pre_days: int = 60,
    post_days: int = 20,
    n_donors: int = 5,
    treatment_effect: float = 0.03,
) -> List[Dict]:
    """
    Generate sample events for testing the backtester.

    Args:
        n_events: Number of events to generate
        pre_days: Pre-treatment period length
        post_days: Post-treatment period length
        n_donors: Number of donors per event
        treatment_effect: Average treatment effect

    Returns:
        List of event dictionaries
    """
    np.random.seed(42)
    events = []

    for i in range(n_events):
        # Generate correlated returns
        common_factor = np.random.randn(pre_days + post_days) * 0.01

        treated = common_factor + np.random.randn(pre_days + post_days) * 0.005
        donors = np.zeros((pre_days + post_days, n_donors))

        for j in range(n_donors):
            donors[:, j] = common_factor + np.random.randn(pre_days + post_days) * 0.005

        # Add treatment effect post-treatment (with noise)
        effect = treatment_effect * (1 + np.random.randn() * 0.5)
        treated[pre_days:] += effect / post_days

        events.append({
            "treated_symbol": f"STOCK_{i}",
            "donor_symbols": [f"DONOR_{i}_{j}" for j in range(n_donors)],
            "event_date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "treated_pre": treated[:pre_days],
            "donors_pre": donors[:pre_days],
            "treated_post": treated[pre_days:],
            "donors_post": donors[pre_days:],
        })

    return events


def print_backtest_report(results: BacktestResults) -> None:
    """Print a formatted backtest report."""
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Events Analyzed:     {results.n_events}")
    print(f"Trades Executed:     {results.n_trades}")
    print("-" * 50)
    print(f"Total Return:        {results.total_return:.2%}")
    print(f"Sharpe Ratio:        {results.sharpe_ratio:.3f}")
    print(f"Sortino Ratio:       {results.sortino_ratio:.3f}")
    print(f"Max Drawdown:        {results.max_drawdown:.2%}")
    print(f"Win Rate:            {results.win_rate:.2%}")
    print(f"Profit Factor:       {results.profit_factor:.2f}")
    print("-" * 50)
    print(f"Average CAR:         {results.avg_car:.4f}")
    print(f"CAR t-statistic:     {results.car_tstat:.3f}")
    print("=" * 50)
