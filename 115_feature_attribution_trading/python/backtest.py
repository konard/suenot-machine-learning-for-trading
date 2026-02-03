"""
Backtesting engine for Feature Attribution Trading strategies.

Provides:
- Configurable backtesting engine with attribution-aware trading
- Performance metrics calculation (Sharpe, Sortino, Max Drawdown, etc.)
- Strategy evaluation with explainability tracking
- Risk management based on feature attribution quality
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import logging

from .data_loader import MarketData, generate_synthetic_data, DEFAULT_FEATURE_NAMES
from .feature_attribution_model import (
    FeatureAttributionModel,
    AttributionConfig,
    AttributionBasedTrader,
    create_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    trading_fee: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    min_confidence: float = 0.5
    max_position_size: float = 0.2
    stop_loss: float = 0.03
    take_profit: float = 0.06
    # Attribution-specific settings
    use_attribution_filter: bool = True
    min_attribution_quality: float = 0.15
    position_scale_by_quality: bool = True
    track_feature_importance: bool = True


@dataclass
class BacktestResult:
    """Results from backtesting with attribution analysis."""
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
    # Attribution-specific results
    attribution_stats: Dict = field(default_factory=dict)
    filtered_by_attribution: int = 0
    average_attribution_quality: float = 0.0

    def summary(self) -> str:
        """Return formatted summary of results."""
        return f"""
Backtest Results:
-----------------
Total Return:           {self.total_return * 100:.2f}%
Annual Return:          {self.annual_return * 100:.2f}%
Sharpe Ratio:           {self.sharpe_ratio:.3f}
Sortino Ratio:          {self.sortino_ratio:.3f}
Max Drawdown:           {self.max_drawdown * 100:.2f}%
Calmar Ratio:           {self.calmar_ratio:.3f}
Win Rate:               {self.win_rate * 100:.2f}%
Profit Factor:          {self.profit_factor:.2f}
Total Trades:           {self.total_trades}
Final Equity:           ${self.final_equity:,.2f}

Attribution Analysis:
---------------------
Filtered by Attribution: {self.filtered_by_attribution}
Avg Attribution Quality: {self.average_attribution_quality * 100:.2f}%
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
            "filtered_by_attribution": self.filtered_by_attribution,
            "average_attribution_quality": self.average_attribution_quality,
            "attribution_stats": self.attribution_stats,
        }

    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame({
            "equity": self.equity_curve,
            "period": range(len(self.equity_curve)),
        })

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        return pd.DataFrame(self.trades)


class BacktestEngine:
    """
    Backtesting engine for Feature Attribution Trading strategies.

    Simulates trading based on model predictions with attribution-aware
    risk management and signal quality filtering.

    Key Features:
    - Attribution-based signal filtering
    - Position sizing based on attribution quality
    - Feature importance tracking over time
    - Comprehensive performance metrics
    """

    def __init__(
        self,
        model: Optional[FeatureAttributionModel] = None,
        config: Optional[BacktestConfig] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            model: Feature attribution model (created if None)
            config: Backtest configuration (default if None)
            feature_names: Names of input features
        """
        self.model = model or create_model(input_dim=len(feature_names or DEFAULT_FEATURE_NAMES))
        self.config = config or BacktestConfig()
        self.feature_names = feature_names or DEFAULT_FEATURE_NAMES

        # Create attribution-based trader
        self.trader = AttributionBasedTrader(
            model=self.model,
            confidence_threshold=self.config.min_confidence,
            attribution_threshold=self.config.min_attribution_quality,
            feature_names=self.feature_names,
        )

        # Tracking
        self._feature_importance_history = []
        self._attribution_quality_history = []

    def run(
        self,
        data: MarketData,
        lookback: int = 64,
        verbose: bool = False,
    ) -> BacktestResult:
        """
        Run backtest on market data.

        Args:
            data: Market data for backtesting
            lookback: Number of periods for model input
            verbose: Whether to print trade details

        Returns:
            BacktestResult with performance metrics and attribution analysis
        """
        features = data.to_features(self.feature_names)
        closes = data.close
        n = len(closes)

        if n < lookback + 2:
            logger.warning(f"Insufficient data: {n} points, need at least {lookback + 2}")
            return self._empty_result()

        # Initialize state
        equity = self.config.initial_capital
        equity_curve = [equity]
        position = 0.0
        entry_price = 0.0
        trades = []
        trade_pnls = []

        # Attribution tracking
        filtered_by_attribution = 0
        attribution_qualities = []
        feature_importance_sum = {name: 0.0 for name in self.feature_names}
        importance_count = 0

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

                    reason = "Stop Loss" if pnl_pct <= -self.config.stop_loss else "Take Profit"
                    trades.append({
                        "type": "CLOSE",
                        "index": i,
                        "price": current_price,
                        "position": position,
                        "pnl": trade_pnl,
                        "reason": reason,
                    })

                    if verbose:
                        logger.info(f"t={i}: {reason} - Closed at {current_price:.2f}, PnL: {trade_pnl*100:.2f}%")

                    position = 0.0
                    entry_price = 0.0
                    continue

            # Get input features
            input_features = features[i - lookback:i]

            # Generate signal with attribution
            if self.config.use_attribution_filter:
                signal_result = self.trader.generate_signal(
                    input_features,
                    require_attribution_quality=True,
                )

                signal = signal_result["signal"]
                confidence = signal_result["confidence"]
                attribution_quality = signal_result["attribution_quality"]
                original_signal = signal_result["original_signal"]

                # Track attribution quality
                attribution_qualities.append(attribution_quality)

                # Track feature importance
                if self.config.track_feature_importance:
                    for name, imp in signal_result["top_features"].items():
                        if name in feature_importance_sum:
                            feature_importance_sum[name] += imp
                    importance_count += 1

                # Check if filtered by attribution
                if signal == "HOLD" and original_signal != "HOLD":
                    filtered_by_attribution += 1

            else:
                # Simple prediction without attribution
                prediction = self.model.predict(input_features)
                signal = prediction["signal"]
                confidence = prediction["confidence"]
                attribution_quality = 1.0

            # Calculate target position
            if confidence < self.config.min_confidence:
                target_position = 0.0
            elif signal == "BUY":
                base_size = self.config.max_position_size
                confidence_scale = (confidence - self.config.min_confidence) / (1 - self.config.min_confidence)

                if self.config.position_scale_by_quality:
                    quality_scale = min(1.0, attribution_quality / self.config.min_attribution_quality)
                else:
                    quality_scale = 1.0

                target_position = base_size * confidence_scale * quality_scale

            elif signal == "SELL":
                base_size = self.config.max_position_size
                confidence_scale = (confidence - self.config.min_confidence) / (1 - self.config.min_confidence)

                if self.config.position_scale_by_quality:
                    quality_scale = min(1.0, attribution_quality / self.config.min_attribution_quality)
                else:
                    quality_scale = 1.0

                target_position = -base_size * confidence_scale * quality_scale

            else:
                target_position = 0.0

            # Execute position change
            position_change = target_position - position

            if abs(position_change) > 0.001:
                # Apply trading costs
                trade_cost = equity * abs(position_change) * (self.config.trading_fee + self.config.slippage)
                equity -= trade_cost

                # Record closing trade
                if abs(position) > 0.001 and np.sign(position_change) != np.sign(position):
                    pnl_pct = (current_price - entry_price) / entry_price * np.sign(position)
                    trade_pnl = pnl_pct * abs(position)
                    trade_pnls.append(trade_pnl)

                    trades.append({
                        "type": "CLOSE",
                        "index": i,
                        "price": current_price,
                        "position": position,
                        "pnl": trade_pnl,
                        "reason": signal,
                    })

                old_position = position
                position += position_change

                # Record opening trade
                if abs(position) > 0.001 and abs(old_position) < 0.001:
                    entry_price = current_price
                    trades.append({
                        "type": "OPEN",
                        "index": i,
                        "signal": signal,
                        "price": current_price,
                        "position": position,
                        "confidence": confidence,
                        "attribution_quality": attribution_quality,
                    })

                    if verbose:
                        logger.info(
                            f"t={i}: {signal} at {current_price:.2f}, "
                            f"conf={confidence:.2f}, attr_q={attribution_quality:.2f}, "
                            f"pos={position:.4f}"
                        )

        # Calculate feature importance stats
        if importance_count > 0:
            avg_feature_importance = {
                name: imp / importance_count
                for name, imp in feature_importance_sum.items()
            }
        else:
            avg_feature_importance = {}

        # Calculate attribution stats
        attribution_stats = {
            "average_quality": np.mean(attribution_qualities) if attribution_qualities else 0.0,
            "min_quality": np.min(attribution_qualities) if attribution_qualities else 0.0,
            "max_quality": np.max(attribution_qualities) if attribution_qualities else 0.0,
            "std_quality": np.std(attribution_qualities) if attribution_qualities else 0.0,
            "feature_importance": avg_feature_importance,
        }

        # Store for later analysis
        self._feature_importance_history = attribution_qualities
        self._attribution_quality_history = attribution_qualities

        return self._calculate_metrics(
            equity_curve, trade_pnls, trades,
            filtered_by_attribution, attribution_stats
        )

    def run_with_baseline(
        self,
        data: MarketData,
        lookback: int = 64,
    ) -> Tuple[BacktestResult, BacktestResult]:
        """
        Run backtest with and without attribution filtering.

        Useful for comparing the impact of attribution-based trading.

        Args:
            data: Market data
            lookback: Lookback period

        Returns:
            Tuple of (with_attribution_result, without_attribution_result)
        """
        # Run with attribution
        original_setting = self.config.use_attribution_filter
        self.config.use_attribution_filter = True
        result_with = self.run(data, lookback)

        # Run without attribution
        self.config.use_attribution_filter = False
        result_without = self.run(data, lookback)

        # Restore setting
        self.config.use_attribution_filter = original_setting

        return result_with, result_without

    def _calculate_metrics(
        self,
        equity_curve: List[float],
        trade_pnls: List[float],
        trades: List[Dict],
        filtered_by_attribution: int,
        attribution_stats: Dict,
    ) -> BacktestResult:
        """Calculate performance metrics."""
        initial = self.config.initial_capital
        final_equity = equity_curve[-1] if equity_curve else initial

        # Returns
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / (equity_array[:-1] + 1e-8)

        # Total return
        total_return = (final_equity - initial) / initial

        # Annualize (assuming hourly data)
        n_periods = max(len(returns), 1)
        periods_per_year = 8760.0  # Hours per year
        annual_factor = periods_per_year / n_periods
        annual_return = (1 + total_return) ** min(annual_factor, 100) - 1  # Cap for stability

        # Sharpe ratio (annualized)
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1) + 1e-8
            sharpe_ratio = mean_return / std_return * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (annualized)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns, ddof=1) + 1e-8
            sortino_ratio = np.mean(returns) / downside_std * np.sqrt(periods_per_year)
        else:
            sortino_ratio = sharpe_ratio

        # Maximum drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / (peak + 1e-8)
        max_drawdown = np.max(drawdown)

        # Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0.001 else 0.0

        # Trade statistics
        trade_pnls_array = np.array(trade_pnls) if trade_pnls else np.array([0.0])
        winning_trades = trade_pnls_array[trade_pnls_array > 0]
        losing_trades = trade_pnls_array[trade_pnls_array < 0]

        n_trades = len(trade_pnls_array)
        win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0.0

        gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        gross_loss = np.abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0.001 else (
            float('inf') if gross_profit > 0 else 0.0
        )

        # Average attribution quality
        avg_attr_quality = attribution_stats.get("average_quality", 0.0)

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len([t for t in trades if t["type"] == "OPEN"]),
            final_equity=final_equity,
            equity_curve=equity_curve,
            trades=trades,
            attribution_stats=attribution_stats,
            filtered_by_attribution=filtered_by_attribution,
            average_attribution_quality=avg_attr_quality,
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
            attribution_stats={},
            filtered_by_attribution=0,
            average_attribution_quality=0.0,
        )

    def get_feature_importance_over_time(self) -> pd.DataFrame:
        """Get feature importance tracking over time."""
        if not self._feature_importance_history:
            return pd.DataFrame()

        return pd.DataFrame({
            "attribution_quality": self._attribution_quality_history,
        })


class WalkForwardBacktest:
    """
    Walk-forward backtesting with periodic model retraining.

    Tests strategy performance with realistic out-of-sample evaluation.
    """

    def __init__(
        self,
        model_factory,
        config: Optional[BacktestConfig] = None,
        train_window: int = 500,
        test_window: int = 100,
        retrain_frequency: int = 100,
    ):
        """
        Initialize walk-forward backtest.

        Args:
            model_factory: Function to create new model instances
            config: Backtest configuration
            train_window: Number of periods for training
            test_window: Number of periods for testing
            retrain_frequency: How often to retrain (in periods)
        """
        self.model_factory = model_factory
        self.config = config or BacktestConfig()
        self.train_window = train_window
        self.test_window = test_window
        self.retrain_frequency = retrain_frequency

    def run(
        self,
        data: MarketData,
        lookback: int = 64,
        verbose: bool = False,
    ) -> List[BacktestResult]:
        """
        Run walk-forward backtest.

        Args:
            data: Full market data
            lookback: Model lookback period
            verbose: Print progress

        Returns:
            List of BacktestResult for each test window
        """
        features = data.to_features()
        n = len(features)

        if n < self.train_window + self.test_window + lookback:
            logger.warning("Insufficient data for walk-forward backtest")
            return []

        results = []
        start = self.train_window + lookback

        while start + self.test_window <= n:
            # Create test data slice
            test_end = min(start + self.test_window, n)
            test_start = start - lookback  # Include lookback

            test_df = data.df.iloc[test_start:test_end].reset_index(drop=True)
            test_data = MarketData(
                df=test_df,
                symbol=data.symbol,
                interval=data.interval,
                source=data.source
            )

            # Create fresh model
            model = self.model_factory()

            # Run backtest
            engine = BacktestEngine(model=model, config=self.config)
            result = engine.run(test_data, lookback=lookback, verbose=verbose)
            results.append(result)

            if verbose:
                logger.info(
                    f"Window {len(results)}: {start}-{test_end}, "
                    f"Return: {result.total_return*100:.2f}%, "
                    f"Trades: {result.total_trades}"
                )

            start += self.retrain_frequency

        return results

    def aggregate_results(self, results: List[BacktestResult]) -> Dict:
        """
        Aggregate results from all test windows.

        Args:
            results: List of BacktestResult

        Returns:
            Dictionary with aggregated statistics
        """
        if not results:
            return {}

        returns = [r.total_return for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        drawdowns = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]

        return {
            "n_windows": len(results),
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
            "mean_max_drawdown": np.mean(drawdowns),
            "worst_drawdown": np.max(drawdowns),
            "mean_win_rate": np.mean(win_rates),
            "total_trades": sum(r.total_trades for r in results),
            "profitable_windows": sum(1 for r in returns if r > 0),
            "profitable_ratio": sum(1 for r in returns if r > 0) / len(returns),
        }


def run_backtest_demo():
    """Run a demonstration backtest."""
    print("Feature Attribution Trading Backtest Demo")
    print("=" * 60)

    # Generate test data
    print("\n1. Generating synthetic market data...")
    data = generate_synthetic_data("BTCUSDT", "60", n_points=2000, seed=42)
    print(f"   Generated {len(data.df)} data points")

    # Create model
    print("\n2. Creating Feature Attribution model...")
    model_config = AttributionConfig(
        input_dim=len(DEFAULT_FEATURE_NAMES),
        hidden_dims=[64, 32],
        dropout=0.2,
    )
    model = create_model(config=model_config, hidden_dim=64, seq_len=64)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")

    # Create backtest config
    backtest_config = BacktestConfig(
        initial_capital=100_000.0,
        trading_fee=0.001,
        slippage=0.0005,
        min_confidence=0.5,
        max_position_size=0.15,
        stop_loss=0.03,
        take_profit=0.06,
        use_attribution_filter=True,
        min_attribution_quality=0.15,
        position_scale_by_quality=True,
        track_feature_importance=True,
    )

    # Run backtest
    print("\n3. Running backtest with attribution filtering...")
    engine = BacktestEngine(
        model=model,
        config=backtest_config,
        feature_names=DEFAULT_FEATURE_NAMES,
    )

    result = engine.run(data, lookback=64, verbose=False)

    # Print results
    print("\n4. Results:")
    print(result.summary())

    # Compare with baseline (no attribution filtering)
    print("5. Comparing with baseline (no attribution filtering)...")
    result_with, result_without = engine.run_with_baseline(data, lookback=64)

    print(f"\n   With Attribution Filter:")
    print(f"      Return: {result_with.total_return * 100:.2f}%")
    print(f"      Sharpe: {result_with.sharpe_ratio:.3f}")
    print(f"      Trades: {result_with.total_trades}")
    print(f"      Filtered: {result_with.filtered_by_attribution}")

    print(f"\n   Without Attribution Filter:")
    print(f"      Return: {result_without.total_return * 100:.2f}%")
    print(f"      Sharpe: {result_without.sharpe_ratio:.3f}")
    print(f"      Trades: {result_without.total_trades}")

    # Feature importance analysis
    print("\n6. Feature Importance Analysis:")
    if result.attribution_stats.get("feature_importance"):
        importance = result.attribution_stats["feature_importance"]
        sorted_importance = sorted(importance.items(), key=lambda x: -x[1])
        for name, imp in sorted_importance[:5]:
            print(f"      {name}: {imp:.4f}")

    # Additional statistics
    print("\n7. Additional Statistics:")
    print(f"   Equity curve length: {len(result.equity_curve)}")
    print(f"   Starting equity: ${result.equity_curve[0]:,.2f}")
    print(f"   Ending equity: ${result.equity_curve[-1]:,.2f}")
    print(f"   Peak equity: ${max(result.equity_curve):,.2f}")
    print(f"   Trough equity: ${min(result.equity_curve):,.2f}")

    print("\n" + "=" * 60)
    print("Backtest demo complete!")

    return result


if __name__ == "__main__":
    run_backtest_demo()
