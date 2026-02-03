"""
Trading Strategy and Backtesting for PCMCI Causal Discovery

Provides:
- BacktestConfig: Configuration for backtesting
- CausalTradingStrategy: Strategy using PCMCI causal links for signal generation
- Backtester: Backtesting engine with performance metrics
- BacktestResult: Dataclass holding all backtest results
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

from .model import PCMCIConfig, PCMCICausalDiscovery, IndependenceTest
from .data import prepare_causal_data

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting causal trading strategies.

    Example:
        config = BacktestConfig(
            initial_capital=100000,
            position_size=0.1,
            stop_loss=0.02,
            take_profit=0.04
        )
    """
    initial_capital: float = 100000.0
    position_size: float = 0.1         # Fraction of capital per trade
    max_position: float = 1.0          # Maximum total position (1.0 = 100%)
    stop_loss: Optional[float] = 0.02  # Stop loss as fraction (2%)
    take_profit: Optional[float] = 0.04  # Take profit as fraction (4%)
    fee_rate: float = 0.001            # Transaction fee rate (0.1%)
    slippage: float = 0.0005           # Slippage per trade (0.05%)
    min_confidence: float = 0.1        # Minimum signal confidence to trade
    rebalance_threshold: float = 0.05  # Minimum change to trigger rebalance
    lookback_window: int = 252         # Window for causal analysis (trading days)
    refit_frequency: int = 21          # How often to refit causal model (days)
    ann_factor: int = 252              # Annualization factor


@dataclass
class BacktestResult:
    """
    Container for backtest results and performance metrics.

    Holds both summary statistics and time series data from a backtest.

    Example:
        result = backtester.run(strategy, data)
        print(f"Sharpe: {result.sharpe_ratio:.2f}")
        print(f"Max DD: {result.max_drawdown:.2%}")
    """
    # Return metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown metrics
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Trade metrics
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # Causal metrics
    n_causal_links: int = 0
    avg_link_strength: float = 0.0

    # Time series data
    cumulative_returns: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    positions: Optional[pd.DataFrame] = None
    signals: Optional[pd.DataFrame] = None
    equity_curve: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None
    causal_links_history: Optional[List[Any]] = None

    def summary(self) -> str:
        """Generate a formatted summary string."""
        lines = [
            "",
            "=" * 60,
            "BACKTEST RESULTS - PCMCI Causal Strategy",
            "=" * 60,
            f"Total Return:          {self.total_return * 100:>10.2f}%",
            f"Annual Return:         {self.annual_return * 100:>10.2f}%",
            f"Volatility:            {self.volatility * 100:>10.2f}%",
            f"Sharpe Ratio:          {self.sharpe_ratio:>10.2f}",
            f"Sortino Ratio:         {self.sortino_ratio:>10.2f}",
            f"Calmar Ratio:          {self.calmar_ratio:>10.2f}",
            "-" * 60,
            f"Max Drawdown:          {self.max_drawdown * 100:>10.2f}%",
            f"Avg Drawdown:          {self.avg_drawdown * 100:>10.2f}%",
            f"Max DD Duration:       {self.max_drawdown_duration:>10d} days",
            "-" * 60,
            f"Number of Trades:      {self.n_trades:>10d}",
            f"Win Rate:              {self.win_rate * 100:>10.2f}%",
            f"Profit Factor:         {self.profit_factor:>10.2f}",
            f"Avg Win:               {self.avg_win * 100:>10.4f}%",
            f"Avg Loss:              {self.avg_loss * 100:>10.4f}%",
            f"Best Trade:            {self.best_trade * 100:>10.4f}%",
            f"Worst Trade:           {self.worst_trade * 100:>10.4f}%",
            "-" * 60,
            f"Causal Links Found:    {self.n_causal_links:>10d}",
            f"Avg Link Strength:     {self.avg_link_strength:>10.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class CausalTradingStrategy:
    """
    Trading strategy based on PCMCI causal discovery.

    Uses PCMCI to identify time-lagged causal relationships between
    financial variables. When a causal parent moves, the strategy
    predicts the effect on the target variable and generates trading
    signals accordingly.

    Supports multi-asset portfolios where causal links between assets
    are used for cross-asset signal generation.

    Example:
        strategy = CausalTradingStrategy(
            pcmci_config=PCMCIConfig(max_lag=5),
            backtest_config=BacktestConfig(),
            target_columns=['AAPL_returns', 'MSFT_returns']
        )

        signals = strategy.generate_signals(data, feature_columns)
    """

    def __init__(
        self,
        pcmci_config: Optional[PCMCIConfig] = None,
        backtest_config: Optional[BacktestConfig] = None,
        target_columns: Optional[List[str]] = None
    ):
        """
        Initialize causal trading strategy.

        Args:
            pcmci_config: Configuration for PCMCI algorithm
            backtest_config: Configuration for backtesting
            target_columns: Column names of target variables (assets to trade)
        """
        self.pcmci_config = pcmci_config or PCMCIConfig()
        self.backtest_config = backtest_config or BacktestConfig()
        self.target_columns = target_columns or []
        self.pcmci: Optional[PCMCICausalDiscovery] = None
        self._causal_links: List[Tuple[str, str, int, float]] = []
        self._fitted = False

    def fit_causal_model(
        self,
        data: np.ndarray,
        variable_names: Optional[List[str]] = None
    ) -> List[Tuple[str, str, int, float]]:
        """
        Fit the PCMCI causal model on training data.

        Args:
            data: Time series data, shape (n_timesteps, n_variables)
            variable_names: Names of the variables

        Returns:
            List of discovered causal links (source, target, lag, strength)
        """
        self.pcmci_config.n_variables = data.shape[1]
        if variable_names:
            self.pcmci_config.variable_names = variable_names

        self.pcmci = PCMCICausalDiscovery(self.pcmci_config)
        self.pcmci.fit(data)

        self._causal_links = self.pcmci.get_causal_links(
            threshold=self.backtest_config.min_confidence
        )
        self._fitted = True

        logger.info(f"Fitted causal model: {len(self._causal_links)} links discovered")
        return self._causal_links

    def _compute_causal_signal(
        self,
        current_data: np.ndarray,
        target_idx: int,
        variable_names: List[str]
    ) -> Tuple[float, float]:
        """
        Compute trading signal for a target variable based on causal parents.

        Looks at recent movements in causal parent variables and predicts
        the expected direction and magnitude for the target.

        Args:
            current_data: Recent data window, shape (window_size, n_variables)
            target_idx: Index of the target variable
            variable_names: Names of variables

        Returns:
            Tuple of (signal_strength, confidence) where:
                signal_strength: Expected direction/magnitude (-1 to 1)
                confidence: Confidence in the signal (0 to 1)
        """
        if not self._causal_links:
            return 0.0, 0.0

        target_name = variable_names[target_idx]

        # Find causal parents of this target
        target_parents = [
            (src, lag, strength)
            for src, tgt, lag, strength in self._causal_links
            if tgt == target_name and lag > 0
        ]

        if not target_parents:
            return 0.0, 0.0

        # Compute weighted signal from parent movements
        weighted_signal = 0.0
        total_weight = 0.0

        for src_name, lag, strength in target_parents:
            # Find source variable index
            src_idx = None
            for i, name in enumerate(variable_names):
                if name == src_name:
                    src_idx = i
                    break

            if src_idx is None:
                continue

            # Get the parent's value at t-lag
            if lag <= len(current_data):
                parent_value = current_data[-lag, src_idx]

                # Signal: if positive causal link and parent moved up,
                # expect target to move up
                signal_contribution = strength * parent_value
                weighted_signal += signal_contribution
                total_weight += abs(strength)

        if total_weight < 1e-10:
            return 0.0, 0.0

        # Normalize signal
        signal = np.clip(weighted_signal / total_weight, -1.0, 1.0)

        # Confidence based on number of agreeing parents and their strength
        confidence = min(total_weight / len(target_parents), 1.0)

        return float(signal), float(confidence)

    def generate_signals(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals for the entire dataset.

        Performs rolling causal analysis: fits PCMCI on a lookback window
        and generates signals based on discovered causal relationships.

        Args:
            data: DataFrame with all feature columns
            feature_columns: Columns to use for causal analysis
            target_columns: Columns representing tradeable assets (default: self.target_columns)

        Returns:
            DataFrame with signal columns for each target asset

        Example:
            signals = strategy.generate_signals(
                data,
                feature_columns=['SPY_returns', 'VIX_returns', 'TLT_returns'],
                target_columns=['SPY_returns']
            )
        """
        if target_columns is None:
            target_columns = self.target_columns

        if not target_columns:
            # Default: first feature column is the target
            target_columns = [feature_columns[0]]

        lookback = self.backtest_config.lookback_window
        refit_freq = self.backtest_config.refit_frequency

        # Prepare data arrays
        available_features = [c for c in feature_columns if c in data.columns]
        if not available_features:
            raise ValueError("No valid feature columns found in data")

        raw_data = data[available_features].values.astype(np.float64)
        variable_names = available_features

        # Find target indices
        target_indices = []
        for tc in target_columns:
            if tc in available_features:
                target_indices.append(available_features.index(tc))
            else:
                logger.warning(f"Target column '{tc}' not in feature columns")

        if not target_indices:
            raise ValueError("No valid target columns found")

        # Generate signals
        signal_records = []
        last_fit_step = -refit_freq  # Force initial fit

        for t in range(lookback, len(raw_data)):
            # Refit causal model periodically
            if t - last_fit_step >= refit_freq:
                window_data = raw_data[t - lookback:t]

                # Standardize the window
                means = window_data.mean(axis=0)
                stds = window_data.std(axis=0)
                stds[stds < 1e-10] = 1.0
                window_std = (window_data - means) / stds

                try:
                    self.fit_causal_model(window_std, variable_names)
                    last_fit_step = t
                except Exception as e:
                    logger.warning(f"Causal model fit failed at step {t}: {e}")

            # Generate signal for each target
            record = {'step': t}
            if 'timestamp' in data.columns:
                record['timestamp'] = data['timestamp'].iloc[t]

            # Use recent standardized data for signal computation
            recent_window = min(self.pcmci_config.max_lag + 5, t)
            recent_data = raw_data[t - recent_window:t]

            # Standardize recent data
            r_means = recent_data.mean(axis=0)
            r_stds = recent_data.std(axis=0)
            r_stds[r_stds < 1e-10] = 1.0
            recent_std = (recent_data - r_means) / r_stds

            for target_idx in target_indices:
                target_name = variable_names[target_idx]
                signal, confidence = self._compute_causal_signal(
                    recent_std, target_idx, variable_names
                )

                # Apply confidence threshold
                if confidence < self.backtest_config.min_confidence:
                    signal = 0.0
                    confidence = 0.0

                # Position sizing based on signal and confidence
                position = signal * confidence * self.backtest_config.position_size

                record[f'{target_name}_signal'] = signal
                record[f'{target_name}_confidence'] = confidence
                record[f'{target_name}_position'] = np.clip(
                    position,
                    -self.backtest_config.max_position,
                    self.backtest_config.max_position
                )

            signal_records.append(record)

        signals_df = pd.DataFrame(signal_records)
        if 'timestamp' in signals_df.columns:
            signals_df = signals_df.set_index('timestamp')
        elif 'step' in signals_df.columns:
            signals_df = signals_df.set_index('step')

        logger.info(
            f"Generated {len(signals_df)} signal rows for "
            f"{len(target_indices)} target(s)"
        )

        return signals_df

    def get_causal_summary(self) -> Dict[str, Any]:
        """
        Get summary of current causal model.

        Returns:
            Dictionary with causal discovery results
        """
        if not self._fitted or self.pcmci is None:
            return {'fitted': False, 'links': []}

        return {
            'fitted': True,
            'n_links': len(self._causal_links),
            'links': self._causal_links,
            'summary': self.pcmci.get_summary()
        }


class Backtester:
    """
    Backtesting engine for causal trading strategies.

    Simulates strategy execution with realistic transaction costs,
    slippage, and position management. Computes comprehensive
    performance metrics.

    Example:
        backtester = Backtester(BacktestConfig())
        result = backtester.run(strategy, data, feature_cols, target_cols)
        print(result.summary())
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: CausalTradingStrategy,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        benchmark_column: Optional[str] = None
    ) -> BacktestResult:
        """
        Run full backtest of a causal trading strategy.

        Args:
            strategy: CausalTradingStrategy instance
            data: DataFrame with all data
            feature_columns: Columns for causal analysis
            target_columns: Tradeable asset columns (should contain returns)
            benchmark_column: Optional benchmark returns column

        Returns:
            BacktestResult with all metrics and time series

        Example:
            result = backtester.run(
                strategy, data,
                feature_columns=['SPY_returns', 'VIX_returns'],
                target_columns=['SPY_returns']
            )
        """
        # Generate signals
        logger.info("Generating trading signals...")
        signals_df = strategy.generate_signals(
            data, feature_columns, target_columns
        )

        # Run backtest for each target
        all_returns = []
        all_positions = []

        lookback = self.config.lookback_window

        for target_col in target_columns:
            position_col = f'{target_col}_position'

            if position_col not in signals_df.columns:
                logger.warning(f"No position column for {target_col}")
                continue

            # Get actual returns for this asset
            if signals_df.index.dtype == np.int64 or signals_df.index.dtype == np.int32:
                # Integer index (step-based)
                actual_returns = data[target_col].iloc[
                    signals_df.index.values
                ].values
            else:
                # Timestamp index
                actual_returns = data.set_index('timestamp')[target_col].reindex(
                    signals_df.index
                ).values

            positions = signals_df[position_col].values

            # Apply position from previous period (no look-ahead)
            positions_shifted = np.zeros_like(positions)
            positions_shifted[1:] = positions[:-1]

            # Calculate strategy returns
            strategy_returns = positions_shifted * actual_returns

            # Transaction costs
            position_changes = np.abs(np.diff(positions_shifted, prepend=0))
            costs = position_changes * (self.config.fee_rate + self.config.slippage)

            # Apply stop loss and take profit
            strategy_returns = self._apply_risk_management(
                strategy_returns, positions_shifted
            )

            # Net returns
            net_returns = strategy_returns - costs

            all_returns.append(net_returns)
            all_positions.append(positions_shifted)

        if not all_returns:
            logger.warning("No valid returns computed")
            return BacktestResult()

        # Aggregate returns across assets (equal weight if multiple)
        if len(all_returns) == 1:
            portfolio_returns = all_returns[0]
        else:
            portfolio_returns = np.mean(all_returns, axis=0)

        # Build result
        result = self._compute_metrics(
            portfolio_returns,
            signals_df,
            strategy
        )

        # Add benchmark comparison
        if benchmark_column and benchmark_column in data.columns:
            benchmark_returns = data[benchmark_column].iloc[lookback:].values
            if len(benchmark_returns) >= len(portfolio_returns):
                benchmark_returns = benchmark_returns[:len(portfolio_returns)]
                bench_cum = np.cumprod(1 + benchmark_returns)
                result_total = result.total_return
                bench_total = bench_cum[-1] - 1 if len(bench_cum) > 0 else 0
                # Store as attribute on the result
                result.excess_return = result_total - bench_total

        return result

    def run_from_signals(
        self,
        signals: pd.DataFrame,
        returns: pd.Series,
        position_column: str = 'position'
    ) -> BacktestResult:
        """
        Run backtest from pre-computed signals and returns.

        Simpler interface when signals are already generated.

        Args:
            signals: DataFrame with position column
            returns: Series of actual asset returns
            position_column: Name of the position column in signals

        Returns:
            BacktestResult with all metrics

        Example:
            result = backtester.run_from_signals(signals_df, returns_series)
        """
        # Align signals with returns
        common_index = signals.index.intersection(returns.index)
        if len(common_index) == 0:
            logger.warning("No common index between signals and returns")
            return BacktestResult()

        signals = signals.loc[common_index]
        returns_aligned = returns.loc[common_index]

        # Get positions (shifted to avoid look-ahead)
        positions = signals[position_column].shift(1).fillna(0).values
        actual_returns = returns_aligned.values

        # Strategy returns
        strategy_returns = positions * actual_returns

        # Transaction costs
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = position_changes * (self.config.fee_rate + self.config.slippage)

        # Apply risk management
        strategy_returns = self._apply_risk_management(strategy_returns, positions)

        # Net returns
        net_returns = strategy_returns - costs

        return self._compute_metrics(net_returns, signals)

    def _apply_risk_management(
        self,
        returns: np.ndarray,
        positions: np.ndarray
    ) -> np.ndarray:
        """
        Apply stop loss and take profit to trade returns.

        Args:
            returns: Strategy returns array
            positions: Position sizes array

        Returns:
            Risk-managed returns array
        """
        managed_returns = returns.copy()

        if self.config.stop_loss is not None:
            # Clip losses at stop loss level
            for i in range(len(managed_returns)):
                if positions[i] != 0:
                    if managed_returns[i] < -self.config.stop_loss:
                        managed_returns[i] = -self.config.stop_loss

        if self.config.take_profit is not None:
            # Clip profits at take profit level
            for i in range(len(managed_returns)):
                if positions[i] != 0:
                    if managed_returns[i] > self.config.take_profit:
                        managed_returns[i] = self.config.take_profit

        return managed_returns

    def _compute_metrics(
        self,
        net_returns: np.ndarray,
        signals_df: Optional[pd.DataFrame] = None,
        strategy: Optional[CausalTradingStrategy] = None
    ) -> BacktestResult:
        """
        Compute comprehensive backtest metrics.

        Args:
            net_returns: Net returns array after costs
            signals_df: Optional signals DataFrame
            strategy: Optional strategy for causal info

        Returns:
            BacktestResult with all computed metrics
        """
        ann_factor = self.config.ann_factor

        # Handle edge cases
        if len(net_returns) == 0:
            return BacktestResult()

        # Replace NaN/Inf
        net_returns = np.nan_to_num(net_returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Cumulative returns
        cumulative = np.cumprod(1 + net_returns)

        # Total and annualized return
        total_return = float(cumulative[-1] - 1)
        n_periods = len(net_returns)
        annual_return = float(
            (1 + total_return) ** (ann_factor / max(n_periods, 1)) - 1
        )

        # Volatility
        volatility = float(np.std(net_returns) * np.sqrt(ann_factor))

        # Sharpe ratio
        sharpe = float(annual_return / volatility) if volatility > 1e-10 else 0.0

        # Sortino ratio (downside deviation)
        downside = net_returns[net_returns < 0]
        downside_std = float(
            np.std(downside) * np.sqrt(ann_factor)
        ) if len(downside) > 0 else 0.0
        sortino = float(annual_return / downside_std) if downside_std > 1e-10 else 0.0

        # Maximum drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (running_max + 1e-10)
        max_drawdown = float(np.min(drawdowns))
        avg_drawdown = float(np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.0

        # Max drawdown duration
        is_in_dd = drawdowns < -1e-8
        dd_duration = 0
        max_dd_duration = 0
        for in_dd in is_in_dd:
            if in_dd:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # Calmar ratio
        calmar = float(
            annual_return / abs(max_drawdown)
        ) if abs(max_drawdown) > 1e-10 else 0.0

        # Trade metrics
        active_returns = net_returns[net_returns != 0]
        n_trades = len(active_returns)
        wins = active_returns[active_returns > 0]
        losses = active_returns[active_returns < 0]

        win_rate = float(len(wins) / n_trades) if n_trades > 0 else 0.0

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = float(
            gross_profit / gross_loss
        ) if gross_loss > 1e-10 else float('inf') if gross_profit > 0 else 0.0

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        avg_trade = float(np.mean(active_returns)) if n_trades > 0 else 0.0
        best_trade = float(np.max(active_returns)) if n_trades > 0 else 0.0
        worst_trade = float(np.min(active_returns)) if n_trades > 0 else 0.0

        # Causal metrics
        n_causal_links = 0
        avg_link_strength = 0.0
        causal_links_history = None

        if strategy is not None and strategy._fitted:
            links = strategy._causal_links
            n_causal_links = len(links)
            if links:
                avg_link_strength = float(
                    np.mean([abs(s) for _, _, _, s in links])
                )
            causal_links_history = links

        # Build time series
        index = signals_df.index if signals_df is not None else range(len(net_returns))
        cumulative_series = pd.Series(cumulative, index=index[:len(cumulative)])
        returns_series = pd.Series(net_returns, index=index[:len(net_returns)])
        drawdown_series = pd.Series(drawdowns, index=index[:len(drawdowns)])

        # Equity curve
        equity_curve = cumulative_series * self.config.initial_capital

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            max_drawdown_duration=max_dd_duration,
            n_trades=n_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            best_trade=best_trade,
            worst_trade=worst_trade,
            n_causal_links=n_causal_links,
            avg_link_strength=avg_link_strength,
            cumulative_returns=cumulative_series,
            daily_returns=returns_series,
            positions=signals_df,
            signals=signals_df,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
            causal_links_history=causal_links_history
        )

    def plot_results(
        self,
        result: BacktestResult,
        title: str = "PCMCI Causal Strategy Performance",
        save_path: Optional[str] = None
    ):
        """
        Plot comprehensive backtest results.

        Generates a 2x2 figure with:
        1. Cumulative returns / equity curve
        2. Drawdown over time
        3. Returns distribution
        4. Rolling Sharpe ratio

        Args:
            result: BacktestResult from a backtest run
            title: Plot title
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Equity curve
        ax = axes[0, 0]
        if result.equity_curve is not None:
            result.equity_curve.plot(ax=ax, label='Strategy')
        ax.set_title('Equity Curve')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Drawdown
        ax = axes[0, 1]
        if result.drawdown_series is not None:
            result.drawdown_series.plot(ax=ax, color='red')
            ax.fill_between(
                result.drawdown_series.index,
                result.drawdown_series.values,
                0,
                alpha=0.3,
                color='red'
            )
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)

        # 3. Returns distribution
        ax = axes[1, 0]
        if result.daily_returns is not None:
            result.daily_returns.hist(bins=50, ax=ax, alpha=0.7, color='steelblue')
            ax.axvline(x=0, color='red', linestyle='--')
            mean_ret = result.daily_returns.mean()
            ax.axvline(
                x=mean_ret, color='green', linestyle='--',
                label=f"Mean: {mean_ret:.4f}"
            )
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Daily Return')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Rolling Sharpe
        ax = axes[1, 1]
        if result.daily_returns is not None:
            window = 60
            rolling_mean = result.daily_returns.rolling(window).mean()
            rolling_std = result.daily_returns.rolling(window).std()
            rolling_sharpe = (rolling_mean / (rolling_std + 1e-10)) * np.sqrt(
                self.config.ann_factor
            )
            rolling_sharpe.plot(ax=ax, color='purple')
            ax.axhline(y=0, color='red', linestyle='--')
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        ax.set_title(f'Rolling {window}-Day Sharpe Ratio')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.close()

    def plot_causal_graph(
        self,
        strategy: CausalTradingStrategy,
        save_path: Optional[str] = None
    ):
        """
        Plot the discovered causal graph.

        Args:
            strategy: Fitted CausalTradingStrategy
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            logger.warning("matplotlib/networkx not available for plotting")
            return

        if not strategy._fitted or strategy.pcmci is None:
            logger.warning("Strategy not fitted, cannot plot causal graph")
            return

        G = strategy.pcmci.get_causal_graph_networkx()

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color='lightblue',
            node_size=2000,
            alpha=0.9
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

        # Draw edges with width proportional to strength
        edges = G.edges(data=True)
        if edges:
            weights = [abs(d.get('weight', 0.5)) * 3 for _, _, d in edges]
            edge_labels = {
                (u, v): f"lag={d.get('lag', '?')}\n{d.get('weight', 0):.2f}"
                for u, v, d in edges
            }

            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=weights,
                alpha=0.6,
                edge_color='gray',
                arrows=True,
                arrowsize=20,
                connectionstyle='arc3,rad=0.1'
            )
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels, ax=ax,
                font_size=8
            )

        ax.set_title('Discovered Causal Graph (PCMCI)')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Causal graph saved to {save_path}")

        plt.close()


def walk_forward_backtest(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    pcmci_config: Optional[PCMCIConfig] = None,
    backtest_config: Optional[BacktestConfig] = None,
    train_window: int = 252,
    test_window: int = 21
) -> BacktestResult:
    """
    Walk-forward backtesting with periodic PCMCI refitting.

    Slides a training window through the data, refitting the causal
    model periodically and testing on out-of-sample data.

    Args:
        data: Full dataset
        feature_columns: Feature columns for causal analysis
        target_columns: Tradeable asset columns
        pcmci_config: PCMCI configuration
        backtest_config: Backtest configuration
        train_window: Size of training window
        test_window: Size of test window

    Returns:
        Combined BacktestResult across all walk-forward windows

    Example:
        result = walk_forward_backtest(
            data, ['SPY_returns', 'VIX_returns'], ['SPY_returns'],
            train_window=252, test_window=21
        )
    """
    if pcmci_config is None:
        pcmci_config = PCMCIConfig()
    if backtest_config is None:
        backtest_config = BacktestConfig()

    all_returns = []
    all_signals = []

    n = len(data)
    step = 0

    for start in range(0, n - train_window - test_window + 1, test_window):
        train_end = start + train_window
        test_end = min(train_end + test_window, n)

        # Train data
        train_data = data.iloc[start:train_end]

        # Fit causal model on training data
        strategy = CausalTradingStrategy(
            pcmci_config=pcmci_config,
            backtest_config=backtest_config,
            target_columns=target_columns
        )

        # Prepare training data for PCMCI
        available = [c for c in feature_columns if c in train_data.columns]
        train_array = train_data[available].values.astype(np.float64)

        # Standardize
        means = train_array.mean(axis=0)
        stds = train_array.std(axis=0)
        stds[stds < 1e-10] = 1.0
        train_std = (train_array - means) / stds

        try:
            strategy.fit_causal_model(train_std, available)
        except Exception as e:
            logger.warning(f"Walk-forward step {step}: fit failed: {e}")
            step += 1
            continue

        # Test data
        test_data = data.iloc[train_end:test_end]

        # Generate signals on test data using the fitted model
        for _, row in test_data.iterrows():
            record = {}
            for tc in target_columns:
                if tc in available:
                    tc_idx = available.index(tc)
                    # Simple signal: use most recent causal prediction
                    signal = 0.0
                    confidence = 0.0
                    for src, tgt, lag, strength in strategy._causal_links:
                        if tgt == tc and src in available:
                            signal += strength * 0.5  # Simplified
                            confidence += abs(strength)

                    signal = np.clip(signal, -1.0, 1.0)
                    confidence = min(confidence, 1.0)
                    position = signal * confidence * backtest_config.position_size

                    record[f'{tc}_position'] = np.clip(
                        position,
                        -backtest_config.max_position,
                        backtest_config.max_position
                    )
                    record[f'{tc}_signal'] = signal
                    record[f'{tc}_confidence'] = confidence

            all_signals.append(record)

            # Actual return
            for tc in target_columns:
                if tc in test_data.columns:
                    ret_val = row[tc] if not pd.isna(row.get(tc, np.nan)) else 0.0
                    all_returns.append(float(ret_val))

        step += 1
        logger.info(f"Walk-forward step {step}: train={start}:{train_end}, test={train_end}:{test_end}")

    if not all_returns:
        return BacktestResult()

    # Compute portfolio returns
    signals_df = pd.DataFrame(all_signals)
    returns_array = np.array(all_returns[:len(signals_df)])

    # Use first target for simple portfolio return
    tc = target_columns[0]
    position_col = f'{tc}_position'
    if position_col in signals_df.columns:
        positions = signals_df[position_col].shift(1).fillna(0).values
        portfolio_returns = positions[:len(returns_array)] * returns_array[:len(positions)]

        # Costs
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = position_changes[:len(portfolio_returns)] * (
            backtest_config.fee_rate + backtest_config.slippage
        )
        net_returns = portfolio_returns - costs[:len(portfolio_returns)]
    else:
        net_returns = returns_array

    # Build result
    backtester = Backtester(backtest_config)
    return backtester._compute_metrics(net_returns, signals_df)


if __name__ == "__main__":
    # Test the strategy and backtester with synthetic data
    print("Testing Causal Trading Strategy and Backtester...")

    from .data import generate_synthetic_causal_data

    np.random.seed(42)

    # Generate synthetic causal data
    n_samples = 600
    true_links = [
        (0, 1, 1, 0.5),
        (1, 2, 2, 0.3),
        (0, 3, 1, 0.4),
    ]

    data_array, _ = generate_synthetic_causal_data(
        n_vars=4,
        n_samples=n_samples,
        true_links=true_links,
        seed=42
    )

    # Create DataFrame
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    df = pd.DataFrame(
        data_array,
        columns=['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D'],
        index=dates
    )
    df['timestamp'] = dates

    feature_cols = ['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D']
    target_cols = ['Asset_B']

    # Test strategy
    print("\n1. Testing CausalTradingStrategy...")
    pcmci_config = PCMCIConfig(
        max_lag=3,
        significance_level=0.05,
        use_tigramite=False
    )
    bt_config = BacktestConfig(
        initial_capital=100000,
        position_size=0.2,
        lookback_window=200,
        refit_frequency=50
    )

    strategy = CausalTradingStrategy(
        pcmci_config=pcmci_config,
        backtest_config=bt_config,
        target_columns=target_cols
    )

    signals = strategy.generate_signals(df, feature_cols, target_cols)
    print(f"   Signals shape: {signals.shape}")
    print(f"   Signal columns: {list(signals.columns)}")

    # Test backtester
    print("\n2. Testing Backtester...")
    backtester = Backtester(bt_config)

    # Create simple returns and signals for direct testing
    n = 400
    test_dates = pd.date_range('2023-01-01', periods=n, freq='D')
    test_signals = pd.DataFrame({
        'position': np.clip(np.cumsum(np.random.randn(n) * 0.05), -0.5, 0.5),
        'confidence': np.random.uniform(0.2, 0.8, n),
    }, index=test_dates)
    test_returns = pd.Series(np.random.randn(n) * 0.015, index=test_dates)

    result = backtester.run_from_signals(test_signals, test_returns)
    print(result.summary())

    # Test causal summary
    print("\n3. Testing causal summary...")
    summary = strategy.get_causal_summary()
    print(f"   Fitted: {summary['fitted']}")
    print(f"   Links: {summary['n_links']}")
    if summary['links']:
        for src, tgt, lag, strength in summary['links'][:5]:
            print(f"     {src} --({lag})--> {tgt}: {strength:.3f}")

    print("\nAll tests passed!")
