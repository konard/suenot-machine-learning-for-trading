"""
Trading Strategy Module Based on Mediation Analysis

This module provides trading strategies and backtesting tools
that leverage mediation analysis insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from .model import baron_kenny_mediation, MediationResults


@dataclass
class MediationSignal:
    """Signal based on mediation pathway activation"""
    timestamp: pd.Timestamp
    treatment: str
    mediator: str
    target: str

    direct_strength: float
    indirect_strength: float
    mediator_activated: bool

    signal: int  # -1, 0, 1 (short, neutral, long)
    confidence: float


class MediationTradingStrategy:
    """
    Trading strategy that exploits mediation pathway dynamics.

    Core insight: When the mediator channel is activated, we expect
    the full effect to materialize. When blocked, only direct effect works.
    """

    def __init__(
        self,
        lookback_window: int = 60,
        mediation_threshold: float = 0.3,
        signal_threshold: float = 0.5,
        min_observations: int = 30
    ):
        """
        Initialize strategy.

        Parameters:
        -----------
        lookback_window : Window for estimating mediation parameters
        mediation_threshold : Min proportion mediated to consider pathway active
        signal_threshold : Min signal strength to trade
        min_observations : Minimum observations before generating signals
        """
        self.lookback = lookback_window
        self.med_threshold = mediation_threshold
        self.signal_threshold = signal_threshold
        self.min_obs = min_observations

        # History buffers
        self.history = {
            'X': [],
            'M': [],
            'Y': []
        }

        # Current parameter estimates
        self.params = {
            'a': None,
            'b': None,
            'c_prime': None,
            'indirect': None,
            'proportion_mediated': None
        }

    def update(
        self,
        X: float,
        M: float,
        Y: float,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Optional[MediationSignal]:
        """
        Update strategy with new observation and generate signal.

        Parameters:
        -----------
        X : Treatment value
        M : Mediator value
        Y : Outcome value
        timestamp : Optional timestamp

        Returns:
        --------
        MediationSignal if enough data, None otherwise
        """
        # Update history
        self.history['X'].append(X)
        self.history['M'].append(M)
        self.history['Y'].append(Y)

        # Keep only lookback window
        for key in self.history:
            if len(self.history[key]) > self.lookback:
                self.history[key] = self.history[key][-self.lookback:]

        # Need minimum observations
        if len(self.history['X']) < self.min_obs:
            return None

        # Estimate mediation parameters
        self._estimate_parameters()

        # Generate signal
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        return self._generate_signal(X, M, timestamp)

    def _estimate_parameters(self):
        """Estimate mediation parameters from history"""
        X = np.array(self.history['X'])
        M = np.array(self.history['M'])
        Y = np.array(self.history['Y'])

        # Use lagged values for estimation
        X_lag = X[:-1]
        M_lag = M[:-1]
        Y_curr = Y[1:]
        M_curr = M[1:]

        n = len(X_lag)
        if n < 10:
            return

        # Standardize
        X_mean, X_std = X_lag.mean(), X_lag.std() + 1e-10
        M_mean, M_std = M_lag.mean(), M_lag.std() + 1e-10
        Y_mean, Y_std = Y_curr.mean(), Y_curr.std() + 1e-10

        X_norm = (X_lag - X_mean) / X_std
        M_norm = (M_lag - M_mean) / M_std
        M_curr_norm = (M_curr - M_mean) / M_std
        Y_norm = (Y_curr - Y_mean) / Y_std

        # Simple regression for a path: M ~ X
        try:
            X_design = np.column_stack([np.ones(n), X_norm])
            beta_a = np.linalg.lstsq(X_design, M_curr_norm, rcond=None)[0]
            self.params['a'] = beta_a[1]
        except:
            self.params['a'] = 0

        # Multiple regression for b and c' paths: Y ~ X + M
        try:
            XM_design = np.column_stack([np.ones(n), X_norm, M_norm])
            beta_y = np.linalg.lstsq(XM_design, Y_norm, rcond=None)[0]
            self.params['c_prime'] = beta_y[1]
            self.params['b'] = beta_y[2]
        except:
            self.params['c_prime'] = 0
            self.params['b'] = 0

        # Indirect effect
        self.params['indirect'] = self.params['a'] * self.params['b']

        # Proportion mediated
        total = self.params['c_prime'] + self.params['indirect']
        if abs(total) > 0.01:
            self.params['proportion_mediated'] = self.params['indirect'] / total
        else:
            self.params['proportion_mediated'] = 0

    def _generate_signal(
        self,
        X: float,
        M: float,
        timestamp: pd.Timestamp
    ) -> MediationSignal:
        """Generate trading signal based on mediation analysis"""
        # Standardize current values
        X_arr = np.array(self.history['X'])
        M_arr = np.array(self.history['M'])

        X_mean, X_std = X_arr.mean(), X_arr.std() + 1e-10
        M_mean, M_std = M_arr.mean(), M_arr.std() + 1e-10

        X_norm = (X - X_mean) / X_std
        M_norm = (M - M_mean) / M_std

        # Expected mediator value given treatment
        M_expected = self.params['a'] * X_norm if self.params['a'] is not None else 0

        # Is mediator activated (stronger than expected)?
        mediator_activated = abs(M_norm) > abs(M_expected) * 1.2

        # Calculate expected effect
        direct = self.params['c_prime'] if self.params['c_prime'] is not None else 0
        indirect = self.params['indirect'] if self.params['indirect'] is not None else 0
        prop_med = self.params['proportion_mediated'] if self.params['proportion_mediated'] is not None else 0

        if mediator_activated and abs(prop_med) > self.med_threshold:
            # Full effect expected (both direct and indirect)
            expected_effect = (direct + indirect) * X_norm
            confidence = 0.8
        else:
            # Only direct effect expected
            expected_effect = direct * X_norm
            confidence = 0.5

        # Generate signal
        if abs(expected_effect) < self.signal_threshold:
            signal = 0  # Neutral
        elif expected_effect > 0:
            signal = 1  # Long
        else:
            signal = -1  # Short

        return MediationSignal(
            timestamp=timestamp,
            treatment='X',
            mediator='M',
            target='Y',
            direct_strength=direct,
            indirect_strength=indirect,
            mediator_activated=mediator_activated,
            signal=signal,
            confidence=confidence * abs(expected_effect)
        )

    def get_parameters(self) -> Dict:
        """Get current parameter estimates"""
        return self.params.copy()

    def reset(self):
        """Reset strategy state"""
        self.history = {'X': [], 'M': [], 'Y': []}
        self.params = {
            'a': None, 'b': None, 'c_prime': None,
            'indirect': None, 'proportion_mediated': None
        }


@dataclass
class BacktestConfig:
    """Configuration for backtest"""
    initial_capital: float = 100000
    transaction_cost_bps: float = 10
    slippage_bps: float = 5
    max_position_size: float = 0.2
    rebalance_frequency: str = 'daily'


class MediationBacktester:
    """Backtest mediation-based trading strategies"""

    def __init__(
        self,
        strategy: MediationTradingStrategy,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtester.

        Parameters:
        -----------
        strategy : MediationTradingStrategy instance
        config : BacktestConfig (uses defaults if None)
        """
        self.strategy = strategy
        self.config = config or BacktestConfig()

    def run(
        self,
        data: pd.DataFrame,
        x_col: str = 'X',
        m_col: str = 'M',
        y_col: str = 'Y'
    ) -> pd.DataFrame:
        """
        Run backtest.

        Parameters:
        -----------
        data : DataFrame with columns for X, M, Y
        x_col, m_col, y_col : Column names

        Returns:
        --------
        DataFrame with backtest results
        """
        results = []
        position = 0
        capital = self.config.initial_capital
        peak_value = capital

        # Reset strategy
        self.strategy.reset()

        for i, (idx, row) in enumerate(data.iterrows()):
            # Update strategy and get signal
            signal = self.strategy.update(
                X=row[x_col],
                M=row[m_col],
                Y=row[y_col],
                timestamp=idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
            )

            if signal is None:
                results.append({
                    'timestamp': idx,
                    'signal': 0,
                    'confidence': 0,
                    'direct_effect': 0,
                    'indirect_effect': 0,
                    'mediator_active': False,
                    'proportion_mediated': 0,
                    'position': 0,
                    'return': 0,
                    'portfolio_value': capital,
                    'drawdown': 0
                })
                continue

            # Calculate position based on signal and confidence
            confidence_threshold = 0.3
            if signal.confidence > confidence_threshold:
                target_position = signal.signal * self.config.max_position_size * capital
            else:
                target_position = 0

            # Transaction costs
            trade_size = abs(target_position - position)
            tc_bps = self.config.transaction_cost_bps + self.config.slippage_bps
            tc = trade_size * tc_bps / 10000

            # P&L from previous position
            if position != 0:
                pnl = position * row[y_col] - tc
            else:
                pnl = -tc if tc > 0 else 0

            # Update capital
            capital += pnl

            # Track drawdown
            peak_value = max(peak_value, capital)
            drawdown = (peak_value - capital) / peak_value

            # Get proportion mediated
            params = self.strategy.get_parameters()
            prop_med = params.get('proportion_mediated', 0) or 0

            results.append({
                'timestamp': idx,
                'signal': signal.signal,
                'confidence': signal.confidence,
                'direct_effect': signal.direct_strength,
                'indirect_effect': signal.indirect_strength,
                'mediator_active': signal.mediator_activated,
                'proportion_mediated': prop_med,
                'position': target_position,
                'return': pnl / (capital - pnl + 1e-10) if capital > pnl else 0,
                'portfolio_value': capital,
                'drawdown': drawdown
            })

            position = target_position

        results_df = pd.DataFrame(results)
        if 'timestamp' in results_df.columns:
            results_df.set_index('timestamp', inplace=True)

        return results_df

    def analyze_results(self, results: pd.DataFrame) -> Dict:
        """
        Analyze backtest results.

        Parameters:
        -----------
        results : DataFrame from run()

        Returns:
        --------
        Dictionary with performance metrics
        """
        if results.empty:
            return {'error': 'No results to analyze'}

        returns = results['return'].dropna()
        portfolio_values = results['portfolio_value']

        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        n_days = len(returns)

        if n_days < 2:
            return {'error': 'Not enough data points'}

        # Annualized metrics
        annual_factor = 252 / n_days
        annual_return = (1 + total_return) ** annual_factor - 1
        annual_vol = returns.std() * np.sqrt(252)

        # Risk metrics
        sharpe = annual_return / (annual_vol + 1e-10)
        max_dd = results['drawdown'].max()
        calmar = annual_return / (max_dd + 1e-10)

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252)
            sortino = annual_return / (downside_vol + 1e-10)
        else:
            sortino = float('inf')

        # Win rate
        trading_returns = returns[returns != 0]
        if len(trading_returns) > 0:
            win_rate = (trading_returns > 0).mean()
        else:
            win_rate = 0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / (gross_loss + 1e-10)

        # Trading statistics
        n_trades = (results['signal'].diff().abs() > 0).sum()
        avg_position = results['position'].abs().mean()

        # Mediation-specific analytics
        mediation_analytics = {}

        if 'proportion_mediated' in results.columns:
            high_med = results[results['proportion_mediated'] > 0.5]
            low_med = results[results['proportion_mediated'] <= 0.5]

            if len(high_med) > 0:
                mediation_analytics['return_high_mediation'] = high_med['return'].mean() * 252
                mediation_analytics['sharpe_high_mediation'] = (
                    high_med['return'].mean() / (high_med['return'].std() + 1e-10) * np.sqrt(252)
                )
            else:
                mediation_analytics['return_high_mediation'] = 0
                mediation_analytics['sharpe_high_mediation'] = 0

            if len(low_med) > 0:
                mediation_analytics['return_low_mediation'] = low_med['return'].mean() * 252
                mediation_analytics['sharpe_low_mediation'] = (
                    low_med['return'].mean() / (low_med['return'].std() + 1e-10) * np.sqrt(252)
                )
            else:
                mediation_analytics['return_low_mediation'] = 0
                mediation_analytics['sharpe_low_mediation'] = 0

        if 'mediator_active' in results.columns:
            active = results[results['mediator_active'] == True]
            inactive = results[results['mediator_active'] == False]

            mediation_analytics['return_mediator_active'] = (
                active['return'].mean() * 252 if len(active) > 0 else 0
            )
            mediation_analytics['return_mediator_inactive'] = (
                inactive['return'].mean() * 252 if len(inactive) > 0 else 0
            )

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_trades': n_trades,
            'avg_position_size': avg_position,
            **mediation_analytics
        }

    def _max_drawdown(self, cumulative: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = cumulative.cummax()
        drawdown = cumulative - rolling_max
        return abs(drawdown.min())
