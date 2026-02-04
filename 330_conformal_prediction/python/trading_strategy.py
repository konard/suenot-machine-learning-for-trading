"""
Trading Strategy with Conformal Prediction

Implements uncertainty-aware position sizing and signal generation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from conformal_predictor import SplitConformalPredictor, PredictionInterval


class SignalType(Enum):
    """Trading signal types."""
    LONG = 1
    SHORT = -1
    HOLD = 0


@dataclass
class TradingSignal:
    """Represents a trading signal with uncertainty information."""
    timestamp: pd.Timestamp
    signal_type: SignalType
    point_prediction: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    confidence: float
    position_size: float

    @property
    def direction(self) -> str:
        """Return signal direction as string."""
        return self.signal_type.name


class ConformalTradingStrategy:
    """
    Trading strategy using conformal prediction intervals.

    Key principles:
    1. Only trade when confident (narrow prediction intervals)
    2. Position size inversely proportional to uncertainty
    3. Enter long when entire interval is positive
    4. Enter short when entire interval is negative
    """

    def __init__(
        self,
        conformal_predictor: SplitConformalPredictor,
        max_position: float = 1.0,
        min_confidence: float = 0.5,
        volatility_threshold: float = 0.02,
        position_scaling: str = 'inverse_width'
    ):
        """
        Initialize conformal trading strategy.

        Args:
            conformal_predictor: Calibrated conformal predictor
            max_position: Maximum position size (fraction of capital)
            min_confidence: Minimum confidence to trade
            volatility_threshold: Maximum interval width to trade
            position_scaling: Method for scaling positions
                             ('inverse_width', 'kelly', 'fixed')
        """
        self.conformal_predictor = conformal_predictor
        self.max_position = max_position
        self.min_confidence = min_confidence
        self.volatility_threshold = volatility_threshold
        self.position_scaling = position_scaling

        self.historical_volatility: Optional[float] = None

    def set_historical_volatility(self, returns: np.ndarray) -> None:
        """
        Set historical volatility for normalization.

        Args:
            returns: Array of historical returns
        """
        self.historical_volatility = np.std(returns)

    def generate_signal(
        self,
        X: np.ndarray,
        timestamp: pd.Timestamp
    ) -> TradingSignal:
        """
        Generate trading signal for a single observation.

        Args:
            X: Feature array (1D or 2D)
            timestamp: Timestamp of observation

        Returns:
            TradingSignal object
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        intervals = self.conformal_predictor.predict_interval(X)
        interval = intervals[0]

        # Normalize interval width
        if self.historical_volatility and self.historical_volatility > 0:
            normalized_width = interval.width / self.historical_volatility
        else:
            normalized_width = interval.width

        # Calculate confidence (inverse of normalized width)
        confidence = 1.0 / (1.0 + normalized_width)

        # Determine signal type
        signal_type = self._determine_signal(
            interval, normalized_width, confidence
        )

        # Calculate position size
        position_size = self._calculate_position_size(
            signal_type, interval, confidence
        )

        return TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            point_prediction=interval.point_prediction,
            lower_bound=interval.lower_bound,
            upper_bound=interval.upper_bound,
            interval_width=interval.width,
            confidence=confidence,
            position_size=position_size
        )

    def _determine_signal(
        self,
        interval: PredictionInterval,
        normalized_width: float,
        confidence: float
    ) -> SignalType:
        """Determine signal type based on prediction interval."""

        # Check confidence threshold
        if confidence < self.min_confidence:
            return SignalType.HOLD

        # Check volatility threshold
        if normalized_width > self.volatility_threshold:
            return SignalType.HOLD

        # Strong signal: entire interval is positive or negative
        if interval.lower_bound > 0:
            return SignalType.LONG

        elif interval.upper_bound < 0:
            return SignalType.SHORT

        # Weaker signal: point prediction is significant
        pred_threshold = 0.001  # 0.1% return threshold

        if interval.point_prediction > pred_threshold and confidence > 0.6:
            return SignalType.LONG

        elif interval.point_prediction < -pred_threshold and confidence > 0.6:
            return SignalType.SHORT

        return SignalType.HOLD

    def _calculate_position_size(
        self,
        signal_type: SignalType,
        interval: PredictionInterval,
        confidence: float
    ) -> float:
        """Calculate position size based on uncertainty."""

        if signal_type == SignalType.HOLD:
            return 0.0

        if self.position_scaling == 'fixed':
            return self.max_position * signal_type.value

        elif self.position_scaling == 'inverse_width':
            # Position inversely proportional to interval width
            scale = confidence ** 2  # Square for stronger effect
            position = self.max_position * scale
            return np.clip(position, 0, self.max_position) * signal_type.value

        elif self.position_scaling == 'kelly':
            # Kelly-inspired sizing
            expected_return = abs(interval.point_prediction)
            uncertainty = interval.width / 2

            if uncertainty > 0:
                kelly_fraction = expected_return / (uncertainty ** 2)
                kelly_fraction = min(kelly_fraction, 1.0)  # Cap at 1
                position = self.max_position * kelly_fraction * confidence
            else:
                position = self.max_position * confidence

            return np.clip(position, 0, self.max_position) * signal_type.value

        return 0.0

    def generate_signals_batch(
        self,
        X: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> List[TradingSignal]:
        """
        Generate signals for multiple observations.

        Args:
            X: Feature array (2D)
            timestamps: Array of timestamps

        Returns:
            List of TradingSignal objects
        """
        signals = []

        for i in range(len(X)):
            signal = self.generate_signal(X[i], timestamps[i])
            signals.append(signal)

        return signals


class PortfolioManager:
    """
    Manages portfolio based on trading signals.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_leverage: float = 1.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005  # 0.05% slippage
    ):
        """
        Initialize portfolio manager.

        Args:
            initial_capital: Starting capital
            max_leverage: Maximum leverage allowed
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self.position = 0.0
        self.entry_price = 0.0

        self.trades: List[dict] = []
        self.equity_curve: List[float] = [initial_capital]

    def execute_signal(
        self,
        signal: TradingSignal,
        current_price: float
    ) -> Optional[dict]:
        """
        Execute trading signal.

        Args:
            signal: Trading signal to execute
            current_price: Current market price

        Returns:
            Trade record if trade executed, None otherwise
        """
        target_position = signal.position_size

        # Limit by max leverage
        target_position = np.clip(
            target_position,
            -self.max_leverage,
            self.max_leverage
        )

        # Calculate position change
        position_change = target_position - self.position

        if abs(position_change) < 0.01:  # Minimum position change
            return None

        # Apply slippage
        if position_change > 0:  # Buying
            execution_price = current_price * (1 + self.slippage)
        else:  # Selling
            execution_price = current_price * (1 - self.slippage)

        # Calculate transaction cost
        trade_value = abs(position_change) * self.capital
        cost = trade_value * self.transaction_cost

        # Update capital for costs
        self.capital -= cost

        # Close existing position P&L
        if self.position != 0:
            pnl = self.position * (current_price - self.entry_price) * self.capital
            self.capital += pnl

        # Update position
        self.position = target_position
        self.entry_price = execution_price

        trade = {
            'timestamp': signal.timestamp,
            'signal_type': signal.direction,
            'position_change': position_change,
            'execution_price': execution_price,
            'position': self.position,
            'cost': cost,
            'capital': self.capital,
            'confidence': signal.confidence
        }

        self.trades.append(trade)

        return trade

    def update_equity(self, current_price: float) -> float:
        """
        Update equity based on current price.

        Args:
            current_price: Current market price

        Returns:
            Current equity value
        """
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl = self.position * (current_price / self.entry_price - 1) * self.capital
            equity = self.capital + unrealized_pnl
        else:
            equity = self.capital

        self.equity_curve.append(equity)
        return equity

    def get_performance_metrics(self) -> dict:
        """Calculate portfolio performance metrics."""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        total_return = (equity[-1] / equity[0]) - 1

        # Sharpe ratio (annualized, assuming hourly data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        else:
            sharpe = 0.0

        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and np.std(negative_returns) > 0:
            sortino = np.mean(returns) / np.std(negative_returns) * np.sqrt(252 * 24)
        else:
            sortino = 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        # Win rate
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            # Simple win rate based on position change direction
            winning_trades = sum(1 for t in self.trades
                               if t.get('position_change', 0) * t.get('capital', 0) > 0)
            win_rate = winning_trades / len(self.trades)
        else:
            win_rate = 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'final_capital': equity[-1],
            'initial_capital': equity[0]
        }


def signals_to_dataframe(signals: List[TradingSignal]) -> pd.DataFrame:
    """Convert list of signals to DataFrame."""
    data = []
    for s in signals:
        data.append({
            'timestamp': s.timestamp,
            'signal': s.direction,
            'point_prediction': s.point_prediction,
            'lower_bound': s.lower_bound,
            'upper_bound': s.upper_bound,
            'interval_width': s.interval_width,
            'confidence': s.confidence,
            'position_size': s.position_size
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import GradientBoostingRegressor
    from conformal_predictor import SplitConformalPredictor

    np.random.seed(42)

    # Generate synthetic price data
    n = 500
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    returns = np.diff(prices) / prices[:-1]

    # Create features (simple example)
    X = np.column_stack([
        returns[:-5],  # Lag 1
        np.roll(returns, 1)[:-5],  # Lag 2
        np.roll(returns, 2)[:-5],  # Lag 3
    ])
    y = returns[5:]  # Future return

    # Split data
    train_end = 300
    cal_end = 400

    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:cal_end], y[train_end:cal_end]
    X_test, y_test = X[cal_end:], y[cal_end:]

    # Create conformal predictor
    base_model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
    cp = SplitConformalPredictor(base_model, alpha=0.1)
    cp.fit_calibrate(X_train, y_train, X_cal, y_cal)

    # Create strategy
    strategy = ConformalTradingStrategy(
        cp,
        max_position=1.0,
        min_confidence=0.3,
        volatility_threshold=0.05
    )
    strategy.set_historical_volatility(y_train)

    # Generate signals
    timestamps = pd.date_range('2024-01-01', periods=len(X_test), freq='1h')
    signals = strategy.generate_signals_batch(X_test, timestamps)

    # Analyze signals
    signals_df = signals_to_dataframe(signals)
    print("\nSignal Distribution:")
    print(signals_df['signal'].value_counts())

    print("\nSample Signals:")
    print(signals_df.head(10))

    # Run simple backtest
    portfolio = PortfolioManager(initial_capital=10000)
    test_prices = prices[cal_end + 5:]

    for signal, price in zip(signals, test_prices):
        portfolio.execute_signal(signal, price)
        portfolio.update_equity(price)

    metrics = portfolio.get_performance_metrics()
    print("\nBacktest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
