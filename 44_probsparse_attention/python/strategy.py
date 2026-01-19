"""
Trading Strategy and Backtesting for Informer Model

Provides:
- SignalGenerator: Generate trading signals from model predictions
- InformerStrategy: Trading strategy based on Informer forecasts
- BacktestEngine: Backtesting framework with metrics
- BacktestResult: Container for backtest results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class TradeRecord:
    """Record of a single trade"""
    timestamp: datetime
    signal: SignalType
    price: float
    position_size: float
    pnl: float
    cumulative_pnl: float
    predicted_return: float
    actual_return: float


@dataclass
class BacktestResult:
    """Container for backtest results"""
    trades: List[TradeRecord]
    equity_curve: np.ndarray
    metrics: Dict[str, float]
    config: Dict

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame"""
        return pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'signal': t.signal.value,
                'price': t.price,
                'position_size': t.position_size,
                'pnl': t.pnl,
                'cumulative_pnl': t.cumulative_pnl,
                'predicted_return': t.predicted_return,
                'actual_return': t.actual_return
            }
            for t in self.trades
        ])

    def summary(self) -> str:
        """Generate summary string"""
        lines = [
            "=" * 50,
            "BACKTEST RESULTS",
            "=" * 50,
            f"Total Return: {self.metrics['total_return']:.2%}",
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}",
            f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}",
            f"Max Drawdown: {self.metrics['max_drawdown']:.2%}",
            f"Win Rate: {self.metrics['win_rate']:.2%}",
            f"Profit Factor: {self.metrics['profit_factor']:.2f}",
            f"Number of Trades: {self.metrics['num_trades']:.0f}",
            "=" * 50
        ]
        return "\n".join(lines)


class SignalGenerator:
    """
    Generate trading signals from model predictions

    Supports multiple signal generation methods:
    - Threshold-based: Signal when predicted return exceeds threshold
    - Quantile-based: Signal based on confidence intervals
    - Direction-based: Direct classification output
    """

    def __init__(
        self,
        method: str = 'threshold',
        long_threshold: float = 0.0005,  # 0.05% for crypto
        short_threshold: float = -0.0005,
        confidence_threshold: float = 0.6
    ):
        """
        Args:
            method: 'threshold', 'quantile', or 'direction'
            long_threshold: Minimum predicted return for long signal
            short_threshold: Maximum predicted return for short signal
            confidence_threshold: Minimum confidence for trading (quantile method)
        """
        self.method = method
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.confidence_threshold = confidence_threshold

    def generate(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        confidence: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> List[SignalType]:
        """
        Generate trading signals from predictions

        Args:
            predictions: Model predictions
            confidence: Optional confidence scores (for quantile method)

        Returns:
            List of trading signals
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if confidence is not None and isinstance(confidence, torch.Tensor):
            confidence = confidence.detach().cpu().numpy()

        signals = []

        # Use first prediction step for immediate signals
        if len(predictions.shape) > 1:
            pred_values = predictions[:, 0]  # First step prediction
        else:
            pred_values = predictions

        for i, pred in enumerate(pred_values):
            if self.method == 'threshold':
                signal = self._threshold_signal(pred)
            elif self.method == 'quantile':
                conf = confidence[i] if confidence is not None else 1.0
                signal = self._quantile_signal(pred, conf)
            elif self.method == 'direction':
                signal = self._direction_signal(predictions[i])
            else:
                signal = SignalType.FLAT

            signals.append(signal)

        return signals

    def _threshold_signal(self, pred: float) -> SignalType:
        """Generate signal based on threshold"""
        if pred > self.long_threshold:
            return SignalType.LONG
        elif pred < self.short_threshold:
            return SignalType.SHORT
        return SignalType.FLAT

    def _quantile_signal(self, pred: float, confidence: float) -> SignalType:
        """Generate signal based on prediction and confidence"""
        if confidence < self.confidence_threshold:
            return SignalType.FLAT
        return self._threshold_signal(pred)

    def _direction_signal(self, probs: np.ndarray) -> SignalType:
        """Generate signal from direction probabilities [down, flat, up]"""
        direction = np.argmax(probs)
        if direction == 2:  # up
            return SignalType.LONG
        elif direction == 0:  # down
            return SignalType.SHORT
        return SignalType.FLAT


class InformerStrategy:
    """
    Trading strategy based on Informer model predictions

    Features:
    - Position sizing based on confidence
    - Stop-loss and take-profit
    - Maximum position limits
    """

    def __init__(
        self,
        model: nn.Module,
        signal_generator: SignalGenerator,
        max_position: float = 1.0,
        use_confidence_sizing: bool = True,
        stop_loss: float = 0.02,  # 2% stop loss
        take_profit: float = 0.05  # 5% take profit
    ):
        """
        Args:
            model: Trained Informer model
            signal_generator: Signal generation module
            max_position: Maximum position size (fraction of capital)
            use_confidence_sizing: Size positions based on model confidence
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
        """
        self.model = model
        self.signal_generator = signal_generator
        self.max_position = max_position
        self.use_confidence_sizing = use_confidence_sizing
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def get_signal(
        self,
        X: torch.Tensor,
        current_price: float
    ) -> Tuple[SignalType, float, float]:
        """
        Get trading signal and position size

        Args:
            X: Input tensor [1, seq_len, features]
            current_price: Current market price

        Returns:
            signal: Trading signal
            position_size: Recommended position size (0 to max_position)
            predicted_return: First step prediction value
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            predictions = output['predictions']
            confidence = output.get('confidence')

        signals = self.signal_generator.generate(predictions, confidence)
        signal = signals[0]

        # Extract predicted return from first step
        pred_return = predictions[0, 0].item() if predictions.dim() > 1 else predictions[0].item()

        # Position sizing
        if signal == SignalType.FLAT:
            position_size = 0.0
        elif self.use_confidence_sizing and confidence is not None:
            conf = confidence[0].item() if isinstance(confidence, torch.Tensor) else confidence[0]
            position_size = self.max_position * conf
        else:
            position_size = self.max_position

        return signal, position_size, pred_return


class BacktestEngine:
    """
    Backtesting engine for trading strategies

    Features:
    - Transaction costs
    - Slippage modeling
    - Performance metrics
    - Equity curve tracking
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
        risk_free_rate: float = 0.02  # 2% annual risk-free rate
    ):
        """
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate

    def run(
        self,
        strategy: InformerStrategy,
        test_data: pd.DataFrame,
        X_test: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 24
    ) -> BacktestResult:
        """
        Run backtest

        Args:
            strategy: Trading strategy
            test_data: DataFrame with OHLCV data
            X_test: Prepared input sequences
            seq_len: Input sequence length
            pred_len: Prediction horizon

        Returns:
            BacktestResult with trades and metrics
        """
        capital = self.initial_capital
        position = SignalType.FLAT
        position_size = 0.0
        entry_price = 0.0

        trades = []
        equity_curve = [capital]

        for i in range(len(X_test)):
            # Get current data
            X = torch.FloatTensor(X_test[i:i+1])
            current_idx = seq_len + i
            if current_idx >= len(test_data) - 1:
                break

            current_price = test_data.iloc[current_idx]['close']
            next_price = test_data.iloc[current_idx + 1]['close']
            actual_return = np.log(next_price / current_price)
            timestamp = test_data.iloc[current_idx]['timestamp']

            # Get new signal (includes prediction to avoid redundant inference)
            new_signal, new_size, pred_return = strategy.get_signal(X, current_price)

            # Check stop-loss / take-profit
            if position != SignalType.FLAT and entry_price > 0:
                price_change = (current_price - entry_price) / entry_price
                if position == SignalType.SHORT:
                    price_change = -price_change

                if price_change < -strategy.stop_loss:
                    new_signal = SignalType.FLAT
                    new_size = 0.0
                    logger.debug(f"Stop-loss triggered at {timestamp}")
                elif price_change > strategy.take_profit:
                    new_signal = SignalType.FLAT
                    new_size = 0.0
                    logger.debug(f"Take-profit triggered at {timestamp}")

            # Position change logic
            pnl = 0.0
            if new_signal != position:
                # Close existing position
                if position != SignalType.FLAT:
                    pnl = self._calculate_pnl(
                        position, entry_price, current_price, position_size, capital
                    )
                    capital += pnl
                    capital *= (1 - self.transaction_cost)

                # Open new position
                if new_signal != SignalType.FLAT:
                    # Apply slippage
                    entry_price = current_price * (1 + self.slippage * new_signal.value)
                    position_size = new_size
                    capital *= (1 - self.transaction_cost)
                else:
                    entry_price = 0.0
                    position_size = 0.0

                position = new_signal
            else:
                # Update PnL for existing position
                if position != SignalType.FLAT:
                    unrealized_pnl = self._calculate_pnl(
                        position, entry_price, current_price, position_size, capital
                    )
                    pnl = unrealized_pnl

            # Record trade (pred_return already obtained from get_signal)
            trades.append(TradeRecord(
                timestamp=timestamp,
                signal=position,
                price=current_price,
                position_size=position_size,
                pnl=pnl,
                cumulative_pnl=capital - self.initial_capital,
                predicted_return=pred_return,
                actual_return=actual_return
            ))

            equity_curve.append(capital)

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)

        return BacktestResult(
            trades=trades,
            equity_curve=np.array(equity_curve),
            metrics=metrics,
            config={
                'initial_capital': self.initial_capital,
                'transaction_cost': self.transaction_cost,
                'slippage': self.slippage
            }
        )

    def _calculate_pnl(
        self,
        position: SignalType,
        entry_price: float,
        exit_price: float,
        position_size: float,
        capital: float
    ) -> float:
        """Calculate PnL for a position"""
        if position == SignalType.FLAT or entry_price == 0:
            return 0.0

        price_return = (exit_price - entry_price) / entry_price
        if position == SignalType.SHORT:
            price_return = -price_return

        return price_return * position_size * capital

    def _calculate_metrics(
        self,
        trades: List[TradeRecord],
        equity_curve: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'num_trades': 0.0
            }

        # Returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[~np.isnan(returns)]

        # Total return
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital

        # Sharpe ratio (annualized, assuming hourly data)
        if len(returns) > 0 and returns.std() > 0:
            excess_returns = returns - self.risk_free_rate / (252 * 24)
            sharpe_ratio = np.sqrt(252 * 24) * excess_returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(252 * 24) * returns.mean() / downside_returns.std()
        else:
            sortino_ratio = sharpe_ratio

        # Max drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (cummax - equity_curve) / cummax
        max_drawdown = drawdown.max()

        # Win rate
        pnls = [t.pnl for t in trades if t.pnl != 0]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Number of trades (position changes)
        position_changes = sum(
            1 for i in range(1, len(trades))
            if trades[i].signal != trades[i-1].signal
        )

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': float(position_changes)
        }


def run_quick_backtest(
    model: nn.Module,
    test_data: pd.DataFrame,
    X_test: np.ndarray,
    seq_len: int = 96,
    initial_capital: float = 100000,
    long_threshold: float = 0.0005,
    short_threshold: float = -0.0005
) -> BacktestResult:
    """
    Quick backtest utility function

    Args:
        model: Trained Informer model
        test_data: DataFrame with OHLCV data
        X_test: Prepared input sequences
        seq_len: Input sequence length
        initial_capital: Starting capital
        long_threshold: Long signal threshold
        short_threshold: Short signal threshold

    Returns:
        BacktestResult
    """
    signal_generator = SignalGenerator(
        method='threshold',
        long_threshold=long_threshold,
        short_threshold=short_threshold
    )

    strategy = InformerStrategy(
        model=model,
        signal_generator=signal_generator,
        max_position=1.0,
        use_confidence_sizing=False
    )

    engine = BacktestEngine(initial_capital=initial_capital)

    return engine.run(strategy, test_data, X_test, seq_len)


if __name__ == "__main__":
    # Test with dummy model and data
    print("Testing strategy module...")

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(6, 24)

        def forward(self, x):
            batch = x.size(0)
            # Random predictions
            preds = torch.randn(batch, 24) * 0.01
            return {'predictions': preds, 'confidence': torch.rand(batch, 24)}

    model = DummyModel()

    # Test signal generator
    print("\n1. Testing SignalGenerator:")
    sg = SignalGenerator(long_threshold=0.001, short_threshold=-0.001)
    preds = torch.tensor([[0.002], [-0.003], [0.0005]])
    signals = sg.generate(preds)
    print(f"   Predictions: {preds.flatten().tolist()}")
    print(f"   Signals: {[s.name for s in signals]}")

    # Test strategy
    print("\n2. Testing InformerStrategy:")
    strategy = InformerStrategy(model, sg)
    X = torch.randn(1, 96, 6)
    signal, size, pred_return = strategy.get_signal(X, 100.0)
    print(f"   Signal: {signal.name}, Size: {size:.2f}, Pred return: {pred_return:.6f}")

    # Test backtest engine
    print("\n3. Testing BacktestEngine:")
    engine = BacktestEngine(initial_capital=10000)

    # Create dummy data
    dates = pd.date_range('2024-01-01', periods=200, freq='h')
    dummy_df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(200) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(200) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(200) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(200) * 0.5),
        'volume': np.random.randint(1000, 10000, 200)
    })
    X_test = np.random.randn(100, 96, 6).astype(np.float32)

    result = engine.run(strategy, dummy_df, X_test, seq_len=96)
    print(result.summary())

    print("\nAll tests completed!")
