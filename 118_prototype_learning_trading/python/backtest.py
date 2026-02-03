"""
Backtesting framework for prototype learning trading strategy.

Implements signal generation and performance evaluation.
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .model import ProtoPNet, MarketState


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    pnl: float
    return_pct: float
    predicted_state: int
    confidence: float


class PrototypeBacktester:
    """
    Backtester for prototype learning trading strategy.

    Generates signals based on prototype similarities and
    evaluates strategy performance.
    """

    def __init__(
        self,
        model: ProtoPNet,
        device: torch.device = None,
        confidence_threshold: float = 0.6,
        position_sizing: str = 'fixed',  # 'fixed', 'confidence', 'kelly'
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        """
        Initialize backtester.

        Args:
            model: Trained ProtoPNet model
            device: Device for inference
            confidence_threshold: Minimum confidence to take a trade
            position_sizing: Position sizing method
            initial_capital: Starting capital
            commission_pct: Commission percentage per trade
            slippage_pct: Slippage percentage per trade
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.confidence_threshold = confidence_threshold
        self.position_sizing = position_sizing
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def generate_signals(
        self,
        X: np.ndarray,
        timestamps: pd.DatetimeIndex,
        prices: np.ndarray,
    ) -> pd.DataFrame:
        """
        Generate trading signals from model predictions.

        Args:
            X: Feature sequences of shape (n_samples, n_features, lookback)
            timestamps: Timestamps for each sample
            prices: Close prices for each sample

        Returns:
            DataFrame with signals and metadata
        """
        # Get predictions
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, similarities, _ = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = probs.max(dim=1)

        predictions = predictions.cpu().numpy()
        confidences = confidences.cpu().numpy()
        similarities = similarities.cpu().numpy()

        # Create signals DataFrame
        signals = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'predicted_state': predictions,
            'confidence': confidences,
        })

        # Add state names
        signals['state_name'] = signals['predicted_state'].apply(
            lambda x: MarketState(x).name
        )

        # Generate position signals
        signals['position'] = 0

        # Long signals: Bullish or Breakout Up with high confidence
        long_mask = (
            (signals['predicted_state'].isin([MarketState.BULLISH, MarketState.BREAKOUT_UP])) &
            (signals['confidence'] >= self.confidence_threshold)
        )
        signals.loc[long_mask, 'position'] = 1

        # Short signals: Bearish or Breakout Down with high confidence
        short_mask = (
            (signals['predicted_state'].isin([MarketState.BEARISH, MarketState.BREAKOUT_DOWN])) &
            (signals['confidence'] >= self.confidence_threshold)
        )
        signals.loc[short_mask, 'position'] = -1

        # Add similarity scores for each prototype
        for i in range(similarities.shape[1]):
            signals[f'similarity_{i}'] = similarities[:, i]

        return signals

    def run_backtest(
        self,
        signals: pd.DataFrame,
        holding_period: int = 5,
    ) -> Dict:
        """
        Run backtest on generated signals.

        Args:
            signals: DataFrame with signals from generate_signals()
            holding_period: Number of periods to hold position

        Returns:
            Dictionary with backtest results
        """
        trades = []
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        current_position = 0
        position_entry_idx = None

        for i in range(len(signals)):
            row = signals.iloc[i]

            # Check if we need to close position
            if current_position != 0:
                # Close after holding period
                if i - position_entry_idx >= holding_period:
                    entry_row = signals.iloc[position_entry_idx]
                    exit_price = row['price'] * (1 - self.slippage_pct * np.sign(current_position))

                    # Calculate P&L
                    if current_position > 0:  # Long
                        return_pct = (exit_price / entry_row['price']) - 1
                    else:  # Short
                        return_pct = 1 - (exit_price / entry_row['price'])

                    return_pct -= self.commission_pct * 2  # Entry and exit

                    position_value = current_capital * abs(current_position)
                    pnl = position_value * return_pct

                    trades.append(Trade(
                        entry_time=entry_row['timestamp'],
                        exit_time=row['timestamp'],
                        entry_price=entry_row['price'],
                        exit_price=exit_price,
                        direction=1 if current_position > 0 else -1,
                        size=abs(current_position),
                        pnl=pnl,
                        return_pct=return_pct,
                        predicted_state=int(entry_row['predicted_state']),
                        confidence=entry_row['confidence'],
                    ))

                    current_capital += pnl
                    current_position = 0
                    position_entry_idx = None

            # Open new position if no current position
            if current_position == 0 and row['position'] != 0:
                current_position = row['position'] * self._calculate_position_size(
                    row['confidence']
                )
                position_entry_idx = i

            equity_curve.append(current_capital)

        # Calculate metrics
        results = self._calculate_metrics(trades, equity_curve, signals)
        results['trades'] = trades
        results['equity_curve'] = equity_curve

        return results

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on sizing method."""
        if self.position_sizing == 'fixed':
            return 1.0
        elif self.position_sizing == 'confidence':
            return confidence
        elif self.position_sizing == 'kelly':
            # Simplified Kelly - would need historical win rate
            return min(confidence * 2, 1.0)
        else:
            return 1.0

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        signals: pd.DataFrame,
    ) -> Dict:
        """Calculate performance metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
            }

        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Basic metrics
        total_return = (equity[-1] / equity[0]) - 1
        total_trades = len(trades)

        # Returns analysis
        if len(returns) > 1:
            sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
            sortino_returns = returns[returns < 0]
            sortino_ratio = np.sqrt(252) * returns.mean() / (sortino_returns.std() + 1e-8) if len(sortino_returns) > 0 else 0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()

        # Trade analysis
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = np.mean([t.return_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.return_pct for t in losing_trades]) if losing_trades else 0

        # Calmar ratio
        annual_return = total_return  # Simplified - would need time normalization
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        # Analysis by market state
        state_performance = {}
        for state in MarketState:
            state_trades = [t for t in trades if t.predicted_state == state.value]
            if state_trades:
                state_performance[state.name] = {
                    'count': len(state_trades),
                    'win_rate': len([t for t in state_trades if t.pnl > 0]) / len(state_trades),
                    'avg_return': np.mean([t.return_pct for t in state_trades]),
                    'total_pnl': sum(t.pnl for t in state_trades),
                }

        return {
            'total_trades': total_trades,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_capital': equity[-1],
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'state_performance': state_performance,
        }

    def walk_forward_backtest(
        self,
        df: pd.DataFrame,
        feature_engineering,
        feature_columns: List[str],
        train_window: int = 252,
        test_window: int = 63,
        retrain_interval: int = 21,
    ) -> Dict:
        """
        Walk-forward backtesting with periodic retraining.

        Args:
            df: Full DataFrame with OHLCV data
            feature_engineering: FeatureEngineering instance
            feature_columns: List of feature columns
            train_window: Training window size in periods
            test_window: Test window size
            retrain_interval: Periods between retraining

        Returns:
            Dictionary with walk-forward results
        """
        from .train import ProtoPNetTrainer

        all_signals = []
        all_periods = []

        start_idx = train_window + feature_engineering.lookback
        current_idx = start_idx

        while current_idx + test_window <= len(df):
            print(f"Walk-forward period: {current_idx} to {current_idx + test_window}")

            # Define train and test periods
            train_start = current_idx - train_window - feature_engineering.lookback
            train_end = current_idx
            test_end = min(current_idx + test_window, len(df))

            # Prepare training data
            train_df = df.iloc[train_start:train_end].copy()
            train_df = feature_engineering.compute_features(train_df)
            train_df = feature_engineering.label_market_states(train_df)

            X_train, y_train = feature_engineering.create_sequences(
                train_df, feature_columns
            )
            X_train = feature_engineering.normalize_features(X_train, fit=True)

            # Retrain model
            trainer = ProtoPNetTrainer(
                ProtoPNet(
                    input_channels=len(feature_columns),
                    sequence_length=feature_engineering.lookback,
                ),
                device=self.device,
            )
            trainer.train(
                X_train, y_train,
                X_train[-50:], y_train[-50:],  # Use end of train as validation
                epochs=30,
                batch_size=32,
                early_stopping_patience=10,
            )

            # Prepare test data
            test_df = df.iloc[train_end - feature_engineering.lookback:test_end].copy()
            test_df = feature_engineering.compute_features(test_df)

            X_test, _ = feature_engineering.create_sequences(
                test_df, feature_columns, label_column='returns_1'  # Dummy label
            )
            X_test = feature_engineering.normalize_features(X_test, fit=False)

            # Generate signals
            test_timestamps = test_df['timestamp'].iloc[feature_engineering.lookback:].values
            test_prices = test_df['close'].iloc[feature_engineering.lookback:].values

            self.model = trainer.model
            signals = self.generate_signals(X_test, test_timestamps, test_prices)

            all_signals.append(signals)
            all_periods.append({
                'start': current_idx,
                'end': test_end,
            })

            current_idx += retrain_interval

        # Combine all signals
        combined_signals = pd.concat(all_signals, ignore_index=True)

        # Run backtest on combined signals
        results = self.run_backtest(combined_signals)
        results['periods'] = all_periods

        return results


def print_backtest_report(results: Dict):
    """Print formatted backtest report."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {results['total_return_pct']:.2f}%")
    print(f"  Final Capital: ${results['final_capital']:,.2f}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {results['sortino_ratio']:.3f}")
    print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"  Calmar Ratio: {results['calmar_ratio']:.3f}")

    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Winning Trades: {results['winning_trades']}")
    print(f"  Losing Trades: {results['losing_trades']}")
    print(f"  Win Rate: {results['win_rate_pct']:.2f}%")
    print(f"  Profit Factor: {results['profit_factor']:.3f}")
    print(f"  Avg Win: {results['avg_win']*100:.2f}%")
    print(f"  Avg Loss: {results['avg_loss']*100:.2f}%")

    if results.get('state_performance'):
        print(f"\nPerformance by Market State:")
        for state, metrics in results['state_performance'].items():
            print(f"  {state}:")
            print(f"    Trades: {metrics['count']}")
            print(f"    Win Rate: {metrics['win_rate']*100:.2f}%")
            print(f"    Avg Return: {metrics['avg_return']*100:.2f}%")
            print(f"    Total P&L: ${metrics['total_pnl']:,.2f}")

    print("=" * 60)
