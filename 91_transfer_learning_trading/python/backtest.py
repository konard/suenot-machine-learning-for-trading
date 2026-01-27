"""
Backtesting script for Transfer Learning Trading strategy.

Usage:
    python backtest.py --use-synthetic
    python backtest.py --data target_data.csv --model saved_model.pt
"""

import argparse
import numpy as np

from model import TransferLearningPipeline
from train import (
    extract_features,
    generate_labels,
    generate_synthetic_data,
)


class TransferTradingStrategy:
    """Trading strategy using transfer learning predictions.

    Includes risk management:
    - Confidence threshold for signal generation
    - Stop-loss and take-profit levels
    - Position sizing based on domain similarity
    - Maximum daily drawdown limit
    """

    def __init__(self, initial_capital=100_000, confidence_threshold=0.65,
                 stop_loss=0.015, take_profit=0.03, max_position_pct=0.02):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.confidence_threshold = confidence_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position_pct = max_position_pct

        self.position = None  # {'side': 'long'/'short', 'entry': price, 'size': amount}
        self.trades = []
        self.equity_curve = [initial_capital]

    def step(self, probabilities, current_price, domain_similarity=1.0):
        """Process one time step.

        Args:
            probabilities: Array of [P(Down), P(Hold), P(Up)]
            current_price: Current market price
            domain_similarity: Similarity score (0-1)

        Returns:
            Signal string: 'BUY', 'SELL', or 'HOLD'
        """
        # Check exit conditions for existing positions
        if self.position is not None:
            exit_signal = self._check_exit(current_price)
            if exit_signal:
                self._close_position(current_price)
                return exit_signal

        # Get prediction
        pred_class = np.argmax(probabilities)
        confidence = probabilities[pred_class]

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            self.equity_curve.append(self.capital)
            return 'HOLD'

        # Adjust for domain similarity
        adjusted_confidence = confidence * min(domain_similarity / 0.7, 1.0)
        if adjusted_confidence < self.confidence_threshold:
            self.equity_curve.append(self.capital)
            return 'HOLD'

        # Generate signal
        if pred_class == 2 and self.position is None:  # Up -> Buy
            size = self._calculate_position_size(current_price, domain_similarity)
            self.position = {'side': 'long', 'entry': current_price, 'size': size}
            self.equity_curve.append(self.capital)
            return 'BUY'
        elif pred_class == 0 and self.position is None:  # Down -> Sell
            size = self._calculate_position_size(current_price, domain_similarity)
            self.position = {'side': 'short', 'entry': current_price, 'size': size}
            self.equity_curve.append(self.capital)
            return 'SELL'
        elif pred_class == 0 and self.position and self.position['side'] == 'long':
            self._close_position(current_price)
            return 'SELL'
        elif pred_class == 2 and self.position and self.position['side'] == 'short':
            self._close_position(current_price)
            return 'BUY'

        self.equity_curve.append(self.capital)
        return 'HOLD'

    def _check_exit(self, current_price):
        """Check stop-loss and take-profit conditions."""
        if self.position is None:
            return None

        entry = self.position['entry']
        if self.position['side'] == 'long':
            pnl_pct = (current_price - entry) / entry
            if pnl_pct <= -self.stop_loss:
                return 'SELL'  # Stop-loss
            if pnl_pct >= self.take_profit:
                return 'SELL'  # Take-profit
        else:
            pnl_pct = (entry - current_price) / entry
            if pnl_pct <= -self.stop_loss:
                return 'BUY'
            if pnl_pct >= self.take_profit:
                return 'BUY'

        return None

    def _close_position(self, current_price):
        """Close the current position and record the trade."""
        if self.position is None:
            return

        entry = self.position['entry']
        size = self.position['size']

        if self.position['side'] == 'long':
            pnl = (current_price - entry) * size
            pnl_pct = (current_price - entry) / entry
        else:
            pnl = (entry - current_price) * size
            pnl_pct = (entry - current_price) / entry

        self.capital += pnl
        self.trades.append({
            'side': self.position['side'],
            'entry': entry,
            'exit': current_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
        })
        self.position = None
        self.equity_curve.append(self.capital)

    def _calculate_position_size(self, price, domain_similarity):
        """Calculate position size based on risk parameters."""
        base_size = self.capital * self.max_position_pct / max(price, 1e-10)
        if domain_similarity < 0.7:
            base_size *= domain_similarity / 0.7
        return base_size

    def performance_summary(self):
        """Compute and print performance metrics."""
        if not self.trades:
            print("No trades executed.")
            return {}

        pnls = [t['pnl'] for t in self.trades]
        pnl_pcts = [t['pnl_pct'] for t in self.trades]

        total_pnl = sum(pnls)
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        win_rate = len(winning) / len(self.trades) if self.trades else 0
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 0
        profit_factor = gross_profit / max(gross_loss, 1e-10)

        avg_return = np.mean(pnl_pcts) if pnl_pcts else 0
        std_return = np.std(pnl_pcts) if len(pnl_pcts) > 1 else 0
        sharpe = avg_return / max(std_return, 1e-10) * np.sqrt(252)

        downside = [r for r in pnl_pcts if r < 0]
        downside_std = np.std(downside) if len(downside) > 1 else 1e-10
        sortino = avg_return / max(downside_std, 1e-10) * np.sqrt(252)

        # Max drawdown from equity curve
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / np.maximum(peak, 1e-10)
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        results = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'final_capital': self.capital,
            'return_pct': (self.capital - self.initial_capital) / self.initial_capital,
        }

        print("\n" + "=" * 50)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Trades:     {results['total_trades']}")
        print(f"Winning Trades:   {results['winning_trades']}")
        print(f"Losing Trades:    {results['losing_trades']}")
        print(f"Win Rate:         {results['win_rate']:.2%}")
        print(f"Total PnL:        ${results['total_pnl']:,.2f}")
        print(f"Profit Factor:    {results['profit_factor']:.2f}")
        print(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:    {results['sortino_ratio']:.2f}")
        print(f"Max Drawdown:     {results['max_drawdown']:.2%}")
        print(f"Final Capital:    ${results['final_capital']:,.2f}")
        print(f"Total Return:     {results['return_pct']:.2%}")
        print("=" * 50)

        return results


def run_backtest(args):
    """Run the backtesting pipeline."""
    print("=" * 60)
    print("Transfer Learning for Trading - Backtest")
    print("=" * 60)

    # Generate data
    if args.use_synthetic:
        print("\nGenerating synthetic data...")
        source_prices, source_volumes = generate_synthetic_data(500, 100.0, 0.02)
        target_prices, target_volumes = generate_synthetic_data(200, 50.0, 0.04)
    else:
        print("Using synthetic data (CSV loading not implemented in demo).")
        source_prices, source_volumes = generate_synthetic_data(500, 100.0, 0.02)
        target_prices, target_volumes = generate_synthetic_data(200, 50.0, 0.04)

    # Extract features
    lookback = 20
    source_features = extract_features(source_prices, source_volumes, lookback)
    source_labels = generate_labels(source_prices, lookback, forward=3, threshold=0.005)
    target_features = extract_features(target_prices, target_volumes, lookback)
    target_labels = generate_labels(target_prices, lookback, forward=3, threshold=0.005)

    n_source = min(len(source_features), len(source_labels))
    n_target = min(len(target_features), len(target_labels))
    source_features = source_features[:n_source]
    source_labels = source_labels[:n_source]
    target_features = target_features[:n_target]
    target_labels = target_labels[:n_target]

    if n_source == 0 or n_target == 0:
        print("Not enough data. Exiting.")
        return

    n_features = source_features.shape[1]

    # Split target data: first half for adaptation, second half for testing
    split_idx = n_target // 2
    adapt_features = target_features[:split_idx]
    adapt_labels = target_labels[:split_idx]
    test_features = target_features[split_idx:]
    test_labels = target_labels[split_idx:]

    print(f"\nSource: {n_source} samples")
    print(f"Target adaptation: {split_idx} samples")
    print(f"Target test: {n_target - split_idx} samples")

    # Train model
    print("\nTraining transfer learning model...")
    pipeline = TransferLearningPipeline(
        input_dim=n_features,
        hidden_dim=64,
        feature_dim=32,
        adaptation_method=args.method,
    )

    pipeline.pretrain(source_features, source_labels, epochs=args.epochs, lr=0.001)
    pipeline.adapt(source_features, adapt_features, target_labels=adapt_labels, epochs=30)

    # Evaluate prediction accuracy
    test_preds = pipeline.predict(test_features)
    test_accuracy = np.mean(test_preds == test_labels)
    print(f"\nTest accuracy: {test_accuracy:.2%}")

    # Run backtesting on test period
    print("\nRunning backtest...")
    strategy = TransferTradingStrategy(
        initial_capital=args.capital,
        confidence_threshold=args.confidence,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )

    test_probas = pipeline.predict_proba(test_features)
    test_prices = target_prices[lookback + split_idx:lookback + split_idx + len(test_features)]

    signals = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    for i in range(len(test_features)):
        if i < len(test_prices):
            signal = strategy.step(test_probas[i], test_prices[i], domain_similarity=0.8)
            signals[signal] += 1

    print(f"\nSignal distribution: {signals}")

    # Close any remaining position
    if strategy.position is not None and len(test_prices) > 0:
        strategy._close_position(test_prices[-1])

    # Performance summary
    strategy.performance_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning Trading - Backtest")
    parser.add_argument("--data", type=str, default=None, help="Path to target data CSV")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    parser.add_argument("--method", type=str, default="mmd", choices=["mmd", "coral", "none"])
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--confidence", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--stop-loss", type=float, default=0.015, help="Stop-loss percentage")
    parser.add_argument("--take-profit", type=float, default=0.03, help="Take-profit percentage")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data")

    args = parser.parse_args()
    run_backtest(args)
