"""
Trading Strategy and Backtesting for Reformer

Provides:
- ReformerStrategy: Trading strategy based on model predictions
- backtest_strategy: Run backtest on historical data
- Performance metrics calculation
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    max_position_size: float = 0.2   # 20% of capital per position
    rebalance_frequency: int = 24    # Hours between rebalancing
    risk_free_rate: float = 0.02     # Annual risk-free rate
    leverage: float = 1.0            # Maximum leverage
    stop_loss: Optional[float] = None  # Stop loss percentage
    take_profit: Optional[float] = None  # Take profit percentage


class ReformerStrategy:
    """
    Trading strategy based on Reformer model predictions.

    Strategy logic:
    1. Use long historical context (Reformer's strength)
    2. Predict returns for each asset
    3. Convert predictions to position signals
    4. Size positions by confidence
    5. Apply risk management rules
    """

    def __init__(self, model, config: BacktestConfig):
        """
        Initialize strategy.

        Args:
            model: Trained ReformerModel
            config: Backtesting configuration
        """
        self.model = model
        self.config = config
        self.model.eval()

    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Generate predictions from model.

        Args:
            x: Input tensor [1, num_tickers, seq_len, features]

        Returns:
            Dictionary with predictions, directions, and confidence
        """
        with torch.no_grad():
            output = self.model(x, return_attention=False)
            predictions = output['predictions'].cpu().numpy().flatten()

        # Convert to directions
        directions = np.sign(predictions)

        # Confidence based on prediction magnitude
        confidence = np.abs(predictions)
        confidence = confidence / (confidence.sum() + 1e-8)  # Normalize

        return {
            'predictions': predictions,
            'directions': directions,
            'confidence': confidence
        }

    def get_position_sizes(
        self,
        predictions: np.ndarray,
        current_capital: float
    ) -> np.ndarray:
        """
        Calculate position sizes based on predictions.

        Uses a modified Kelly criterion approach:
        - Position size proportional to prediction magnitude
        - Capped by max_position_size
        - Leverage applied if configured

        Args:
            predictions: Model predictions [num_tickers]
            current_capital: Current portfolio value

        Returns:
            Position sizes in dollars [num_tickers]
        """
        # Scale predictions to position weights
        weights = np.tanh(predictions * 2)  # Bound to [-1, 1]

        # Normalize
        total_weight = np.abs(weights).sum()
        if total_weight > 1:
            weights = weights / total_weight

        # Apply position size limits
        weights = np.clip(
            weights,
            -self.config.max_position_size,
            self.config.max_position_size
        )

        # Apply leverage
        weights = weights * self.config.leverage

        # Convert to dollar positions
        positions = weights * current_capital

        return positions

    def apply_risk_management(
        self,
        positions: np.ndarray,
        current_prices: np.ndarray,
        entry_prices: np.ndarray
    ) -> np.ndarray:
        """
        Apply stop-loss and take-profit rules.

        Args:
            positions: Current positions
            current_prices: Current asset prices
            entry_prices: Entry prices for positions

        Returns:
            Adjusted positions
        """
        if self.config.stop_loss is None and self.config.take_profit is None:
            return positions

        adjusted_positions = positions.copy()

        for i in range(len(positions)):
            if positions[i] == 0 or entry_prices[i] == 0:
                continue

            pnl_pct = (current_prices[i] - entry_prices[i]) / entry_prices[i]

            # Adjust for short positions
            if positions[i] < 0:
                pnl_pct = -pnl_pct

            # Check stop-loss
            if self.config.stop_loss and pnl_pct < -self.config.stop_loss:
                adjusted_positions[i] = 0
                logger.debug(f"Stop-loss triggered for asset {i}")

            # Check take-profit
            if self.config.take_profit and pnl_pct > self.config.take_profit:
                adjusted_positions[i] = 0
                logger.debug(f"Take-profit triggered for asset {i}")

        return adjusted_positions


def backtest_strategy(
    model,
    test_data: Dict,
    config: BacktestConfig
) -> pd.DataFrame:
    """
    Backtest Reformer-based trading strategy.

    Args:
        model: Trained ReformerModel
        test_data: Test data dictionary with X, y
        config: Backtesting configuration

    Returns:
        DataFrame with backtest results
    """
    strategy = ReformerStrategy(model, config)

    X = test_data['X']
    y_true = test_data['y']
    symbols = test_data['symbols']
    n_assets = len(symbols)

    # Initialize tracking
    capital = config.initial_capital
    positions = np.zeros(n_assets)
    entry_prices = np.zeros(n_assets)

    # Results tracking
    portfolio_values = [capital]
    position_history = []
    trade_history = []

    # Iterate through test data
    n_steps = len(X)
    rebalance_steps = range(0, n_steps, config.rebalance_frequency)

    logger.info(f"Backtesting {n_steps} steps with {len(list(rebalance_steps))} rebalances")

    for step_idx, i in enumerate(rebalance_steps):
        # Get model predictions
        x = torch.FloatTensor(X[i:i+1])
        pred_output = strategy.predict(x)
        predictions = pred_output['predictions']

        # Calculate target positions
        target_positions = strategy.get_position_sizes(predictions, capital)

        # Calculate transaction costs for position changes
        position_changes = np.abs(target_positions - positions)
        turnover = position_changes.sum()
        costs = turnover * config.transaction_cost

        # Record trades
        for asset_idx in range(n_assets):
            if position_changes[asset_idx] > 0:
                trade_history.append({
                    'step': i,
                    'symbol': symbols[asset_idx],
                    'old_position': positions[asset_idx],
                    'new_position': target_positions[asset_idx],
                    'prediction': predictions[asset_idx]
                })

        # Update positions
        positions = target_positions.copy()
        entry_prices = np.where(
            position_changes > 0,
            np.ones(n_assets),  # Placeholder price
            entry_prices
        )

        # Calculate returns for this period
        end_idx = min(i + config.rebalance_frequency, len(y_true))
        period_returns = y_true[i:end_idx]

        if len(period_returns) > 0:
            # Position weights
            weights = positions / (capital + 1e-8)

            # Compound returns for the period
            for ret in period_returns:
                portfolio_return = np.sum(weights * ret)
                capital = capital * (1 + portfolio_return)

            # Subtract transaction costs
            capital -= costs

        portfolio_values.append(capital)

        position_history.append({
            'step': i,
            'capital': capital,
            'positions': positions.copy(),
            'predictions': predictions.copy(),
            'costs': costs
        })

    # Create results DataFrame
    results = pd.DataFrame({
        'step': list(range(len(portfolio_values))),
        'portfolio_value': portfolio_values
    })

    # Calculate returns series
    returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])

    # Calculate performance metrics
    metrics = calculate_metrics(returns, config.risk_free_rate)

    # Store metrics and history as attributes
    results.attrs['metrics'] = metrics
    results.attrs['position_history'] = position_history
    results.attrs['trade_history'] = trade_history
    results.attrs['symbols'] = symbols

    return results


def calculate_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0,
            'volatility': 0.0
        }

    # Basic metrics
    total_return = np.prod(1 + returns) - 1

    # Annualized volatility (assuming hourly returns)
    volatility = returns.std() * np.sqrt(365 * 24)

    # Sharpe ratio
    excess_returns = returns - risk_free_rate / (365 * 24)
    if volatility > 0:
        sharpe_ratio = np.sqrt(365 * 24) * excess_returns.mean() / returns.std()
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (downside deviation only)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = np.sqrt(365 * 24) * excess_returns.mean() / downside_returns.std()
    else:
        sortino_ratio = float('inf') if excess_returns.mean() > 0 else 0.0

    # Maximum drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0

    # Win rate
    wins = (returns > 0).sum()
    win_rate = wins / len(returns) if len(returns) > 0 else 0.0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Calmar ratio (return / max drawdown)
    annualized_return = (1 + total_return) ** (365 * 24 / len(returns)) - 1 if len(returns) > 0 else 0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio if sortino_ratio != float('inf') else 999.0,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor if profit_factor != float('inf') else 999.0,
        'calmar_ratio': calmar_ratio if calmar_ratio != float('inf') else 999.0,
        'volatility': volatility,
        'num_periods': len(returns)
    }


def print_backtest_report(results: pd.DataFrame):
    """
    Print a formatted backtest report.

    Args:
        results: DataFrame from backtest_strategy
    """
    metrics = results.attrs['metrics']
    symbols = results.attrs['symbols']
    trades = results.attrs['trade_history']

    print("\n" + "=" * 60)
    print("REFORMER BACKTEST REPORT")
    print("=" * 60)

    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Total periods: {metrics['num_periods']}")

    print("\n--- Performance Metrics ---")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Volatility: {metrics['volatility']:.2%}")

    print(f"\n--- Trading Activity ---")
    print(f"Total Trades: {len(trades)}")

    print("\n" + "=" * 60)


def compare_strategies(
    model,
    test_data: Dict,
    configs: List[BacktestConfig],
    config_names: List[str]
) -> pd.DataFrame:
    """
    Compare multiple strategy configurations.

    Args:
        model: Trained model
        test_data: Test data
        configs: List of backtest configurations
        config_names: Names for each configuration

    Returns:
        DataFrame comparing strategy performance
    """
    results = []

    for config, name in zip(configs, config_names):
        logger.info(f"Running backtest for: {name}")
        backtest_results = backtest_strategy(model, test_data, config)
        metrics = backtest_results.attrs['metrics']
        metrics['strategy'] = name
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('strategy')

    return comparison_df


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing strategy module with synthetic data...")

    # Create mock data
    np.random.seed(42)
    n_samples = 500
    n_tickers = 3
    seq_len = 168
    n_features = 6

    test_data = {
        'X': np.random.randn(n_samples, n_tickers, seq_len, n_features).astype(np.float32),
        'y': np.random.randn(n_samples, n_tickers).astype(np.float32) * 0.02,
        'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    }

    # Create mock model
    from model import ReformerModel, ReformerConfig

    config = ReformerConfig(
        num_tickers=n_tickers,
        seq_len=seq_len,
        input_features=n_features,
        d_model=32,
        n_heads=2,
        n_layers=1,
        n_buckets=16,
        n_rounds=2,
        chunk_size=16
    )

    model = ReformerModel(config)

    # Run backtest
    backtest_config = BacktestConfig(
        initial_capital=100000,
        transaction_cost=0.001,
        max_position_size=0.2,
        rebalance_frequency=24
    )

    results = backtest_strategy(model, test_data, backtest_config)

    # Print report
    print_backtest_report(results)

    print("\nAll tests passed!")
