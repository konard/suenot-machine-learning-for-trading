"""
Backtesting framework for SSM-GNN Hybrid trading strategy.

Evaluates trading signals on historical data with realistic
transaction costs, slippage, and performance metrics.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtesting results."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    n_trades: int = 0
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))


class Backtester:
    """
    Backtesting engine for multi-asset trading strategies.

    Supports:
    - Long/short/neutral signals per asset
    - Transaction costs (commission + slippage)
    - Portfolio-level performance metrics
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        """
        Args:
            initial_capital: Starting capital.
            commission: Commission rate per trade (fraction).
            slippage: Slippage per trade (fraction).
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            periods_per_year: Trading periods per year (252 for daily, 8760 for hourly).
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def run(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        confidences: np.ndarray = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            signals: (n_periods, n_assets) array of signals {-1, 0, 1}.
            prices: (n_periods, n_assets) array of prices.
            confidences: Optional (n_periods, n_assets) confidence weights.

        Returns:
            BacktestResult with all performance metrics.
        """
        n_periods, n_assets = signals.shape
        assert prices.shape == signals.shape, "signals and prices must have same shape"

        if confidences is None:
            confidences = np.ones_like(signals, dtype=float)

        logger.info("Running backtest: %d periods, %d assets", n_periods, n_assets)

        # Compute portfolio weights
        weights = signals.astype(float) * confidences
        weight_sum = np.sum(np.abs(weights), axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
        weights = weights / weight_sum

        # Compute returns
        price_returns = np.zeros_like(prices)
        price_returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-10)

        # Portfolio returns (shifted signals: signal at t trades at t+1)
        portfolio_returns = np.zeros(n_periods)
        n_trades = 0

        for t in range(1, n_periods):
            # Use previous period's weights
            port_ret = np.sum(weights[t-1] * price_returns[t])

            # Transaction costs on weight changes
            if t >= 2:
                turnover = np.sum(np.abs(weights[t-1] - weights[t-2]))
                cost = turnover * (self.commission + self.slippage)
                port_ret -= cost
                if turnover > 0.01:
                    n_trades += 1

            portfolio_returns[t] = port_ret

        # Equity curve
        equity_curve = self.initial_capital * np.cumprod(1 + portfolio_returns)

        # Performance metrics
        result = BacktestResult()
        result.equity_curve = equity_curve
        result.daily_returns = portfolio_returns
        result.n_trades = n_trades

        # Total return
        result.total_return = equity_curve[-1] / self.initial_capital - 1

        # Annualized return
        n_years = n_periods / self.periods_per_year
        if n_years > 0:
            result.annualized_return = (1 + result.total_return) ** (1 / n_years) - 1

        # Sharpe ratio
        excess_returns = portfolio_returns - self.risk_free_rate / self.periods_per_year
        std_returns = np.std(portfolio_returns)
        if std_returns > 0:
            result.sharpe_ratio = (np.mean(excess_returns) / std_returns) * np.sqrt(self.periods_per_year)

        # Sortino ratio
        downside = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-10
        result.sortino_ratio = (np.mean(excess_returns) / downside_std) * np.sqrt(self.periods_per_year)

        # Maximum drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / (running_max + 1e-10)
        result.max_drawdown = np.max(drawdowns)

        # Win rate
        winning_periods = np.sum(portfolio_returns > 0)
        active_periods = np.sum(portfolio_returns != 0)
        result.win_rate = winning_periods / max(active_periods, 1)

        # Profit factor
        gross_profit = np.sum(portfolio_returns[portfolio_returns > 0])
        gross_loss = abs(np.sum(portfolio_returns[portfolio_returns < 0]))
        result.profit_factor = gross_profit / max(gross_loss, 1e-10)

        # Calmar ratio
        result.calmar_ratio = result.annualized_return / max(result.max_drawdown, 1e-10)

        logger.info("Backtest complete: return=%.4f, sharpe=%.3f, max_dd=%.4f",
                     result.total_return, result.sharpe_ratio, result.max_drawdown)

        return result


def run_ssm_gnn_backtest(
    symbols: list = None,
    interval: str = "60",
    limit: int = 200,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """
    End-to-end backtest of the SSM-GNN hybrid strategy.

    Args:
        symbols: List of trading pair symbols.
        interval: Kline interval.
        limit: Number of data points.
        initial_capital: Starting capital.

    Returns:
        BacktestResult with full performance metrics.
    """
    from python.data_loader import load_multi_asset_data, build_correlation_graph, prepare_features
    from python.ssm_gnn_model import SSMGNNHybrid

    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT"]

    # Load data
    prices_dict, all_close = load_multi_asset_data(symbols, interval, limit)
    features = prepare_features(prices_dict, symbols)

    # Build graph
    edge_index, edge_weight = build_correlation_graph(all_close)

    # Create model
    n_features = features.shape[2]
    model = SSMGNNHybrid(
        n_features=n_features,
        d_model=64,
        d_state=16,
        n_gnn_layers=2,
        n_heads=4,
        n_classes=3,
    )

    # Generate rolling signals
    n_assets, seq_len, _ = features.shape
    window_size = 50
    all_signals = np.zeros((seq_len, n_assets), dtype=int)
    all_confidences = np.zeros((seq_len, n_assets))

    for t in range(window_size, seq_len):
        window_features = features[:, t-window_size:t, :]
        signals, confidences = model.predict_signals(window_features, edge_index, edge_weight)
        all_signals[t] = signals
        all_confidences[t] = confidences

    # Run backtest
    bt = Backtester(
        initial_capital=initial_capital,
        commission=0.001,
        slippage=0.0005,
    )

    result = bt.run(all_signals, all_close.T, all_confidences)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_ssm_gnn_backtest()
    print(f"Total Return: {result.total_return:.4f}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"Sortino Ratio: {result.sortino_ratio:.3f}")
    print(f"Max Drawdown: {result.max_drawdown:.4f}")
    print(f"Win Rate: {result.win_rate:.3f}")
    print(f"Profit Factor: {result.profit_factor:.3f}")
    print(f"Number of Trades: {result.n_trades}")
