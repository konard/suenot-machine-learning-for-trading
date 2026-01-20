"""
Portfolio Backtesting Module

This module provides classes for backtesting portfolio strategies
including LLM-based portfolio construction.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from portfolio backtest."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    num_trades: int
    turnover: float
    portfolio_values: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)

    def summary(self) -> str:
        return f"""
Portfolio Backtest Results
==========================
Total Return:      {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Volatility:        {self.volatility:.2%}
Sharpe Ratio:      {self.sharpe_ratio:.2f}
Sortino Ratio:     {self.sortino_ratio:.2f}
Max Drawdown:      {self.max_drawdown:.2%}
Calmar Ratio:      {self.calmar_ratio:.2f}
Win Rate:          {self.win_rate:.2%}
Turnover:          {self.turnover:.2%}
Number of Trades:  {self.num_trades}
"""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        return pd.DataFrame({
            "date": self.dates,
            "portfolio_value": self.portfolio_values,
            "return": [0] + self.returns if self.returns else [0] * len(self.dates)
        })


class PortfolioBacktester:
    """
    Backtest portfolio strategies on historical data.

    Supports various rebalancing frequencies and transaction costs.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        rebalance_frequency: str = "weekly",  # daily, weekly, monthly
        transaction_cost: float = 0.001,  # 0.1%
        slippage: float = 0.0005  # 0.05%
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            rebalance_frequency: How often to rebalance
            transaction_cost: Transaction cost as fraction
            slippage: Slippage cost as fraction
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def run(
        self,
        price_data: pd.DataFrame,
        weight_generator: Callable[[pd.DataFrame, str], Dict[str, float]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResult:
        """
        Run backtest with dynamic weight generation.

        Args:
            price_data: DataFrame with asset prices (columns = symbols)
            weight_generator: Function that generates weights given historical data and date
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResult with performance metrics
        """
        # Filter data by date range
        if start_date:
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]

        if len(price_data) < 2:
            raise ValueError("Not enough data for backtest")

        # Initialize
        capital = self.initial_capital
        portfolio_values = [capital]
        dates = [str(price_data.index[0])]
        returns_list = []
        current_weights = {}
        num_trades = 0
        total_turnover = 0.0

        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(price_data.index)

        for i in range(1, len(price_data)):
            date = price_data.index[i]
            prev_date = price_data.index[i-1]

            # Calculate daily returns
            daily_returns = (price_data.iloc[i] / price_data.iloc[i-1]) - 1

            # Check for rebalance
            if date in rebalance_dates:
                # Get historical data up to this point
                historical_data = price_data.iloc[:i]

                try:
                    new_weights = weight_generator(historical_data, str(date))

                    # Calculate turnover and costs
                    turnover = self._calculate_turnover(current_weights, new_weights)
                    total_turnover += turnover
                    cost = turnover * (self.transaction_cost + self.slippage)
                    capital *= (1 - cost)

                    # Count trades
                    for symbol in set(current_weights.keys()) | set(new_weights.keys()):
                        old_w = current_weights.get(symbol, 0)
                        new_w = new_weights.get(symbol, 0)
                        if abs(new_w - old_w) > 0.01:  # Meaningful change
                            num_trades += 1

                    current_weights = new_weights

                except Exception as e:
                    logger.warning(f"Weight generation failed for {date}: {e}")

            # Calculate portfolio return
            port_return = 0.0
            for symbol, weight in current_weights.items():
                if symbol in daily_returns.index:
                    port_return += weight * daily_returns[symbol]

            capital *= (1 + port_return)
            portfolio_values.append(capital)
            dates.append(str(date))
            returns_list.append(port_return)

        # Calculate metrics
        returns_series = pd.Series(returns_list)

        total_return = (capital / self.initial_capital) - 1
        trading_days = len(returns_series)

        # Annualize based on assumed trading days
        days_per_year = 365 if self._is_crypto(price_data) else 252
        annualized_return = (1 + total_return) ** (days_per_year / trading_days) - 1
        volatility = returns_series.std() * np.sqrt(days_per_year)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # Sortino (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        downside_std = downside_returns.std() * np.sqrt(days_per_year) if len(downside_returns) > 0 else 0.001
        sortino = annualized_return / downside_std if downside_std > 0 else 0

        # Max drawdown
        cumulative = pd.Series(portfolio_values)
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Calmar
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        win_rate = len(returns_series[returns_series > 0]) / len(returns_series) if len(returns_series) > 0 else 0

        # Average turnover
        num_rebalances = len(rebalance_dates)
        avg_turnover = total_turnover / num_rebalances if num_rebalances > 0 else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            win_rate=win_rate,
            num_trades=num_trades,
            turnover=avg_turnover,
            portfolio_values=portfolio_values,
            dates=dates,
            returns=returns_list
        )

    def run_static_weights(
        self,
        price_data: pd.DataFrame,
        weights: Dict[str, float],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResult:
        """
        Run backtest with static (fixed) weights.

        Args:
            price_data: DataFrame with asset prices
            weights: Fixed portfolio weights
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResult with performance metrics
        """
        def static_weight_generator(hist_data: pd.DataFrame, date: str) -> Dict[str, float]:
            return weights

        return self.run(price_data, static_weight_generator, start_date, end_date)

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> set:
        """Get rebalance dates based on frequency."""
        if self.rebalance_frequency == "daily":
            return set(dates)
        elif self.rebalance_frequency == "weekly":
            # Rebalance on first day of each week
            return set(dates.to_series().groupby(dates.isocalendar().week).first())
        elif self.rebalance_frequency == "monthly":
            # Rebalance on first trading day of month
            return set(dates.to_series().groupby(dates.to_period('M')).first())
        return set()

    def _calculate_turnover(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        """Calculate portfolio turnover."""
        all_symbols = set(old_weights.keys()) | set(new_weights.keys())
        turnover = sum(
            abs(new_weights.get(s, 0) - old_weights.get(s, 0))
            for s in all_symbols
        ) / 2
        return turnover

    def _is_crypto(self, price_data: pd.DataFrame) -> bool:
        """Detect if data is crypto (trades on weekends)."""
        dates = price_data.index
        if len(dates) < 7:
            return False
        # Check if there are weekend data points
        weekend_count = sum(1 for d in dates[:30] if d.dayofweek >= 5)
        return weekend_count > 0


class StrategyComparison:
    """
    Compare multiple portfolio strategies.
    """

    def __init__(self, backtester: Optional[PortfolioBacktester] = None):
        """
        Initialize comparison.

        Args:
            backtester: Backtester to use (creates default if None)
        """
        self.backtester = backtester or PortfolioBacktester()
        self.results: Dict[str, BacktestResult] = {}

    def add_strategy(
        self,
        name: str,
        price_data: pd.DataFrame,
        weight_generator: Callable,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Add a strategy to comparison.

        Args:
            name: Strategy name
            price_data: Price data
            weight_generator: Weight generation function
            start_date: Start date
            end_date: End date
        """
        result = self.backtester.run(
            price_data, weight_generator, start_date, end_date
        )
        self.results[name] = result
        logger.info(f"Added strategy: {name}")

    def add_equal_weight(
        self,
        price_data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """Add equal-weight benchmark strategy."""
        symbols = price_data.columns.tolist()
        weights = {s: 1.0/len(symbols) for s in symbols}

        def equal_weight_gen(hist_data: pd.DataFrame, date: str) -> Dict[str, float]:
            return weights

        self.add_strategy("Equal Weight", price_data, equal_weight_gen, start_date, end_date)

    def summary(self) -> pd.DataFrame:
        """
        Get summary comparison of all strategies.

        Returns:
            DataFrame with metrics for each strategy
        """
        data = []
        for name, result in self.results.items():
            data.append({
                "Strategy": name,
                "Total Return": f"{result.total_return:.2%}",
                "Ann. Return": f"{result.annualized_return:.2%}",
                "Volatility": f"{result.volatility:.2%}",
                "Sharpe": f"{result.sharpe_ratio:.2f}",
                "Sortino": f"{result.sortino_ratio:.2f}",
                "Max DD": f"{result.max_drawdown:.2%}",
                "Calmar": f"{result.calmar_ratio:.2f}",
                "Win Rate": f"{result.win_rate:.2%}",
            })
        return pd.DataFrame(data)

    def plot_equity_curves(self, figsize: tuple = (12, 6)):
        """
        Plot equity curves for all strategies.

        Args:
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=figsize)

            for name, result in self.results.items():
                ax.plot(range(len(result.portfolio_values)),
                       result.portfolio_values,
                       label=name)

            ax.set_xlabel("Days")
            ax.set_ylabel("Portfolio Value")
            ax.set_title("Strategy Comparison - Equity Curves")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None


# Example usage
if __name__ == "__main__":
    print("Portfolio Backtester Demo")
    print("=" * 50)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

    # Simulate prices with random walk
    prices = {}
    for symbol in symbols:
        base_price = {"BTCUSDT": 30000, "ETHUSDT": 2000, "SOLUSDT": 25, "BNBUSDT": 300}[symbol]
        returns = np.random.normal(0.0003, 0.03, len(dates))
        prices[symbol] = base_price * np.exp(np.cumsum(returns))

    price_df = pd.DataFrame(prices, index=dates)

    # Initialize backtester
    backtester = PortfolioBacktester(
        initial_capital=100000,
        rebalance_frequency="weekly",
        transaction_cost=0.001
    )

    # Test equal weight strategy
    print("\nRunning equal weight backtest...")
    equal_weights = {s: 0.25 for s in symbols}
    result_equal = backtester.run_static_weights(price_df, equal_weights)
    print(result_equal.summary())

    # Test momentum-based rebalancing
    print("\nRunning momentum-based backtest...")

    def momentum_weights(hist_data: pd.DataFrame, date: str) -> Dict[str, float]:
        """Simple momentum strategy: higher weight to recent winners."""
        if len(hist_data) < 20:
            return {s: 0.25 for s in hist_data.columns}

        # Calculate 20-day returns
        recent = hist_data.iloc[-20:]
        returns = (recent.iloc[-1] / recent.iloc[0]) - 1

        # Convert to weights (positive momentum = higher weight)
        scores = returns + 1  # Shift to all positive
        weights = scores / scores.sum()

        return weights.to_dict()

    result_momentum = backtester.run(price_df, momentum_weights)
    print(result_momentum.summary())

    # Compare strategies
    print("\n" + "=" * 50)
    print("Strategy Comparison")

    comparison = StrategyComparison(backtester)
    comparison.add_equal_weight(price_df)
    comparison.add_strategy("Momentum", price_df, momentum_weights)

    print(comparison.summary().to_string(index=False))
