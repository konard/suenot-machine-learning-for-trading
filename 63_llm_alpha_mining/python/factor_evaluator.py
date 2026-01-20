"""
Factor Evaluator Module for LLM Alpha Mining

This module provides tools for evaluating alpha factor performance using
standard quantitative finance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class FactorMetrics:
    """
    Container for alpha factor evaluation metrics.

    Attributes:
        ic: Information Coefficient (Pearson correlation with future returns)
        ic_ir: IC Information Ratio (IC mean / IC std)
        rank_ic: Rank IC (Spearman correlation)
        ic_positive_pct: Percentage of positive IC values
        turnover: Average daily turnover
        sharpe_ratio: Annualized Sharpe ratio
        max_drawdown: Maximum drawdown
        mean_return: Mean annualized return
        volatility: Annualized volatility
        win_rate: Percentage of profitable periods
        t_stat: T-statistic for IC
        p_value: P-value for IC significance
    """
    ic: float
    ic_ir: float
    rank_ic: float
    ic_positive_pct: float
    turnover: float
    sharpe_ratio: float
    max_drawdown: float
    mean_return: float
    volatility: float
    win_rate: float
    t_stat: float
    p_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "ic": self.ic,
            "ic_ir": self.ic_ir,
            "rank_ic": self.rank_ic,
            "ic_positive_pct": self.ic_positive_pct,
            "turnover": self.turnover,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "win_rate": self.win_rate,
            "t_stat": self.t_stat,
            "p_value": self.p_value
        }

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if IC is statistically significant."""
        return self.p_value < alpha

    def quality_score(self) -> float:
        """
        Calculate overall factor quality score (0-100).

        Considers IC, Sharpe, drawdown, and significance.
        """
        scores = []

        # IC score (0-30 points)
        ic_score = min(30, abs(self.ic) * 150)
        scores.append(ic_score)

        # IC IR score (0-20 points)
        ir_score = min(20, abs(self.ic_ir) * 10)
        scores.append(ir_score)

        # Sharpe score (0-25 points)
        sharpe_score = min(25, max(0, self.sharpe_ratio) * 12.5)
        scores.append(sharpe_score)

        # Drawdown score (0-15 points)
        dd_score = max(0, 15 - abs(self.max_drawdown) * 30)
        scores.append(dd_score)

        # Significance bonus (0-10 points)
        sig_score = 10 if self.is_significant() else 5 if self.p_value < 0.1 else 0
        scores.append(sig_score)

        return sum(scores)


class FactorEvaluator:
    """
    Evaluate alpha factors using standard quantitative metrics.

    The evaluator calculates various performance metrics including:
    - Information Coefficient (IC)
    - Sharpe Ratio
    - Maximum Drawdown
    - Factor Turnover

    Examples:
        >>> evaluator = FactorEvaluator()
        >>> metrics = evaluator.evaluate(factor_values, returns)
        >>> print(f"IC: {metrics.ic:.4f}")
        >>> print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
    """

    def __init__(
        self,
        periods_per_year: int = 252,
        forward_periods: int = 1,
        quantiles: int = 5
    ):
        """
        Initialize the evaluator.

        Args:
            periods_per_year: Trading periods per year (252 for daily, 8760 for hourly)
            forward_periods: Periods ahead for return calculation
            quantiles: Number of quantiles for portfolio analysis
        """
        self.periods_per_year = periods_per_year
        self.forward_periods = forward_periods
        self.quantiles = quantiles

    def calculate_ic(
        self,
        factor: pd.Series,
        returns: pd.Series
    ) -> Tuple[float, float, float, float]:
        """
        Calculate Information Coefficient.

        Args:
            factor: Factor values
            returns: Forward returns

        Returns:
            Tuple of (IC, Rank IC, T-stat, P-value)
        """
        # Align data
        aligned = pd.DataFrame({"factor": factor, "returns": returns}).dropna()

        if len(aligned) < 30:
            return 0.0, 0.0, 0.0, 1.0

        # Pearson correlation (IC)
        ic, p_value = stats.pearsonr(aligned["factor"], aligned["returns"])

        # Spearman correlation (Rank IC)
        rank_ic, _ = stats.spearmanr(aligned["factor"], aligned["returns"])

        # T-statistic
        n = len(aligned)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2) if abs(ic) < 1 else 0

        return float(ic), float(rank_ic), float(t_stat), float(p_value)

    def calculate_ic_series(
        self,
        factor: pd.Series,
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate rolling IC series.

        Args:
            factor: Factor values
            returns: Forward returns
            window: Rolling window size

        Returns:
            Series of rolling IC values
        """
        aligned = pd.DataFrame({"factor": factor, "returns": returns}).dropna()

        ic_series = aligned["factor"].rolling(window).corr(aligned["returns"])

        return ic_series

    def calculate_turnover(self, factor: pd.Series) -> float:
        """
        Calculate factor turnover.

        Turnover measures how much the factor changes period to period.
        High turnover = more trading required.

        Args:
            factor: Factor values

        Returns:
            Average turnover (0-1 scale)
        """
        factor_clean = factor.dropna()

        if len(factor_clean) < 2:
            return 0.0

        # Normalize factor to ranks for turnover calculation
        ranks = factor_clean.rank(pct=True)

        # Calculate change in ranks
        rank_changes = ranks.diff().abs()

        return float(rank_changes.mean())

    def calculate_sharpe(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Period returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Annualized Sharpe ratio
        """
        returns_clean = returns.dropna()

        if len(returns_clean) < 2:
            return 0.0

        excess_returns = returns_clean - risk_free_rate / self.periods_per_year
        mean_return = excess_returns.mean()
        std_return = excess_returns.std()

        if std_return == 0:
            return 0.0

        return float(mean_return / std_return * np.sqrt(self.periods_per_year))

    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            cumulative_returns: Cumulative return series (not compounded)

        Returns:
            Maximum drawdown (negative number)
        """
        wealth = (1 + cumulative_returns)
        peak = wealth.expanding().max()
        drawdown = (wealth - peak) / peak

        return float(drawdown.min())

    def factor_returns(
        self,
        factor: pd.Series,
        returns: pd.Series,
        long_short: bool = True
    ) -> pd.Series:
        """
        Calculate factor portfolio returns.

        Creates a portfolio that goes long high factor values
        and short low factor values.

        Args:
            factor: Factor values
            returns: Forward returns
            long_short: If True, long top quintile and short bottom

        Returns:
            Factor portfolio returns series
        """
        aligned = pd.DataFrame({"factor": factor, "returns": returns}).dropna()

        # Calculate quintile positions
        aligned["quintile"] = pd.qcut(
            aligned["factor"],
            q=self.quantiles,
            labels=False,
            duplicates="drop"
        )

        # Long top quintile
        top_returns = aligned[aligned["quintile"] == self.quantiles - 1]["returns"]

        if long_short:
            # Short bottom quintile
            bottom_returns = aligned[aligned["quintile"] == 0]["returns"]
            portfolio_returns = top_returns.mean() - bottom_returns.mean()
        else:
            portfolio_returns = top_returns.mean()

        # Build return series
        result = pd.Series(index=aligned.index, dtype=float)
        for date in aligned.index:
            row = aligned.loc[date]
            if row["quintile"] == self.quantiles - 1:
                result[date] = row["returns"]
            elif long_short and row["quintile"] == 0:
                result[date] = -row["returns"]
            else:
                result[date] = 0.0

        return result

    def evaluate(
        self,
        factor: pd.Series,
        returns: pd.Series,
        prices: Optional[pd.Series] = None
    ) -> FactorMetrics:
        """
        Comprehensive factor evaluation.

        Args:
            factor: Factor values (aligned with returns index)
            returns: Forward returns (1-period ahead returns)
            prices: Optional price series for additional analysis

        Returns:
            FactorMetrics with all calculated metrics
        """
        # Calculate IC metrics
        ic, rank_ic, t_stat, p_value = self.calculate_ic(factor, returns)

        # Rolling IC for IC-IR
        ic_series = self.calculate_ic_series(factor, returns)
        ic_ir = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0
        ic_positive_pct = (ic_series > 0).mean()

        # Turnover
        turnover = self.calculate_turnover(factor)

        # Factor returns for Sharpe, drawdown
        factor_rets = self.factor_returns(factor, returns)
        sharpe = self.calculate_sharpe(factor_rets)
        cum_rets = factor_rets.cumsum()
        max_dd = self.calculate_max_drawdown(cum_rets)

        # Additional metrics
        mean_return = factor_rets.mean() * self.periods_per_year
        volatility = factor_rets.std() * np.sqrt(self.periods_per_year)
        win_rate = (factor_rets > 0).mean()

        return FactorMetrics(
            ic=ic,
            ic_ir=float(ic_ir),
            rank_ic=rank_ic,
            ic_positive_pct=float(ic_positive_pct),
            turnover=turnover,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            mean_return=float(mean_return),
            volatility=float(volatility),
            win_rate=float(win_rate),
            t_stat=t_stat,
            p_value=p_value
        )

    def compare_factors(
        self,
        factors: Dict[str, pd.Series],
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple factors.

        Args:
            factors: Dict mapping factor names to factor values
            returns: Forward returns

        Returns:
            DataFrame comparing all factors
        """
        results = []

        for name, factor in factors.items():
            metrics = self.evaluate(factor, returns)
            result = metrics.to_dict()
            result["name"] = name
            result["quality_score"] = metrics.quality_score()
            results.append(result)

        df = pd.DataFrame(results).set_index("name")

        # Sort by quality score
        df = df.sort_values("quality_score", ascending=False)

        return df

    def ic_decay_analysis(
        self,
        factor: pd.Series,
        prices: pd.Series,
        max_periods: int = 20
    ) -> pd.DataFrame:
        """
        Analyze IC decay over different holding periods.

        Args:
            factor: Factor values
            prices: Price series
            max_periods: Maximum holding period to analyze

        Returns:
            DataFrame with IC at different horizons
        """
        results = []

        for periods in range(1, max_periods + 1):
            # Calculate forward returns for this period
            forward_returns = prices.pct_change(periods).shift(-periods)

            ic, rank_ic, t_stat, p_value = self.calculate_ic(factor, forward_returns)

            results.append({
                "periods": periods,
                "ic": ic,
                "rank_ic": rank_ic,
                "t_stat": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            })

        return pd.DataFrame(results).set_index("periods")


def evaluate_alpha_expression(
    expression: str,
    data: pd.DataFrame,
    forward_periods: int = 1,
    periods_per_year: int = 252
) -> FactorMetrics:
    """
    Convenience function to evaluate an alpha expression.

    Args:
        expression: Alpha expression string
        data: OHLCV DataFrame
        forward_periods: Periods ahead for return calculation
        periods_per_year: Trading periods per year

    Returns:
        FactorMetrics for the expression
    """
    from .alpha_generator import AlphaExpressionParser

    parser = AlphaExpressionParser()
    evaluator = FactorEvaluator(
        periods_per_year=periods_per_year,
        forward_periods=forward_periods
    )

    # Calculate factor values
    factor = parser.evaluate(expression, data)

    # Calculate forward returns
    returns = data["close"].pct_change(forward_periods).shift(-forward_periods)

    return evaluator.evaluate(factor, returns)


if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    from alpha_generator import AlphaExpressionParser, PREDEFINED_FACTORS

    print("LLM Alpha Mining - Factor Evaluator Demo")
    print("=" * 60)

    # Generate data
    print("\n1. Loading Data")
    print("-" * 40)
    data = generate_synthetic_data(["BTCUSDT"], days=500)
    btc_data = data["BTCUSDT"].ohlcv
    print(f"Data shape: {btc_data.shape}")
    print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")

    # Calculate forward returns
    returns = btc_data["close"].pct_change().shift(-1)

    # Initialize components
    parser = AlphaExpressionParser()
    evaluator = FactorEvaluator(periods_per_year=252)

    # Evaluate predefined factors
    print("\n2. Evaluating Predefined Factors")
    print("-" * 40)

    factor_values = {}
    for factor in PREDEFINED_FACTORS:
        try:
            values = parser.evaluate(factor.expression, btc_data)
            factor_values[factor.name] = values

            metrics = evaluator.evaluate(values, returns)
            print(f"\n{factor.name}:")
            print(f"  IC: {metrics.ic:.4f} (p={metrics.p_value:.4f})")
            print(f"  Rank IC: {metrics.rank_ic:.4f}")
            print(f"  Sharpe: {metrics.sharpe_ratio:.2f}")
            print(f"  Max DD: {metrics.max_drawdown:.2%}")
            print(f"  Quality Score: {metrics.quality_score():.1f}/100")
            print(f"  Significant: {metrics.is_significant()}")
        except Exception as e:
            print(f"\n{factor.name}: Error - {e}")

    # Compare all factors
    print("\n3. Factor Comparison")
    print("-" * 40)

    comparison = evaluator.compare_factors(factor_values, returns)
    print("\nFactor Rankings:")
    print(comparison[["ic", "sharpe_ratio", "max_drawdown", "quality_score"]].round(4))

    # IC decay analysis
    print("\n4. IC Decay Analysis (momentum_5d)")
    print("-" * 40)

    if "momentum_5d" in factor_values:
        decay = evaluator.ic_decay_analysis(
            factor_values["momentum_5d"],
            btc_data["close"],
            max_periods=10
        )
        print("\nIC at different horizons:")
        print(decay[["ic", "rank_ic", "significant"]].round(4))

    # Quick expression evaluation
    print("\n5. Quick Expression Evaluation")
    print("-" * 40)

    test_expression = "(close - ts_mean(close, 10)) / ts_std(close, 10)"
    print(f"Expression: {test_expression}")

    metrics = evaluate_alpha_expression(
        test_expression,
        btc_data,
        forward_periods=1
    )
    print(f"IC: {metrics.ic:.4f}")
    print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
    print(f"Quality Score: {metrics.quality_score():.1f}/100")

    print("\n" + "=" * 60)
    print("Demo complete!")
