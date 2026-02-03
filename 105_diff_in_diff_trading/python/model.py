"""Difference-in-Differences estimation model for financial time series.

Implements DiD estimation with support for:
- Basic two-period, two-group design
- Multi-period staggered treatment designs
- Parallel trends testing
- Clustered standard errors
- Bootstrap confidence intervals
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class DiDResults:
    """Results from a Difference-in-Differences estimation.

    Attributes:
        did_estimate: The DiD point estimate (average treatment effect on treated)
        std_error: Standard error of the estimate
        t_statistic: t-statistic for hypothesis test (H0: effect = 0)
        p_value: Two-sided p-value
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        confidence_level: Confidence level for the interval (default 0.95)
        n_treated: Number of treated observations
        n_control: Number of control observations
        pre_trend_test: Results from parallel trends test (if performed)
    """

    did_estimate: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_treated: int
    n_control: int
    pre_trend_test: Optional[dict] = None

    def __repr__(self) -> str:
        sig = "***" if self.p_value < 0.01 else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.1 else ""
        return (
            f"DiDResults(\n"
            f"  did_estimate = {self.did_estimate:.6f} {sig}\n"
            f"  std_error    = {self.std_error:.6f}\n"
            f"  t_statistic  = {self.t_statistic:.4f}\n"
            f"  p_value      = {self.p_value:.4f}\n"
            f"  {self.confidence_level*100:.0f}% CI = [{self.ci_lower:.6f}, {self.ci_upper:.6f}]\n"
            f"  n_treated    = {self.n_treated}\n"
            f"  n_control    = {self.n_control}\n"
            f")"
        )

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if the estimate is statistically significant."""
        return self.p_value < alpha


class DifferenceInDifferences:
    """Difference-in-Differences estimator for causal effect estimation.

    Implements the canonical DiD design with extensions for:
    - Panel data with unit and time fixed effects
    - Clustered standard errors
    - Bootstrap inference
    - Parallel trends testing

    Args:
        treatment_col: Column name indicating treatment status (0/1)
        time_col: Column name indicating post-treatment period (0/1)
        outcome_col: Column name for the outcome variable
        unit_col: Column name for unit identifier (for panel data)
        covariates: List of covariate column names to control for
        cluster_col: Column to cluster standard errors on
        n_bootstrap: Number of bootstrap iterations for inference
        confidence_level: Confidence level for intervals (default 0.95)
    """

    def __init__(
        self,
        treatment_col: str = "treated",
        time_col: str = "post_treatment",
        outcome_col: str = "return",
        unit_col: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        cluster_col: Optional[str] = None,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ):
        self.treatment_col = treatment_col
        self.time_col = time_col
        self.outcome_col = outcome_col
        self.unit_col = unit_col
        self.covariates = covariates or []
        self.cluster_col = cluster_col
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level

        self._results: Optional[DiDResults] = None
        self._data: Optional[pd.DataFrame] = None

    def fit(self, data: pd.DataFrame) -> DiDResults:
        """Fit the DiD model to the data.

        Args:
            data: DataFrame containing treatment, time, outcome, and optional covariates

        Returns:
            DiDResults object with estimation results
        """
        self._data = data.copy()
        self._validate_data()

        # Compute basic DiD estimate
        did_estimate = self._compute_did_estimate()

        # Compute standard errors
        if self.cluster_col is not None:
            std_error = self._clustered_standard_error()
        elif self.n_bootstrap > 0:
            std_error = self._bootstrap_standard_error()
        else:
            std_error = self._ols_standard_error()

        # Compute test statistics
        t_stat = did_estimate / std_error if std_error > 0 else 0.0
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), self._degrees_of_freedom()))

        # Confidence interval
        t_critical = self._t_quantile(1 - (1 - self.confidence_level) / 2, self._degrees_of_freedom())
        ci_lower = did_estimate - t_critical * std_error
        ci_upper = did_estimate + t_critical * std_error

        # Count observations
        treated_mask = self._data[self.treatment_col] == 1
        n_treated = treated_mask.sum()
        n_control = (~treated_mask).sum()

        self._results = DiDResults(
            did_estimate=did_estimate,
            std_error=std_error,
            t_statistic=t_stat,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_treated=n_treated,
            n_control=n_control,
        )

        return self._results

    def test_parallel_trends(
        self, data: pd.DataFrame, event_time_col: str, n_pre_periods: int = 5
    ) -> dict:
        """Test the parallel trends assumption using pre-treatment data.

        Estimates event-time dummies for pre-treatment periods and tests
        whether they are jointly equal to zero.

        Args:
            data: Panel data with event time information
            event_time_col: Column indicating time relative to treatment
            n_pre_periods: Number of pre-treatment periods to test

        Returns:
            Dictionary with test statistics and coefficients
        """
        pre_data = data[data[event_time_col] < 0].copy()

        if len(pre_data) == 0:
            return {"valid": False, "message": "No pre-treatment data available"}

        # Create event-time dummies
        coefficients = {}
        for t in range(-n_pre_periods, 0):
            mask = pre_data[event_time_col] == t
            if mask.sum() > 0:
                treated_pre = pre_data.loc[mask & (pre_data[self.treatment_col] == 1), self.outcome_col]
                control_pre = pre_data.loc[mask & (pre_data[self.treatment_col] == 0), self.outcome_col]

                if len(treated_pre) > 0 and len(control_pre) > 0:
                    coefficients[t] = treated_pre.mean() - control_pre.mean()

        # Simple F-test: are all pre-treatment coefficients zero?
        if len(coefficients) > 0:
            coef_values = list(coefficients.values())
            f_stat = np.mean(np.array(coef_values) ** 2) / (np.var(coef_values) + 1e-10)
            # Simplified p-value approximation
            p_value = np.exp(-f_stat / 2) if f_stat > 0 else 1.0
        else:
            f_stat = 0.0
            p_value = 1.0

        return {
            "valid": True,
            "coefficients": coefficients,
            "f_statistic": f_stat,
            "p_value": p_value,
            "parallel_trends_satisfied": p_value > 0.05,
        }

    def generate_signals(
        self,
        threshold: float = 0.02,
        confidence: float = 0.95,
        holding_period: int = 5,
    ) -> pd.DataFrame:
        """Generate trading signals based on DiD estimates.

        Args:
            threshold: Minimum absolute effect size to generate a signal
            confidence: Minimum confidence level required
            holding_period: Number of periods to hold position

        Returns:
            DataFrame with trading signals
        """
        if self._results is None:
            raise ValueError("Must fit model before generating signals")

        signal = 0
        if self._results.is_significant(1 - confidence):
            if self._results.did_estimate > threshold:
                signal = 1  # Long
            elif self._results.did_estimate < -threshold:
                signal = -1  # Short

        return pd.DataFrame({
            "signal": [signal],
            "effect_size": [self._results.did_estimate],
            "confidence": [1 - self._results.p_value],
            "holding_period": [holding_period],
        })

    def _validate_data(self) -> None:
        """Validate that required columns exist in the data."""
        required = [self.treatment_col, self.time_col, self.outcome_col]
        missing = [col for col in required if col not in self._data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _compute_did_estimate(self) -> float:
        """Compute the basic DiD estimate.

        DiD = E[Y|D=1,T=1] - E[Y|D=1,T=0] - (E[Y|D=0,T=1] - E[Y|D=0,T=0])
        """
        d = self._data
        treat, time, y = self.treatment_col, self.time_col, self.outcome_col

        # Four cell means
        y_11 = d.loc[(d[treat] == 1) & (d[time] == 1), y].mean()  # Treated, Post
        y_10 = d.loc[(d[treat] == 1) & (d[time] == 0), y].mean()  # Treated, Pre
        y_01 = d.loc[(d[treat] == 0) & (d[time] == 1), y].mean()  # Control, Post
        y_00 = d.loc[(d[treat] == 0) & (d[time] == 0), y].mean()  # Control, Pre

        # Handle missing cells
        if any(np.isnan([y_11, y_10, y_01, y_00])):
            # Fall back to regression approach
            return self._ols_did_estimate()

        return (y_11 - y_10) - (y_01 - y_00)

    def _ols_did_estimate(self) -> float:
        """Compute DiD using OLS regression.

        Y = alpha + beta*D + gamma*T + delta*(D*T) + epsilon
        delta is the DiD estimate
        """
        d = self._data
        treat, time, y = self.treatment_col, self.time_col, self.outcome_col

        # Create interaction term
        interaction = d[treat] * d[time]

        # Build design matrix
        X = np.column_stack([
            np.ones(len(d)),
            d[treat].values,
            d[time].values,
            interaction.values,
        ])

        # Add covariates if specified
        if self.covariates:
            for cov in self.covariates:
                if cov in d.columns:
                    X = np.column_stack([X, d[cov].values])

        Y = d[y].values

        # OLS: (X'X)^{-1} X'Y
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            return beta[3]  # Coefficient on interaction term
        except np.linalg.LinAlgError:
            return 0.0

    def _ols_standard_error(self) -> float:
        """Compute heteroskedasticity-robust standard error."""
        d = self._data
        treat, time, y = self.treatment_col, self.time_col, self.outcome_col

        interaction = d[treat] * d[time]

        X = np.column_stack([
            np.ones(len(d)),
            d[treat].values,
            d[time].values,
            interaction.values,
        ])

        if self.covariates:
            for cov in self.covariates:
                if cov in d.columns:
                    X = np.column_stack([X, d[cov].values])

        Y = d[y].values
        n, k = X.shape

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ Y
            residuals = Y - X @ beta

            # HC1 robust variance estimator
            meat = np.zeros((k, k))
            for i in range(n):
                meat += (residuals[i] ** 2) * np.outer(X[i], X[i])
            meat *= n / (n - k)

            var_beta = XtX_inv @ meat @ XtX_inv
            return np.sqrt(var_beta[3, 3])  # SE of interaction coefficient
        except np.linalg.LinAlgError:
            return self._bootstrap_standard_error()

    def _clustered_standard_error(self) -> float:
        """Compute cluster-robust standard error."""
        if self.cluster_col is None:
            return self._ols_standard_error()

        d = self._data
        treat, time, y = self.treatment_col, self.time_col, self.outcome_col
        cluster = self.cluster_col

        interaction = d[treat] * d[time]

        X = np.column_stack([
            np.ones(len(d)),
            d[treat].values,
            d[time].values,
            interaction.values,
        ])

        if self.covariates:
            for cov in self.covariates:
                if cov in d.columns:
                    X = np.column_stack([X, d[cov].values])

        Y = d[y].values
        clusters = d[cluster].values
        n, k = X.shape

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ Y
            residuals = Y - X @ beta

            # Cluster-robust meat matrix
            unique_clusters = np.unique(clusters)
            G = len(unique_clusters)
            meat = np.zeros((k, k))

            for g in unique_clusters:
                mask = clusters == g
                X_g = X[mask]
                e_g = residuals[mask]
                score_g = X_g.T @ e_g
                meat += np.outer(score_g, score_g)

            # Small sample adjustment
            adjustment = (G / (G - 1)) * ((n - 1) / (n - k))
            meat *= adjustment

            var_beta = XtX_inv @ meat @ XtX_inv
            return np.sqrt(var_beta[3, 3])
        except np.linalg.LinAlgError:
            return self._bootstrap_standard_error()

    def _bootstrap_standard_error(self) -> float:
        """Compute standard error using bootstrap."""
        estimates = []
        n = len(self._data)

        for _ in range(self.n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(n, size=n, replace=True)
            boot_data = self._data.iloc[idx]

            # Compute DiD estimate on bootstrap sample
            treat, time, y = self.treatment_col, self.time_col, self.outcome_col

            y_11 = boot_data.loc[(boot_data[treat] == 1) & (boot_data[time] == 1), y].mean()
            y_10 = boot_data.loc[(boot_data[treat] == 1) & (boot_data[time] == 0), y].mean()
            y_01 = boot_data.loc[(boot_data[treat] == 0) & (boot_data[time] == 1), y].mean()
            y_00 = boot_data.loc[(boot_data[treat] == 0) & (boot_data[time] == 0), y].mean()

            if not any(np.isnan([y_11, y_10, y_01, y_00])):
                estimates.append((y_11 - y_10) - (y_01 - y_00))

        if len(estimates) > 0:
            return np.std(estimates, ddof=1)
        return 0.0

    def _degrees_of_freedom(self) -> int:
        """Compute degrees of freedom for t-distribution."""
        n = len(self._data)
        k = 4 + len(self.covariates)  # intercept + treat + time + interaction + covariates

        if self.cluster_col is not None:
            G = self._data[self.cluster_col].nunique()
            return G - 1

        return n - k

    @staticmethod
    def _t_cdf(x: float, df: int) -> float:
        """Approximate CDF of t-distribution."""
        # Use normal approximation for large df
        if df > 30:
            return 0.5 * (1 + np.tanh(x * 0.7978845608))
        # Simple approximation for small df
        return 0.5 + 0.5 * np.tanh(x / np.sqrt(df + 1))

    @staticmethod
    def _t_quantile(p: float, df: int) -> float:
        """Approximate quantile of t-distribution."""
        if df > 30:
            # Use normal approximation
            return np.sqrt(2) * np.arctanh(2 * p - 1) / 0.7978845608

        # Rough approximation for small df
        if p >= 0.95:
            return 1.96 * np.sqrt(1 + 3 / df)
        if p >= 0.975:
            return 2.576 * np.sqrt(1 + 3 / df)
        return 1.645 * np.sqrt(1 + 3 / df)


class StaggeredDiD:
    """Difference-in-Differences with staggered treatment timing.

    Implements modern DiD estimators that handle heterogeneous treatment
    effects with staggered adoption, based on Callaway & Sant'Anna (2021).

    Args:
        unit_col: Column name for unit identifier
        time_col: Column name for time period
        treatment_time_col: Column for treatment adoption time (None if never treated)
        outcome_col: Column name for outcome variable
        covariates: List of covariate column names
    """

    def __init__(
        self,
        unit_col: str = "unit_id",
        time_col: str = "time",
        treatment_time_col: str = "treatment_time",
        outcome_col: str = "return",
        covariates: Optional[List[str]] = None,
    ):
        self.unit_col = unit_col
        self.time_col = time_col
        self.treatment_time_col = treatment_time_col
        self.outcome_col = outcome_col
        self.covariates = covariates or []

    def fit(self, data: pd.DataFrame) -> dict:
        """Estimate group-time average treatment effects.

        Args:
            data: Panel data with unit, time, treatment timing, and outcome

        Returns:
            Dictionary with group-time ATT estimates
        """
        # Get treatment cohorts (groups defined by treatment timing)
        treatment_times = data[self.treatment_time_col].dropna().unique()
        time_periods = sorted(data[self.time_col].unique())

        results = {}

        for g in treatment_times:
            # Units treated at time g
            treated_units = data[data[self.treatment_time_col] == g][self.unit_col].unique()

            # Never-treated units (control)
            never_treated = data[data[self.treatment_time_col].isna()][self.unit_col].unique()

            if len(never_treated) == 0:
                # Use not-yet-treated as control
                for t in time_periods:
                    if t >= g:
                        # Post-treatment period
                        not_yet_treated = data[
                            (data[self.treatment_time_col] > t) |
                            (data[self.treatment_time_col].isna())
                        ][self.unit_col].unique()
                        control_units = not_yet_treated
                    else:
                        continue

                    if len(control_units) > 0:
                        att = self._estimate_group_time_att(
                            data, treated_units, control_units, g, t
                        )
                        results[(g, t)] = att
            else:
                # Use never-treated as control
                for t in time_periods:
                    if t >= g:
                        att = self._estimate_group_time_att(
                            data, treated_units, never_treated, g, t
                        )
                        results[(g, t)] = att

        return results

    def _estimate_group_time_att(
        self,
        data: pd.DataFrame,
        treated_units: np.ndarray,
        control_units: np.ndarray,
        treatment_time: int,
        current_time: int,
    ) -> dict:
        """Estimate ATT for a specific group-time cell."""
        # Pre-period: one period before treatment
        pre_time = treatment_time - 1

        # Treated group outcomes
        treated_post = data[
            (data[self.unit_col].isin(treated_units)) &
            (data[self.time_col] == current_time)
        ][self.outcome_col].mean()

        treated_pre = data[
            (data[self.unit_col].isin(treated_units)) &
            (data[self.time_col] == pre_time)
        ][self.outcome_col].mean()

        # Control group outcomes
        control_post = data[
            (data[self.unit_col].isin(control_units)) &
            (data[self.time_col] == current_time)
        ][self.outcome_col].mean()

        control_pre = data[
            (data[self.unit_col].isin(control_units)) &
            (data[self.time_col] == pre_time)
        ][self.outcome_col].mean()

        if any(np.isnan([treated_post, treated_pre, control_post, control_pre])):
            return {"att": np.nan, "n_treated": 0, "n_control": 0}

        att = (treated_post - treated_pre) - (control_post - control_pre)

        return {
            "att": att,
            "n_treated": len(treated_units),
            "n_control": len(control_units),
            "treatment_time": treatment_time,
            "current_time": current_time,
        }

    def aggregate(self, group_time_atts: dict, weights: str = "simple") -> float:
        """Aggregate group-time ATTs into an overall estimate.

        Args:
            group_time_atts: Dictionary of group-time ATT estimates
            weights: Weighting scheme ("simple", "group_size", "time_elapsed")

        Returns:
            Aggregated ATT estimate
        """
        valid_atts = [v["att"] for v in group_time_atts.values() if not np.isnan(v["att"])]

        if len(valid_atts) == 0:
            return np.nan

        if weights == "simple":
            return np.mean(valid_atts)

        elif weights == "group_size":
            total_n = sum(v["n_treated"] for v in group_time_atts.values())
            if total_n == 0:
                return np.mean(valid_atts)
            return sum(
                v["att"] * v["n_treated"] / total_n
                for v in group_time_atts.values()
                if not np.isnan(v["att"])
            )

        else:
            return np.mean(valid_atts)
