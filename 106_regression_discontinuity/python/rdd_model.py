"""
Regression Discontinuity Design Model Implementation

This module provides core functionality for RDD estimation including:
- Local linear regression with kernel weighting
- Optimal bandwidth selection (IK and CCT methods)
- Sharp and fuzzy RDD estimation
- Validation tests (McCrary density test, covariate balance)
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Callable
from scipy import stats
from scipy.optimize import minimize_scalar


class Kernel(Enum):
    """Kernel functions for local linear regression."""
    TRIANGULAR = "triangular"
    UNIFORM = "uniform"
    EPANECHNIKOV = "epanechnikov"


def kernel_function(kernel: Kernel) -> Callable[[np.ndarray], np.ndarray]:
    """Return kernel function for given kernel type."""
    if kernel == Kernel.TRIANGULAR:
        return lambda u: np.maximum(0, 1 - np.abs(u))
    elif kernel == Kernel.UNIFORM:
        return lambda u: np.where(np.abs(u) <= 1, 0.5, 0)
    elif kernel == Kernel.EPANECHNIKOV:
        return lambda u: np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


@dataclass
class RDDResults:
    """Results from RDD estimation."""
    tau: float  # Treatment effect estimate
    se: float  # Standard error
    ci_lower: float  # Lower bound of confidence interval
    ci_upper: float  # Upper bound of confidence interval
    t_stat: float  # t-statistic
    p_value: float  # p-value for test that tau = 0
    bandwidth: float  # Bandwidth used
    n_left: int  # Number of observations left of cutoff
    n_right: int  # Number of observations right of cutoff
    alpha_left: float  # Intercept estimate left of cutoff
    alpha_right: float  # Intercept estimate right of cutoff
    beta_left: float  # Slope estimate left of cutoff
    beta_right: float  # Slope estimate right of cutoff


@dataclass
class DensityTestResult:
    """Results from McCrary density test."""
    theta: float  # Log difference in density
    se: float  # Standard error
    z_stat: float  # z-statistic
    p_value: float  # p-value for test of no discontinuity
    density_left: float  # Estimated density left of cutoff
    density_right: float  # Estimated density right of cutoff


@dataclass
class PlaceboTestResult:
    """Results from placebo test at false cutoffs."""
    cutoff: float  # Placebo cutoff value
    tau: float  # Treatment effect estimate
    se: float  # Standard error
    p_value: float  # p-value


class RegressionDiscontinuity:
    """
    Regression Discontinuity Design estimator.

    Implements local linear regression with kernel weighting for both
    sharp and fuzzy RDD designs.

    Parameters
    ----------
    cutoff : float
        The threshold value where treatment assignment changes.
    bandwidth : float or str
        Bandwidth for local regression. If 'optimal', uses IK method.
        If 'cct', uses CCT robust bandwidth.
    kernel : Kernel
        Kernel function for weighting observations.
    order : int
        Order of local polynomial (1 for local linear).
    alpha : float
        Significance level for confidence intervals.
    fuzzy : bool
        If True, use fuzzy RDD (treatment is probabilistic).
    """

    def __init__(
        self,
        cutoff: float,
        bandwidth: float | str = 'optimal',
        kernel: Kernel = Kernel.TRIANGULAR,
        order: int = 1,
        alpha: float = 0.05,
        fuzzy: bool = False,
    ):
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.order = order
        self.alpha = alpha
        self.fuzzy = fuzzy
        self._kernel_fn = kernel_function(kernel)
        self._fitted = False
        self._results: Optional[RDDResults] = None

    def fit(
        self,
        running_var: np.ndarray,
        outcome: np.ndarray,
        treatment: Optional[np.ndarray] = None,
        covariates: Optional[np.ndarray] = None,
    ) -> RDDResults:
        """
        Fit the RDD model.

        Parameters
        ----------
        running_var : np.ndarray
            The running variable (e.g., market cap rank).
        outcome : np.ndarray
            The outcome variable (e.g., returns).
        treatment : np.ndarray, optional
            Treatment indicator for fuzzy RDD.
        covariates : np.ndarray, optional
            Additional covariates to control for.

        Returns
        -------
        RDDResults
            Estimation results including treatment effect and standard error.
        """
        running_var = np.asarray(running_var, dtype=np.float64)
        outcome = np.asarray(outcome, dtype=np.float64)

        # Center running variable at cutoff
        x = running_var - self.cutoff
        y = outcome

        # Determine bandwidth
        if isinstance(self.bandwidth, str):
            if self.bandwidth == 'optimal':
                h = self._ik_bandwidth(x, y)
            elif self.bandwidth == 'cct':
                h = self._cct_bandwidth(x, y)
            else:
                raise ValueError(f"Unknown bandwidth method: {self.bandwidth}")
        else:
            h = self.bandwidth

        # Select observations within bandwidth
        in_window = np.abs(x) <= h
        x_w = x[in_window]
        y_w = y[in_window]

        # Separate left and right of cutoff
        left_mask = x_w < 0
        right_mask = x_w >= 0

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left < 5 or n_right < 5:
            raise ValueError(
                f"Insufficient observations: {n_left} left, {n_right} right. "
                "Consider increasing bandwidth."
            )

        # Compute kernel weights
        weights = self._kernel_fn(x_w / h)

        # Fit local linear regression on each side
        if self.fuzzy and treatment is not None:
            # Fuzzy RDD: use 2SLS
            treatment = np.asarray(treatment, dtype=np.float64)
            d_w = treatment[in_window]
            results = self._fit_fuzzy(x_w, y_w, d_w, weights, left_mask, right_mask)
        else:
            # Sharp RDD: regular local linear
            results = self._fit_sharp(x_w, y_w, weights, left_mask, right_mask)

        # Add bandwidth and sample sizes
        results.bandwidth = h
        results.n_left = n_left
        results.n_right = n_right

        self._fitted = True
        self._results = results
        self._data = (running_var, outcome)

        return results

    def _fit_sharp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> RDDResults:
        """Fit sharp RDD using weighted local linear regression."""
        # Left side
        x_left = x[left_mask]
        y_left = y[left_mask]
        w_left = weights[left_mask]
        alpha_left, beta_left, var_left = self._weighted_ols(x_left, y_left, w_left)

        # Right side
        x_right = x[right_mask]
        y_right = y[right_mask]
        w_right = weights[right_mask]
        alpha_right, beta_right, var_right = self._weighted_ols(x_right, y_right, w_right)

        # Treatment effect is difference in intercepts
        tau = alpha_right - alpha_left

        # Standard error (assuming independence)
        se = np.sqrt(var_left + var_right)

        # Confidence interval
        z = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = tau - z * se
        ci_upper = tau + z * se

        # Test statistic and p-value
        t_stat = tau / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

        return RDDResults(
            tau=tau,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            t_stat=t_stat,
            p_value=p_value,
            bandwidth=0,  # Set later
            n_left=0,
            n_right=0,
            alpha_left=alpha_left,
            alpha_right=alpha_right,
            beta_left=beta_left,
            beta_right=beta_right,
        )

    def _fit_fuzzy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        weights: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> RDDResults:
        """Fit fuzzy RDD using instrumental variables."""
        # First stage: treatment on cutoff
        d_left = d[left_mask]
        d_right = d[right_mask]
        w_left = weights[left_mask]
        w_right = weights[right_mask]
        x_left = x[left_mask]
        x_right = x[right_mask]

        alpha_d_left, _, _ = self._weighted_ols(x_left, d_left, w_left)
        alpha_d_right, _, _ = self._weighted_ols(x_right, d_right, w_right)

        first_stage = alpha_d_right - alpha_d_left

        if np.abs(first_stage) < 0.01:
            raise ValueError(
                "First stage is too weak for fuzzy RDD. "
                f"Probability jump: {first_stage:.4f}"
            )

        # Reduced form: outcome on cutoff
        y_left = y[left_mask]
        y_right = y[right_mask]

        alpha_left, beta_left, var_left = self._weighted_ols(x_left, y_left, w_left)
        alpha_right, beta_right, var_right = self._weighted_ols(x_right, y_right, w_right)

        reduced_form = alpha_right - alpha_left

        # IV estimate: reduced form / first stage
        tau = reduced_form / first_stage

        # Standard error (delta method approximation)
        se_rf = np.sqrt(var_left + var_right)
        se = se_rf / np.abs(first_stage)

        # Confidence interval
        z = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = tau - z * se
        ci_upper = tau + z * se

        # Test statistic and p-value
        t_stat = tau / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

        return RDDResults(
            tau=tau,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            t_stat=t_stat,
            p_value=p_value,
            bandwidth=0,
            n_left=0,
            n_right=0,
            alpha_left=alpha_left,
            alpha_right=alpha_right,
            beta_left=beta_left,
            beta_right=beta_right,
        )

    def _weighted_ols(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Weighted OLS for local linear regression.

        Returns intercept, slope, and variance of intercept.
        """
        n = len(x)
        if n < 2:
            return np.mean(y), 0.0, np.var(y) / n if n > 0 else 0.0

        # Weighted design matrix
        W = np.diag(w)
        X = np.column_stack([np.ones(n), x])

        # Weighted OLS: (X'WX)^{-1} X'Wy
        XtW = X.T @ W
        XtWX = XtW @ X
        XtWy = XtW @ y

        # Add small ridge for numerical stability
        XtWX += np.eye(2) * 1e-10

        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

        alpha = beta[0]
        slope = beta[1]

        # Residuals and variance
        resid = y - X @ beta
        sigma2 = np.sum(w * resid**2) / (np.sum(w) - 2)

        # Variance of intercept
        try:
            XtWX_inv = np.linalg.inv(XtWX)
            var_alpha = sigma2 * XtWX_inv[0, 0]
        except np.linalg.LinAlgError:
            var_alpha = sigma2 / np.sum(w)

        return alpha, slope, max(0, var_alpha)

    def _ik_bandwidth(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Imbens-Kalyanaraman optimal bandwidth selection.

        Implements the method from:
        Imbens, G., & Kalyanaraman, K. (2012). Optimal Bandwidth Choice for the
        Regression Discontinuity Estimator. Review of Economic Studies, 79(3), 933-959.
        """
        n = len(x)
        left_mask = x < 0
        right_mask = x >= 0

        # Step 1: Estimate variance
        h_pilot = 2.84 * np.std(x) * n**(-1/5)  # Silverman's rule

        in_pilot = np.abs(x) <= h_pilot
        x_pilot = x[in_pilot]
        y_pilot = y[in_pilot]

        if len(x_pilot) < 10:
            # Fallback to fraction of data range
            return (np.max(x) - np.min(x)) / 4

        # Estimate conditional variance
        sigma2_left = np.var(y[left_mask]) if np.sum(left_mask) > 1 else np.var(y)
        sigma2_right = np.var(y[right_mask]) if np.sum(right_mask) > 1 else np.var(y)
        sigma2 = (sigma2_left + sigma2_right) / 2

        # Step 2: Estimate curvature
        # Use global polynomial to estimate second derivative
        try:
            coef_left = np.polyfit(x[left_mask], y[left_mask], 2)
            coef_right = np.polyfit(x[right_mask], y[right_mask], 2)
            m2_left = 2 * coef_left[0]  # Second derivative
            m2_right = 2 * coef_right[0]
        except (np.linalg.LinAlgError, ValueError):
            m2_left = m2_right = 0.1

        # Regularization term
        r = 2.702 * ((sigma2_left + sigma2_right) / (
            np.abs(m2_right - m2_left) + 0.001))**(1/5)

        # IK optimal bandwidth
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_eff = n_left + n_right

        h_opt = 2.84 * (sigma2 / (n_eff * (np.abs(m2_right - m2_left) + 0.001)**2))**(1/5)

        # Bound bandwidth
        data_range = np.max(x) - np.min(x)
        h_opt = min(h_opt, data_range / 2)
        h_opt = max(h_opt, data_range / 20)

        return h_opt

    def _cct_bandwidth(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calonico-Cattaneo-Titiunik robust bandwidth.

        Simplified version of the CCT method.
        """
        # Start with IK bandwidth, then adjust
        h_ik = self._ik_bandwidth(x, y)

        # CCT suggests using slightly smaller bandwidth for robustness
        h_cct = h_ik * 0.85

        return h_cct

    def predict(self, running_var: np.ndarray) -> np.ndarray:
        """Predict outcome for new running variable values."""
        if not self._fitted or self._results is None:
            raise ValueError("Model must be fitted before prediction.")

        x = np.asarray(running_var) - self.cutoff
        predictions = np.zeros_like(x, dtype=np.float64)

        left_mask = x < 0
        right_mask = x >= 0

        predictions[left_mask] = (
            self._results.alpha_left +
            self._results.beta_left * x[left_mask]
        )
        predictions[right_mask] = (
            self._results.alpha_right +
            self._results.beta_right * x[right_mask]
        )

        return predictions


class RDDValidator:
    """
    Validation tools for RDD analysis.

    Provides tests to verify RDD assumptions:
    - Density test (McCrary) for manipulation
    - Placebo tests at false cutoffs
    - Covariate balance tests
    """

    def __init__(self, rdd_model: RegressionDiscontinuity):
        self.model = rdd_model

    def density_test(
        self,
        running_var: Optional[np.ndarray] = None,
        n_bins: int = 20,
    ) -> DensityTestResult:
        """
        McCrary density test for manipulation of the running variable.

        Tests if there is a discontinuity in the density of the running
        variable at the cutoff. A significant discontinuity suggests
        potential manipulation.

        Parameters
        ----------
        running_var : np.ndarray, optional
            Running variable. If None, uses data from fitted model.
        n_bins : int
            Number of bins for histogram on each side.

        Returns
        -------
        DensityTestResult
            Test results including z-statistic and p-value.
        """
        if running_var is None:
            if not hasattr(self.model, '_data'):
                raise ValueError("Must fit model or provide running_var")
            running_var = self.model._data[0]

        x = np.asarray(running_var) - self.model.cutoff

        # Separate observations
        left = x[x < 0]
        right = x[x >= 0]

        if len(left) < 10 or len(right) < 10:
            raise ValueError("Insufficient observations for density test")

        # Compute bin width
        h = (np.max(x) - np.min(x)) / (2 * n_bins)

        # Count in bins near cutoff
        bins_left = np.linspace(-h * n_bins, 0, n_bins + 1)
        bins_right = np.linspace(0, h * n_bins, n_bins + 1)

        counts_left, _ = np.histogram(left, bins=bins_left)
        counts_right, _ = np.histogram(right, bins=bins_right)

        # Estimate density just below and above cutoff
        density_left = counts_left[-1] / (h * len(x))
        density_right = counts_right[0] / (h * len(x))

        # Log difference
        if density_left <= 0 or density_right <= 0:
            theta = 0.0
            se = 1.0
        else:
            theta = np.log(density_right) - np.log(density_left)

            # Standard error (asymptotic)
            se = np.sqrt(1 / max(counts_left[-1], 1) + 1 / max(counts_right[0], 1))

        z_stat = theta / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        return DensityTestResult(
            theta=theta,
            se=se,
            z_stat=z_stat,
            p_value=p_value,
            density_left=density_left,
            density_right=density_right,
        )

    def placebo_test(
        self,
        cutoffs: List[float],
        running_var: Optional[np.ndarray] = None,
        outcome: Optional[np.ndarray] = None,
    ) -> List[PlaceboTestResult]:
        """
        Test for discontinuities at placebo (false) cutoffs.

        If there are significant effects at placebo cutoffs, it suggests
        the effect at the true cutoff might be spurious.

        Parameters
        ----------
        cutoffs : list of float
            Placebo cutoff values to test.
        running_var : np.ndarray, optional
            Running variable.
        outcome : np.ndarray, optional
            Outcome variable.

        Returns
        -------
        list of PlaceboTestResult
            Results for each placebo cutoff.
        """
        if running_var is None or outcome is None:
            if not hasattr(self.model, '_data'):
                raise ValueError("Must fit model or provide data")
            running_var, outcome = self.model._data

        results = []
        for c in cutoffs:
            # Create new RDD model at placebo cutoff
            placebo_model = RegressionDiscontinuity(
                cutoff=c,
                bandwidth=self.model.bandwidth,
                kernel=self.model.kernel,
                order=self.model.order,
            )

            try:
                placebo_results = placebo_model.fit(running_var, outcome)
                results.append(PlaceboTestResult(
                    cutoff=c,
                    tau=placebo_results.tau,
                    se=placebo_results.se,
                    p_value=placebo_results.p_value,
                ))
            except ValueError:
                # Not enough data at this cutoff
                results.append(PlaceboTestResult(
                    cutoff=c,
                    tau=np.nan,
                    se=np.nan,
                    p_value=np.nan,
                ))

        return results

    def covariate_balance(
        self,
        covariates: np.ndarray,
        running_var: Optional[np.ndarray] = None,
        covariate_names: Optional[List[str]] = None,
    ) -> dict:
        """
        Test for balance of covariates at the cutoff.

        Covariates should not show discontinuities at the cutoff,
        as this would indicate potential confounding.

        Parameters
        ----------
        covariates : np.ndarray
            Matrix of covariates (n_obs x n_covariates).
        running_var : np.ndarray, optional
            Running variable.
        covariate_names : list of str, optional
            Names for each covariate.

        Returns
        -------
        dict
            Balance test results for each covariate.
        """
        if running_var is None:
            if not hasattr(self.model, '_data'):
                raise ValueError("Must fit model or provide running_var")
            running_var = self.model._data[0]

        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)

        n_covariates = covariates.shape[1]
        if covariate_names is None:
            covariate_names = [f"covariate_{i}" for i in range(n_covariates)]

        results = {}
        for i, name in enumerate(covariate_names):
            cov_model = RegressionDiscontinuity(
                cutoff=self.model.cutoff,
                bandwidth=self.model.bandwidth,
                kernel=self.model.kernel,
            )

            try:
                cov_results = cov_model.fit(running_var, covariates[:, i])
                results[name] = {
                    'difference': cov_results.tau,
                    'se': cov_results.se,
                    'p_value': cov_results.p_value,
                    'balanced': cov_results.p_value > 0.05,
                }
            except ValueError:
                results[name] = {
                    'difference': np.nan,
                    'se': np.nan,
                    'p_value': np.nan,
                    'balanced': None,
                }

        return results


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Relative Strength Index."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros(len(prices))
    avg_losses = np.zeros(len(prices))

    # Initial SMA
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])

    # Exponential smoothing
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.where(avg_losses > 0, avg_gains / avg_losses, 100)
    rsi = 100 - (100 / (1 + rs))

    return rsi


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulate data with treatment effect at cutoff
    n = 1000
    x = np.random.uniform(-100, 100, n)  # Running variable

    # True treatment effect = 5 for x >= 0
    treatment = (x >= 0).astype(float)
    y = 2 + 0.5 * x + 5 * treatment + np.random.normal(0, 10, n)

    # Fit RDD model
    rdd = RegressionDiscontinuity(cutoff=0, bandwidth='optimal')
    results = rdd.fit(x, y)

    print("RDD Results:")
    print(f"  Treatment Effect: {results.tau:.3f} (true: 5.0)")
    print(f"  Standard Error: {results.se:.3f}")
    print(f"  95% CI: [{results.ci_lower:.3f}, {results.ci_upper:.3f}]")
    print(f"  p-value: {results.p_value:.4f}")
    print(f"  Bandwidth: {results.bandwidth:.3f}")
    print(f"  N left: {results.n_left}, N right: {results.n_right}")

    # Validation tests
    validator = RDDValidator(rdd)

    density_result = validator.density_test(x)
    print(f"\nDensity Test (McCrary):")
    print(f"  p-value: {density_result.p_value:.4f}")
    print(f"  (p > 0.05 suggests no manipulation)")

    placebo = validator.placebo_test([-50, 50], x, y)
    print(f"\nPlacebo Tests:")
    for p in placebo:
        print(f"  Cutoff {p.cutoff}: tau={p.tau:.3f}, p={p.p_value:.4f}")
