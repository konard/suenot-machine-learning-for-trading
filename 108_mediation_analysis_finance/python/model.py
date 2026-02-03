"""
Mediation Analysis Models

This module implements classical and causal mediation analysis methods
for financial applications.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import warnings


@dataclass
class MediationResults:
    """Results container for mediation analysis"""
    total_effect: float
    total_effect_se: float
    total_effect_pvalue: float

    direct_effect: float
    direct_effect_se: float
    direct_effect_pvalue: float

    indirect_effect: float
    indirect_effect_se: float
    indirect_effect_pvalue: float

    a_path: float  # X → M
    a_path_se: float
    a_path_pvalue: float

    b_path: float  # M → Y (controlling for X)
    b_path_se: float
    b_path_pvalue: float

    proportion_mediated: float
    sobel_z: float
    sobel_pvalue: float

    def __str__(self):
        mediation_type = "Full Mediation" if abs(self.direct_effect_pvalue) > 0.05 and self.indirect_effect_pvalue < 0.05 \
            else "Partial Mediation" if self.indirect_effect_pvalue < 0.05 and self.direct_effect_pvalue < 0.05 \
            else "No Mediation" if self.indirect_effect_pvalue > 0.05 \
            else "Direct Effect Only"

        return f"""
========== MEDIATION ANALYSIS RESULTS ==========

Path Coefficients:
    a (X → M):        {self.a_path:.4f} (SE={self.a_path_se:.4f}, p={self.a_path_pvalue:.4f})
    b (M → Y|X):      {self.b_path:.4f} (SE={self.b_path_se:.4f}, p={self.b_path_pvalue:.4f})

Effect Decomposition:
    Total Effect (c):     {self.total_effect:.4f} (SE={self.total_effect_se:.4f}, p={self.total_effect_pvalue:.4f})
    Direct Effect (c'):   {self.direct_effect:.4f} (SE={self.direct_effect_se:.4f}, p={self.direct_effect_pvalue:.4f})
    Indirect Effect (ab): {self.indirect_effect:.4f} (SE={self.indirect_effect_se:.4f}, p={self.indirect_effect_pvalue:.4f})

Sobel Test:
    Z-statistic: {self.sobel_z:.4f}
    P-value:     {self.sobel_pvalue:.4f}

Proportion Mediated: {self.proportion_mediated:.2%}

Interpretation: {mediation_type}
"""

    def to_dict(self) -> Dict:
        """Convert results to dictionary"""
        return {
            'total_effect': self.total_effect,
            'total_effect_se': self.total_effect_se,
            'total_effect_pvalue': self.total_effect_pvalue,
            'direct_effect': self.direct_effect,
            'direct_effect_se': self.direct_effect_se,
            'direct_effect_pvalue': self.direct_effect_pvalue,
            'indirect_effect': self.indirect_effect,
            'indirect_effect_se': self.indirect_effect_se,
            'indirect_effect_pvalue': self.indirect_effect_pvalue,
            'a_path': self.a_path,
            'a_path_se': self.a_path_se,
            'a_path_pvalue': self.a_path_pvalue,
            'b_path': self.b_path,
            'b_path_se': self.b_path_se,
            'b_path_pvalue': self.b_path_pvalue,
            'proportion_mediated': self.proportion_mediated,
            'sobel_z': self.sobel_z,
            'sobel_pvalue': self.sobel_pvalue
        }


def _ols_regression(X_design: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Simple OLS regression.

    Returns:
        coefficients, standard errors, p-values, mse
    """
    n, k = X_design.shape

    # Solve normal equations
    try:
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.zeros(k), np.full(k, np.inf), np.ones(k), np.inf

    # Residuals and MSE
    residuals = y - X_design @ beta
    mse = np.sum(residuals ** 2) / (n - k)

    # Standard errors
    try:
        var_beta = mse * np.linalg.inv(X_design.T @ X_design)
        se = np.sqrt(np.diag(var_beta))
    except np.linalg.LinAlgError:
        se = np.full(k, np.inf)

    # t-statistics and p-values
    t_stats = beta / (se + 1e-10)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    return beta, se, p_values, mse


def baron_kenny_mediation(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    covariates: Optional[np.ndarray] = None
) -> MediationResults:
    """
    Perform Baron-Kenny mediation analysis.

    Parameters:
    -----------
    X : Treatment variable (n,)
    M : Mediator variable (n,)
    Y : Outcome variable (n,)
    covariates : Optional control variables (n, k)

    Returns:
    --------
    MediationResults object
    """
    X = np.asarray(X).flatten()
    M = np.asarray(M).flatten()
    Y = np.asarray(Y).flatten()

    n = len(X)

    # Prepare design matrices
    if covariates is not None:
        covariates = np.asarray(covariates)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        X_with_const = np.column_stack([np.ones(n), X, covariates])
        XM_with_const = np.column_stack([np.ones(n), X, M, covariates])
    else:
        X_with_const = np.column_stack([np.ones(n), X])
        XM_with_const = np.column_stack([np.ones(n), X, M])

    # Step 1: Total effect (c) - Y ~ X
    beta_total, se_total, pval_total, _ = _ols_regression(X_with_const, Y)
    c = beta_total[1]
    c_se = se_total[1]
    c_pvalue = pval_total[1]

    # Step 2: a path - M ~ X
    beta_a, se_a, pval_a, _ = _ols_regression(X_with_const, M)
    a = beta_a[1]
    a_se = se_a[1]
    a_pvalue = pval_a[1]

    # Step 3: Direct effect (c') and b path - Y ~ X + M
    beta_direct, se_direct, pval_direct, _ = _ols_regression(XM_with_const, Y)
    c_prime = beta_direct[1]
    c_prime_se = se_direct[1]
    c_prime_pvalue = pval_direct[1]

    b = beta_direct[2]
    b_se = se_direct[2]
    b_pvalue = pval_direct[2]

    # Indirect effect
    indirect = a * b

    # Sobel test for indirect effect
    sobel_se = np.sqrt(b**2 * a_se**2 + a**2 * b_se**2)
    sobel_z = indirect / (sobel_se + 1e-10)
    sobel_pvalue = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    # Proportion mediated
    if abs(c) > 1e-10:
        prop_mediated = indirect / c
    else:
        prop_mediated = np.nan

    return MediationResults(
        total_effect=c,
        total_effect_se=c_se,
        total_effect_pvalue=c_pvalue,
        direct_effect=c_prime,
        direct_effect_se=c_prime_se,
        direct_effect_pvalue=c_prime_pvalue,
        indirect_effect=indirect,
        indirect_effect_se=sobel_se,
        indirect_effect_pvalue=sobel_pvalue,
        a_path=a,
        a_path_se=a_se,
        a_path_pvalue=a_pvalue,
        b_path=b,
        b_path_se=b_se,
        b_path_pvalue=b_pvalue,
        proportion_mediated=prop_mediated,
        sobel_z=sobel_z,
        sobel_pvalue=sobel_pvalue
    )


def bootstrap_mediation(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    covariates: Optional[np.ndarray] = None,
    random_state: Optional[int] = None
) -> Dict:
    """
    Bootstrap confidence intervals for mediation effects.
    More reliable than Sobel test, especially for small samples.

    Parameters:
    -----------
    X : Treatment variable
    M : Mediator variable
    Y : Outcome variable
    n_bootstrap : Number of bootstrap samples
    confidence_level : Confidence level for intervals
    covariates : Optional control variables
    random_state : Random seed for reproducibility

    Returns:
    --------
    Dictionary with bootstrap results
    """
    X = np.asarray(X).flatten()
    M = np.asarray(M).flatten()
    Y = np.asarray(Y).flatten()

    n = len(X)

    if random_state is not None:
        np.random.seed(random_state)

    indirect_effects = np.zeros(n_bootstrap)
    direct_effects = np.zeros(n_bootstrap)
    total_effects = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        M_boot = M[idx]
        Y_boot = Y[idx]
        cov_boot = covariates[idx] if covariates is not None else None

        # Fit models
        if cov_boot is not None:
            X_design = np.column_stack([np.ones(n), X_boot, cov_boot])
            XM_design = np.column_stack([np.ones(n), X_boot, M_boot, cov_boot])
        else:
            X_design = np.column_stack([np.ones(n), X_boot])
            XM_design = np.column_stack([np.ones(n), X_boot, M_boot])

        try:
            # Total effect
            beta_total = np.linalg.lstsq(X_design, Y_boot, rcond=None)[0]
            total_effects[i] = beta_total[1]

            # a path
            beta_a = np.linalg.lstsq(X_design, M_boot, rcond=None)[0]
            a = beta_a[1]

            # b and c' paths
            beta_direct = np.linalg.lstsq(XM_design, Y_boot, rcond=None)[0]
            b = beta_direct[2]
            c_prime = beta_direct[1]

            indirect_effects[i] = a * b
            direct_effects[i] = c_prime
        except:
            indirect_effects[i] = np.nan
            direct_effects[i] = np.nan
            total_effects[i] = np.nan

    # Remove failed bootstraps
    valid = ~np.isnan(indirect_effects)
    indirect_effects = indirect_effects[valid]
    direct_effects = direct_effects[valid]
    total_effects = total_effects[valid]

    alpha = 1 - confidence_level

    # Calculate proportion mediated
    prop_mediated = indirect_effects / (total_effects + 1e-10)

    return {
        'indirect_mean': np.mean(indirect_effects),
        'indirect_se': np.std(indirect_effects),
        'indirect_ci_lower': np.percentile(indirect_effects, 100 * alpha / 2),
        'indirect_ci_upper': np.percentile(indirect_effects, 100 * (1 - alpha / 2)),
        'indirect_significant': (
            np.percentile(indirect_effects, 100 * alpha / 2) > 0 or
            np.percentile(indirect_effects, 100 * (1 - alpha / 2)) < 0
        ),
        'direct_mean': np.mean(direct_effects),
        'direct_se': np.std(direct_effects),
        'direct_ci_lower': np.percentile(direct_effects, 100 * alpha / 2),
        'direct_ci_upper': np.percentile(direct_effects, 100 * (1 - alpha / 2)),
        'total_mean': np.mean(total_effects),
        'total_se': np.std(total_effects),
        'total_ci_lower': np.percentile(total_effects, 100 * alpha / 2),
        'total_ci_upper': np.percentile(total_effects, 100 * (1 - alpha / 2)),
        'proportion_mediated_mean': np.mean(prop_mediated),
        'proportion_mediated_ci_lower': np.percentile(prop_mediated, 100 * alpha / 2),
        'proportion_mediated_ci_upper': np.percentile(prop_mediated, 100 * (1 - alpha / 2)),
        'n_valid_bootstraps': len(indirect_effects)
    }


class MediationAnalyzer:
    """
    Rolling mediation analyzer for streaming financial data.

    Useful for real-time trading applications where mediation
    parameters need to be updated with each new observation.
    """

    def __init__(
        self,
        lookback_window: int = 60,
        min_observations: int = 30
    ):
        """
        Initialize analyzer.

        Parameters:
        -----------
        lookback_window : Window size for rolling estimation
        min_observations : Minimum observations before analysis
        """
        self.lookback = lookback_window
        self.min_obs = min_observations

        # History buffers
        self.X_history = []
        self.M_history = []
        self.Y_history = []

        # Current parameter estimates
        self._results = None

    def update(self, X: float, M: float, Y: float) -> Optional[MediationResults]:
        """
        Update with new observation and optionally return results.

        Parameters:
        -----------
        X : Treatment value
        M : Mediator value
        Y : Outcome value

        Returns:
        --------
        MediationResults if enough data, None otherwise
        """
        # Update history
        self.X_history.append(X)
        self.M_history.append(M)
        self.Y_history.append(Y)

        # Keep only lookback window
        if len(self.X_history) > self.lookback:
            self.X_history = self.X_history[-self.lookback:]
            self.M_history = self.M_history[-self.lookback:]
            self.Y_history = self.Y_history[-self.lookback:]

        # Run analysis if enough data
        if len(self.X_history) >= self.min_obs:
            self._results = baron_kenny_mediation(
                np.array(self.X_history),
                np.array(self.M_history),
                np.array(self.Y_history)
            )
            return self._results

        return None

    def get_current_results(self) -> Optional[MediationResults]:
        """Get most recent analysis results"""
        return self._results

    def reset(self):
        """Clear all history"""
        self.X_history = []
        self.M_history = []
        self.Y_history = []
        self._results = None


class CausalMediationAnalysis:
    """
    Causal mediation analysis with sensitivity analysis.

    Based on Imai, Keele, & Tingley (2010) framework.
    """

    def __init__(
        self,
        X: np.ndarray,
        M: np.ndarray,
        Y: np.ndarray,
        covariates: Optional[np.ndarray] = None,
        treatment_m_interaction: bool = False
    ):
        """
        Initialize causal mediation analysis.

        Parameters:
        -----------
        X : Treatment variable (n,)
        M : Mediator variable (n,)
        Y : Outcome variable (n,)
        covariates : Optional control variables (n, k)
        treatment_m_interaction : Whether to include X*M interaction
        """
        self.X = np.asarray(X).flatten()
        self.M = np.asarray(M).flatten()
        self.Y = np.asarray(Y).flatten()
        self.covariates = covariates
        self.n = len(self.X)
        self.interaction = treatment_m_interaction

        # Fit models
        self._fit_models()

    def _fit_models(self):
        """Fit mediator and outcome models"""
        n = self.n

        # Mediator model: M ~ X + C
        if self.covariates is not None:
            design_m = np.column_stack([np.ones(n), self.X, self.covariates])
        else:
            design_m = np.column_stack([np.ones(n), self.X])

        self.beta_m, self.se_m, _, self.mse_m = _ols_regression(design_m, self.M)

        # Outcome model: Y ~ X + M + (X*M) + C
        if self.interaction:
            XM_interaction = self.X * self.M
            if self.covariates is not None:
                design_y = np.column_stack([
                    np.ones(n), self.X, self.M, XM_interaction, self.covariates
                ])
            else:
                design_y = np.column_stack([
                    np.ones(n), self.X, self.M, XM_interaction
                ])
        else:
            if self.covariates is not None:
                design_y = np.column_stack([
                    np.ones(n), self.X, self.M, self.covariates
                ])
            else:
                design_y = np.column_stack([np.ones(n), self.X, self.M])

        self.beta_y, self.se_y, _, self.mse_y = _ols_regression(design_y, self.Y)

    def estimate_effects(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None
    ) -> Dict:
        """
        Estimate natural direct and indirect effects using simulation.

        Parameters:
        -----------
        n_simulations : Number of Monte Carlo simulations
        confidence_level : Confidence level for intervals
        random_state : Random seed

        Returns:
        --------
        Dictionary with effect estimates and confidence intervals
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Storage for effects
        acme = np.zeros(n_simulations)  # Average Causal Mediation Effect
        ade = np.zeros(n_simulations)   # Average Direct Effect
        total = np.zeros(n_simulations)

        sigma_m = np.sqrt(self.mse_m)
        sigma_y = np.sqrt(self.mse_y)

        for i in range(n_simulations):
            effects_indirect = []
            effects_direct = []

            # Subsample for speed
            sample_size = min(100, self.n)
            sample_idx = np.random.choice(self.n, size=sample_size, replace=False)

            for j in sample_idx:
                # Get covariates for this observation
                if self.covariates is not None:
                    c_j = self.covariates[j]

                # Potential mediators
                if self.covariates is not None:
                    M_0 = self.beta_m[0] + self.beta_m[1] * 0 + np.dot(c_j, self.beta_m[2:])
                    M_1 = self.beta_m[0] + self.beta_m[1] * 1 + np.dot(c_j, self.beta_m[2:])
                else:
                    M_0 = self.beta_m[0] + self.beta_m[1] * 0
                    M_1 = self.beta_m[0] + self.beta_m[1] * 1

                # Add noise
                M_0 += np.random.randn() * sigma_m
                M_1 += np.random.randn() * sigma_m

                # Potential outcomes
                if self.interaction:
                    if self.covariates is not None:
                        Y_00 = self.beta_y[0] + self.beta_y[1]*0 + self.beta_y[2]*M_0 + self.beta_y[3]*0*M_0 + np.dot(c_j, self.beta_y[4:])
                        Y_01 = self.beta_y[0] + self.beta_y[1]*0 + self.beta_y[2]*M_1 + self.beta_y[3]*0*M_1 + np.dot(c_j, self.beta_y[4:])
                        Y_10 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_0 + self.beta_y[3]*1*M_0 + np.dot(c_j, self.beta_y[4:])
                        Y_11 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_1 + self.beta_y[3]*1*M_1 + np.dot(c_j, self.beta_y[4:])
                    else:
                        Y_00 = self.beta_y[0] + self.beta_y[1]*0 + self.beta_y[2]*M_0 + self.beta_y[3]*0*M_0
                        Y_01 = self.beta_y[0] + self.beta_y[1]*0 + self.beta_y[2]*M_1 + self.beta_y[3]*0*M_1
                        Y_10 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_0 + self.beta_y[3]*1*M_0
                        Y_11 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_1 + self.beta_y[3]*1*M_1
                else:
                    if self.covariates is not None:
                        Y_00 = self.beta_y[0] + self.beta_y[1]*0 + self.beta_y[2]*M_0 + np.dot(c_j, self.beta_y[3:])
                        Y_01 = self.beta_y[0] + self.beta_y[1]*0 + self.beta_y[2]*M_1 + np.dot(c_j, self.beta_y[3:])
                        Y_10 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_0 + np.dot(c_j, self.beta_y[3:])
                        Y_11 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_1 + np.dot(c_j, self.beta_y[3:])
                    else:
                        Y_00 = self.beta_y[0] + self.beta_y[1]*0 + self.beta_y[2]*M_0
                        Y_01 = self.beta_y[0] + self.beta_y[1]*0 + self.beta_y[2]*M_1
                        Y_10 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_0
                        Y_11 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_1

                # NIE(1) = Y(1,M(1)) - Y(1,M(0))
                # NDE(0) = Y(1,M(0)) - Y(0,M(0))
                effects_indirect.append(Y_11 - Y_10)
                effects_direct.append(Y_10 - Y_00)

            acme[i] = np.mean(effects_indirect)
            ade[i] = np.mean(effects_direct)
            total[i] = acme[i] + ade[i]

        # Compute summary statistics
        alpha = 1 - confidence_level

        def compute_pvalue(samples):
            """Compute two-sided p-value"""
            mean_sign = np.sign(np.mean(samples))
            p = np.mean(np.sign(samples) != mean_sign) * 2
            return min(p, 1.0)

        # Proportion mediated
        prop_mediated = acme / (total + 1e-10)

        results = {
            'ACME': {
                'estimate': np.mean(acme),
                'se': np.std(acme),
                'ci_lower': np.percentile(acme, 100 * alpha / 2),
                'ci_upper': np.percentile(acme, 100 * (1 - alpha / 2)),
                'pvalue': compute_pvalue(acme)
            },
            'ADE': {
                'estimate': np.mean(ade),
                'se': np.std(ade),
                'ci_lower': np.percentile(ade, 100 * alpha / 2),
                'ci_upper': np.percentile(ade, 100 * (1 - alpha / 2)),
                'pvalue': compute_pvalue(ade)
            },
            'Total_Effect': {
                'estimate': np.mean(total),
                'se': np.std(total),
                'ci_lower': np.percentile(total, 100 * alpha / 2),
                'ci_upper': np.percentile(total, 100 * (1 - alpha / 2)),
            },
            'Proportion_Mediated': {
                'estimate': np.mean(prop_mediated),
                'ci_lower': np.percentile(prop_mediated, 100 * alpha / 2),
                'ci_upper': np.percentile(prop_mediated, 100 * (1 - alpha / 2)),
            }
        }

        return results

    def sensitivity_analysis(
        self,
        rho_range: np.ndarray = None,
        n_simulations: int = 500,
        random_state: Optional[int] = None
    ) -> Dict:
        """
        Sensitivity analysis for unmeasured confounding.

        Assesses how results change if there's correlation between
        mediator and outcome residuals (violation of sequential ignorability).

        Parameters:
        -----------
        rho_range : Correlation values to test
        n_simulations : Simulations per rho value
        random_state : Random seed

        Returns:
        --------
        Dictionary with sensitivity analysis results
        """
        if rho_range is None:
            rho_range = np.linspace(-0.9, 0.9, 19)

        if random_state is not None:
            np.random.seed(random_state)

        results = {
            'rho': rho_range,
            'acme': np.zeros(len(rho_range)),
            'acme_ci_lower': np.zeros(len(rho_range)),
            'acme_ci_upper': np.zeros(len(rho_range))
        }

        sigma_m = np.sqrt(self.mse_m)
        sigma_y = np.sqrt(self.mse_y)

        for i, rho in enumerate(rho_range):
            acme_samples = np.zeros(n_simulations)

            # Correlation matrix for errors
            cov_matrix = np.array([
                [sigma_m**2, rho * sigma_m * sigma_y],
                [rho * sigma_m * sigma_y, sigma_y**2]
            ])

            for s in range(n_simulations):
                effects = []

                for j in range(min(50, self.n)):
                    # Generate correlated errors
                    try:
                        eps_m, eps_y = np.random.multivariate_normal([0, 0], cov_matrix)
                    except:
                        eps_m, eps_y = 0, 0

                    if self.covariates is not None:
                        c_j = self.covariates[j]
                        M_0 = self.beta_m[0] + self.beta_m[1]*0 + np.dot(c_j, self.beta_m[2:]) + eps_m
                        M_1 = self.beta_m[0] + self.beta_m[1]*1 + np.dot(c_j, self.beta_m[2:]) + eps_m
                    else:
                        M_0 = self.beta_m[0] + self.beta_m[1]*0 + eps_m
                        M_1 = self.beta_m[0] + self.beta_m[1]*1 + eps_m

                    if self.interaction:
                        if self.covariates is not None:
                            Y_11 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_1 + self.beta_y[3]*1*M_1 + np.dot(c_j, self.beta_y[4:]) + eps_y
                            Y_10 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_0 + self.beta_y[3]*1*M_0 + np.dot(c_j, self.beta_y[4:]) + eps_y
                        else:
                            Y_11 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_1 + self.beta_y[3]*1*M_1 + eps_y
                            Y_10 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_0 + self.beta_y[3]*1*M_0 + eps_y
                    else:
                        if self.covariates is not None:
                            Y_11 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_1 + np.dot(c_j, self.beta_y[3:]) + eps_y
                            Y_10 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_0 + np.dot(c_j, self.beta_y[3:]) + eps_y
                        else:
                            Y_11 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_1 + eps_y
                            Y_10 = self.beta_y[0] + self.beta_y[1]*1 + self.beta_y[2]*M_0 + eps_y

                    effects.append(Y_11 - Y_10)

                acme_samples[s] = np.mean(effects)

            results['acme'][i] = np.mean(acme_samples)
            results['acme_ci_lower'][i] = np.percentile(acme_samples, 2.5)
            results['acme_ci_upper'][i] = np.percentile(acme_samples, 97.5)

        # Find breakdown point
        signs = np.sign(results['acme'])
        sign_changes = np.where(np.diff(signs) != 0)[0]

        if len(sign_changes) > 0:
            results['breakdown_rho'] = rho_range[sign_changes[0]]
        else:
            results['breakdown_rho'] = None

        return results
