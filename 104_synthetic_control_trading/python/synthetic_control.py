"""
Synthetic Control Method implementation for trading.

Provides the core SCM estimator with:
- Weight optimization using quadratic programming
- Treatment effect estimation
- Placebo test inference
- Visualization utilities
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SCMResults:
    """Results from Synthetic Control Method analysis."""
    weights: np.ndarray
    donor_labels: List[str]
    pre_treatment_rmse: float
    treatment_effects: np.ndarray
    cumulative_effect: float
    car: float  # Cumulative abnormal return
    pre_treatment_fit: np.ndarray
    post_treatment_synthetic: np.ndarray


@dataclass
class PlaceboResults:
    """Results from placebo tests."""
    placebo_effects: Dict[str, np.ndarray]
    placebo_rmse: Dict[str, float]
    p_value: float
    rank: int
    n_placebos: int


class SyntheticControlModel:
    """
    Synthetic Control Method model for causal inference in trading.

    This model constructs a synthetic version of a treated unit (e.g., a stock
    affected by an event) using a weighted combination of donor units (similar
    stocks unaffected by the event).

    Example:
        >>> from python.synthetic_control import SyntheticControlModel
        >>> model = SyntheticControlModel()
        >>> model.fit(treated_pre, donors_pre)
        >>> effects = model.estimate_effects(treated_post, donors_post)
        >>> print(f"CAR: {effects['car']:.4f}")
    """

    def __init__(
        self,
        v_penalty: float = 0.0,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        verbose: bool = False,
    ):
        """
        Initialize Synthetic Control Model.

        Args:
            v_penalty: Regularization penalty for weight optimization
            max_weight: Maximum weight for any single donor
            min_weight: Minimum weight for any donor (usually 0)
            verbose: Whether to print optimization progress
        """
        self.v_penalty = v_penalty
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.verbose = verbose

        self.weights_: Optional[np.ndarray] = None
        self.donor_labels_: Optional[List[str]] = None
        self.pre_treatment_rmse_: Optional[float] = None
        self._treated_pre: Optional[np.ndarray] = None
        self._donors_pre: Optional[np.ndarray] = None

    def fit(
        self,
        treated: np.ndarray,
        donors: np.ndarray,
        donor_labels: Optional[List[str]] = None,
        predictors: Optional[List[str]] = None,
    ) -> "SyntheticControlModel":
        """
        Fit the synthetic control model to pre-treatment data.

        Finds optimal weights W* such that:
            Σⱼ wⱼ * donor_j ≈ treated

        Args:
            treated: Pre-treatment outcomes for treated unit (T_pre,)
            donors: Pre-treatment outcomes for donors (T_pre, J)
            donor_labels: Optional labels for donor units
            predictors: Optional list of predictor names (for documentation)

        Returns:
            Self for method chaining
        """
        treated = np.asarray(treated).flatten()
        donors = np.asarray(donors)

        if donors.ndim == 1:
            donors = donors.reshape(-1, 1)

        T_pre, J = donors.shape

        if len(treated) != T_pre:
            raise ValueError(
                f"Treated length ({len(treated)}) != donors rows ({T_pre})"
            )

        self._treated_pre = treated
        self._donors_pre = donors
        self.donor_labels_ = donor_labels or [f"Donor_{i}" for i in range(J)]

        # Optimize weights using constrained least squares
        self.weights_ = self._optimize_weights(treated, donors)

        # Calculate pre-treatment fit
        synthetic_pre = donors @ self.weights_
        self.pre_treatment_rmse_ = np.sqrt(np.mean((treated - synthetic_pre) ** 2))

        if self.verbose:
            logger.info(f"Pre-treatment RMSE: {self.pre_treatment_rmse_:.6f}")
            for label, weight in zip(self.donor_labels_, self.weights_):
                if weight > 0.01:
                    logger.info(f"  {label}: {weight:.4f}")

        return self

    def _optimize_weights(
        self,
        treated: np.ndarray,
        donors: np.ndarray,
    ) -> np.ndarray:
        """
        Optimize donor weights to minimize pre-treatment prediction error.

        Uses constrained optimization:
        - Weights sum to 1
        - Weights are non-negative (or within [min_weight, max_weight])

        Args:
            treated: Pre-treatment treated outcomes (T,)
            donors: Pre-treatment donor outcomes (T, J)

        Returns:
            Optimal weights (J,)
        """
        T, J = donors.shape

        # Try scipy optimization first, fall back to simple least squares
        try:
            from scipy.optimize import minimize

            def objective(w):
                synthetic = donors @ w
                loss = np.sum((treated - synthetic) ** 2)
                # Add L2 regularization
                loss += self.v_penalty * np.sum(w ** 2)
                return loss

            # Constraints: weights sum to 1
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            # Bounds: weights in [min_weight, max_weight]
            bounds = [(self.min_weight, self.max_weight) for _ in range(J)]

            # Initial guess: equal weights
            w0 = np.ones(J) / J

            result = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-10},
            )

            if result.success:
                weights = result.x
                # Ensure constraints are satisfied
                weights = np.maximum(weights, self.min_weight)
                weights = np.minimum(weights, self.max_weight)
                weights = weights / np.sum(weights)
                return weights
            else:
                logger.warning(f"Optimization failed: {result.message}")

        except ImportError:
            logger.warning("scipy not available, using simple least squares")

        # Fallback: Simple least squares with projection
        weights = self._simple_ols_weights(treated, donors)
        return weights

    def _simple_ols_weights(
        self,
        treated: np.ndarray,
        donors: np.ndarray,
    ) -> np.ndarray:
        """
        Simple OLS-based weight estimation with projection to constraints.

        Args:
            treated: Pre-treatment treated outcomes (T,)
            donors: Pre-treatment donor outcomes (T, J)

        Returns:
            Weights projected to simplex (J,)
        """
        # OLS solution
        try:
            weights = np.linalg.lstsq(donors, treated, rcond=None)[0]
        except np.linalg.LinAlgError:
            weights = np.ones(donors.shape[1]) / donors.shape[1]

        # Project to simplex (non-negative, sum to 1)
        weights = self._project_to_simplex(weights)
        return weights

    def _project_to_simplex(self, v: np.ndarray) -> np.ndarray:
        """
        Project vector onto probability simplex.

        Args:
            v: Input vector

        Returns:
            Projected vector (non-negative, sums to 1)
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(1, n + 1)
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        w = np.maximum(v - theta, 0)
        return w

    def estimate_effects(
        self,
        treated_post: np.ndarray,
        donors_post: np.ndarray,
    ) -> Dict:
        """
        Estimate treatment effects using fitted weights.

        Args:
            treated_post: Post-treatment outcomes for treated unit (T_post,)
            donors_post: Post-treatment outcomes for donors (T_post, J)

        Returns:
            Dictionary with:
            - effects: Point-by-point treatment effects
            - cumulative: Cumulative treatment effect
            - car: Cumulative abnormal return
            - synthetic_post: Synthetic control post-treatment values
        """
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before estimating effects")

        treated_post = np.asarray(treated_post).flatten()
        donors_post = np.asarray(donors_post)

        if donors_post.ndim == 1:
            donors_post = donors_post.reshape(-1, 1)

        # Construct synthetic control for post-treatment period
        synthetic_post = donors_post @ self.weights_

        # Treatment effects (gap)
        effects = treated_post - synthetic_post

        # Cumulative effect
        cumulative_effect = np.sum(effects)

        # CAR (for returns data, this is the cumulative abnormal return)
        car = cumulative_effect

        return {
            "effects": effects,
            "cumulative": cumulative_effect,
            "car": car,
            "synthetic_post": synthetic_post,
            "treated_post": treated_post,
        }

    def get_results(
        self,
        treated_post: np.ndarray,
        donors_post: np.ndarray,
    ) -> SCMResults:
        """
        Get comprehensive SCM results.

        Args:
            treated_post: Post-treatment outcomes for treated unit
            donors_post: Post-treatment outcomes for donors

        Returns:
            SCMResults dataclass with all analysis outputs
        """
        effects = self.estimate_effects(treated_post, donors_post)

        pre_fit = self._donors_pre @ self.weights_

        return SCMResults(
            weights=self.weights_,
            donor_labels=self.donor_labels_,
            pre_treatment_rmse=self.pre_treatment_rmse_,
            treatment_effects=effects["effects"],
            cumulative_effect=effects["cumulative"],
            car=effects["car"],
            pre_treatment_fit=pre_fit,
            post_treatment_synthetic=effects["synthetic_post"],
        )

    def weights(self) -> Dict[str, float]:
        """Get donor weights as a dictionary."""
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted first")
        return dict(zip(self.donor_labels_, self.weights_))

    def run_placebo_tests(
        self,
        treated_post: np.ndarray,
        donors_post: np.ndarray,
    ) -> PlaceboResults:
        """
        Run in-space placebo tests for inference.

        For each donor unit, pretend it was the treated unit and
        estimate a placebo treatment effect. The true treatment
        effect is significant if it is extreme relative to placebos.

        Args:
            treated_post: Post-treatment outcomes for treated unit
            donors_post: Post-treatment outcomes for donors

        Returns:
            PlaceboResults with p-value and rank
        """
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted first")

        treated_post = np.asarray(treated_post).flatten()
        donors_post = np.asarray(donors_post)

        if donors_post.ndim == 1:
            donors_post = donors_post.reshape(-1, 1)

        # Get actual treatment effect
        actual_effects = self.estimate_effects(treated_post, donors_post)
        actual_car = np.abs(actual_effects["car"])

        # Run placebos
        placebo_effects = {}
        placebo_rmse = {}
        placebo_cars = []

        J = self._donors_pre.shape[1]

        for j in range(J):
            # Donor j as treated, others as donors
            placebo_treated_pre = self._donors_pre[:, j]
            placebo_donors_pre = np.delete(self._donors_pre, j, axis=1)

            placebo_treated_post = donors_post[:, j]
            placebo_donors_post = np.delete(donors_post, j, axis=1)

            # Fit placebo model
            placebo_model = SyntheticControlModel(
                v_penalty=self.v_penalty,
                max_weight=self.max_weight,
                min_weight=self.min_weight,
            )
            placebo_model.fit(placebo_treated_pre, placebo_donors_pre)

            # Only include if pre-treatment fit is reasonable
            if placebo_model.pre_treatment_rmse_ < 2 * self.pre_treatment_rmse_:
                effects = placebo_model.estimate_effects(
                    placebo_treated_post, placebo_donors_post
                )
                label = self.donor_labels_[j]
                placebo_effects[label] = effects["effects"]
                placebo_rmse[label] = placebo_model.pre_treatment_rmse_
                placebo_cars.append(np.abs(effects["car"]))

        # Calculate p-value (rank-based)
        all_cars = placebo_cars + [actual_car]
        rank = sum(1 for x in all_cars if x >= actual_car)
        p_value = rank / len(all_cars)

        return PlaceboResults(
            placebo_effects=placebo_effects,
            placebo_rmse=placebo_rmse,
            p_value=p_value,
            rank=rank,
            n_placebos=len(placebo_cars),
        )


def run_scm_analysis(
    treated_pre: np.ndarray,
    donors_pre: np.ndarray,
    treated_post: np.ndarray,
    donors_post: np.ndarray,
    donor_labels: Optional[List[str]] = None,
    run_placebos: bool = True,
) -> Tuple[SCMResults, Optional[PlaceboResults]]:
    """
    Run complete SCM analysis.

    Convenience function that fits the model, estimates effects,
    and optionally runs placebo tests.

    Args:
        treated_pre: Pre-treatment treated outcomes
        donors_pre: Pre-treatment donor outcomes
        treated_post: Post-treatment treated outcomes
        donors_post: Post-treatment donor outcomes
        donor_labels: Optional donor labels
        run_placebos: Whether to run placebo tests

    Returns:
        Tuple of (SCMResults, PlaceboResults or None)
    """
    model = SyntheticControlModel(verbose=True)
    model.fit(treated_pre, donors_pre, donor_labels)

    results = model.get_results(treated_post, donors_post)

    placebo_results = None
    if run_placebos:
        placebo_results = model.run_placebo_tests(treated_post, donors_post)

    return results, placebo_results
