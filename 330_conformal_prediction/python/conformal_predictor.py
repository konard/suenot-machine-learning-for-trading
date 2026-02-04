"""
Conformal Prediction Implementation

Provides split conformal prediction, conformalized quantile regression,
and adaptive conformal inference for time series.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Callable
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin


@dataclass
class PredictionInterval:
    """Represents a prediction interval."""
    point_prediction: float
    lower_bound: float
    upper_bound: float
    width: float
    alpha: float

    @property
    def confidence(self) -> float:
        """Return confidence level (1 - alpha)."""
        return 1.0 - self.alpha


class SplitConformalPredictor:
    """
    Split (Inductive) Conformal Prediction.

    Uses a held-out calibration set to compute non-conformity scores
    and construct prediction intervals.
    """

    def __init__(
        self,
        base_model: BaseEstimator,
        alpha: float = 0.1,
        score_function: str = 'absolute'
    ):
        """
        Initialize split conformal predictor.

        Args:
            base_model: Underlying prediction model
            alpha: Miscoverage rate (default 0.1 for 90% coverage)
            score_function: Type of non-conformity score
                           ('absolute', 'normalized', 'signed')
        """
        self.base_model = base_model
        self.alpha = alpha
        self.score_function = score_function
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile: Optional[float] = None
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'SplitConformalPredictor':
        """
        Fit the base model on training data.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            self
        """
        self.base_model.fit(X_train, y_train)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> 'SplitConformalPredictor':
        """
        Calibrate conformal predictor using held-out calibration data.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets

        Returns:
            self
        """
        # Get predictions on calibration set
        y_pred_cal = self.base_model.predict(X_cal)

        # Compute non-conformity scores
        self.calibration_scores = self._compute_scores(y_cal, y_pred_cal)

        # Compute quantile for coverage guarantee
        n_cal = len(self.calibration_scores)
        quantile_level = np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal
        quantile_level = min(quantile_level, 1.0)

        self.quantile = np.quantile(self.calibration_scores, quantile_level)
        self._fitted = True

        return self

    def fit_calibrate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> 'SplitConformalPredictor':
        """
        Fit model and calibrate in one step.

        Args:
            X_train: Training features
            y_train: Training targets
            X_cal: Calibration features
            y_cal: Calibration targets

        Returns:
            self
        """
        self.fit(X_train, y_train)
        self.calibrate(X_cal, y_cal)
        return self

    def predict_interval(self, X: np.ndarray) -> List[PredictionInterval]:
        """
        Generate prediction intervals for new data.

        Args:
            X: Features for prediction

        Returns:
            List of PredictionInterval objects
        """
        if not self._fitted:
            raise ValueError("Predictor must be calibrated before making predictions")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        y_pred = self.base_model.predict(X)

        intervals = []
        for pred in y_pred:
            lower = pred - self.quantile
            upper = pred + self.quantile
            width = upper - lower

            intervals.append(PredictionInterval(
                point_prediction=pred,
                lower_bound=lower,
                upper_bound=upper,
                width=width,
                alpha=self.alpha
            ))

        return intervals

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with intervals (array format).

        Args:
            X: Features for prediction

        Returns:
            Tuple of (point_predictions, lower_bounds, upper_bounds)
        """
        intervals = self.predict_interval(X)

        point_preds = np.array([i.point_prediction for i in intervals])
        lower_bounds = np.array([i.lower_bound for i in intervals])
        upper_bounds = np.array([i.upper_bound for i in intervals])

        return point_preds, lower_bounds, upper_bounds

    def _compute_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute non-conformity scores."""
        if self.score_function == 'absolute':
            return np.abs(y_true - y_pred)
        elif self.score_function == 'signed':
            return y_true - y_pred
        elif self.score_function == 'normalized':
            # Normalize by local variance estimate
            residuals = y_true - y_pred
            mad = np.median(np.abs(residuals - np.median(residuals)))
            return np.abs(residuals) / (mad + 1e-8)
        else:
            return np.abs(y_true - y_pred)

    def evaluate_coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate coverage and interval width on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Tuple of (coverage_rate, average_interval_width)
        """
        point_preds, lower_bounds, upper_bounds = self.predict(X_test)

        # Coverage: fraction of true values in intervals
        covered = (y_test >= lower_bounds) & (y_test <= upper_bounds)
        coverage = np.mean(covered)

        # Average interval width
        avg_width = np.mean(upper_bounds - lower_bounds)

        return coverage, avg_width


class ConformalQuantileRegressor:
    """
    Conformalized Quantile Regression (CQR).

    Combines quantile regression with conformal calibration for
    potentially asymmetric and heteroscedastic prediction intervals.
    """

    def __init__(
        self,
        quantile_model_low: BaseEstimator,
        quantile_model_high: BaseEstimator,
        alpha: float = 0.1
    ):
        """
        Initialize CQR predictor.

        Args:
            quantile_model_low: Model predicting lower quantile (alpha/2)
            quantile_model_high: Model predicting upper quantile (1 - alpha/2)
            alpha: Miscoverage rate
        """
        self.quantile_model_low = quantile_model_low
        self.quantile_model_high = quantile_model_high
        self.alpha = alpha
        self.quantile_adjustment: Optional[float] = None
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'ConformalQuantileRegressor':
        """Fit both quantile models."""
        self.quantile_model_low.fit(X_train, y_train)
        self.quantile_model_high.fit(X_train, y_train)
        return self

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> 'ConformalQuantileRegressor':
        """
        Calibrate using CQR score function.

        Score = max(q_low(x) - y, y - q_high(x))
        """
        q_low = self.quantile_model_low.predict(X_cal)
        q_high = self.quantile_model_high.predict(X_cal)

        # CQR non-conformity scores
        scores = np.maximum(q_low - y_cal, y_cal - q_high)

        # Compute quantile for adjustment
        n_cal = len(scores)
        quantile_level = np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal
        quantile_level = min(quantile_level, 1.0)

        self.quantile_adjustment = np.quantile(scores, quantile_level)
        self._fitted = True

        return self

    def fit_calibrate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> 'ConformalQuantileRegressor':
        """Fit and calibrate in one step."""
        self.fit(X_train, y_train)
        self.calibrate(X_cal, y_cal)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with calibrated intervals.

        Returns:
            Tuple of (point_predictions, lower_bounds, upper_bounds)
        """
        if not self._fitted:
            raise ValueError("Predictor must be calibrated before making predictions")

        q_low = self.quantile_model_low.predict(X)
        q_high = self.quantile_model_high.predict(X)

        # Apply conformal adjustment
        lower_bounds = q_low - self.quantile_adjustment
        upper_bounds = q_high + self.quantile_adjustment

        # Point prediction as midpoint
        point_preds = (lower_bounds + upper_bounds) / 2

        return point_preds, lower_bounds, upper_bounds


class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Inference (ACI) for time series.

    Adjusts the miscoverage level alpha online to maintain coverage
    under distribution shift.
    """

    def __init__(
        self,
        base_predictor: SplitConformalPredictor,
        target_alpha: float = 0.1,
        gamma: float = 0.01,
        window_size: int = 100
    ):
        """
        Initialize adaptive conformal predictor.

        Args:
            base_predictor: Underlying conformal predictor
            target_alpha: Target miscoverage rate
            gamma: Learning rate for alpha updates
            window_size: Window size for rolling calibration
        """
        self.base_predictor = base_predictor
        self.target_alpha = target_alpha
        self.alpha_t = target_alpha  # Time-varying alpha
        self.gamma = gamma
        self.window_size = window_size

        self.coverage_history: List[bool] = []
        self.alpha_history: List[float] = []

    def update(self, y_true: float, lower: float, upper: float) -> None:
        """
        Update alpha based on observed coverage.

        Args:
            y_true: True observed value
            lower: Predicted lower bound
            upper: Predicted upper bound
        """
        covered = lower <= y_true <= upper
        self.coverage_history.append(covered)

        # Update alpha: increase if not covered, decrease if covered
        error = (1 - int(covered)) - self.target_alpha
        self.alpha_t = np.clip(
            self.alpha_t + self.gamma * error,
            0.01, 0.5
        )

        self.alpha_history.append(self.alpha_t)

    def get_adjusted_quantile(self) -> float:
        """Get current quantile level (1 - alpha_t)."""
        return 1 - self.alpha_t

    def rolling_calibrate(
        self,
        X_history: np.ndarray,
        y_history: np.ndarray
    ) -> None:
        """
        Recalibrate using recent history.

        Args:
            X_history: Recent feature history
            y_history: Recent target history
        """
        if len(X_history) < self.window_size:
            return

        # Use most recent window for calibration
        X_cal = X_history[-self.window_size:]
        y_cal = y_history[-self.window_size:]

        # Temporarily adjust alpha
        original_alpha = self.base_predictor.alpha
        self.base_predictor.alpha = self.alpha_t

        # Recalibrate
        self.base_predictor.calibrate(X_cal, y_cal)

        # Restore alpha
        self.base_predictor.alpha = original_alpha

    def get_rolling_coverage(self, window: int = 50) -> float:
        """Get recent rolling coverage rate."""
        if len(self.coverage_history) < window:
            return np.mean(self.coverage_history) if self.coverage_history else 0.5
        return np.mean(self.coverage_history[-window:])


def compute_coverage_metrics(
    y_true: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    target_alpha: float = 0.1
) -> dict:
    """
    Compute comprehensive coverage metrics.

    Args:
        y_true: True values
        lower_bounds: Lower interval bounds
        upper_bounds: Upper interval bounds
        target_alpha: Target miscoverage rate

    Returns:
        Dictionary with coverage metrics
    """
    covered = (y_true >= lower_bounds) & (y_true <= upper_bounds)

    coverage = np.mean(covered)
    target_coverage = 1 - target_alpha
    coverage_gap = abs(coverage - target_coverage)

    widths = upper_bounds - lower_bounds

    return {
        'coverage': coverage,
        'target_coverage': target_coverage,
        'coverage_gap': coverage_gap,
        'mean_width': np.mean(widths),
        'median_width': np.median(widths),
        'std_width': np.std(widths),
        'min_width': np.min(widths),
        'max_width': np.max(widths),
        'fraction_covered': np.sum(covered),
        'fraction_not_covered': np.sum(~covered),
        'total_samples': len(y_true)
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import QuantileRegressor

    np.random.seed(42)

    # Generate synthetic data
    n = 1000
    X = np.random.randn(n, 5)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.5

    # Split data
    train_end = int(0.6 * n)
    cal_end = int(0.8 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_cal, y_cal = X[train_end:cal_end], y[train_end:cal_end]
    X_test, y_test = X[cal_end:], y[cal_end:]

    # Test Split Conformal
    print("=" * 50)
    print("Split Conformal Prediction")
    print("=" * 50)

    base_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
    cp = SplitConformalPredictor(base_model, alpha=0.1)
    cp.fit_calibrate(X_train, y_train, X_cal, y_cal)

    coverage, avg_width = cp.evaluate_coverage(X_test, y_test)
    print(f"Coverage: {coverage:.3f} (target: 0.90)")
    print(f"Average interval width: {avg_width:.3f}")

    # Show some predictions
    intervals = cp.predict_interval(X_test[:5])
    print("\nSample predictions:")
    for i, (interval, true_val) in enumerate(zip(intervals, y_test[:5])):
        covered = interval.lower_bound <= true_val <= interval.upper_bound
        print(f"  {i+1}. Pred: {interval.point_prediction:.3f}, "
              f"Interval: [{interval.lower_bound:.3f}, {interval.upper_bound:.3f}], "
              f"True: {true_val:.3f}, Covered: {covered}")
