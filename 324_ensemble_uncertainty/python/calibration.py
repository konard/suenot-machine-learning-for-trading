"""
Model Calibration module.

This module provides methods for calibrating model predictions
to ensure that predicted uncertainties match actual error frequencies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Calibrator:
    """
    Calibrates model predictions and uncertainties.

    Supports:
    - Expected Calibration Error (ECE)
    - Reliability diagrams
    - Platt scaling
    - Isotonic regression
    - Temperature scaling
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize the calibrator.

        Args:
            n_bins: Number of bins for calibration metrics
        """
        self.n_bins = n_bins
        self.calibration_model = None
        self.temperature = 1.0

    def expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: Optional[int] = None
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE = sum_b |B_b|/n * |acc(b) - conf(b)|

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            n_bins: Number of bins (uses default if None)

        Returns:
            ECE value (lower is better)
        """
        if n_bins is None:
            n_bins = self.n_bins

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (y_pred_proba >= bin_boundaries[i]) & \
                   (y_pred_proba < bin_boundaries[i + 1])

            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_pred_proba[mask].mean()
                bin_size = mask.sum() / len(y_true)
                ece += bin_size * np.abs(bin_accuracy - bin_confidence)

        return ece

    def maximum_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: Optional[int] = None
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins

        Returns:
            MCE value (lower is better)
        """
        if n_bins is None:
            n_bins = self.n_bins

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            mask = (y_pred_proba >= bin_boundaries[i]) & \
                   (y_pred_proba < bin_boundaries[i + 1])

            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = y_pred_proba[mask].mean()
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        return mce

    def reliability_diagram_data(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute data for reliability diagram.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins

        Returns:
            DataFrame with bin_center, accuracy, confidence, count
        """
        if n_bins is None:
            n_bins = self.n_bins

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        data = []
        for i in range(n_bins):
            mask = (y_pred_proba >= bin_boundaries[i]) & \
                   (y_pred_proba < bin_boundaries[i + 1])

            if mask.sum() > 0:
                data.append({
                    'bin_center': bin_centers[i],
                    'accuracy': y_true[mask].mean(),
                    'confidence': y_pred_proba[mask].mean(),
                    'count': mask.sum(),
                })

        return pd.DataFrame(data)

    def regression_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        n_bins: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute calibration error for regression with uncertainty.

        Checks if actual coverage matches predicted confidence levels.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_std: Predicted standard deviations
            n_bins: Number of confidence levels to check

        Returns:
            Dictionary with calibration metrics
        """
        if n_bins is None:
            n_bins = self.n_bins

        # Check coverage at different confidence levels
        z_scores = np.abs(y_true - y_pred) / (y_std + 1e-8)

        calibration_errors = []
        coverage_data = []

        for confidence in np.linspace(0.1, 0.95, n_bins):
            z_threshold = stats.norm.ppf((1 + confidence) / 2)
            expected_coverage = confidence
            actual_coverage = np.mean(z_scores <= z_threshold)
            error = np.abs(expected_coverage - actual_coverage)
            calibration_errors.append(error)
            coverage_data.append({
                'confidence': confidence,
                'expected_coverage': expected_coverage,
                'actual_coverage': actual_coverage,
                'error': error,
            })

        return {
            'mean_calibration_error': np.mean(calibration_errors),
            'max_calibration_error': np.max(calibration_errors),
            'coverage_data': pd.DataFrame(coverage_data),
        }

    def fit_platt_scaling(
        self,
        y_true: np.ndarray,
        logits: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fit Platt scaling parameters.

        P(y=1|f) = 1 / (1 + exp(A*f + B))

        Args:
            y_true: True binary labels
            logits: Raw model outputs (logits)

        Returns:
            Tuple of (A, B) parameters
        """
        from scipy.optimize import minimize

        def negative_log_likelihood(params):
            A, B = params
            probs = 1 / (1 + np.exp(A * logits + B))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            return -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))

        result = minimize(negative_log_likelihood, [1.0, 0.0], method='L-BFGS-B')
        A, B = result.x
        logger.info(f"Platt scaling fitted: A={A:.4f}, B={B:.4f}")
        return A, B

    def apply_platt_scaling(
        self,
        logits: np.ndarray,
        A: float,
        B: float
    ) -> np.ndarray:
        """
        Apply Platt scaling to logits.

        Args:
            logits: Raw model outputs
            A, B: Platt scaling parameters

        Returns:
            Calibrated probabilities
        """
        return 1 / (1 + np.exp(A * logits + B))

    def fit_isotonic_regression(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> IsotonicRegression:
        """
        Fit isotonic regression for calibration.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities

        Returns:
            Fitted IsotonicRegression model
        """
        self.calibration_model = IsotonicRegression(out_of_bounds='clip')
        self.calibration_model.fit(y_pred_proba, y_true)
        logger.info("Isotonic regression fitted for calibration")
        return self.calibration_model

    def apply_isotonic_calibration(
        self,
        y_pred_proba: np.ndarray
    ) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            y_pred_proba: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if self.calibration_model is None:
            raise ValueError("Calibration model not fitted. Call fit_isotonic_regression first.")
        return self.calibration_model.predict(y_pred_proba)

    def fit_temperature_scaling(
        self,
        y_true: np.ndarray,
        logits: np.ndarray
    ) -> float:
        """
        Fit temperature for temperature scaling.

        Args:
            y_true: True labels (one-hot or integer)
            logits: Raw model outputs

        Returns:
            Optimal temperature
        """
        def nll_with_temperature(T):
            scaled_logits = logits / T
            probs = self._softmax(scaled_logits)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            if y_true.ndim == 1:
                # Integer labels
                log_probs = np.log(probs[np.arange(len(y_true)), y_true])
            else:
                # One-hot labels
                log_probs = np.sum(y_true * np.log(probs), axis=1)

            return -np.mean(log_probs)

        result = minimize_scalar(nll_with_temperature, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        logger.info(f"Temperature scaling fitted: T={self.temperature:.4f}")
        return self.temperature

    def apply_temperature_scaling(
        self,
        logits: np.ndarray,
        temperature: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply temperature scaling.

        Args:
            logits: Raw model outputs
            temperature: Temperature (uses fitted if None)

        Returns:
            Calibrated probabilities
        """
        if temperature is None:
            temperature = self.temperature
        return self._softmax(logits / temperature)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def calibrate_uncertainty(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray
    ) -> float:
        """
        Calibrate uncertainty estimates by finding scaling factor.

        Args:
            y_true: True values
            y_pred: Predictions
            y_std: Uncertainty estimates

        Returns:
            Scaling factor for uncertainties
        """
        def miscalibration(scale):
            scaled_std = y_std * scale
            z_scores = np.abs(y_true - y_pred) / (scaled_std + 1e-8)

            # Target: 68% within 1 std, 95% within 2 std
            actual_68 = np.mean(z_scores <= 1.0)
            actual_95 = np.mean(z_scores <= 2.0)

            error = (actual_68 - 0.68)**2 + (actual_95 - 0.95)**2
            return error

        result = minimize_scalar(miscalibration, bounds=(0.1, 10.0), method='bounded')
        scale_factor = result.x
        logger.info(f"Uncertainty calibration scale factor: {scale_factor:.4f}")
        return scale_factor


def main():
    """Example usage of calibration."""
    np.random.seed(42)

    print("=" * 60)
    print("Classification Calibration")
    print("=" * 60)

    # Simulate classification predictions
    n_samples = 1000
    y_true = np.random.binomial(1, 0.5, n_samples)

    # Simulate uncalibrated probabilities (overconfident)
    raw_probs = np.random.beta(2, 5, n_samples)  # Skewed towards low
    raw_probs = np.where(y_true == 1, 1 - raw_probs, raw_probs)

    calibrator = Calibrator(n_bins=10)

    # ECE before calibration
    ece_before = calibrator.expected_calibration_error(y_true, raw_probs)
    mce_before = calibrator.maximum_calibration_error(y_true, raw_probs)
    print(f"\nBefore calibration:")
    print(f"  ECE: {ece_before:.4f}")
    print(f"  MCE: {mce_before:.4f}")

    # Reliability diagram data
    reliability = calibrator.reliability_diagram_data(y_true, raw_probs)
    print(f"\nReliability diagram data:")
    print(reliability.to_string())

    # Apply isotonic calibration
    calibrator.fit_isotonic_regression(y_true, raw_probs)
    calibrated_probs = calibrator.apply_isotonic_calibration(raw_probs)

    # ECE after calibration
    ece_after = calibrator.expected_calibration_error(y_true, calibrated_probs)
    mce_after = calibrator.maximum_calibration_error(y_true, calibrated_probs)
    print(f"\nAfter isotonic calibration:")
    print(f"  ECE: {ece_after:.4f}")
    print(f"  MCE: {mce_after:.4f}")

    print("\n" + "=" * 60)
    print("Regression Calibration")
    print("=" * 60)

    # Simulate regression predictions
    y_true_reg = np.random.randn(n_samples)
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 0.5
    y_std_reg = np.abs(np.random.randn(n_samples) * 0.3) + 0.2  # Uncertain std estimates

    # Check calibration
    reg_calibration = calibrator.regression_calibration_error(
        y_true_reg, y_pred_reg, y_std_reg
    )
    print(f"\nRegression calibration before:")
    print(f"  Mean calibration error: {reg_calibration['mean_calibration_error']:.4f}")
    print(f"  Max calibration error: {reg_calibration['max_calibration_error']:.4f}")
    print(f"\nCoverage data:")
    print(reg_calibration['coverage_data'].to_string())

    # Calibrate uncertainty
    scale_factor = calibrator.calibrate_uncertainty(y_true_reg, y_pred_reg, y_std_reg)
    calibrated_std = y_std_reg * scale_factor

    # Check calibration after
    reg_calibration_after = calibrator.regression_calibration_error(
        y_true_reg, y_pred_reg, calibrated_std
    )
    print(f"\nRegression calibration after:")
    print(f"  Mean calibration error: {reg_calibration_after['mean_calibration_error']:.4f}")
    print(f"  Max calibration error: {reg_calibration_after['max_calibration_error']:.4f}")

    print("\n" + "=" * 60)
    print("Temperature Scaling")
    print("=" * 60)

    # Simulate multi-class logits
    n_classes = 5
    logits = np.random.randn(n_samples, n_classes)
    y_true_mc = np.random.randint(0, n_classes, n_samples)

    # Fit temperature
    temperature = calibrator.fit_temperature_scaling(y_true_mc, logits)
    print(f"\nFitted temperature: {temperature:.4f}")

    # Apply temperature scaling
    original_probs = calibrator._softmax(logits)
    calibrated_probs_mc = calibrator.apply_temperature_scaling(logits)

    print(f"Max confidence before: {original_probs.max(axis=1).mean():.4f}")
    print(f"Max confidence after: {calibrated_probs_mc.max(axis=1).mean():.4f}")


if __name__ == '__main__':
    main()
