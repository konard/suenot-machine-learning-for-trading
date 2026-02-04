//! Model calibration methods.

use ndarray::Array1;
use std::f64::consts::PI;

/// Calibrator for model predictions and uncertainties
pub struct Calibrator {
    n_bins: usize,
    temperature: f64,
    scale_factor: f64,
}

impl Calibrator {
    /// Create new calibrator
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins,
            temperature: 1.0,
            scale_factor: 1.0,
        }
    }

    /// Compute Expected Calibration Error (ECE) for classification
    pub fn expected_calibration_error(
        &self,
        y_true: &Array1<f64>,
        y_pred_proba: &Array1<f64>,
    ) -> f64 {
        let n_samples = y_true.len();
        let bin_boundaries: Vec<f64> = (0..=self.n_bins)
            .map(|i| i as f64 / self.n_bins as f64)
            .collect();

        let mut ece = 0.0;

        for i in 0..self.n_bins {
            let mask: Vec<bool> = y_pred_proba
                .iter()
                .map(|&p| p >= bin_boundaries[i] && p < bin_boundaries[i + 1])
                .collect();

            let count: usize = mask.iter().filter(|&&m| m).count();
            if count > 0 {
                let bin_accuracy: f64 = y_true
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&y, _)| y)
                    .sum::<f64>()
                    / count as f64;

                let bin_confidence: f64 = y_pred_proba
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&p, _)| p)
                    .sum::<f64>()
                    / count as f64;

                let bin_size = count as f64 / n_samples as f64;
                ece += bin_size * (bin_accuracy - bin_confidence).abs();
            }
        }

        ece
    }

    /// Compute Maximum Calibration Error (MCE)
    pub fn maximum_calibration_error(
        &self,
        y_true: &Array1<f64>,
        y_pred_proba: &Array1<f64>,
    ) -> f64 {
        let bin_boundaries: Vec<f64> = (0..=self.n_bins)
            .map(|i| i as f64 / self.n_bins as f64)
            .collect();

        let mut mce = 0.0;

        for i in 0..self.n_bins {
            let mask: Vec<bool> = y_pred_proba
                .iter()
                .map(|&p| p >= bin_boundaries[i] && p < bin_boundaries[i + 1])
                .collect();

            let count: usize = mask.iter().filter(|&&m| m).count();
            if count > 0 {
                let bin_accuracy: f64 = y_true
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&y, _)| y)
                    .sum::<f64>()
                    / count as f64;

                let bin_confidence: f64 = y_pred_proba
                    .iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&p, _)| p)
                    .sum::<f64>()
                    / count as f64;

                mce = mce.max((bin_accuracy - bin_confidence).abs());
            }
        }

        mce
    }

    /// Compute regression calibration error
    pub fn regression_calibration_error(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        y_std: &Array1<f64>,
    ) -> CalibrationResult {
        let n_samples = y_true.len();
        let confidence_levels: Vec<f64> = (1..=9).map(|i| i as f64 / 10.0).collect();

        let mut calibration_errors = Vec::new();
        let mut coverage_data = Vec::new();

        for &confidence in &confidence_levels {
            // Z-score for this confidence level
            let z_threshold = self.standard_normal_quantile((1.0 + confidence) / 2.0);

            // Count samples within interval
            let mut within_interval = 0;
            for i in 0..n_samples {
                let z_score = (y_true[i] - y_pred[i]).abs() / (y_std[i] + 1e-8);
                if z_score <= z_threshold {
                    within_interval += 1;
                }
            }

            let actual_coverage = within_interval as f64 / n_samples as f64;
            let error = (confidence - actual_coverage).abs();
            calibration_errors.push(error);

            coverage_data.push(CoveragePoint {
                confidence,
                expected_coverage: confidence,
                actual_coverage,
                error,
            });
        }

        let mean_error = calibration_errors.iter().sum::<f64>() / calibration_errors.len() as f64;
        let max_error = calibration_errors
            .iter()
            .cloned()
            .fold(0.0, f64::max);

        CalibrationResult {
            mean_calibration_error: mean_error,
            max_calibration_error: max_error,
            coverage_data,
        }
    }

    /// Calibrate uncertainty estimates
    pub fn calibrate_uncertainty(
        &mut self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        y_std: &Array1<f64>,
    ) -> f64 {
        // Find scale factor that minimizes calibration error
        let mut best_scale = 1.0;
        let mut best_error = f64::MAX;

        for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0] {
            let scaled_std = y_std.mapv(|s| s * scale);
            let result = self.regression_calibration_error(y_true, y_pred, &scaled_std);

            if result.mean_calibration_error < best_error {
                best_error = result.mean_calibration_error;
                best_scale = scale;
            }
        }

        self.scale_factor = best_scale;
        best_scale
    }

    /// Apply calibration to uncertainty
    pub fn apply_calibration(&self, y_std: &Array1<f64>) -> Array1<f64> {
        y_std.mapv(|s| s * self.scale_factor)
    }

    /// Get scale factor
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Approximate standard normal quantile
    fn standard_normal_quantile(&self, p: f64) -> f64 {
        // Approximation using Abramowitz and Stegun formula
        if p <= 0.0 || p >= 1.0 {
            return 0.0;
        }

        let a = [
            -3.969683028665376e1,
            2.209460984245205e2,
            -2.759285104469687e2,
            1.383577518672690e2,
            -3.066479806614716e1,
            2.506628277459239e0,
        ];
        let b = [
            -5.447609879822406e1,
            1.615858368580409e2,
            -1.556989798598866e2,
            6.680131188771972e1,
            -1.328068155288572e1,
        ];
        let c = [
            -7.784894002430293e-3,
            -3.223964580411365e-1,
            -2.400758277161838e0,
            -2.549732539343734e0,
            4.374664141464968e0,
            2.938163982698783e0,
        ];
        let d = [
            7.784695709041462e-3,
            3.224671290700398e-1,
            2.445134137142996e0,
            3.754408661907416e0,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }
    }
}

impl Default for Calibrator {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Calibration result
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    pub mean_calibration_error: f64,
    pub max_calibration_error: f64,
    pub coverage_data: Vec<CoveragePoint>,
}

/// Coverage data point
#[derive(Debug, Clone)]
pub struct CoveragePoint {
    pub confidence: f64,
    pub expected_coverage: f64,
    pub actual_coverage: f64,
    pub error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_calibrator() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.1, 2.2, 2.9, 4.1, 4.8]);
        let y_std = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.5]);

        let calibrator = Calibrator::new(10);
        let result = calibrator.regression_calibration_error(&y_true, &y_pred, &y_std);

        assert!(result.mean_calibration_error >= 0.0);
        assert!(result.max_calibration_error >= 0.0);
    }
}
