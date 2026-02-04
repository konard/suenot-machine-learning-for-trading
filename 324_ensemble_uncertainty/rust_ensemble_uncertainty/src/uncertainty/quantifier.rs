//! Uncertainty quantification methods.

use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Uncertainty quantifier for ensemble predictions
pub struct UncertaintyQuantifier;

impl UncertaintyQuantifier {
    /// Create new quantifier
    pub fn new() -> Self {
        Self
    }

    /// Compute prediction variance across ensemble members
    pub fn prediction_variance(&self, predictions: &Array2<f64>) -> Array1<f64> {
        let mean = predictions.mean_axis(Axis(0)).unwrap();
        let n_models = predictions.nrows() as f64;

        let mut variance = Array1::zeros(predictions.ncols());
        for row in predictions.axis_iter(Axis(0)) {
            let diff = &row.to_owned() - &mean;
            variance = variance + &diff.mapv(|x| x.powi(2));
        }
        variance / n_models
    }

    /// Compute prediction standard deviation
    pub fn prediction_std(&self, predictions: &Array2<f64>) -> Array1<f64> {
        self.prediction_variance(predictions).mapv(|v| v.sqrt())
    }

    /// Compute coefficient of variation (relative uncertainty)
    pub fn coefficient_of_variation(&self, predictions: &Array2<f64>) -> Array1<f64> {
        let mean = predictions.mean_axis(Axis(0)).unwrap();
        let std = self.prediction_std(predictions);
        &std / &(mean.mapv(|m| m.abs()) + 1e-8)
    }

    /// Compute interquartile range
    pub fn interquartile_range(&self, predictions: &Array2<f64>) -> Array1<f64> {
        let n_samples = predictions.ncols();
        let mut iqr = Array1::zeros(n_samples);

        for (i, col) in predictions.axis_iter(Axis(1)).enumerate() {
            let mut values: Vec<f64> = col.to_vec();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = values.len();
            let q25_idx = (n as f64 * 0.25) as usize;
            let q75_idx = (n as f64 * 0.75) as usize;

            iqr[i] = values[q75_idx.min(n - 1)] - values[q25_idx];
        }

        iqr
    }

    /// Compute confidence interval
    pub fn confidence_interval(
        &self,
        predictions: &Array2<f64>,
        confidence: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let n_samples = predictions.ncols();
        let alpha = 1.0 - confidence;

        let mut lower = Array1::zeros(n_samples);
        let mut upper = Array1::zeros(n_samples);

        for (i, col) in predictions.axis_iter(Axis(1)).enumerate() {
            let mut values: Vec<f64> = col.to_vec();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = values.len();
            let lower_idx = ((alpha / 2.0) * n as f64) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * n as f64) as usize;

            lower[i] = values[lower_idx];
            upper[i] = values[upper_idx.min(n - 1)];
        }

        (lower, upper)
    }

    /// Compute all uncertainty metrics
    pub fn compute_all_metrics(&self, predictions: &Array2<f64>) -> UncertaintyMetrics {
        let mean = predictions.mean_axis(Axis(0)).unwrap();
        let (ci_lower, ci_upper) = self.confidence_interval(predictions, 0.95);

        UncertaintyMetrics {
            mean,
            std: self.prediction_std(predictions),
            variance: self.prediction_variance(predictions),
            cv: self.coefficient_of_variation(predictions),
            iqr: self.interquartile_range(predictions),
            ci_lower,
            ci_upper,
        }
    }

    /// Decompose uncertainty into epistemic and aleatoric components
    pub fn decompose_uncertainty(
        &self,
        predictions: &Array2<f64>,
        leaf_variances: Option<&Array2<f64>>,
    ) -> UncertaintyDecomposition {
        let total_var = self.prediction_variance(predictions);

        let (epistemic_var, aleatoric_var) = if let Some(leaf_vars) = leaf_variances {
            let aleatoric = leaf_vars.mean_axis(Axis(0)).unwrap();
            let epistemic = (&total_var - &aleatoric).mapv(|v| v.max(0.0));
            (epistemic, aleatoric)
        } else {
            (total_var.clone(), Array1::zeros(total_var.len()))
        };

        UncertaintyDecomposition {
            epistemic: epistemic_var.mapv(|v| v.sqrt()),
            aleatoric: aleatoric_var.mapv(|v| v.sqrt()),
            total: total_var.mapv(|v| v.sqrt()),
        }
    }
}

impl Default for UncertaintyQuantifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Computed uncertainty metrics
#[derive(Debug, Clone)]
pub struct UncertaintyMetrics {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
    pub variance: Array1<f64>,
    pub cv: Array1<f64>,
    pub iqr: Array1<f64>,
    pub ci_lower: Array1<f64>,
    pub ci_upper: Array1<f64>,
}

impl UncertaintyMetrics {
    /// Get CI width
    pub fn ci_width(&self) -> Array1<f64> {
        &self.ci_upper - &self.ci_lower
    }
}

/// Uncertainty decomposition
#[derive(Debug, Clone)]
pub struct UncertaintyDecomposition {
    pub epistemic: Array1<f64>,
    pub aleatoric: Array1<f64>,
    pub total: Array1<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_uncertainty_quantifier() {
        let n_models = 10;
        let n_samples = 100;

        let predictions = Array2::random((n_models, n_samples), Uniform::new(-1.0, 1.0));

        let quantifier = UncertaintyQuantifier::new();

        let std = quantifier.prediction_std(&predictions);
        assert_eq!(std.len(), n_samples);
        assert!(std.iter().all(|&s| s >= 0.0));

        let metrics = quantifier.compute_all_metrics(&predictions);
        assert_eq!(metrics.mean.len(), n_samples);
        assert_eq!(metrics.std.len(), n_samples);
        assert_eq!(metrics.ci_lower.len(), n_samples);
        assert_eq!(metrics.ci_upper.len(), n_samples);
    }
}
