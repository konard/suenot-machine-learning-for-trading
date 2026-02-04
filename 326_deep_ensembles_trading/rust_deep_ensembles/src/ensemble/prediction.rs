//! Ensemble prediction results

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Container for ensemble prediction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePrediction {
    /// Mean prediction (ensemble average)
    pub mean: Array1<f64>,
    /// Total standard deviation
    pub total_std: Array1<f64>,
    /// Epistemic uncertainty (model disagreement)
    pub epistemic_std: Array1<f64>,
    /// Aleatoric uncertainty (data noise)
    pub aleatoric_std: Array1<f64>,
    /// Individual model means
    pub individual_means: Vec<Array1<f64>>,
    /// Individual model standard deviations
    pub individual_stds: Vec<Array1<f64>>,
}

impl EnsemblePrediction {
    /// Create a new ensemble prediction
    pub fn new(
        individual_means: Vec<Array1<f64>>,
        individual_stds: Vec<Array1<f64>>,
    ) -> Self {
        let num_models = individual_means.len();
        let num_samples = individual_means[0].len();

        // Compute ensemble mean
        let mut mean = Array1::zeros(num_samples);
        for model_mean in &individual_means {
            mean = mean + model_mean;
        }
        mean = mean / num_models as f64;

        // Compute epistemic variance (variance of means)
        let mut epistemic_var = Array1::zeros(num_samples);
        for model_mean in &individual_means {
            let diff = model_mean - &mean;
            epistemic_var = epistemic_var + &diff * &diff;
        }
        epistemic_var = epistemic_var / num_models as f64;

        // Compute aleatoric variance (mean of variances)
        let mut aleatoric_var = Array1::zeros(num_samples);
        for model_std in &individual_stds {
            aleatoric_var = aleatoric_var + model_std.mapv(|s| s * s);
        }
        aleatoric_var = aleatoric_var / num_models as f64;

        // Total variance
        let total_var = &epistemic_var + &aleatoric_var;

        Self {
            mean,
            total_std: total_var.mapv(|v: f64| v.sqrt()),
            epistemic_std: epistemic_var.mapv(|v: f64| v.sqrt()),
            aleatoric_std: aleatoric_var.mapv(|v: f64| v.sqrt()),
            individual_means,
            individual_stds,
        }
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.mean.len()
    }

    /// Get the number of models
    pub fn num_models(&self) -> usize {
        self.individual_means.len()
    }

    /// Get epistemic ratio (epistemic / total)
    pub fn epistemic_ratio(&self) -> Array1<f64> {
        let total = &self.total_std;
        &self.epistemic_std / total.mapv(|t| if t > 0.0 { t } else { 1e-10 })
    }

    /// Get prediction confidence (1 - normalized uncertainty)
    pub fn confidence(&self) -> Array1<f64> {
        let max_std = self.total_std.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_std > 0.0 {
            self.total_std.mapv(|s| 1.0 - s / max_std)
        } else {
            Array1::ones(self.num_samples())
        }
    }

    /// Get signal strength (mean / std, like Sharpe ratio)
    pub fn signal_strength(&self) -> Array1<f64> {
        &self.mean / self.total_std.mapv(|s| if s > 0.0 { s } else { 1e-10 })
    }

    /// Get prediction at index
    pub fn get_prediction(&self, idx: usize) -> SinglePrediction {
        SinglePrediction {
            mean: self.mean[idx],
            total_std: self.total_std[idx],
            epistemic_std: self.epistemic_std[idx],
            aleatoric_std: self.aleatoric_std[idx],
        }
    }
}

/// Single prediction result
#[derive(Debug, Clone, Copy)]
pub struct SinglePrediction {
    pub mean: f64,
    pub total_std: f64,
    pub epistemic_std: f64,
    pub aleatoric_std: f64,
}

impl SinglePrediction {
    /// Get epistemic ratio
    pub fn epistemic_ratio(&self) -> f64 {
        if self.total_std > 0.0 {
            self.epistemic_std / self.total_std
        } else {
            0.0
        }
    }

    /// Get signal strength
    pub fn signal_strength(&self) -> f64 {
        if self.total_std > 0.0 {
            self.mean / self.total_std
        } else {
            0.0
        }
    }

    /// Check if prediction is confident (low epistemic uncertainty)
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.epistemic_ratio() < threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_prediction() {
        let means = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![1.1, 2.1, 2.9]),
            Array1::from_vec(vec![0.9, 1.9, 3.1]),
        ];

        let stds = vec![
            Array1::from_vec(vec![0.1, 0.1, 0.1]),
            Array1::from_vec(vec![0.1, 0.1, 0.1]),
            Array1::from_vec(vec![0.1, 0.1, 0.1]),
        ];

        let pred = EnsemblePrediction::new(means, stds);

        assert_eq!(pred.num_samples(), 3);
        assert_eq!(pred.num_models(), 3);

        // Check mean is close to expected
        assert!((pred.mean[0] - 1.0).abs() < 0.1);
        assert!((pred.mean[1] - 2.0).abs() < 0.1);
        assert!((pred.mean[2] - 3.0).abs() < 0.1);
    }
}
