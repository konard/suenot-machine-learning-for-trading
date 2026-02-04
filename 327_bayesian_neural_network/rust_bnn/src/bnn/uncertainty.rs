//! Uncertainty estimation for Bayesian Neural Networks.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Type of uncertainty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintyType {
    /// Epistemic (model) uncertainty - reducible with more data
    Epistemic,
    /// Aleatoric (data) uncertainty - irreducible noise
    Aleatoric,
    /// Total uncertainty (epistemic + aleatoric)
    Total,
}

/// Uncertainty estimate from a Bayesian model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyEstimate {
    /// Mean prediction
    pub mean: Vec<f64>,
    /// Standard deviation of prediction
    pub std: Vec<f64>,
    /// Epistemic uncertainty (variance of means)
    pub epistemic: Vec<f64>,
    /// Aleatoric uncertainty (mean of variances, if available)
    pub aleatoric: Option<Vec<f64>>,
    /// Total uncertainty
    pub total: Vec<f64>,
    /// Confidence (1 / (1 + uncertainty))
    pub confidence: Vec<f64>,
    /// Number of MC samples used
    pub n_samples: usize,
}

impl UncertaintyEstimate {
    /// Create a new uncertainty estimate from MC samples.
    ///
    /// # Arguments
    /// * `samples` - MC samples of shape (n_samples, output_dim)
    pub fn from_samples(samples: &[Vec<f64>]) -> Self {
        let n_samples = samples.len();
        let output_dim = samples.first().map(|s| s.len()).unwrap_or(0);

        if n_samples == 0 || output_dim == 0 {
            return Self {
                mean: vec![],
                std: vec![],
                epistemic: vec![],
                aleatoric: None,
                total: vec![],
                confidence: vec![],
                n_samples: 0,
            };
        }

        // Calculate mean for each output dimension
        let mut mean = vec![0.0; output_dim];
        for sample in samples {
            for (i, &val) in sample.iter().enumerate() {
                mean[i] += val;
            }
        }
        for m in &mut mean {
            *m /= n_samples as f64;
        }

        // Calculate variance (epistemic uncertainty)
        let mut variance = vec![0.0; output_dim];
        for sample in samples {
            for (i, &val) in sample.iter().enumerate() {
                variance[i] += (val - mean[i]).powi(2);
            }
        }
        for v in &mut variance {
            *v /= n_samples as f64;
        }

        let std: Vec<f64> = variance.iter().map(|v| v.sqrt()).collect();
        let epistemic = variance.clone();
        let total = variance.clone();

        // Confidence as inverse of total uncertainty
        let confidence: Vec<f64> = total
            .iter()
            .map(|&t| 1.0 / (1.0 + t.sqrt()))
            .collect();

        Self {
            mean,
            std,
            epistemic,
            aleatoric: None,
            total,
            confidence,
            n_samples,
        }
    }

    /// Create estimate with separate epistemic and aleatoric uncertainties.
    pub fn from_samples_heteroscedastic(
        mean_samples: &[Vec<f64>],
        var_samples: &[Vec<f64>],
    ) -> Self {
        let n_samples = mean_samples.len();
        let output_dim = mean_samples.first().map(|s| s.len()).unwrap_or(0);

        if n_samples == 0 || output_dim == 0 {
            return Self::from_samples(&[]);
        }

        // Mean of means
        let mut mean = vec![0.0; output_dim];
        for sample in mean_samples {
            for (i, &val) in sample.iter().enumerate() {
                mean[i] += val;
            }
        }
        for m in &mut mean {
            *m /= n_samples as f64;
        }

        // Epistemic: variance of means
        let mut epistemic = vec![0.0; output_dim];
        for sample in mean_samples {
            for (i, &val) in sample.iter().enumerate() {
                epistemic[i] += (val - mean[i]).powi(2);
            }
        }
        for e in &mut epistemic {
            *e /= n_samples as f64;
        }

        // Aleatoric: mean of variances
        let mut aleatoric = vec![0.0; output_dim];
        for sample in var_samples {
            for (i, &val) in sample.iter().enumerate() {
                aleatoric[i] += val;
            }
        }
        for a in &mut aleatoric {
            *a /= n_samples as f64;
        }

        // Total uncertainty
        let total: Vec<f64> = epistemic
            .iter()
            .zip(aleatoric.iter())
            .map(|(&e, &a)| e + a)
            .collect();

        let std: Vec<f64> = total.iter().map(|t| t.sqrt()).collect();

        let confidence: Vec<f64> = total
            .iter()
            .map(|&t| 1.0 / (1.0 + t.sqrt()))
            .collect();

        Self {
            mean,
            std,
            epistemic,
            aleatoric: Some(aleatoric),
            total,
            confidence,
            n_samples,
        }
    }

    /// Get the mean confidence (average across dimensions).
    pub fn mean_confidence(&self) -> f64 {
        if self.confidence.is_empty() {
            return 0.0;
        }
        self.confidence.iter().sum::<f64>() / self.confidence.len() as f64
    }

    /// Get the mean epistemic uncertainty.
    pub fn mean_epistemic(&self) -> f64 {
        if self.epistemic.is_empty() {
            return 0.0;
        }
        self.epistemic.iter().sum::<f64>() / self.epistemic.len() as f64
    }

    /// Get the mean aleatoric uncertainty.
    pub fn mean_aleatoric(&self) -> Option<f64> {
        self.aleatoric.as_ref().map(|a| {
            if a.is_empty() {
                0.0
            } else {
                a.iter().sum::<f64>() / a.len() as f64
            }
        })
    }

    /// Check if this is a high-confidence prediction.
    pub fn is_high_confidence(&self, threshold: f64) -> bool {
        self.mean_confidence() >= threshold
    }

    /// Get predicted class (index of maximum mean).
    pub fn predicted_class(&self) -> usize {
        self.mean
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get probability of predicted class (assuming softmax output).
    pub fn predicted_probability(&self) -> f64 {
        let class = self.predicted_class();
        self.mean.get(class).copied().unwrap_or(0.0)
    }
}

/// Calculate predictive entropy from probability distribution.
pub fn predictive_entropy(probs: &[f64]) -> f64 {
    let eps = 1e-10;
    -probs
        .iter()
        .map(|&p| {
            if p > eps {
                p * p.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

/// Calculate mutual information from MC probability samples.
///
/// MI = H[E[p]] - E[H[p]]
/// This measures information gain about prediction from knowing weights.
pub fn mutual_information(mc_probs: &[Vec<f64>]) -> f64 {
    if mc_probs.is_empty() {
        return 0.0;
    }

    let n_samples = mc_probs.len();
    let n_classes = mc_probs[0].len();

    // Mean probabilities
    let mut mean_probs = vec![0.0; n_classes];
    for probs in mc_probs {
        for (i, &p) in probs.iter().enumerate() {
            mean_probs[i] += p;
        }
    }
    for p in &mut mean_probs {
        *p /= n_samples as f64;
    }

    // Entropy of mean
    let entropy_of_mean = predictive_entropy(&mean_probs);

    // Mean of entropies
    let mean_of_entropies: f64 = mc_probs
        .iter()
        .map(|probs| predictive_entropy(probs))
        .sum::<f64>()
        / n_samples as f64;

    entropy_of_mean - mean_of_entropies
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_from_samples() {
        let samples = vec![
            vec![0.7, 0.2, 0.1],
            vec![0.65, 0.25, 0.1],
            vec![0.75, 0.15, 0.1],
        ];

        let estimate = UncertaintyEstimate::from_samples(&samples);

        assert_eq!(estimate.n_samples, 3);
        assert_eq!(estimate.mean.len(), 3);
        assert!(estimate.mean[0] > 0.6);
    }

    #[test]
    fn test_predictive_entropy() {
        let probs = vec![0.5, 0.5]; // Maximum entropy for 2 classes
        let entropy = predictive_entropy(&probs);
        assert!((entropy - 2.0_f64.ln()).abs() < 1e-6);

        let certain = vec![1.0, 0.0]; // Minimum entropy
        let entropy_certain = predictive_entropy(&certain);
        assert!(entropy_certain < 1e-6);
    }
}
