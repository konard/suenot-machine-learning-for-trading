//! Prediction types for VAE outputs

use serde::{Deserialize, Serialize};

/// Prediction with uncertainty from VAE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Expected return
    pub expected_return: f64,

    /// Standard deviation of return prediction
    pub return_std: f64,

    /// Probability of positive return
    pub prob_positive: f64,

    /// Predicted market regime (0=bull, 1=bear, 2=sideways)
    pub regime: usize,

    /// Regime probabilities
    pub regime_probs: Vec<f64>,

    /// Latent mean
    pub latent_mu: Vec<f64>,

    /// Latent log variance
    pub latent_logvar: Vec<f64>,

    /// Reconstruction error (optional, for anomaly detection)
    pub reconstruction_error: Option<f64>,
}

impl Prediction {
    /// Create a new prediction
    pub fn new(
        expected_return: f64,
        return_std: f64,
        regime: usize,
        regime_probs: Vec<f64>,
        latent_mu: Vec<f64>,
        latent_logvar: Vec<f64>,
    ) -> Self {
        // Calculate probability of positive return assuming Gaussian
        let prob_positive = 1.0 - normal_cdf(-expected_return / return_std.max(1e-8));

        Self {
            expected_return,
            return_std,
            prob_positive,
            regime,
            regime_probs,
            latent_mu,
            latent_logvar,
            reconstruction_error: None,
        }
    }

    /// Get confidence (based on probability away from 0.5)
    pub fn confidence(&self) -> f64 {
        (self.prob_positive - 0.5).abs() * 2.0
    }

    /// Check if prediction is bullish
    pub fn is_bullish(&self) -> bool {
        self.prob_positive > 0.5
    }

    /// Check if prediction is bearish
    pub fn is_bearish(&self) -> bool {
        self.prob_positive < 0.5
    }

    /// Get regime name
    pub fn regime_name(&self) -> &'static str {
        match self.regime {
            0 => "Bull",
            1 => "Bear",
            2 => "Sideways",
            _ => "Unknown",
        }
    }

    /// Calculate information ratio (expected return / uncertainty)
    pub fn information_ratio(&self) -> f64 {
        self.expected_return / self.return_std.max(1e-8)
    }

    /// Check if this is a high-confidence prediction
    pub fn is_high_confidence(&self, threshold: f64) -> bool {
        self.confidence() > threshold
    }

    /// Check if prediction indicates anomaly (high reconstruction error)
    pub fn is_anomaly(&self, threshold: f64) -> bool {
        self.reconstruction_error
            .map(|e| e > threshold)
            .unwrap_or(false)
    }
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Horner form)
fn erf(x: f64) -> f64 {
    // Approximation constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Aggregate predictions from multiple samples
#[derive(Debug, Clone)]
pub struct AggregatedPrediction {
    /// Mean expected return across samples
    pub mean_return: f64,

    /// Standard deviation of expected returns
    pub return_uncertainty: f64,

    /// Mean probability of positive return
    pub mean_prob_positive: f64,

    /// Mode regime (most common)
    pub mode_regime: usize,

    /// Average regime probabilities
    pub mean_regime_probs: Vec<f64>,

    /// Number of samples
    pub n_samples: usize,
}

impl AggregatedPrediction {
    /// Aggregate multiple predictions
    pub fn from_predictions(predictions: &[Prediction]) -> Self {
        if predictions.is_empty() {
            return Self {
                mean_return: 0.0,
                return_uncertainty: 0.0,
                mean_prob_positive: 0.5,
                mode_regime: 0,
                mean_regime_probs: vec![],
                n_samples: 0,
            };
        }

        let n = predictions.len() as f64;

        // Calculate mean return
        let mean_return: f64 = predictions.iter().map(|p| p.expected_return).sum::<f64>() / n;

        // Calculate uncertainty (std of expected returns)
        let variance: f64 = predictions.iter()
            .map(|p| (p.expected_return - mean_return).powi(2))
            .sum::<f64>() / n;
        let return_uncertainty = variance.sqrt();

        // Calculate mean probability
        let mean_prob_positive: f64 = predictions.iter()
            .map(|p| p.prob_positive)
            .sum::<f64>() / n;

        // Find mode regime
        let mut regime_counts = vec![0usize; 3];
        for p in predictions {
            if p.regime < 3 {
                regime_counts[p.regime] += 1;
            }
        }
        let mode_regime = regime_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Calculate mean regime probs
        let n_regimes = predictions[0].regime_probs.len();
        let mut mean_regime_probs = vec![0.0; n_regimes];
        for p in predictions {
            for (i, &prob) in p.regime_probs.iter().enumerate() {
                if i < n_regimes {
                    mean_regime_probs[i] += prob;
                }
            }
        }
        for prob in &mut mean_regime_probs {
            *prob /= n;
        }

        Self {
            mean_return,
            return_uncertainty,
            mean_prob_positive,
            mode_regime,
            mean_regime_probs,
            n_samples: predictions.len(),
        }
    }

    /// Get overall confidence
    pub fn confidence(&self) -> f64 {
        (self.mean_prob_positive - 0.5).abs() * 2.0
    }

    /// Check if aggregated prediction is bullish
    pub fn is_bullish(&self) -> bool {
        self.mean_prob_positive > 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_confidence() {
        let pred = Prediction::new(
            0.01,
            0.005,
            0,
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.2],
            vec![-0.1, -0.2],
        );

        assert!(pred.is_bullish());
        assert!(pred.confidence() > 0.5);
        assert_eq!(pred.regime_name(), "Bull");
    }

    #[test]
    fn test_aggregated_prediction() {
        let predictions = vec![
            Prediction::new(0.01, 0.005, 0, vec![0.7, 0.2, 0.1], vec![], vec![]),
            Prediction::new(0.015, 0.006, 0, vec![0.8, 0.1, 0.1], vec![], vec![]),
            Prediction::new(0.008, 0.004, 1, vec![0.4, 0.4, 0.2], vec![], vec![]),
        ];

        let agg = AggregatedPrediction::from_predictions(&predictions);

        assert_eq!(agg.n_samples, 3);
        assert_eq!(agg.mode_regime, 0); // Two predictions have regime 0
        assert!(agg.mean_return > 0.0);
    }

    #[test]
    fn test_normal_cdf() {
        // Test some known values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }
}
