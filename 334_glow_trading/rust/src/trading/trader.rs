//! GLOW Trader - Trading system based on GLOW model
//!
//! Uses GLOW's likelihood and latent space for trading decisions

use crate::model::GLOWModel;
use crate::data::{FeatureExtractor, MarketFeatures, Normalizer};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for GLOW trader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraderConfig {
    /// Minimum log-likelihood threshold for trading
    pub likelihood_threshold: f64,
    /// Minimum confidence for taking a position
    pub confidence_threshold: f64,
    /// Number of regimes for clustering
    pub num_regimes: usize,
    /// Lookback period for feature extraction
    pub lookback: usize,
    /// Maximum position size (0-1)
    pub max_position: f64,
}

impl Default for TraderConfig {
    fn default() -> Self {
        Self {
            likelihood_threshold: -20.0,
            confidence_threshold: 0.3,
            num_regimes: 4,
            lookback: 20,
            max_position: 1.0,
        }
    }
}

/// Trading signal from GLOW model
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Position size (-1 to 1)
    pub signal: f64,
    /// Log-likelihood of current state
    pub log_likelihood: f64,
    /// Whether current state is in learned distribution
    pub in_distribution: bool,
    /// Detected market regime
    pub regime: usize,
    /// Latent representation
    pub latent: Array1<f64>,
    /// Confidence level (0-1)
    pub confidence: f64,
}

impl TradingSignal {
    /// Create a neutral (no trade) signal
    pub fn neutral() -> Self {
        Self {
            signal: 0.0,
            log_likelihood: f64::NEG_INFINITY,
            in_distribution: false,
            regime: 0,
            latent: Array1::zeros(1),
            confidence: 0.0,
        }
    }
}

/// GLOW-based trader
pub struct GLOWTrader {
    /// GLOW model
    model: GLOWModel,
    /// Feature normalizer
    normalizer: Option<Normalizer>,
    /// Configuration
    config: TraderConfig,
    /// Regime centroids from clustering
    regime_centroids: Option<Array2<f64>>,
}

impl GLOWTrader {
    /// Create a new GLOW trader
    pub fn new(model: GLOWModel, config: TraderConfig) -> Self {
        Self {
            model,
            normalizer: None,
            config,
            regime_centroids: None,
        }
    }

    /// Fit the normalizer on training data
    pub fn fit_normalizer(&mut self, features: &Array2<f64>) {
        self.normalizer = Some(Normalizer::fit(features));
    }

    /// Set pre-fitted normalizer
    pub fn set_normalizer(&mut self, normalizer: Normalizer) {
        self.normalizer = Some(normalizer);
    }

    /// Fit regime clustering on training data
    pub fn fit_regimes(&mut self, features: &Array2<f64>) {
        // Normalize features
        let normalized = if let Some(ref norm) = self.normalizer {
            norm.transform(features)
        } else {
            features.clone()
        };

        // Get latent representations
        let (z, _) = self.model.forward(&normalized);

        // Simple k-means clustering
        self.regime_centroids = Some(self.kmeans(&z, self.config.num_regimes));
    }

    /// Simple k-means clustering
    fn kmeans(&self, data: &Array2<f64>, k: usize) -> Array2<f64> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Initialize centroids with first k samples
        let mut centroids = Array2::zeros((k, n_features));
        let step = n_samples / k;
        for i in 0..k {
            let idx = i * step;
            centroids.row_mut(i).assign(&data.row(idx));
        }

        // Run k-means for fixed iterations
        for _ in 0..20 {
            // Assign samples to nearest centroid
            let mut assignments = vec![0usize; n_samples];
            let mut counts = vec![0usize; k];

            for i in 0..n_samples {
                let sample = data.row(i);
                let mut min_dist = f64::INFINITY;
                let mut min_idx = 0;

                for j in 0..k {
                    let centroid = centroids.row(j);
                    let dist: f64 = sample
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum();
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = j;
                    }
                }

                assignments[i] = min_idx;
                counts[min_idx] += 1;
            }

            // Update centroids
            let mut new_centroids = Array2::zeros((k, n_features));
            for i in 0..n_samples {
                let cluster = assignments[i];
                new_centroids
                    .row_mut(cluster)
                    .zip_mut_with(&data.row(i), |a, &b| *a += b);
            }

            for j in 0..k {
                if counts[j] > 0 {
                    new_centroids
                        .row_mut(j)
                        .mapv_inplace(|v| v / counts[j] as f64);
                }
            }

            centroids = new_centroids;
        }

        centroids
    }

    /// Get regime for a latent vector
    fn get_regime(&self, z: &Array1<f64>) -> usize {
        if let Some(ref centroids) = self.regime_centroids {
            let mut min_dist = f64::INFINITY;
            let mut min_idx = 0;

            for (idx, centroid) in centroids.outer_iter().enumerate() {
                let dist: f64 = z
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = idx;
                }
            }

            min_idx
        } else {
            0
        }
    }

    /// Generate trading signal from market features
    pub fn generate_signal(&mut self, features: &MarketFeatures) -> TradingSignal {
        let feature_array = features.to_array();
        self.generate_signal_from_array(&feature_array)
    }

    /// Generate trading signal from feature array
    pub fn generate_signal_from_array(&mut self, features: &Array1<f64>) -> TradingSignal {
        // Normalize features
        let normalized = if let Some(ref norm) = self.normalizer {
            norm.transform_sample(features)
        } else {
            features.clone()
        };

        // Convert to 2D for model
        let x = normalized.clone().insert_axis(ndarray::Axis(0));

        // Compute log probability
        let log_prob = self.model.log_prob(&x);
        let log_likelihood = log_prob[0];

        // Get latent representation
        let (z, _) = self.model.forward(&x);
        let latent = z.row(0).to_owned();

        // Check if in distribution
        let in_distribution = log_likelihood > self.config.likelihood_threshold;

        // Get regime
        let regime = self.get_regime(&latent);

        // Compute signal from latent
        // Use first latent dimension as primary signal
        let raw_signal = latent[0].tanh();

        // Compute confidence based on likelihood
        let confidence = if in_distribution {
            let normalized_ll = (log_likelihood - self.config.likelihood_threshold)
                / (-self.config.likelihood_threshold);
            normalized_ll.clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Final signal
        let signal = if in_distribution && confidence >= self.config.confidence_threshold {
            raw_signal * confidence * self.config.max_position
        } else {
            0.0
        };

        TradingSignal {
            signal,
            log_likelihood,
            in_distribution,
            regime,
            latent,
            confidence,
        }
    }

    /// Generate multiple scenarios for risk analysis
    pub fn generate_scenarios(&mut self, num_scenarios: usize, temperature: f64) -> Array2<f64> {
        let samples = self.model.sample(num_scenarios, temperature);

        // Denormalize if normalizer is available
        if let Some(ref norm) = self.normalizer {
            norm.inverse_transform(&samples)
        } else {
            samples
        }
    }

    /// Compute Value at Risk from scenarios
    pub fn compute_var(&mut self, num_scenarios: usize, confidence: f64) -> f64 {
        let scenarios = self.generate_scenarios(num_scenarios, 1.0);

        // Assuming first column is returns
        let mut returns: Vec<f64> = scenarios.column(0).to_vec();
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = ((1.0 - confidence) * num_scenarios as f64) as usize;
        returns.get(idx).copied().unwrap_or(0.0)
    }

    /// Compute Conditional Value at Risk (Expected Shortfall)
    pub fn compute_cvar(&mut self, num_scenarios: usize, confidence: f64) -> f64 {
        let scenarios = self.generate_scenarios(num_scenarios, 1.0);

        // Assuming first column is returns
        let mut returns: Vec<f64> = scenarios.column(0).to_vec();
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let var_idx = ((1.0 - confidence) * num_scenarios as f64) as usize;
        let var = returns.get(var_idx).copied().unwrap_or(0.0);

        // Average of returns below VaR
        let tail: Vec<f64> = returns.iter().filter(|&&r| r <= var).copied().collect();
        if tail.is_empty() {
            var
        } else {
            tail.iter().sum::<f64>() / tail.len() as f64
        }
    }

    /// Get reference to underlying model
    pub fn model(&self) -> &GLOWModel {
        &self.model
    }

    /// Get mutable reference to underlying model
    pub fn model_mut(&mut self) -> &mut GLOWModel {
        &mut self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GLOWConfig;

    #[test]
    fn test_trader_creation() {
        let config = GLOWConfig::with_features(16);
        let model = GLOWModel::new(config);
        let trader_config = TraderConfig::default();

        let trader = GLOWTrader::new(model, trader_config);
        assert!(trader.normalizer.is_none());
    }

    #[test]
    fn test_generate_signal() {
        let config = GLOWConfig::with_features(16);
        let model = GLOWModel::new(config);
        let trader_config = TraderConfig::default();

        let mut trader = GLOWTrader::new(model, trader_config);

        let features = Array1::from_vec(vec![0.0; 16]);
        let signal = trader.generate_signal_from_array(&features);

        assert!(signal.log_likelihood.is_finite());
        assert!(signal.signal >= -1.0 && signal.signal <= 1.0);
    }

    #[test]
    fn test_scenario_generation() {
        let config = GLOWConfig::with_features(16);
        let model = GLOWModel::new(config);
        let trader_config = TraderConfig::default();

        let mut trader = GLOWTrader::new(model, trader_config);

        let scenarios = trader.generate_scenarios(100, 1.0);
        assert_eq!(scenarios.nrows(), 100);
        assert_eq!(scenarios.ncols(), 16);
    }
}
