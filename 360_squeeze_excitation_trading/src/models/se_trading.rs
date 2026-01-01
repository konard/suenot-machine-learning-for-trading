//! SE-Enhanced Trading Model
//!
//! This module provides a complete trading model that uses SE blocks
//! for dynamic feature weighting and generates trading signals.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;

use super::se_block::{SEBlock, SqueezeType};
use super::activation::{tanh, sigmoid};

/// Configuration for the SE Trading Model
#[derive(Debug, Clone)]
pub struct SETradingConfig {
    /// Number of input features (channels)
    pub num_features: usize,
    /// Reduction ratio for SE block
    pub reduction_ratio: usize,
    /// Type of squeeze operation
    pub squeeze_type: SqueezeType,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Dropout rate (for training)
    pub dropout_rate: f64,
    /// Whether to use residual connection
    pub use_residual: bool,
}

impl Default for SETradingConfig {
    fn default() -> Self {
        Self {
            num_features: 10,
            reduction_ratio: 4,
            squeeze_type: SqueezeType::GlobalAveragePooling,
            hidden_size: 32,
            dropout_rate: 0.1,
            use_residual: true,
        }
    }
}

/// SE-Enhanced Trading Model
///
/// A complete model that uses SE blocks to dynamically weight features
/// and produces trading signals.
#[derive(Debug, Clone)]
pub struct SETradingModel {
    /// Configuration
    config: SETradingConfig,
    /// SE block for feature attention
    se_block: SEBlock,
    /// Weights for feature aggregation layer
    agg_weights: Array2<f64>,
    /// Bias for aggregation layer
    agg_bias: Array1<f64>,
    /// Weights for output layer
    output_weights: Array1<f64>,
    /// Bias for output layer
    output_bias: f64,
}

impl SETradingModel {
    /// Create a new SE Trading Model
    pub fn new(config: SETradingConfig) -> Self {
        let mut rng = rand::thread_rng();

        let se_block = SEBlock::with_squeeze_type(
            config.num_features,
            config.reduction_ratio,
            config.squeeze_type,
        );

        // Xavier initialization for aggregation layer
        let scale_agg = (2.0 / (config.num_features + config.hidden_size) as f64).sqrt();
        let agg_weights = Array2::from_shape_fn(
            (config.hidden_size, config.num_features),
            |_| rng.gen_range(-scale_agg..scale_agg),
        );

        // Output layer initialization
        let scale_out = (2.0 / (config.hidden_size + 1) as f64).sqrt();
        let output_weights = Array1::from_shape_fn(config.hidden_size, |_| {
            rng.gen_range(-scale_out..scale_out)
        });

        Self {
            config,
            se_block,
            agg_weights,
            agg_bias: Array1::zeros(config.hidden_size),
            output_weights,
            output_bias: 0.0,
        }
    }

    /// Create with default configuration
    pub fn default_model() -> Self {
        Self::new(SETradingConfig::default())
    }

    /// Forward pass to generate trading signal
    ///
    /// # Arguments
    ///
    /// * `x` - Input features of shape (time_steps, num_features)
    ///
    /// # Returns
    ///
    /// Trading signal in range [-1, 1] where:
    /// - -1: Strong sell/short signal
    /// - 0: Neutral
    /// - +1: Strong buy/long signal
    pub fn forward(&self, x: &Array2<f64>) -> f64 {
        // Apply SE block for feature attention
        let se_output = self.se_block.forward(x);

        // Optional residual connection
        let features = if self.config.use_residual {
            &se_output + x
        } else {
            se_output
        };

        // Aggregate temporal information (take last row after attention)
        let last_features = features.row(features.nrows() - 1);

        // Hidden layer with tanh activation
        let hidden = tanh(&(self.agg_weights.dot(&last_features) + &self.agg_bias));

        // Output layer with tanh for [-1, 1] range
        let raw_output = self.output_weights.dot(&hidden) + self.output_bias;
        raw_output.tanh()
    }

    /// Get detailed output including attention weights
    pub fn forward_with_attention(&self, x: &Array2<f64>) -> ModelOutput {
        let attention_weights = self.se_block.get_attention_weights(x);
        let signal = self.forward(x);

        ModelOutput {
            signal,
            attention_weights,
            confidence: signal.abs(),
        }
    }

    /// Get the SE block's attention weights
    pub fn get_attention(&self, x: &Array2<f64>) -> Array1<f64> {
        self.se_block.get_attention_weights(x)
    }

    /// Batch prediction for multiple samples
    pub fn predict_batch(&self, batch: &[Array2<f64>]) -> Vec<f64> {
        batch.iter().map(|x| self.forward(x)).collect()
    }
}

/// Output from the SE Trading Model
#[derive(Debug, Clone)]
pub struct ModelOutput {
    /// Trading signal in [-1, 1]
    pub signal: f64,
    /// Feature attention weights
    pub attention_weights: Array1<f64>,
    /// Confidence level (absolute value of signal)
    pub confidence: f64,
}

impl ModelOutput {
    /// Check if signal suggests going long
    pub fn is_long(&self, threshold: f64) -> bool {
        self.signal > threshold
    }

    /// Check if signal suggests going short
    pub fn is_short(&self, threshold: f64) -> bool {
        self.signal < -threshold
    }

    /// Check if signal is neutral
    pub fn is_neutral(&self, threshold: f64) -> bool {
        self.signal.abs() <= threshold
    }

    /// Get the top-k most attended features
    pub fn top_k_features(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .attention_weights
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);
        indexed
    }
}

/// Ensemble of SE Trading Models for more robust predictions
#[derive(Debug)]
pub struct SEEnsemble {
    /// Individual models
    models: Vec<SETradingModel>,
    /// Ensemble weights
    weights: Array1<f64>,
}

impl SEEnsemble {
    /// Create an ensemble with n models
    pub fn new(n_models: usize, config: SETradingConfig) -> Self {
        let models: Vec<SETradingModel> = (0..n_models)
            .map(|_| SETradingModel::new(config.clone()))
            .collect();

        let weights = Array1::from_elem(n_models, 1.0 / n_models as f64);

        Self { models, weights }
    }

    /// Forward pass through ensemble (weighted average)
    pub fn forward(&self, x: &Array2<f64>) -> f64 {
        let predictions: Vec<f64> = self.models.iter().map(|m| m.forward(x)).collect();

        predictions
            .iter()
            .zip(self.weights.iter())
            .map(|(&p, &w)| p * w)
            .sum()
    }

    /// Get predictions with disagreement measure
    pub fn forward_with_uncertainty(&self, x: &Array2<f64>) -> EnsembleOutput {
        let predictions: Vec<f64> = self.models.iter().map(|m| m.forward(x)).collect();

        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions
            .iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        EnsembleOutput {
            mean_signal: mean,
            std_signal: variance.sqrt(),
            individual_signals: predictions,
        }
    }
}

/// Output from ensemble model
#[derive(Debug, Clone)]
pub struct EnsembleOutput {
    /// Mean prediction across models
    pub mean_signal: f64,
    /// Standard deviation of predictions
    pub std_signal: f64,
    /// Individual model predictions
    pub individual_signals: Vec<f64>,
}

impl EnsembleOutput {
    /// Check if models agree (low uncertainty)
    pub fn models_agree(&self, threshold: f64) -> bool {
        self.std_signal < threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_se_trading_model() {
        let config = SETradingConfig {
            num_features: 5,
            reduction_ratio: 2,
            hidden_size: 16,
            ..Default::default()
        };

        let model = SETradingModel::new(config);
        let input = Array2::ones((50, 5));
        let signal = model.forward(&input);

        assert!(signal >= -1.0 && signal <= 1.0);
    }

    #[test]
    fn test_forward_with_attention() {
        let config = SETradingConfig {
            num_features: 5,
            ..Default::default()
        };

        let model = SETradingModel::new(config);
        let input = Array2::from_shape_fn((50, 5), |(i, j)| (i * j) as f64 * 0.01);
        let output = model.forward_with_attention(&input);

        assert_eq!(output.attention_weights.len(), 5);
        assert!(output.signal >= -1.0 && output.signal <= 1.0);
    }

    #[test]
    fn test_ensemble() {
        let config = SETradingConfig {
            num_features: 5,
            ..Default::default()
        };

        let ensemble = SEEnsemble::new(3, config);
        let input = Array2::ones((50, 5));
        let output = ensemble.forward_with_uncertainty(&input);

        assert_eq!(output.individual_signals.len(), 3);
        assert!(output.mean_signal >= -1.0 && output.mean_signal <= 1.0);
    }
}
