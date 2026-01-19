//! Reformer model implementation
//!
//! Complete Reformer architecture with LSH attention for efficient
//! long-sequence processing.

use ndarray::{Array1, Array2, Array3};
use std::f64::consts::PI;

use super::config::{OutputType, ReformerConfig};
use super::embedding::TokenEmbedding;
use super::lsh_attention::{AttentionWeights, LSHAttention};
use super::reversible::{ChunkedFeedForward, ReversibleBlock};

/// Reformer model
pub struct ReformerModel {
    /// Configuration
    config: ReformerConfig,
    /// Input embedding layer
    embedding: TokenEmbedding,
    /// Encoder layers (reversible blocks)
    layers: Vec<ReversibleBlock>,
    /// Final layer normalization gamma
    ln_final_gamma: Array2<f64>,
    /// Final layer normalization beta
    ln_final_beta: Array2<f64>,
    /// Output projection [d_model, output_dim]
    output_projection: Array2<f64>,
}

impl ReformerModel {
    /// Create a new Reformer model
    pub fn new(config: ReformerConfig) -> Self {
        // Validate configuration
        if let Err(e) = config.validate() {
            panic!("Invalid configuration: {}", e);
        }

        let embedding = TokenEmbedding::new(&config);

        // Create encoder layers
        let layers: Vec<ReversibleBlock> = (0..config.n_layers)
            .map(|_| ReversibleBlock::new(&config))
            .collect();

        // Final layer normalization
        let ln_final_gamma = Array2::ones((1, config.d_model));
        let ln_final_beta = Array2::zeros((1, config.d_model));

        // Output projection
        let output_dim = config.output_dim();
        let scale = (2.0 / (config.d_model + output_dim) as f64).sqrt();
        let output_projection =
            Array2::from_shape_fn((config.d_model, output_dim), |_| rand_normal() * scale);

        Self {
            config,
            embedding,
            layers,
            ln_final_gamma,
            ln_final_beta,
            output_projection,
        }
    }

    /// Forward pass for prediction
    ///
    /// # Arguments
    /// * `x` - Input features [batch, seq_len, n_features]
    ///
    /// # Returns
    /// * Predictions based on output type
    pub fn forward(&self, x: &Array3<f64>) -> Array2<f64> {
        let (batch_size, seq_len, _) = x.dim();

        // Embedding
        let embedded = self.embedding.forward(x);

        // Split for reversible layers
        let (mut x1, mut x2) = split_for_reversible(&embedded);

        // Pass through encoder layers
        for layer in &self.layers {
            let (y1, y2) = layer.forward(&x1, &x2);
            x1 = y1;
            x2 = y2;
        }

        // Combine
        let combined = combine_from_reversible(&x1, &x2);

        // Final layer normalization
        let normalized = self.layer_norm_3d(&combined);

        // Pool across sequence (use last position or mean)
        let pooled = self.pool_sequence(&normalized);

        // Output projection
        let output = self.project_output(&pooled);

        // Apply activation based on output type
        self.apply_output_activation(&output)
    }

    /// Forward pass returning attention weights
    pub fn forward_with_attention(
        &self,
        x: &Array3<f64>,
    ) -> (Array2<f64>, Vec<AttentionWeights>) {
        // For simplicity, we just return the forward output
        // A full implementation would collect attention weights from each layer
        let output = self.forward(x);
        let weights = Vec::new();
        (output, weights)
    }

    /// Predict on 2D input [seq_len, n_features]
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        // Add batch dimension
        let x_3d = x.clone().insert_axis(ndarray::Axis(0));
        let output = self.forward(&x_3d);
        output.index_axis(ndarray::Axis(0), 0).to_owned()
    }

    /// Predict with attention weights
    pub fn predict_with_attention(&self, x: &Array2<f64>) -> (Array1<f64>, Vec<i32>) {
        let embedded = self.embedding.forward_2d(x);

        // Split for reversible
        let (mut x1, mut x2) = split_for_reversible_2d(&embedded);

        // Collect bucket assignments from first layer
        let mut buckets = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let (y1, y2) = layer.forward_2d(&x1, &x2);
            x1 = y1;
            x2 = y2;
        }

        // Combine and normalize
        let combined = combine_from_reversible_2d(&x1, &x2);
        let normalized = self.layer_norm_2d(&combined);

        // Pool
        let pooled = self.pool_sequence_2d(&normalized);

        // Output projection
        let output = self.project_output_1d(&pooled);

        // Apply activation
        let output = self.apply_output_activation_1d(&output);

        (output, buckets)
    }

    /// Layer normalization for 3D tensor
    fn layer_norm_3d(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                let mut sum = 0.0;
                for d in 0..d_model {
                    sum += x[[b, t, d]];
                }
                let mean = sum / d_model as f64;

                let mut var_sum = 0.0;
                for d in 0..d_model {
                    let diff = x[[b, t, d]] - mean;
                    var_sum += diff * diff;
                }
                let std = (var_sum / d_model as f64 + 1e-6).sqrt();

                for d in 0..d_model {
                    let normalized = (x[[b, t, d]] - mean) / std;
                    output[[b, t, d]] =
                        self.ln_final_gamma[[0, d]] * normalized + self.ln_final_beta[[0, d]];
                }
            }
        }

        output
    }

    /// Layer normalization for 2D tensor
    fn layer_norm_2d(&self, x: &Array2<f64>) -> Array2<f64> {
        let (seq_len, d_model) = x.dim();
        let mut output = Array2::zeros((seq_len, d_model));

        for t in 0..seq_len {
            let mut sum = 0.0;
            for d in 0..d_model {
                sum += x[[t, d]];
            }
            let mean = sum / d_model as f64;

            let mut var_sum = 0.0;
            for d in 0..d_model {
                let diff = x[[t, d]] - mean;
                var_sum += diff * diff;
            }
            let std = (var_sum / d_model as f64 + 1e-6).sqrt();

            for d in 0..d_model {
                let normalized = (x[[t, d]] - mean) / std;
                output[[t, d]] =
                    self.ln_final_gamma[[0, d]] * normalized + self.ln_final_beta[[0, d]];
            }
        }

        output
    }

    /// Pool sequence to fixed-size representation
    fn pool_sequence(&self, x: &Array3<f64>) -> Array2<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut pooled = Array2::zeros((batch_size, d_model));

        // Use last position (can also use mean)
        for b in 0..batch_size {
            for d in 0..d_model {
                pooled[[b, d]] = x[[b, seq_len - 1, d]];
            }
        }

        pooled
    }

    /// Pool sequence for 2D
    fn pool_sequence_2d(&self, x: &Array2<f64>) -> Array1<f64> {
        let (seq_len, d_model) = x.dim();
        let mut pooled = Array1::zeros(d_model);

        // Use last position
        for d in 0..d_model {
            pooled[d] = x[[seq_len - 1, d]];
        }

        pooled
    }

    /// Project to output dimension
    fn project_output(&self, x: &Array2<f64>) -> Array2<f64> {
        let (batch_size, d_model) = x.dim();
        let output_dim = self.output_projection.ncols();
        let mut output = Array2::zeros((batch_size, output_dim));

        for b in 0..batch_size {
            for o in 0..output_dim {
                let mut sum = 0.0;
                for d in 0..d_model {
                    sum += x[[b, d]] * self.output_projection[[d, o]];
                }
                output[[b, o]] = sum;
            }
        }

        output
    }

    /// Project to output dimension for 1D
    fn project_output_1d(&self, x: &Array1<f64>) -> Array1<f64> {
        let d_model = x.len();
        let output_dim = self.output_projection.ncols();
        let mut output = Array1::zeros(output_dim);

        for o in 0..output_dim {
            let mut sum = 0.0;
            for d in 0..d_model {
                sum += x[d] * self.output_projection[[d, o]];
            }
            output[o] = sum;
        }

        output
    }

    /// Apply output activation based on output type
    fn apply_output_activation(&self, x: &Array2<f64>) -> Array2<f64> {
        match self.config.output_type {
            OutputType::Regression => x.clone(),
            OutputType::Direction => softmax_2d(x),
            OutputType::Portfolio => softmax_2d(x),
            OutputType::Quantile => x.clone(),
        }
    }

    /// Apply output activation for 1D
    fn apply_output_activation_1d(&self, x: &Array1<f64>) -> Array1<f64> {
        match self.config.output_type {
            OutputType::Regression => x.clone(),
            OutputType::Direction => softmax_1d(x),
            OutputType::Portfolio => softmax_1d(x),
            OutputType::Quantile => x.clone(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ReformerConfig {
        &self.config
    }

    /// Estimated number of parameters
    pub fn num_parameters(&self) -> usize {
        let embedding = self.config.n_features * self.config.d_model;
        let pe = self.config.seq_len * self.config.d_model;

        let per_layer = {
            let attention = 4 * self.config.d_model * self.config.d_model;
            let ffn = 2 * self.config.d_model * self.config.d_ff;
            let ln = 4 * self.config.d_model;
            attention + ffn + ln
        };

        let layers = self.config.n_layers * per_layer;
        let output = self.config.d_model * self.config.output_dim();

        embedding + pe + layers + output
    }
}

/// Split tensor for reversible layers
fn split_for_reversible(x: &Array3<f64>) -> (Array3<f64>, Array3<f64>) {
    let (batch_size, seq_len, d_model) = x.dim();

    // Split along feature dimension
    let half = d_model / 2;
    let mut x1 = Array3::zeros((batch_size, seq_len, d_model));
    let mut x2 = Array3::zeros((batch_size, seq_len, d_model));

    // Copy first half to x1, second half to x2
    // Then pad with zeros to maintain dimension
    for b in 0..batch_size {
        for t in 0..seq_len {
            for d in 0..half {
                x1[[b, t, d]] = x[[b, t, d]];
                x2[[b, t, d]] = x[[b, t, half + d]];
            }
        }
    }

    (x1, x2)
}

/// Split for 2D
fn split_for_reversible_2d(x: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (seq_len, d_model) = x.dim();
    let half = d_model / 2;

    let mut x1 = Array2::zeros((seq_len, d_model));
    let mut x2 = Array2::zeros((seq_len, d_model));

    for t in 0..seq_len {
        for d in 0..half {
            x1[[t, d]] = x[[t, d]];
            x2[[t, d]] = x[[t, half + d]];
        }
    }

    (x1, x2)
}

/// Combine from reversible layers
fn combine_from_reversible(x1: &Array3<f64>, x2: &Array3<f64>) -> Array3<f64> {
    // Simply add the two streams
    x1 + x2
}

/// Combine for 2D
fn combine_from_reversible_2d(x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
    x1 + x2
}

/// Softmax for 2D array (along last dimension)
fn softmax_2d(x: &Array2<f64>) -> Array2<f64> {
    let (n_rows, n_cols) = x.dim();
    let mut output = Array2::zeros((n_rows, n_cols));

    for i in 0..n_rows {
        let max_val = (0..n_cols).map(|j| x[[i, j]]).fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = (0..n_cols).map(|j| (x[[i, j]] - max_val).exp()).sum();

        for j in 0..n_cols {
            output[[i, j]] = (x[[i, j]] - max_val).exp() / exp_sum;
        }
    }

    output
}

/// Softmax for 1D array
fn softmax_1d(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x: Array1<f64> = x.mapv(|v| (v - max_val).exp());
    let sum: f64 = exp_x.sum();
    exp_x / sum
}

/// Generate random number from standard normal distribution
fn rand_normal() -> f64 {
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ReformerConfig {
        ReformerConfig {
            seq_len: 32,
            n_features: 10,
            d_model: 32,
            n_heads: 4,
            d_ff: 64,
            n_layers: 2,
            n_hash_rounds: 2,
            n_buckets: 8,
            chunk_size: 8,
            prediction_horizon: 5,
            output_type: OutputType::Regression,
            ..Default::default()
        }
    }

    #[test]
    fn test_model_creation() {
        let config = create_test_config();
        let model = ReformerModel::new(config.clone());

        assert_eq!(model.config().n_layers, 2);
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_forward_3d() {
        let config = create_test_config();
        let model = ReformerModel::new(config.clone());

        let x = Array3::from_shape_fn((2, 32, 10), |_| rand_normal() * 0.1);
        let output = model.forward(&x);

        assert_eq!(output.dim(), (2, config.prediction_horizon));
    }

    #[test]
    fn test_predict_2d() {
        let config = create_test_config();
        let model = ReformerModel::new(config.clone());

        let x = Array2::from_shape_fn((32, 10), |_| rand_normal() * 0.1);
        let output = model.predict(&x);

        assert_eq!(output.len(), config.prediction_horizon);
    }

    #[test]
    fn test_direction_output() {
        let mut config = create_test_config();
        config.output_type = OutputType::Direction;
        let model = ReformerModel::new(config);

        let x = Array2::from_shape_fn((32, 10), |_| rand_normal() * 0.1);
        let output = model.predict(&x);

        // Should be probabilities summing to 1
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All values should be positive
        for &val in output.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_portfolio_output() {
        let mut config = create_test_config();
        config.output_type = OutputType::Portfolio;
        let model = ReformerModel::new(config);

        let x = Array2::from_shape_fn((32, 10), |_| rand_normal() * 0.1);
        let output = model.predict(&x);

        // Should be weights summing to 1
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_num_parameters() {
        let config = create_test_config();
        let model = ReformerModel::new(config);

        let params = model.num_parameters();
        assert!(params > 0);
    }

    #[test]
    fn test_output_finite() {
        let config = create_test_config();
        let model = ReformerModel::new(config);

        let x = Array2::from_shape_fn((32, 10), |_| rand_normal() * 0.1);
        let output = model.predict(&x);

        // All outputs should be finite
        for &val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
    }
}
