//! Universal encoder for task-agnostic representation learning
//!
//! The encoder maps raw market features into a shared representation space
//! that is useful across multiple downstream trading tasks.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for the universal encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output representation dimension
    pub repr_dim: usize,
    /// Dropout rate for regularization
    pub dropout_rate: f64,
    /// Whether to use batch normalization
    pub use_batch_norm: bool,
    /// Whether to use residual connections
    pub use_residual: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            input_dim: 20,
            hidden_dims: vec![64, 32],
            repr_dim: 16,
            dropout_rate: 0.1,
            use_batch_norm: true,
            use_residual: true,
        }
    }
}

/// Layer weights for a dense layer
#[derive(Debug, Clone)]
struct DenseLayer {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization
    fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);

        Self { weights, bias }
    }

    /// Forward pass through the dense layer
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let output = input.dot(&self.weights);
        output + &self.bias
    }
}

/// Universal encoder that produces task-agnostic representations
pub struct UniversalEncoder {
    /// Configuration
    pub config: EncoderConfig,
    /// Dense layers
    layers: Vec<DenseLayer>,
    /// Projection layer to representation space
    projection: DenseLayer,
    /// Running mean for batch normalization
    bn_means: Vec<Array1<f64>>,
    /// Running variance for batch normalization
    #[allow(dead_code)]
    bn_vars: Vec<Array1<f64>>,
}

impl UniversalEncoder {
    /// Create a new universal encoder
    pub fn new(config: EncoderConfig) -> Self {
        let mut layers = Vec::new();
        let mut bn_means = Vec::new();
        let mut bn_vars = Vec::new();

        let mut prev_dim = config.input_dim;
        for (i, &hidden_dim) in config.hidden_dims.iter().enumerate() {
            layers.push(DenseLayer::new(prev_dim, hidden_dim, (i * 42 + 7) as u64));
            if config.use_batch_norm {
                bn_means.push(Array1::zeros(hidden_dim));
                bn_vars.push(Array1::ones(hidden_dim));
            }
            prev_dim = hidden_dim;
        }

        let projection = DenseLayer::new(prev_dim, config.repr_dim, 999);

        Self {
            config,
            layers,
            projection,
            bn_means,
            bn_vars,
        }
    }

    /// Forward pass: raw features -> task-agnostic representation
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let mut h = layer.forward(&x);

            // Batch normalization
            if self.config.use_batch_norm && i < self.bn_means.len() {
                h = self.batch_normalize(&h, i);
            }

            // ReLU activation
            h.mapv_inplace(|v| v.max(0.0));

            // Residual connection if dimensions match
            if self.config.use_residual && x.ncols() == h.ncols() {
                h = &h + &x;
            }

            x = h;
        }

        // Project to representation space
        let repr = self.projection.forward(&x);

        // L2 normalize representations
        self.l2_normalize(&repr)
    }

    /// Apply batch normalization
    fn batch_normalize(&self, input: &Array2<f64>, layer_idx: usize) -> Array2<f64> {
        let mean = input.mean_axis(Axis(0)).unwrap();
        let var = input.var_axis(Axis(0), 0.0);
        let eps = 1e-5;

        let mut output = input.clone();
        for mut row in output.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                *val = (*val - mean[j]) / (var[j] + eps).sqrt();
            }
        }
        let _ = layer_idx; // Used for running stats in training mode
        output
    }

    /// L2 normalize each row
    fn l2_normalize(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = input.clone();
        for mut row in output.rows_mut() {
            let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
            row.mapv_inplace(|v| v / norm);
        }
        output
    }

    /// Get the output representation dimension
    pub fn repr_dim(&self) -> usize {
        self.config.repr_dim
    }
}

use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_forward() {
        let config = EncoderConfig {
            input_dim: 10,
            hidden_dims: vec![32, 16],
            repr_dim: 8,
            ..Default::default()
        };
        let encoder = UniversalEncoder::new(config);
        // Use varied input (constant input gets zeroed by batch norm)
        let input = Array2::from_shape_fn((5, 10), |(i, j)| {
            (i as f64 * 0.3 + j as f64 * 0.7 + 1.0) / 10.0
        });
        let output = encoder.forward(&input);

        assert_eq!(output.shape(), &[5, 8]);

        // Check L2 normalization: each row should have unit norm
        for row in output.rows() {
            let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 0.01, "norm was {}", norm);
        }
    }
}
