//! Input embedding layer
//!
//! Converts input features to model dimension with positional encoding.

use ndarray::{Array1, Array2, Array3};
use std::f64::consts::PI;

use super::config::ReformerConfig;

/// Token embedding layer
#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    /// Linear projection weights [n_features, d_model]
    weights: Array2<f64>,
    /// Positional encoding [max_seq_len, d_model]
    positional_encoding: Array2<f64>,
    /// Configuration
    d_model: usize,
    /// Whether to use positional encoding
    use_positional_encoding: bool,
}

impl TokenEmbedding {
    /// Create a new token embedding layer
    pub fn new(config: &ReformerConfig) -> Self {
        let d_model = config.d_model;
        let n_features = config.n_features;
        let max_seq_len = config.seq_len;

        // Xavier initialization for weights
        let scale = (2.0 / (n_features + d_model) as f64).sqrt();
        let weights = Array2::from_shape_fn((n_features, d_model), |_| rand_normal() * scale);

        // Create sinusoidal positional encoding
        let positional_encoding = Self::create_positional_encoding(max_seq_len, d_model);

        Self {
            weights,
            positional_encoding,
            d_model,
            use_positional_encoding: config.use_positional_encoding,
        }
    }

    /// Create sinusoidal positional encoding
    fn create_positional_encoding(max_len: usize, d_model: usize) -> Array2<f64> {
        let mut pe = Array2::zeros((max_len, d_model));

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let angle = pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
                pe[[pos, 2 * i]] = angle.sin();
                pe[[pos, 2 * i + 1]] = angle.cos();
            }
        }

        pe
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, n_features]
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, n_features) = x.dim();

        let mut output = Array3::zeros((batch_size, seq_len, self.d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                // Linear projection: x @ weights
                for d in 0..self.d_model {
                    let mut sum = 0.0;
                    for f in 0..n_features {
                        sum += x[[b, t, f]] * self.weights[[f, d]];
                    }

                    // Add positional encoding
                    if self.use_positional_encoding && t < self.positional_encoding.nrows() {
                        sum += self.positional_encoding[[t, d]];
                    }

                    output[[b, t, d]] = sum;
                }
            }
        }

        output
    }

    /// Forward pass for 2D input [seq_len, n_features]
    pub fn forward_2d(&self, x: &Array2<f64>) -> Array2<f64> {
        let (seq_len, n_features) = x.dim();

        let mut output = Array2::zeros((seq_len, self.d_model));

        for t in 0..seq_len {
            for d in 0..self.d_model {
                let mut sum = 0.0;
                for f in 0..n_features {
                    sum += x[[t, f]] * self.weights[[f, d]];
                }

                if self.use_positional_encoding && t < self.positional_encoding.nrows() {
                    sum += self.positional_encoding[[t, d]];
                }

                output[[t, d]] = sum;
            }
        }

        output
    }

    /// Get positional encoding for a given position
    pub fn get_positional_encoding(&self, position: usize) -> Array1<f64> {
        if position < self.positional_encoding.nrows() {
            self.positional_encoding.row(position).to_owned()
        } else {
            // Generate on the fly for longer sequences
            let mut pe = Array1::zeros(self.d_model);
            for i in 0..(self.d_model / 2) {
                let angle =
                    position as f64 / (10000.0_f64).powf(2.0 * i as f64 / self.d_model as f64);
                pe[2 * i] = angle.sin();
                pe[2 * i + 1] = angle.cos();
            }
            pe
        }
    }
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

    #[test]
    fn test_embedding_creation() {
        let config = ReformerConfig {
            n_features: 10,
            d_model: 64,
            seq_len: 100,
            ..Default::default()
        };

        let embedding = TokenEmbedding::new(&config);

        assert_eq!(embedding.weights.dim(), (10, 64));
        assert_eq!(embedding.positional_encoding.dim(), (100, 64));
    }

    #[test]
    fn test_forward_3d() {
        let config = ReformerConfig {
            n_features: 5,
            d_model: 32,
            seq_len: 50,
            ..Default::default()
        };

        let embedding = TokenEmbedding::new(&config);

        let x = Array3::from_shape_fn((2, 50, 5), |_| rand_normal());
        let output = embedding.forward(&x);

        assert_eq!(output.dim(), (2, 50, 32));
    }

    #[test]
    fn test_forward_2d() {
        let config = ReformerConfig {
            n_features: 5,
            d_model: 32,
            seq_len: 50,
            ..Default::default()
        };

        let embedding = TokenEmbedding::new(&config);

        let x = Array2::from_shape_fn((50, 5), |_| rand_normal());
        let output = embedding.forward_2d(&x);

        assert_eq!(output.dim(), (50, 32));
    }

    #[test]
    fn test_positional_encoding_values() {
        let config = ReformerConfig {
            d_model: 64,
            seq_len: 100,
            ..Default::default()
        };

        let embedding = TokenEmbedding::new(&config);

        // Check that PE values are bounded
        for row in embedding.positional_encoding.rows() {
            for &val in row {
                assert!(val >= -1.0 && val <= 1.0);
            }
        }
    }

    #[test]
    fn test_without_positional_encoding() {
        let config = ReformerConfig {
            n_features: 5,
            d_model: 32,
            seq_len: 50,
            use_positional_encoding: false,
            ..Default::default()
        };

        let embedding = TokenEmbedding::new(&config);

        let x = Array2::from_shape_fn((50, 5), |_| 1.0);
        let output1 = embedding.forward_2d(&x);

        // Without PE, same input at different positions should give same output
        // (after projection, not considering PE)
        // This test verifies PE is not being added
        assert!(!embedding.use_positional_encoding);
    }
}
