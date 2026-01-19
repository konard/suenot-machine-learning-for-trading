//! FNet Encoder Block
//!
//! Single encoder block combining Fourier layer and feed-forward network.

use ndarray::{Array2, Array3};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::fourier::FourierLayer;

/// Feed-forward network weights.
pub struct FeedForward {
    /// First linear layer weights [d_model, d_ff]
    pub w1: Array2<f64>,
    /// First linear layer bias [d_ff]
    pub b1: Vec<f64>,
    /// Second linear layer weights [d_ff, d_model]
    pub w2: Array2<f64>,
    /// Second linear layer bias [d_model]
    pub b2: Vec<f64>,
    /// Dropout probability
    pub dropout: f64,
}

impl FeedForward {
    /// Create a new feed-forward network with Xavier initialization.
    pub fn new(d_model: usize, d_ff: usize, dropout: f64) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let std1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let std2 = (2.0 / (d_ff + d_model) as f64).sqrt();

        let normal1 = Normal::new(0.0, std1).unwrap();
        let normal2 = Normal::new(0.0, std2).unwrap();

        let w1 = Array2::from_shape_fn((d_model, d_ff), |_| normal1.sample(&mut rng));
        let b1 = vec![0.0; d_ff];
        let w2 = Array2::from_shape_fn((d_ff, d_model), |_| normal2.sample(&mut rng));
        let b2 = vec![0.0; d_model];

        Self { w1, b1, w2, b2, dropout }
    }

    /// Forward pass through feed-forward network.
    ///
    /// Structure: Linear -> GELU -> Dropout -> Linear -> Dropout
    pub fn forward(&self, x: &Array3<f64>, training: bool) -> Array3<f64> {
        let (batch, seq_len, d_model) = x.dim();
        let d_ff = self.w1.dim().1;

        let mut output = Array3::<f64>::zeros((batch, seq_len, d_model));

        for b in 0..batch {
            for t in 0..seq_len {
                // First linear: [d_model] -> [d_ff]
                let mut hidden = vec![0.0; d_ff];
                for j in 0..d_ff {
                    let mut sum = self.b1[j];
                    for i in 0..d_model {
                        sum += x[[b, t, i]] * self.w1[[i, j]];
                    }
                    // GELU activation
                    hidden[j] = gelu(sum);
                }

                // Dropout on hidden layer
                if training {
                    apply_dropout_inplace(&mut hidden, self.dropout);
                }

                // Second linear: [d_ff] -> [d_model]
                for i in 0..d_model {
                    let mut sum = self.b2[i];
                    for j in 0..d_ff {
                        sum += hidden[j] * self.w2[[j, i]];
                    }
                    output[[b, t, i]] = sum;
                }
            }
        }

        // Dropout on output
        if training {
            apply_dropout_3d(&mut output, self.dropout);
        }

        output
    }
}

/// Layer normalization parameters.
pub struct LayerNorm {
    /// Scale parameter (gamma)
    pub gamma: Vec<f64>,
    /// Shift parameter (beta)
    pub beta: Vec<f64>,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl LayerNorm {
    /// Create new layer normalization.
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: vec![1.0; d_model],
            beta: vec![0.0; d_model],
            eps: 1e-5,
        }
    }

    /// Apply layer normalization.
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, d_model) = x.dim();
        let mut output = Array3::<f64>::zeros((batch, seq_len, d_model));

        for b in 0..batch {
            for t in 0..seq_len {
                // Calculate mean and variance over d_model dimension
                let mut mean = 0.0;
                for i in 0..d_model {
                    mean += x[[b, t, i]];
                }
                mean /= d_model as f64;

                let mut var = 0.0;
                for i in 0..d_model {
                    let diff = x[[b, t, i]] - mean;
                    var += diff * diff;
                }
                var /= d_model as f64;

                // Normalize
                let std = (var + self.eps).sqrt();
                for i in 0..d_model {
                    output[[b, t, i]] = self.gamma[i] * (x[[b, t, i]] - mean) / std + self.beta[i];
                }
            }
        }

        output
    }
}

/// FNet encoder block.
///
/// Structure:
/// 1. Fourier Transform + Residual + LayerNorm
/// 2. Feed-Forward + Residual + LayerNorm
pub struct FNetEncoderBlock {
    fourier: FourierLayer,
    norm1: LayerNorm,
    ff: FeedForward,
    norm2: LayerNorm,
}

impl FNetEncoderBlock {
    /// Create a new encoder block.
    pub fn new(d_model: usize, d_ff: usize, dropout: f64) -> Self {
        Self {
            fourier: FourierLayer::new(),
            norm1: LayerNorm::new(d_model),
            ff: FeedForward::new(d_model, d_ff, dropout),
            norm2: LayerNorm::new(d_model),
        }
    }

    /// Forward pass through encoder block.
    pub fn forward(&mut self, x: &Array3<f64>, training: bool) -> Array3<f64> {
        // Fourier sublayer with residual
        let fourier_out = self.fourier.forward(x);
        let residual1 = add_arrays(x, &fourier_out);
        let normed1 = self.norm1.forward(&residual1);

        // Feed-forward sublayer with residual
        let ff_out = self.ff.forward(&normed1, training);
        let residual2 = add_arrays(&normed1, &ff_out);
        let normed2 = self.norm2.forward(&residual2);

        normed2
    }

    /// Forward pass with frequency map output.
    pub fn forward_with_frequencies(
        &mut self,
        x: &Array3<f64>,
        training: bool
    ) -> (Array3<f64>, Array3<f64>) {
        let fourier_out = self.fourier.forward(x);
        let residual1 = add_arrays(x, &fourier_out);
        let normed1 = self.norm1.forward(&residual1);

        let ff_out = self.ff.forward(&normed1, training);
        let residual2 = add_arrays(&normed1, &ff_out);
        let normed2 = self.norm2.forward(&residual2);

        (normed2, fourier_out)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// GELU activation function.
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
}

/// Apply dropout in place.
fn apply_dropout_inplace(x: &mut [f64], p: f64) {
    // Guard against p >= 1.0 to avoid NaN/inf from 1/(1-p)
    if p <= 0.0 {
        return; // No dropout
    }
    if p >= 1.0 {
        // Drop everything
        for val in x.iter_mut() {
            *val = 0.0;
        }
        return;
    }

    let mut rng = rand::thread_rng();
    let scale = 1.0 / (1.0 - p);

    for val in x.iter_mut() {
        if rng.gen::<f64>() < p {
            *val = 0.0;
        } else {
            *val *= scale;
        }
    }
}

/// Apply dropout to 3D array.
fn apply_dropout_3d(x: &mut Array3<f64>, p: f64) {
    // Guard against p >= 1.0 to avoid NaN/inf from 1/(1-p)
    if p <= 0.0 {
        return; // No dropout
    }
    if p >= 1.0 {
        // Drop everything
        for val in x.iter_mut() {
            *val = 0.0;
        }
        return;
    }

    let mut rng = rand::thread_rng();
    let scale = 1.0 / (1.0 - p);

    for val in x.iter_mut() {
        if rng.gen::<f64>() < p {
            *val = 0.0;
        } else {
            *val *= scale;
        }
    }
}

/// Element-wise addition of two 3D arrays.
fn add_arrays(a: &Array3<f64>, b: &Array3<f64>) -> Array3<f64> {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let norm = LayerNorm::new(4);
        let input = Array3::from_shape_fn((2, 3, 4), |(b, t, d)| {
            (b * 12 + t * 4 + d) as f64
        });

        let output = norm.forward(&input);

        // Check output shape
        assert_eq!(output.dim(), input.dim());

        // After normalization, each position should have mean ~0 and std ~1
        for b in 0..2 {
            for t in 0..3 {
                let mut mean = 0.0;
                for d in 0..4 {
                    mean += output[[b, t, d]];
                }
                mean /= 4.0;
                assert!((mean.abs()) < 1e-5, "Mean should be ~0");
            }
        }
    }

    #[test]
    fn test_feed_forward() {
        let ff = FeedForward::new(4, 16, 0.0);
        let input = Array3::from_shape_fn((2, 3, 4), |(_, _, _)| 0.5);

        let output = ff.forward(&input, false);
        assert_eq!(output.dim(), input.dim());
    }

    #[test]
    fn test_encoder_block() {
        let mut block = FNetEncoderBlock::new(4, 16, 0.0);
        let input = Array3::from_shape_fn((2, 8, 4), |(_, i, _)| {
            (i as f64 / 8.0).sin()
        });

        let output = block.forward(&input, false);
        assert_eq!(output.dim(), input.dim());
    }

    #[test]
    fn test_gelu() {
        // GELU(0) ≈ 0
        assert!((gelu(0.0)).abs() < 1e-5);

        // GELU is positive for positive input
        assert!(gelu(1.0) > 0.0);

        // GELU is approximately linear for large positive values
        assert!((gelu(5.0) - 5.0).abs() < 0.1);
    }
}
