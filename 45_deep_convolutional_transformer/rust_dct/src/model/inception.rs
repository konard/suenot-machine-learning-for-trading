//! Inception Convolutional Token Embedding
//!
//! Multi-scale feature extraction using parallel convolutions with different kernel sizes.

use ndarray::{Array2, Array3, Axis};

/// Inception module for multi-scale pattern extraction
#[derive(Debug, Clone)]
pub struct InceptionEmbedding {
    /// Output dimension
    d_model: usize,
    /// Kernel sizes for different branches
    kernel_sizes: Vec<usize>,
    /// Weights for each kernel size
    weights: Vec<Array3<f64>>,
    /// Biases for each branch
    biases: Vec<Array2<f64>>,
}

impl InceptionEmbedding {
    /// Create new inception embedding
    pub fn new(input_features: usize, d_model: usize) -> Self {
        let kernel_sizes = vec![1, 3, 5];
        let channels_per_branch = d_model / 4; // Each branch produces 1/4 of channels

        // Initialize weights with Xavier/Glorot initialization
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for &ks in &kernel_sizes {
            // Weight shape: (channels_per_branch, input_features, kernel_size)
            let scale = (2.0 / (input_features * ks) as f64).sqrt();
            let weight = Array3::from_shape_fn((channels_per_branch, input_features, ks), |_| {
                rand_normal() * scale
            });
            weights.push(weight);

            let bias = Array2::zeros((1, channels_per_branch));
            biases.push(bias);
        }

        // Max pool branch weight
        let scale = (2.0 / input_features as f64).sqrt();
        let mp_weight = Array3::from_shape_fn((channels_per_branch, input_features, 1), |_| {
            rand_normal() * scale
        });
        weights.push(mp_weight);
        biases.push(Array2::zeros((1, channels_per_branch)));

        Self {
            d_model,
            kernel_sizes,
            weights,
            biases,
        }
    }

    /// Forward pass through inception embedding
    /// Input: (batch, seq_len, input_features)
    /// Output: (batch, seq_len, d_model)
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, _) = x.dim();
        let mut outputs = Vec::new();

        // Apply each convolutional branch
        for (i, &ks) in self.kernel_sizes.iter().enumerate() {
            let conv_out = conv1d_same(x, &self.weights[i], ks);
            let activated = relu(&conv_out);
            outputs.push(activated);
        }

        // Max pool branch: pool then 1x1 conv
        let pooled = max_pool1d(x, 3);
        let mp_conv = conv1d_same(&pooled, &self.weights[3], 1);
        let mp_activated = relu(&mp_conv);
        outputs.push(mp_activated);

        // Concatenate all branches along feature dimension
        let channels_per_branch = self.d_model / 4;
        let mut result = Array3::zeros((batch, seq_len, self.d_model));

        for (i, output) in outputs.iter().enumerate() {
            let start = i * channels_per_branch;
            let end = start + channels_per_branch;
            for b in 0..batch {
                for t in 0..seq_len {
                    for c in 0..channels_per_branch {
                        result[[b, t, start + c]] = output[[b, t, c]];
                    }
                }
            }
        }

        result
    }
}

/// Simple pseudo-random normal distribution
fn rand_normal() -> f64 {
    // Box-Muller transform using a simple PRNG
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(12345);

    let s = SEED.fetch_add(1, Ordering::Relaxed);
    let u1 = ((s.wrapping_mul(1103515245).wrapping_add(12345) % (1 << 31)) as f64) / (1u64 << 31) as f64;
    let u2 = ((s.wrapping_mul(1103515245).wrapping_add(54321) % (1 << 31)) as f64) / (1u64 << 31) as f64;

    let u1 = u1.max(1e-10);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 1D convolution with same padding
fn conv1d_same(x: &Array3<f64>, weight: &Array3<f64>, kernel_size: usize) -> Array3<f64> {
    let (batch, seq_len, in_features) = x.dim();
    let (out_channels, w_in_features, _) = weight.dim();

    assert_eq!(in_features, w_in_features, "Input features must match weight");

    let pad = kernel_size / 2;
    let mut output = Array3::zeros((batch, seq_len, out_channels));

    for b in 0..batch {
        for t in 0..seq_len {
            for oc in 0..out_channels {
                let mut sum = 0.0;
                for ic in 0..in_features {
                    for k in 0..kernel_size {
                        let idx = t as isize + k as isize - pad as isize;
                        if idx >= 0 && (idx as usize) < seq_len {
                            sum += x[[b, idx as usize, ic]] * weight[[oc, ic, k]];
                        }
                    }
                }
                output[[b, t, oc]] = sum;
            }
        }
    }

    output
}

/// Max pooling 1D with same output size
fn max_pool1d(x: &Array3<f64>, kernel_size: usize) -> Array3<f64> {
    let (batch, seq_len, features) = x.dim();
    let pad = kernel_size / 2;
    let mut output = Array3::zeros((batch, seq_len, features));

    for b in 0..batch {
        for t in 0..seq_len {
            for f in 0..features {
                let mut max_val = f64::NEG_INFINITY;
                for k in 0..kernel_size {
                    let idx = t as isize + k as isize - pad as isize;
                    if idx >= 0 && (idx as usize) < seq_len {
                        max_val = max_val.max(x[[b, idx as usize, f]]);
                    }
                }
                output[[b, t, f]] = max_val;
            }
        }
    }

    output
}

/// ReLU activation
fn relu(x: &Array3<f64>) -> Array3<f64> {
    x.mapv(|v| v.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inception_embedding() {
        let inception = InceptionEmbedding::new(13, 64);
        let input = Array3::from_shape_fn((2, 30, 13), |_| rand_normal());

        let output = inception.forward(&input);
        assert_eq!(output.dim(), (2, 30, 64));
    }
}
