//! Dense Layer Implementation
//!
//! A single layer within a DenseNet dense block.
//! Each layer receives feature maps from ALL previous layers.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// A single dense layer: BN → ReLU → Conv → Dropout
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Input channels (sum of all previous outputs)
    pub in_channels: usize,

    /// Output channels (growth rate)
    pub out_channels: usize,

    /// Convolution kernel size
    pub kernel_size: usize,

    /// Dropout rate
    pub dropout: f64,

    /// Whether to use bottleneck (1x1 conv before 3x1)
    pub use_bottleneck: bool,

    /// Bottleneck intermediate channels
    pub bottleneck_channels: usize,

    // Learnable parameters
    /// Batch normalization gamma (scale)
    pub bn_gamma: Array1<f64>,

    /// Batch normalization beta (shift)
    pub bn_beta: Array1<f64>,

    /// Running mean for batch norm
    pub bn_running_mean: Array1<f64>,

    /// Running variance for batch norm
    pub bn_running_var: Array1<f64>,

    /// Bottleneck convolution weights (if used)
    pub bottleneck_weights: Option<Array2<f64>>,

    /// Main convolution weights
    pub conv_weights: Array2<f64>,

    /// Convolution bias
    pub conv_bias: Array1<f64>,
}

impl DenseLayer {
    /// Create a new dense layer
    pub fn new(
        in_channels: usize,
        growth_rate: usize,
        kernel_size: usize,
        dropout: f64,
        use_bottleneck: bool,
        bottleneck_factor: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let bottleneck_channels = if use_bottleneck {
            bottleneck_factor * growth_rate
        } else {
            in_channels
        };

        // Initialize batch norm parameters
        let bn_gamma = Array1::ones(in_channels);
        let bn_beta = Array1::zeros(in_channels);
        let bn_running_mean = Array1::zeros(in_channels);
        let bn_running_var = Array1::ones(in_channels);

        // Initialize bottleneck weights if needed
        let bottleneck_weights = if use_bottleneck {
            let weights: Vec<f64> = (0..in_channels * bottleneck_channels)
                .map(|_| rng.sample(normal))
                .collect();
            Some(Array2::from_shape_vec((bottleneck_channels, in_channels), weights).unwrap())
        } else {
            None
        };

        // Initialize main convolution weights
        let conv_in = if use_bottleneck { bottleneck_channels } else { in_channels };
        let weights: Vec<f64> = (0..conv_in * growth_rate * kernel_size)
            .map(|_| rng.sample(normal))
            .collect();
        let conv_weights = Array2::from_shape_vec(
            (growth_rate, conv_in * kernel_size),
            weights
        ).unwrap();

        let conv_bias = Array1::zeros(growth_rate);

        Self {
            in_channels,
            out_channels: growth_rate,
            kernel_size,
            dropout,
            use_bottleneck,
            bottleneck_channels,
            bn_gamma,
            bn_beta,
            bn_running_mean,
            bn_running_var,
            bottleneck_weights,
            conv_weights,
            conv_bias,
        }
    }

    /// Forward pass through the layer
    ///
    /// Input: [batch_size, in_channels, sequence_length]
    /// Output: [batch_size, growth_rate, sequence_length]
    pub fn forward(&self, x: &Array2<f64>, training: bool) -> Array2<f64> {
        // x shape: [channels, sequence_length]

        // 1. Batch Normalization
        let normalized = self.batch_norm(x);

        // 2. ReLU activation
        let activated = normalized.mapv(|v| v.max(0.0));

        // 3. Bottleneck 1x1 convolution (if used)
        let bottleneck_out = if let Some(ref bn_weights) = self.bottleneck_weights {
            bn_weights.dot(&activated)
        } else {
            activated
        };

        // 4. ReLU after bottleneck
        let bn_activated = bottleneck_out.mapv(|v| v.max(0.0));

        // 5. 3x1 Convolution (simplified as matrix multiplication for 1D)
        let conv_out = self.conv1d(&bn_activated);

        // 6. Dropout (only during training)
        if training && self.dropout > 0.0 {
            self.apply_dropout(&conv_out)
        } else {
            conv_out
        }
    }

    /// Apply batch normalization
    fn batch_norm(&self, x: &Array2<f64>) -> Array2<f64> {
        let eps = 1e-5;
        let mut result = x.clone();

        for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
            if i < self.in_channels {
                let mean = self.bn_running_mean[i];
                let var = self.bn_running_var[i];
                let gamma = self.bn_gamma[i];
                let beta = self.bn_beta[i];

                row.mapv_inplace(|v| {
                    gamma * (v - mean) / (var + eps).sqrt() + beta
                });
            }
        }

        result
    }

    /// Apply 1D convolution
    fn conv1d(&self, x: &Array2<f64>) -> Array2<f64> {
        let (in_ch, seq_len) = x.dim();
        let out_ch = self.out_channels;
        let k = self.kernel_size;
        let padding = k / 2;

        let mut output = Array2::zeros((out_ch, seq_len));

        for out_c in 0..out_ch {
            for pos in 0..seq_len {
                let mut sum = self.conv_bias[out_c];

                for in_c in 0..in_ch.min(self.bottleneck_channels) {
                    for ki in 0..k {
                        let input_pos = pos as i64 + ki as i64 - padding as i64;
                        if input_pos >= 0 && input_pos < seq_len as i64 {
                            let weight_idx = in_c * k + ki;
                            if weight_idx < self.conv_weights.ncols() {
                                sum += x[[in_c, input_pos as usize]]
                                    * self.conv_weights[[out_c, weight_idx]];
                            }
                        }
                    }
                }

                output[[out_c, pos]] = sum;
            }
        }

        output
    }

    /// Apply dropout
    fn apply_dropout(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.dropout);

        x.mapv(|v| {
            if rng.gen::<f64>() < self.dropout {
                0.0
            } else {
                v * scale
            }
        })
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        // Batch norm
        count += 2 * self.in_channels; // gamma, beta

        // Bottleneck weights
        if let Some(ref w) = self.bottleneck_weights {
            count += w.len();
        }

        // Conv weights and bias
        count += self.conv_weights.len();
        count += self.conv_bias.len();

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer_creation() {
        let layer = DenseLayer::new(64, 32, 3, 0.2, true, 4);
        assert_eq!(layer.in_channels, 64);
        assert_eq!(layer.out_channels, 32);
        assert!(layer.bottleneck_weights.is_some());
    }

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer::new(32, 16, 3, 0.0, false, 4);
        let input = Array2::ones((32, 64));
        let output = layer.forward(&input, false);
        assert_eq!(output.dim(), (16, 64));
    }

    #[test]
    fn test_layer_parameters() {
        let layer = DenseLayer::new(64, 32, 3, 0.2, true, 4);
        assert!(layer.num_parameters() > 0);
    }
}
