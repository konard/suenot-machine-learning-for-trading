//! Transition Layer Implementation
//!
//! Transition layers reduce feature map dimensions between dense blocks.
//! They apply: BN → ReLU → 1x1 Conv → 2x2 AvgPool

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// Transition layer for dimensionality reduction
///
/// Reduces channels by compression factor θ and halves spatial dimensions
#[derive(Debug, Clone)]
pub struct TransitionLayer {
    /// Input channels
    pub in_channels: usize,

    /// Output channels (floor(in_channels * compression))
    pub out_channels: usize,

    /// Compression factor (0 < θ <= 1)
    pub compression: f64,

    /// Pool size for average pooling
    pub pool_size: usize,

    // Learnable parameters
    /// Batch norm gamma
    pub bn_gamma: Array1<f64>,

    /// Batch norm beta
    pub bn_beta: Array1<f64>,

    /// Running mean
    pub bn_running_mean: Array1<f64>,

    /// Running variance
    pub bn_running_var: Array1<f64>,

    /// 1x1 convolution weights
    pub conv_weights: Array2<f64>,

    /// Convolution bias
    pub conv_bias: Array1<f64>,
}

impl TransitionLayer {
    /// Create a new transition layer
    pub fn new(in_channels: usize, compression: f64, pool_size: usize) -> Self {
        let out_channels = (in_channels as f64 * compression).floor() as usize;
        let out_channels = out_channels.max(1); // At least 1 channel

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        // Batch norm parameters
        let bn_gamma = Array1::ones(in_channels);
        let bn_beta = Array1::zeros(in_channels);
        let bn_running_mean = Array1::zeros(in_channels);
        let bn_running_var = Array1::ones(in_channels);

        // 1x1 convolution weights (channel reduction)
        let weights: Vec<f64> = (0..out_channels * in_channels)
            .map(|_| rng.sample(normal))
            .collect();
        let conv_weights = Array2::from_shape_vec((out_channels, in_channels), weights).unwrap();

        let conv_bias = Array1::zeros(out_channels);

        Self {
            in_channels,
            out_channels,
            compression,
            pool_size,
            bn_gamma,
            bn_beta,
            bn_running_mean,
            bn_running_var,
            conv_weights,
            conv_bias,
        }
    }

    /// Forward pass through transition layer
    ///
    /// Input: [in_channels, sequence_length]
    /// Output: [out_channels, sequence_length / pool_size]
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // 1. Batch Normalization
        let normalized = self.batch_norm(x);

        // 2. ReLU activation
        let activated = normalized.mapv(|v| v.max(0.0));

        // 3. 1x1 Convolution (channel reduction)
        let conv_out = self.conv1x1(&activated);

        // 4. Average Pooling
        self.avg_pool(&conv_out)
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

    /// Apply 1x1 convolution
    fn conv1x1(&self, x: &Array2<f64>) -> Array2<f64> {
        let (_, seq_len) = x.dim();
        let mut output = Array2::zeros((self.out_channels, seq_len));

        for (out_c, mut out_row) in output.axis_iter_mut(Axis(0)).enumerate() {
            for pos in 0..seq_len {
                let mut sum = self.conv_bias[out_c];

                for in_c in 0..self.in_channels.min(x.dim().0) {
                    sum += x[[in_c, pos]] * self.conv_weights[[out_c, in_c]];
                }

                out_row[pos] = sum;
            }
        }

        output
    }

    /// Apply average pooling
    fn avg_pool(&self, x: &Array2<f64>) -> Array2<f64> {
        let (channels, seq_len) = x.dim();
        let new_len = seq_len / self.pool_size;

        if new_len == 0 {
            // If sequence is too short, return as-is
            return x.clone();
        }

        let mut output = Array2::zeros((channels, new_len));

        for c in 0..channels {
            for i in 0..new_len {
                let start = i * self.pool_size;
                let end = (start + self.pool_size).min(seq_len);

                let sum: f64 = (start..end).map(|j| x[[c, j]]).sum();
                output[[c, i]] = sum / (end - start) as f64;
            }
        }

        output
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        // BN: gamma + beta
        // Conv: weights + bias
        2 * self.in_channels + self.conv_weights.len() + self.conv_bias.len()
    }

    /// Get layer info
    pub fn info(&self) -> TransitionInfo {
        TransitionInfo {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            compression: self.compression,
            pool_size: self.pool_size,
            parameters: self.num_parameters(),
        }
    }
}

/// Information about a transition layer
#[derive(Debug, Clone)]
pub struct TransitionInfo {
    pub in_channels: usize,
    pub out_channels: usize,
    pub compression: f64,
    pub pool_size: usize,
    pub parameters: usize,
}

impl std::fmt::Display for TransitionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Transition(in={}, out={}, θ={:.2}, pool={}, params={})",
            self.in_channels,
            self.out_channels,
            self.compression,
            self.pool_size,
            self.parameters
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_creation() {
        let trans = TransitionLayer::new(256, 0.5, 2);
        assert_eq!(trans.in_channels, 256);
        assert_eq!(trans.out_channels, 128);
    }

    #[test]
    fn test_transition_forward() {
        let trans = TransitionLayer::new(64, 0.5, 2);
        let input = Array2::ones((64, 32));
        let output = trans.forward(&input);

        assert_eq!(output.dim().0, 32); // Channels reduced by 0.5
        assert_eq!(output.dim().1, 16); // Sequence reduced by pool_size
    }

    #[test]
    fn test_compression_factor() {
        for compression in [0.25, 0.5, 0.75, 1.0] {
            let trans = TransitionLayer::new(100, compression, 2);
            let expected = (100.0 * compression).floor() as usize;
            assert_eq!(trans.out_channels, expected);
        }
    }

    #[test]
    fn test_transition_info() {
        let trans = TransitionLayer::new(128, 0.5, 2);
        let info = trans.info();
        assert_eq!(info.in_channels, 128);
        assert_eq!(info.out_channels, 64);
    }
}
