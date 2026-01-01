//! Basic neural network layers for ConvNeXt

use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// 1D Convolution layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conv1d {
    /// Weight tensor [out_channels, in_channels/groups, kernel_size]
    pub weight: Array3<f64>,
    /// Bias vector [out_channels]
    pub bias: Array1<f64>,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Number of groups for grouped convolution
    pub groups: usize,
}

impl Conv1d {
    /// Create a new Conv1d layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weight = Array3::from_shape_fn(
            (out_channels, in_channels / groups, kernel_size),
            |_| normal.sample(&mut rng),
        );
        let bias = Array1::zeros(out_channels);

        Self {
            weight,
            bias,
            stride,
            padding,
            groups,
        }
    }

    /// Create a depthwise convolution (groups = in_channels = out_channels)
    pub fn depthwise(channels: usize, kernel_size: usize, padding: usize) -> Self {
        Self::new(channels, channels, kernel_size, 1, padding, channels)
    }

    /// Create a pointwise convolution (1x1 conv)
    pub fn pointwise(in_channels: usize, out_channels: usize) -> Self {
        Self::new(in_channels, out_channels, 1, 1, 0, 1)
    }

    /// Forward pass
    /// Input shape: [batch, in_channels, length]
    /// Output shape: [batch, out_channels, new_length]
    pub fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let (batch, in_channels, length) = input.dim();
        let out_channels = self.weight.dim().0;
        let kernel_size = self.weight.dim().2;

        // Calculate output length
        let out_length = (length + 2 * self.padding - kernel_size) / self.stride + 1;

        let mut output = Array3::zeros((batch, out_channels, out_length));

        // Apply padding
        let padded = if self.padding > 0 {
            let mut padded = Array3::zeros((batch, in_channels, length + 2 * self.padding));
            padded
                .slice_mut(ndarray::s![.., .., self.padding..self.padding + length])
                .assign(input);
            padded
        } else {
            input.clone()
        };

        // Grouped convolution
        let channels_per_group = in_channels / self.groups;
        let out_channels_per_group = out_channels / self.groups;

        for b in 0..batch {
            for g in 0..self.groups {
                let in_start = g * channels_per_group;
                let out_start = g * out_channels_per_group;

                for oc in 0..out_channels_per_group {
                    let out_idx = out_start + oc;

                    for ol in 0..out_length {
                        let in_start_l = ol * self.stride;
                        let mut sum = self.bias[out_idx];

                        for ic in 0..channels_per_group {
                            for k in 0..kernel_size {
                                sum += padded[[b, in_start + ic, in_start_l + k]]
                                    * self.weight[[out_idx, ic, k]];
                            }
                        }

                        output[[b, out_idx, ol]] = sum;
                    }
                }
            }
        }

        output
    }
}

/// Layer Normalization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNorm {
    /// Normalized shape (number of features)
    pub normalized_shape: usize,
    /// Learnable scale parameter
    pub gamma: Array1<f64>,
    /// Learnable shift parameter
    pub beta: Array1<f64>,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl LayerNorm {
    /// Create a new LayerNorm layer
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            gamma: Array1::ones(normalized_shape),
            beta: Array1::zeros(normalized_shape),
            eps: 1e-6,
        }
    }

    /// Forward pass for 2D input [batch, features]
    pub fn forward_2d(&self, input: &Array2<f64>) -> Array2<f64> {
        let mean = input.mean_axis(Axis(1)).unwrap();
        let variance = input.var_axis(Axis(1), 0.0);

        let mut output = input.clone();

        for (i, mut row) in output.axis_iter_mut(Axis(0)).enumerate() {
            let std = (variance[i] + self.eps).sqrt();
            for (j, val) in row.iter_mut().enumerate() {
                *val = (*val - mean[i]) / std * self.gamma[j] + self.beta[j];
            }
        }

        output
    }

    /// Forward pass for 3D input [batch, channels, length]
    /// Normalizes over the channels dimension
    pub fn forward_3d(&self, input: &Array3<f64>) -> Array3<f64> {
        let (batch, channels, length) = input.dim();
        let mut output = input.clone();

        for b in 0..batch {
            for l in 0..length {
                // Get slice for this position
                let slice: Vec<f64> = (0..channels).map(|c| input[[b, c, l]]).collect();

                // Compute mean and variance
                let mean = slice.iter().sum::<f64>() / channels as f64;
                let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / channels as f64;
                let std = (variance + self.eps).sqrt();

                // Normalize
                for c in 0..channels {
                    output[[b, c, l]] = (input[[b, c, l]] - mean) / std * self.gamma[c] + self.beta[c];
                }
            }
        }

        output
    }
}

/// Linear (fully connected) layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    pub weight: Array2<f64>,
    /// Bias vector [out_features]
    pub bias: Array1<f64>,
}

impl Linear {
    /// Create a new Linear layer
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / in_features as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weight = Array2::from_shape_fn((out_features, in_features), |_| normal.sample(&mut rng));
        let bias = Array1::zeros(out_features);

        Self { weight, bias }
    }

    /// Forward pass
    /// Input shape: [batch, in_features]
    /// Output shape: [batch, out_features]
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.dot(&self.weight.t()) + &self.bias
    }
}

/// Drop Path (Stochastic Depth) for regularization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DropPath {
    /// Drop probability
    pub drop_prob: f64,
    /// Whether in training mode
    pub training: bool,
}

impl DropPath {
    /// Create a new DropPath layer
    pub fn new(drop_prob: f64) -> Self {
        Self {
            drop_prob,
            training: true,
        }
    }

    /// Forward pass
    pub fn forward<T: Clone>(&self, x: T, residual: &T) -> T
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<f64, Output = T>,
    {
        if !self.training || self.drop_prob == 0.0 {
            return x + residual.clone();
        }

        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.drop_prob {
            residual.clone()
        } else {
            let scale = 1.0 / (1.0 - self.drop_prob);
            x * scale + residual.clone()
        }
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// GELU activation function
pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Apply GELU to array
pub fn gelu_array(input: &Array3<f64>) -> Array3<f64> {
    input.mapv(gelu)
}

/// Softmax function
pub fn softmax(input: &Array2<f64>) -> Array2<f64> {
    let max = input.map_axis(Axis(1), |row| {
        row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });

    let mut exp_input = input.clone();
    for (i, mut row) in exp_input.axis_iter_mut(Axis(0)).enumerate() {
        for val in row.iter_mut() {
            *val = (*val - max[i]).exp();
        }
    }

    let sum = exp_input.sum_axis(Axis(1));

    let mut output = exp_input;
    for (i, mut row) in output.axis_iter_mut(Axis(0)).enumerate() {
        for val in row.iter_mut() {
            *val /= sum[i];
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_conv1d() {
        let conv = Conv1d::new(3, 6, 3, 1, 1, 1);
        let input = Array3::ones((2, 3, 10));
        let output = conv.forward(&input);
        assert_eq!(output.dim(), (2, 6, 10));
    }

    #[test]
    fn test_depthwise_conv() {
        let conv = Conv1d::depthwise(4, 7, 3);
        let input = Array3::ones((1, 4, 20));
        let output = conv.forward(&input);
        assert_eq!(output.dim(), (1, 4, 20));
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(4);
        let input = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
        let output = ln.forward_2d(&input);

        // Check that output has zero mean
        for row in output.axis_iter(Axis(0)) {
            let mean = row.mean().unwrap();
            assert_relative_eq!(mean, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_linear() {
        let linear = Linear::new(10, 5);
        let input = Array2::ones((3, 10));
        let output = linear.forward(&input);
        assert_eq!(output.dim(), (3, 5));
    }

    #[test]
    fn test_gelu() {
        assert_relative_eq!(gelu(0.0), 0.0, epsilon = 1e-6);
        assert!(gelu(1.0) > 0.0);
        assert!(gelu(-1.0) < 0.0);
    }

    #[test]
    fn test_softmax() {
        let input = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = softmax(&input);

        // Sum should be 1
        let sum = output.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Values should be in ascending order
        assert!(output[[0, 0]] < output[[0, 1]]);
        assert!(output[[0, 1]] < output[[0, 2]]);
    }
}
