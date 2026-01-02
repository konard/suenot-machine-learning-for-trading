//! Neural network layer implementations

use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// 1D Convolutional layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv1d {
    /// Weight tensor [out_channels, in_channels, kernel_size]
    pub weight: Array3<f32>,
    /// Bias vector [out_channels]
    pub bias: Option<Array1<f32>>,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
}

impl Conv1d {
    /// Create a new Conv1d layer with random initialization
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let mut rng = rand::thread_rng();
        // Kaiming/He initialization
        let std = (2.0 / (in_channels * kernel_size) as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weight = Array3::from_shape_fn((out_channels, in_channels, kernel_size), |_| {
            rng.sample(normal)
        });

        let bias_arr = if bias {
            Some(Array1::zeros(out_channels))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_arr,
            stride,
            padding,
            in_channels,
            out_channels,
            kernel_size,
        }
    }

    /// Forward pass
    /// Input shape: [batch, in_channels, length]
    /// Output shape: [batch, out_channels, new_length]
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let batch_size = input.shape()[0];
        let in_len = input.shape()[2];

        // Calculate output length
        let out_len = (in_len + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let mut output = Array3::zeros((batch_size, self.out_channels, out_len));

        for b in 0..batch_size {
            for oc in 0..self.out_channels {
                for ol in 0..out_len {
                    let mut sum = 0.0f32;
                    let start_idx = ol * self.stride;

                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let in_idx = start_idx + k;
                            let padded_idx = in_idx as i32 - self.padding as i32;

                            if padded_idx >= 0 && (padded_idx as usize) < in_len {
                                sum += input[[b, ic, padded_idx as usize]] * self.weight[[oc, ic, k]];
                            }
                        }
                    }

                    if let Some(ref bias) = self.bias {
                        sum += bias[oc];
                    }

                    output[[b, oc, ol]] = sum;
                }
            }
        }

        output
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        let weight_params = self.out_channels * self.in_channels * self.kernel_size;
        let bias_params = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        weight_params + bias_params
    }
}

/// Batch Normalization 1D
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNorm1d {
    /// Number of features (channels)
    pub num_features: usize,
    /// Scale parameter (gamma)
    pub weight: Array1<f32>,
    /// Shift parameter (beta)
    pub bias: Array1<f32>,
    /// Running mean
    pub running_mean: Array1<f32>,
    /// Running variance
    pub running_var: Array1<f32>,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Momentum for running stats
    pub momentum: f32,
    /// Training mode
    pub training: bool,
}

impl BatchNorm1d {
    /// Create a new BatchNorm1d layer
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            weight: Array1::ones(num_features),
            bias: Array1::zeros(num_features),
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            eps: 1e-5,
            momentum: 0.1,
            training: true,
        }
    }

    /// Forward pass
    /// Input shape: [batch, channels, length]
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let batch_size = input.shape()[0];
        let length = input.shape()[2];

        let mut output = input.clone();

        for c in 0..self.num_features {
            let (mean, var) = if self.training {
                // Calculate batch statistics
                let mut sum = 0.0f32;
                let mut sq_sum = 0.0f32;
                let n = (batch_size * length) as f32;

                for b in 0..batch_size {
                    for l in 0..length {
                        let val = input[[b, c, l]];
                        sum += val;
                        sq_sum += val * val;
                    }
                }

                let mean = sum / n;
                let var = sq_sum / n - mean * mean;
                (mean, var)
            } else {
                (self.running_mean[c], self.running_var[c])
            };

            let std = (var + self.eps).sqrt();

            for b in 0..batch_size {
                for l in 0..length {
                    let normalized = (input[[b, c, l]] - mean) / std;
                    output[[b, c, l]] = self.weight[c] * normalized + self.bias[c];
                }
            }
        }

        output
    }

    /// Set training mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }
}

/// ReLU activation function
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReLU;

impl ReLU {
    /// Create a new ReLU layer
    pub fn new() -> Self {
        Self
    }

    /// Forward pass
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        input.mapv(|x| x.max(0.0))
    }

    /// Forward pass for 2D input
    pub fn forward_2d(&self, input: &Array2<f32>) -> Array2<f32> {
        input.mapv(|x| x.max(0.0))
    }
}

/// Max Pooling 1D
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool1d {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
}

impl MaxPool1d {
    /// Create a new MaxPool1d layer
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let in_len = input.shape()[2];

        let out_len = (in_len + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let mut output = Array3::from_elem((batch_size, channels, out_len), f32::NEG_INFINITY);

        for b in 0..batch_size {
            for c in 0..channels {
                for ol in 0..out_len {
                    let start_idx = ol * self.stride;

                    for k in 0..self.kernel_size {
                        let in_idx = start_idx + k;
                        let padded_idx = in_idx as i32 - self.padding as i32;

                        if padded_idx >= 0 && (padded_idx as usize) < in_len {
                            output[[b, c, ol]] =
                                output[[b, c, ol]].max(input[[b, c, padded_idx as usize]]);
                        }
                    }
                }
            }
        }

        output
    }
}

/// Adaptive Average Pooling 1D
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveAvgPool1d {
    /// Output size
    pub output_size: usize,
}

impl AdaptiveAvgPool1d {
    /// Create a new AdaptiveAvgPool1d layer
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let batch_size = input.shape()[0];
        let channels = input.shape()[1];
        let in_len = input.shape()[2];

        let mut output = Array3::zeros((batch_size, channels, self.output_size));

        for b in 0..batch_size {
            for c in 0..channels {
                for o in 0..self.output_size {
                    let start = (o * in_len) / self.output_size;
                    let end = ((o + 1) * in_len) / self.output_size;
                    let count = end - start;

                    let mut sum = 0.0f32;
                    for i in start..end {
                        sum += input[[b, c, i]];
                    }
                    output[[b, c, o]] = sum / count as f32;
                }
            }
        }

        output
    }
}

/// Linear (Fully Connected) layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    /// Weight matrix [out_features, in_features]
    pub weight: Array2<f32>,
    /// Bias vector [out_features]
    pub bias: Option<Array1<f32>>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
}

impl Linear {
    /// Create a new Linear layer with random initialization
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let mut rng = rand::thread_rng();
        let std = (1.0 / in_features as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weight =
            Array2::from_shape_fn((out_features, in_features), |_| rng.sample(normal));

        let bias_arr = if bias {
            Some(Array1::zeros(out_features))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_arr,
            in_features,
            out_features,
        }
    }

    /// Forward pass
    /// Input shape: [batch, in_features]
    /// Output shape: [batch, out_features]
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.dot(&self.weight.t());

        if let Some(ref bias) = self.bias {
            for mut row in output.axis_iter_mut(Axis(0)) {
                row += bias;
            }
        }

        output
    }

    /// Get number of parameters
    pub fn num_params(&self) -> usize {
        let weight_params = self.in_features * self.out_features;
        let bias_params = if self.bias.is_some() {
            self.out_features
        } else {
            0
        };
        weight_params + bias_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d_forward() {
        let conv = Conv1d::new(2, 4, 3, 1, 1, true);
        let input = Array3::ones((1, 2, 10));
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[1, 4, 10]);
    }

    #[test]
    fn test_batchnorm1d_forward() {
        let bn = BatchNorm1d::new(4);
        let input = Array3::ones((2, 4, 10));
        let output = bn.forward(&input);
        assert_eq!(output.shape(), &[2, 4, 10]);
    }

    #[test]
    fn test_relu_forward() {
        let relu = ReLU::new();
        let input = Array3::from_elem((1, 2, 3), -1.0);
        let output = relu.forward(&input);
        assert!(output.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_maxpool1d_forward() {
        let pool = MaxPool1d::new(2, 2, 0);
        let input = Array3::ones((1, 2, 10));
        let output = pool.forward(&input);
        assert_eq!(output.shape(), &[1, 2, 5]);
    }

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(10, 5, true);
        let input = Array2::ones((2, 10));
        let output = linear.forward(&input);
        assert_eq!(output.shape(), &[2, 5]);
    }
}
