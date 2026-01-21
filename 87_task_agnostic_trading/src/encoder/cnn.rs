//! CNN encoder for local pattern extraction in market data

use super::SharedEncoder;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};

/// CNN encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CNNConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Output embedding dimension
    pub embedding_dim: usize,
    /// Kernel sizes for different scales
    pub kernel_sizes: Vec<usize>,
    /// Number of filters per kernel size
    pub num_filters: usize,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for CNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 20,
            embedding_dim: 64,
            kernel_sizes: vec![3, 5, 7],
            num_filters: 32,
            dropout: 0.1,
        }
    }
}

/// 1D convolution kernel
struct Conv1D {
    weights: Array2<f64>,
    bias: Array1<f64>,
    kernel_size: usize,
}

impl Conv1D {
    fn new(kernel_size: usize, num_filters: usize) -> Self {
        let scale = (2.0 / kernel_size as f64).sqrt();
        Self {
            weights: Array2::random((num_filters, kernel_size), Uniform::new(-scale, scale)),
            bias: Array1::zeros(num_filters),
            kernel_size,
        }
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let input_len = input.len();
        let num_filters = self.weights.nrows();

        // Apply convolution with padding to maintain size
        let pad = self.kernel_size / 2;
        let mut output = Array1::zeros(num_filters);

        for (f, filter) in self.weights.axis_iter(Axis(0)).enumerate() {
            let mut sum = 0.0;
            for (k, &w) in filter.iter().enumerate() {
                let idx = k as isize - pad as isize;
                if idx >= 0 && (idx as usize) < input_len {
                    sum += w * input[idx as usize];
                }
            }
            output[f] = sum + self.bias[f];
        }

        // Global max pooling across spatial dimension approximated
        // by taking max activation per filter
        output.mapv(|x| x.max(0.0)) // ReLU activation
    }
}

/// Multi-scale CNN block
struct MultiScaleCNN {
    convs: Vec<Conv1D>,
    num_filters: usize,
}

impl MultiScaleCNN {
    fn new(kernel_sizes: &[usize], num_filters: usize) -> Self {
        let convs: Vec<_> = kernel_sizes
            .iter()
            .map(|&k| Conv1D::new(k, num_filters))
            .collect();
        Self { convs, num_filters }
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // Apply each convolution and concatenate
        let outputs: Vec<f64> = self.convs
            .iter()
            .flat_map(|conv| conv.forward(input).to_vec())
            .collect();
        Array1::from_vec(outputs)
    }

    fn output_dim(&self) -> usize {
        self.convs.len() * self.num_filters
    }
}

/// CNN encoder for market data
pub struct CNNEncoder {
    config: CNNConfig,
    multi_scale: MultiScaleCNN,
    fc1: Array2<f64>,
    fc2: Array2<f64>,
}

impl CNNEncoder {
    /// Create a new CNN encoder
    pub fn new(config: CNNConfig) -> Self {
        let multi_scale = MultiScaleCNN::new(&config.kernel_sizes, config.num_filters);
        let conv_out_dim = multi_scale.output_dim();

        // Fully connected layers
        let hidden_dim = config.embedding_dim * 2;
        let scale1 = (2.0 / (conv_out_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (hidden_dim + config.embedding_dim) as f64).sqrt();

        let fc1 = Array2::random((conv_out_dim, hidden_dim), Uniform::new(-scale1, scale1));
        let fc2 = Array2::random((hidden_dim, config.embedding_dim), Uniform::new(-scale2, scale2));

        Self {
            config,
            multi_scale,
            fc1,
            fc2,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CNNConfig {
        &self.config
    }
}

impl SharedEncoder for CNNEncoder {
    fn encode(&self, input: &Array1<f64>) -> Array1<f64> {
        // Multi-scale convolution
        let conv_out = self.multi_scale.forward(input);

        // FC layers with ReLU
        let hidden = conv_out.dot(&self.fc1).mapv(|x| x.max(0.0));
        hidden.dot(&self.fc2)
    }

    fn encode_batch(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut outputs = Vec::with_capacity(inputs.nrows());
        for row in inputs.axis_iter(Axis(0)) {
            let embedding = self.encode(&row.to_owned());
            outputs.push(embedding);
        }

        let flat: Vec<f64> = outputs.iter().flat_map(|e| e.to_vec()).collect();
        Array2::from_shape_vec((inputs.nrows(), self.config.embedding_dim), flat)
            .expect("Shape mismatch in encode_batch")
    }

    fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    fn parameters(&self) -> Vec<Array2<f64>> {
        vec![self.fc1.clone(), self.fc2.clone()]
    }

    fn update_parameters(&mut self, gradients: &[Array2<f64>], learning_rate: f64) {
        if gradients.len() >= 2 {
            self.fc1 = &self.fc1 - &(&gradients[0] * learning_rate);
            self.fc2 = &self.fc2 - &(&gradients[1] * learning_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_cnn_encoder() {
        let config = CNNConfig {
            input_dim: 20,
            embedding_dim: 32,
            kernel_sizes: vec![3, 5],
            num_filters: 16,
            dropout: 0.1,
        };

        let encoder = CNNEncoder::new(config);
        let input = Array::random(20, Uniform::new(-1.0, 1.0));
        let output = encoder.encode(&input);

        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_multi_scale() {
        let ms = MultiScaleCNN::new(&[3, 5, 7], 16);
        let input = Array::random(20, Uniform::new(-1.0, 1.0));
        let output = ms.forward(&input);

        assert_eq!(output.len(), 48); // 3 * 16
    }
}
