//! Embedding Network for feature transformation
//!
//! Transforms raw market features into a learned embedding space where
//! similar market conditions are close together.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;
use std::f64::consts::E;

/// Multi-layer embedding network with ReLU activations
#[derive(Debug, Clone)]
pub struct EmbeddingNetwork {
    /// Weight matrices for each layer
    weights: Vec<Array2<f64>>,
    /// Bias vectors for each layer
    biases: Vec<Array1<f64>>,
    /// Layer dimensions [input, hidden1, hidden2, ..., output]
    layer_dims: Vec<usize>,
    /// Whether to apply batch normalization
    use_batch_norm: bool,
    /// Running mean for batch normalization
    running_mean: Vec<Array1<f64>>,
    /// Running variance for batch normalization
    running_var: Vec<Array1<f64>>,
}

impl EmbeddingNetwork {
    /// Create a new embedding network
    ///
    /// # Arguments
    /// * `input_dim` - Dimension of input features
    /// * `hidden_dims` - Dimensions of hidden layers
    /// * `output_dim` - Dimension of output embedding
    /// * `use_batch_norm` - Whether to use batch normalization
    ///
    /// # Example
    /// ```
    /// use matching_networks_finance::network::EmbeddingNetwork;
    ///
    /// let network = EmbeddingNetwork::new(20, &[64, 64], 32, true);
    /// ```
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        use_batch_norm: bool,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut layer_dims = vec![input_dim];
        layer_dims.extend_from_slice(hidden_dims);
        layer_dims.push(output_dim);

        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut running_mean = Vec::new();
        let mut running_var = Vec::new();

        // Initialize weights using He initialization
        for i in 0..layer_dims.len() - 1 {
            let fan_in = layer_dims[i];
            let fan_out = layer_dims[i + 1];
            let std_dev = (2.0 / fan_in as f64).sqrt();
            let normal = Normal::new(0.0, std_dev).unwrap();

            let weight = Array2::from_shape_fn((fan_in, fan_out), |_| rng.sample(&normal));
            let bias = Array1::zeros(fan_out);

            weights.push(weight);
            biases.push(bias);

            if use_batch_norm && i < layer_dims.len() - 2 {
                running_mean.push(Array1::zeros(fan_out));
                running_var.push(Array1::ones(fan_out));
            }
        }

        Self {
            weights,
            biases,
            layer_dims,
            use_batch_norm,
            running_mean,
            running_var,
        }
    }

    /// Forward pass through the embedding network
    ///
    /// # Arguments
    /// * `input` - Input features of shape [batch_size, input_dim]
    ///
    /// # Returns
    /// Embeddings of shape [batch_size, output_dim]
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();

        for (i, (weight, bias)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            // Linear transformation
            x = x.dot(weight) + bias;

            // Apply batch normalization (except for the last layer)
            if self.use_batch_norm && i < self.weights.len() - 1 && !self.running_mean.is_empty() {
                x = self.batch_normalize(&x, i);
            }

            // Apply ReLU activation (except for the last layer)
            if i < self.weights.len() - 1 {
                x = x.mapv(|v| v.max(0.0));
            }
        }

        // L2 normalize the final embeddings
        self.l2_normalize(&x)
    }

    /// Forward pass for a single sample
    pub fn forward_single(&self, input: &Array1<f64>) -> Array1<f64> {
        let input_2d = input.clone().insert_axis(Axis(0));
        let output = self.forward(&input_2d);
        output.index_axis(Axis(0), 0).to_owned()
    }

    /// Batch normalization
    fn batch_normalize(&self, x: &Array2<f64>, layer_idx: usize) -> Array2<f64> {
        let eps = 1e-5;

        if layer_idx >= self.running_mean.len() {
            return x.clone();
        }

        let mean = &self.running_mean[layer_idx];
        let var = &self.running_var[layer_idx];

        let mut result = x.clone();
        for mut row in result.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                *val = (*val - mean[j]) / (var[j] + eps).sqrt();
            }
        }
        result
    }

    /// L2 normalize embeddings
    fn l2_normalize(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = x.clone();
        for mut row in result.rows_mut() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-10 {
                row.mapv_inplace(|v| v / norm);
            }
        }
        result
    }

    /// Get the input dimension
    pub fn input_dim(&self) -> usize {
        self.layer_dims[0]
    }

    /// Get the output dimension (embedding dimension)
    pub fn output_dim(&self) -> usize {
        *self.layer_dims.last().unwrap()
    }

    /// Get the total number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params: usize = self.weights.iter().map(|w| w.len()).sum();
        let bias_params: usize = self.biases.iter().map(|b| b.len()).sum();
        weight_params + bias_params
    }

    /// Update weights (for training)
    pub fn update_weights(&mut self, gradients: &[Array2<f64>], learning_rate: f64) {
        for (weight, grad) in self.weights.iter_mut().zip(gradients.iter()) {
            *weight = &*weight - &(grad * learning_rate);
        }
    }

    /// Get weights (for serialization)
    pub fn get_weights(&self) -> &[Array2<f64>] {
        &self.weights
    }

    /// Get biases (for serialization)
    pub fn get_biases(&self) -> &[Array1<f64>] {
        &self.biases
    }

    /// Set weights (for loading)
    pub fn set_weights(&mut self, weights: Vec<Array2<f64>>) {
        self.weights = weights;
    }

    /// Set biases (for loading)
    pub fn set_biases(&mut self, biases: Vec<Array1<f64>>) {
        self.biases = biases;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_embedding_network_creation() {
        let network = EmbeddingNetwork::new(20, &[64, 64], 32, true);
        assert_eq!(network.input_dim(), 20);
        assert_eq!(network.output_dim(), 32);
        assert_eq!(network.weights.len(), 3); // 3 layers
    }

    #[test]
    fn test_forward_pass() {
        let network = EmbeddingNetwork::new(10, &[32], 16, false);
        let input = Array2::from_shape_fn((5, 10), |_| rand::random::<f64>());

        let output = network.forward(&input);

        assert_eq!(output.shape(), &[5, 16]);

        // Check L2 normalization
        for row in output.rows() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_single_forward() {
        let network = EmbeddingNetwork::new(10, &[32], 16, false);
        let input = Array1::from_shape_fn(10, |_| rand::random::<f64>());

        let output = network.forward_single(&input);

        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_num_parameters() {
        let network = EmbeddingNetwork::new(10, &[20, 20], 5, false);
        // 10*20 + 20 + 20*20 + 20 + 20*5 + 5 = 200 + 20 + 400 + 20 + 100 + 5 = 745
        assert_eq!(network.num_parameters(), 745);
    }
}
