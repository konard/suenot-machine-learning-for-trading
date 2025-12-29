//! Dense (Fully Connected) Layer Implementation
//!
//! A dense layer performs: output = activation(input * weights + bias)

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::activation::{create_activation, Activation, ActivationType};

/// Dense layer with weights, biases, and activation function
#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    /// Weight matrix (input_size x output_size)
    pub weights: Array2<f64>,
    /// Bias vector (output_size)
    pub biases: Array1<f64>,
    /// Activation function type
    pub activation_type: ActivationType,
    /// Input size
    pub input_size: usize,
    /// Output size (number of neurons)
    pub output_size: usize,
    /// Dropout rate (0.0 = no dropout)
    pub dropout_rate: f64,

    // Cached values for backpropagation (not serialized)
    #[serde(skip)]
    pub last_input: Option<Array2<f64>>,
    #[serde(skip)]
    pub last_z: Option<Array2<f64>>,
    #[serde(skip)]
    pub last_output: Option<Array2<f64>>,
    #[serde(skip)]
    pub dropout_mask: Option<Array2<f64>>,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType) -> Self {
        // Xavier/Glorot initialization
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();
        let weights = Array2::random((input_size, output_size), Uniform::new(-limit, limit));
        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            activation_type: activation,
            input_size,
            output_size,
            dropout_rate: 0.0,
            last_input: None,
            last_z: None,
            last_output: None,
            dropout_mask: None,
        }
    }

    /// Create layer with specific dropout rate
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Forward pass through the layer
    pub fn forward(&mut self, input: &Array2<f64>, training: bool) -> Array2<f64> {
        self.last_input = Some(input.clone());

        // Linear transformation: z = input @ weights + bias
        let mut z = input.dot(&self.weights);
        for mut row in z.rows_mut() {
            row += &self.biases;
        }
        self.last_z = Some(z.clone());

        // Apply activation
        let activation = create_activation(self.activation_type);
        let mut output = activation.forward_batch(&z);

        // Apply dropout during training
        if training && self.dropout_rate > 0.0 {
            let mut rng = rand::thread_rng();
            let mask = Array2::from_shape_fn(output.dim(), |_| {
                if rng.gen::<f64>() > self.dropout_rate {
                    1.0 / (1.0 - self.dropout_rate) // Scale to maintain expected value
                } else {
                    0.0
                }
            });
            output = &output * &mask;
            self.dropout_mask = Some(mask);
        }

        self.last_output = Some(output.clone());
        output
    }

    /// Backward pass - compute gradients
    /// Returns: (input_gradient, weight_gradient, bias_gradient)
    pub fn backward(&self, output_gradient: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let z = self.last_z.as_ref().expect("Must call forward before backward");
        let input = self.last_input.as_ref().expect("Must call forward before backward");

        // Apply dropout mask to gradient
        let grad = if let Some(mask) = &self.dropout_mask {
            output_gradient * mask
        } else {
            output_gradient.clone()
        };

        // Compute activation derivative
        let activation = create_activation(self.activation_type);
        let activation_grad = activation.backward_batch(z);
        let delta = &grad * &activation_grad;

        // Gradient with respect to weights
        let weight_gradient = input.t().dot(&delta);

        // Gradient with respect to biases
        let bias_gradient = delta.sum_axis(Axis(0));

        // Gradient with respect to input (for previous layer)
        let input_gradient = delta.dot(&self.weights.t());

        (input_gradient, weight_gradient, bias_gradient)
    }

    /// Update weights using gradients
    pub fn update_weights(&mut self, weight_gradient: &Array2<f64>, bias_gradient: &Array1<f64>, learning_rate: f64) {
        self.weights = &self.weights - &(learning_rate * weight_gradient);
        self.biases = &self.biases - &(learning_rate * bias_gradient);
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weights.len() + self.biases.len()
    }
}

impl Clone for DenseLayer {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            biases: self.biases.clone(),
            activation_type: self.activation_type,
            input_size: self.input_size,
            output_size: self.output_size,
            dropout_rate: self.dropout_rate,
            last_input: None,
            last_z: None,
            last_output: None,
            dropout_mask: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = DenseLayer::new(10, 5, ActivationType::ReLU);
        assert_eq!(layer.weights.dim(), (10, 5));
        assert_eq!(layer.biases.len(), 5);
    }

    #[test]
    fn test_forward_pass() {
        let mut layer = DenseLayer::new(4, 3, ActivationType::ReLU);
        let input = Array2::ones((2, 4)); // batch of 2, input size 4
        let output = layer.forward(&input, false);
        assert_eq!(output.dim(), (2, 3));
    }

    #[test]
    fn test_num_parameters() {
        let layer = DenseLayer::new(10, 5, ActivationType::ReLU);
        assert_eq!(layer.num_parameters(), 10 * 5 + 5);
    }
}
