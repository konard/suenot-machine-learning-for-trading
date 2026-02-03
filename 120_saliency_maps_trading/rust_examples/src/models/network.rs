//! Neural network implementation for trading predictions
//!
//! This module provides a simple feedforward neural network that can be used
//! for computing saliency maps via numerical gradients.

use ndarray::{Array1, Array2};
use rand::Rng;

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
}

impl Activation {
    /// Apply activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
        }
    }

    /// Compute derivative
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
        }
    }
}

/// A single layer in the network
#[derive(Debug, Clone)]
pub struct Layer {
    /// Weight matrix (input_size x output_size)
    pub weights: Array2<f64>,
    /// Bias vector
    pub bias: Array1<f64>,
    /// Activation function
    pub activation: Activation,
}

impl Layer {
    /// Create a new layer with random weights
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_size as f64).sqrt(); // Xavier initialization

        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rng.gen_range(-scale..scale)
        });

        let bias = Array1::zeros(output_size);

        Self {
            weights,
            bias,
            activation,
        }
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let linear = input.dot(&self.weights) + &self.bias;
        linear.mapv(|x| self.activation.apply(x))
    }

    /// Forward pass returning pre-activation values
    pub fn forward_with_preact(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let linear = input.dot(&self.weights) + &self.bias;
        let output = linear.mapv(|x| self.activation.apply(x));
        (linear, output)
    }
}

/// Simple feedforward neural network for trading
#[derive(Debug, Clone)]
pub struct TradingNetwork {
    /// Network layers
    layers: Vec<Layer>,
    /// Input size (flattened sequence)
    input_size: usize,
    /// Sequence length
    sequence_length: usize,
    /// Number of features per time step
    num_features: usize,
}

impl TradingNetwork {
    /// Create a new trading network
    ///
    /// # Arguments
    /// * `sequence_length` - Number of time steps in input
    /// * `num_features` - Number of features per time step
    /// * `hidden_sizes` - Sizes of hidden layers
    pub fn new(sequence_length: usize, num_features: usize, hidden_sizes: &[usize]) -> Self {
        let input_size = sequence_length * num_features;
        let mut layers = Vec::new();

        let mut prev_size = input_size;
        for (i, &size) in hidden_sizes.iter().enumerate() {
            let activation = if i == hidden_sizes.len() - 1 {
                Activation::Sigmoid // Output layer
            } else {
                Activation::ReLU // Hidden layers
            };
            layers.push(Layer::new(prev_size, size, activation));
            prev_size = size;
        }

        // Final output layer
        layers.push(Layer::new(prev_size, 1, Activation::Sigmoid));

        Self {
            layers,
            input_size,
            sequence_length,
            num_features,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &Array2<f64>) -> f64 {
        // Flatten input
        let flat_input = Array1::from_iter(input.iter().cloned());

        let mut current = flat_input;
        for layer in &self.layers {
            current = layer.forward(&current);
        }

        current[0]
    }

    /// Predict with confidence
    pub fn predict(&self, input: &Array2<f64>) -> (i32, f64) {
        let prob = self.forward(input);
        let direction = if prob > 0.5 { 1 } else { -1 };
        let confidence = (prob - 0.5).abs() * 2.0;
        (direction, confidence)
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get sequence length
    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Simple training step (gradient descent)
    pub fn train_step(&mut self, input: &Array2<f64>, label: f64, learning_rate: f64) {
        let prediction = self.forward(input);
        let error = prediction - label;

        // Numerical gradient computation and update
        let epsilon = 1e-5;

        for layer in &mut self.layers {
            let (rows, cols) = layer.weights.dim();
            for i in 0..rows {
                for j in 0..cols {
                    let original = layer.weights[[i, j]];

                    layer.weights[[i, j]] = original + epsilon;
                    let loss_plus = (self.forward(input) - label).powi(2);

                    layer.weights[[i, j]] = original - epsilon;
                    let loss_minus = (self.forward(input) - label).powi(2);

                    layer.weights[[i, j]] = original;

                    let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                    layer.weights[[i, j]] -= learning_rate * grad;
                }
            }

            for i in 0..layer.bias.len() {
                let original = layer.bias[i];

                layer.bias[i] = original + epsilon;
                let loss_plus = (self.forward(input) - label).powi(2);

                layer.bias[i] = original - epsilon;
                let loss_minus = (self.forward(input) - label).powi(2);

                layer.bias[i] = original;

                let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
                layer.bias[i] -= learning_rate * grad;
            }
        }
    }

    /// Train on a batch of samples
    pub fn train(&mut self, samples: &[Array2<f64>], labels: &[f64], epochs: usize, learning_rate: f64) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input, &label) in samples.iter().zip(labels.iter()) {
                let prediction = self.forward(input);
                total_loss += (prediction - label).powi(2);
                self.train_step(input, label, learning_rate);
            }

            if (epoch + 1) % 10 == 0 {
                let avg_loss = total_loss / samples.len() as f64;
                log::info!("Epoch {}: avg_loss = {:.6}", epoch + 1, avg_loss);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_forward() {
        let network = TradingNetwork::new(30, 5, &[64, 32]);
        let input = Array2::from_shape_fn((30, 5), |_| 0.5);
        let output = network.forward(&input);

        assert!(output >= 0.0 && output <= 1.0);
    }

    #[test]
    fn test_activation() {
        let relu = Activation::ReLU;
        assert_eq!(relu.apply(-1.0), 0.0);
        assert_eq!(relu.apply(1.0), 1.0);

        let sigmoid = Activation::Sigmoid;
        assert!((sigmoid.apply(0.0) - 0.5).abs() < 1e-10);
    }
}
