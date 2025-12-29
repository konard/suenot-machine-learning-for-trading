//! Simple neural network implementation for DQN.

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Activation function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Linear,
}

impl Activation {
    fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Linear => x,
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::Linear => 1.0,
        }
    }
}

/// A single layer in the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Layer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    activation: Activation,
}

impl Layer {
    fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        // Xavier initialization
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weights = Array2::random((input_size, output_size), Uniform::new(-scale, scale));
        let biases = Array1::zeros(output_size);

        Self {
            weights,
            biases,
            activation,
        }
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = input.dot(&self.weights) + &self.biases;
        z.mapv(|x| self.activation.apply(x))
    }
}

/// Neural network for Q-value approximation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl NeuralNetwork {
    /// Create a new neural network
    pub fn new(layer_sizes: &[usize], learning_rate: f64) -> Self {
        assert!(layer_sizes.len() >= 2, "Need at least input and output layers");

        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let activation = if i == layer_sizes.len() - 2 {
                Activation::Linear // Output layer
            } else {
                Activation::ReLU // Hidden layers
            };

            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1], activation));
        }

        Self {
            layers,
            learning_rate,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    /// Predict Q-values for all actions
    pub fn predict(&self, state: &Array1<f64>) -> Array1<f64> {
        self.forward(state)
    }

    /// Get the best action (argmax of Q-values)
    pub fn best_action(&self, state: &Array1<f64>) -> usize {
        let q_values = self.predict(state);
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Train the network on a batch (simplified gradient descent)
    pub fn train_batch(
        &mut self,
        states: &[Array1<f64>],
        actions: &[usize],
        targets: &[f64],
    ) {
        for ((state, &action), &target) in states.iter().zip(actions).zip(targets) {
            self.train_single(state, action, target);
        }
    }

    /// Train on a single sample using simple gradient descent
    fn train_single(&mut self, state: &Array1<f64>, action: usize, target: f64) {
        // Forward pass storing activations
        let mut activations = vec![state.clone()];
        let mut output = state.clone();

        for layer in &self.layers {
            output = layer.forward(&output);
            activations.push(output.clone());
        }

        // Compute error for the specific action
        let mut error = Array1::zeros(output.len());
        error[action] = target - output[action];

        // Backward pass (simplified)
        for i in (0..self.layers.len()).rev() {
            let input = &activations[i];
            let output = &activations[i + 1];

            // Compute gradients
            let delta: Array1<f64> = if i == self.layers.len() - 1 {
                error.clone()
            } else {
                let layer_output = output;
                error.iter()
                    .zip(layer_output.iter())
                    .map(|(&e, &o)| e * self.layers[i].activation.derivative(o))
                    .collect::<Array1<f64>>()
            };

            // Update weights
            for j in 0..self.layers[i].weights.nrows() {
                for k in 0..self.layers[i].weights.ncols() {
                    self.layers[i].weights[[j, k]] +=
                        self.learning_rate * input[j] * delta[k];
                }
            }

            // Update biases
            for k in 0..self.layers[i].biases.len() {
                self.layers[i].biases[k] += self.learning_rate * delta[k];
            }

            // Propagate error backward
            if i > 0 {
                let mut new_error = Array1::zeros(self.layers[i].weights.nrows());
                for j in 0..new_error.len() {
                    for k in 0..delta.len() {
                        new_error[j] += delta[k] * self.layers[i].weights[[j, k]];
                    }
                }
                error = new_error;
            }
        }
    }

    /// Copy weights from another network
    pub fn copy_from(&mut self, other: &NeuralNetwork) {
        for (self_layer, other_layer) in self.layers.iter_mut().zip(other.layers.iter()) {
            self_layer.weights.assign(&other_layer.weights);
            self_layer.biases.assign(&other_layer.biases);
        }
    }

    /// Soft update weights from another network (for target network)
    pub fn soft_update(&mut self, other: &NeuralNetwork, tau: f64) {
        for (self_layer, other_layer) in self.layers.iter_mut().zip(other.layers.iter()) {
            for i in 0..self_layer.weights.nrows() {
                for j in 0..self_layer.weights.ncols() {
                    self_layer.weights[[i, j]] = tau * other_layer.weights[[i, j]]
                        + (1.0 - tau) * self_layer.weights[[i, j]];
                }
            }
            for i in 0..self_layer.biases.len() {
                self_layer.biases[i] =
                    tau * other_layer.biases[i] + (1.0 - tau) * self_layer.biases[i];
            }
        }
    }

    /// Save network to file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Load network from file
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let network = serde_json::from_reader(reader)?;
        Ok(network)
    }

    /// Get learning rate
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = NeuralNetwork::new(&[10, 64, 32, 3], 0.001);
        assert_eq!(network.layers.len(), 3);
    }

    #[test]
    fn test_forward_pass() {
        let network = NeuralNetwork::new(&[10, 64, 32, 3], 0.001);
        let input = Array1::from_vec(vec![0.1; 10]);
        let output = network.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_best_action() {
        let network = NeuralNetwork::new(&[10, 32, 3], 0.001);
        let input = Array1::from_vec(vec![0.1; 10]);
        let action = network.best_action(&input);
        assert!(action < 3);
    }

    #[test]
    fn test_soft_update() {
        let mut network1 = NeuralNetwork::new(&[10, 32, 3], 0.001);
        let network2 = NeuralNetwork::new(&[10, 32, 3], 0.001);

        let input = Array1::from_vec(vec![0.1; 10]);
        let before = network1.forward(&input).clone();

        network1.soft_update(&network2, 0.5);

        let after = network1.forward(&input);
        // Values should have changed
        assert!(before != after);
    }
}
