//! Neural Network Implementation
//!
//! Full feedforward neural network with training capabilities

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

use super::activation::ActivationType;
use super::layer::DenseLayer;
use super::optimizer::{Adam, Optimizer};

/// Loss function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error (for regression)
    MSE,
    /// Binary Cross-Entropy (for binary classification)
    BinaryCrossEntropy,
    /// Categorical Cross-Entropy (for multi-class classification)
    CategoricalCrossEntropy,
}

/// Neural Network configuration
#[derive(Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub layer_sizes: Vec<usize>,
    pub activations: Vec<ActivationType>,
    pub dropout_rates: Vec<f64>,
    pub loss_function: LossFunction,
}

impl NetworkConfig {
    pub fn new(input_size: usize) -> Self {
        Self {
            layer_sizes: vec![input_size],
            activations: vec![],
            dropout_rates: vec![],
            loss_function: LossFunction::MSE,
        }
    }

    /// Add a hidden layer
    pub fn add_layer(mut self, size: usize, activation: ActivationType) -> Self {
        self.layer_sizes.push(size);
        self.activations.push(activation);
        self.dropout_rates.push(0.0);
        self
    }

    /// Add a hidden layer with dropout
    pub fn add_layer_with_dropout(
        mut self,
        size: usize,
        activation: ActivationType,
        dropout: f64,
    ) -> Self {
        self.layer_sizes.push(size);
        self.activations.push(activation);
        self.dropout_rates.push(dropout);
        self
    }

    /// Set output layer
    pub fn output_layer(mut self, size: usize, activation: ActivationType) -> Self {
        self.layer_sizes.push(size);
        self.activations.push(activation);
        self.dropout_rates.push(0.0);
        self
    }

    /// Set loss function
    pub fn with_loss(mut self, loss: LossFunction) -> Self {
        self.loss_function = loss;
        self
    }
}

/// Feedforward Neural Network
pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
    pub config: NetworkConfig,
    optimizers: Vec<Box<dyn Optimizer>>,
}

impl NeuralNetwork {
    /// Create network from configuration
    pub fn from_config(config: NetworkConfig) -> Self {
        let mut layers = Vec::new();

        for i in 0..config.activations.len() {
            let input_size = config.layer_sizes[i];
            let output_size = config.layer_sizes[i + 1];
            let activation = config.activations[i];
            let dropout = config.dropout_rates[i];

            let layer = DenseLayer::new(input_size, output_size, activation).with_dropout(dropout);
            layers.push(layer);
        }

        // Default optimizer: Adam
        let optimizers: Vec<Box<dyn Optimizer>> = (0..layers.len())
            .map(|_| Box::new(Adam::new(0.001)) as Box<dyn Optimizer>)
            .collect();

        Self {
            layers,
            config,
            optimizers,
        }
    }

    /// Create a simple network for regression
    pub fn regression(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> Self {
        let mut config = NetworkConfig::new(input_size);

        for &size in hidden_sizes {
            config = config.add_layer(size, ActivationType::ReLU);
        }

        config = config
            .output_layer(output_size, ActivationType::Linear)
            .with_loss(LossFunction::MSE);

        Self::from_config(config)
    }

    /// Create a simple network for binary classification
    pub fn binary_classification(input_size: usize, hidden_sizes: &[usize]) -> Self {
        let mut config = NetworkConfig::new(input_size);

        for &size in hidden_sizes {
            config = config.add_layer(size, ActivationType::ReLU);
        }

        config = config
            .output_layer(1, ActivationType::Sigmoid)
            .with_loss(LossFunction::BinaryCrossEntropy);

        Self::from_config(config)
    }

    /// Set optimizer for all layers
    pub fn set_optimizer(&mut self, optimizer: Box<dyn Optimizer>) {
        self.optimizers = self
            .layers
            .iter()
            .map(|_| optimizer.clone_box())
            .collect();
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &Array2<f64>, training: bool) -> Array2<f64> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output, training);
        }
        output
    }

    /// Predict (forward pass without training mode)
    pub fn predict(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.forward(input, false)
    }

    /// Compute loss
    pub fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
        let n = predictions.len() as f64;

        match self.config.loss_function {
            LossFunction::MSE => {
                let diff = predictions - targets;
                (&diff * &diff).sum() / n
            }
            LossFunction::BinaryCrossEntropy => {
                let epsilon = 1e-15;
                let p = predictions.mapv(|v| v.clamp(epsilon, 1.0 - epsilon));
                let loss = targets * &p.mapv(f64::ln)
                    + &(1.0 - targets) * &(1.0 - &p).mapv(f64::ln);
                -loss.sum() / n
            }
            LossFunction::CategoricalCrossEntropy => {
                let epsilon = 1e-15;
                let p = predictions.mapv(|v| v.clamp(epsilon, 1.0 - epsilon));
                let loss = targets * &p.mapv(f64::ln);
                -loss.sum() / n
            }
        }
    }

    /// Compute loss gradient
    fn compute_loss_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let n = predictions.nrows() as f64;

        match self.config.loss_function {
            LossFunction::MSE => {
                2.0 * (predictions - targets) / n
            }
            LossFunction::BinaryCrossEntropy => {
                let epsilon = 1e-15;
                let p = predictions.mapv(|v| v.clamp(epsilon, 1.0 - epsilon));
                (((&p - targets) / (&p * &(1.0 - &p))) / n)
            }
            LossFunction::CategoricalCrossEntropy => {
                let epsilon = 1e-15;
                let p = predictions.mapv(|v| v.clamp(epsilon, 1.0 - epsilon));
                (-targets / &p) / n
            }
        }
    }

    /// Backward pass and weight update
    pub fn backward(&mut self, predictions: &Array2<f64>, targets: &Array2<f64>) {
        // Compute loss gradient
        let mut gradient = self.compute_loss_gradient(predictions, targets);

        // Backpropagate through layers
        for i in (0..self.layers.len()).rev() {
            let (input_grad, weight_grad, bias_grad) = self.layers[i].backward(&gradient);

            // Update weights
            self.optimizers[i].update_weights(&mut self.layers[i].weights, &weight_grad);
            self.optimizers[i].update_biases(&mut self.layers[i].biases, &bias_grad);

            gradient = input_grad;
        }
    }

    /// Train for one epoch
    pub fn train_epoch(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array2<f64>,
        batch_size: usize,
    ) -> f64 {
        let n_samples = x_train.nrows();
        let n_batches = (n_samples + batch_size - 1) / batch_size;
        let mut total_loss = 0.0;

        // Shuffle indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        for batch_idx in 0..n_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(n_samples);
            let batch_indices = &indices[start..end];

            // Get batch data
            let x_batch = x_train.select(Axis(0), batch_indices);
            let y_batch = y_train.select(Axis(0), batch_indices);

            // Forward pass
            let predictions = self.forward(&x_batch, true);

            // Compute loss
            total_loss += self.compute_loss(&predictions, &y_batch);

            // Backward pass
            self.backward(&predictions, &y_batch);
        }

        total_loss / n_batches as f64
    }

    /// Train the network
    pub fn train(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array2<f64>,
        epochs: usize,
        batch_size: usize,
        verbose: bool,
    ) -> Vec<f64> {
        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let loss = self.train_epoch(x_train, y_train, batch_size);
            losses.push(loss);

            if verbose && (epoch + 1) % 10 == 0 {
                println!("Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, loss);
            }
        }

        losses
    }

    /// Evaluate on test data
    pub fn evaluate(&mut self, x_test: &Array2<f64>, y_test: &Array2<f64>) -> f64 {
        let predictions = self.predict(x_test);
        self.compute_loss(&predictions, y_test)
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }

    /// Save model to file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        let data = (&self.config, &self.layers);
        serde_json::to_writer(writer, &data)?;

        Ok(())
    }

    /// Load model from file
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let (config, layers): (NetworkConfig, Vec<DenseLayer>) = serde_json::from_reader(reader)?;

        let optimizers: Vec<Box<dyn Optimizer>> = (0..layers.len())
            .map(|_| Box::new(Adam::new(0.001)) as Box<dyn Optimizer>)
            .collect();

        Ok(Self {
            layers,
            config,
            optimizers,
        })
    }

    /// Print network summary
    pub fn summary(&self) {
        println!("Neural Network Summary");
        println!("======================");
        println!("Input size: {}", self.config.layer_sizes[0]);

        for (i, layer) in self.layers.iter().enumerate() {
            println!(
                "Layer {}: {} -> {} ({:?}), params: {}",
                i + 1,
                layer.input_size,
                layer.output_size,
                layer.activation_type,
                layer.num_parameters()
            );
        }

        println!("======================");
        println!("Total parameters: {}", self.num_parameters());
        println!("Loss function: {:?}", self.config.loss_function);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let config = NetworkConfig::new(10)
            .add_layer(32, ActivationType::ReLU)
            .add_layer(16, ActivationType::ReLU)
            .output_layer(1, ActivationType::Sigmoid);

        let network = NeuralNetwork::from_config(config);
        assert_eq!(network.layers.len(), 3);
    }

    #[test]
    fn test_forward_pass() {
        let mut network = NeuralNetwork::regression(4, &[8, 4], 1);
        let input = Array2::ones((10, 4));
        let output = network.predict(&input);
        assert_eq!(output.dim(), (10, 1));
    }

    #[test]
    fn test_training() {
        let mut network = NeuralNetwork::regression(2, &[4], 1);

        // Simple XOR-like problem
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        )
        .unwrap();
        let y = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

        let initial_loss = network.evaluate(&x, &y);
        network.train(&x, &y, 100, 4, false);
        let final_loss = network.evaluate(&x, &y);

        assert!(final_loss < initial_loss);
    }
}
