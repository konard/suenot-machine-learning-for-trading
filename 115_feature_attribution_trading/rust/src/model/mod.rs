//! Neural Network Models for Trading
//!
//! This module provides neural network implementations for trading signal
//! prediction with gradient computation support for attribution methods.

pub mod attribution;

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Trait for neural networks that support gradient computation
pub trait NeuralNetwork: Send + Sync {
    /// Forward pass: compute output from input
    fn forward(&self, input: &Array1<f64>) -> Array1<f64>;

    /// Compute gradient of output with respect to input
    fn gradient(&self, input: &Array1<f64>) -> Array1<f64>;

    /// Get the number of input features
    fn input_size(&self) -> usize;

    /// Get the number of outputs
    fn output_size(&self) -> usize;
}

/// Activation function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Leaky ReLU: max(0.01*x, x)
    LeakyReLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// No activation (identity)
    Linear,
    /// Softplus: log(1 + exp(x))
    Softplus,
    /// Exponential Linear Unit
    ELU,
}

impl Activation {
    /// Apply activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Linear => x,
            Activation::Softplus => (1.0 + x.exp()).ln(),
            Activation::ELU => if x > 0.0 { x } else { x.exp() - 1.0 },
        }
    }

    /// Compute derivative of activation function
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::LeakyReLU => if x > 0.0 { 1.0 } else { 0.01 },
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Linear => 1.0,
            Activation::Softplus => 1.0 / (1.0 + (-x).exp()),
            Activation::ELU => if x > 0.0 { 1.0 } else { x.exp() },
        }
    }
}

/// Dense (fully connected) layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    weights: Array2<f64>,
    bias: Array1<f64>,
    activation: Activation,
    input_size: usize,
    output_size: usize,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier/He initialization
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();

        // He initialization for ReLU variants, Xavier for others
        let scale = match activation {
            Activation::ReLU | Activation::LeakyReLU | Activation::ELU => {
                (2.0 / input_size as f64).sqrt()
            }
            _ => (2.0 / (input_size + output_size) as f64).sqrt(),
        };

        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.gen::<f64>() * 2.0 * scale - scale
        });

        let bias = Array1::zeros(output_size);

        Self {
            weights,
            bias,
            activation,
            input_size,
            output_size,
        }
    }

    /// Create layer with specified weights
    pub fn with_weights(
        weights: Array2<f64>,
        bias: Array1<f64>,
        activation: Activation,
    ) -> Self {
        let (output_size, input_size) = weights.dim();
        Self {
            weights,
            bias,
            activation,
            input_size,
            output_size,
        }
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = self.weights.dot(input) + &self.bias;
        z.mapv(|x| self.activation.apply(x))
    }

    /// Compute pre-activation values (before activation function)
    pub fn pre_activation(&self, input: &Array1<f64>) -> Array1<f64> {
        self.weights.dot(input) + &self.bias
    }

    /// Get weights reference
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Get activation function
    pub fn activation(&self) -> Activation {
        self.activation
    }
}

/// Multi-layer perceptron trading model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingModel {
    layers: Vec<DenseLayer>,
    input_size: usize,
    hidden_sizes: Vec<usize>,
    output_size: usize,
    dropout_rate: f64,
}

impl TradingModel {
    /// Create a new trading model
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `hidden_sizes` - Sizes of hidden layers
    /// * `output_size` - Number of outputs (1 for binary classification)
    ///
    /// # Example
    ///
    /// ```
    /// use feature_attribution_trading::TradingModel;
    ///
    /// // Create model with 10 inputs, two hidden layers (64, 32), and 1 output
    /// let model = TradingModel::new(10, vec![64, 32], 1);
    /// ```
    pub fn new(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        let mut layers = Vec::new();
        let mut in_size = input_size;

        // Hidden layers with ReLU activation
        for &hidden_size in &hidden_sizes {
            layers.push(DenseLayer::new(in_size, hidden_size, Activation::ReLU));
            in_size = hidden_size;
        }

        // Output layer with linear activation (use sigmoid externally for probability)
        layers.push(DenseLayer::new(in_size, output_size, Activation::Linear));

        Self {
            layers,
            input_size,
            hidden_sizes,
            output_size,
            dropout_rate: 0.0,
        }
    }

    /// Create model with custom activation functions
    pub fn with_activations(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        hidden_activation: Activation,
        output_activation: Activation,
    ) -> Self {
        let mut layers = Vec::new();
        let mut in_size = input_size;

        for &hidden_size in &hidden_sizes {
            layers.push(DenseLayer::new(in_size, hidden_size, hidden_activation));
            in_size = hidden_size;
        }

        layers.push(DenseLayer::new(in_size, output_size, output_activation));

        Self {
            layers,
            input_size,
            hidden_sizes,
            output_size,
            dropout_rate: 0.0,
        }
    }

    /// Set dropout rate (for training)
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate.clamp(0.0, 0.9);
        self
    }

    /// Get model architecture as string
    pub fn architecture(&self) -> String {
        let mut sizes = vec![self.input_size];
        sizes.extend(&self.hidden_sizes);
        sizes.push(self.output_size);

        sizes
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                layer.weights.len() + layer.bias.len()
            })
            .sum()
    }

    /// Predict probability (sigmoid of output)
    pub fn predict_proba(&self, input: &Array1<f64>) -> f64 {
        let output = self.forward(input);
        1.0 / (1.0 + (-output[0]).exp())
    }

    /// Predict class (threshold at 0.5)
    pub fn predict_class(&self, input: &Array1<f64>) -> i32 {
        if self.predict_proba(input) > 0.5 { 1 } else { 0 }
    }

    /// Batch prediction
    pub fn predict_batch(&self, inputs: &Array2<f64>) -> Vec<f64> {
        (0..inputs.nrows())
            .map(|i| self.predict_proba(&inputs.row(i).to_owned()))
            .collect()
    }

    /// Save model to JSON file
    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Load model from JSON file
    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        let model: Self = serde_json::from_str(&json)?;
        Ok(model)
    }

    /// Get reference to layers
    pub fn layers(&self) -> &[DenseLayer] {
        &self.layers
    }
}

impl NeuralNetwork for TradingModel {
    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    fn gradient(&self, input: &Array1<f64>) -> Array1<f64> {
        // Forward pass, storing intermediate values
        let mut activations = vec![input.clone()];
        let mut pre_activations = Vec::new();

        let mut current = input.clone();
        for layer in &self.layers {
            let z = layer.pre_activation(&current);
            pre_activations.push(z.clone());
            current = z.mapv(|x| layer.activation().apply(x));
            activations.push(current.clone());
        }

        // Backward pass using chain rule
        // Start with gradient of output (d/d(output) of output[0] = 1)
        let mut delta = Array1::ones(self.output_size);

        // Backpropagate through layers
        for (i, layer) in self.layers.iter().enumerate().rev() {
            // Apply activation derivative
            let z = &pre_activations[i];
            let act_deriv: Array1<f64> = z.mapv(|x| layer.activation().derivative(x));
            delta = &delta * &act_deriv;

            // Propagate through weights (transpose)
            delta = layer.weights().t().dot(&delta);
        }

        delta
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn output_size(&self) -> usize {
        self.output_size
    }
}

/// Ensemble of trading models for robust predictions
#[derive(Debug, Clone)]
pub struct EnsembleModel {
    models: Vec<TradingModel>,
    aggregation: AggregationType,
}

/// Method for aggregating ensemble predictions
#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    Mean,
    Median,
    WeightedMean(f64), // weight for first model, rest equally distributed
}

impl EnsembleModel {
    /// Create a new ensemble from models
    pub fn new(models: Vec<TradingModel>, aggregation: AggregationType) -> Self {
        assert!(!models.is_empty(), "Ensemble must have at least one model");
        Self { models, aggregation }
    }

    /// Create ensemble with identical architecture but different initialization
    pub fn with_random_initialization(
        n_models: usize,
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
    ) -> Self {
        let models: Vec<TradingModel> = (0..n_models)
            .map(|_| TradingModel::new(input_size, hidden_sizes.clone(), output_size))
            .collect();

        Self::new(models, AggregationType::Mean)
    }

    /// Predict with ensemble
    pub fn predict(&self, input: &Array1<f64>) -> f64 {
        let predictions: Vec<f64> = self.models
            .iter()
            .map(|m| m.predict_proba(input))
            .collect();

        match self.aggregation {
            AggregationType::Mean => {
                predictions.iter().sum::<f64>() / predictions.len() as f64
            }
            AggregationType::Median => {
                let mut sorted = predictions.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                }
            }
            AggregationType::WeightedMean(first_weight) => {
                let remaining_weight = (1.0 - first_weight) / (predictions.len() - 1) as f64;
                predictions[0] * first_weight
                    + predictions[1..].iter().sum::<f64>() * remaining_weight
            }
        }
    }

    /// Get uncertainty estimate (standard deviation of predictions)
    pub fn uncertainty(&self, input: &Array1<f64>) -> f64 {
        let predictions: Vec<f64> = self.models
            .iter()
            .map(|m| m.predict_proba(input))
            .collect();

        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions
            .iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_functions() {
        let relu = Activation::ReLU;
        assert_eq!(relu.apply(-1.0), 0.0);
        assert_eq!(relu.apply(1.0), 1.0);
        assert_eq!(relu.derivative(-1.0), 0.0);
        assert_eq!(relu.derivative(1.0), 1.0);

        let sigmoid = Activation::Sigmoid;
        assert!((sigmoid.apply(0.0) - 0.5).abs() < 1e-10);

        let tanh = Activation::Tanh;
        assert!((tanh.apply(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_model_forward() {
        let model = TradingModel::new(5, vec![10, 5], 1);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_model_gradient() {
        let model = TradingModel::new(5, vec![10], 1);
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        let grad = model.gradient(&input);
        assert_eq!(grad.len(), 5);
    }

    #[test]
    fn test_model_architecture() {
        let model = TradingModel::new(10, vec![64, 32], 1);
        assert_eq!(model.architecture(), "10 -> 64 -> 32 -> 1");
    }

    #[test]
    fn test_model_save_load() {
        let model = TradingModel::new(5, vec![10], 1);
        let path = "/tmp/test_model.json";

        model.save(path).unwrap();
        let loaded = TradingModel::load(path).unwrap();

        assert_eq!(model.architecture(), loaded.architecture());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_ensemble_prediction() {
        let ensemble = EnsembleModel::with_random_initialization(
            3, 5, vec![10], 1
        );

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let pred = ensemble.predict(&input);
        let uncertainty = ensemble.uncertainty(&input);

        assert!(pred >= 0.0 && pred <= 1.0);
        assert!(uncertainty >= 0.0);
    }
}
