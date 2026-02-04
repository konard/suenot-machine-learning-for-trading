//! MC Dropout model implementation

use super::config::MCDropoutConfig;
use super::layers::{DropoutLayer, LinearLayer, relu};
use super::prediction::PredictionResult;
use ndarray::Array1;

/// MC Dropout Neural Network
#[derive(Debug, Clone)]
pub struct MCDropoutModel {
    /// Model configuration
    pub config: MCDropoutConfig,
    /// Linear layers
    layers: Vec<LinearLayer>,
    /// Dropout layers
    dropouts: Vec<DropoutLayer>,
}

impl MCDropoutModel {
    /// Create a new MC Dropout model
    pub fn new(config: MCDropoutConfig) -> Self {
        config.validate().expect("Invalid configuration");

        let mut layers = Vec::new();
        let mut dropouts = Vec::new();

        // Build layers
        let mut prev_dim = config.input_dim;
        for &hidden_dim in &config.hidden_dims {
            layers.push(LinearLayer::new(prev_dim, hidden_dim));
            dropouts.push(DropoutLayer::new(config.dropout_rate));
            prev_dim = hidden_dim;
        }

        // Output layer
        layers.push(LinearLayer::new(prev_dim, config.output_dim));

        Self {
            config,
            layers,
            dropouts,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut current = x.clone();

        // Hidden layers with dropout and ReLU
        for i in 0..self.dropouts.len() {
            current = self.layers[i].forward(&current);
            current = relu(&current);
            current = self.dropouts[i].forward(&current);
        }

        // Output layer (no dropout, no activation for regression)
        self.layers.last().unwrap().forward(&current)
    }

    /// Make predictions with uncertainty estimation using MC Dropout
    ///
    /// This method runs multiple forward passes with dropout active
    /// to estimate the uncertainty in the prediction.
    pub fn predict_with_uncertainty(
        &self,
        x: &Array1<f64>,
        n_samples: Option<usize>,
    ) -> PredictionResult {
        let n = n_samples.unwrap_or(self.config.n_mc_samples);

        let samples: Vec<f64> = (0..n)
            .map(|_| {
                let output = self.forward(x);
                output[0] // Assuming single output
            })
            .collect();

        PredictionResult::from_samples(samples)
    }

    /// Predict without dropout (standard mode)
    pub fn predict(&mut self, x: &Array1<f64>) -> f64 {
        // Disable dropout
        for dropout in &mut self.dropouts {
            dropout.set_training(false);
        }

        let output = self.forward(x);

        // Re-enable dropout
        for dropout in &mut self.dropouts {
            dropout.set_training(true);
        }

        output[0]
    }

    /// Get the number of parameters in the model
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| {
            l.weights.len() + l.bias.len()
        }).sum()
    }

    /// Enable/disable training mode (affects dropout)
    pub fn set_training(&mut self, training: bool) {
        for dropout in &mut self.dropouts {
            dropout.set_training(training);
        }
    }
}

impl Default for MCDropoutModel {
    fn default() -> Self {
        Self::new(MCDropoutConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let config = MCDropoutConfig::new(10)
            .with_hidden_dims(vec![32, 16]);
        let model = MCDropoutModel::new(config);
        assert_eq!(model.layers.len(), 3); // 2 hidden + 1 output
        assert_eq!(model.dropouts.len(), 2); // dropout after each hidden
    }

    #[test]
    fn test_forward_pass() {
        let config = MCDropoutConfig::new(10)
            .with_hidden_dims(vec![32, 16]);
        let model = MCDropoutModel::new(config);

        let input = Array1::ones(10);
        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_mc_dropout_inference() {
        let config = MCDropoutConfig::new(10)
            .with_hidden_dims(vec![32, 16])
            .with_mc_samples(100);
        let model = MCDropoutModel::new(config);

        let input = Array1::ones(10);
        let result = model.predict_with_uncertainty(&input, None);

        assert_eq!(result.n_samples, 100);
        assert!(result.std > 0.0); // Should have some variance due to dropout
    }

    #[test]
    fn test_uncertainty_varies() {
        let config = MCDropoutConfig::new(10)
            .with_hidden_dims(vec![64, 32])
            .with_dropout_rate(0.3)
            .with_mc_samples(50);
        let model = MCDropoutModel::new(config);

        let input = Array1::ones(10);
        let result = model.predict_with_uncertainty(&input, None);

        // With dropout, we should see variance in predictions
        let min = result.samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = result.samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max > min); // Predictions should vary
    }
}
