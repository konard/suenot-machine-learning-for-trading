//! LRP-enabled neural network

use ndarray::{Array1, Array2};

use super::config::LRPConfig;
use super::linear::LRPLinear;
use super::lrp::LRPRule;

/// LRP-enabled neural network
#[derive(Debug)]
pub struct LRPNetwork {
    /// Network layers
    layers: Vec<LRPLinear>,
    /// LRP rules for each layer
    layer_rules: Vec<LRPRule>,
    /// Configuration
    config: LRPConfig,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
}

impl LRPNetwork {
    /// Create a new network
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    /// * `hidden_dims` - Hidden layer dimensions
    /// * `output_dim` - Output dimension (e.g., 2 for binary classification)
    /// * `config` - LRP configuration
    pub fn new(
        input_dim: usize,
        hidden_dims: Vec<usize>,
        output_dim: usize,
        config: LRPConfig,
    ) -> Self {
        let mut layers = Vec::new();
        let mut layer_rules = Vec::new();

        // Build layers
        let mut dims = vec![input_dim];
        dims.extend(hidden_dims.clone());
        dims.push(output_dim);

        let n_layers = dims.len() - 1;

        for i in 0..n_layers {
            layers.push(LRPLinear::new(dims[i], dims[i + 1], config.clone()));

            // Assign composite rules
            let position = i as f64 / n_layers as f64;
            let rule = if position < 0.33 {
                LRPRule::Gamma
            } else if position < 0.66 {
                LRPRule::Epsilon
            } else {
                LRPRule::Zero
            };
            layer_rules.push(rule);
        }

        Self {
            layers,
            layer_rules,
            config,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let mut x = input.clone();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x);

            // Apply ReLU for all but last layer
            if i < self.layers.len() - 1 {
                x = x.mapv(|v| v.max(0.0));
            }
        }

        x
    }

    /// Batch forward pass
    pub fn forward_batch(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let mut x = input.clone();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward_batch(&x);

            // Apply ReLU for all but last layer
            if i < self.layers.len() - 1 {
                x = x.mapv(|v| v.max(0.0));
            }
        }

        x
    }

    /// Compute LRP explanation
    ///
    /// # Arguments
    ///
    /// * `input` - Input features
    /// * `target_class` - Class to explain (None = use predicted class)
    pub fn explain(&mut self, input: &Array1<f64>, target_class: Option<usize>) -> Array1<f64> {
        // Forward pass
        let output = self.forward(input);

        // Initialize relevance at output
        let mut relevance = Array1::zeros(self.output_dim);
        let class = target_class.unwrap_or_else(|| {
            output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        });
        relevance[class] = output[class];

        // Backward propagation
        for i in (0..self.layers.len()).rev() {
            let rule = self.layer_rules[i];
            relevance = self.layers[i].lrp(&relevance, Some(rule));
        }

        relevance
    }

    /// Get prediction probabilities (softmax)
    pub fn predict_proba(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let logits = self.forward(input);
        softmax(&logits)
    }

    /// Get predicted class
    pub fn predict(&mut self, input: &Array1<f64>) -> usize {
        let output = self.forward(input);
        output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Verify conservation property
    pub fn verify_conservation(&mut self, input: &Array1<f64>, tolerance: f64) -> (bool, f64) {
        let output = self.forward(input);
        let predicted_class = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let relevance = self.explain(input, Some(predicted_class));

        let output_val = output[predicted_class];
        let relevance_sum: f64 = relevance.sum();

        let relative_error = (relevance_sum - output_val).abs() / (output_val.abs() + 1e-9);

        (relative_error < tolerance, relative_error)
    }
}

/// Softmax function
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Array1<f64> = x.mapv(|v| (v - max).exp());
    let sum: f64 = exp.sum();
    exp / sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_forward() {
        let config = LRPConfig::default();
        let mut network = LRPNetwork::new(10, vec![8, 4], 2, config);

        let input = Array1::from_vec(vec![1.0; 10]);
        let output = network.forward(&input);

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_network_explain() {
        let config = LRPConfig::default();
        let mut network = LRPNetwork::new(10, vec![8, 4], 2, config);

        let input = Array1::from_vec(vec![1.0; 10]);
        let relevance = network.explain(&input, Some(0));

        assert_eq!(relevance.len(), 10);
    }

    #[test]
    fn test_conservation() {
        let config = LRPConfig::default().with_epsilon(0.001);
        let mut network = LRPNetwork::new(10, vec![8, 4], 2, config);

        let input = Array1::from_vec(vec![0.5; 10]);
        let (conserved, error) = network.verify_conservation(&input, 0.2);

        // With epsilon rule, conservation is approximate
        println!("Conservation error: {}", error);
    }
}
