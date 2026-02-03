//! LRP-enabled linear layer

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::Normal;

use super::config::LRPConfig;
use super::lrp::{LRPRule, LRPRules};

/// Linear layer with LRP support
#[derive(Debug, Clone)]
pub struct LRPLinear {
    /// Weight matrix [in_features, out_features]
    pub weights: Array2<f64>,
    /// Bias vector [out_features]
    pub bias: Array1<f64>,
    /// Stored activations for LRP
    activations: Option<Array1<f64>>,
    /// LRP configuration
    config: LRPConfig,
}

impl LRPLinear {
    /// Create a new linear layer with random initialization
    pub fn new(in_features: usize, out_features: usize, config: LRPConfig) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let std = (2.0 / (in_features + out_features) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = Array2::from_shape_fn((in_features, out_features), |_| rng.sample(normal));
        let bias = Array1::zeros(out_features);

        Self {
            weights,
            bias,
            activations: None,
            config,
        }
    }

    /// Create with specific weights
    pub fn with_weights(
        weights: Array2<f64>,
        bias: Array1<f64>,
        config: LRPConfig,
    ) -> Self {
        Self {
            weights,
            bias,
            activations: None,
            config,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        // Store activations for LRP
        self.activations = Some(input.clone());

        // Linear transformation: output = input @ weights + bias
        let output = input.dot(&self.weights) + &self.bias;

        output
    }

    /// Batch forward pass
    pub fn forward_batch(&mut self, input: &Array2<f64>) -> Array2<f64> {
        // For batch, we store the last sample's activations (simplified)
        let last_row = input.row(input.nrows() - 1).to_owned();
        self.activations = Some(last_row);

        // Batch linear transformation
        let output = input.dot(&self.weights);

        // Add bias to each row
        let mut result = output;
        for mut row in result.rows_mut() {
            row += &self.bias;
        }

        result
    }

    /// LRP backward pass
    pub fn lrp(&self, relevance: &Array1<f64>, rule: Option<LRPRule>) -> Array1<f64> {
        let activations = self.activations.as_ref()
            .expect("No activations stored. Call forward() first.");

        let rule = rule.unwrap_or(self.config.default_rule);

        LRPRules::apply(
            rule,
            activations,
            &self.weights,
            relevance,
            self.config.epsilon,
            self.config.gamma,
            self.config.alpha,
            self.config.beta,
        )
    }

    /// Get input features
    pub fn in_features(&self) -> usize {
        self.weights.nrows()
    }

    /// Get output features
    pub fn out_features(&self) -> usize {
        self.weights.ncols()
    }

    /// Update weights (for training)
    pub fn update_weights(&mut self, gradient: &Array2<f64>, learning_rate: f64) {
        self.weights = &self.weights - &(learning_rate * gradient);
    }

    /// Update bias (for training)
    pub fn update_bias(&mut self, gradient: &Array1<f64>, learning_rate: f64) {
        self.bias = &self.bias - &(learning_rate * gradient);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_forward() {
        let config = LRPConfig::default();
        let mut layer = LRPLinear::new(3, 2, config);

        let input = array![1.0, 2.0, 3.0];
        let output = layer.forward(&input);

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_lrp() {
        let config = LRPConfig::default();
        let mut layer = LRPLinear::new(3, 2, config);

        let input = array![1.0, 2.0, 3.0];
        let _ = layer.forward(&input);

        let relevance = array![1.0, 0.5];
        let r_prev = layer.lrp(&relevance, None);

        assert_eq!(r_prev.len(), 3);
    }
}
