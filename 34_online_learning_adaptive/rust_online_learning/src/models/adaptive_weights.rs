//! Adaptive Momentum Weights
//!
//! Implements online learning of optimal momentum factor weights
//! that adapt to changing market conditions.

use super::OnlineModel;
use std::collections::VecDeque;

/// Adaptive Momentum Weights Model
///
/// Uses online gradient descent to learn optimal weights for
/// combining multiple momentum signals.
#[derive(Debug, Clone)]
pub struct AdaptiveMomentumWeights {
    /// Current weights for each factor
    weights: Vec<f64>,
    /// Learning rate
    learning_rate: f64,
    /// Factor names
    factor_names: Vec<String>,
    /// Number of factors
    n_factors: usize,
    /// Number of updates
    n_updates: u64,
    /// Weight history for visualization
    weight_history: VecDeque<Vec<f64>>,
    /// Maximum history size
    max_history: usize,
    /// L2 regularization
    l2_reg: f64,
    /// Momentum coefficient for gradient
    momentum: f64,
    /// Previous gradient for momentum
    prev_gradient: Vec<f64>,
    /// Use exponential weights normalization
    use_softmax: bool,
    /// Temperature for softmax
    temperature: f64,
}

impl AdaptiveMomentumWeights {
    /// Create a new adaptive momentum weights model
    ///
    /// # Arguments
    ///
    /// * `n_factors` - Number of momentum factors
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `factor_names` - Names of momentum factors
    pub fn new(n_factors: usize, learning_rate: f64, factor_names: Vec<String>) -> Self {
        assert_eq!(n_factors, factor_names.len(), "Factor names must match n_factors");

        Self {
            weights: vec![1.0 / n_factors as f64; n_factors],
            learning_rate,
            factor_names,
            n_factors,
            n_updates: 0,
            weight_history: VecDeque::with_capacity(1000),
            max_history: 1000,
            l2_reg: 0.0001,
            momentum: 0.9,
            prev_gradient: vec![0.0; n_factors],
            use_softmax: false,
            temperature: 1.0,
        }
    }

    /// Create with equal initial weights
    pub fn with_equal_weights(n_factors: usize, learning_rate: f64) -> Self {
        let names: Vec<String> = (0..n_factors).map(|i| format!("factor_{}", i)).collect();
        Self::new(n_factors, learning_rate, names)
    }

    /// Enable softmax normalization
    pub fn with_softmax(mut self, temperature: f64) -> Self {
        self.use_softmax = true;
        self.temperature = temperature;
        self
    }

    /// Set L2 regularization
    pub fn with_l2(mut self, l2_reg: f64) -> Self {
        self.l2_reg = l2_reg;
        self
    }

    /// Set momentum coefficient
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Apply softmax normalization to weights
    fn apply_softmax(&self, weights: &[f64]) -> Vec<f64> {
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_weights: Vec<f64> = weights
            .iter()
            .map(|w| ((w - max_w) / self.temperature).exp())
            .collect();
        let sum: f64 = exp_weights.iter().sum();
        exp_weights.iter().map(|w| w / sum).collect()
    }

    /// Normalize weights to sum to 1 (absolute values)
    fn normalize_weights(&mut self) {
        if self.use_softmax {
            self.weights = self.apply_softmax(&self.weights);
        } else {
            let sum: f64 = self.weights.iter().map(|w| w.abs()).sum();
            if sum > 1e-10 {
                for w in self.weights.iter_mut() {
                    *w /= sum;
                }
            }
        }
    }

    /// Get current weights
    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get weight history
    pub fn get_weight_history(&self) -> &VecDeque<Vec<f64>> {
        &self.weight_history
    }

    /// Get factor importance (weights with names)
    pub fn get_factor_importance(&self) -> Vec<(String, f64)> {
        self.factor_names
            .iter()
            .zip(self.weights.iter())
            .map(|(name, weight)| (name.clone(), *weight))
            .collect()
    }

    /// Update with learning rate decay
    pub fn update(&mut self, factor_signals: &[f64], actual_return: f64) {
        assert_eq!(
            factor_signals.len(),
            self.n_factors,
            "Signal dimension mismatch"
        );

        // Compute prediction
        let prediction: f64 = self.weights
            .iter()
            .zip(factor_signals.iter())
            .map(|(w, s)| w * s)
            .sum();

        // Compute error
        let error = actual_return - prediction;

        // Compute gradients (negative of MSE gradient for minimization)
        // Gradient of MSE: d/dw (y - w*x)^2 = -2 * (y - w*x) * x
        let gradients: Vec<f64> = factor_signals
            .iter()
            .zip(self.weights.iter())
            .map(|(xi, wi)| -2.0 * error * xi + 2.0 * self.l2_reg * wi)
            .collect();

        // Apply momentum
        let momentum_gradients: Vec<f64> = gradients
            .iter()
            .zip(self.prev_gradient.iter())
            .map(|(g, pg)| g + self.momentum * pg)
            .collect();

        // Learning rate decay
        let lr = self.learning_rate / (1.0 + 0.001 * (self.n_updates as f64).sqrt());

        // Update weights
        for i in 0..self.n_factors {
            self.weights[i] -= lr * momentum_gradients[i];
        }

        // Store gradient for next iteration
        self.prev_gradient = gradients;

        // Normalize weights
        self.normalize_weights();

        // Store history
        if self.weight_history.len() >= self.max_history {
            self.weight_history.pop_front();
        }
        self.weight_history.push_back(self.weights.clone());

        self.n_updates += 1;
    }

    /// Predict weighted signal
    pub fn predict(&self, factor_signals: &[f64]) -> f64 {
        assert_eq!(
            factor_signals.len(),
            self.n_factors,
            "Signal dimension mismatch"
        );

        self.weights
            .iter()
            .zip(factor_signals.iter())
            .map(|(w, s)| w * s)
            .sum()
    }

    /// Get number of updates performed
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Get the most influential factor
    pub fn best_factor(&self) -> Option<(String, f64)> {
        self.weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, w)| (self.factor_names[i].clone(), *w))
    }

    /// Calculate recent prediction error (MSE over last n samples)
    pub fn recent_mse(&self, predictions: &[f64], actuals: &[f64], n: usize) -> f64 {
        let n = n.min(predictions.len()).min(actuals.len());
        if n == 0 {
            return 0.0;
        }

        let start = predictions.len() - n;
        predictions[start..]
            .iter()
            .zip(actuals[start..].iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / n as f64
    }
}

impl OnlineModel for AdaptiveMomentumWeights {
    fn predict(&self, x: &[f64]) -> f64 {
        self.predict(x)
    }

    fn learn(&mut self, x: &[f64], y: f64) {
        self.update(x, y);
    }

    fn get_params(&self) -> Vec<f64> {
        self.weights.clone()
    }

    fn reset(&mut self) {
        self.weights = vec![1.0 / self.n_factors as f64; self.n_factors];
        self.n_updates = 0;
        self.weight_history.clear();
        self.prev_gradient = vec![0.0; self.n_factors];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_weights_initialization() {
        let model = AdaptiveMomentumWeights::new(
            4,
            0.01,
            vec![
                "mom_1m".to_string(),
                "mom_3m".to_string(),
                "mom_6m".to_string(),
                "mom_12m".to_string(),
            ],
        );

        assert_eq!(model.get_weights().len(), 4);
        assert!((model.get_weights().iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_weights_learning() {
        let mut model = AdaptiveMomentumWeights::with_equal_weights(3, 0.1);

        // Simulate where first factor is always best predictor
        for _ in 0..100 {
            let signals = vec![0.01, 0.001, -0.005];
            let actual = 0.01; // Matches first factor
            model.update(&signals, actual);
        }

        // First factor should have highest weight
        let importance = model.get_factor_importance();
        assert!(importance[0].1 > importance[1].1);
        assert!(importance[0].1 > importance[2].1);
    }

    #[test]
    fn test_weight_normalization() {
        let mut model = AdaptiveMomentumWeights::with_equal_weights(3, 0.1);

        // After updates, weights should still sum to ~1
        for _ in 0..50 {
            model.update(&[0.01, 0.02, 0.03], 0.015);
        }

        let sum: f64 = model.get_weights().iter().map(|w| w.abs()).sum();
        assert!((sum - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_softmax_weights() {
        let model = AdaptiveMomentumWeights::with_equal_weights(3, 0.1).with_softmax(1.0);

        // Weights should sum to 1 with softmax
        let sum: f64 = model.get_weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
