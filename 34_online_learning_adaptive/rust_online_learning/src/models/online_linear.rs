//! Online Linear Regression
//!
//! Implements stochastic gradient descent for linear regression
//! with support for various optimizers.

use super::OnlineModel;

/// Online Linear Regression with SGD
///
/// Updates weights incrementally using stochastic gradient descent.
/// Supports L2 regularization and various learning rate schedules.
#[derive(Debug, Clone)]
pub struct OnlineLinearRegression {
    /// Model weights
    weights: Vec<f64>,
    /// Bias term
    bias: f64,
    /// Learning rate
    learning_rate: f64,
    /// L2 regularization strength
    l2_reg: f64,
    /// Number of features
    n_features: usize,
    /// Number of samples seen
    n_samples: u64,
    /// Running mean for input normalization
    running_mean: Vec<f64>,
    /// Running variance for input normalization
    running_var: Vec<f64>,
    /// Whether to use online normalization
    normalize: bool,
    /// Adam optimizer state: first moment
    m: Vec<f64>,
    /// Adam optimizer state: second moment
    v: Vec<f64>,
    /// Adam beta1
    beta1: f64,
    /// Adam beta2
    beta2: f64,
    /// Use Adam optimizer
    use_adam: bool,
}

impl OnlineLinearRegression {
    /// Create a new online linear regression model
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of input features
    /// * `learning_rate` - Learning rate for SGD
    pub fn new(n_features: usize, learning_rate: f64) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            learning_rate,
            l2_reg: 0.001,
            n_features,
            n_samples: 0,
            running_mean: vec![0.0; n_features],
            running_var: vec![1.0; n_features],
            normalize: true,
            m: vec![0.0; n_features + 1], // +1 for bias
            v: vec![0.0; n_features + 1],
            beta1: 0.9,
            beta2: 0.999,
            use_adam: true,
        }
    }

    /// Create with custom L2 regularization
    pub fn with_l2(mut self, l2_reg: f64) -> Self {
        self.l2_reg = l2_reg;
        self
    }

    /// Disable online normalization
    pub fn without_normalization(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Use simple SGD instead of Adam
    pub fn with_sgd(mut self) -> Self {
        self.use_adam = false;
        self
    }

    /// Normalize input using running statistics
    fn normalize_input(&self, x: &[f64]) -> Vec<f64> {
        if !self.normalize || self.n_samples < 2 {
            return x.to_vec();
        }

        x.iter()
            .zip(self.running_mean.iter())
            .zip(self.running_var.iter())
            .map(|((xi, mean), var)| {
                if *var > 1e-10 {
                    (xi - mean) / var.sqrt()
                } else {
                    xi - mean
                }
            })
            .collect()
    }

    /// Update running statistics
    fn update_statistics(&mut self, x: &[f64]) {
        let n = self.n_samples as f64;

        for i in 0..self.n_features {
            // Welford's online algorithm for mean and variance
            let delta = x[i] - self.running_mean[i];
            self.running_mean[i] += delta / (n + 1.0);
            let delta2 = x[i] - self.running_mean[i];
            self.running_var[i] += (delta * delta2 - self.running_var[i]) / (n + 1.0);
        }
    }

    /// Adam optimizer update
    fn adam_update(&mut self, gradients: &[f64], bias_gradient: f64) {
        let t = self.n_samples as f64 + 1.0;
        let epsilon = 1e-8;

        // Bias correction terms
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        // Update weights
        for i in 0..self.n_features {
            // Update first moment
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradients[i];
            // Update second moment
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradients[i].powi(2);

            // Bias-corrected estimates
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            // Update weight
            self.weights[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        }

        // Update bias
        let bias_idx = self.n_features;
        self.m[bias_idx] = self.beta1 * self.m[bias_idx] + (1.0 - self.beta1) * bias_gradient;
        self.v[bias_idx] = self.beta2 * self.v[bias_idx] + (1.0 - self.beta2) * bias_gradient.powi(2);

        let m_hat = self.m[bias_idx] / bc1;
        let v_hat = self.v[bias_idx] / bc2;
        self.bias -= self.learning_rate * m_hat / (v_hat.sqrt() + epsilon);
    }

    /// Simple SGD update
    fn sgd_update(&mut self, gradients: &[f64], bias_gradient: f64) {
        for i in 0..self.n_features {
            self.weights[i] -= self.learning_rate * gradients[i];
        }
        self.bias -= self.learning_rate * bias_gradient;
    }

    /// Get current learning rate (with decay)
    pub fn effective_learning_rate(&self) -> f64 {
        // Use inverse sqrt decay
        self.learning_rate / (1.0 + 0.01 * (self.n_samples as f64).sqrt())
    }

    /// Get number of samples processed
    pub fn samples_seen(&self) -> u64 {
        self.n_samples
    }

    /// Get model weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get bias term
    pub fn bias(&self) -> f64 {
        self.bias
    }
}

impl OnlineModel for OnlineLinearRegression {
    /// Predict output for input features
    fn predict(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.n_features, "Feature dimension mismatch");

        let x_norm = self.normalize_input(x);

        let mut prediction = self.bias;
        for (w, xi) in self.weights.iter().zip(x_norm.iter()) {
            prediction += w * xi;
        }
        prediction
    }

    /// Learn from a single observation using SGD
    fn learn(&mut self, x: &[f64], y: f64) {
        assert_eq!(x.len(), self.n_features, "Feature dimension mismatch");

        // Update running statistics
        if self.normalize {
            self.update_statistics(x);
        }

        // Normalize input
        let x_norm = self.normalize_input(x);

        // Compute prediction
        let prediction = self.predict(x);

        // Compute error
        let error = prediction - y;

        // Compute gradients (MSE loss with L2 regularization)
        let mut gradients: Vec<f64> = x_norm
            .iter()
            .zip(self.weights.iter())
            .map(|(xi, wi)| 2.0 * error * xi + 2.0 * self.l2_reg * wi)
            .collect();

        let bias_gradient = 2.0 * error;

        // Apply gradient clipping
        let max_grad = 1.0;
        for g in gradients.iter_mut() {
            *g = g.clamp(-max_grad, max_grad);
        }

        // Update weights
        if self.use_adam {
            self.adam_update(&gradients, bias_gradient.clamp(-max_grad, max_grad));
        } else {
            self.sgd_update(&gradients, bias_gradient.clamp(-max_grad, max_grad));
        }

        self.n_samples += 1;
    }

    /// Get current model parameters
    fn get_params(&self) -> Vec<f64> {
        let mut params = self.weights.clone();
        params.push(self.bias);
        params
    }

    /// Reset model to initial state
    fn reset(&mut self) {
        self.weights = vec![0.0; self.n_features];
        self.bias = 0.0;
        self.n_samples = 0;
        self.running_mean = vec![0.0; self.n_features];
        self.running_var = vec![1.0; self.n_features];
        self.m = vec![0.0; self.n_features + 1];
        self.v = vec![0.0; self.n_features + 1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_linear_regression() {
        let mut model = OnlineLinearRegression::new(2, 0.1).without_normalization();

        // Simple linear function: y = 2*x1 + 3*x2 + 1
        let data = vec![
            (vec![1.0, 0.0], 3.0),
            (vec![0.0, 1.0], 4.0),
            (vec![1.0, 1.0], 6.0),
            (vec![2.0, 1.0], 8.0),
        ];

        // Train for multiple epochs
        for _ in 0..100 {
            for (x, y) in &data {
                model.learn(x, *y);
            }
        }

        // Test prediction
        let pred = model.predict(&[1.0, 1.0]);
        assert!((pred - 6.0).abs() < 1.0, "Prediction: {}", pred);
    }

    #[test]
    fn test_model_reset() {
        let mut model = OnlineLinearRegression::new(3, 0.01);
        model.learn(&[1.0, 2.0, 3.0], 10.0);

        assert!(model.samples_seen() > 0);

        model.reset();

        assert_eq!(model.samples_seen(), 0);
        assert!(model.weights().iter().all(|&w| w == 0.0));
    }
}
