//! Online prediction and adaptive learning

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Online learner with forgetting for non-stationary environments
#[derive(Debug, Clone)]
pub struct OnlineLearner {
    /// Current output weights
    w_out: Array2<f64>,
    /// Covariance matrix inverse (for RLS)
    p: Array2<f64>,
    /// Forgetting factor
    forgetting_factor: f64,
    /// Learning rate (for gradient descent methods)
    learning_rate: f64,
    /// Method type
    method: OnlineMethod,
    /// Number of updates performed
    update_count: usize,
}

/// Online learning method
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OnlineMethod {
    /// Recursive Least Squares
    RLS,
    /// Stochastic Gradient Descent
    SGD,
    /// Least Mean Squares
    LMS,
    /// Normalized LMS
    NLMS,
}

impl OnlineLearner {
    /// Create new online learner with RLS
    pub fn new_rls(state_dim: usize, output_dim: usize, forgetting_factor: f64) -> Self {
        Self {
            w_out: Array2::zeros((output_dim, state_dim)),
            p: Array2::eye(state_dim) * 1000.0,
            forgetting_factor,
            learning_rate: 0.0,
            method: OnlineMethod::RLS,
            update_count: 0,
        }
    }

    /// Create new online learner with SGD
    pub fn new_sgd(state_dim: usize, output_dim: usize, learning_rate: f64) -> Self {
        Self {
            w_out: Array2::zeros((output_dim, state_dim)),
            p: Array2::zeros((state_dim, state_dim)),
            forgetting_factor: 1.0,
            learning_rate,
            method: OnlineMethod::SGD,
            update_count: 0,
        }
    }

    /// Create new online learner with LMS
    pub fn new_lms(state_dim: usize, output_dim: usize, learning_rate: f64) -> Self {
        Self {
            w_out: Array2::zeros((output_dim, state_dim)),
            p: Array2::zeros((state_dim, state_dim)),
            forgetting_factor: 1.0,
            learning_rate,
            method: OnlineMethod::LMS,
            update_count: 0,
        }
    }

    /// Create new online learner with Normalized LMS
    pub fn new_nlms(state_dim: usize, output_dim: usize, learning_rate: f64) -> Self {
        Self {
            w_out: Array2::zeros((output_dim, state_dim)),
            p: Array2::zeros((state_dim, state_dim)),
            forgetting_factor: 1.0,
            learning_rate,
            method: OnlineMethod::NLMS,
            update_count: 0,
        }
    }

    /// Initialize weights from pretrained model
    pub fn with_initial_weights(mut self, w_out: Array2<f64>) -> Self {
        self.w_out = w_out;
        self
    }

    /// Predict output for given state
    pub fn predict(&self, state: &Array1<f64>) -> Array1<f64> {
        self.w_out.dot(state)
    }

    /// Update weights with new observation
    pub fn update(&mut self, state: &Array1<f64>, target: &Array1<f64>) {
        match self.method {
            OnlineMethod::RLS => self.update_rls(state, target),
            OnlineMethod::SGD => self.update_sgd(state, target),
            OnlineMethod::LMS => self.update_lms(state, target),
            OnlineMethod::NLMS => self.update_nlms(state, target),
        }
        self.update_count += 1;
    }

    /// RLS update
    fn update_rls(&mut self, state: &Array1<f64>, target: &Array1<f64>) {
        // Gain vector: k = P * x / (λ + x^T * P * x)
        let px = self.p.dot(state);
        let xpx: f64 = state.dot(&px);
        let denom = self.forgetting_factor + xpx;
        let k = &px / denom;

        // Error
        let prediction = self.w_out.dot(state);
        let error = target - &prediction;

        // Update weights
        for i in 0..self.w_out.nrows() {
            for j in 0..self.w_out.ncols() {
                self.w_out[[i, j]] += error[i] * k[j];
            }
        }

        // Update covariance
        let n = self.p.nrows();
        for i in 0..n {
            for j in 0..n {
                self.p[[i, j]] = (self.p[[i, j]] - k[i] * px[j]) / self.forgetting_factor;
            }
        }
    }

    /// SGD update
    fn update_sgd(&mut self, state: &Array1<f64>, target: &Array1<f64>) {
        let prediction = self.w_out.dot(state);
        let error = target - &prediction;

        // Gradient descent: W += η * error * x^T
        for i in 0..self.w_out.nrows() {
            for j in 0..self.w_out.ncols() {
                self.w_out[[i, j]] += self.learning_rate * error[i] * state[j];
            }
        }
    }

    /// LMS update (same as SGD but typically with smaller learning rate)
    fn update_lms(&mut self, state: &Array1<f64>, target: &Array1<f64>) {
        self.update_sgd(state, target);
    }

    /// Normalized LMS update
    fn update_nlms(&mut self, state: &Array1<f64>, target: &Array1<f64>) {
        let prediction = self.w_out.dot(state);
        let error = target - &prediction;

        // Normalize by state energy
        let state_norm_sq: f64 = state.dot(state);
        let epsilon = 1e-10; // Prevent division by zero
        let normalized_lr = self.learning_rate / (epsilon + state_norm_sq);

        for i in 0..self.w_out.nrows() {
            for j in 0..self.w_out.ncols() {
                self.w_out[[i, j]] += normalized_lr * error[i] * state[j];
            }
        }
    }

    /// Get current weights
    pub fn weights(&self) -> &Array2<f64> {
        &self.w_out
    }

    /// Get number of updates performed
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// Reset learner state
    pub fn reset(&mut self) {
        self.w_out.fill(0.0);
        if matches!(self.method, OnlineMethod::RLS) {
            let n = self.p.nrows();
            self.p = Array2::eye(n) * 1000.0;
        }
        self.update_count = 0;
    }
}

/// Sliding window predictor for concept drift detection
pub struct AdaptivePredictor {
    /// Online learner
    learner: OnlineLearner,
    /// Recent errors for drift detection
    error_history: Vec<f64>,
    /// Window size for drift detection
    window_size: usize,
    /// Drift threshold
    drift_threshold: f64,
    /// Current mean error
    current_error_mean: f64,
    /// Drift detected flag
    drift_detected: bool,
}

impl AdaptivePredictor {
    /// Create new adaptive predictor
    pub fn new(
        learner: OnlineLearner,
        window_size: usize,
        drift_threshold: f64,
    ) -> Self {
        Self {
            learner,
            error_history: Vec::with_capacity(window_size * 2),
            window_size,
            drift_threshold,
            current_error_mean: 0.0,
            drift_detected: false,
        }
    }

    /// Predict and update
    pub fn predict_update(
        &mut self,
        state: &Array1<f64>,
        target: &Array1<f64>,
    ) -> (Array1<f64>, bool) {
        // Predict
        let prediction = self.learner.predict(state);

        // Calculate error
        let error: f64 = (&prediction - target)
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
            .sqrt();

        // Update error history
        self.error_history.push(error);
        if self.error_history.len() > self.window_size * 2 {
            self.error_history.remove(0);
        }

        // Detect drift
        self.drift_detected = self.detect_drift();

        // Update learner
        self.learner.update(state, target);

        (prediction, self.drift_detected)
    }

    /// Detect concept drift using Page-Hinkley test
    fn detect_drift(&mut self) -> bool {
        if self.error_history.len() < self.window_size {
            return false;
        }

        // Calculate mean of recent window
        let recent_start = self.error_history.len().saturating_sub(self.window_size);
        let recent_mean: f64 = self.error_history[recent_start..]
            .iter()
            .sum::<f64>() / self.window_size as f64;

        // Calculate overall mean
        let overall_mean: f64 = self.error_history.iter().sum::<f64>()
            / self.error_history.len() as f64;

        self.current_error_mean = recent_mean;

        // Drift if recent error significantly higher than overall
        (recent_mean - overall_mean).abs() > self.drift_threshold
    }

    /// Get current error mean
    pub fn current_error_mean(&self) -> f64 {
        self.current_error_mean
    }

    /// Check if drift was detected
    pub fn drift_detected(&self) -> bool {
        self.drift_detected
    }

    /// Reset the predictor
    pub fn reset(&mut self) {
        self.learner.reset();
        self.error_history.clear();
        self.current_error_mean = 0.0;
        self.drift_detected = false;
    }
}

/// Multi-step ahead predictor using iterated predictions
pub struct MultiStepPredictor {
    /// Number of steps to predict ahead
    horizon: usize,
    /// Prediction buffer
    predictions: Vec<Array1<f64>>,
}

impl MultiStepPredictor {
    /// Create new multi-step predictor
    pub fn new(horizon: usize) -> Self {
        Self {
            horizon,
            predictions: Vec::with_capacity(horizon),
        }
    }

    /// Predict multiple steps ahead
    pub fn predict_horizon<F>(&mut self, initial_state: &Array1<f64>, mut step_fn: F) -> Vec<Array1<f64>>
    where
        F: FnMut(&Array1<f64>) -> Array1<f64>,
    {
        self.predictions.clear();
        let mut current = initial_state.clone();

        for _ in 0..self.horizon {
            let prediction = step_fn(&current);
            self.predictions.push(prediction.clone());
            current = prediction;
        }

        self.predictions.clone()
    }

    /// Get prediction horizon
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// Set prediction horizon
    pub fn set_horizon(&mut self, horizon: usize) {
        self.horizon = horizon;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rls_learner() {
        let mut learner = OnlineLearner::new_rls(2, 1, 0.99);

        // Learn y = 2*x + 1
        for i in 0..1000 {
            let x = i as f64 / 1000.0;
            let state = Array1::from_vec(vec![1.0, x]);
            let target = Array1::from_vec(vec![1.0 + 2.0 * x]);
            learner.update(&state, &target);
        }

        // Test prediction
        let test_state = Array1::from_vec(vec![1.0, 0.5]);
        let prediction = learner.predict(&test_state);

        // Should predict ~2.0
        assert!((prediction[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_sgd_learner() {
        let mut learner = OnlineLearner::new_sgd(2, 1, 0.1);

        // Learn y = x
        for _ in 0..1000 {
            let state = Array1::from_vec(vec![1.0, 0.5]);
            let target = Array1::from_vec(vec![0.5]);
            learner.update(&state, &target);
        }

        let prediction = learner.predict(&Array1::from_vec(vec![1.0, 0.5]));
        assert!((prediction[0] - 0.5).abs() < 0.2);
    }

    #[test]
    fn test_adaptive_predictor() {
        let learner = OnlineLearner::new_rls(2, 1, 0.99);
        let mut predictor = AdaptivePredictor::new(learner, 50, 0.1);

        // Simulate stable period
        for i in 0..100 {
            let x = i as f64 / 100.0;
            let state = Array1::from_vec(vec![1.0, x]);
            let target = Array1::from_vec(vec![x]);
            let (_, drift) = predictor.predict_update(&state, &target);
            assert!(!drift || i < 50); // No drift expected in stable period
        }
    }

    #[test]
    fn test_multi_step_predictor() {
        let mut predictor = MultiStepPredictor::new(5);

        let initial = Array1::from_vec(vec![1.0]);
        let predictions = predictor.predict_horizon(&initial, |x| x * 2.0);

        assert_eq!(predictions.len(), 5);
        assert!((predictions[0][0] - 2.0).abs() < 1e-10);
        assert!((predictions[1][0] - 4.0).abs() < 1e-10);
    }
}
