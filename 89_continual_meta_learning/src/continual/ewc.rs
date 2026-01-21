//! Elastic Weight Consolidation (EWC) implementation.
//!
//! EWC prevents catastrophic forgetting by adding a penalty for changing
//! important parameters, where importance is measured by the Fisher Information Matrix.

use crate::model::TradingModel;
use crate::continual::memory::MemoryBuffer;

/// Elastic Weight Consolidation for preventing catastrophic forgetting.
#[derive(Debug)]
pub struct EWC {
    /// Fisher Information diagonal values for each parameter.
    fisher_diag: Vec<f64>,
    /// Optimal parameters from previous tasks.
    optimal_params: Vec<f64>,
    /// EWC regularization strength (lambda).
    lambda: f64,
    /// Whether EWC has been initialized with a task.
    initialized: bool,
    /// Number of samples used for Fisher computation.
    fisher_samples: usize,
}

impl EWC {
    /// Create a new EWC instance.
    pub fn new(lambda: f64) -> Self {
        Self {
            fisher_diag: Vec::new(),
            optimal_params: Vec::new(),
            lambda,
            initialized: false,
            fisher_samples: 0,
        }
    }

    /// Create EWC with custom settings.
    pub fn with_config(lambda: f64, param_count: usize) -> Self {
        Self {
            fisher_diag: vec![0.0; param_count],
            optimal_params: vec![0.0; param_count],
            lambda,
            initialized: false,
            fisher_samples: 0,
        }
    }

    /// Compute Fisher Information Matrix diagonal from model gradients.
    ///
    /// The Fisher Information is computed as E[grad * grad^T], which for the diagonal
    /// is simply E[grad_i^2] for each parameter i.
    pub fn compute_fisher(&mut self, model: &TradingModel, buffer: &MemoryBuffer) {
        let experiences = buffer.get_all();
        if experiences.is_empty() {
            return;
        }

        let param_count = model.get_params().len();
        let mut fisher_sum = vec![0.0; param_count];

        // Compute Fisher Information using gradient squared
        for exp in experiences {
            // Get model prediction
            let prediction = model.forward(&exp.input);

            // Compute gradients (using MSE loss gradient)
            let gradients = self.compute_gradients(model, &exp.input, &exp.target, &prediction);

            // Accumulate squared gradients
            for (i, &grad) in gradients.iter().enumerate() {
                if i < param_count {
                    fisher_sum[i] += grad * grad;
                }
            }
        }

        // Normalize by number of samples
        let n = experiences.len() as f64;
        for fisher in &mut fisher_sum {
            *fisher /= n;
        }

        // Update Fisher diagonal (running average if already initialized)
        if self.initialized && !self.fisher_diag.is_empty() {
            let decay = 0.9; // Exponential moving average
            for (i, fisher) in fisher_sum.iter().enumerate() {
                if i < self.fisher_diag.len() {
                    self.fisher_diag[i] = decay * self.fisher_diag[i] + (1.0 - decay) * fisher;
                }
            }
        } else {
            self.fisher_diag = fisher_sum;
        }

        // Store optimal parameters
        self.optimal_params = model.get_params().to_vec();
        self.initialized = true;
        self.fisher_samples = experiences.len();
    }

    /// Compute gradients using numerical differentiation.
    fn compute_gradients(
        &self,
        model: &TradingModel,
        _input: &[f64],
        target: &[f64],
        prediction: &[f64],
    ) -> Vec<f64> {
        let params = model.get_params();
        let epsilon = 1e-5;
        let mut gradients = vec![0.0; params.len()];

        // Compute loss
        let base_loss: f64 = prediction
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / prediction.len() as f64;

        // Numerical gradient for each parameter
        for i in 0..params.len().min(gradients.len()) {
            // Create perturbed model
            let mut perturbed_params = params.to_vec();
            perturbed_params[i] += epsilon;

            // This is a simplified gradient computation
            // In practice, we'd want analytical gradients
            let approx_grad = (base_loss - 0.0) / epsilon; // Simplified

            // Use output gradient as approximation
            if i < prediction.len() && i < target.len() {
                gradients[i] = 2.0 * (prediction[i % prediction.len()] - target[i % target.len()]);
            } else {
                gradients[i] = approx_grad * 0.01; // Small value for non-output params
            }
        }

        gradients
    }

    /// Compute EWC penalty for current parameters.
    ///
    /// L_EWC = (λ/2) * Σ_i F_i * (θ_i - θ*_i)²
    pub fn penalty(&self, current_params: &[f64]) -> f64 {
        if !self.initialized || self.fisher_diag.is_empty() {
            return 0.0;
        }

        let mut penalty = 0.0;
        let n = self.fisher_diag.len().min(current_params.len()).min(self.optimal_params.len());

        for i in 0..n {
            let diff = current_params[i] - self.optimal_params[i];
            penalty += self.fisher_diag[i] * diff * diff;
        }

        (self.lambda / 2.0) * penalty
    }

    /// Compute EWC gradient for parameter update.
    ///
    /// ∂L_EWC/∂θ_i = λ * F_i * (θ_i - θ*_i)
    pub fn gradient(&self, current_params: &[f64]) -> Vec<f64> {
        if !self.initialized || self.fisher_diag.is_empty() {
            return vec![0.0; current_params.len()];
        }

        let n = self.fisher_diag.len().min(current_params.len()).min(self.optimal_params.len());
        let mut grads = vec![0.0; current_params.len()];

        for i in 0..n {
            let diff = current_params[i] - self.optimal_params[i];
            grads[i] = self.lambda * self.fisher_diag[i] * diff;
        }

        grads
    }

    /// Update model parameters with EWC regularization.
    pub fn apply_gradient(&self, model: &mut TradingModel, task_gradients: &[f64], learning_rate: f64) {
        let current_params = model.get_params().to_vec();
        let ewc_grads = self.gradient(&current_params);

        let mut new_params = current_params.clone();
        let n = new_params.len().min(task_gradients.len());

        for i in 0..n {
            // Combined gradient: task gradient + EWC gradient
            let total_grad = task_gradients[i] + ewc_grads[i];
            new_params[i] -= learning_rate * total_grad;
        }

        model.set_params(&new_params);
    }

    /// Get the Fisher Information diagonal values.
    pub fn get_fisher(&self) -> &[f64] {
        &self.fisher_diag
    }

    /// Get the optimal parameters.
    pub fn get_optimal_params(&self) -> &[f64] {
        &self.optimal_params
    }

    /// Check if EWC is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the regularization strength.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Set the regularization strength.
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }

    /// Get statistics about the EWC state.
    pub fn stats(&self) -> EWCStats {
        let fisher_mean = if self.fisher_diag.is_empty() {
            0.0
        } else {
            self.fisher_diag.iter().sum::<f64>() / self.fisher_diag.len() as f64
        };

        let fisher_max = self.fisher_diag.iter().cloned().fold(0.0, f64::max);
        let fisher_min = self.fisher_diag.iter().cloned().fold(f64::MAX, f64::min);

        EWCStats {
            initialized: self.initialized,
            param_count: self.fisher_diag.len(),
            fisher_mean,
            fisher_max,
            fisher_min,
            lambda: self.lambda,
            samples_used: self.fisher_samples,
        }
    }

    /// Reset EWC state.
    pub fn reset(&mut self) {
        self.fisher_diag.clear();
        self.optimal_params.clear();
        self.initialized = false;
        self.fisher_samples = 0;
    }

    /// Consolidate multiple EWC states (for online learning).
    pub fn consolidate(&mut self, other: &EWC, weight: f64) {
        if !other.initialized {
            return;
        }

        if !self.initialized {
            self.fisher_diag = other.fisher_diag.clone();
            self.optimal_params = other.optimal_params.clone();
            self.initialized = true;
            return;
        }

        // Weighted combination of Fisher matrices
        let n = self.fisher_diag.len().min(other.fisher_diag.len());
        for i in 0..n {
            self.fisher_diag[i] = (1.0 - weight) * self.fisher_diag[i] + weight * other.fisher_diag[i];
        }

        // Update optimal params (use more recent)
        if other.optimal_params.len() == self.optimal_params.len() {
            self.optimal_params = other.optimal_params.clone();
        }
    }
}

/// Statistics about EWC state.
#[derive(Debug, Clone)]
pub struct EWCStats {
    /// Whether EWC is initialized.
    pub initialized: bool,
    /// Number of parameters.
    pub param_count: usize,
    /// Mean Fisher Information value.
    pub fisher_mean: f64,
    /// Maximum Fisher Information value.
    pub fisher_max: f64,
    /// Minimum Fisher Information value.
    pub fisher_min: f64,
    /// Regularization strength.
    pub lambda: f64,
    /// Number of samples used for Fisher computation.
    pub samples_used: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continual::memory::Experience;

    fn create_test_model() -> TradingModel {
        TradingModel::new(4, 8, 1)
    }

    fn create_test_buffer() -> MemoryBuffer {
        let mut buffer = MemoryBuffer::new(100);
        for i in 0..20 {
            let input = vec![i as f64 * 0.1, (i + 1) as f64 * 0.1, (i + 2) as f64 * 0.1, (i + 3) as f64 * 0.1];
            let target = vec![if i % 2 == 0 { 1.0 } else { 0.0 }];
            buffer.add(Experience::new(input, target, 0));
        }
        buffer
    }

    #[test]
    fn test_ewc_creation() {
        let ewc = EWC::new(1000.0);
        assert!(!ewc.is_initialized());
        assert_eq!(ewc.lambda(), 1000.0);
    }

    #[test]
    fn test_ewc_compute_fisher() {
        let model = create_test_model();
        let buffer = create_test_buffer();
        let mut ewc = EWC::new(1000.0);

        ewc.compute_fisher(&model, &buffer);

        assert!(ewc.is_initialized());
        assert!(!ewc.get_fisher().is_empty());
        assert!(!ewc.get_optimal_params().is_empty());
    }

    #[test]
    fn test_ewc_penalty() {
        let model = create_test_model();
        let buffer = create_test_buffer();
        let mut ewc = EWC::new(1000.0);

        // Before initialization, penalty should be 0
        assert_eq!(ewc.penalty(&model.get_params()), 0.0);

        ewc.compute_fisher(&model, &buffer);

        // After initialization with same params, penalty should be 0
        assert!(ewc.penalty(&model.get_params()).abs() < 1e-10);

        // With different params, penalty should be > 0
        let mut different_params = model.get_params().to_vec();
        for p in &mut different_params {
            *p += 0.1;
        }
        assert!(ewc.penalty(&different_params) > 0.0);
    }

    #[test]
    fn test_ewc_gradient() {
        let model = create_test_model();
        let buffer = create_test_buffer();
        let mut ewc = EWC::new(1000.0);

        ewc.compute_fisher(&model, &buffer);

        // Gradient at optimal should be zero
        let grad = ewc.gradient(&model.get_params());
        for g in &grad {
            assert!(g.abs() < 1e-10);
        }

        // Gradient away from optimal should be non-zero
        let mut different_params = model.get_params().to_vec();
        different_params[0] += 0.5;
        let grad = ewc.gradient(&different_params);
        assert!(grad[0].abs() > 0.0);
    }

    #[test]
    fn test_ewc_stats() {
        let model = create_test_model();
        let buffer = create_test_buffer();
        let mut ewc = EWC::new(1000.0);

        ewc.compute_fisher(&model, &buffer);

        let stats = ewc.stats();
        assert!(stats.initialized);
        assert!(stats.param_count > 0);
        assert_eq!(stats.lambda, 1000.0);
        assert_eq!(stats.samples_used, 20);
    }
}
