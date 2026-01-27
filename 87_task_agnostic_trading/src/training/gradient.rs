//! Gradient harmonization for multi-task learning
//!
//! Prevents any single task from dominating the shared encoder's training
//! by balancing gradient magnitudes and directions.

use serde::{Deserialize, Serialize};

/// Method for harmonizing gradients across tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarmonizationMethod {
    /// No harmonization - raw gradient sum
    None,
    /// GradNorm: normalize gradient magnitudes
    GradNorm,
    /// PCGrad: project conflicting gradients
    PCGrad,
    /// Uncertainty weighting: weight by task uncertainty
    UncertaintyWeighting,
}

/// Configuration for gradient harmonization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonizerConfig {
    /// Harmonization method
    pub method: HarmonizationMethod,
    /// Alpha parameter for GradNorm
    pub grad_norm_alpha: f64,
    /// Learning rate for task weights
    pub weight_lr: f64,
    /// Minimum task weight
    pub min_weight: f64,
    /// Maximum task weight
    pub max_weight: f64,
}

impl Default for HarmonizerConfig {
    fn default() -> Self {
        Self {
            method: HarmonizationMethod::GradNorm,
            grad_norm_alpha: 1.5,
            weight_lr: 0.01,
            min_weight: 0.1,
            max_weight: 5.0,
        }
    }
}

/// Gradient harmonizer that balances multi-task gradients
pub struct GradientHarmonizer {
    config: HarmonizerConfig,
    /// Current task weights
    task_weights: Vec<f64>,
    /// Initial loss values for relative loss tracking
    initial_losses: Option<Vec<f64>>,
    /// Running loss ratios
    loss_ratios: Vec<f64>,
}

impl GradientHarmonizer {
    /// Create a new gradient harmonizer
    pub fn new(config: HarmonizerConfig, n_tasks: usize) -> Self {
        Self {
            config,
            task_weights: vec![1.0; n_tasks],
            initial_losses: None,
            loss_ratios: vec![1.0; n_tasks],
        }
    }

    /// Update task weights based on current losses
    pub fn update_weights(&mut self, task_losses: &[f64]) {
        match self.config.method {
            HarmonizationMethod::None => {
                // Keep all weights at 1.0
            }
            HarmonizationMethod::GradNorm => {
                self.grad_norm_update(task_losses);
            }
            HarmonizationMethod::PCGrad => {
                self.pcgrad_update(task_losses);
            }
            HarmonizationMethod::UncertaintyWeighting => {
                self.uncertainty_update(task_losses);
            }
        }
    }

    /// Get current task weights
    pub fn weights(&self) -> &[f64] {
        &self.task_weights
    }

    /// Compute weighted loss from individual task losses
    pub fn weighted_loss(&self, task_losses: &[f64]) -> f64 {
        task_losses
            .iter()
            .zip(&self.task_weights)
            .map(|(loss, weight)| loss * weight)
            .sum()
    }

    /// Get harmonization statistics
    pub fn stats(&self) -> HarmonizationStats {
        let mean_weight = self.task_weights.iter().sum::<f64>() / self.task_weights.len() as f64;
        let max_weight = self.task_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_weight = self.task_weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let weight_variance = self
            .task_weights
            .iter()
            .map(|w| (w - mean_weight).powi(2))
            .sum::<f64>()
            / self.task_weights.len() as f64;

        HarmonizationStats {
            task_weights: self.task_weights.clone(),
            mean_weight,
            max_weight,
            min_weight,
            weight_variance,
            loss_ratios: self.loss_ratios.clone(),
        }
    }

    // --- Private methods ---

    /// GradNorm: adjust weights so all tasks train at similar rates
    fn grad_norm_update(&mut self, task_losses: &[f64]) {
        // Initialize baseline losses
        if self.initial_losses.is_none() {
            self.initial_losses = Some(task_losses.to_vec());
        }

        let initial = self.initial_losses.as_ref().unwrap();

        // Compute loss ratios (current / initial)
        for i in 0..task_losses.len() {
            if initial[i] > 1e-10 {
                self.loss_ratios[i] = task_losses[i] / initial[i];
            }
        }

        // Compute average loss ratio
        let avg_ratio: f64 =
            self.loss_ratios.iter().sum::<f64>() / self.loss_ratios.len() as f64;

        if avg_ratio < 1e-10 {
            return;
        }

        // Update weights: tasks that train slower get higher weights
        let alpha = self.config.grad_norm_alpha;
        for i in 0..self.task_weights.len() {
            let relative_rate = self.loss_ratios[i] / avg_ratio;
            let target_weight = relative_rate.powf(-alpha);

            // Smooth update
            self.task_weights[i] += self.config.weight_lr * (target_weight - self.task_weights[i]);

            // Clamp weights
            self.task_weights[i] = self.task_weights[i]
                .clamp(self.config.min_weight, self.config.max_weight);
        }

        // Normalize so weights sum to n_tasks
        let n = self.task_weights.len() as f64;
        let sum: f64 = self.task_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.task_weights {
                *w = *w * n / sum;
            }
        }
    }

    /// PCGrad: project conflicting gradients (simplified version)
    fn pcgrad_update(&mut self, task_losses: &[f64]) {
        // In a full implementation, this would operate on actual gradient vectors.
        // Here we approximate by detecting conflicting loss trends.
        if self.initial_losses.is_none() {
            self.initial_losses = Some(task_losses.to_vec());
            return;
        }

        let prev = self.initial_losses.as_ref().unwrap();
        let n = task_losses.len();

        // Detect if tasks have conflicting loss changes
        let loss_changes: Vec<f64> = task_losses
            .iter()
            .zip(prev.iter())
            .map(|(curr, prev)| curr - prev)
            .collect();

        for i in 0..n {
            let mut conflict_count = 0;
            for j in 0..n {
                if i != j && loss_changes[i] * loss_changes[j] < 0.0 {
                    conflict_count += 1;
                }
            }
            // Reduce weight for tasks that conflict with many others
            let conflict_ratio = conflict_count as f64 / (n - 1).max(1) as f64;
            self.task_weights[i] *= 1.0 - 0.1 * conflict_ratio;
            self.task_weights[i] = self.task_weights[i]
                .clamp(self.config.min_weight, self.config.max_weight);
        }

        self.initial_losses = Some(task_losses.to_vec());
    }

    /// Uncertainty weighting: weight inversely by loss (lower loss = lower uncertainty)
    fn uncertainty_update(&mut self, task_losses: &[f64]) {
        let total: f64 = task_losses.iter().sum();
        if total < 1e-10 {
            return;
        }

        // Higher loss -> higher uncertainty -> lower weight
        for (i, loss) in task_losses.iter().enumerate() {
            let log_sigma = (loss + 1e-10).ln();
            // Weight = 1 / (2 * sigma^2) ≈ 1 / (2 * exp(2 * log_sigma))
            self.task_weights[i] = 1.0 / (2.0 * (2.0 * log_sigma).exp() + 1e-10);
            self.task_weights[i] = self.task_weights[i]
                .clamp(self.config.min_weight, self.config.max_weight);
        }

        // Normalize
        let n = self.task_weights.len() as f64;
        let sum: f64 = self.task_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.task_weights {
                *w = *w * n / sum;
            }
        }
    }
}

/// Statistics about gradient harmonization
#[derive(Debug, Clone)]
pub struct HarmonizationStats {
    /// Current task weights
    pub task_weights: Vec<f64>,
    /// Mean weight
    pub mean_weight: f64,
    /// Maximum weight
    pub max_weight: f64,
    /// Minimum weight
    pub min_weight: f64,
    /// Variance of weights
    pub weight_variance: f64,
    /// Current loss ratios
    pub loss_ratios: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_norm() {
        let config = HarmonizerConfig {
            method: HarmonizationMethod::GradNorm,
            ..Default::default()
        };
        let mut harmonizer = GradientHarmonizer::new(config, 4);

        // Initial losses
        harmonizer.update_weights(&[1.0, 0.5, 2.0, 1.5]);
        assert_eq!(harmonizer.weights().len(), 4);

        // Second update with different losses
        harmonizer.update_weights(&[0.8, 0.6, 1.5, 1.0]);
        let stats = harmonizer.stats();
        assert!(stats.weight_variance >= 0.0);
    }

    #[test]
    fn test_weighted_loss() {
        let config = HarmonizerConfig::default();
        let harmonizer = GradientHarmonizer::new(config, 3);
        let losses = vec![1.0, 2.0, 3.0];
        let total = harmonizer.weighted_loss(&losses);
        assert!((total - 6.0).abs() < 1e-10);
    }
}
