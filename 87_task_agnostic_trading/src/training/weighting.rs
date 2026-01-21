//! Task weighting strategies for multi-task learning

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::tasks::TaskType;

/// Task weighting strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightingStrategy {
    /// Equal weights for all tasks
    Equal,
    /// Uncertainty-based weighting (homoscedastic)
    Uncertainty,
    /// Dynamic weighting based on loss magnitude
    DynamicLoss,
    /// Custom fixed weights
    Custom,
}

impl Default for WeightingStrategy {
    fn default() -> Self {
        Self::Equal
    }
}

/// Task weighter for balancing losses
pub struct TaskWeighter {
    strategy: WeightingStrategy,
    /// Custom weights (used for Custom strategy)
    custom_weights: HashMap<TaskType, f64>,
    /// Log variance parameters for uncertainty weighting
    log_vars: HashMap<TaskType, f64>,
    /// Historical losses for dynamic weighting
    loss_history: HashMap<TaskType, Vec<f64>>,
    /// History window size
    history_size: usize,
    /// Learning rate for log_var updates
    var_lr: f64,
}

impl TaskWeighter {
    /// Create a new task weighter
    pub fn new(strategy: WeightingStrategy) -> Self {
        let mut default_weights = HashMap::new();
        for task in [TaskType::Direction, TaskType::Volatility, TaskType::Regime, TaskType::Returns] {
            default_weights.insert(task, 1.0);
        }

        let mut default_log_vars = HashMap::new();
        for task in [TaskType::Direction, TaskType::Volatility, TaskType::Regime, TaskType::Returns] {
            default_log_vars.insert(task, 0.0); // log(1) = 0
        }

        Self {
            strategy,
            custom_weights: default_weights,
            log_vars: default_log_vars,
            loss_history: HashMap::new(),
            history_size: 100,
            var_lr: 0.01,
        }
    }

    /// Set custom weights
    pub fn with_custom_weights(mut self, weights: HashMap<TaskType, f64>) -> Self {
        self.custom_weights = weights;
        self
    }

    /// Get weight for a specific task
    pub fn get_weight(&self, task: TaskType) -> f64 {
        match self.strategy {
            WeightingStrategy::Equal => 1.0,
            WeightingStrategy::Custom => *self.custom_weights.get(&task).unwrap_or(&1.0),
            WeightingStrategy::Uncertainty => {
                let log_var = *self.log_vars.get(&task).unwrap_or(&0.0);
                // Weight = 1 / (2 * var) = 1 / (2 * exp(log_var))
                1.0 / (2.0 * log_var.exp())
            }
            WeightingStrategy::DynamicLoss => {
                self.compute_dynamic_weight(task)
            }
        }
    }

    /// Get all task weights
    pub fn get_all_weights(&self) -> HashMap<TaskType, f64> {
        let tasks = [TaskType::Direction, TaskType::Volatility, TaskType::Regime, TaskType::Returns];
        tasks.iter().map(|&t| (t, self.get_weight(t))).collect()
    }

    /// Update weights based on loss (for adaptive strategies)
    pub fn update(&mut self, task: TaskType, loss: f64) {
        // Update loss history
        self.loss_history.entry(task).or_insert_with(Vec::new).push(loss);
        if self.loss_history[&task].len() > self.history_size {
            self.loss_history.get_mut(&task).unwrap().remove(0);
        }

        // Update log variance for uncertainty weighting
        if self.strategy == WeightingStrategy::Uncertainty {
            let log_var = self.log_vars.entry(task).or_insert(0.0);
            // Gradient: d/d(log_var) [loss / (2*var) + log_var/2] = loss/(2*var) - 1/2
            // Simplified update towards matching loss
            let var = log_var.exp();
            let grad = loss / (2.0 * var) - 0.5;
            *log_var += self.var_lr * grad;
            *log_var = log_var.max(-5.0).min(5.0); // Clip for stability
        }
    }

    /// Compute dynamic weight based on recent losses
    fn compute_dynamic_weight(&self, task: TaskType) -> f64 {
        let history = self.loss_history.get(&task);

        if let Some(losses) = history {
            if !losses.is_empty() {
                let avg_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;
                // Higher loss -> higher weight (to focus learning)
                // But normalized to avoid extreme values
                return (1.0 + avg_loss).ln().max(0.1).min(10.0);
            }
        }

        1.0
    }

    /// Compute total weighted loss
    pub fn weighted_loss(&self, task_losses: &HashMap<TaskType, f64>) -> f64 {
        task_losses.iter()
            .map(|(task, loss)| self.get_weight(*task) * loss)
            .sum()
    }

    /// Get weighted loss with regularization (for uncertainty weighting)
    pub fn weighted_loss_with_reg(&self, task_losses: &HashMap<TaskType, f64>) -> f64 {
        if self.strategy != WeightingStrategy::Uncertainty {
            return self.weighted_loss(task_losses);
        }

        task_losses.iter()
            .map(|(task, loss)| {
                let log_var = *self.log_vars.get(task).unwrap_or(&0.0);
                let var = log_var.exp();
                // L = loss / (2*var) + log(var) / 2
                loss / (2.0 * var) + log_var / 2.0
            })
            .sum()
    }

    /// Get strategy
    pub fn strategy(&self) -> WeightingStrategy {
        self.strategy
    }

    /// Get uncertainty log variances
    pub fn log_variances(&self) -> &HashMap<TaskType, f64> {
        &self.log_vars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_weighting() {
        let weighter = TaskWeighter::new(WeightingStrategy::Equal);

        assert_eq!(weighter.get_weight(TaskType::Direction), 1.0);
        assert_eq!(weighter.get_weight(TaskType::Volatility), 1.0);
    }

    #[test]
    fn test_custom_weighting() {
        let mut weights = HashMap::new();
        weights.insert(TaskType::Direction, 2.0);
        weights.insert(TaskType::Volatility, 0.5);

        let weighter = TaskWeighter::new(WeightingStrategy::Custom)
            .with_custom_weights(weights);

        assert_eq!(weighter.get_weight(TaskType::Direction), 2.0);
        assert_eq!(weighter.get_weight(TaskType::Volatility), 0.5);
    }

    #[test]
    fn test_uncertainty_weighting() {
        let weighter = TaskWeighter::new(WeightingStrategy::Uncertainty);

        // Initial log_var = 0, so var = 1, weight = 1/(2*1) = 0.5
        let weight = weighter.get_weight(TaskType::Direction);
        assert!((weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_loss() {
        let weighter = TaskWeighter::new(WeightingStrategy::Equal);

        let mut losses = HashMap::new();
        losses.insert(TaskType::Direction, 0.5);
        losses.insert(TaskType::Volatility, 0.3);

        let total = weighter.weighted_loss(&losses);
        assert!((total - 0.8).abs() < 1e-10);
    }
}
