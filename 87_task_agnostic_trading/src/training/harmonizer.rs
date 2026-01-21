//! Gradient harmonization methods for multi-task learning

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::tasks::TaskType;

/// Gradient harmonization method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HarmonizerType {
    /// No harmonization (simple sum)
    None,
    /// PCGrad: Project conflicting gradients
    PCGrad,
    /// GradNorm: Normalize gradient magnitudes
    GradNorm,
    /// Gradient vaccine: adaptive conflict resolution
    GradientVaccine,
}

impl Default for HarmonizerType {
    fn default() -> Self {
        Self::PCGrad
    }
}

/// Gradient harmonizer for resolving task conflicts
pub struct GradientHarmonizer {
    method: HarmonizerType,
    /// Historical gradient norms for GradNorm
    grad_norms: HashMap<TaskType, f64>,
    /// Target ratio for GradNorm (usually 1.0 for equal importance)
    target_ratio: f64,
    /// Smoothing factor for exponential moving average
    alpha: f64,
}

impl GradientHarmonizer {
    /// Create a new gradient harmonizer
    pub fn new(method: HarmonizerType) -> Self {
        Self {
            method,
            grad_norms: HashMap::new(),
            target_ratio: 1.0,
            alpha: 0.1,
        }
    }

    /// Set target gradient ratio
    pub fn with_target_ratio(mut self, ratio: f64) -> Self {
        self.target_ratio = ratio;
        self
    }

    /// Harmonize gradients from multiple tasks
    pub fn harmonize(
        &mut self,
        task_gradients: &HashMap<TaskType, Vec<Array2<f64>>>,
    ) -> Vec<Array2<f64>> {
        if task_gradients.is_empty() {
            return vec![];
        }

        match self.method {
            HarmonizerType::None => self.simple_sum(task_gradients),
            HarmonizerType::PCGrad => self.pcgrad(task_gradients),
            HarmonizerType::GradNorm => self.gradnorm(task_gradients),
            HarmonizerType::GradientVaccine => self.gradient_vaccine(task_gradients),
        }
    }

    /// Simple gradient summation (no harmonization)
    fn simple_sum(&self, task_gradients: &HashMap<TaskType, Vec<Array2<f64>>>) -> Vec<Array2<f64>> {
        let first_grads = task_gradients.values().next().unwrap();
        let mut combined: Vec<Array2<f64>> = first_grads.iter().map(|g| g.clone()).collect();

        for (i, grad) in task_gradients.values().skip(1).flat_map(|gs| gs.iter()).enumerate() {
            let param_idx = i % combined.len();
            combined[param_idx] = &combined[param_idx] + grad;
        }

        combined
    }

    /// PCGrad: Project conflicting gradients
    fn pcgrad(&self, task_gradients: &HashMap<TaskType, Vec<Array2<f64>>>) -> Vec<Array2<f64>> {
        let tasks: Vec<_> = task_gradients.keys().collect();
        let n_tasks = tasks.len();

        if n_tasks <= 1 {
            return self.simple_sum(task_gradients);
        }

        let first_grads = task_gradients.values().next().unwrap();
        let n_params = first_grads.len();

        // Process each parameter
        let mut combined = Vec::with_capacity(n_params);

        for param_idx in 0..n_params {
            // Collect gradients for this parameter from all tasks
            let param_grads: Vec<&Array2<f64>> = task_gradients.values()
                .filter_map(|gs| gs.get(param_idx))
                .collect();

            if param_grads.is_empty() {
                continue;
            }

            // Project gradients
            let mut projected: Vec<Array2<f64>> = param_grads.iter().map(|&g| g.clone()).collect();

            for i in 0..projected.len() {
                for j in 0..projected.len() {
                    if i != j {
                        let gi_flat: Vec<f64> = projected[i].iter().cloned().collect();
                        let gj_flat: Vec<f64> = projected[j].iter().cloned().collect();

                        let dot: f64 = gi_flat.iter().zip(&gj_flat).map(|(a, b)| a * b).sum();
                        let gj_norm_sq: f64 = gj_flat.iter().map(|x| x * x).sum();

                        // If conflicting (negative dot product), project out
                        if dot < 0.0 && gj_norm_sq > 1e-10 {
                            let scale = dot / gj_norm_sq;
                            let shape = projected[i].raw_dim();
                            let projected_i: Vec<f64> = gi_flat.iter()
                                .zip(&gj_flat)
                                .map(|(gi, gj)| gi - scale * gj)
                                .collect();
                            projected[i] = Array2::from_shape_vec(shape, projected_i)
                                .expect("Shape mismatch in PCGrad");
                        }
                    }
                }
            }

            // Sum projected gradients
            let mut param_combined = projected[0].clone();
            for grad in projected.iter().skip(1) {
                param_combined = &param_combined + grad;
            }

            combined.push(param_combined);
        }

        combined
    }

    /// GradNorm: Normalize gradient magnitudes
    fn gradnorm(&mut self, task_gradients: &HashMap<TaskType, Vec<Array2<f64>>>) -> Vec<Array2<f64>> {
        // Compute gradient norms for each task
        let mut task_norms: HashMap<TaskType, f64> = HashMap::new();

        for (task, grads) in task_gradients {
            let norm: f64 = grads.iter()
                .map(|g| g.iter().map(|x| x * x).sum::<f64>())
                .sum::<f64>()
                .sqrt();
            task_norms.insert(*task, norm);

            // Update historical norm with EMA
            let hist_norm = self.grad_norms.entry(*task).or_insert(norm);
            *hist_norm = self.alpha * norm + (1.0 - self.alpha) * *hist_norm;
        }

        // Compute average norm
        let avg_norm: f64 = task_norms.values().sum::<f64>() / task_norms.len() as f64;

        // Scale gradients to match target norm
        let first_grads = task_gradients.values().next().unwrap();
        let n_params = first_grads.len();
        let mut combined = vec![Array2::zeros(first_grads[0].raw_dim()); n_params];

        for (task, grads) in task_gradients {
            let task_norm = task_norms[task];
            let scale = if task_norm > 1e-10 {
                avg_norm * self.target_ratio / task_norm
            } else {
                1.0
            };

            for (param_idx, grad) in grads.iter().enumerate() {
                if param_idx < combined.len() {
                    combined[param_idx] = &combined[param_idx] + &(grad * scale);
                }
            }
        }

        combined
    }

    /// Gradient vaccine: adaptive conflict resolution
    fn gradient_vaccine(&self, task_gradients: &HashMap<TaskType, Vec<Array2<f64>>>) -> Vec<Array2<f64>> {
        // Simplified gradient vaccine: soft PCGrad with partial projection
        let tasks: Vec<_> = task_gradients.keys().collect();
        let n_tasks = tasks.len();

        if n_tasks <= 1 {
            return self.simple_sum(task_gradients);
        }

        let first_grads = task_gradients.values().next().unwrap();
        let n_params = first_grads.len();
        let mut combined = Vec::with_capacity(n_params);

        let vaccine_strength = 0.5; // How much to project (0 = no projection, 1 = full PCGrad)

        for param_idx in 0..n_params {
            let param_grads: Vec<&Array2<f64>> = task_gradients.values()
                .filter_map(|gs| gs.get(param_idx))
                .collect();

            if param_grads.is_empty() {
                continue;
            }

            let mut vaccinated: Vec<Array2<f64>> = param_grads.iter().map(|&g| g.clone()).collect();

            for i in 0..vaccinated.len() {
                for j in 0..vaccinated.len() {
                    if i != j {
                        let gi_flat: Vec<f64> = vaccinated[i].iter().cloned().collect();
                        let gj_flat: Vec<f64> = vaccinated[j].iter().cloned().collect();

                        let dot: f64 = gi_flat.iter().zip(&gj_flat).map(|(a, b)| a * b).sum();
                        let gj_norm_sq: f64 = gj_flat.iter().map(|x| x * x).sum();

                        if dot < 0.0 && gj_norm_sq > 1e-10 {
                            // Partial projection (vaccine)
                            let scale = vaccine_strength * dot / gj_norm_sq;
                            let shape = vaccinated[i].raw_dim();
                            let vaccinated_i: Vec<f64> = gi_flat.iter()
                                .zip(&gj_flat)
                                .map(|(gi, gj)| gi - scale * gj)
                                .collect();
                            vaccinated[i] = Array2::from_shape_vec(shape, vaccinated_i)
                                .expect("Shape mismatch");
                        }
                    }
                }
            }

            let mut param_combined = vaccinated[0].clone();
            for grad in vaccinated.iter().skip(1) {
                param_combined = &param_combined + grad;
            }

            combined.push(param_combined);
        }

        combined
    }

    /// Get current method
    pub fn method(&self) -> HarmonizerType {
        self.method
    }

    /// Get historical gradient norms
    pub fn gradient_norms(&self) -> &HashMap<TaskType, f64> {
        &self.grad_norms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonizer_creation() {
        let harmonizer = GradientHarmonizer::new(HarmonizerType::PCGrad);
        assert_eq!(harmonizer.method(), HarmonizerType::PCGrad);
    }

    #[test]
    fn test_simple_sum() {
        let harmonizer = GradientHarmonizer::new(HarmonizerType::None);

        let mut task_grads = HashMap::new();
        task_grads.insert(TaskType::Direction, vec![Array2::ones((2, 2))]);
        task_grads.insert(TaskType::Volatility, vec![Array2::ones((2, 2))]);

        let combined = harmonizer.simple_sum(&task_grads);

        assert_eq!(combined.len(), 1);
        assert!((combined[0][[0, 0]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_pcgrad_no_conflict() {
        let harmonizer = GradientHarmonizer::new(HarmonizerType::PCGrad);

        // Aligned gradients (same direction)
        let mut task_grads = HashMap::new();
        task_grads.insert(TaskType::Direction, vec![Array2::ones((2, 2))]);
        task_grads.insert(TaskType::Volatility, vec![Array2::ones((2, 2)) * 0.5]);

        let combined = harmonizer.pcgrad(&task_grads);

        // Should be simple sum when no conflict
        assert_eq!(combined.len(), 1);
        assert!(combined[0][[0, 0]] > 1.0); // Should be > 1 (sum)
    }
}
