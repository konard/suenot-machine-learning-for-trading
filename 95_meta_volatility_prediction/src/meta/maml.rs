//! MAML (Model-Agnostic Meta-Learning) implementation for volatility prediction.
//!
//! Implements the inner loop (task-specific adaptation) and outer loop
//! (meta-update) of MAML, using finite-difference gradient approximation.

use crate::model::network::MetaVolatilityNet;
use tracing::info;

/// A single volatility prediction task for meta-learning.
#[derive(Debug, Clone)]
pub struct VolatilityTask {
    pub support_x: Vec<Vec<f64>>,
    pub support_y: Vec<f64>,
    pub query_x: Vec<Vec<f64>>,
    pub query_y: Vec<f64>,
}

/// MAML trainer for meta-volatility prediction.
pub struct MAMLTrainer {
    pub net: MetaVolatilityNet,
    pub inner_lr: f64,
    pub outer_lr: f64,
    pub inner_steps: usize,
    epsilon: f64,
}

impl MAMLTrainer {
    /// Create a new MAML trainer.
    pub fn new(
        net: MetaVolatilityNet,
        inner_lr: f64,
        outer_lr: f64,
        inner_steps: usize,
    ) -> Self {
        Self {
            net,
            inner_lr,
            outer_lr,
            inner_steps,
            epsilon: 1e-5,
        }
    }

    /// Compute volatility loss: MSE + lambda * MAE.
    fn volatility_loss(predictions: &[f64], targets: &[f64]) -> f64 {
        let n = predictions.len() as f64;
        let lambda = 0.1;

        let mse: f64 = predictions
            .iter()
            .zip(targets)
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n;

        let mae: f64 = predictions
            .iter()
            .zip(targets)
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / n;

        mse + lambda * mae
    }

    /// Evaluate model with given parameters on a dataset.
    fn evaluate_with_params(
        &self,
        params: &[f64],
        inputs: &[Vec<f64>],
        targets: &[f64],
    ) -> f64 {
        let predictions: Vec<f64> = inputs
            .iter()
            .map(|x| self.net.forward_with_params(x, params))
            .collect();
        Self::volatility_loss(&predictions, targets)
    }

    /// Compute gradient via finite differences.
    fn compute_gradient(
        &self,
        params: &[f64],
        inputs: &[Vec<f64>],
        targets: &[f64],
    ) -> Vec<f64> {
        let n = params.len();
        let mut grad = vec![0.0; n];
        let base_loss = self.evaluate_with_params(params, inputs, targets);

        for i in 0..n {
            let mut perturbed = params.to_vec();
            perturbed[i] += self.epsilon;
            let perturbed_loss = self.evaluate_with_params(&perturbed, inputs, targets);
            grad[i] = (perturbed_loss - base_loss) / self.epsilon;
        }

        grad
    }

    /// Inner loop: adapt parameters to a specific task.
    fn inner_update(&self, task: &VolatilityTask) -> Vec<f64> {
        let mut params = self.net.flatten_params();

        for _step in 0..self.inner_steps {
            let grad = self.compute_gradient(
                &params,
                &task.support_x,
                &task.support_y,
            );
            for (p, g) in params.iter_mut().zip(grad.iter()) {
                *p -= self.inner_lr * g;
            }
        }

        params
    }

    /// One step of MAML meta-training across a batch of tasks.
    ///
    /// Returns the average meta-loss.
    pub fn meta_train_step(&mut self, tasks: &[&VolatilityTask]) -> f64 {
        let num_params = self.net.num_params();
        let mut meta_grad = vec![0.0; num_params];
        let mut total_loss = 0.0;

        let base_params = self.net.flatten_params();

        for task in tasks {
            // Inner loop: adapt to task
            let adapted_params = self.inner_update(task);

            // Evaluate adapted params on query set
            let query_loss = self.evaluate_with_params(
                &adapted_params,
                &task.query_x,
                &task.query_y,
            );
            total_loss += query_loss;

            // Compute meta-gradient via finite differences on the outer loss
            for i in 0..num_params {
                let mut perturbed_base = base_params.clone();
                perturbed_base[i] += self.epsilon;

                // Re-run inner loop with perturbed base params
                let mut perturbed_adapted = perturbed_base.clone();
                for _step in 0..self.inner_steps {
                    let grad = self.compute_gradient(
                        &perturbed_adapted,
                        &task.support_x,
                        &task.support_y,
                    );
                    for (p, g) in perturbed_adapted.iter_mut().zip(grad.iter()) {
                        *p -= self.inner_lr * g;
                    }
                }

                let perturbed_loss = self.evaluate_with_params(
                    &perturbed_adapted,
                    &task.query_x,
                    &task.query_y,
                );
                meta_grad[i] += (perturbed_loss - query_loss) / self.epsilon;
            }
        }

        let n_tasks = tasks.len() as f64;
        total_loss /= n_tasks;

        // Apply meta-update
        let mut params = base_params;
        for (p, g) in params.iter_mut().zip(meta_grad.iter()) {
            *p -= self.outer_lr * g / n_tasks;
        }
        self.net.unflatten_params(&params);

        total_loss
    }

    /// Adapt to a new task and predict on query inputs.
    pub fn adapt_and_predict(
        &self,
        task: &VolatilityTask,
    ) -> Vec<f64> {
        let adapted_params = self.inner_update(task);
        task.query_x
            .iter()
            .map(|x| self.net.forward_with_params(x, &adapted_params))
            .collect()
    }

    /// Get reference to the underlying network.
    pub fn network(&self) -> &MetaVolatilityNet {
        &self.net
    }

    /// Log training status.
    pub fn log_status(&self, epoch: usize, loss: f64) {
        info!(epoch = epoch, loss = loss, "Meta-training step");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_task() -> VolatilityTask {
        VolatilityTask {
            support_x: vec![vec![0.01; 10]; 5],
            support_y: vec![0.02; 5],
            query_x: vec![vec![0.01; 10]; 3],
            query_y: vec![0.02; 3],
        }
    }

    #[test]
    fn test_inner_update_reduces_loss() {
        let net = MetaVolatilityNet::new(10, 16);
        let trainer = MAMLTrainer::new(net, 0.01, 0.001, 3);
        let task = make_dummy_task();

        let initial_params = trainer.net.flatten_params();
        let initial_loss = trainer.evaluate_with_params(
            &initial_params,
            &task.support_x,
            &task.support_y,
        );

        let adapted_params = trainer.inner_update(&task);
        let adapted_loss = trainer.evaluate_with_params(
            &adapted_params,
            &task.support_x,
            &task.support_y,
        );

        assert!(
            adapted_loss <= initial_loss + 1e-6,
            "Inner loop should reduce or maintain loss: {} -> {}",
            initial_loss,
            adapted_loss
        );
    }

    #[test]
    fn test_meta_train_step() {
        let net = MetaVolatilityNet::new(10, 16);
        let mut trainer = MAMLTrainer::new(net, 0.01, 0.001, 2);
        let task = make_dummy_task();
        let tasks: Vec<&VolatilityTask> = vec![&task];

        let loss = trainer.meta_train_step(&tasks);
        assert!(loss.is_finite(), "Meta-loss should be finite");
    }

    #[test]
    fn test_adapt_and_predict() {
        let net = MetaVolatilityNet::new(10, 16);
        let trainer = MAMLTrainer::new(net, 0.01, 0.001, 3);
        let task = make_dummy_task();

        let preds = trainer.adapt_and_predict(&task);
        assert_eq!(preds.len(), task.query_x.len());
        for p in &preds {
            assert!(*p > 0.0, "Volatility predictions must be positive");
        }
    }
}
