//! MAML meta-learning algorithm.
//!
//! Model-Agnostic Meta-Learning (MAML) is an algorithm that learns a good
//! initialization for neural network parameters that enables fast adaptation
//! to new tasks with only a few gradient steps.
//!
//! Reference: Finn, C., Abbeel, P., & Levine, S. (2017).
//! "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML.

use crate::model::network::TradingModel;

/// Task data for meta-learning
#[derive(Debug, Clone)]
pub struct TaskData {
    /// Support set features for adaptation
    pub support_features: Vec<Vec<f64>>,
    /// Support set labels
    pub support_labels: Vec<f64>,
    /// Query set features for evaluation
    pub query_features: Vec<Vec<f64>>,
    /// Query set labels
    pub query_labels: Vec<f64>,
}

impl TaskData {
    /// Create new task data
    pub fn new(
        support_features: Vec<Vec<f64>>,
        support_labels: Vec<f64>,
        query_features: Vec<Vec<f64>>,
        query_labels: Vec<f64>,
    ) -> Self {
        Self {
            support_features,
            support_labels,
            query_features,
            query_labels,
        }
    }
}

/// MAML meta-learning trainer
///
/// Implements both full MAML (second-order) and FOMAML (first-order approximation).
#[derive(Debug)]
pub struct MAMLTrainer {
    /// The model being trained
    model: TradingModel,
    /// Inner loop learning rate (alpha - for task adaptation)
    inner_lr: f64,
    /// Outer loop learning rate (beta - meta-learning rate)
    outer_lr: f64,
    /// Number of SGD steps per task in inner loop
    inner_steps: usize,
    /// Whether to use first-order approximation (FOMAML)
    first_order: bool,
    /// Epsilon for numerical gradients
    gradient_epsilon: f64,
}

impl MAMLTrainer {
    /// Create a new MAML trainer
    ///
    /// # Arguments
    /// * `model` - The trading model to train
    /// * `inner_lr` - Learning rate for task-specific adaptation (alpha)
    /// * `outer_lr` - Meta-learning rate (beta)
    /// * `inner_steps` - Number of gradient steps per task in inner loop
    /// * `first_order` - If true, use FOMAML (first-order approximation)
    pub fn new(
        model: TradingModel,
        inner_lr: f64,
        outer_lr: f64,
        inner_steps: usize,
        first_order: bool,
    ) -> Self {
        Self {
            model,
            inner_lr,
            outer_lr,
            inner_steps,
            first_order,
            gradient_epsilon: 1e-4,
        }
    }

    /// Set the gradient epsilon for numerical differentiation
    pub fn set_gradient_epsilon(&mut self, epsilon: f64) {
        self.gradient_epsilon = epsilon;
    }

    /// Perform inner loop adaptation on a single task
    ///
    /// This is the task-specific learning step that adapts the model
    /// to a particular task using a few gradient steps on the support set.
    ///
    /// Returns the adapted model and the query loss
    fn inner_loop(&self, task: &TaskData) -> (TradingModel, f64) {
        let mut adapted_model = self.model.clone_model();

        // Perform k steps of SGD on the support set
        for _ in 0..self.inner_steps {
            let gradients = adapted_model.compute_gradients(
                &task.support_features,
                &task.support_labels,
                self.gradient_epsilon,
            );
            adapted_model.sgd_step(&gradients, self.inner_lr);
        }

        // Evaluate on query set
        let query_predictions = adapted_model.predict_batch(&task.query_features);
        let query_loss = adapted_model.compute_loss(&query_predictions, &task.query_labels);

        (adapted_model, query_loss)
    }

    /// Compute meta-gradients for a batch of tasks
    ///
    /// For full MAML, this computes the gradient of the query loss
    /// with respect to the original parameters, accounting for the
    /// inner loop adaptation.
    ///
    /// For FOMAML, this simply uses the gradients at the adapted parameters.
    fn compute_meta_gradients(&self, tasks: &[TaskData]) -> Vec<f64> {
        let original_params = self.model.get_parameters();
        let num_params = original_params.len();
        let mut meta_gradients = vec![0.0; num_params];

        for task in tasks {
            if self.first_order {
                // FOMAML: Use gradients at adapted parameters
                let (adapted_model, _) = self.inner_loop(task);
                let gradients = adapted_model.compute_gradients(
                    &task.query_features,
                    &task.query_labels,
                    self.gradient_epsilon,
                );
                for (mg, g) in meta_gradients.iter_mut().zip(gradients.iter()) {
                    *mg += g;
                }
            } else {
                // Full MAML: Approximate second-order gradients numerically
                // This is a numerical approximation to the true MAML gradient
                for i in 0..num_params {
                    // Compute loss after inner loop with param[i] + epsilon
                    let mut params_plus = original_params.clone();
                    params_plus[i] += self.gradient_epsilon;
                    let mut model_plus = self.model.clone_model();
                    model_plus.set_parameters(&params_plus);
                    let trainer_plus = MAMLTrainer::new(
                        model_plus,
                        self.inner_lr,
                        self.outer_lr,
                        self.inner_steps,
                        true, // Use FOMAML for inner computation
                    );
                    let (_, loss_plus) = trainer_plus.inner_loop(task);

                    // Compute loss after inner loop with param[i] - epsilon
                    let mut params_minus = original_params.clone();
                    params_minus[i] -= self.gradient_epsilon;
                    let mut model_minus = self.model.clone_model();
                    model_minus.set_parameters(&params_minus);
                    let trainer_minus = MAMLTrainer::new(
                        model_minus,
                        self.inner_lr,
                        self.outer_lr,
                        self.inner_steps,
                        true,
                    );
                    let (_, loss_minus) = trainer_minus.inner_loop(task);

                    // Numerical gradient
                    meta_gradients[i] += (loss_plus - loss_minus) / (2.0 * self.gradient_epsilon);
                }
            }
        }

        // Average over tasks
        let num_tasks = tasks.len() as f64;
        for g in meta_gradients.iter_mut() {
            *g /= num_tasks;
        }

        meta_gradients
    }

    /// Perform one meta-training step using MAML
    ///
    /// # Arguments
    /// * `tasks` - Batch of tasks for meta-training
    ///
    /// # Returns
    /// Average query loss across all tasks
    pub fn meta_train_step(&mut self, tasks: &[TaskData]) -> f64 {
        if tasks.is_empty() {
            return 0.0;
        }

        // Compute meta-gradients
        let meta_gradients = self.compute_meta_gradients(tasks);

        // Apply meta-update: θ ← θ - β * ∇_θ L
        self.model.sgd_step(&meta_gradients, self.outer_lr);

        // Compute average query loss for reporting
        let mut total_query_loss = 0.0;
        for task in tasks {
            let (_, query_loss) = self.inner_loop(task);
            total_query_loss += query_loss;
        }

        total_query_loss / tasks.len() as f64
    }

    /// Adapt the meta-learned model to a new task
    ///
    /// # Arguments
    /// * `support_features` - Features from the new task
    /// * `support_labels` - Labels from the new task
    /// * `adaptation_steps` - Number of gradient steps (default: inner_steps)
    ///
    /// # Returns
    /// Adapted model ready for prediction
    pub fn adapt(
        &self,
        support_features: &[Vec<f64>],
        support_labels: &[f64],
        adaptation_steps: Option<usize>,
    ) -> TradingModel {
        let steps = adaptation_steps.unwrap_or(self.inner_steps);
        let mut adapted_model = self.model.clone_model();

        for _ in 0..steps {
            let gradients = adapted_model.compute_gradients(
                support_features,
                support_labels,
                self.gradient_epsilon,
            );
            adapted_model.sgd_step(&gradients, self.inner_lr);
        }

        adapted_model
    }

    /// Get a reference to the current model
    pub fn model(&self) -> &TradingModel {
        &self.model
    }

    /// Get a mutable reference to the current model
    pub fn model_mut(&mut self) -> &mut TradingModel {
        &mut self.model
    }

    /// Get the base (meta-learned) model (alias for model())
    pub fn base_model(&self) -> &TradingModel {
        &self.model
    }

    /// Get the inner learning rate
    pub fn inner_lr(&self) -> f64 {
        self.inner_lr
    }

    /// Get the outer learning rate
    pub fn outer_lr(&self) -> f64 {
        self.outer_lr
    }

    /// Get the number of inner steps
    pub fn inner_steps(&self) -> usize {
        self.inner_steps
    }

    /// Check if using first-order approximation
    pub fn is_first_order(&self) -> bool {
        self.first_order
    }

    /// Set the inner learning rate
    pub fn set_inner_lr(&mut self, lr: f64) {
        self.inner_lr = lr;
    }

    /// Set the outer learning rate
    pub fn set_outer_lr(&mut self, lr: f64) {
        self.outer_lr = lr;
    }

    /// Set the number of inner steps
    pub fn set_inner_steps(&mut self, steps: usize) {
        self.inner_steps = steps;
    }

    /// Set first-order mode
    pub fn set_first_order(&mut self, first_order: bool) {
        self.first_order = first_order;
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Epoch number
    pub epoch: usize,
    /// Average query loss
    pub avg_loss: f64,
    /// Minimum loss in this epoch
    pub min_loss: f64,
    /// Maximum loss in this epoch
    pub max_loss: f64,
}

/// Meta-training loop with logging
pub fn train_maml(
    trainer: &mut MAMLTrainer,
    task_generator: impl Iterator<Item = Vec<TaskData>>,
    num_epochs: usize,
    log_interval: usize,
) -> Vec<TrainingStats> {
    let mut stats_history = Vec::new();
    let mut task_iter = task_generator;

    for epoch in 0..num_epochs {
        if let Some(tasks) = task_iter.next() {
            let losses: Vec<f64> = tasks.iter().map(|t| {
                let (_, loss) = trainer.inner_loop(t);
                loss
            }).collect();

            let avg_loss = trainer.meta_train_step(&tasks);
            let min_loss = losses.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let stats = TrainingStats {
                epoch,
                avg_loss,
                min_loss,
                max_loss,
            };

            if epoch % log_interval == 0 {
                tracing::info!(
                    "Epoch {}: avg_loss={:.6}, min_loss={:.6}, max_loss={:.6}",
                    epoch, avg_loss, min_loss, max_loss
                );
            }

            stats_history.push(stats);
        } else {
            break;
        }
    }

    stats_history
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_task() -> TaskData {
        TaskData::new(
            vec![vec![0.1, 0.2, 0.3, 0.4]; 10],
            vec![0.5; 10],
            vec![vec![0.2, 0.3, 0.4, 0.5]; 5],
            vec![0.6; 5],
        )
    }

    #[test]
    fn test_maml_trainer_creation() {
        let model = TradingModel::new(4, 16, 1);
        let trainer = MAMLTrainer::new(model, 0.01, 0.001, 5, false);

        assert_eq!(trainer.inner_lr(), 0.01);
        assert_eq!(trainer.outer_lr(), 0.001);
        assert_eq!(trainer.inner_steps(), 5);
        assert!(!trainer.is_first_order());
    }

    #[test]
    fn test_fomaml_trainer_creation() {
        let model = TradingModel::new(4, 16, 1);
        let trainer = MAMLTrainer::new(model, 0.01, 0.001, 5, true);

        assert!(trainer.is_first_order());
    }

    #[test]
    fn test_inner_loop() {
        let model = TradingModel::new(4, 8, 1);
        let trainer = MAMLTrainer::new(model, 0.01, 0.001, 3, true);
        let task = create_dummy_task();

        let (_adapted_model, loss) = trainer.inner_loop(&task);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_meta_train_step_fomaml() {
        let model = TradingModel::new(4, 8, 1);
        let mut trainer = MAMLTrainer::new(model, 0.01, 0.001, 3, true);
        let tasks = vec![create_dummy_task(), create_dummy_task()];

        let loss = trainer.meta_train_step(&tasks);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_adapt() {
        let model = TradingModel::new(4, 8, 1);
        let trainer = MAMLTrainer::new(model, 0.01, 0.001, 3, true);

        let features = vec![vec![0.1, 0.2, 0.3, 0.4]; 10];
        let labels = vec![0.5; 10];

        let adapted = trainer.adapt(&features, &labels, Some(5));
        let prediction = adapted.predict(&[0.1, 0.2, 0.3, 0.4]);
        assert!(prediction.is_finite());
    }

    #[test]
    fn test_parameter_update() {
        let model = TradingModel::new(4, 8, 1);
        let params_before = model.get_parameters();
        let mut trainer = MAMLTrainer::new(model, 0.01, 0.001, 3, true);
        let tasks = vec![create_dummy_task()];

        trainer.meta_train_step(&tasks);

        let params_after = trainer.model().get_parameters();

        // Parameters should have changed
        let mut changed = false;
        for (before, after) in params_before.iter().zip(params_after.iter()) {
            if (before - after).abs() > 1e-10 {
                changed = true;
                break;
            }
        }
        assert!(changed, "Parameters should change after meta-training step");
    }
}
