//! Meta-Gradient Optimization algorithm.
//!
//! Learns optimal hyperparameters (learning rates, loss function parameters)
//! by differentiating through the inner optimization loop.
//!
//! Reference: Xu, Z., van Hasselt, H., & Silver, D. (2018).
//! "Meta-Gradient Reinforcement Learning." NeurIPS.

use crate::model::network::TradingModel;

/// Task data for meta-gradient optimization
#[derive(Debug, Clone)]
pub struct TaskData {
    /// Training set features (for inner loop adaptation)
    pub train_features: Vec<Vec<f64>>,
    /// Training set labels
    pub train_labels: Vec<f64>,
    /// Validation set features (for meta-gradient computation)
    pub val_features: Vec<Vec<f64>>,
    /// Validation set labels
    pub val_labels: Vec<f64>,
}

impl TaskData {
    /// Create new task data
    pub fn new(
        train_features: Vec<Vec<f64>>,
        train_labels: Vec<f64>,
        val_features: Vec<Vec<f64>>,
        val_labels: Vec<f64>,
    ) -> Self {
        Self {
            train_features,
            train_labels,
            val_features,
            val_labels,
        }
    }
}

/// Learnable loss function parameters
#[derive(Debug, Clone)]
pub struct LearnableLossParams {
    /// Weight for directional accuracy penalty
    pub direction_weight: f64,
    /// Weight for magnitude error
    pub magnitude_weight: f64,
    /// Asymmetry factor (penalize negative errors more)
    pub asymmetry: f64,
}

impl Default for LearnableLossParams {
    fn default() -> Self {
        Self {
            direction_weight: 1.0,
            magnitude_weight: 1.0,
            asymmetry: 0.0,
        }
    }
}

impl LearnableLossParams {
    /// Compute learned loss
    pub fn compute_loss(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        assert_eq!(predictions.len(), targets.len());
        let n = predictions.len() as f64;

        let mut total_loss = 0.0;
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let error = pred - target;

            // Direction penalty
            let direction_error = if pred.signum() != target.signum() {
                softplus(self.direction_weight)
            } else {
                0.0
            };

            // Magnitude loss
            let magnitude_loss = error.powi(2) * softplus(self.magnitude_weight);

            // Asymmetric weighting
            let sigmoid_asym = sigmoid(self.asymmetry);
            let asymmetric_weight = if error < 0.0 {
                1.0 + sigmoid_asym
            } else {
                1.0 - sigmoid_asym * 0.5
            };

            total_loss += magnitude_loss * asymmetric_weight + direction_error;
        }

        total_loss / n
    }
}

/// Softplus activation
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid activation
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Meta-Gradient Optimizer
///
/// Optimizes hyperparameters (learning rates, loss function parameters)
/// by computing gradients of validation loss with respect to these
/// hyperparameters through the inner optimization loop.
#[derive(Debug)]
pub struct MetaGradientOptimizer {
    /// The model being trained
    model: TradingModel,
    /// Per-parameter learning rates (in log space)
    log_learning_rates: Vec<f64>,
    /// Meta-learning rate for hyperparameter updates
    meta_lr: f64,
    /// Number of inner loop steps
    inner_steps: usize,
    /// Whether to learn per-parameter learning rates
    per_param_lr: bool,
    /// Whether to learn the loss function
    learn_loss: bool,
    /// Learnable loss parameters
    loss_params: LearnableLossParams,
    /// Epsilon for numerical gradients
    gradient_epsilon: f64,
    /// Adam optimizer state for meta-parameters
    meta_m: Vec<f64>,
    meta_v: Vec<f64>,
    meta_t: usize,
}

impl MetaGradientOptimizer {
    /// Create a new Meta-Gradient Optimizer
    ///
    /// # Arguments
    /// * `model` - The trading model to optimize
    /// * `inner_lr_init` - Initial inner learning rate
    /// * `meta_lr` - Meta-learning rate for hyperparameter updates
    /// * `inner_steps` - Number of inner optimization steps
    /// * `per_param_lr` - Use per-parameter learning rates
    /// * `learn_loss` - Learn the loss function parameters
    pub fn new(
        model: TradingModel,
        inner_lr_init: f64,
        meta_lr: f64,
        inner_steps: usize,
        per_param_lr: bool,
        learn_loss: bool,
    ) -> Self {
        let num_params = model.num_parameters();
        let num_lr = if per_param_lr { num_params } else { 1 };
        let log_lr = inner_lr_init.ln();

        let num_meta_params = num_lr + if learn_loss { 3 } else { 0 };

        Self {
            model,
            log_learning_rates: vec![log_lr; num_lr],
            meta_lr,
            inner_steps,
            per_param_lr,
            learn_loss,
            loss_params: LearnableLossParams::default(),
            gradient_epsilon: 1e-4,
            meta_m: vec![0.0; num_meta_params],
            meta_v: vec![0.0; num_meta_params],
            meta_t: 0,
        }
    }

    /// Get current learning rates (exponentiated from log space)
    pub fn get_learning_rates(&self) -> Vec<f64> {
        self.log_learning_rates.iter().map(|lr| lr.exp()).collect()
    }

    /// Get reference to the loss parameters
    pub fn loss_params(&self) -> &LearnableLossParams {
        &self.loss_params
    }

    /// Compute loss (MSE or learned)
    fn compute_loss(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        if self.learn_loss {
            self.loss_params.compute_loss(predictions, targets)
        } else {
            let n = predictions.len() as f64;
            predictions
                .iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / n
        }
    }

    /// Perform inner loop: adapt model on training data
    fn inner_loop(&self, task: &TaskData) -> (TradingModel, f64) {
        let mut adapted_model = self.model.clone_model();
        let learning_rates = self.get_learning_rates();

        for _ in 0..self.inner_steps {
            let gradients = adapted_model.compute_gradients(
                &task.train_features,
                &task.train_labels,
                self.gradient_epsilon,
            );

            if self.per_param_lr {
                // Per-parameter learning rates
                let mut params = adapted_model.get_parameters();
                for (i, (p, g)) in params.iter_mut().zip(gradients.iter()).enumerate() {
                    let lr = if i < learning_rates.len() {
                        learning_rates[i]
                    } else {
                        learning_rates[0]
                    };
                    *p -= lr * g;
                }
                adapted_model.set_parameters(&params);
            } else {
                adapted_model.sgd_step(&gradients, learning_rates[0]);
            }
        }

        // Evaluate on validation set
        let val_predictions = adapted_model.predict_batch(&task.val_features);
        let val_loss = self.compute_loss(&val_predictions, &task.val_labels);

        (adapted_model, val_loss)
    }

    /// Collect all meta-parameters into a single vector
    fn get_meta_parameters(&self) -> Vec<f64> {
        let mut params = self.log_learning_rates.clone();
        if self.learn_loss {
            params.push(self.loss_params.direction_weight);
            params.push(self.loss_params.magnitude_weight);
            params.push(self.loss_params.asymmetry);
        }
        params
    }

    /// Set meta-parameters from a flat vector
    fn set_meta_parameters(&mut self, params: &[f64]) {
        let num_lr = self.log_learning_rates.len();
        for i in 0..num_lr {
            self.log_learning_rates[i] = params[i];
        }
        if self.learn_loss {
            self.loss_params.direction_weight = params[num_lr];
            self.loss_params.magnitude_weight = params[num_lr + 1];
            self.loss_params.asymmetry = params[num_lr + 2];
        }
    }

    /// Compute meta-gradients numerically
    fn compute_meta_gradients(&self, tasks: &[TaskData]) -> Vec<f64> {
        let meta_params = self.get_meta_parameters();
        let num_meta = meta_params.len();
        let mut meta_gradients = vec![0.0; num_meta];

        for i in 0..num_meta {
            // Evaluate with meta_param[i] + epsilon
            let mut params_plus = meta_params.clone();
            params_plus[i] += self.gradient_epsilon;
            let mut opt_plus = self.clone_with_meta_params(&params_plus);
            let mut loss_plus = 0.0;
            for task in tasks {
                let (_, vl) = opt_plus.inner_loop(task);
                loss_plus += vl;
            }

            // Evaluate with meta_param[i] - epsilon
            let mut params_minus = meta_params.clone();
            params_minus[i] -= self.gradient_epsilon;
            let mut opt_minus = self.clone_with_meta_params(&params_minus);
            let mut loss_minus = 0.0;
            for task in tasks {
                let (_, vl) = opt_minus.inner_loop(task);
                loss_minus += vl;
            }

            meta_gradients[i] =
                (loss_plus - loss_minus) / (2.0 * self.gradient_epsilon * tasks.len() as f64);
        }

        meta_gradients
    }

    /// Clone optimizer with different meta-parameters
    fn clone_with_meta_params(&self, params: &[f64]) -> Self {
        let mut cloned = Self {
            model: self.model.clone_model(),
            log_learning_rates: self.log_learning_rates.clone(),
            meta_lr: self.meta_lr,
            inner_steps: self.inner_steps,
            per_param_lr: self.per_param_lr,
            learn_loss: self.learn_loss,
            loss_params: self.loss_params.clone(),
            gradient_epsilon: self.gradient_epsilon,
            meta_m: self.meta_m.clone(),
            meta_v: self.meta_v.clone(),
            meta_t: self.meta_t,
        };
        cloned.set_meta_parameters(params);
        cloned
    }

    /// Adam update for meta-parameters
    fn adam_update(&mut self, gradients: &[f64]) {
        self.meta_t += 1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        let mut meta_params = self.get_meta_parameters();

        for i in 0..gradients.len() {
            self.meta_m[i] = beta1 * self.meta_m[i] + (1.0 - beta1) * gradients[i];
            self.meta_v[i] = beta2 * self.meta_v[i] + (1.0 - beta2) * gradients[i].powi(2);

            let m_hat = self.meta_m[i] / (1.0 - beta1.powi(self.meta_t as i32));
            let v_hat = self.meta_v[i] / (1.0 - beta2.powi(self.meta_t as i32));

            meta_params[i] -= self.meta_lr * m_hat / (v_hat.sqrt() + eps);
        }

        self.set_meta_parameters(&meta_params);
    }

    /// Perform one meta-training step
    ///
    /// # Arguments
    /// * `tasks` - Batch of tasks for meta-training
    ///
    /// # Returns
    /// Average validation loss across tasks
    pub fn meta_train_step(&mut self, tasks: &[TaskData]) -> f64 {
        if tasks.is_empty() {
            return 0.0;
        }

        // Compute meta-gradients
        let meta_gradients = self.compute_meta_gradients(tasks);

        // Apply Adam update to meta-parameters
        self.adam_update(&meta_gradients);

        // Also update the base model using FOMAML-style update
        let mut total_val_loss = 0.0;
        for task in tasks {
            let (adapted_model, val_loss) = self.inner_loop(task);
            total_val_loss += val_loss;

            // Use adapted model's gradients on val set for base model update
            let val_grads = adapted_model.compute_gradients(
                &task.val_features,
                &task.val_labels,
                self.gradient_epsilon,
            );
            self.model.sgd_step(&val_grads, self.get_learning_rates()[0] * 0.1);
        }

        total_val_loss / tasks.len() as f64
    }

    /// Adapt model to new data and return adapted model
    pub fn adapt(
        &self,
        train_features: &[Vec<f64>],
        train_labels: &[f64],
        adaptation_steps: Option<usize>,
    ) -> TradingModel {
        let steps = adaptation_steps.unwrap_or(self.inner_steps);
        let mut adapted_model = self.model.clone_model();
        let learning_rates = self.get_learning_rates();
        let lr = learning_rates[0];

        for _ in 0..steps {
            let gradients = adapted_model.compute_gradients(
                train_features,
                train_labels,
                self.gradient_epsilon,
            );
            adapted_model.sgd_step(&gradients, lr);
        }

        adapted_model
    }

    /// Get reference to the current model
    pub fn model(&self) -> &TradingModel {
        &self.model
    }

    /// Get mutable reference to the current model
    pub fn model_mut(&mut self) -> &mut TradingModel {
        &mut self.model
    }

    /// Get the meta-learning rate
    pub fn meta_lr(&self) -> f64 {
        self.meta_lr
    }

    /// Get the number of inner steps
    pub fn inner_steps(&self) -> usize {
        self.inner_steps
    }

    /// Check if using per-parameter learning rates
    pub fn is_per_param_lr(&self) -> bool {
        self.per_param_lr
    }

    /// Check if learning the loss function
    pub fn is_learning_loss(&self) -> bool {
        self.learn_loss
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub epoch: usize,
    pub avg_val_loss: f64,
    pub learning_rates: Vec<f64>,
}

/// Meta-gradient training loop
pub fn train_meta_gradient(
    optimizer: &mut MetaGradientOptimizer,
    task_generator: impl Iterator<Item = Vec<TaskData>>,
    num_epochs: usize,
    log_interval: usize,
) -> Vec<TrainingStats> {
    let mut stats_history = Vec::new();
    let mut task_iter = task_generator;

    for epoch in 0..num_epochs {
        if let Some(tasks) = task_iter.next() {
            let avg_loss = optimizer.meta_train_step(&tasks);

            let stats = TrainingStats {
                epoch,
                avg_val_loss: avg_loss,
                learning_rates: optimizer.get_learning_rates(),
            };

            if epoch % log_interval == 0 {
                tracing::info!(
                    "Epoch {}: avg_val_loss={:.6}, lr={:.6}",
                    epoch,
                    avg_loss,
                    stats.learning_rates[0]
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
            vec![0.05; 10],
            vec![vec![0.2, 0.3, 0.4, 0.5]; 5],
            vec![0.06; 5],
        )
    }

    #[test]
    fn test_optimizer_creation() {
        let model = TradingModel::new(4, 16, 1);
        let opt = MetaGradientOptimizer::new(model, 0.01, 0.001, 5, false, false);

        assert_eq!(opt.inner_steps(), 5);
        assert!(!opt.is_per_param_lr());
        assert!(!opt.is_learning_loss());

        let lrs = opt.get_learning_rates();
        assert_eq!(lrs.len(), 1);
        assert!((lrs[0] - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_per_param_lr() {
        let model = TradingModel::new(4, 8, 1);
        let num_params = model.num_parameters();
        let opt = MetaGradientOptimizer::new(model, 0.01, 0.001, 3, true, false);

        let lrs = opt.get_learning_rates();
        assert_eq!(lrs.len(), num_params);
    }

    #[test]
    fn test_inner_loop() {
        let model = TradingModel::new(4, 8, 1);
        let opt = MetaGradientOptimizer::new(model, 0.01, 0.001, 3, false, false);
        let task = create_dummy_task();

        let (_adapted, val_loss) = opt.inner_loop(&task);
        assert!(val_loss.is_finite());
        assert!(val_loss >= 0.0);
    }

    #[test]
    fn test_meta_train_step() {
        let model = TradingModel::new(4, 8, 1);
        let mut opt = MetaGradientOptimizer::new(model, 0.01, 0.001, 3, false, false);
        let tasks = vec![create_dummy_task(), create_dummy_task()];

        let loss = opt.meta_train_step(&tasks);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_learning_rate_updates() {
        let model = TradingModel::new(4, 8, 1);
        let mut opt = MetaGradientOptimizer::new(model, 0.01, 0.001, 2, false, false);

        let lr_before = opt.get_learning_rates()[0];
        let tasks = vec![create_dummy_task()];
        opt.meta_train_step(&tasks);
        let lr_after = opt.get_learning_rates()[0];

        // Learning rate should have changed
        assert!(
            (lr_before - lr_after).abs() > 1e-12,
            "Learning rate should change after meta-training"
        );
    }

    #[test]
    fn test_learnable_loss() {
        let model = TradingModel::new(4, 8, 1);
        let opt = MetaGradientOptimizer::new(model, 0.01, 0.001, 3, false, true);

        assert!(opt.is_learning_loss());

        let predictions = vec![0.1, -0.05, 0.02];
        let targets = vec![0.05, 0.03, -0.01];
        let loss = opt.loss_params().compute_loss(&predictions, &targets);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_adapt() {
        let model = TradingModel::new(4, 8, 1);
        let opt = MetaGradientOptimizer::new(model, 0.01, 0.001, 3, false, false);

        let features = vec![vec![0.1, 0.2, 0.3, 0.4]; 10];
        let labels = vec![0.05; 10];

        let adapted = opt.adapt(&features, &labels, Some(5));
        let prediction = adapted.predict(&[0.1, 0.2, 0.3, 0.4]);
        assert!(prediction.is_finite());
    }
}
