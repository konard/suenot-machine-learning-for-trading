//! Continual Meta-Learning algorithm combining MAML with EWC and replay.
//!
//! This module provides a trainer that can learn new market regimes
//! sequentially without catastrophic forgetting of previously learned regimes.
//!
//! Key components:
//! - MAML inner/outer loop for fast adaptation
//! - Elastic Weight Consolidation (EWC) to protect important parameters
//! - Experience replay buffer to revisit past regime tasks
//!
//! References:
//! - Finn et al., 2017 (MAML)
//! - Kirkpatrick et al., 2017 (EWC)
//! - Javed & White, 2019 (Continual Meta-Learning)

use crate::model::network::TradingModel;
use rand::seq::SliceRandom;

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
    /// Market regime identifier
    pub regime_id: usize,
}

impl TaskData {
    /// Create new task data
    pub fn new(
        support_features: Vec<Vec<f64>>,
        support_labels: Vec<f64>,
        query_features: Vec<Vec<f64>>,
        query_labels: Vec<f64>,
        regime_id: usize,
    ) -> Self {
        Self {
            support_features,
            support_labels,
            query_features,
            query_labels,
            regime_id,
        }
    }
}

/// Experience replay buffer for continual learning
#[derive(Debug)]
pub struct ReplayBuffer {
    buffer: Vec<TaskData>,
    max_size: usize,
    tasks_per_regime: usize,
    regime_counts: std::collections::HashMap<usize, usize>,
}

impl ReplayBuffer {
    /// Create a new replay buffer
    pub fn new(max_size: usize, tasks_per_regime: usize) -> Self {
        Self {
            buffer: Vec::new(),
            max_size,
            tasks_per_regime,
            regime_counts: std::collections::HashMap::new(),
        }
    }

    /// Add a task to the buffer
    pub fn add(&mut self, task: TaskData) {
        let regime = task.regime_id;
        let count = self.regime_counts.get(&regime).copied().unwrap_or(0);

        if count < self.tasks_per_regime {
            if self.buffer.len() < self.max_size {
                self.buffer.push(task);
                *self.regime_counts.entry(regime).or_insert(0) += 1;
            }
        }
    }

    /// Add multiple tasks
    pub fn add_batch(&mut self, tasks: &[TaskData]) {
        for task in tasks {
            self.add(task.clone());
        }
    }

    /// Sample n random tasks from the buffer
    pub fn sample(&self, n: usize) -> Vec<&TaskData> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        let mut rng = rand::thread_rng();
        let n = n.min(self.buffer.len());
        let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(n);

        indices.iter().map(|&i| &self.buffer[i]).collect()
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.buffer.len()
    }

    /// Get number of distinct regimes stored
    pub fn num_regimes(&self) -> usize {
        self.regime_counts.len()
    }
}

/// Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting
///
/// Computes the Fisher Information Matrix to identify important parameters
/// and penalizes changes to them during training on new regimes.
#[derive(Debug)]
pub struct EWC {
    /// Fisher Information diagonal (importance weights per parameter)
    fisher: Vec<f64>,
    /// Optimal parameters from previous regimes
    optimal_params: Vec<f64>,
    /// EWC regularization strength
    lambda: f64,
    /// Whether Fisher has been computed
    initialized: bool,
}

impl EWC {
    /// Create a new EWC instance
    pub fn new(lambda: f64) -> Self {
        Self {
            fisher: Vec::new(),
            optimal_params: Vec::new(),
            lambda,
            initialized: false,
        }
    }

    /// Compute the Fisher Information Matrix from a set of tasks
    ///
    /// Uses numerical gradient approximation to estimate parameter importance
    pub fn compute_fisher(
        &mut self,
        model: &TradingModel,
        tasks: &[TaskData],
        epsilon: f64,
    ) {
        let params = model.get_parameters();
        let num_params = params.len();
        let mut fisher = vec![0.0; num_params];
        let mut n_samples = 0;

        for task in tasks {
            // Compute gradients on support set
            let gradients = model.compute_gradients(
                &task.support_features,
                &task.support_labels,
                epsilon,
            );

            // Fisher = E[grad^2]
            for (f, g) in fisher.iter_mut().zip(gradients.iter()) {
                *f += g * g;
            }
            n_samples += 1;
        }

        // Average
        if n_samples > 0 {
            let n = n_samples as f64;
            for f in fisher.iter_mut() {
                *f /= n;
            }
        }

        self.fisher = fisher;
        self.optimal_params = params;
        self.initialized = true;
    }

    /// Compute the EWC penalty for a given set of parameters
    ///
    /// penalty = (lambda/2) * sum_i F_i * (theta_i - theta_i*)^2
    pub fn penalty(&self, current_params: &[f64]) -> f64 {
        if !self.initialized {
            return 0.0;
        }

        let mut penalty = 0.0;
        for i in 0..current_params.len().min(self.fisher.len()) {
            let diff = current_params[i] - self.optimal_params[i];
            penalty += self.fisher[i] * diff * diff;
        }

        (self.lambda / 2.0) * penalty
    }

    /// Compute the EWC gradient penalty
    ///
    /// grad_penalty_i = lambda * F_i * (theta_i - theta_i*)
    pub fn penalty_gradients(&self, current_params: &[f64]) -> Vec<f64> {
        if !self.initialized {
            return vec![0.0; current_params.len()];
        }

        current_params
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                if i < self.fisher.len() {
                    self.lambda * self.fisher[i] * (p - self.optimal_params[i])
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Check if Fisher has been computed
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Continual Meta-Learning trainer
///
/// Combines MAML with EWC and experience replay for learning
/// sequential market regimes without catastrophic forgetting.
#[derive(Debug)]
pub struct ContinualMAMLTrainer {
    /// The model being trained
    model: TradingModel,
    /// Inner loop learning rate (alpha)
    inner_lr: f64,
    /// Outer loop learning rate (beta)
    outer_lr: f64,
    /// Number of SGD steps per task in inner loop
    inner_steps: usize,
    /// Whether to use first-order approximation (FOMAML)
    first_order: bool,
    /// Epsilon for numerical gradients
    gradient_epsilon: f64,
    /// EWC regularizer
    ewc: EWC,
    /// Replay buffer
    replay_buffer: ReplayBuffer,
    /// Ratio of replay tasks to current tasks
    replay_ratio: f64,
    /// List of regime IDs seen so far
    regimes_seen: Vec<usize>,
}

impl ContinualMAMLTrainer {
    /// Create a new Continual MAML trainer
    ///
    /// # Arguments
    /// * `model` - The trading model to train
    /// * `inner_lr` - Learning rate for task-specific adaptation
    /// * `outer_lr` - Meta-learning rate
    /// * `inner_steps` - Number of gradient steps per task in inner loop
    /// * `first_order` - If true, use FOMAML
    /// * `ewc_lambda` - EWC regularization strength
    /// * `replay_buffer_size` - Maximum tasks in replay buffer
    pub fn new(
        model: TradingModel,
        inner_lr: f64,
        outer_lr: f64,
        inner_steps: usize,
        first_order: bool,
        ewc_lambda: f64,
        replay_buffer_size: usize,
    ) -> Self {
        Self {
            model,
            inner_lr,
            outer_lr,
            inner_steps,
            first_order,
            gradient_epsilon: 1e-4,
            ewc: EWC::new(ewc_lambda),
            replay_buffer: ReplayBuffer::new(replay_buffer_size, 5),
            replay_ratio: 0.5,
            regimes_seen: Vec::new(),
        }
    }

    /// Set the replay ratio
    pub fn set_replay_ratio(&mut self, ratio: f64) {
        self.replay_ratio = ratio;
    }

    /// Perform inner loop adaptation on a single task
    fn inner_loop(&self, task: &TaskData) -> (TradingModel, f64) {
        let mut adapted_model = self.model.clone_model();

        for _ in 0..self.inner_steps {
            let gradients = adapted_model.compute_gradients(
                &task.support_features,
                &task.support_labels,
                self.gradient_epsilon,
            );
            adapted_model.sgd_step(&gradients, self.inner_lr);
        }

        let query_predictions = adapted_model.predict_batch(&task.query_features);
        let query_loss = adapted_model.compute_loss(&query_predictions, &task.query_labels);

        (adapted_model, query_loss)
    }

    /// Compute meta-gradients with EWC penalty
    fn compute_meta_gradients(&self, tasks: &[TaskData]) -> Vec<f64> {
        let num_params = self.model.get_parameters().len();
        let mut meta_gradients = vec![0.0; num_params];

        for task in tasks {
            // FOMAML: use gradients at adapted parameters on query set
            let (adapted_model, _) = self.inner_loop(task);
            let gradients = adapted_model.compute_gradients(
                &task.query_features,
                &task.query_labels,
                self.gradient_epsilon,
            );
            for (mg, g) in meta_gradients.iter_mut().zip(gradients.iter()) {
                *mg += g;
            }
        }

        // Average over tasks
        let num_tasks = tasks.len() as f64;
        for g in meta_gradients.iter_mut() {
            *g /= num_tasks;
        }

        // Add EWC penalty gradients
        let ewc_grads = self.ewc.penalty_gradients(&self.model.get_parameters());
        for (mg, eg) in meta_gradients.iter_mut().zip(ewc_grads.iter()) {
            *mg += eg;
        }

        meta_gradients
    }

    /// Perform one meta-training step with continual learning
    ///
    /// Combines MAML loss on current + replay tasks with EWC penalty.
    pub fn meta_train_step(&mut self, tasks: &[TaskData]) -> f64 {
        if tasks.is_empty() {
            return 0.0;
        }

        // Combine current tasks with replay tasks
        let mut all_tasks: Vec<TaskData> = tasks.to_vec();
        if self.replay_buffer.size() > 0 {
            let n_replay = (tasks.len() as f64 * self.replay_ratio).ceil() as usize;
            let replay_tasks = self.replay_buffer.sample(n_replay.max(1));
            for rt in replay_tasks {
                all_tasks.push(rt.clone());
            }
        }

        // Compute meta-gradients (includes EWC penalty)
        let meta_gradients = self.compute_meta_gradients(&all_tasks);

        // Apply meta-update
        self.model.sgd_step(&meta_gradients, self.outer_lr);

        // Compute average query loss for reporting
        let mut total_query_loss = 0.0;
        for task in tasks {
            let (_, query_loss) = self.inner_loop(task);
            total_query_loss += query_loss;
        }

        total_query_loss / tasks.len() as f64
    }

    /// Learn a new market regime while preserving past knowledge
    ///
    /// 1. Meta-train on the new regime
    /// 2. Update EWC Fisher information
    /// 3. Store tasks in replay buffer
    pub fn learn_regime(
        &mut self,
        tasks: &[TaskData],
        regime_id: usize,
        num_epochs: usize,
        log_interval: usize,
    ) -> Vec<f64> {
        let mut losses = Vec::new();

        for epoch in 0..num_epochs {
            let loss = self.meta_train_step(tasks);
            losses.push(loss);

            if log_interval > 0 && (epoch + 1) % log_interval == 0 {
                tracing::info!(
                    "Regime {} | Epoch {}: loss={:.6}, replay_buffer={}",
                    regime_id, epoch + 1, loss, self.replay_buffer.size()
                );
            }
        }

        // Update EWC with combined data
        let mut all_tasks = tasks.to_vec();
        if self.replay_buffer.size() > 0 {
            let n = tasks.len().min(self.replay_buffer.size());
            let replay = self.replay_buffer.sample(n);
            for rt in replay {
                all_tasks.push(rt.clone());
            }
        }
        self.ewc.compute_fisher(&self.model, &all_tasks, self.gradient_epsilon);

        // Store in replay buffer
        self.replay_buffer.add_batch(tasks);

        if !self.regimes_seen.contains(&regime_id) {
            self.regimes_seen.push(regime_id);
        }

        losses
    }

    /// Adapt the meta-learned model to a new task
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

    /// Evaluate model on a specific regime's tasks
    ///
    /// Returns (average_loss, directional_accuracy)
    pub fn evaluate_regime(&self, tasks: &[TaskData]) -> (f64, f64) {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total_predictions = 0;

        for task in tasks {
            let (adapted_model, query_loss) = self.inner_loop(task);
            total_loss += query_loss;

            let predictions = adapted_model.predict_batch(&task.query_features);
            for (pred, actual) in predictions.iter().zip(task.query_labels.iter()) {
                if (pred > &0.0 && actual > &0.0) || (pred < &0.0 && actual < &0.0) {
                    total_correct += 1;
                }
                total_predictions += 1;
            }
        }

        let avg_loss = if !tasks.is_empty() {
            total_loss / tasks.len() as f64
        } else {
            0.0
        };

        let accuracy = if total_predictions > 0 {
            total_correct as f64 / total_predictions as f64
        } else {
            0.0
        };

        (avg_loss, accuracy)
    }

    /// Get a reference to the current model
    pub fn model(&self) -> &TradingModel {
        &self.model
    }

    /// Get a mutable reference to the current model
    pub fn model_mut(&mut self) -> &mut TradingModel {
        &mut self.model
    }

    /// Get the base (meta-learned) model
    pub fn base_model(&self) -> &TradingModel {
        &self.model
    }

    /// Get the list of regimes seen
    pub fn regimes_seen(&self) -> &[usize] {
        &self.regimes_seen
    }

    /// Get the replay buffer size
    pub fn replay_buffer_size(&self) -> usize {
        self.replay_buffer.size()
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
    pub epoch: usize,
    pub avg_loss: f64,
    pub min_loss: f64,
    pub max_loss: f64,
    pub regime_id: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_task(regime_id: usize) -> TaskData {
        TaskData::new(
            vec![vec![0.1, 0.2, 0.3, 0.4]; 10],
            vec![0.5; 10],
            vec![vec![0.2, 0.3, 0.4, 0.5]; 5],
            vec![0.6; 5],
            regime_id,
        )
    }

    #[test]
    fn test_trainer_creation() {
        let model = TradingModel::new(4, 16, 1);
        let trainer = ContinualMAMLTrainer::new(model, 0.01, 0.001, 5, true, 100.0, 50);

        assert_eq!(trainer.inner_lr(), 0.01);
        assert_eq!(trainer.outer_lr(), 0.001);
        assert_eq!(trainer.inner_steps(), 5);
        assert!(trainer.is_first_order());
        assert_eq!(trainer.regimes_seen().len(), 0);
    }

    #[test]
    fn test_inner_loop() {
        let model = TradingModel::new(4, 8, 1);
        let trainer = ContinualMAMLTrainer::new(model, 0.01, 0.001, 3, true, 100.0, 50);
        let task = create_dummy_task(0);

        let (_adapted_model, loss) = trainer.inner_loop(&task);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_meta_train_step() {
        let model = TradingModel::new(4, 8, 1);
        let mut trainer = ContinualMAMLTrainer::new(model, 0.01, 0.001, 3, true, 100.0, 50);
        let tasks = vec![create_dummy_task(0), create_dummy_task(0)];

        let loss = trainer.meta_train_step(&tasks);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_learn_regime() {
        let model = TradingModel::new(4, 8, 1);
        let mut trainer = ContinualMAMLTrainer::new(model, 0.01, 0.001, 3, true, 100.0, 50);

        // Learn regime 0
        let tasks_0 = vec![create_dummy_task(0), create_dummy_task(0)];
        let losses_0 = trainer.learn_regime(&tasks_0, 0, 5, 0);
        assert_eq!(losses_0.len(), 5);
        assert_eq!(trainer.regimes_seen().len(), 1);

        // Learn regime 1
        let tasks_1 = vec![create_dummy_task(1), create_dummy_task(1)];
        let losses_1 = trainer.learn_regime(&tasks_1, 1, 5, 0);
        assert_eq!(losses_1.len(), 5);
        assert_eq!(trainer.regimes_seen().len(), 2);
    }

    #[test]
    fn test_adapt() {
        let model = TradingModel::new(4, 8, 1);
        let trainer = ContinualMAMLTrainer::new(model, 0.01, 0.001, 3, true, 100.0, 50);

        let features = vec![vec![0.1, 0.2, 0.3, 0.4]; 10];
        let labels = vec![0.5; 10];

        let adapted = trainer.adapt(&features, &labels, Some(5));
        let prediction = adapted.predict(&[0.1, 0.2, 0.3, 0.4]);
        assert!(prediction.is_finite());
    }

    #[test]
    fn test_evaluate_regime() {
        let model = TradingModel::new(4, 8, 1);
        let trainer = ContinualMAMLTrainer::new(model, 0.01, 0.001, 3, true, 100.0, 50);
        let tasks = vec![create_dummy_task(0)];

        let (loss, accuracy) = trainer.evaluate_regime(&tasks);
        assert!(loss.is_finite());
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(10, 3);

        for i in 0..5 {
            buffer.add(create_dummy_task(0));
            buffer.add(create_dummy_task(1));
            let _ = i;
        }

        assert!(buffer.size() <= 10);
        assert_eq!(buffer.num_regimes(), 2);

        let sampled = buffer.sample(3);
        assert_eq!(sampled.len(), 3);
    }

    #[test]
    fn test_ewc() {
        let model = TradingModel::new(4, 8, 1);
        let tasks = vec![create_dummy_task(0)];

        let mut ewc = EWC::new(100.0);
        assert!(!ewc.is_initialized());

        ewc.compute_fisher(&model, &tasks, 1e-4);
        assert!(ewc.is_initialized());

        let params = model.get_parameters();
        let penalty = ewc.penalty(&params);
        // Penalty should be zero since params haven't changed
        assert!(penalty.abs() < 1e-10);

        // Modified params should give non-zero penalty
        let modified: Vec<f64> = params.iter().map(|p| p + 0.1).collect();
        let penalty_modified = ewc.penalty(&modified);
        assert!(penalty_modified > 0.0);
    }

    #[test]
    fn test_parameter_update_with_ewc() {
        let model = TradingModel::new(4, 8, 1);
        let mut trainer = ContinualMAMLTrainer::new(model, 0.01, 0.001, 3, true, 100.0, 50);

        // Learn first regime
        let tasks_0 = vec![create_dummy_task(0)];
        trainer.learn_regime(&tasks_0, 0, 3, 0);
        let params_after_regime0 = trainer.model().get_parameters();

        // Learn second regime
        let tasks_1 = vec![create_dummy_task(1)];
        trainer.learn_regime(&tasks_1, 1, 3, 0);
        let params_after_regime1 = trainer.model().get_parameters();

        // Parameters should have changed
        let mut changed = false;
        for (p0, p1) in params_after_regime0.iter().zip(params_after_regime1.iter()) {
            if (p0 - p1).abs() > 1e-10 {
                changed = true;
                break;
            }
        }
        assert!(changed, "Parameters should change after learning new regime");
    }
}
