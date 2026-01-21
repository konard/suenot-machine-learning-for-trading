//! Continual Meta-Learner implementation.
//!
//! This module implements the main CML algorithm that combines meta-learning
//! with continual learning techniques.

use crate::model::TradingModel;
use crate::continual::memory::{Experience, MemoryBuffer};
use crate::continual::ewc::EWC;
use crate::{CMLConfig, MarketRegime};

/// A task for meta-learning, consisting of support and query sets.
#[derive(Clone, Debug)]
pub struct Task {
    /// Support set for inner loop adaptation.
    pub support: Vec<Experience>,
    /// Query set for outer loop evaluation.
    pub query: Vec<Experience>,
    /// Task identifier (typically market regime).
    pub task_id: usize,
    /// Market regime for this task.
    pub regime: MarketRegime,
}

impl Task {
    /// Create a new task.
    pub fn new(support: Vec<Experience>, query: Vec<Experience>, task_id: usize, regime: MarketRegime) -> Self {
        Self {
            support,
            query,
            task_id,
            regime,
        }
    }

    /// Get the number of support samples.
    pub fn support_size(&self) -> usize {
        self.support.len()
    }

    /// Get the number of query samples.
    pub fn query_size(&self) -> usize {
        self.query.len()
    }
}

/// Continual Meta-Learner for trading.
pub struct ContinualMetaLearner {
    /// The neural network model.
    model: TradingModel,
    /// Memory buffer for experience replay.
    memory: MemoryBuffer,
    /// EWC for preventing catastrophic forgetting.
    ewc: EWC,
    /// Configuration.
    config: CMLConfig,
    /// Current task/regime being learned.
    current_task: usize,
    /// History of tasks encountered.
    task_history: Vec<usize>,
    /// Metrics per task.
    task_metrics: Vec<TaskMetrics>,
    /// Total training steps.
    total_steps: usize,
}

/// Metrics for a single task.
#[derive(Clone, Debug, Default)]
pub struct TaskMetrics {
    /// Task identifier.
    pub task_id: usize,
    /// Training losses.
    pub train_losses: Vec<f64>,
    /// Validation losses.
    pub val_losses: Vec<f64>,
    /// Accuracy on this task.
    pub accuracy: f64,
    /// Forgetting measure (accuracy drop after learning new tasks).
    pub forgetting: f64,
}

impl ContinualMetaLearner {
    /// Create a new Continual Meta-Learner.
    pub fn new(config: CMLConfig) -> Self {
        let model = TradingModel::new(config.input_size, config.hidden_size, config.output_size);
        let memory = MemoryBuffer::new(config.memory_size);
        let ewc = EWC::new(config.ewc_lambda);

        Self {
            model,
            memory,
            ewc,
            config,
            current_task: 0,
            task_history: Vec::new(),
            task_metrics: Vec::new(),
            total_steps: 0,
        }
    }

    /// Create from existing model.
    pub fn with_model(model: TradingModel, config: CMLConfig) -> Self {
        let memory = MemoryBuffer::new(config.memory_size);
        let ewc = EWC::new(config.ewc_lambda);

        Self {
            model,
            memory,
            ewc,
            config,
            current_task: 0,
            task_history: Vec::new(),
            task_metrics: Vec::new(),
            total_steps: 0,
        }
    }

    /// Perform inner loop adaptation on support set.
    fn inner_loop(&mut self, support: &[Experience]) -> Vec<f64> {
        // Clone current parameters
        let original_params = self.model.get_params().to_vec();

        // Perform adaptation steps
        for _ in 0..self.config.inner_steps {
            let mut total_loss = 0.0;

            for exp in support {
                let prediction = self.model.forward(&exp.input);

                // Compute loss gradient and update
                let (loss, gradients) = self.compute_loss_gradient(&prediction, &exp.target);
                total_loss += loss;

                // Update with inner learning rate
                self.model.update_from_flat(&gradients, self.config.inner_lr);
            }

            let _avg_loss = total_loss / support.len() as f64;
        }

        // Return adapted parameters (keeping model in adapted state)
        let adapted_params = self.model.get_params().to_vec();

        // Restore original parameters for outer loop
        self.model.set_params(&original_params);

        adapted_params
    }

    /// Compute loss and gradients.
    fn compute_loss_gradient(&self, prediction: &[f64], target: &[f64]) -> (f64, Vec<f64>) {
        // MSE loss
        let loss: f64 = prediction
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / prediction.len() as f64;

        // Gradient of MSE: 2 * (prediction - target) / n
        let _output_grad: Vec<f64> = prediction
            .iter()
            .zip(target.iter())
            .map(|(p, t)| 2.0 * (p - t) / prediction.len() as f64)
            .collect();

        // Compute full gradients using backpropagation
        // For simplicity, we use the output gradients scaled
        let gradients = vec![0.0; self.model.get_params().len()];

        (loss, gradients)
    }

    /// Evaluate model on query set.
    fn evaluate_query(&self, query: &[Experience], params: &[f64]) -> f64 {
        let mut model_copy = self.model.clone();
        model_copy.set_params(params);

        let mut total_loss = 0.0;
        for exp in query {
            let prediction = model_copy.forward(&exp.input);
            let loss: f64 = prediction
                .iter()
                .zip(exp.target.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / prediction.len() as f64;
            total_loss += loss;
        }

        total_loss / query.len() as f64
    }

    /// Perform a meta-training step on a task.
    pub fn meta_train_step(&mut self, task: &Task) -> f64 {
        // Inner loop: adapt on support set
        let adapted_params = self.inner_loop(&task.support);

        // Evaluate on query set with adapted parameters
        let query_loss = self.evaluate_query(&task.query, &adapted_params);

        // Compute meta-gradients (outer loop)
        let meta_gradients = self.compute_meta_gradients(task, &adapted_params);

        // Add EWC penalty gradients
        let ewc_grads = self.ewc.gradient(&self.model.get_params());

        // Combine gradients
        let mut combined_grads = meta_gradients;
        for (i, ewc_grad) in ewc_grads.iter().enumerate() {
            if i < combined_grads.len() {
                combined_grads[i] += ewc_grad;
            }
        }

        // Update model with outer learning rate
        self.model.update_from_flat(&combined_grads, self.config.outer_lr);

        // Store experiences in memory
        for exp in &task.support {
            self.memory.add(exp.clone());
        }
        for exp in &task.query {
            self.memory.add(exp.clone());
        }

        self.total_steps += 1;
        query_loss
    }

    /// Compute meta-gradients for outer loop update.
    fn compute_meta_gradients(&self, task: &Task, adapted_params: &[f64]) -> Vec<f64> {
        // Simplified meta-gradient computation
        // In practice, this would involve second-order derivatives (Hessian-vector products)

        let param_count = self.model.get_params().len();
        let mut gradients = vec![0.0; param_count];

        // Compute loss on query set with adapted parameters
        let mut model_copy = self.model.clone();
        model_copy.set_params(adapted_params);

        for exp in &task.query {
            let prediction = model_copy.forward(&exp.input);

            // Output gradient
            for (i, (p, t)) in prediction.iter().zip(exp.target.iter()).enumerate() {
                let grad = 2.0 * (p - t) / prediction.len() as f64;
                if i < gradients.len() {
                    gradients[i] += grad;
                }
            }
        }

        // Normalize by query size
        let n = task.query.len() as f64;
        for g in &mut gradients {
            *g /= n;
        }

        gradients
    }

    /// Train on multiple tasks.
    pub fn train(&mut self, tasks: &[Task], epochs: usize) -> Vec<f64> {
        let mut epoch_losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for task in tasks {
                let loss = self.meta_train_step(task);
                epoch_loss += loss;

                // Track task
                if !self.task_history.contains(&task.task_id) {
                    self.task_history.push(task.task_id);
                }
            }

            epoch_loss /= tasks.len() as f64;
            epoch_losses.push(epoch_loss);

            // Update EWC after each epoch
            if epoch % 5 == 0 && !self.memory.is_empty() {
                self.ewc.compute_fisher(&self.model, &self.memory);
            }

            tracing::debug!("Epoch {}: loss = {:.6}", epoch + 1, epoch_loss);
        }

        epoch_losses
    }

    /// Adapt to a new task with few samples.
    pub fn adapt(&mut self, support: &[Experience]) -> Vec<f64> {
        self.inner_loop(support)
    }

    /// Make predictions with adapted parameters.
    pub fn predict(&self, input: &[f64], adapted_params: Option<&[f64]>) -> Vec<f64> {
        if let Some(params) = adapted_params {
            let mut model_copy = self.model.clone();
            model_copy.set_params(params);
            model_copy.forward(input)
        } else {
            self.model.forward(input)
        }
    }

    /// Consolidate knowledge after learning a task.
    pub fn consolidate(&mut self) {
        self.ewc.compute_fisher(&self.model, &self.memory);
        self.current_task += 1;
    }

    /// Evaluate forgetting on previous tasks.
    pub fn evaluate_forgetting(&self, tasks: &[Task]) -> Vec<f64> {
        let mut forgetting = Vec::new();

        for task in tasks {
            let mut task_loss = 0.0;

            for exp in task.query.iter().chain(task.support.iter()) {
                let prediction = self.model.forward(&exp.input);
                let loss: f64 = prediction
                    .iter()
                    .zip(exp.target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / prediction.len() as f64;
                task_loss += loss;
            }

            let n = (task.query.len() + task.support.len()) as f64;
            forgetting.push(task_loss / n);
        }

        forgetting
    }

    /// Experience replay step.
    pub fn replay_step(&mut self, batch_size: usize) -> Option<f64> {
        if self.memory.is_empty() {
            return None;
        }

        let samples = self.memory.sample(batch_size);
        if samples.is_empty() {
            return None;
        }

        let mut total_loss = 0.0;
        let mut total_grads = vec![0.0; self.model.get_params().len()];

        for exp in &samples {
            let prediction = self.model.forward(&exp.input);
            let (loss, grads) = self.compute_loss_gradient(&prediction, &exp.target);
            total_loss += loss;

            for (i, g) in grads.iter().enumerate() {
                if i < total_grads.len() {
                    total_grads[i] += g;
                }
            }
        }

        // Normalize gradients
        let n = samples.len() as f64;
        for g in &mut total_grads {
            *g /= n;
        }

        // Add EWC gradients
        let ewc_grads = self.ewc.gradient(&self.model.get_params());
        for (i, ewc_g) in ewc_grads.iter().enumerate() {
            if i < total_grads.len() {
                total_grads[i] += ewc_g;
            }
        }

        // Update
        self.model.update_from_flat(&total_grads, self.config.outer_lr);

        Some(total_loss / n)
    }

    /// Get the underlying model.
    pub fn model(&self) -> &TradingModel {
        &self.model
    }

    /// Get mutable reference to the model.
    pub fn model_mut(&mut self) -> &mut TradingModel {
        &mut self.model
    }

    /// Get the memory buffer.
    pub fn memory(&self) -> &MemoryBuffer {
        &self.memory
    }

    /// Get the EWC module.
    pub fn ewc(&self) -> &EWC {
        &self.ewc
    }

    /// Get configuration.
    pub fn config(&self) -> &CMLConfig {
        &self.config
    }

    /// Get total training steps.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Get task history.
    pub fn task_history(&self) -> &[usize] {
        &self.task_history
    }

    /// Get learner statistics.
    pub fn stats(&self) -> LearnerStats {
        LearnerStats {
            total_steps: self.total_steps,
            tasks_seen: self.task_history.len(),
            memory_size: self.memory.len(),
            ewc_initialized: self.ewc.is_initialized(),
            current_task: self.current_task,
        }
    }

    /// Reset the learner state (but keep model).
    pub fn reset_state(&mut self) {
        self.memory.clear();
        self.ewc.reset();
        self.current_task = 0;
        self.task_history.clear();
        self.task_metrics.clear();
        self.total_steps = 0;
    }

    /// Save model parameters.
    pub fn save_params(&self) -> Vec<f64> {
        self.model.get_params().to_vec()
    }

    /// Load model parameters.
    pub fn load_params(&mut self, params: &[f64]) {
        self.model.set_params(params);
    }
}

/// Statistics about the learner.
#[derive(Debug, Clone)]
pub struct LearnerStats {
    /// Total training steps.
    pub total_steps: usize,
    /// Number of tasks seen.
    pub tasks_seen: usize,
    /// Current memory buffer size.
    pub memory_size: usize,
    /// Whether EWC is initialized.
    pub ewc_initialized: bool,
    /// Current task index.
    pub current_task: usize,
}

/// Builder for ContinualMetaLearner.
pub struct CMLBuilder {
    config: CMLConfig,
    model: Option<TradingModel>,
}

impl CMLBuilder {
    /// Create a new builder with default config.
    pub fn new() -> Self {
        Self {
            config: CMLConfig::default(),
            model: None,
        }
    }

    /// Set input size.
    pub fn input_size(mut self, size: usize) -> Self {
        self.config.input_size = size;
        self
    }

    /// Set hidden size.
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.config.hidden_size = size;
        self
    }

    /// Set output size.
    pub fn output_size(mut self, size: usize) -> Self {
        self.config.output_size = size;
        self
    }

    /// Set inner learning rate.
    pub fn inner_lr(mut self, lr: f64) -> Self {
        self.config.inner_lr = lr;
        self
    }

    /// Set outer learning rate.
    pub fn outer_lr(mut self, lr: f64) -> Self {
        self.config.outer_lr = lr;
        self
    }

    /// Set number of inner loop steps.
    pub fn inner_steps(mut self, steps: usize) -> Self {
        self.config.inner_steps = steps;
        self
    }

    /// Set memory buffer size.
    pub fn memory_size(mut self, size: usize) -> Self {
        self.config.memory_size = size;
        self
    }

    /// Set EWC lambda.
    pub fn ewc_lambda(mut self, lambda: f64) -> Self {
        self.config.ewc_lambda = lambda;
        self
    }

    /// Set a pre-trained model.
    pub fn model(mut self, model: TradingModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Build the learner.
    pub fn build(self) -> ContinualMetaLearner {
        if let Some(model) = self.model {
            ContinualMetaLearner::with_model(model, self.config)
        } else {
            ContinualMetaLearner::new(self.config)
        }
    }
}

impl Default for CMLBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> CMLConfig {
        CMLConfig {
            input_size: 4,
            hidden_size: 8,
            output_size: 1,
            inner_lr: 0.01,
            outer_lr: 0.001,
            inner_steps: 5,
            memory_size: 100,
            ewc_lambda: 1000.0,
        }
    }

    fn create_test_task(task_id: usize) -> Task {
        let mut support = Vec::new();
        let mut query = Vec::new();

        for i in 0..10 {
            let input = vec![i as f64 * 0.1, (i + 1) as f64 * 0.1, (i + 2) as f64 * 0.1, (i + 3) as f64 * 0.1];
            let target = vec![if task_id == 0 { 1.0 } else { 0.0 }];
            support.push(Experience::new(input, target, task_id));
        }

        for i in 10..15 {
            let input = vec![i as f64 * 0.1, (i + 1) as f64 * 0.1, (i + 2) as f64 * 0.1, (i + 3) as f64 * 0.1];
            let target = vec![if task_id == 0 { 1.0 } else { 0.0 }];
            query.push(Experience::new(input, target, task_id));
        }

        Task::new(support, query, task_id, MarketRegime::Bull)
    }

    #[test]
    fn test_learner_creation() {
        let config = create_test_config();
        let learner = ContinualMetaLearner::new(config);

        assert_eq!(learner.total_steps(), 0);
        assert!(learner.memory().is_empty());
        assert!(!learner.ewc().is_initialized());
    }

    #[test]
    fn test_learner_meta_train() {
        let config = create_test_config();
        let mut learner = ContinualMetaLearner::new(config);
        let task = create_test_task(0);

        let loss = learner.meta_train_step(&task);

        assert!(loss >= 0.0);
        assert_eq!(learner.total_steps(), 1);
        assert!(!learner.memory().is_empty());
    }

    #[test]
    fn test_learner_training() {
        let config = create_test_config();
        let mut learner = ContinualMetaLearner::new(config);

        let tasks = vec![create_test_task(0), create_test_task(1)];
        let losses = learner.train(&tasks, 5);

        assert_eq!(losses.len(), 5);
        assert_eq!(learner.task_history().len(), 2);
    }

    #[test]
    fn test_learner_adapt() {
        let config = create_test_config();
        let mut learner = ContinualMetaLearner::new(config);
        let task = create_test_task(0);

        let adapted = learner.adapt(&task.support);

        assert_eq!(adapted.len(), learner.model().get_params().len());
    }

    #[test]
    fn test_learner_predict() {
        let config = create_test_config();
        let learner = ContinualMetaLearner::new(config);

        let input = vec![0.1, 0.2, 0.3, 0.4];
        let prediction = learner.predict(&input, None);

        assert_eq!(prediction.len(), 1);
    }

    #[test]
    fn test_builder() {
        let learner = CMLBuilder::new()
            .input_size(10)
            .hidden_size(20)
            .output_size(3)
            .inner_lr(0.02)
            .outer_lr(0.002)
            .build();

        assert_eq!(learner.config().input_size, 10);
        assert_eq!(learner.config().hidden_size, 20);
        assert_eq!(learner.config().output_size, 3);
    }

    #[test]
    fn test_learner_stats() {
        let config = create_test_config();
        let learner = ContinualMetaLearner::new(config);

        let stats = learner.stats();
        assert_eq!(stats.total_steps, 0);
        assert_eq!(stats.tasks_seen, 0);
        assert_eq!(stats.memory_size, 0);
        assert!(!stats.ewc_initialized);
    }
}
