//! Multi-task trainer for task-agnostic learning

use super::{GradientHarmonizer, HarmonizerType, TaskWeighter, WeightingStrategy};
use crate::encoder::{create_encoder, EncoderConfig, SharedEncoder};
use crate::tasks::{
    TaskHead, TaskType, MultiTaskPrediction,
    DirectionHead, DirectionConfig,
    VolatilityHead, VolatilityConfig,
    RegimeHead, RegimeConfig,
    ReturnsHead, ReturnsConfig,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trainer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Encoder configuration
    pub encoder_config: EncoderConfig,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Gradient harmonization method
    pub harmonizer_type: HarmonizerType,
    /// Task weighting strategy
    pub weighting_strategy: WeightingStrategy,
    /// Enabled tasks
    pub enabled_tasks: Vec<TaskType>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            encoder_config: EncoderConfig::default(),
            learning_rate: 0.001,
            batch_size: 32,
            harmonizer_type: HarmonizerType::PCGrad,
            weighting_strategy: WeightingStrategy::Uncertainty,
            enabled_tasks: vec![
                TaskType::Direction,
                TaskType::Volatility,
                TaskType::Regime,
                TaskType::Returns,
            ],
        }
    }
}

impl TrainerConfig {
    /// Set encoder config
    pub fn with_encoder(mut self, config: EncoderConfig) -> Self {
        self.encoder_config = config;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set harmonizer type
    pub fn with_harmonizer(mut self, harmonizer: HarmonizerType) -> Self {
        self.harmonizer_type = harmonizer;
        self
    }

    /// Set weighting strategy
    pub fn with_weighting(mut self, strategy: WeightingStrategy) -> Self {
        self.weighting_strategy = strategy;
        self
    }

    /// Set enabled tasks
    pub fn with_tasks(mut self, tasks: Vec<TaskType>) -> Self {
        self.enabled_tasks = tasks;
        self
    }
}

/// Training result for one step
#[derive(Debug, Clone)]
pub struct TrainingStep {
    /// Total loss
    pub total_loss: f64,
    /// Per-task losses
    pub task_losses: HashMap<TaskType, f64>,
    /// Task weights used
    pub task_weights: HashMap<TaskType, f64>,
}

/// Multi-task trainer
pub struct MultiTaskTrainer {
    config: TrainerConfig,
    encoder: Box<dyn SharedEncoder>,
    direction_head: Option<DirectionHead>,
    volatility_head: Option<VolatilityHead>,
    regime_head: Option<RegimeHead>,
    returns_head: Option<ReturnsHead>,
    harmonizer: GradientHarmonizer,
    weighter: TaskWeighter,
    step_count: usize,
}

impl MultiTaskTrainer {
    /// Create a new multi-task trainer
    pub fn new(config: TrainerConfig) -> Self {
        let encoder = create_encoder(&config.encoder_config);
        let embedding_dim = config.encoder_config.embedding_dim;

        let direction_head = if config.enabled_tasks.contains(&TaskType::Direction) {
            Some(DirectionHead::new(DirectionConfig {
                embedding_dim,
                ..Default::default()
            }))
        } else {
            None
        };

        let volatility_head = if config.enabled_tasks.contains(&TaskType::Volatility) {
            Some(VolatilityHead::new(VolatilityConfig {
                embedding_dim,
                ..Default::default()
            }))
        } else {
            None
        };

        let regime_head = if config.enabled_tasks.contains(&TaskType::Regime) {
            Some(RegimeHead::new(RegimeConfig {
                embedding_dim,
                ..Default::default()
            }))
        } else {
            None
        };

        let returns_head = if config.enabled_tasks.contains(&TaskType::Returns) {
            Some(ReturnsHead::new(ReturnsConfig {
                embedding_dim,
                ..Default::default()
            }))
        } else {
            None
        };

        let harmonizer = GradientHarmonizer::new(config.harmonizer_type);
        let weighter = TaskWeighter::new(config.weighting_strategy);

        Self {
            config,
            encoder,
            direction_head,
            volatility_head,
            regime_head,
            returns_head,
            harmonizer,
            weighter,
            step_count: 0,
        }
    }

    /// Forward pass through encoder and all task heads
    pub fn forward(&self, features: &Array2<f64>) -> HashMap<TaskType, Array2<f64>> {
        // Encode features
        let embeddings = self.encoder.encode_batch(features);

        let mut outputs = HashMap::new();

        if let Some(ref head) = self.direction_head {
            outputs.insert(TaskType::Direction, head.forward_batch(&embeddings));
        }

        if let Some(ref head) = self.volatility_head {
            outputs.insert(TaskType::Volatility, head.forward_batch(&embeddings));
        }

        if let Some(ref head) = self.regime_head {
            outputs.insert(TaskType::Regime, head.forward_batch(&embeddings));
        }

        if let Some(ref head) = self.returns_head {
            outputs.insert(TaskType::Returns, head.forward_batch(&embeddings));
        }

        outputs
    }

    /// Predict on a single sample
    pub fn predict(&self, features: &Array1<f64>) -> MultiTaskPrediction {
        let embedding = self.encoder.encode(features);
        let mut prediction = MultiTaskPrediction::new();

        if let Some(ref head) = self.direction_head {
            prediction.direction = Some(head.predict(&embedding));
        }

        if let Some(ref head) = self.volatility_head {
            prediction.volatility = Some(head.predict(&embedding));
        }

        if let Some(ref head) = self.regime_head {
            prediction.regime = Some(head.predict(&embedding));
        }

        if let Some(ref head) = self.returns_head {
            prediction.returns = Some(head.predict(&embedding));
        }

        prediction
    }

    /// Compute losses for all tasks
    pub fn compute_losses(
        &self,
        predictions: &HashMap<TaskType, Array2<f64>>,
        targets: &HashMap<TaskType, Array2<f64>>,
    ) -> HashMap<TaskType, f64> {
        let mut losses = HashMap::new();

        if let (Some(pred), Some(target)) = (predictions.get(&TaskType::Direction), targets.get(&TaskType::Direction)) {
            if let Some(ref head) = self.direction_head {
                losses.insert(TaskType::Direction, head.compute_loss(pred, target));
            }
        }

        if let (Some(pred), Some(target)) = (predictions.get(&TaskType::Volatility), targets.get(&TaskType::Volatility)) {
            if let Some(ref head) = self.volatility_head {
                losses.insert(TaskType::Volatility, head.compute_loss(pred, target));
            }
        }

        if let (Some(pred), Some(target)) = (predictions.get(&TaskType::Regime), targets.get(&TaskType::Regime)) {
            if let Some(ref head) = self.regime_head {
                losses.insert(TaskType::Regime, head.compute_loss(pred, target));
            }
        }

        if let (Some(pred), Some(target)) = (predictions.get(&TaskType::Returns), targets.get(&TaskType::Returns)) {
            if let Some(ref head) = self.returns_head {
                losses.insert(TaskType::Returns, head.compute_loss(pred, target));
            }
        }

        losses
    }

    /// Single training step (forward + backward + update)
    /// Note: This is a simplified version without automatic differentiation
    pub fn train_step(
        &mut self,
        features: &Array2<f64>,
        targets: &HashMap<TaskType, Array2<f64>>,
    ) -> TrainingStep {
        // Forward pass
        let predictions = self.forward(features);

        // Compute losses
        let task_losses = self.compute_losses(&predictions, targets);

        // Update weighter
        for (task, &loss) in &task_losses {
            self.weighter.update(*task, loss);
        }

        // Get weighted total loss
        let task_weights = self.weighter.get_all_weights();
        let total_loss = self.weighter.weighted_loss(&task_losses);

        self.step_count += 1;

        TrainingStep {
            total_loss,
            task_losses,
            task_weights,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &TrainerConfig {
        &self.config
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get task weights
    pub fn get_task_weights(&self) -> HashMap<TaskType, f64> {
        self.weighter.get_all_weights()
    }

    /// Check if a task is enabled
    pub fn is_task_enabled(&self, task: TaskType) -> bool {
        self.config.enabled_tasks.contains(&task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_trainer_creation() {
        let config = TrainerConfig::default();
        let trainer = MultiTaskTrainer::new(config);

        assert!(trainer.is_task_enabled(TaskType::Direction));
        assert!(trainer.is_task_enabled(TaskType::Volatility));
    }

    #[test]
    fn test_forward() {
        let config = TrainerConfig::default()
            .with_encoder(EncoderConfig::default().with_input_dim(10));
        let trainer = MultiTaskTrainer::new(config);

        let features = Array2::random((5, 10), Uniform::new(-1.0, 1.0));
        let outputs = trainer.forward(&features);

        assert!(outputs.contains_key(&TaskType::Direction));
        assert_eq!(outputs[&TaskType::Direction].nrows(), 5);
    }

    #[test]
    fn test_predict() {
        let config = TrainerConfig::default()
            .with_encoder(EncoderConfig::default().with_input_dim(10));
        let trainer = MultiTaskTrainer::new(config);

        let features = Array1::random(10, Uniform::new(-1.0, 1.0));
        let prediction = trainer.predict(&features);

        assert!(prediction.direction.is_some());
        assert!(prediction.volatility.is_some());
        assert!(prediction.regime.is_some());
        assert!(prediction.returns.is_some());
    }
}
