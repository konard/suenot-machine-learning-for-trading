//! Learning rate schedulers for training
//!
//! Provides various learning rate scheduling strategies.

use serde::{Deserialize, Serialize};

/// Types of learning rate schedulers
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SchedulerType {
    /// Constant learning rate
    Constant,
    /// Step decay: multiply by gamma every step_size epochs
    StepDecay,
    /// Exponential decay
    ExponentialDecay,
    /// Cosine annealing
    CosineAnnealing,
    /// Linear warmup followed by decay
    WarmupDecay,
}

/// Learning rate scheduler
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    scheduler_type: SchedulerType,
    initial_lr: f64,
    min_lr: f64,
    /// For step decay
    step_size: usize,
    gamma: f64,
    /// For warmup
    warmup_steps: usize,
    /// Total training steps (for cosine annealing)
    total_steps: usize,
}

impl LearningRateScheduler {
    /// Create a constant learning rate scheduler
    pub fn constant(lr: f64) -> Self {
        Self {
            scheduler_type: SchedulerType::Constant,
            initial_lr: lr,
            min_lr: lr,
            step_size: 1,
            gamma: 1.0,
            warmup_steps: 0,
            total_steps: 1,
        }
    }

    /// Create a step decay scheduler
    ///
    /// Learning rate is multiplied by gamma every step_size steps
    pub fn step_decay(initial_lr: f64, step_size: usize, gamma: f64, min_lr: f64) -> Self {
        Self {
            scheduler_type: SchedulerType::StepDecay,
            initial_lr,
            min_lr,
            step_size,
            gamma,
            warmup_steps: 0,
            total_steps: 1,
        }
    }

    /// Create an exponential decay scheduler
    ///
    /// lr = initial_lr * gamma^step
    pub fn exponential_decay(initial_lr: f64, gamma: f64, min_lr: f64) -> Self {
        Self {
            scheduler_type: SchedulerType::ExponentialDecay,
            initial_lr,
            min_lr,
            step_size: 1,
            gamma,
            warmup_steps: 0,
            total_steps: 1,
        }
    }

    /// Create a cosine annealing scheduler
    ///
    /// lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * step / total_steps))
    pub fn cosine_annealing(initial_lr: f64, total_steps: usize, min_lr: f64) -> Self {
        Self {
            scheduler_type: SchedulerType::CosineAnnealing,
            initial_lr,
            min_lr,
            step_size: 1,
            gamma: 1.0,
            warmup_steps: 0,
            total_steps,
        }
    }

    /// Create a warmup + decay scheduler
    ///
    /// Linear warmup for warmup_steps, then exponential decay
    pub fn warmup_decay(
        initial_lr: f64,
        warmup_steps: usize,
        gamma: f64,
        min_lr: f64,
    ) -> Self {
        Self {
            scheduler_type: SchedulerType::WarmupDecay,
            initial_lr,
            min_lr,
            step_size: 1,
            gamma,
            warmup_steps,
            total_steps: 1,
        }
    }

    /// Get the learning rate for a given step
    pub fn step(&self, current_step: usize) -> f64 {
        let lr = match self.scheduler_type {
            SchedulerType::Constant => self.initial_lr,

            SchedulerType::StepDecay => {
                let num_decays = current_step / self.step_size;
                self.initial_lr * self.gamma.powi(num_decays as i32)
            }

            SchedulerType::ExponentialDecay => {
                self.initial_lr * self.gamma.powi(current_step as i32)
            }

            SchedulerType::CosineAnnealing => {
                let progress = current_step as f64 / self.total_steps as f64;
                let cosine = (std::f64::consts::PI * progress).cos();
                self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + cosine)
            }

            SchedulerType::WarmupDecay => {
                if current_step < self.warmup_steps {
                    // Linear warmup
                    self.initial_lr * (current_step as f64 + 1.0) / self.warmup_steps as f64
                } else {
                    // Exponential decay after warmup
                    let decay_step = current_step - self.warmup_steps;
                    self.initial_lr * self.gamma.powi(decay_step as i32)
                }
            }
        };

        lr.max(self.min_lr)
    }

    /// Get learning rates for a range of steps (useful for visualization)
    pub fn get_schedule(&self, n_steps: usize) -> Vec<f64> {
        (0..n_steps).map(|step| self.step(step)).collect()
    }

    /// Get the scheduler type
    pub fn scheduler_type(&self) -> SchedulerType {
        self.scheduler_type
    }
}

impl Default for LearningRateScheduler {
    fn default() -> Self {
        Self::constant(0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let scheduler = LearningRateScheduler::constant(0.01);

        assert_eq!(scheduler.step(0), 0.01);
        assert_eq!(scheduler.step(100), 0.01);
        assert_eq!(scheduler.step(1000), 0.01);
    }

    #[test]
    fn test_step_decay_scheduler() {
        let scheduler = LearningRateScheduler::step_decay(0.1, 100, 0.5, 0.0001);

        assert!((scheduler.step(0) - 0.1).abs() < 1e-10);
        assert!((scheduler.step(99) - 0.1).abs() < 1e-10);
        assert!((scheduler.step(100) - 0.05).abs() < 1e-10);
        assert!((scheduler.step(200) - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_decay_scheduler() {
        let scheduler = LearningRateScheduler::exponential_decay(0.1, 0.99, 0.0001);

        assert!((scheduler.step(0) - 0.1).abs() < 1e-10);
        assert!(scheduler.step(100) < scheduler.step(0));
        assert!(scheduler.step(100) >= 0.0001);
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        let scheduler = LearningRateScheduler::cosine_annealing(0.1, 1000, 0.0001);

        // At start, should be near initial_lr
        assert!((scheduler.step(0) - 0.1).abs() < 0.01);

        // At middle, should be between initial and min
        let mid_lr = scheduler.step(500);
        assert!(mid_lr < 0.1 && mid_lr > 0.0001);

        // At end, should be near min_lr
        assert!((scheduler.step(1000) - 0.0001).abs() < 0.01);
    }

    #[test]
    fn test_warmup_decay_scheduler() {
        let scheduler = LearningRateScheduler::warmup_decay(0.1, 100, 0.99, 0.0001);

        // During warmup, should increase
        assert!(scheduler.step(0) < scheduler.step(50));
        assert!(scheduler.step(50) < scheduler.step(99));

        // After warmup, should decrease
        assert!(scheduler.step(100) > scheduler.step(200));
    }

    #[test]
    fn test_min_lr_enforcement() {
        let scheduler = LearningRateScheduler::exponential_decay(0.1, 0.5, 0.01);

        // After many steps, should not go below min_lr
        assert!(scheduler.step(1000) >= 0.01);
    }

    #[test]
    fn test_get_schedule() {
        let scheduler = LearningRateScheduler::step_decay(0.1, 10, 0.9, 0.001);
        let schedule = scheduler.get_schedule(30);

        assert_eq!(schedule.len(), 30);
        assert!((schedule[0] - 0.1).abs() < 1e-10);
        assert!(schedule[20] < schedule[0]);
    }
}
