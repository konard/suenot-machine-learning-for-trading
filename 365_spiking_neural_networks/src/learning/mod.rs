//! Learning rules for Spiking Neural Networks
//!
//! This module implements various learning rules:
//! - STDP (Spike-Timing Dependent Plasticity)
//! - R-STDP (Reward-modulated STDP)

mod stdp;
mod reward;

pub use stdp::STDP;
pub use reward::RewardModulatedSTDP;

use crate::neuron::Spike;

/// Common trait for learning rules
pub trait LearningRule: Send + Sync + std::fmt::Debug {
    /// Compute weight change based on pre/post spikes and reward
    fn compute_weight_change(
        &self,
        pre_spike: Option<&Spike>,
        post_spike: Option<&Spike>,
        reward: f64,
    ) -> f64;

    /// Update learning parameters (e.g., eligibility traces)
    fn update(&mut self, dt: f64);

    /// Reset learning state
    fn reset(&mut self);
}

/// Simple Hebbian learning (rate-based)
#[derive(Debug, Clone)]
pub struct HebbianRule {
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay
    pub weight_decay: f64,
}

impl HebbianRule {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            weight_decay: 0.0001,
        }
    }
}

impl LearningRule for HebbianRule {
    fn compute_weight_change(
        &self,
        pre_spike: Option<&Spike>,
        post_spike: Option<&Spike>,
        _reward: f64,
    ) -> f64 {
        match (pre_spike, post_spike) {
            (Some(_), Some(_)) => self.learning_rate,
            (Some(_), None) => -self.weight_decay,
            (None, Some(_)) => -self.weight_decay,
            (None, None) => 0.0,
        }
    }

    fn update(&mut self, _dt: f64) {}

    fn reset(&mut self) {}
}
