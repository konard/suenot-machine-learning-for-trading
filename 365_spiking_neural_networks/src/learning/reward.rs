//! Reward-modulated STDP (R-STDP)
//!
//! Combines STDP with reinforcement learning by modulating
//! weight changes with a reward signal.

use crate::learning::{LearningRule, STDP, STDPParams};
use crate::neuron::Spike;

/// Eligibility trace for R-STDP
#[derive(Debug, Clone)]
struct EligibilityTrace {
    /// Current trace value
    value: f64,
    /// Time constant for decay
    tau: f64,
}

impl EligibilityTrace {
    fn new(tau: f64) -> Self {
        Self { value: 0.0, tau }
    }

    fn update(&mut self, stdp_value: f64, dt: f64) {
        // Decay existing trace
        self.value *= (-dt / self.tau).exp();
        // Add new STDP contribution
        self.value += stdp_value;
    }

    fn decay(&mut self, dt: f64) {
        self.value *= (-dt / self.tau).exp();
    }

    fn reset(&mut self) {
        self.value = 0.0;
    }
}

/// Reward-modulated STDP learning rule
#[derive(Debug, Clone)]
pub struct RewardModulatedSTDP {
    /// Base STDP rule
    stdp: STDP,
    /// Eligibility traces for each synapse
    eligibility_traces: Vec<Vec<EligibilityTrace>>,
    /// Time constant for eligibility trace
    tau_eligibility: f64,
    /// Reward modulation strength
    reward_factor: f64,
    /// Baseline reward (for computing reward prediction error)
    baseline_reward: f64,
    /// Learning rate for baseline
    baseline_lr: f64,
}

impl RewardModulatedSTDP {
    /// Create a new R-STDP rule
    pub fn new(num_pre: usize, num_post: usize) -> Self {
        let tau_eligibility = 100.0; // 100ms default

        let eligibility_traces = (0..num_pre)
            .map(|_| {
                (0..num_post)
                    .map(|_| EligibilityTrace::new(tau_eligibility))
                    .collect()
            })
            .collect();

        Self {
            stdp: STDP::new(),
            eligibility_traces,
            tau_eligibility,
            reward_factor: 1.0,
            baseline_reward: 0.0,
            baseline_lr: 0.01,
        }
    }

    /// Create with custom STDP parameters
    pub fn with_stdp_params(mut self, params: STDPParams) -> Self {
        self.stdp = STDP::with_params(params);
        self
    }

    /// Set eligibility trace time constant
    pub fn with_tau_eligibility(mut self, tau: f64) -> Self {
        self.tau_eligibility = tau;
        for row in &mut self.eligibility_traces {
            for trace in row {
                trace.tau = tau;
            }
        }
        self
    }

    /// Set reward factor
    pub fn with_reward_factor(mut self, factor: f64) -> Self {
        self.reward_factor = factor;
        self
    }

    /// Update eligibility trace for a specific synapse
    pub fn update_eligibility(
        &mut self,
        pre_idx: usize,
        post_idx: usize,
        pre_spike: Option<&Spike>,
        post_spike: Option<&Spike>,
        dt: f64,
    ) {
        if pre_idx < self.eligibility_traces.len()
            && post_idx < self.eligibility_traces[pre_idx].len()
        {
            let stdp_value = self.stdp.compute_weight_change(pre_spike, post_spike, 0.0);
            self.eligibility_traces[pre_idx][post_idx].update(stdp_value, dt);
        }
    }

    /// Get weight change with reward modulation
    pub fn get_weight_change(&self, pre_idx: usize, post_idx: usize, reward: f64) -> f64 {
        if pre_idx < self.eligibility_traces.len()
            && post_idx < self.eligibility_traces[pre_idx].len()
        {
            let trace = self.eligibility_traces[pre_idx][post_idx].value;
            let reward_error = reward - self.baseline_reward;
            trace * reward_error * self.reward_factor
        } else {
            0.0
        }
    }

    /// Apply reward and get all weight changes
    pub fn apply_reward(&mut self, reward: f64) -> Vec<Vec<f64>> {
        let reward_error = reward - self.baseline_reward;

        // Update baseline with exponential moving average
        self.baseline_reward += self.baseline_lr * reward_error;

        // Compute weight changes
        self.eligibility_traces
            .iter()
            .map(|row| {
                row.iter()
                    .map(|trace| trace.value * reward_error * self.reward_factor)
                    .collect()
            })
            .collect()
    }

    /// Decay all eligibility traces
    pub fn decay_traces(&mut self, dt: f64) {
        for row in &mut self.eligibility_traces {
            for trace in row {
                trace.decay(dt);
            }
        }
    }

    /// Get current baseline reward
    pub fn baseline_reward(&self) -> f64 {
        self.baseline_reward
    }

    /// Reset all state
    pub fn reset_all(&mut self) {
        for row in &mut self.eligibility_traces {
            for trace in row {
                trace.reset();
            }
        }
        self.baseline_reward = 0.0;
    }
}

impl LearningRule for RewardModulatedSTDP {
    fn compute_weight_change(
        &self,
        pre_spike: Option<&Spike>,
        post_spike: Option<&Spike>,
        reward: f64,
    ) -> f64 {
        // Compute STDP
        let stdp_value = self.stdp.compute_weight_change(pre_spike, post_spike, 0.0);

        // Modulate by reward
        let reward_error = reward - self.baseline_reward;
        stdp_value * reward_error * self.reward_factor
    }

    fn update(&mut self, dt: f64) {
        self.decay_traces(dt);
    }

    fn reset(&mut self) {
        self.reset_all();
    }
}

/// Dopamine-modulated STDP
///
/// More biologically realistic model with dopamine dynamics.
#[derive(Debug, Clone)]
pub struct DopamineSTDP {
    /// Base R-STDP
    r_stdp: RewardModulatedSTDP,
    /// Current dopamine level
    dopamine: f64,
    /// Dopamine decay time constant
    tau_dopamine: f64,
    /// Dopamine release rate
    release_rate: f64,
}

impl DopamineSTDP {
    pub fn new(num_pre: usize, num_post: usize) -> Self {
        Self {
            r_stdp: RewardModulatedSTDP::new(num_pre, num_post),
            dopamine: 0.0,
            tau_dopamine: 200.0,
            release_rate: 0.5,
        }
    }

    /// Release dopamine in response to reward
    pub fn release_dopamine(&mut self, reward: f64) {
        let reward_error = reward - self.r_stdp.baseline_reward;
        self.dopamine += reward_error * self.release_rate;

        // Update baseline
        self.r_stdp.baseline_reward += self.r_stdp.baseline_lr * reward_error;
    }

    /// Update dopamine decay
    pub fn update_dopamine(&mut self, dt: f64) {
        self.dopamine *= (-dt / self.tau_dopamine).exp();
    }

    /// Get current dopamine level
    pub fn dopamine_level(&self) -> f64 {
        self.dopamine
    }

    /// Apply dopamine to get weight changes
    pub fn apply_dopamine(&self) -> Vec<Vec<f64>> {
        self.r_stdp
            .eligibility_traces
            .iter()
            .map(|row| {
                row.iter()
                    .map(|trace| trace.value * self.dopamine)
                    .collect()
            })
            .collect()
    }
}

impl LearningRule for DopamineSTDP {
    fn compute_weight_change(
        &self,
        pre_spike: Option<&Spike>,
        post_spike: Option<&Spike>,
        _reward: f64,
    ) -> f64 {
        // Use current dopamine level instead of immediate reward
        let stdp_value = self.r_stdp.stdp.compute_weight_change(pre_spike, post_spike, 0.0);
        stdp_value * self.dopamine
    }

    fn update(&mut self, dt: f64) {
        self.r_stdp.decay_traces(dt);
        self.update_dopamine(dt);
    }

    fn reset(&mut self) {
        self.r_stdp.reset_all();
        self.dopamine = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r_stdp_creation() {
        let r_stdp = RewardModulatedSTDP::new(10, 5);
        assert_eq!(r_stdp.eligibility_traces.len(), 10);
        assert_eq!(r_stdp.eligibility_traces[0].len(), 5);
    }

    #[test]
    fn test_r_stdp_positive_reward() {
        let r_stdp = RewardModulatedSTDP::new(1, 1);

        let pre_spike = Spike::positive(0.0);
        let post_spike = Spike::positive(10.0);

        // Positive STDP + positive reward = positive weight change
        let dw = r_stdp.compute_weight_change(
            Some(&pre_spike),
            Some(&post_spike),
            1.0,
        );

        assert!(dw > 0.0);
    }

    #[test]
    fn test_r_stdp_negative_reward() {
        let r_stdp = RewardModulatedSTDP::new(1, 1);

        let pre_spike = Spike::positive(0.0);
        let post_spike = Spike::positive(10.0);

        // Positive STDP + negative reward = negative weight change
        let dw = r_stdp.compute_weight_change(
            Some(&pre_spike),
            Some(&post_spike),
            -1.0,
        );

        assert!(dw < 0.0);
    }

    #[test]
    fn test_eligibility_trace_decay() {
        let mut r_stdp = RewardModulatedSTDP::new(1, 1);

        let pre_spike = Spike::positive(0.0);
        let post_spike = Spike::positive(10.0);

        // Update eligibility
        r_stdp.update_eligibility(0, 0, Some(&pre_spike), Some(&post_spike), 1.0);

        let initial_trace = r_stdp.eligibility_traces[0][0].value;

        // Decay
        r_stdp.decay_traces(50.0);

        let decayed_trace = r_stdp.eligibility_traces[0][0].value;

        assert!(decayed_trace < initial_trace);
    }

    #[test]
    fn test_baseline_update() {
        let mut r_stdp = RewardModulatedSTDP::new(1, 1);

        assert_eq!(r_stdp.baseline_reward(), 0.0);

        // Apply positive rewards
        for _ in 0..100 {
            r_stdp.apply_reward(1.0);
        }

        // Baseline should move towards 1.0
        assert!(r_stdp.baseline_reward() > 0.5);
    }

    #[test]
    fn test_dopamine_stdp() {
        let mut da_stdp = DopamineSTDP::new(1, 1);

        assert_eq!(da_stdp.dopamine_level(), 0.0);

        // Release dopamine
        da_stdp.release_dopamine(1.0);
        assert!(da_stdp.dopamine_level() > 0.0);

        // Decay
        da_stdp.update_dopamine(100.0);
        let after_decay = da_stdp.dopamine_level();

        da_stdp.release_dopamine(1.0);
        da_stdp.update_dopamine(100.0);

        // Should be decaying
        assert!(da_stdp.dopamine_level() < after_decay * 2.0);
    }
}
