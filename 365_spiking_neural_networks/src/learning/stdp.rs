//! Spike-Timing Dependent Plasticity (STDP)
//!
//! Classic biological learning rule where weight changes depend on
//! the relative timing of pre- and post-synaptic spikes.

use crate::learning::LearningRule;
use crate::neuron::Spike;

/// STDP learning rule parameters
#[derive(Debug, Clone)]
pub struct STDPParams {
    /// Amplitude of potentiation (LTP)
    pub a_plus: f64,
    /// Amplitude of depression (LTD)
    pub a_minus: f64,
    /// Time constant for potentiation (ms)
    pub tau_plus: f64,
    /// Time constant for depression (ms)
    pub tau_minus: f64,
    /// Maximum weight
    pub w_max: f64,
    /// Minimum weight
    pub w_min: f64,
}

impl Default for STDPParams {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,  // Slightly stronger depression
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_max: 1.0,
            w_min: 0.0,
        }
    }
}

impl STDPParams {
    /// Create symmetric STDP parameters
    pub fn symmetric(amplitude: f64, tau: f64) -> Self {
        Self {
            a_plus: amplitude,
            a_minus: amplitude,
            tau_plus: tau,
            tau_minus: tau,
            ..Default::default()
        }
    }

    /// Create parameters for excitatory synapses
    pub fn excitatory() -> Self {
        Self::default()
    }

    /// Create parameters for inhibitory synapses
    pub fn inhibitory() -> Self {
        Self {
            a_plus: 0.005,
            a_minus: 0.005,
            tau_plus: 30.0,
            tau_minus: 30.0,
            w_max: 0.0,
            w_min: -1.0,
        }
    }
}

/// Standard STDP learning rule
#[derive(Debug, Clone)]
pub struct STDP {
    /// STDP parameters
    params: STDPParams,
}

impl STDP {
    /// Create a new STDP rule with default parameters
    pub fn new() -> Self {
        Self {
            params: STDPParams::default(),
        }
    }

    /// Create STDP with custom parameters
    pub fn with_params(params: STDPParams) -> Self {
        Self { params }
    }

    /// Compute STDP weight change from spike times
    pub fn compute_stdp(&self, pre_time: f64, post_time: f64) -> f64 {
        let dt = post_time - pre_time;

        if dt > 0.0 {
            // Pre before post: potentiation (LTP)
            self.params.a_plus * (-dt / self.params.tau_plus).exp()
        } else if dt < 0.0 {
            // Post before pre: depression (LTD)
            -self.params.a_minus * (dt / self.params.tau_minus).exp()
        } else {
            0.0
        }
    }

    /// Get parameters
    pub fn params(&self) -> &STDPParams {
        &self.params
    }

    /// Set learning rates
    pub fn set_learning_rates(&mut self, a_plus: f64, a_minus: f64) {
        self.params.a_plus = a_plus;
        self.params.a_minus = a_minus;
    }
}

impl Default for STDP {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningRule for STDP {
    fn compute_weight_change(
        &self,
        pre_spike: Option<&Spike>,
        post_spike: Option<&Spike>,
        _reward: f64,
    ) -> f64 {
        match (pre_spike, post_spike) {
            (Some(pre), Some(post)) => self.compute_stdp(pre.time, post.time),
            _ => 0.0,
        }
    }

    fn update(&mut self, _dt: f64) {
        // Standard STDP doesn't have internal state to update
    }

    fn reset(&mut self) {
        // Nothing to reset
    }
}

/// Triplet STDP rule
///
/// More accurate model that considers triplets of spikes.
#[derive(Debug, Clone)]
pub struct TripletSTDP {
    /// Base STDP parameters
    params: STDPParams,
    /// Additional amplitude for triplet term
    a_triplet: f64,
    /// Time constant for triplet term
    tau_triplet: f64,
    /// Trace of pre-synaptic activity
    pre_trace: f64,
    /// Trace of post-synaptic activity
    post_trace: f64,
    /// Slow trace for triplet
    slow_trace: f64,
}

impl TripletSTDP {
    pub fn new() -> Self {
        Self {
            params: STDPParams::default(),
            a_triplet: 0.005,
            tau_triplet: 100.0,
            pre_trace: 0.0,
            post_trace: 0.0,
            slow_trace: 0.0,
        }
    }

    /// Update traces with time step
    fn decay_traces(&mut self, dt: f64) {
        self.pre_trace *= (-dt / self.params.tau_plus).exp();
        self.post_trace *= (-dt / self.params.tau_minus).exp();
        self.slow_trace *= (-dt / self.tau_triplet).exp();
    }
}

impl Default for TripletSTDP {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningRule for TripletSTDP {
    fn compute_weight_change(
        &self,
        pre_spike: Option<&Spike>,
        post_spike: Option<&Spike>,
        _reward: f64,
    ) -> f64 {
        let mut dw = 0.0;

        if pre_spike.is_some() {
            // Pre spike: LTD
            dw -= self.params.a_minus * self.post_trace;
        }

        if post_spike.is_some() {
            // Post spike: LTP + triplet term
            dw += self.params.a_plus * self.pre_trace;
            dw += self.a_triplet * self.pre_trace * self.slow_trace;
        }

        dw
    }

    fn update(&mut self, dt: f64) {
        self.decay_traces(dt);
    }

    fn reset(&mut self) {
        self.pre_trace = 0.0;
        self.post_trace = 0.0;
        self.slow_trace = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_ltp() {
        let stdp = STDP::new();

        // Pre before post (dt > 0) should give LTP
        let dw = stdp.compute_stdp(0.0, 10.0);
        assert!(dw > 0.0, "Pre before post should potentiate");
    }

    #[test]
    fn test_stdp_ltd() {
        let stdp = STDP::new();

        // Post before pre (dt < 0) should give LTD
        let dw = stdp.compute_stdp(10.0, 0.0);
        assert!(dw < 0.0, "Post before pre should depress");
    }

    #[test]
    fn test_stdp_timing_dependence() {
        let stdp = STDP::new();

        // Closer spikes should have larger effect
        let dw_close = stdp.compute_stdp(0.0, 5.0);
        let dw_far = stdp.compute_stdp(0.0, 50.0);

        assert!(dw_close > dw_far);
    }

    #[test]
    fn test_stdp_with_spikes() {
        let stdp = STDP::new();

        let pre_spike = Spike::positive(0.0);
        let post_spike = Spike::positive(10.0);

        let dw = stdp.compute_weight_change(
            Some(&pre_spike),
            Some(&post_spike),
            0.0,
        );

        assert!(dw > 0.0);
    }

    #[test]
    fn test_no_spikes() {
        let stdp = STDP::new();

        let dw = stdp.compute_weight_change(None, None, 0.0);
        assert!((dw - 0.0).abs() < 1e-10);
    }
}
