//! Synapse Model
//!
//! Synapses connect neurons and transmit spikes with configurable
//! weights and delays.
//!
//! ## Features
//!
//! - Configurable synaptic weight
//! - Transmission delay
//! - Spike-timing-dependent plasticity (STDP) support

use super::SpikeEvent;

/// Synapse configuration
#[derive(Debug, Clone, Copy)]
pub struct SynapseConfig {
    /// Synaptic weight (positive for excitatory, negative for inhibitory)
    pub weight: f64,
    /// Transmission delay in ms
    pub delay: f64,
    /// Maximum weight (for plasticity bounds)
    pub weight_max: f64,
    /// Minimum weight
    pub weight_min: f64,
    /// Whether synapse is plastic (can change weight)
    pub plastic: bool,
}

impl Default for SynapseConfig {
    fn default() -> Self {
        Self {
            weight: 0.5,
            delay: 1.0,
            weight_max: 1.0,
            weight_min: 0.0,
            plastic: true,
        }
    }
}

impl SynapseConfig {
    /// Create an excitatory synapse
    pub fn excitatory(weight: f64) -> Self {
        Self {
            weight: weight.abs(),
            ..Default::default()
        }
    }

    /// Create an inhibitory synapse
    pub fn inhibitory(weight: f64) -> Self {
        Self {
            weight: -weight.abs(),
            weight_min: -1.0,
            ..Default::default()
        }
    }
}

/// Synapse connecting two neurons
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Pre-synaptic neuron ID
    pub pre_id: usize,
    /// Post-synaptic neuron ID
    pub post_id: usize,
    /// Configuration
    config: SynapseConfig,
    /// Queue of spikes waiting to be delivered (time, weight)
    spike_queue: Vec<(f64, f64)>,
    /// Last pre-synaptic spike time (for STDP)
    last_pre_spike: f64,
    /// Last post-synaptic spike time (for STDP)
    last_post_spike: f64,
    /// Eligibility trace for learning
    eligibility_trace: f64,
}

impl Synapse {
    /// Create a new synapse
    pub fn new(pre_id: usize, post_id: usize, config: SynapseConfig) -> Self {
        Self {
            pre_id,
            post_id,
            config,
            spike_queue: Vec::new(),
            last_pre_spike: -f64::INFINITY,
            last_post_spike: -f64::INFINITY,
            eligibility_trace: 0.0,
        }
    }

    /// Get the current weight
    pub fn weight(&self) -> f64 {
        self.config.weight
    }

    /// Set the weight (clamped to bounds)
    pub fn set_weight(&mut self, weight: f64) {
        self.config.weight = weight.clamp(self.config.weight_min, self.config.weight_max);
    }

    /// Get the delay
    pub fn delay(&self) -> f64 {
        self.config.delay
    }

    /// Check if synapse is excitatory
    pub fn is_excitatory(&self) -> bool {
        self.config.weight > 0.0
    }

    /// Check if synapse is inhibitory
    pub fn is_inhibitory(&self) -> bool {
        self.config.weight < 0.0
    }

    /// Check if synapse is plastic
    pub fn is_plastic(&self) -> bool {
        self.config.plastic
    }

    /// Record a pre-synaptic spike
    pub fn pre_spike(&mut self, time: f64) {
        self.last_pre_spike = time;
        // Add spike to queue with delay
        self.spike_queue.push((time + self.config.delay, self.config.weight));
    }

    /// Record a post-synaptic spike
    pub fn post_spike(&mut self, time: f64) {
        self.last_post_spike = time;
    }

    /// Get spikes ready to be delivered at current time
    pub fn get_deliverable_spikes(&mut self, current_time: f64) -> Vec<f64> {
        let mut deliverable = Vec::new();
        self.spike_queue.retain(|&(time, weight)| {
            if time <= current_time {
                deliverable.push(weight);
                false
            } else {
                true
            }
        });
        deliverable
    }

    /// Apply STDP learning rule
    ///
    /// Returns the weight change
    pub fn apply_stdp(&mut self, a_plus: f64, a_minus: f64, tau_plus: f64, tau_minus: f64) -> f64 {
        if !self.config.plastic {
            return 0.0;
        }

        let dt = self.last_post_spike - self.last_pre_spike;
        let dw = if dt > 0.0 {
            // Pre before post -> potentiation
            a_plus * (-dt / tau_plus).exp()
        } else if dt < 0.0 {
            // Post before pre -> depression
            -a_minus * (dt / tau_minus).exp()
        } else {
            0.0
        };

        let new_weight = self.config.weight + dw;
        self.set_weight(new_weight);

        dw
    }

    /// Get eligibility trace
    pub fn eligibility(&self) -> f64 {
        self.eligibility_trace
    }

    /// Update eligibility trace
    pub fn update_eligibility(&mut self, stdp_value: f64, decay: f64) {
        self.eligibility_trace = self.eligibility_trace * decay + stdp_value;
    }

    /// Apply reward-modulated learning
    pub fn apply_reward(&mut self, reward: f64, learning_rate: f64) {
        if !self.config.plastic {
            return;
        }

        let dw = reward * self.eligibility_trace * learning_rate;
        let new_weight = self.config.weight + dw;
        self.set_weight(new_weight);
    }

    /// Reset spike timing information
    pub fn reset_timing(&mut self) {
        self.last_pre_spike = -f64::INFINITY;
        self.last_post_spike = -f64::INFINITY;
        self.spike_queue.clear();
        self.eligibility_trace = 0.0;
    }
}

/// Builder for creating synapses
pub struct SynapseBuilder {
    pre_id: usize,
    post_id: usize,
    config: SynapseConfig,
}

impl SynapseBuilder {
    /// Create a new synapse builder
    pub fn new(pre_id: usize, post_id: usize) -> Self {
        Self {
            pre_id,
            post_id,
            config: SynapseConfig::default(),
        }
    }

    /// Set the weight
    pub fn weight(mut self, weight: f64) -> Self {
        self.config.weight = weight;
        self
    }

    /// Set the delay
    pub fn delay(mut self, delay: f64) -> Self {
        self.config.delay = delay;
        self
    }

    /// Set weight bounds
    pub fn bounds(mut self, min: f64, max: f64) -> Self {
        self.config.weight_min = min;
        self.config.weight_max = max;
        self
    }

    /// Set plasticity
    pub fn plastic(mut self, plastic: bool) -> Self {
        self.config.plastic = plastic;
        self
    }

    /// Build the synapse
    pub fn build(self) -> Synapse {
        Synapse::new(self.pre_id, self.post_id, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synapse_creation() {
        let synapse = Synapse::new(0, 1, SynapseConfig::excitatory(0.8));
        assert_eq!(synapse.pre_id, 0);
        assert_eq!(synapse.post_id, 1);
        assert_eq!(synapse.weight(), 0.8);
        assert!(synapse.is_excitatory());
    }

    #[test]
    fn test_inhibitory_synapse() {
        let synapse = Synapse::new(0, 1, SynapseConfig::inhibitory(0.5));
        assert!(synapse.is_inhibitory());
        assert_eq!(synapse.weight(), -0.5);
    }

    #[test]
    fn test_spike_delay() {
        let mut synapse = Synapse::new(0, 1, SynapseConfig {
            delay: 5.0,
            ..Default::default()
        });

        synapse.pre_spike(10.0);

        // Too early
        assert!(synapse.get_deliverable_spikes(12.0).is_empty());

        // Just right
        let spikes = synapse.get_deliverable_spikes(15.0);
        assert_eq!(spikes.len(), 1);
    }

    #[test]
    fn test_stdp_potentiation() {
        let mut synapse = Synapse::new(0, 1, SynapseConfig::excitatory(0.5));

        synapse.pre_spike(10.0);
        synapse.post_spike(15.0);  // Post after pre -> potentiation

        let dw = synapse.apply_stdp(0.1, 0.1, 20.0, 20.0);
        assert!(dw > 0.0, "Pre before post should increase weight");
    }

    #[test]
    fn test_stdp_depression() {
        let mut synapse = Synapse::new(0, 1, SynapseConfig::excitatory(0.5));

        synapse.post_spike(10.0);
        synapse.pre_spike(15.0);  // Pre after post -> depression

        let dw = synapse.apply_stdp(0.1, 0.1, 20.0, 20.0);
        assert!(dw < 0.0, "Post before pre should decrease weight");
    }

    #[test]
    fn test_weight_bounds() {
        let mut synapse = Synapse::new(0, 1, SynapseConfig {
            weight: 0.5,
            weight_min: 0.0,
            weight_max: 1.0,
            ..Default::default()
        });

        synapse.set_weight(1.5);
        assert_eq!(synapse.weight(), 1.0);

        synapse.set_weight(-0.5);
        assert_eq!(synapse.weight(), 0.0);
    }

    #[test]
    fn test_synapse_builder() {
        let synapse = SynapseBuilder::new(0, 1)
            .weight(0.7)
            .delay(2.0)
            .bounds(0.1, 0.9)
            .plastic(true)
            .build();

        assert_eq!(synapse.weight(), 0.7);
        assert_eq!(synapse.delay(), 2.0);
        assert!(synapse.is_plastic());
    }
}
