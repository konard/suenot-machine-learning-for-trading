//! Leaky Integrate-and-Fire (LIF) Neuron Model
//!
//! The LIF model is the simplest and most commonly used spiking neuron model.
//! It captures the essential dynamics of neural integration with a leak term.
//!
//! ## Dynamics
//!
//! ```text
//! τ_m * dV/dt = -(V - V_rest) + R * I(t)
//!
//! if V >= V_threshold:
//!     emit spike
//!     V = V_reset
//! ```
//!
//! ## Example
//!
//! ```rust
//! use neuromorphic_trading::neuron::lif::{LIFNeuron, LIFConfig};
//!
//! let config = LIFConfig {
//!     tau_m: 20.0,      // membrane time constant (ms)
//!     threshold: 1.0,    // spike threshold
//!     reset: 0.0,        // reset potential after spike
//!     rest: 0.0,         // resting potential
//!     refractory_period: 2.0,  // refractory period (ms)
//! };
//!
//! let mut neuron = LIFNeuron::new(0, config);
//! ```

use super::{Neuron, NeuronState, SpikeEvent};

/// Configuration for LIF neuron
#[derive(Debug, Clone, Copy)]
pub struct LIFConfig {
    /// Membrane time constant (ms)
    pub tau_m: f64,
    /// Spike threshold
    pub threshold: f64,
    /// Reset potential after spike
    pub reset: f64,
    /// Resting potential
    pub rest: f64,
    /// Refractory period (ms)
    pub refractory_period: f64,
}

impl Default for LIFConfig {
    fn default() -> Self {
        Self {
            tau_m: 20.0,
            threshold: 1.0,
            reset: 0.0,
            rest: 0.0,
            refractory_period: 2.0,
        }
    }
}

/// Leaky Integrate-and-Fire neuron
#[derive(Debug, Clone)]
pub struct LIFNeuron {
    /// Unique neuron ID
    id: usize,
    /// Configuration
    config: LIFConfig,
    /// Current membrane potential
    membrane_potential: f64,
    /// Current simulation time
    current_time: f64,
    /// Time of last spike
    last_spike_time: f64,
    /// Whether in refractory period
    refractory: bool,
    /// Total spike count
    spike_count: u64,
    /// Accumulated input current
    input_buffer: f64,
    /// Spike times for rate calculation
    recent_spikes: Vec<f64>,
}

impl LIFNeuron {
    /// Create a new LIF neuron with given ID and configuration
    pub fn new(id: usize, config: LIFConfig) -> Self {
        Self {
            id,
            config,
            membrane_potential: config.rest,
            current_time: 0.0,
            last_spike_time: -f64::INFINITY,
            refractory: false,
            spike_count: 0,
            input_buffer: 0.0,
            recent_spikes: Vec::new(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &LIFConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: LIFConfig) {
        self.config = config;
    }

    /// Get time since last spike
    pub fn time_since_spike(&self) -> f64 {
        self.current_time - self.last_spike_time
    }

    /// Clean up old spike times for rate calculation
    fn cleanup_spike_history(&mut self, window: f64) {
        let cutoff = self.current_time - window;
        self.recent_spikes.retain(|&t| t > cutoff);
    }
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self::new(0, LIFConfig::default())
    }
}

impl Neuron for LIFNeuron {
    fn id(&self) -> usize {
        self.id
    }

    fn state(&self) -> NeuronState {
        NeuronState {
            membrane_potential: self.membrane_potential,
            recovery: 0.0,
            time_since_spike: self.time_since_spike(),
            refractory: self.refractory,
            spike_count: self.spike_count,
        }
    }

    fn reset(&mut self) {
        self.membrane_potential = self.config.rest;
        self.current_time = 0.0;
        self.last_spike_time = -f64::INFINITY;
        self.refractory = false;
        self.spike_count = 0;
        self.input_buffer = 0.0;
        self.recent_spikes.clear();
    }

    fn step(&mut self, input_current: f64, dt: f64) -> Option<SpikeEvent> {
        self.current_time += dt;

        // Check if still in refractory period
        if self.refractory {
            if self.time_since_spike() >= self.config.refractory_period {
                self.refractory = false;
            } else {
                return None;
            }
        }

        // Total input = external + buffered
        let total_input = input_current + self.input_buffer;
        self.input_buffer = 0.0;

        // LIF dynamics: τ_m * dV/dt = -(V - V_rest) + I
        let dv = (-( self.membrane_potential - self.config.rest) + total_input) * dt / self.config.tau_m;
        self.membrane_potential += dv;

        // Check for spike
        if self.membrane_potential >= self.config.threshold {
            // Generate spike
            self.membrane_potential = self.config.reset;
            self.last_spike_time = self.current_time;
            self.refractory = true;
            self.spike_count += 1;
            self.recent_spikes.push(self.current_time);

            // Clean up old spikes
            self.cleanup_spike_history(1000.0);

            return Some(SpikeEvent::new(self.id, self.current_time));
        }

        None
    }

    fn is_refractory(&self) -> bool {
        self.refractory
    }

    fn membrane_potential(&self) -> f64 {
        self.membrane_potential
    }

    fn threshold(&self) -> f64 {
        self.config.threshold
    }

    fn apply_input(&mut self, input: f64) {
        self.input_buffer += input;
    }

    fn spike_rate(&self, time_window: f64) -> f64 {
        let cutoff = self.current_time - time_window;
        let count = self.recent_spikes.iter().filter(|&&t| t > cutoff).count();
        (count as f64) / (time_window / 1000.0)  // spikes per second
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_default() {
        let neuron = LIFNeuron::default();
        assert_eq!(neuron.id(), 0);
        assert_eq!(neuron.membrane_potential(), 0.0);
        assert_eq!(neuron.threshold(), 1.0);
    }

    #[test]
    fn test_lif_integration() {
        let mut neuron = LIFNeuron::default();

        // Weak input should not cause spike
        for _ in 0..10 {
            assert!(neuron.step(0.01, 1.0).is_none());
        }
    }

    #[test]
    fn test_lif_spike() {
        let mut neuron = LIFNeuron::default();

        // Strong constant input should eventually cause spike
        let mut spike_time = None;
        for _ in 0..100 {
            if let Some(spike) = neuron.step(0.1, 1.0) {
                spike_time = Some(spike.time);
                break;
            }
        }

        assert!(spike_time.is_some(), "Neuron should have spiked");
        assert_eq!(neuron.state().spike_count, 1);
    }

    #[test]
    fn test_lif_refractory() {
        let config = LIFConfig {
            refractory_period: 5.0,
            ..Default::default()
        };
        let mut neuron = LIFNeuron::new(0, config);

        // Force spike
        neuron.step(10.0, 1.0);

        // Should be refractory
        assert!(neuron.is_refractory());

        // Strong input during refractory should not cause spike
        for _ in 0..4 {
            assert!(neuron.step(10.0, 1.0).is_none());
        }
    }

    #[test]
    fn test_lif_reset() {
        let mut neuron = LIFNeuron::default();

        // Generate some spikes
        for _ in 0..100 {
            neuron.step(0.1, 1.0);
        }

        // Reset
        neuron.reset();

        assert_eq!(neuron.membrane_potential(), 0.0);
        assert_eq!(neuron.state().spike_count, 0);
        assert!(!neuron.is_refractory());
    }
}
