//! Izhikevich Neuron Model
//!
//! The Izhikevich model provides a balance between biological plausibility
//! and computational efficiency. It can reproduce various firing patterns
//! observed in biological neurons.
//!
//! ## Dynamics
//!
//! ```text
//! dv/dt = 0.04v² + 5v + 140 - u + I
//! du/dt = a(bv - u)
//!
//! if v >= 30mV:
//!     v = c
//!     u = u + d
//! ```
//!
//! ## Neuron Types
//!
//! | Type | a | b | c | d | Behavior |
//! |------|---|---|---|---|----------|
//! | Regular Spiking | 0.02 | 0.2 | -65 | 8 | Most common excitatory |
//! | Fast Spiking | 0.1 | 0.2 | -65 | 2 | Fast inhibitory |
//! | Bursting | 0.02 | 0.2 | -50 | 2 | Burst firing |
//! | Chattering | 0.02 | 0.2 | -50 | 2 | Rhythmic bursts |

use super::{Neuron, NeuronState, SpikeEvent};

/// Predefined neuron types with their parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronType {
    /// Regular spiking (most common excitatory neurons)
    RegularSpiking,
    /// Fast spiking (inhibitory interneurons)
    FastSpiking,
    /// Intrinsically bursting
    Bursting,
    /// Chattering (rhythmic bursting)
    Chattering,
    /// Low-threshold spiking
    LowThreshold,
    /// Custom parameters
    Custom,
}

impl NeuronType {
    /// Get the (a, b, c, d) parameters for this neuron type
    pub fn parameters(&self) -> (f64, f64, f64, f64) {
        match self {
            NeuronType::RegularSpiking => (0.02, 0.2, -65.0, 8.0),
            NeuronType::FastSpiking => (0.1, 0.2, -65.0, 2.0),
            NeuronType::Bursting => (0.02, 0.2, -50.0, 2.0),
            NeuronType::Chattering => (0.02, 0.2, -50.0, 2.0),
            NeuronType::LowThreshold => (0.02, 0.25, -65.0, 2.0),
            NeuronType::Custom => (0.02, 0.2, -65.0, 8.0),
        }
    }
}

/// Configuration for Izhikevich neuron
#[derive(Debug, Clone, Copy)]
pub struct IzhikevichConfig {
    /// Time scale of recovery variable
    pub a: f64,
    /// Sensitivity of recovery variable to membrane potential
    pub b: f64,
    /// After-spike reset value of membrane potential
    pub c: f64,
    /// After-spike reset of recovery variable
    pub d: f64,
    /// Spike threshold (typically 30mV)
    pub threshold: f64,
    /// Neuron type
    pub neuron_type: NeuronType,
}

impl Default for IzhikevichConfig {
    fn default() -> Self {
        Self::from_type(NeuronType::RegularSpiking)
    }
}

impl IzhikevichConfig {
    /// Create configuration from predefined neuron type
    pub fn from_type(neuron_type: NeuronType) -> Self {
        let (a, b, c, d) = neuron_type.parameters();
        Self {
            a,
            b,
            c,
            d,
            threshold: 30.0,
            neuron_type,
        }
    }

    /// Create custom configuration
    pub fn custom(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            a,
            b,
            c,
            d,
            threshold: 30.0,
            neuron_type: NeuronType::Custom,
        }
    }
}

/// Izhikevich neuron model
#[derive(Debug, Clone)]
pub struct IzhikevichNeuron {
    /// Unique neuron ID
    id: usize,
    /// Configuration
    config: IzhikevichConfig,
    /// Membrane potential (v)
    v: f64,
    /// Recovery variable (u)
    u: f64,
    /// Current simulation time
    current_time: f64,
    /// Time of last spike
    last_spike_time: f64,
    /// Total spike count
    spike_count: u64,
    /// Accumulated input current
    input_buffer: f64,
    /// Recent spike times
    recent_spikes: Vec<f64>,
}

impl IzhikevichNeuron {
    /// Create a new Izhikevich neuron
    pub fn new(id: usize, config: IzhikevichConfig) -> Self {
        let v = config.c;
        let u = config.b * v;
        Self {
            id,
            config,
            v,
            u,
            current_time: 0.0,
            last_spike_time: -f64::INFINITY,
            spike_count: 0,
            input_buffer: 0.0,
            recent_spikes: Vec::new(),
        }
    }

    /// Create from predefined neuron type
    pub fn from_type(id: usize, neuron_type: NeuronType) -> Self {
        Self::new(id, IzhikevichConfig::from_type(neuron_type))
    }

    /// Get the configuration
    pub fn config(&self) -> &IzhikevichConfig {
        &self.config
    }

    /// Get the recovery variable
    pub fn recovery(&self) -> f64 {
        self.u
    }

    /// Clean up old spike times
    fn cleanup_spike_history(&mut self, window: f64) {
        let cutoff = self.current_time - window;
        self.recent_spikes.retain(|&t| t > cutoff);
    }
}

impl Default for IzhikevichNeuron {
    fn default() -> Self {
        Self::new(0, IzhikevichConfig::default())
    }
}

impl Neuron for IzhikevichNeuron {
    fn id(&self) -> usize {
        self.id
    }

    fn state(&self) -> NeuronState {
        NeuronState {
            membrane_potential: self.v,
            recovery: self.u,
            time_since_spike: self.current_time - self.last_spike_time,
            refractory: false,  // Izhikevich doesn't have explicit refractory
            spike_count: self.spike_count,
        }
    }

    fn reset(&mut self) {
        self.v = self.config.c;
        self.u = self.config.b * self.v;
        self.current_time = 0.0;
        self.last_spike_time = -f64::INFINITY;
        self.spike_count = 0;
        self.input_buffer = 0.0;
        self.recent_spikes.clear();
    }

    fn step(&mut self, input_current: f64, dt: f64) -> Option<SpikeEvent> {
        self.current_time += dt;

        // Total input
        let input = input_current + self.input_buffer;
        self.input_buffer = 0.0;

        // Izhikevich dynamics (using Euler method with smaller steps for stability)
        let steps = 4;
        let dt_small = dt / steps as f64;

        for _ in 0..steps {
            // dv/dt = 0.04v² + 5v + 140 - u + I
            let dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + input) * dt_small;
            // du/dt = a(bv - u)
            let du = self.config.a * (self.config.b * self.v - self.u) * dt_small;

            self.v += dv;
            self.u += du;
        }

        // Check for spike
        if self.v >= self.config.threshold {
            self.v = self.config.c;
            self.u += self.config.d;
            self.last_spike_time = self.current_time;
            self.spike_count += 1;
            self.recent_spikes.push(self.current_time);
            self.cleanup_spike_history(1000.0);

            return Some(SpikeEvent::new(self.id, self.current_time));
        }

        None
    }

    fn is_refractory(&self) -> bool {
        // Izhikevich model doesn't have explicit refractory period
        // but the recovery variable provides similar behavior
        false
    }

    fn membrane_potential(&self) -> f64 {
        self.v
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
        (count as f64) / (time_window / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_izhikevich_regular_spiking() {
        let mut neuron = IzhikevichNeuron::from_type(0, NeuronType::RegularSpiking);

        // Apply strong input
        let mut spike_count = 0;
        for _ in 0..1000 {
            if neuron.step(15.0, 0.5).is_some() {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0, "Regular spiking neuron should fire");
    }

    #[test]
    fn test_izhikevich_fast_spiking() {
        let mut neuron = IzhikevichNeuron::from_type(0, NeuronType::FastSpiking);

        let mut spike_count = 0;
        for _ in 0..1000 {
            if neuron.step(20.0, 0.5).is_some() {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0, "Fast spiking neuron should fire");
    }

    #[test]
    fn test_izhikevich_bursting() {
        let mut neuron = IzhikevichNeuron::from_type(0, NeuronType::Bursting);

        let mut spike_count = 0;
        for _ in 0..1000 {
            if neuron.step(10.0, 0.5).is_some() {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0, "Bursting neuron should fire");
    }

    #[test]
    fn test_neuron_types() {
        let types = vec![
            NeuronType::RegularSpiking,
            NeuronType::FastSpiking,
            NeuronType::Bursting,
            NeuronType::Chattering,
            NeuronType::LowThreshold,
        ];

        for neuron_type in types {
            let (a, b, c, d) = neuron_type.parameters();
            assert!(a > 0.0);
            assert!(b > 0.0);
            assert!(c < 0.0);
            assert!(d > 0.0);
        }
    }
}
