//! Neuron models for Spiking Neural Networks
//!
//! This module provides various neuron models:
//! - Leaky Integrate-and-Fire (LIF)
//! - Izhikevich model

mod lif;
mod izhikevich;

pub use lif::LIFNeuron;
pub use izhikevich::IzhikevichNeuron;

/// Represents a spike event
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spike {
    /// Time of the spike in milliseconds
    pub time: f64,
    /// Polarity of the spike (positive for excitatory, negative for inhibitory)
    pub polarity: SpikePolarity,
    /// Source neuron index (if applicable)
    pub source: Option<usize>,
}

/// Spike polarity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpikePolarity {
    /// Excitatory (positive) spike
    Positive,
    /// Inhibitory (negative) spike
    Negative,
}

impl Spike {
    /// Create a new positive spike
    pub fn positive(time: f64) -> Self {
        Self {
            time,
            polarity: SpikePolarity::Positive,
            source: None,
        }
    }

    /// Create a new negative spike
    pub fn negative(time: f64) -> Self {
        Self {
            time,
            polarity: SpikePolarity::Negative,
            source: None,
        }
    }

    /// Create a spike with source information
    pub fn with_source(mut self, source: usize) -> Self {
        self.source = Some(source);
        self
    }

    /// Get the sign of the spike (+1 or -1)
    pub fn sign(&self) -> f64 {
        match self.polarity {
            SpikePolarity::Positive => 1.0,
            SpikePolarity::Negative => -1.0,
        }
    }
}

/// Common trait for all neuron models
pub trait Neuron: Send + Sync {
    /// Reset the neuron to its initial state
    fn reset(&mut self);

    /// Update the neuron state and return whether it spiked
    fn step(&mut self, input_current: f64, dt: f64) -> bool;

    /// Get the current membrane potential
    fn membrane_potential(&self) -> f64;

    /// Check if the neuron is in refractory period
    fn is_refractory(&self) -> bool;

    /// Get the last spike time (if any)
    fn last_spike_time(&self) -> Option<f64>;

    /// Receive a spike from another neuron
    fn receive_spike(&mut self, spike: &Spike, weight: f64);
}

/// Neuron parameters for configuration
#[derive(Debug, Clone)]
pub struct NeuronParams {
    /// Resting membrane potential (mV)
    pub v_rest: f64,
    /// Firing threshold (mV)
    pub v_thresh: f64,
    /// Reset potential after spike (mV)
    pub v_reset: f64,
    /// Membrane time constant (ms)
    pub tau_m: f64,
    /// Membrane resistance (MOhm)
    pub resistance: f64,
    /// Refractory period (ms)
    pub t_refrac: f64,
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            v_rest: -70.0,
            v_thresh: -55.0,
            v_reset: -75.0,
            tau_m: 10.0,
            resistance: 10.0,
            t_refrac: 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_creation() {
        let spike = Spike::positive(10.0);
        assert_eq!(spike.time, 10.0);
        assert_eq!(spike.polarity, SpikePolarity::Positive);
        assert_eq!(spike.sign(), 1.0);

        let spike = Spike::negative(20.0).with_source(5);
        assert_eq!(spike.time, 20.0);
        assert_eq!(spike.polarity, SpikePolarity::Negative);
        assert_eq!(spike.source, Some(5));
        assert_eq!(spike.sign(), -1.0);
    }
}
