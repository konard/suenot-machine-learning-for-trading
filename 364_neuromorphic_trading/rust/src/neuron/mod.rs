//! Neuron Models Module
//!
//! This module provides implementations of various spiking neuron models
//! used in neuromorphic computing.
//!
//! ## Available Models
//!
//! - **LIF (Leaky Integrate-and-Fire)**: Simple and computationally efficient
//! - **Izhikevich**: Biologically plausible with rich dynamics
//!
//! ## Example
//!
//! ```rust
//! use neuromorphic_trading::neuron::{Neuron, lif::LIFNeuron};
//!
//! let mut neuron = LIFNeuron::default();
//! let input_current = 1.5;
//!
//! // Simulate for 100ms
//! for _ in 0..100 {
//!     if let Some(spike) = neuron.step(input_current, 1.0) {
//!         println!("Spike at t={}", spike.time);
//!     }
//! }
//! ```

pub mod lif;
pub mod izhikevich;
pub mod synapse;

use chrono::{DateTime, Utc};

/// Spike event representing a neuron firing
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeEvent {
    /// Neuron ID that fired
    pub neuron_id: usize,
    /// Time of the spike
    pub time: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl SpikeEvent {
    /// Create a new spike event
    pub fn new(neuron_id: usize, time: f64) -> Self {
        Self {
            neuron_id,
            time,
            timestamp: Utc::now(),
        }
    }
}

/// Neuron state containing current values
#[derive(Debug, Clone, Copy, Default)]
pub struct NeuronState {
    /// Membrane potential
    pub membrane_potential: f64,
    /// Recovery variable (for Izhikevich)
    pub recovery: f64,
    /// Time since last spike
    pub time_since_spike: f64,
    /// Whether neuron is in refractory period
    pub refractory: bool,
    /// Cumulative spike count
    pub spike_count: u64,
}

/// Common trait for all neuron models
pub trait Neuron: Send + Sync {
    /// Get the neuron's unique ID
    fn id(&self) -> usize;

    /// Get the current state
    fn state(&self) -> NeuronState;

    /// Reset the neuron to initial state
    fn reset(&mut self);

    /// Process one timestep with given input current
    /// Returns Some(SpikeEvent) if the neuron fires
    fn step(&mut self, input_current: f64, dt: f64) -> Option<SpikeEvent>;

    /// Check if the neuron is currently in refractory period
    fn is_refractory(&self) -> bool;

    /// Get the membrane potential
    fn membrane_potential(&self) -> f64;

    /// Get the spike threshold
    fn threshold(&self) -> f64;

    /// Apply external input to the neuron
    fn apply_input(&mut self, input: f64);

    /// Get the neuron's current spike rate (spikes per second)
    fn spike_rate(&self, time_window: f64) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use lif::LIFNeuron;

    #[test]
    fn test_spike_event_creation() {
        let spike = SpikeEvent::new(42, 10.5);
        assert_eq!(spike.neuron_id, 42);
        assert_eq!(spike.time, 10.5);
    }

    #[test]
    fn test_neuron_fires() {
        let mut neuron = LIFNeuron::default();

        // Apply strong input until spike
        let mut spiked = false;
        for _ in 0..100 {
            if neuron.step(2.0, 1.0).is_some() {
                spiked = true;
                break;
            }
        }
        assert!(spiked, "Neuron should have fired with strong input");
    }
}
