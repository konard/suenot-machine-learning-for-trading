//! Leaky Integrate-and-Fire (LIF) Neuron Model
//!
//! The LIF model is the most common spiking neuron model, described by:
//!
//! τ_m * dV/dt = -(V - V_rest) + R * I(t)
//!
//! When V >= V_thresh: emit spike, V = V_reset

use super::{Neuron, NeuronParams, Spike};

/// Leaky Integrate-and-Fire neuron
#[derive(Debug, Clone)]
pub struct LIFNeuron {
    /// Current membrane potential (mV)
    membrane_potential: f64,
    /// Neuron parameters
    params: NeuronParams,
    /// Current simulation time (ms)
    current_time: f64,
    /// Time of last spike (ms)
    last_spike: Option<f64>,
    /// Accumulated input current from spikes
    input_buffer: f64,
}

impl LIFNeuron {
    /// Create a new LIF neuron with default parameters
    pub fn new() -> Self {
        Self::with_params(NeuronParams::default())
    }

    /// Create a new LIF neuron with custom parameters
    pub fn with_params(params: NeuronParams) -> Self {
        Self {
            membrane_potential: params.v_rest,
            params,
            current_time: 0.0,
            last_spike: None,
            input_buffer: 0.0,
        }
    }

    /// Create a fast-spiking neuron (shorter time constant)
    pub fn fast_spiking() -> Self {
        Self::with_params(NeuronParams {
            tau_m: 5.0,
            t_refrac: 1.0,
            ..Default::default()
        })
    }

    /// Create a slow-spiking neuron (longer time constant)
    pub fn slow_spiking() -> Self {
        Self::with_params(NeuronParams {
            tau_m: 20.0,
            t_refrac: 3.0,
            ..Default::default()
        })
    }

    /// Get the neuron parameters
    pub fn params(&self) -> &NeuronParams {
        &self.params
    }

    /// Set the firing threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.params.v_thresh = threshold;
    }

    /// Get the current time
    pub fn current_time(&self) -> f64 {
        self.current_time
    }
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl Neuron for LIFNeuron {
    fn reset(&mut self) {
        self.membrane_potential = self.params.v_rest;
        self.current_time = 0.0;
        self.last_spike = None;
        self.input_buffer = 0.0;
    }

    fn step(&mut self, input_current: f64, dt: f64) -> bool {
        self.current_time += dt;

        // Check if in refractory period
        if self.is_refractory() {
            self.input_buffer = 0.0;
            return false;
        }

        // Total input current (external + spike-induced)
        let total_current = input_current + self.input_buffer;
        self.input_buffer = 0.0;

        // LIF dynamics: τ_m * dV/dt = -(V - V_rest) + R * I
        let dv = (-( self.membrane_potential - self.params.v_rest)
            + self.params.resistance * total_current)
            * dt
            / self.params.tau_m;

        self.membrane_potential += dv;

        // Check for spike
        if self.membrane_potential >= self.params.v_thresh {
            self.membrane_potential = self.params.v_reset;
            self.last_spike = Some(self.current_time);
            true
        } else {
            false
        }
    }

    fn membrane_potential(&self) -> f64 {
        self.membrane_potential
    }

    fn is_refractory(&self) -> bool {
        if let Some(last_spike) = self.last_spike {
            self.current_time - last_spike < self.params.t_refrac
        } else {
            false
        }
    }

    fn last_spike_time(&self) -> Option<f64> {
        self.last_spike
    }

    fn receive_spike(&mut self, spike: &Spike, weight: f64) {
        // Add weighted spike to input buffer
        self.input_buffer += weight * spike.sign();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_creation() {
        let neuron = LIFNeuron::new();
        assert_eq!(neuron.membrane_potential(), neuron.params().v_rest);
        assert!(!neuron.is_refractory());
        assert!(neuron.last_spike_time().is_none());
    }

    #[test]
    fn test_lif_spike() {
        let mut neuron = LIFNeuron::new();

        // Apply strong current until spike
        let dt = 0.1;
        let mut spiked = false;

        for _ in 0..1000 {
            if neuron.step(50.0, dt) {
                spiked = true;
                break;
            }
        }

        assert!(spiked, "Neuron should spike with strong input");
        assert!(neuron.last_spike_time().is_some());
    }

    #[test]
    fn test_lif_refractory() {
        let mut neuron = LIFNeuron::new();
        let dt = 0.1;

        // Force spike with very strong current
        while !neuron.step(100.0, dt) {}

        // Should be in refractory period
        assert!(neuron.is_refractory());

        // Shouldn't spike during refractory period
        assert!(!neuron.step(100.0, dt));
    }

    #[test]
    fn test_lif_reset() {
        let mut neuron = LIFNeuron::new();

        // Modify state
        neuron.step(50.0, 1.0);

        // Reset
        neuron.reset();

        assert_eq!(neuron.membrane_potential(), neuron.params().v_rest);
        assert_eq!(neuron.current_time(), 0.0);
        assert!(neuron.last_spike_time().is_none());
    }

    #[test]
    fn test_lif_receive_spike() {
        let mut neuron = LIFNeuron::new();

        let spike = Spike::positive(0.0);
        neuron.receive_spike(&spike, 10.0);

        // The input should be applied in the next step
        let initial_v = neuron.membrane_potential();
        neuron.step(0.0, 0.1);

        assert!(neuron.membrane_potential() > initial_v);
    }
}
