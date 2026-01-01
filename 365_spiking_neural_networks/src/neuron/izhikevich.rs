//! Izhikevich Neuron Model
//!
//! A more biologically realistic model that can reproduce various
//! firing patterns observed in real neurons.
//!
//! Dynamics:
//! dv/dt = 0.04v² + 5v + 140 - u + I
//! du/dt = a(bv - u)
//!
//! When v >= 30: v = c, u = u + d

use super::{Neuron, Spike};

/// Izhikevich neuron model parameters
#[derive(Debug, Clone, Copy)]
pub struct IzhikevichParams {
    /// Time scale of recovery variable
    pub a: f64,
    /// Sensitivity of recovery variable to membrane potential
    pub b: f64,
    /// After-spike reset value of membrane potential
    pub c: f64,
    /// After-spike reset increment of recovery variable
    pub d: f64,
}

impl IzhikevichParams {
    /// Regular spiking (RS) neuron - most common excitatory
    pub fn regular_spiking() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        }
    }

    /// Fast spiking (FS) neuron - typical inhibitory interneuron
    pub fn fast_spiking() -> Self {
        Self {
            a: 0.1,
            b: 0.2,
            c: -65.0,
            d: 2.0,
        }
    }

    /// Intrinsically bursting (IB) neuron
    pub fn intrinsically_bursting() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -55.0,
            d: 4.0,
        }
    }

    /// Chattering (CH) neuron
    pub fn chattering() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -50.0,
            d: 2.0,
        }
    }

    /// Low-threshold spiking (LTS) neuron
    pub fn low_threshold_spiking() -> Self {
        Self {
            a: 0.02,
            b: 0.25,
            c: -65.0,
            d: 2.0,
        }
    }

    /// Thalamo-cortical (TC) neuron
    pub fn thalamocortical() -> Self {
        Self {
            a: 0.02,
            b: 0.25,
            c: -65.0,
            d: 0.05,
        }
    }

    /// Resonator (RZ) neuron
    pub fn resonator() -> Self {
        Self {
            a: 0.1,
            b: 0.26,
            c: -65.0,
            d: 2.0,
        }
    }
}

impl Default for IzhikevichParams {
    fn default() -> Self {
        Self::regular_spiking()
    }
}

/// Izhikevich neuron implementation
#[derive(Debug, Clone)]
pub struct IzhikevichNeuron {
    /// Membrane potential
    v: f64,
    /// Recovery variable
    u: f64,
    /// Parameters
    params: IzhikevichParams,
    /// Current simulation time
    current_time: f64,
    /// Last spike time
    last_spike: Option<f64>,
    /// Input buffer from incoming spikes
    input_buffer: f64,
}

impl IzhikevichNeuron {
    /// Create a new Izhikevich neuron with default (regular spiking) parameters
    pub fn new() -> Self {
        Self::with_params(IzhikevichParams::default())
    }

    /// Create a new Izhikevich neuron with custom parameters
    pub fn with_params(params: IzhikevichParams) -> Self {
        let v = params.c;
        let u = params.b * v;
        Self {
            v,
            u,
            params,
            current_time: 0.0,
            last_spike: None,
            input_buffer: 0.0,
        }
    }

    /// Create a regular spiking neuron
    pub fn regular_spiking() -> Self {
        Self::with_params(IzhikevichParams::regular_spiking())
    }

    /// Create a fast spiking neuron
    pub fn fast_spiking() -> Self {
        Self::with_params(IzhikevichParams::fast_spiking())
    }

    /// Create an intrinsically bursting neuron
    pub fn intrinsically_bursting() -> Self {
        Self::with_params(IzhikevichParams::intrinsically_bursting())
    }

    /// Create a chattering neuron
    pub fn chattering() -> Self {
        Self::with_params(IzhikevichParams::chattering())
    }

    /// Get the recovery variable
    pub fn recovery_variable(&self) -> f64 {
        self.u
    }

    /// Get the parameters
    pub fn params(&self) -> &IzhikevichParams {
        &self.params
    }
}

impl Default for IzhikevichNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl Neuron for IzhikevichNeuron {
    fn reset(&mut self) {
        self.v = self.params.c;
        self.u = self.params.b * self.v;
        self.current_time = 0.0;
        self.last_spike = None;
        self.input_buffer = 0.0;
    }

    fn step(&mut self, input_current: f64, dt: f64) -> bool {
        self.current_time += dt;

        let total_input = input_current + self.input_buffer;
        self.input_buffer = 0.0;

        // Izhikevich dynamics (using 0.5ms steps for stability)
        let steps = (dt / 0.5).ceil() as usize;
        let sub_dt = dt / steps as f64;

        for _ in 0..steps {
            // dv/dt = 0.04v² + 5v + 140 - u + I
            let dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + total_input) * sub_dt;
            // du/dt = a(bv - u)
            let du = self.params.a * (self.params.b * self.v - self.u) * sub_dt;

            self.v += dv;
            self.u += du;

            // Spike condition
            if self.v >= 30.0 {
                self.v = self.params.c;
                self.u += self.params.d;
                self.last_spike = Some(self.current_time);
                return true;
            }
        }

        false
    }

    fn membrane_potential(&self) -> f64 {
        self.v
    }

    fn is_refractory(&self) -> bool {
        // Izhikevich model has implicit refractoriness through the recovery variable
        false
    }

    fn last_spike_time(&self) -> Option<f64> {
        self.last_spike
    }

    fn receive_spike(&mut self, spike: &Spike, weight: f64) {
        self.input_buffer += weight * spike.sign();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_izhikevich_creation() {
        let neuron = IzhikevichNeuron::new();
        assert!(neuron.last_spike_time().is_none());
    }

    #[test]
    fn test_regular_spiking() {
        let mut neuron = IzhikevichNeuron::regular_spiking();
        let dt = 1.0;
        let mut spike_count = 0;

        for _ in 0..1000 {
            if neuron.step(10.0, dt) {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0, "Regular spiking neuron should spike");
    }

    #[test]
    fn test_fast_spiking() {
        let mut fast = IzhikevichNeuron::fast_spiking();
        let mut regular = IzhikevichNeuron::regular_spiking();
        let dt = 1.0;

        let mut fast_spikes = 0;
        let mut regular_spikes = 0;

        for _ in 0..500 {
            if fast.step(14.0, dt) {
                fast_spikes += 1;
            }
            if regular.step(14.0, dt) {
                regular_spikes += 1;
            }
        }

        assert!(fast_spikes >= regular_spikes,
            "Fast spiking should produce at least as many spikes");
    }

    #[test]
    fn test_reset() {
        let mut neuron = IzhikevichNeuron::new();

        // Run for a while
        for _ in 0..100 {
            neuron.step(10.0, 1.0);
        }

        neuron.reset();

        assert_eq!(neuron.v, neuron.params().c);
        assert!(neuron.last_spike_time().is_none());
    }
}
