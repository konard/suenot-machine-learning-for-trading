//! Layer implementation for Spiking Neural Networks

use crate::neuron::{LIFNeuron, Neuron, Spike, SpikePolarity};
use rand::Rng;

/// Layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Input,
    Hidden,
    Output,
}

/// Configuration for a layer
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Number of neurons in the layer
    pub size: usize,
    /// Layer type
    pub layer_type: LayerType,
    /// Membrane time constant
    pub tau_m: f64,
    /// Firing threshold
    pub threshold: f64,
}

impl LayerConfig {
    /// Create a new input layer config
    pub fn input(size: usize) -> Self {
        Self {
            size,
            layer_type: LayerType::Input,
            tau_m: 10.0,
            threshold: -55.0,
        }
    }

    /// Create a new hidden layer config
    pub fn hidden(size: usize) -> Self {
        Self {
            size,
            layer_type: LayerType::Hidden,
            tau_m: 10.0,
            threshold: -55.0,
        }
    }

    /// Create a new output layer config
    pub fn output(size: usize) -> Self {
        Self {
            size,
            layer_type: LayerType::Output,
            tau_m: 20.0,  // Slower for integration
            threshold: -55.0,
        }
    }

    /// Set membrane time constant
    pub fn with_tau(mut self, tau_m: f64) -> Self {
        self.tau_m = tau_m;
        self
    }

    /// Set firing threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

/// A layer of spiking neurons
#[derive(Debug, Clone)]
pub struct SNNLayer {
    /// Neurons in the layer
    neurons: Vec<LIFNeuron>,
    /// Layer configuration
    config: LayerConfig,
    /// Weights to next layer (if any)
    weights: Option<Vec<Vec<f64>>>,
    /// Spikes generated in the last timestep
    last_spikes: Vec<Option<Spike>>,
    /// Spike history for learning
    spike_history: Vec<Vec<f64>>,
}

impl SNNLayer {
    /// Create a new layer from configuration
    pub fn new(config: LayerConfig) -> Self {
        let neurons = (0..config.size)
            .map(|_| {
                let mut neuron = LIFNeuron::new();
                neuron.set_threshold(config.threshold);
                neuron
            })
            .collect();

        Self {
            neurons,
            config,
            weights: None,
            last_spikes: vec![None; config.size],
            spike_history: vec![Vec::new(); config.size],
        }
    }

    /// Initialize random weights to next layer
    pub fn init_weights(&mut self, next_layer_size: usize) {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (self.config.size as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..self.config.size)
            .map(|_| {
                (0..next_layer_size)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        self.weights = Some(weights);
    }

    /// Get weights (if set)
    pub fn weights(&self) -> Option<&Vec<Vec<f64>>> {
        self.weights.as_ref()
    }

    /// Get mutable weights
    pub fn weights_mut(&mut self) -> Option<&mut Vec<Vec<f64>>> {
        self.weights.as_mut()
    }

    /// Set weights manually
    pub fn set_weights(&mut self, weights: Vec<Vec<f64>>) {
        self.weights = Some(weights);
    }

    /// Get layer size
    pub fn size(&self) -> usize {
        self.config.size
    }

    /// Get layer type
    pub fn layer_type(&self) -> LayerType {
        self.config.layer_type
    }

    /// Reset all neurons
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        self.last_spikes = vec![None; self.config.size];
        for history in &mut self.spike_history {
            history.clear();
        }
    }

    /// Process input and return output spikes
    pub fn forward(&mut self, inputs: &[f64], dt: f64, current_time: f64) -> Vec<bool> {
        assert_eq!(inputs.len(), self.config.size,
            "Input size mismatch: expected {}, got {}", self.config.size, inputs.len());

        let mut spikes = Vec::with_capacity(self.config.size);

        for (i, (neuron, &input)) in self.neurons.iter_mut().zip(inputs.iter()).enumerate() {
            let spiked = neuron.step(input, dt);
            spikes.push(spiked);

            if spiked {
                self.last_spikes[i] = Some(Spike {
                    time: current_time,
                    polarity: SpikePolarity::Positive,
                    source: Some(i),
                });
                self.spike_history[i].push(current_time);
            } else {
                self.last_spikes[i] = None;
            }
        }

        spikes
    }

    /// Receive spikes from previous layer
    pub fn receive_spikes(&mut self, spikes: &[bool], source_weights: &[Vec<f64>]) {
        for (src_idx, &spiked) in spikes.iter().enumerate() {
            if spiked {
                let spike = Spike::positive(0.0).with_source(src_idx);
                for (dst_idx, neuron) in self.neurons.iter_mut().enumerate() {
                    let weight = source_weights[src_idx][dst_idx];
                    neuron.receive_spike(&spike, weight);
                }
            }
        }
    }

    /// Get last spikes
    pub fn last_spikes(&self) -> &[Option<Spike>] {
        &self.last_spikes
    }

    /// Get spike history for a specific neuron
    pub fn spike_history(&self, neuron_idx: usize) -> &[f64] {
        &self.spike_history[neuron_idx]
    }

    /// Get membrane potentials
    pub fn membrane_potentials(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.membrane_potential()).collect()
    }

    /// Count spikes in last window
    pub fn spike_count(&self, window_start: f64) -> Vec<usize> {
        self.spike_history
            .iter()
            .map(|history| {
                history.iter().filter(|&&t| t >= window_start).count()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let config = LayerConfig::hidden(10);
        let layer = SNNLayer::new(config);
        assert_eq!(layer.size(), 10);
        assert_eq!(layer.layer_type(), LayerType::Hidden);
    }

    #[test]
    fn test_layer_forward() {
        let config = LayerConfig::input(5);
        let mut layer = SNNLayer::new(config);

        let inputs = vec![50.0, 50.0, 0.0, 0.0, 50.0];
        let dt = 0.1;

        // Run for a while and check for spikes
        let mut total_spikes = 0;
        for t in 0..100 {
            let spikes = layer.forward(&inputs, dt, t as f64 * dt);
            total_spikes += spikes.iter().filter(|&&s| s).count();
        }

        assert!(total_spikes > 0, "Should have some spikes with input");
    }

    #[test]
    fn test_weight_initialization() {
        let config = LayerConfig::hidden(10);
        let mut layer = SNNLayer::new(config);

        layer.init_weights(5);

        let weights = layer.weights().unwrap();
        assert_eq!(weights.len(), 10);
        assert_eq!(weights[0].len(), 5);
    }
}
