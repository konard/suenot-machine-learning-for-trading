//! Neural Layer Module
//!
//! A layer is a collection of neurons that can be processed together.

use crate::neuron::{Neuron, NeuronState, SpikeEvent, lif::{LIFNeuron, LIFConfig}};

/// Layer configuration
#[derive(Debug, Clone, Copy)]
pub struct LayerConfig {
    /// Number of neurons in the layer
    pub size: usize,
    /// Configuration for neurons
    pub neuron_config: LIFConfig,
    /// Layer ID
    pub layer_id: usize,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            size: 64,
            neuron_config: LIFConfig::default(),
            layer_id: 0,
        }
    }
}

/// A layer of spiking neurons
#[derive(Debug)]
pub struct Layer {
    /// Layer configuration
    config: LayerConfig,
    /// Neurons in the layer
    neurons: Vec<LIFNeuron>,
    /// Current simulation time
    current_time: f64,
    /// Accumulated inputs for each neuron
    input_buffer: Vec<f64>,
}

impl Layer {
    /// Create a new layer
    pub fn new(config: LayerConfig) -> Self {
        let neurons = (0..config.size)
            .map(|i| LIFNeuron::new(i, config.neuron_config))
            .collect();

        let input_buffer = vec![0.0; config.size];

        Self {
            config,
            neurons,
            current_time: 0.0,
            input_buffer,
        }
    }

    /// Get layer size
    pub fn size(&self) -> usize {
        self.config.size
    }

    /// Get layer ID
    pub fn layer_id(&self) -> usize {
        self.config.layer_id
    }

    /// Apply a spike to a neuron
    pub fn apply_spike(&mut self, neuron_id: usize, weight: f64) {
        if neuron_id < self.input_buffer.len() {
            self.input_buffer[neuron_id] += weight;
        }
    }

    /// Process one timestep, returning spikes
    pub fn step(&mut self, dt: f64) -> Vec<SpikeEvent> {
        self.current_time += dt;
        let mut spikes = Vec::new();

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let input = self.input_buffer[i];
            self.input_buffer[i] = 0.0;

            if let Some(mut spike) = neuron.step(input, dt) {
                // Adjust neuron ID to global space
                spike.neuron_id = self.config.layer_id * 1000 + i;
                spikes.push(spike);
            }
        }

        spikes
    }

    /// Get states of all neurons
    pub fn neuron_states(&self) -> Vec<NeuronState> {
        self.neurons.iter().map(|n| n.state()).collect()
    }

    /// Reset all neurons
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        for input in &mut self.input_buffer {
            *input = 0.0;
        }
        self.current_time = 0.0;
    }

    /// Get a specific neuron's state
    pub fn get_neuron_state(&self, index: usize) -> Option<NeuronState> {
        self.neurons.get(index).map(|n| n.state())
    }

    /// Get average membrane potential
    pub fn avg_membrane_potential(&self) -> f64 {
        let sum: f64 = self.neurons.iter().map(|n| n.membrane_potential()).sum();
        sum / self.neurons.len() as f64
    }

    /// Get spike counts for all neurons
    pub fn spike_counts(&self) -> Vec<u64> {
        self.neurons.iter().map(|n| n.state().spike_count).collect()
    }

    /// Apply lateral inhibition (winner-take-all)
    pub fn apply_lateral_inhibition(&mut self, strength: f64) {
        // Find the neuron with highest membrane potential
        let max_potential = self.neurons
            .iter()
            .map(|n| n.membrane_potential())
            .fold(f64::NEG_INFINITY, f64::max);

        // Inhibit all other neurons proportionally
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let potential = neuron.membrane_potential();
            if potential < max_potential {
                let inhibition = strength * (max_potential - potential);
                self.input_buffer[i] -= inhibition;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let config = LayerConfig {
            size: 10,
            ..Default::default()
        };
        let layer = Layer::new(config);

        assert_eq!(layer.size(), 10);
    }

    #[test]
    fn test_layer_step() {
        let mut layer = Layer::new(LayerConfig {
            size: 5,
            ..Default::default()
        });

        // Apply strong input to first neuron
        layer.apply_spike(0, 2.0);

        // Step multiple times
        let mut total_spikes = 0;
        for _ in 0..100 {
            total_spikes += layer.step(1.0).len();
            layer.apply_spike(0, 0.1);
        }

        assert!(total_spikes > 0, "Layer should produce spikes with input");
    }

    #[test]
    fn test_layer_reset() {
        let mut layer = Layer::new(LayerConfig::default());

        // Generate some activity
        for _ in 0..50 {
            layer.apply_spike(0, 0.1);
            layer.step(1.0);
        }

        // Reset
        layer.reset();

        // All neurons should be at rest
        for state in layer.neuron_states() {
            assert_eq!(state.membrane_potential, 0.0);
        }
    }
}
