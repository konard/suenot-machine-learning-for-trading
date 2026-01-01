//! Neural Network Module
//!
//! This module provides the infrastructure for building and running
//! spiking neural networks.
//!
//! ## Components
//!
//! - **Layer**: Groups of neurons with configurable connectivity
//! - **Topology**: Network structure and connectivity patterns
//! - **Learning**: STDP and other learning rules

pub mod layer;
pub mod topology;
pub mod learning;

use crate::neuron::{Neuron, SpikeEvent, NeuronState, lif::{LIFNeuron, LIFConfig}, synapse::Synapse};
use layer::{Layer, LayerConfig};
use learning::{STDPConfig, LearningRule};

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Number of input neurons
    pub input_size: usize,
    /// Sizes of hidden layers
    pub hidden_sizes: Vec<usize>,
    /// Number of output neurons
    pub output_size: usize,
    /// Membrane time constant
    pub tau_m: f64,
    /// Spike threshold
    pub threshold: f64,
    /// Reset potential
    pub reset: f64,
    /// Resting potential
    pub rest: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            input_size: 128,
            hidden_sizes: vec![64],
            output_size: 3,
            tau_m: 20.0,
            threshold: 1.0,
            reset: 0.0,
            rest: 0.0,
        }
    }
}

/// Network state snapshot
#[derive(Debug, Clone)]
pub struct NetworkState {
    /// Average membrane potential across all neurons
    pub avg_membrane_potential: f64,
    /// Average spike rate (spikes per second)
    pub avg_spike_rate: f64,
    /// Total number of spikes in last timestep
    pub spike_count: usize,
    /// Number of active neurons
    pub active_neurons: usize,
    /// Current simulation time
    pub current_time: f64,
}

/// Spiking Neural Network
#[derive(Debug)]
pub struct SpikingNetwork {
    /// Network configuration
    config: NetworkConfig,
    /// Input layer
    input_layer: Layer,
    /// Hidden layers
    hidden_layers: Vec<Layer>,
    /// Output layer
    output_layer: Layer,
    /// Synapses between layers
    synapses: Vec<Vec<Synapse>>,
    /// Current simulation time
    current_time: f64,
    /// STDP configuration
    stdp_config: STDPConfig,
    /// Whether learning is enabled
    learning_enabled: bool,
}

impl SpikingNetwork {
    /// Create a new spiking neural network
    pub fn new(config: NetworkConfig) -> Self {
        let neuron_config = LIFConfig {
            tau_m: config.tau_m,
            threshold: config.threshold,
            reset: config.reset,
            rest: config.rest,
            ..Default::default()
        };

        // Create input layer
        let input_layer = Layer::new(LayerConfig {
            size: config.input_size,
            neuron_config,
            layer_id: 0,
        });

        // Create hidden layers
        let mut hidden_layers = Vec::new();
        for (i, &size) in config.hidden_sizes.iter().enumerate() {
            hidden_layers.push(Layer::new(LayerConfig {
                size,
                neuron_config,
                layer_id: i + 1,
            }));
        }

        // Create output layer
        let output_layer = Layer::new(LayerConfig {
            size: config.output_size,
            neuron_config,
            layer_id: config.hidden_sizes.len() + 1,
        });

        // Create synapses between layers
        let mut synapses = Vec::new();

        // Input -> First hidden (or output if no hidden)
        let first_target = if config.hidden_sizes.is_empty() {
            config.output_size
        } else {
            config.hidden_sizes[0]
        };
        synapses.push(Self::create_dense_connections(
            config.input_size,
            first_target,
            0,
        ));

        // Hidden -> Hidden
        for i in 0..config.hidden_sizes.len().saturating_sub(1) {
            synapses.push(Self::create_dense_connections(
                config.hidden_sizes[i],
                config.hidden_sizes[i + 1],
                config.input_size + config.hidden_sizes[0..i].iter().sum::<usize>(),
            ));
        }

        // Last hidden -> Output
        if !config.hidden_sizes.is_empty() {
            let last_hidden_size = *config.hidden_sizes.last().unwrap();
            let last_hidden_offset = config.input_size +
                config.hidden_sizes[0..config.hidden_sizes.len()-1].iter().sum::<usize>();
            synapses.push(Self::create_dense_connections(
                last_hidden_size,
                config.output_size,
                last_hidden_offset,
            ));
        }

        Self {
            config,
            input_layer,
            hidden_layers,
            output_layer,
            synapses,
            current_time: 0.0,
            stdp_config: STDPConfig::default(),
            learning_enabled: true,
        }
    }

    /// Create dense connections between layers
    fn create_dense_connections(pre_size: usize, post_size: usize, pre_offset: usize) -> Vec<Synapse> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut synapses = Vec::new();

        for i in 0..pre_size {
            for j in 0..post_size {
                let weight = rng.gen_range(0.1..0.5);
                synapses.push(Synapse::new(
                    pre_offset + i,
                    j,
                    crate::neuron::synapse::SynapseConfig {
                        weight,
                        delay: 1.0,
                        ..Default::default()
                    },
                ));
            }
        }

        synapses
    }

    /// Process one timestep
    ///
    /// Takes input spikes and returns output spikes
    pub fn step(&mut self, input_spikes: &[SpikeEvent], dt: f64) -> Vec<SpikeEvent> {
        self.current_time += dt;

        // Process input layer
        let mut all_spikes: Vec<SpikeEvent> = Vec::new();

        // Apply input spikes to input layer
        for spike in input_spikes {
            if spike.neuron_id < self.input_layer.size() {
                self.input_layer.apply_spike(spike.neuron_id, 1.0);
            }
        }

        // Step input layer
        let input_spikes = self.input_layer.step(dt);
        all_spikes.extend(input_spikes.iter().cloned());

        // Propagate through synapses and hidden layers
        let mut current_spikes = input_spikes;

        for (layer_idx, layer) in self.hidden_layers.iter_mut().enumerate() {
            // Apply spikes through synapses
            if layer_idx < self.synapses.len() {
                for synapse in &mut self.synapses[layer_idx] {
                    for spike in &current_spikes {
                        if spike.neuron_id == synapse.pre_id {
                            synapse.pre_spike(self.current_time);
                        }
                    }

                    // Deliver spikes
                    for weight in synapse.get_deliverable_spikes(self.current_time) {
                        layer.apply_spike(synapse.post_id, weight);
                    }
                }
            }

            // Step layer
            current_spikes = layer.step(dt);
            all_spikes.extend(current_spikes.iter().cloned());
        }

        // Process output layer
        let synapse_idx = self.synapses.len().saturating_sub(1);
        if !self.synapses.is_empty() {
            for synapse in &mut self.synapses[synapse_idx] {
                for spike in &current_spikes {
                    if spike.neuron_id == synapse.pre_id {
                        synapse.pre_spike(self.current_time);
                    }
                }

                for weight in synapse.get_deliverable_spikes(self.current_time) {
                    self.output_layer.apply_spike(synapse.post_id, weight);
                }
            }
        }

        let output_spikes = self.output_layer.step(dt);

        // Apply STDP if learning is enabled
        if self.learning_enabled {
            self.apply_stdp(&all_spikes, &output_spikes);
        }

        output_spikes
    }

    /// Apply STDP learning to all synapses
    fn apply_stdp(&mut self, _pre_spikes: &[SpikeEvent], post_spikes: &[SpikeEvent]) {
        for synapse_layer in &mut self.synapses {
            for synapse in synapse_layer {
                // Record post-synaptic spikes
                for spike in post_spikes {
                    if spike.neuron_id == synapse.post_id {
                        synapse.post_spike(self.current_time);
                    }
                }

                // Apply STDP
                synapse.apply_stdp(
                    self.stdp_config.a_plus,
                    self.stdp_config.a_minus,
                    self.stdp_config.tau_plus,
                    self.stdp_config.tau_minus,
                );
            }
        }
    }

    /// Get current network state
    pub fn get_state(&self) -> NetworkState {
        let mut total_potential = 0.0;
        let mut total_rate = 0.0;
        let mut neuron_count = 0;
        let mut active_count = 0;

        // Collect stats from all layers
        for state in self.input_layer.neuron_states() {
            total_potential += state.membrane_potential;
            if state.membrane_potential > self.config.rest + 0.1 {
                active_count += 1;
            }
            neuron_count += 1;
        }

        for layer in &self.hidden_layers {
            for state in layer.neuron_states() {
                total_potential += state.membrane_potential;
                if state.membrane_potential > self.config.rest + 0.1 {
                    active_count += 1;
                }
                neuron_count += 1;
            }
        }

        for state in self.output_layer.neuron_states() {
            total_potential += state.membrane_potential;
            if state.membrane_potential > self.config.rest + 0.1 {
                active_count += 1;
            }
            neuron_count += 1;
        }

        NetworkState {
            avg_membrane_potential: total_potential / neuron_count as f64,
            avg_spike_rate: total_rate / neuron_count as f64,
            spike_count: 0,  // Would need to track this
            active_neurons: active_count,
            current_time: self.current_time,
        }
    }

    /// Reset the network
    pub fn reset(&mut self) {
        self.input_layer.reset();
        for layer in &mut self.hidden_layers {
            layer.reset();
        }
        self.output_layer.reset();

        for synapse_layer in &mut self.synapses {
            for synapse in synapse_layer {
                synapse.reset_timing();
            }
        }

        self.current_time = 0.0;
    }

    /// Enable or disable learning
    pub fn set_learning(&mut self, enabled: bool) {
        self.learning_enabled = enabled;
    }

    /// Set STDP configuration
    pub fn set_stdp_config(&mut self, config: STDPConfig) {
        self.stdp_config = config;
    }

    /// Get the number of neurons in each layer
    pub fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![self.input_layer.size()];
        for layer in &self.hidden_layers {
            sizes.push(layer.size());
        }
        sizes.push(self.output_layer.size());
        sizes
    }

    /// Get total number of neurons
    pub fn total_neurons(&self) -> usize {
        self.layer_sizes().iter().sum()
    }

    /// Get total number of synapses
    pub fn total_synapses(&self) -> usize {
        self.synapses.iter().map(|s| s.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let config = NetworkConfig {
            input_size: 10,
            hidden_sizes: vec![5],
            output_size: 3,
            ..Default::default()
        };

        let network = SpikingNetwork::new(config);

        assert_eq!(network.layer_sizes(), vec![10, 5, 3]);
        assert_eq!(network.total_neurons(), 18);
    }

    #[test]
    fn test_network_step() {
        let config = NetworkConfig {
            input_size: 4,
            hidden_sizes: vec![2],
            output_size: 2,
            ..Default::default()
        };

        let mut network = SpikingNetwork::new(config);

        // Create some input spikes
        let input_spikes = vec![
            SpikeEvent::new(0, 0.0),
            SpikeEvent::new(1, 0.0),
        ];

        // Step the network
        let _output = network.step(&input_spikes, 1.0);

        // Network should have advanced time
        assert!(network.current_time > 0.0);
    }

    #[test]
    fn test_network_reset() {
        let config = NetworkConfig::default();
        let mut network = SpikingNetwork::new(config);

        // Run some steps
        for _ in 0..10 {
            network.step(&[], 1.0);
        }

        // Reset
        network.reset();

        assert_eq!(network.current_time, 0.0);
    }
}
