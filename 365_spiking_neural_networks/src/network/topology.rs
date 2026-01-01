//! Network topology and architecture

use crate::network::layer::{SNNLayer, LayerConfig, LayerType};
use crate::learning::LearningRule;

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Simulation timestep
    pub dt: f64,
    /// Learning enabled
    pub learning_enabled: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            dt: 1.0,
            learning_enabled: true,
        }
    }
}

/// A complete Spiking Neural Network
#[derive(Debug)]
pub struct SNNNetwork {
    /// Network layers
    layers: Vec<SNNLayer>,
    /// Network configuration
    config: NetworkConfig,
    /// Current simulation time
    current_time: f64,
    /// Learning rule (optional)
    learning_rule: Option<Box<dyn LearningRule>>,
}

impl SNNNetwork {
    /// Create a new network builder
    pub fn builder() -> SNNNetworkBuilder {
        SNNNetworkBuilder::new()
    }

    /// Create a network from layers
    pub fn new(layers: Vec<SNNLayer>, config: NetworkConfig) -> Self {
        Self {
            layers,
            config,
            current_time: 0.0,
            learning_rule: None,
        }
    }

    /// Set learning rule
    pub fn set_learning_rule(&mut self, rule: Box<dyn LearningRule>) {
        self.learning_rule = Some(rule);
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get input layer size
    pub fn input_size(&self) -> usize {
        self.layers.first().map(|l| l.size()).unwrap_or(0)
    }

    /// Get output layer size
    pub fn output_size(&self) -> usize {
        self.layers.last().map(|l| l.size()).unwrap_or(0)
    }

    /// Reset the network
    pub fn reset(&mut self) {
        self.current_time = 0.0;
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &[f64]) -> Vec<bool> {
        if self.layers.is_empty() {
            return vec![];
        }

        let dt = self.config.dt;
        self.current_time += dt;

        // Process input layer
        let mut current_spikes = self.layers[0].forward(input, dt, self.current_time);

        // Process hidden and output layers
        for i in 1..self.layers.len() {
            // Get weights from previous layer
            let weights = self.layers[i - 1].weights().cloned();

            if let Some(ref w) = weights {
                self.layers[i].receive_spikes(&current_spikes, w);
            }

            // Create zero input for hidden/output layers
            let zero_input = vec![0.0; self.layers[i].size()];
            current_spikes = self.layers[i].forward(&zero_input, dt, self.current_time);
        }

        current_spikes
    }

    /// Run for multiple timesteps
    pub fn run(&mut self, input: &[f64], steps: usize) -> Vec<Vec<bool>> {
        let mut all_spikes = Vec::with_capacity(steps);

        for _ in 0..steps {
            let spikes = self.forward(input);
            all_spikes.push(spikes);
        }

        all_spikes
    }

    /// Get output spike counts
    pub fn output_spike_counts(&self, window_start: f64) -> Vec<usize> {
        if let Some(output_layer) = self.layers.last() {
            output_layer.spike_count(window_start)
        } else {
            vec![]
        }
    }

    /// Get output membrane potentials
    pub fn output_potentials(&self) -> Vec<f64> {
        if let Some(output_layer) = self.layers.last() {
            output_layer.membrane_potentials()
        } else {
            vec![]
        }
    }

    /// Apply learning with reward signal
    pub fn learn(&mut self, reward: f64) {
        if !self.config.learning_enabled {
            return;
        }

        if let Some(ref mut rule) = self.learning_rule {
            for i in 0..self.layers.len() - 1 {
                if let Some(weights) = self.layers[i].weights_mut() {
                    let pre_spikes = self.layers[i].last_spikes();
                    let post_spikes = self.layers[i + 1].last_spikes();

                    for (pre_idx, pre_spike) in pre_spikes.iter().enumerate() {
                        for (post_idx, post_spike) in post_spikes.iter().enumerate() {
                            let dw = rule.compute_weight_change(
                                pre_spike.as_ref(),
                                post_spike.as_ref(),
                                reward,
                            );
                            weights[pre_idx][post_idx] += dw;
                        }
                    }
                }
            }
        }
    }

    /// Get current simulation time
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Get layer by index
    pub fn layer(&self, idx: usize) -> Option<&SNNLayer> {
        self.layers.get(idx)
    }

    /// Get mutable layer by index
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut SNNLayer> {
        self.layers.get_mut(idx)
    }
}

/// Builder for constructing SNNNetwork
pub struct SNNNetworkBuilder {
    layer_configs: Vec<LayerConfig>,
    config: NetworkConfig,
}

impl SNNNetworkBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            layer_configs: Vec::new(),
            config: NetworkConfig::default(),
        }
    }

    /// Add input layer
    pub fn input_layer(mut self, size: usize) -> Self {
        self.layer_configs.push(LayerConfig::input(size));
        self
    }

    /// Add hidden layer
    pub fn hidden_layer(mut self, size: usize) -> Self {
        self.layer_configs.push(LayerConfig::hidden(size));
        self
    }

    /// Add output layer
    pub fn output_layer(mut self, size: usize) -> Self {
        self.layer_configs.push(LayerConfig::output(size));
        self
    }

    /// Add custom layer
    pub fn layer(mut self, config: LayerConfig) -> Self {
        self.layer_configs.push(config);
        self
    }

    /// Set simulation timestep
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.config.dt = dt;
        self
    }

    /// Enable or disable learning
    pub fn with_learning(mut self, enabled: bool) -> Self {
        self.config.learning_enabled = enabled;
        self
    }

    /// Build the network
    pub fn build(self) -> SNNNetwork {
        let mut layers: Vec<SNNLayer> = self.layer_configs
            .into_iter()
            .map(SNNLayer::new)
            .collect();

        // Initialize weights between layers
        for i in 0..layers.len() - 1 {
            let next_size = layers[i + 1].size();
            layers[i].init_weights(next_size);
        }

        SNNNetwork::new(layers, self.config)
    }
}

impl Default for SNNNetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_builder() {
        let network = SNNNetwork::builder()
            .input_layer(10)
            .hidden_layer(20)
            .output_layer(2)
            .build();

        assert_eq!(network.num_layers(), 3);
        assert_eq!(network.input_size(), 10);
        assert_eq!(network.output_size(), 2);
    }

    #[test]
    fn test_network_forward() {
        let mut network = SNNNetwork::builder()
            .input_layer(5)
            .hidden_layer(10)
            .output_layer(2)
            .build();

        let input = vec![50.0, 50.0, 50.0, 50.0, 50.0];
        let output = network.forward(&input);

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_network_run() {
        let mut network = SNNNetwork::builder()
            .input_layer(5)
            .output_layer(2)
            .build();

        let input = vec![50.0; 5];
        let spikes = network.run(&input, 100);

        assert_eq!(spikes.len(), 100);
        assert!(spikes.iter().any(|s| s.iter().any(|&v| v)),
            "Should have some output spikes");
    }

    #[test]
    fn test_network_reset() {
        let mut network = SNNNetwork::builder()
            .input_layer(5)
            .output_layer(2)
            .build();

        // Run for a while
        let input = vec![50.0; 5];
        network.run(&input, 10);

        let time_before = network.current_time();
        assert!(time_before > 0.0);

        network.reset();

        assert_eq!(network.current_time(), 0.0);
    }
}
