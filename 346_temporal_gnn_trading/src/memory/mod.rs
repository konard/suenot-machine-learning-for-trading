//! Memory module for TGN
//!
//! This module implements the temporal memory mechanism that allows
//! TGN to maintain state across events.

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Memory dimension per node
    pub dim: usize,
    /// Decay rate for stale memories
    pub decay_rate: f64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            decay_rate: 0.01,
        }
    }
}

/// Memory state for a single node
#[derive(Debug, Clone)]
pub struct MemoryState {
    /// Memory vector
    pub vector: Array1<f64>,
    /// Last update timestamp
    pub last_update: u64,
    /// Number of updates
    pub update_count: u64,
}

impl MemoryState {
    /// Create new memory state
    pub fn new(dim: usize) -> Self {
        Self {
            vector: Array1::zeros(dim),
            last_update: 0,
            update_count: 0,
        }
    }

    /// Apply time decay to memory
    pub fn apply_decay(&mut self, current_time: u64, decay_rate: f64) {
        if current_time > self.last_update {
            let delta_t = (current_time - self.last_update) as f64 / 1000.0; // Convert to seconds
            let decay_factor = (-decay_rate * delta_t).exp();
            self.vector.mapv_inplace(|x| x * decay_factor);
        }
    }
}

/// Memory updater types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UpdaterType {
    /// GRU-based update
    GRU,
    /// Simple mean update
    Mean,
    /// Last message only
    Last,
    /// Decay-based update
    Decay,
}

/// Memory updater implementing various update mechanisms
#[derive(Debug)]
pub struct MemoryUpdater {
    /// Updater type
    updater_type: UpdaterType,
    /// Hidden dimension
    dim: usize,
    /// GRU weights (if using GRU)
    gru_weights: Option<GRUWeights>,
}

/// GRU weights for memory update
#[derive(Debug)]
struct GRUWeights {
    /// Update gate weights
    w_z: Array2<f64>,
    /// Reset gate weights
    w_r: Array2<f64>,
    /// Candidate weights
    w_h: Array2<f64>,
}

impl MemoryUpdater {
    /// Create new memory updater
    pub fn new(updater_type: UpdaterType, dim: usize) -> Self {
        let gru_weights = match updater_type {
            UpdaterType::GRU => Some(GRUWeights {
                w_z: Array2::from_shape_fn((dim * 2, dim), |_| {
                    rand::random::<f64>() * 0.1 - 0.05
                }),
                w_r: Array2::from_shape_fn((dim * 2, dim), |_| {
                    rand::random::<f64>() * 0.1 - 0.05
                }),
                w_h: Array2::from_shape_fn((dim * 2, dim), |_| {
                    rand::random::<f64>() * 0.1 - 0.05
                }),
            }),
            _ => None,
        };

        Self {
            updater_type,
            dim,
            gru_weights,
        }
    }

    /// Update memory state with new message
    pub fn update(&self, memory: &Array1<f64>, message: &Array1<f64>) -> Array1<f64> {
        match self.updater_type {
            UpdaterType::GRU => self.gru_update(memory, message),
            UpdaterType::Mean => self.mean_update(memory, message),
            UpdaterType::Last => message.clone(),
            UpdaterType::Decay => self.decay_update(memory, message),
        }
    }

    /// GRU-based update
    fn gru_update(&self, memory: &Array1<f64>, message: &Array1<f64>) -> Array1<f64> {
        let weights = self.gru_weights.as_ref().unwrap();

        // Concatenate memory and message
        let mut concat = Array1::zeros(self.dim * 2);
        concat.slice_mut(ndarray::s![..self.dim]).assign(memory);
        concat.slice_mut(ndarray::s![self.dim..]).assign(message);

        // Update gate
        let z_linear = concat.dot(&weights.w_z);
        let z: Array1<f64> = z_linear.mapv(sigmoid);

        // Reset gate
        let r_linear = concat.dot(&weights.w_r);
        let r: Array1<f64> = r_linear.mapv(sigmoid);

        // Reset memory
        let reset_memory: Array1<f64> = memory * &r;

        // Candidate
        let mut concat_reset = Array1::zeros(self.dim * 2);
        concat_reset.slice_mut(ndarray::s![..self.dim]).assign(&reset_memory);
        concat_reset.slice_mut(ndarray::s![self.dim..]).assign(message);
        let h_linear = concat_reset.dot(&weights.w_h);
        let h_candidate: Array1<f64> = h_linear.mapv(|x| x.tanh());

        // Final update
        let one_minus_z: Array1<f64> = z.mapv(|x| 1.0 - x);
        &one_minus_z * memory + &z * &h_candidate
    }

    /// Mean update
    fn mean_update(&self, memory: &Array1<f64>, message: &Array1<f64>) -> Array1<f64> {
        (memory + message) / 2.0
    }

    /// Decay-based update
    fn decay_update(&self, memory: &Array1<f64>, message: &Array1<f64>) -> Array1<f64> {
        let alpha = 0.5; // Learning rate for new information
        memory * (1.0 - alpha) + message * alpha
    }
}

/// Main memory structure for TGN
#[derive(Debug)]
pub struct Memory {
    /// Memory configuration
    config: MemoryConfig,
    /// Memory states per node
    states: HashMap<usize, MemoryState>,
    /// Memory updater
    updater: MemoryUpdater,
}

impl Memory {
    /// Create new memory
    pub fn new(config: MemoryConfig) -> Self {
        let updater = MemoryUpdater::new(UpdaterType::GRU, config.dim);
        Self {
            config,
            states: HashMap::new(),
            updater,
        }
    }

    /// Initialize memory for a new node
    pub fn initialize_node(&mut self, node_idx: usize) {
        self.states.insert(node_idx, MemoryState::new(self.config.dim));
    }

    /// Get memory state for a node
    pub fn get_state(&self, node_idx: usize) -> Array1<f64> {
        self.states
            .get(&node_idx)
            .map(|s| s.vector.clone())
            .unwrap_or_else(|| Array1::zeros(self.config.dim))
    }

    /// Update memory for a node
    pub fn update(&mut self, node_idx: usize, message: &Array1<f64>, timestamp: u64) {
        let state = self.states
            .entry(node_idx)
            .or_insert_with(|| MemoryState::new(self.config.dim));

        // Apply time decay
        state.apply_decay(timestamp, self.config.decay_rate);

        // Update memory
        state.vector = self.updater.update(&state.vector, message);
        state.last_update = timestamp;
        state.update_count += 1;
    }

    /// Get time since last update for a node
    pub fn time_since_update(&self, node_idx: usize, current_time: u64) -> f64 {
        self.states
            .get(&node_idx)
            .map(|s| (current_time.saturating_sub(s.last_update)) as f64 / 1000.0)
            .unwrap_or(0.0)
    }

    /// Get all last update times
    pub fn get_all_last_update_times(&self) -> Vec<(usize, u64)> {
        self.states
            .iter()
            .map(|(&idx, state)| (idx, state.last_update))
            .collect()
    }

    /// Reset all memories
    pub fn reset(&mut self) {
        self.states.clear();
    }

    /// Get number of nodes with memory
    pub fn num_nodes(&self) -> usize {
        self.states.len()
    }
}

/// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_creation() {
        let config = MemoryConfig::default();
        let memory = Memory::new(config);
        assert_eq!(memory.num_nodes(), 0);
    }

    #[test]
    fn test_memory_update() {
        let config = MemoryConfig { dim: 4, decay_rate: 0.01 };
        let mut memory = Memory::new(config);

        memory.initialize_node(0);
        let message = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        memory.update(0, &message, 1000);

        let state = memory.get_state(0);
        assert_eq!(state.len(), 4);
    }

    #[test]
    fn test_time_decay() {
        let mut state = MemoryState::new(4);
        state.vector = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        state.last_update = 0;

        state.apply_decay(1000, 0.001);

        // After decay, values should be smaller
        assert!(state.vector.iter().all(|&x| x < 1.0));
    }
}
