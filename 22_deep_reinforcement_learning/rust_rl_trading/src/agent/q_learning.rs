//! Tabular Q-Learning agent for simple environments.

use crate::agent::Agent;
use crate::environment::{TradingAction, TradingState};
use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Discretization configuration for continuous states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discretizer {
    /// Number of bins per feature
    bins: Vec<usize>,
    /// Minimum values per feature
    mins: Vec<f64>,
    /// Maximum values per feature
    maxs: Vec<f64>,
}

impl Discretizer {
    /// Create a new discretizer with uniform bins
    pub fn new(num_features: usize, num_bins: usize, min_val: f64, max_val: f64) -> Self {
        Self {
            bins: vec![num_bins; num_features],
            mins: vec![min_val; num_features],
            maxs: vec![max_val; num_features],
        }
    }

    /// Create with custom ranges per feature
    pub fn with_ranges(bins: Vec<usize>, mins: Vec<f64>, maxs: Vec<f64>) -> Self {
        Self { bins, mins, maxs }
    }

    /// Discretize a continuous state into a discrete state key
    pub fn discretize(&self, state: &[f64]) -> Vec<usize> {
        state
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let min = self.mins.get(i).copied().unwrap_or(-1.0);
                let max = self.maxs.get(i).copied().unwrap_or(1.0);
                let num_bins = self.bins.get(i).copied().unwrap_or(10);

                let normalized = (val - min) / (max - min);
                let bin = (normalized * num_bins as f64).floor() as usize;
                bin.min(num_bins - 1)
            })
            .collect()
    }
}

/// Q-Learning agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QLearningConfig {
    /// Learning rate
    pub alpha: f64,
    /// Discount factor
    pub gamma: f64,
    /// Initial epsilon for exploration
    pub epsilon_start: f64,
    /// Final epsilon
    pub epsilon_end: f64,
    /// Epsilon decay rate
    pub epsilon_decay: f64,
    /// Number of bins for discretization
    pub num_bins: usize,
}

impl Default for QLearningConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            num_bins: 10,
        }
    }
}

/// Tabular Q-Learning agent
#[derive(Serialize, Deserialize)]
pub struct QLearningAgent {
    /// Q-table mapping state-action to value
    q_table: HashMap<(Vec<usize>, usize), f64>,
    /// Discretizer for continuous states
    discretizer: Discretizer,
    /// Configuration
    config: QLearningConfig,
    /// Current epsilon
    epsilon: f64,
    /// Number of state features
    state_size: usize,
    /// Number of actions
    action_size: usize,
}

impl QLearningAgent {
    /// Create a new Q-Learning agent
    pub fn new(state_size: usize, action_size: usize, config: QLearningConfig) -> Self {
        let discretizer = Discretizer::new(state_size, config.num_bins, -1.0, 1.0);

        Self {
            q_table: HashMap::new(),
            discretizer,
            config: config.clone(),
            epsilon: config.epsilon_start,
            state_size,
            action_size,
        }
    }

    /// Get Q-value for state-action pair
    fn get_q_value(&self, state: &Vec<usize>, action: usize) -> f64 {
        self.q_table
            .get(&(state.clone(), action))
            .copied()
            .unwrap_or(0.0)
    }

    /// Set Q-value for state-action pair
    fn set_q_value(&mut self, state: Vec<usize>, action: usize, value: f64) {
        self.q_table.insert((state, action), value);
    }

    /// Get best action for a discretized state
    fn best_action_for_state(&self, state: &Vec<usize>) -> usize {
        (0..self.action_size)
            .max_by(|&a, &b| {
                let q_a = self.get_q_value(state, a);
                let q_b = self.get_q_value(state, b);
                q_a.partial_cmp(&q_b).unwrap()
            })
            .unwrap_or(0)
    }

    /// Update Q-value using Q-learning update rule
    fn update(
        &mut self,
        state: Vec<usize>,
        action: usize,
        reward: f64,
        next_state: Vec<usize>,
        done: bool,
    ) {
        let current_q = self.get_q_value(&state, action);

        let target = if done {
            reward
        } else {
            let best_next_action = self.best_action_for_state(&next_state);
            let max_next_q = self.get_q_value(&next_state, best_next_action);
            reward + self.config.gamma * max_next_q
        };

        let new_q = current_q + self.config.alpha * (target - current_q);
        self.set_q_value(state, action, new_q);
    }

    /// Get statistics about the Q-table
    pub fn stats(&self) -> (usize, f64, f64) {
        let num_entries = self.q_table.len();
        let values: Vec<f64> = self.q_table.values().copied().collect();

        if values.is_empty() {
            return (0, 0.0, 0.0);
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        (num_entries, mean, std)
    }
}

impl Agent for QLearningAgent {
    fn select_action(&self, state: &TradingState, epsilon: f64) -> TradingAction {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < epsilon {
            // Random action (exploration)
            let action_idx = rng.gen_range(0..self.action_size);
            TradingAction::from_index(action_idx).unwrap_or(TradingAction::Hold)
        } else {
            // Best action (exploitation)
            let state_array = state.to_array();
            let discrete_state = self.discretizer.discretize(state_array.as_slice().unwrap());
            let action_idx = self.best_action_for_state(&discrete_state);
            TradingAction::from_index(action_idx).unwrap_or(TradingAction::Hold)
        }
    }

    fn learn(&mut self, experiences: &[(TradingState, TradingAction, f64, TradingState, bool)]) {
        for (state, action, reward, next_state, done) in experiences {
            let state_array = state.to_array();
            let next_state_array = next_state.to_array();

            let discrete_state = self.discretizer.discretize(state_array.as_slice().unwrap());
            let discrete_next = self.discretizer.discretize(next_state_array.as_slice().unwrap());

            self.update(
                discrete_state,
                action.to_index(),
                *reward,
                discrete_next,
                *done,
            );
        }
    }

    fn get_epsilon(&self) -> f64 {
        self.epsilon
    }

    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);
    }

    fn save(&self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let loaded: QLearningAgent = serde_json::from_reader(reader)?;
        *self = loaded;
        Ok(())
    }

    fn name(&self) -> &str {
        "Q-Learning"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_discretizer() {
        let discretizer = Discretizer::new(3, 10, -1.0, 1.0);
        let state = vec![0.0, 0.5, -0.5];
        let discrete = discretizer.discretize(&state);

        assert_eq!(discrete.len(), 3);
        assert!(discrete.iter().all(|&b| b < 10));
    }

    #[test]
    fn test_q_learning_agent() {
        let config = QLearningConfig::default();
        let mut agent = QLearningAgent::new(7, 3, config);

        let state = TradingState::new(Array1::zeros(7), 0.0, 0.0, 0.0, 0.0);
        let action = agent.select_action(&state, 0.5);

        assert!(action.to_index() < 3);
    }

    #[test]
    fn test_learning() {
        let config = QLearningConfig::default();
        let mut agent = QLearningAgent::new(7, 3, config);

        let state = TradingState::new(Array1::from_vec(vec![0.1; 7]), 0.0, 0.0, 0.0, 0.0);
        let next_state = TradingState::new(Array1::from_vec(vec![0.2; 7]), 1.0, 0.05, 0.1, 0.1);

        let experiences = vec![(state, TradingAction::Long, 1.0, next_state, false)];

        agent.learn(&experiences);

        let (entries, _, _) = agent.stats();
        assert!(entries > 0);
    }
}
