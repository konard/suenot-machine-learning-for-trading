//! Deep Q-Network (DQN) agent implementation.

use crate::agent::{Agent, Experience, NeuralNetwork, ReplayBuffer};
use crate::environment::{TradingAction, TradingState};
use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// DQN agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DQNConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Discount factor
    pub gamma: f64,
    /// Initial epsilon for exploration
    pub epsilon_start: f64,
    /// Final epsilon
    pub epsilon_end: f64,
    /// Epsilon decay rate
    pub epsilon_decay: f64,
    /// Replay buffer capacity
    pub buffer_size: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Target network update frequency
    pub target_update_freq: usize,
    /// Soft update coefficient (tau)
    pub tau: f64,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Whether to use Double DQN
    pub double_dqn: bool,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            buffer_size: 100_000,
            batch_size: 64,
            target_update_freq: 100,
            tau: 0.005,
            hidden_layers: vec![128, 64],
            double_dqn: true,
        }
    }
}

/// Deep Q-Network agent
pub struct DQNAgent {
    /// Q-network (policy network)
    q_network: NeuralNetwork,
    /// Target network
    target_network: NeuralNetwork,
    /// Experience replay buffer
    replay_buffer: ReplayBuffer,
    /// Configuration
    config: DQNConfig,
    /// Current epsilon
    epsilon: f64,
    /// State size
    state_size: usize,
    /// Action size
    action_size: usize,
    /// Training steps counter
    train_step: usize,
}

impl DQNAgent {
    /// Create a new DQN agent
    pub fn new(state_size: usize, action_size: usize, config: DQNConfig) -> Self {
        // Build network architecture
        let mut layer_sizes = vec![state_size];
        layer_sizes.extend(&config.hidden_layers);
        layer_sizes.push(action_size);

        let q_network = NeuralNetwork::new(&layer_sizes, config.learning_rate);
        let mut target_network = NeuralNetwork::new(&layer_sizes, config.learning_rate);
        target_network.copy_from(&q_network);

        let replay_buffer = ReplayBuffer::new(config.buffer_size);

        Self {
            q_network,
            target_network,
            replay_buffer,
            config: config.clone(),
            epsilon: config.epsilon_start,
            state_size,
            action_size,
            train_step: 0,
        }
    }

    /// Store experience in replay buffer
    pub fn remember(&mut self, experience: Experience) {
        self.replay_buffer.push(experience);
    }

    /// Store experience from components
    pub fn remember_transition(
        &mut self,
        state: TradingState,
        action: TradingAction,
        reward: f64,
        next_state: TradingState,
        done: bool,
    ) {
        self.remember(Experience::new(state, action, reward, next_state, done));
    }

    /// Train the network on a batch of experiences
    pub fn train_step(&mut self) -> Option<f64> {
        if !self.replay_buffer.can_sample(self.config.batch_size) {
            return None;
        }

        let batch = self.replay_buffer.sample(self.config.batch_size);
        let mut total_loss = 0.0;

        // Prepare batch data
        let mut states = Vec::with_capacity(batch.len());
        let mut actions = Vec::with_capacity(batch.len());
        let mut targets = Vec::with_capacity(batch.len());

        for exp in &batch {
            let state_array = exp.state.to_array();
            let next_state_array = exp.next_state.to_array();

            // Compute target Q-value
            let target = if exp.done {
                exp.reward
            } else if self.config.double_dqn {
                // Double DQN: use Q-network to select action, target network to evaluate
                let next_action = self.q_network.best_action(&next_state_array);
                let next_q_values = self.target_network.predict(&next_state_array);
                exp.reward + self.config.gamma * next_q_values[next_action]
            } else {
                // Standard DQN: use target network for both
                let next_q_values = self.target_network.predict(&next_state_array);
                let max_next_q = next_q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                exp.reward + self.config.gamma * max_next_q
            };

            // Compute TD error for loss tracking
            let current_q = self.q_network.predict(&state_array);
            let td_error = target - current_q[exp.action.to_index()];
            total_loss += td_error.powi(2);

            states.push(state_array);
            actions.push(exp.action.to_index());
            targets.push(target);
        }

        // Train Q-network
        self.q_network.train_batch(&states, &actions, &targets);

        self.train_step += 1;

        // Update target network
        if self.train_step % self.config.target_update_freq == 0 {
            self.update_target_network();
        }

        Some(total_loss / batch.len() as f64)
    }

    /// Update target network (soft update)
    fn update_target_network(&mut self) {
        self.target_network
            .soft_update(&self.q_network, self.config.tau);
    }

    /// Hard update target network
    pub fn hard_update_target(&mut self) {
        self.target_network.copy_from(&self.q_network);
    }

    /// Get current replay buffer size
    pub fn buffer_size(&self) -> usize {
        self.replay_buffer.len()
    }

    /// Check if ready to train
    pub fn can_train(&self) -> bool {
        self.replay_buffer.can_sample(self.config.batch_size)
    }

    /// Get Q-values for a state
    pub fn get_q_values(&self, state: &TradingState) -> Vec<f64> {
        let state_array = state.to_array();
        self.q_network.predict(&state_array).to_vec()
    }
}

impl Agent for DQNAgent {
    fn select_action(&self, state: &TradingState, epsilon: f64) -> TradingAction {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < epsilon {
            // Random action (exploration)
            let action_idx = rng.gen_range(0..self.action_size);
            TradingAction::from_index(action_idx).unwrap_or(TradingAction::Hold)
        } else {
            // Best action (exploitation)
            let state_array = state.to_array();
            let action_idx = self.q_network.best_action(&state_array);
            TradingAction::from_index(action_idx).unwrap_or(TradingAction::Hold)
        }
    }

    fn learn(&mut self, experiences: &[(TradingState, TradingAction, f64, TradingState, bool)]) {
        // Add experiences to buffer
        for (state, action, reward, next_state, done) in experiences {
            self.remember(Experience::new(
                state.clone(),
                *action,
                *reward,
                next_state.clone(),
                *done,
            ));
        }

        // Train if buffer is ready
        self.train_step();
    }

    fn get_epsilon(&self) -> f64 {
        self.epsilon
    }

    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);
    }

    fn save(&self, path: &str) -> Result<()> {
        self.q_network.save(path)
    }

    fn load(&mut self, path: &str) -> Result<()> {
        self.q_network = NeuralNetwork::load(path)?;
        self.target_network.copy_from(&self.q_network);
        Ok(())
    }

    fn name(&self) -> &str {
        if self.config.double_dqn {
            "Double DQN"
        } else {
            "DQN"
        }
    }
}

/// Serializable agent state for saving/loading
#[derive(Serialize, Deserialize)]
struct DQNAgentState {
    q_network: NeuralNetwork,
    config: DQNConfig,
    epsilon: f64,
    state_size: usize,
    action_size: usize,
    train_step: usize,
}

impl DQNAgent {
    /// Save complete agent state
    pub fn save_full(&self, path: &str) -> Result<()> {
        let state = DQNAgentState {
            q_network: self.q_network.clone(),
            config: self.config.clone(),
            epsilon: self.epsilon,
            state_size: self.state_size,
            action_size: self.action_size,
            train_step: self.train_step,
        };

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &state)?;
        Ok(())
    }

    /// Load complete agent state
    pub fn load_full(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state: DQNAgentState = serde_json::from_reader(reader)?;

        let mut layer_sizes = vec![state.state_size];
        layer_sizes.extend(&state.config.hidden_layers);
        layer_sizes.push(state.action_size);

        let mut target_network = NeuralNetwork::new(&layer_sizes, state.config.learning_rate);
        target_network.copy_from(&state.q_network);

        Ok(Self {
            q_network: state.q_network,
            target_network,
            replay_buffer: ReplayBuffer::new(state.config.buffer_size),
            config: state.config,
            epsilon: state.epsilon,
            state_size: state.state_size,
            action_size: state.action_size,
            train_step: state.train_step,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_dqn_creation() {
        let config = DQNConfig::default();
        let agent = DQNAgent::new(11, 3, config);

        assert_eq!(agent.state_size, 11);
        assert_eq!(agent.action_size, 3);
    }

    #[test]
    fn test_action_selection() {
        let config = DQNConfig::default();
        let agent = DQNAgent::new(11, 3, config);

        let state = TradingState::new(Array1::zeros(7), 0.0, 0.0, 0.0, 0.0);

        // With epsilon=1.0, should always be random
        let action = agent.select_action(&state, 1.0);
        assert!(action.to_index() < 3);

        // With epsilon=0.0, should always be greedy
        let action = agent.select_action(&state, 0.0);
        assert!(action.to_index() < 3);
    }

    #[test]
    fn test_remember_and_train() {
        let config = DQNConfig {
            batch_size: 4,
            buffer_size: 100,
            ..Default::default()
        };
        let mut agent = DQNAgent::new(11, 3, config);

        // Add experiences
        for i in 0..10 {
            let state = TradingState::new(
                Array1::from_vec(vec![i as f64 * 0.1; 7]),
                0.0,
                0.0,
                0.0,
                0.0,
            );
            let next_state = TradingState::new(
                Array1::from_vec(vec![(i + 1) as f64 * 0.1; 7]),
                1.0,
                0.05,
                0.1,
                0.1,
            );
            agent.remember_transition(state, TradingAction::Long, 0.5, next_state, i == 9);
        }

        assert!(agent.can_train());

        // Train
        let loss = agent.train_step();
        assert!(loss.is_some());
    }
}
