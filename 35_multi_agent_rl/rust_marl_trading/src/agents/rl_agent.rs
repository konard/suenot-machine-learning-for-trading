//! Reinforcement Learning agent implementation

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::{Agent, AgentAction, AgentId, AgentState};
use crate::environment::Observation;

/// Experience tuple for replay buffer
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
}

/// Simple replay buffer
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    buffer: Vec<Experience>,
    capacity: usize,
    position: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(experience);
        } else {
            self.buffer[self.position] = experience;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(batch_size);

        for _ in 0..batch_size.min(self.buffer.len()) {
            let idx = rng.gen_range(0..self.buffer.len());
            samples.push(self.buffer[idx].clone());
        }

        samples
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Simple neural network for Q-learning
#[derive(Debug, Clone)]
pub struct SimpleNetwork {
    weights1: Array2<f64>,
    bias1: Array1<f64>,
    weights2: Array2<f64>,
    bias2: Array1<f64>,
}

impl SimpleNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        let weights1 = Array2::from_shape_fn((input_size, hidden_size), |_| normal.sample(&mut rng));
        let bias1 = Array1::zeros(hidden_size);
        let weights2 = Array2::from_shape_fn((hidden_size, output_size), |_| normal.sample(&mut rng));
        let bias2 = Array1::zeros(output_size);

        Self {
            weights1,
            bias1,
            weights2,
            bias2,
        }
    }

    /// Forward pass with ReLU activation
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        // Hidden layer
        let hidden = input.dot(&self.weights1) + &self.bias1;
        let hidden = hidden.mapv(|x| x.max(0.0)); // ReLU

        // Output layer
        hidden.dot(&self.weights2) + &self.bias2
    }

    /// Get Q-values for a state
    pub fn q_values(&self, state: &[f64]) -> Vec<f64> {
        let input = Array1::from_vec(state.to_vec());
        self.forward(&input).to_vec()
    }

    /// Simple gradient update (SGD)
    pub fn update(&mut self, gradient: &NetworkGradient, learning_rate: f64) {
        self.weights1 = &self.weights1 - &(&gradient.weights1 * learning_rate);
        self.bias1 = &self.bias1 - &(&gradient.bias1 * learning_rate);
        self.weights2 = &self.weights2 - &(&gradient.weights2 * learning_rate);
        self.bias2 = &self.bias2 - &(&gradient.bias2 * learning_rate);
    }

    /// Copy weights from another network
    pub fn copy_from(&mut self, other: &SimpleNetwork) {
        self.weights1.assign(&other.weights1);
        self.bias1.assign(&other.bias1);
        self.weights2.assign(&other.weights2);
        self.bias2.assign(&other.bias2);
    }

    /// Soft update (for target networks)
    pub fn soft_update(&mut self, other: &SimpleNetwork, tau: f64) {
        self.weights1 = &self.weights1 * (1.0 - tau) + &other.weights1 * tau;
        self.bias1 = &self.bias1 * (1.0 - tau) + &other.bias1 * tau;
        self.weights2 = &self.weights2 * (1.0 - tau) + &other.weights2 * tau;
        self.bias2 = &self.bias2 * (1.0 - tau) + &other.bias2 * tau;
    }
}

/// Gradient storage
#[derive(Debug, Clone)]
pub struct NetworkGradient {
    weights1: Array2<f64>,
    bias1: Array1<f64>,
    weights2: Array2<f64>,
    bias2: Array1<f64>,
}

impl NetworkGradient {
    pub fn zeros_like(network: &SimpleNetwork) -> Self {
        Self {
            weights1: Array2::zeros(network.weights1.raw_dim()),
            bias1: Array1::zeros(network.bias1.len()),
            weights2: Array2::zeros(network.weights2.raw_dim()),
            bias2: Array1::zeros(network.bias2.len()),
        }
    }
}

/// RL Agent using Deep Q-Learning
#[derive(Debug, Clone)]
pub struct RLAgent {
    id: AgentId,
    /// Q-network
    network: SimpleNetwork,
    /// Target network
    target_network: SimpleNetwork,
    /// Replay buffer
    replay_buffer: ReplayBuffer,
    /// Exploration rate
    epsilon: f64,
    /// Epsilon decay
    epsilon_decay: f64,
    /// Minimum epsilon
    epsilon_min: f64,
    /// Discount factor
    gamma: f64,
    /// Learning rate
    learning_rate: f64,
    /// Batch size for training
    batch_size: usize,
    /// Target network update frequency
    target_update_freq: usize,
    /// Steps counter
    steps: usize,
    /// Number of actions
    n_actions: usize,
    /// State size
    state_size: usize,
    /// Last state for learning
    last_state: Option<Vec<f64>>,
    /// Last action for learning
    last_action: Option<usize>,
    /// Trade sizes for each action
    action_sizes: Vec<f64>,
}

impl RLAgent {
    /// Create a new RL agent
    pub fn new(state_size: usize, hidden_size: usize) -> Self {
        // Actions: Hold, Small Buy, Medium Buy, Large Buy, Small Sell, Medium Sell, Large Sell
        let n_actions = 7;
        let action_sizes = vec![0.0, 0.1, 0.5, 1.0, 0.1, 0.5, 1.0];

        Self {
            id: 0,
            network: SimpleNetwork::new(state_size, hidden_size, n_actions),
            target_network: SimpleNetwork::new(state_size, hidden_size, n_actions),
            replay_buffer: ReplayBuffer::new(10000),
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            gamma: 0.99,
            learning_rate: 0.001,
            batch_size: 64,
            target_update_freq: 100,
            steps: 0,
            n_actions,
            state_size,
            last_state: None,
            last_action: None,
            action_sizes,
        }
    }

    /// Convert observation to state vector
    fn observation_to_state(&self, obs: &Observation, agent_state: &AgentState) -> Vec<f64> {
        vec![
            obs.mid_price / 50000.0, // Normalized price
            obs.spread / obs.mid_price,
            obs.imbalance,
            obs.volatility,
            agent_state.inventory / 10.0,
            agent_state.momentum(10),
            agent_state.momentum(20),
            agent_state.volatility(10),
        ]
    }

    /// Convert action index to AgentAction
    fn index_to_action(&self, action_idx: usize, state: &AgentState, price: f64) -> AgentAction {
        let size = self.action_sizes.get(action_idx).copied().unwrap_or(0.0);

        match action_idx {
            0 => AgentAction::Hold,
            1 | 2 | 3 => {
                // Buy actions
                let quantity = size;
                if state.cash >= quantity * price {
                    AgentAction::MarketBuy { quantity }
                } else {
                    AgentAction::Hold
                }
            }
            4 | 5 | 6 => {
                // Sell actions
                let quantity = size.min(state.inventory);
                if quantity > 0.0 {
                    AgentAction::MarketSell { quantity }
                } else {
                    AgentAction::Hold
                }
            }
            _ => AgentAction::Hold,
        }
    }

    /// Select action using epsilon-greedy
    fn select_action(&self, state: &[f64]) -> usize {
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < self.epsilon {
            // Random action
            rng.gen_range(0..self.n_actions)
        } else {
            // Greedy action
            let q_values = self.network.q_values(state);
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// Train on a batch of experiences
    fn train_step(&mut self) {
        if self.replay_buffer.len() < self.batch_size {
            return;
        }

        let batch = self.replay_buffer.sample(self.batch_size);

        for experience in batch {
            // Compute target Q-value
            let target_q = if experience.done {
                experience.reward
            } else {
                let next_q_values = self.target_network.q_values(&experience.next_state);
                let max_next_q = next_q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                experience.reward + self.gamma * max_next_q
            };

            // Compute current Q-value
            let current_q_values = self.network.q_values(&experience.state);
            let current_q = current_q_values[experience.action];

            // Compute TD error
            let td_error = target_q - current_q;

            // Simple update (in practice, would use proper backprop)
            // This is a simplified version for demonstration
            let input = Array1::from_vec(experience.state.clone());
            let hidden = input.dot(&self.network.weights1) + &self.network.bias1;
            let hidden_activated = hidden.mapv(|x| x.max(0.0));

            // Update output layer
            for i in 0..self.network.weights2.nrows() {
                if hidden_activated[i] > 0.0 {
                    self.network.weights2[[i, experience.action]] +=
                        self.learning_rate * td_error * hidden_activated[i];
                }
            }
            self.network.bias2[experience.action] += self.learning_rate * td_error;
        }

        // Update epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }
}

impl Agent for RLAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn set_id(&mut self, id: AgentId) {
        self.id = id;
    }

    fn name(&self) -> &str {
        "RLAgent"
    }

    fn act(&mut self, observation: &Observation, state: &AgentState) -> AgentAction {
        let current_state = self.observation_to_state(observation, state);
        let action_idx = self.select_action(&current_state);

        // Store for learning
        self.last_state = Some(current_state);
        self.last_action = Some(action_idx);

        self.index_to_action(action_idx, state, observation.mid_price)
    }

    fn learn(&mut self, reward: f64, next_observation: &Observation, done: bool) {
        let dummy_state = AgentState::new(100_000.0);

        if let (Some(state), Some(action)) = (self.last_state.take(), self.last_action.take()) {
            let next_state = self.observation_to_state(next_observation, &dummy_state);

            let experience = Experience {
                state,
                action,
                reward,
                next_state,
                done,
            };

            self.replay_buffer.push(experience);
            self.train_step();

            self.steps += 1;

            // Update target network
            if self.steps % self.target_update_freq == 0 {
                self.target_network.copy_from(&self.network);
            }
        }
    }

    fn reset(&mut self) {
        self.last_state = None;
        self.last_action = None;
    }

    fn clone_box(&self) -> Box<dyn Agent> {
        Box::new(self.clone())
    }

    fn params_string(&self) -> String {
        format!(
            "epsilon={:.3}, gamma={:.3}, lr={:.5}",
            self.epsilon, self.gamma, self.learning_rate
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_observation(price: f64) -> Observation {
        Observation {
            mid_price: price,
            spread: 1.0,
            bid_depth: 100.0,
            ask_depth: 100.0,
            imbalance: 0.0,
            last_price: price,
            volume: 1000.0,
            volatility: 0.02,
            timestamp: 0,
        }
    }

    #[test]
    fn test_rl_agent_creation() {
        let agent = RLAgent::new(8, 64);
        assert_eq!(agent.n_actions, 7);
        assert_eq!(agent.state_size, 8);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);

        for i in 0..50 {
            buffer.push(Experience {
                state: vec![i as f64],
                action: 0,
                reward: 1.0,
                next_state: vec![i as f64 + 1.0],
                done: false,
            });
        }

        assert_eq!(buffer.len(), 50);

        let samples = buffer.sample(10);
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn test_simple_network() {
        let network = SimpleNetwork::new(8, 32, 7);
        let input = Array1::from_vec(vec![1.0; 8]);
        let output = network.forward(&input);

        assert_eq!(output.len(), 7);
    }

    #[test]
    fn test_rl_agent_act() {
        let mut agent = RLAgent::new(8, 64);
        let state = AgentState::new(100_000.0);
        let obs = create_test_observation(50000.0);

        let action = agent.act(&obs, &state);

        // Should return some action
        println!("Action: {:?}", action);
    }
}
