use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// =============================================================================
// Constants
// =============================================================================

/// Number of atoms in the categorical distribution
pub const NUM_ATOMS: usize = 51;
/// Minimum value of the support
pub const V_MIN: f64 = -10.0;
/// Maximum value of the support
pub const V_MAX: f64 = 10.0;
/// Spacing between atoms
pub const DZ: f64 = (V_MAX - V_MIN) / (NUM_ATOMS as f64 - 1.0);

/// Number of actions: Buy(0), Sell(1), Hold(2)
pub const NUM_ACTIONS: usize = 3;

// =============================================================================
// Support (atom values)
// =============================================================================

/// Generate the fixed support vector of 51 atoms
pub fn generate_support() -> Array1<f64> {
    Array1::from_iter((0..NUM_ATOMS).map(|i| V_MIN + i as f64 * DZ))
}

// =============================================================================
// Softmax
// =============================================================================

/// Compute softmax over a 1D array
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = logits.mapv(|x| (x - max_val).exp());
    let sum: f64 = exp_vals.sum();
    if sum == 0.0 {
        Array1::from_elem(logits.len(), 1.0 / logits.len() as f64)
    } else {
        exp_vals / sum
    }
}

// =============================================================================
// Dense Layer
// =============================================================================

/// A simple fully-connected (dense) layer
#[derive(Clone)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub use_relu: bool,
}

impl DenseLayer {
    /// Create a new dense layer with random initialization
    pub fn new(input_size: usize, output_size: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_size as f64).sqrt();
        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_size);
        Self {
            weights,
            biases,
            use_relu,
        }
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let output = input.dot(&self.weights) + &self.biases;
        if self.use_relu {
            output.mapv(|x| x.max(0.0))
        } else {
            output
        }
    }
}

// =============================================================================
// C51 Network
// =============================================================================

/// C51 neural network: state -> distribution over atoms for each action
#[derive(Clone)]
pub struct C51Network {
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
    pub output_layer: DenseLayer,
    pub num_actions: usize,
    pub num_atoms: usize,
    pub support: Array1<f64>,
    pub learning_rate: f64,
}

impl C51Network {
    /// Create a new C51 network
    pub fn new(state_size: usize, num_actions: usize, num_atoms: usize) -> Self {
        let hidden_size = 128;
        Self {
            layer1: DenseLayer::new(state_size, hidden_size, true),
            layer2: DenseLayer::new(hidden_size, hidden_size, true),
            output_layer: DenseLayer::new(hidden_size, num_actions * num_atoms, false),
            num_actions,
            num_atoms,
            support: generate_support(),
            learning_rate: 0.001,
        }
    }

    /// Forward pass: returns probability distributions for all actions
    /// Shape: [num_actions, num_atoms]
    pub fn forward(&self, state: &Array1<f64>) -> Array2<f64> {
        let h1 = self.layer1.forward(state);
        let h2 = self.layer2.forward(&h1);
        let logits = self.output_layer.forward(&h2);

        // Reshape to [num_actions, num_atoms] and apply softmax per action
        let mut distributions = Array2::zeros((self.num_actions, self.num_atoms));
        for a in 0..self.num_actions {
            let start = a * self.num_atoms;
            let end = start + self.num_atoms;
            let action_logits = Array1::from_iter(logits.iter().skip(start).take(end - start).cloned());
            let probs = softmax(&action_logits);
            distributions.row_mut(a).assign(&probs);
        }
        distributions
    }

    /// Get Q-values by computing expected value of each action's distribution
    pub fn q_values(&self, state: &Array1<f64>) -> Array1<f64> {
        let distributions = self.forward(state);
        let mut q = Array1::zeros(self.num_actions);
        for a in 0..self.num_actions {
            q[a] = distributions.row(a).dot(&self.support);
        }
        q
    }

    /// Select the best action (greedy)
    pub fn best_action(&self, state: &Array1<f64>) -> usize {
        let q = self.q_values(state);
        q.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Apply simple gradient update to weights (simplified SGD)
    pub fn update_weights(&mut self, gradients: &WeightGradients) {
        let lr = self.learning_rate;
        self.layer1.weights = &self.layer1.weights - &(&gradients.layer1_w * lr);
        self.layer1.biases = &self.layer1.biases - &(&gradients.layer1_b * lr);
        self.layer2.weights = &self.layer2.weights - &(&gradients.layer2_w * lr);
        self.layer2.biases = &self.layer2.biases - &(&gradients.layer2_b * lr);
        self.output_layer.weights = &self.output_layer.weights - &(&gradients.output_w * lr);
        self.output_layer.biases = &self.output_layer.biases - &(&gradients.output_b * lr);
    }
}

/// Gradients for all layers
pub struct WeightGradients {
    pub layer1_w: Array2<f64>,
    pub layer1_b: Array1<f64>,
    pub layer2_w: Array2<f64>,
    pub layer2_b: Array1<f64>,
    pub output_w: Array2<f64>,
    pub output_b: Array1<f64>,
}

// =============================================================================
// Categorical Projection
// =============================================================================

/// Project a target distribution onto the fixed support
pub struct CategoricalProjection {
    pub v_min: f64,
    pub v_max: f64,
    pub num_atoms: usize,
    pub dz: f64,
    pub support: Array1<f64>,
}

impl CategoricalProjection {
    pub fn new() -> Self {
        Self {
            v_min: V_MIN,
            v_max: V_MAX,
            num_atoms: NUM_ATOMS,
            dz: DZ,
            support: generate_support(),
        }
    }

    /// Project the target distribution after Bellman update
    /// Given reward, gamma, done flag, and next-state distribution,
    /// returns the projected distribution on the fixed support.
    pub fn project(
        &self,
        reward: f64,
        gamma: f64,
        done: bool,
        next_dist: &Array1<f64>,
    ) -> Array1<f64> {
        let mut projected = Array1::zeros(self.num_atoms);
        let gamma_effective = if done { 0.0 } else { gamma };

        for j in 0..self.num_atoms {
            // Compute the projected atom
            let tz = (reward + gamma_effective * self.support[j])
                .clamp(self.v_min, self.v_max);

            // Find position on support
            let b = (tz - self.v_min) / self.dz;
            let l = b.floor() as usize;
            let u = (b.ceil() as usize).min(self.num_atoms - 1);

            if l == u {
                projected[l] += next_dist[j];
            } else {
                // Distribute probability using linear interpolation
                projected[l] += next_dist[j] * (u as f64 - b);
                projected[u] += next_dist[j] * (b - l as f64);
            }
        }
        projected
    }
}

impl Default for CategoricalProjection {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// KL Divergence / Cross-Entropy Loss
// =============================================================================

/// Compute cross-entropy loss between target distribution m and predicted distribution p
/// L = -sum(m_i * log(p_i))
pub fn cross_entropy_loss(target: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let epsilon = 1e-8;
    -target
        .iter()
        .zip(predicted.iter())
        .map(|(m, p)| m * (p + epsilon).ln())
        .sum::<f64>()
}

/// Compute KL divergence: KL(target || predicted)
pub fn kl_divergence(target: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let epsilon = 1e-8;
    target
        .iter()
        .zip(predicted.iter())
        .map(|(m, p)| {
            if *m > epsilon {
                m * ((m + epsilon) / (p + epsilon)).ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

// =============================================================================
// Experience Replay Buffer
// =============================================================================

/// A single transition
#[derive(Clone)]
pub struct Transition {
    pub state: Array1<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Array1<f64>,
    pub done: bool,
}

/// Experience replay buffer with uniform sampling
pub struct ReplayBuffer {
    pub buffer: Vec<Transition>,
    pub capacity: usize,
    pub position: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Transition> {
        let mut rng = rand::thread_rng();
        let len = self.buffer.len();
        (0..batch_size)
            .map(|_| {
                let idx = rng.gen_range(0..len);
                self.buffer[idx].clone()
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// =============================================================================
// C51 Agent
// =============================================================================

/// C51 agent with epsilon-greedy exploration
pub struct C51Agent {
    pub network: C51Network,
    pub target_network: C51Network,
    pub replay_buffer: ReplayBuffer,
    pub projection: CategoricalProjection,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_min: f64,
    pub epsilon_decay: f64,
    pub batch_size: usize,
    pub target_update_freq: usize,
    pub step_count: usize,
}

impl C51Agent {
    pub fn new(state_size: usize) -> Self {
        let network = C51Network::new(state_size, NUM_ACTIONS, NUM_ATOMS);
        let target_network = network.clone();
        Self {
            network,
            target_network,
            replay_buffer: ReplayBuffer::new(10000),
            projection: CategoricalProjection::new(),
            gamma: 0.99,
            epsilon: 1.0,
            epsilon_min: 0.01,
            epsilon_decay: 0.995,
            batch_size: 32,
            target_update_freq: 100,
            step_count: 0,
        }
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: &Array1<f64>) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..NUM_ACTIONS)
        } else {
            self.network.best_action(state)
        }
    }

    /// Store a transition in the replay buffer
    pub fn store_transition(&mut self, transition: Transition) {
        self.replay_buffer.push(transition);
    }

    /// Train the agent on a batch from the replay buffer
    pub fn train_step(&mut self) -> Option<f64> {
        if self.replay_buffer.len() < self.batch_size {
            return None;
        }

        let batch = self.replay_buffer.sample(self.batch_size);
        let mut total_loss = 0.0;

        // Compute numerical gradients and update (simplified training)
        for transition in &batch {
            // Get current distribution for the taken action
            let current_dists = self.network.forward(&transition.state);
            let current_dist = current_dists.row(transition.action).to_owned();

            // Get next-state distribution from target network
            let next_q = self.target_network.q_values(&transition.next_state);
            let best_next_action = next_q
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let next_dists = self.target_network.forward(&transition.next_state);
            let next_dist = next_dists.row(best_next_action).to_owned();

            // Project target distribution
            let target_dist = self.projection.project(
                transition.reward,
                self.gamma,
                transition.done,
                &next_dist,
            );

            // Compute loss
            let loss = cross_entropy_loss(&target_dist, &current_dist);
            total_loss += loss;

            // Simplified weight perturbation update
            let perturbation_scale = 0.001;
            let mut rng = rand::thread_rng();
            self.perturb_weights(&mut rng, perturbation_scale, &transition.state, transition.action, &target_dist);
        }

        self.step_count += 1;

        // Update target network periodically
        if self.step_count % self.target_update_freq == 0 {
            self.target_network = self.network.clone();
        }

        // Decay epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);

        Some(total_loss / batch.len() as f64)
    }

    /// Simple weight perturbation for training (evolutionary-style update)
    fn perturb_weights(
        &mut self,
        rng: &mut impl Rng,
        scale: f64,
        state: &Array1<f64>,
        action: usize,
        target_dist: &Array1<f64>,
    ) {
        // Get current loss
        let current_dist = self.network.forward(state);
        let current_loss = cross_entropy_loss(target_dist, &current_dist.row(action).to_owned());

        // Perturb output layer biases (most impactful for distribution shaping)
        let start = action * self.network.num_atoms;
        for i in 0..self.network.num_atoms {
            let idx = start + i;
            if idx < self.network.output_layer.biases.len() {
                let original = self.network.output_layer.biases[idx];

                // Try positive perturbation
                let delta: f64 = rng.gen_range(-scale..scale);
                self.network.output_layer.biases[idx] = original + delta;
                let new_dist = self.network.forward(state);
                let new_loss = cross_entropy_loss(target_dist, &new_dist.row(action).to_owned());

                // Keep if improved, otherwise revert
                if new_loss >= current_loss {
                    self.network.output_layer.biases[idx] = original;
                }
            }
        }
    }

    /// Get the return distribution for a given state and action
    pub fn get_distribution(&self, state: &Array1<f64>, action: usize) -> Array1<f64> {
        let dists = self.network.forward(state);
        dists.row(action).to_owned()
    }

    /// Compute VaR at a given confidence level
    pub fn compute_var(&self, state: &Array1<f64>, action: usize, alpha: f64) -> f64 {
        let dist = self.get_distribution(state, action);
        let support = generate_support();
        let mut cumulative = 0.0;
        for i in 0..NUM_ATOMS {
            cumulative += dist[i];
            if cumulative >= alpha {
                return support[i];
            }
        }
        support[NUM_ATOMS - 1]
    }

    /// Compute CVaR (Expected Shortfall) at a given confidence level
    pub fn compute_cvar(&self, state: &Array1<f64>, action: usize, alpha: f64) -> f64 {
        let dist = self.get_distribution(state, action);
        let support = generate_support();
        let mut cumulative = 0.0;
        let mut cvar = 0.0;
        for i in 0..NUM_ATOMS {
            if cumulative + dist[i] >= alpha {
                let remaining = alpha - cumulative;
                cvar += remaining * support[i];
                break;
            }
            cvar += dist[i] * support[i];
            cumulative += dist[i];
        }
        cvar / alpha
    }
}

// =============================================================================
// Trading Environment
// =============================================================================

/// Simple trading environment using historical price data
pub struct TradingEnvironment {
    pub prices: Vec<f64>,
    pub current_step: usize,
    pub position: i32, // -1: short, 0: flat, 1: long
    pub entry_price: f64,
    pub window_size: usize,
    pub fee_rate: f64,
}

impl TradingEnvironment {
    pub fn new(prices: Vec<f64>, window_size: usize) -> Self {
        Self {
            prices,
            current_step: window_size,
            position: 0,
            entry_price: 0.0,
            window_size,
            fee_rate: 0.00075, // 0.075% Bybit futures fee
        }
    }

    /// Get state features from current market data
    pub fn get_state(&self) -> Array1<f64> {
        let mut features = Vec::new();
        let idx = self.current_step;

        // Log returns at different timeframes
        for &lookback in &[1, 5, 10, 20] {
            if idx >= lookback && idx < self.prices.len() {
                let ret = (self.prices[idx] / self.prices[idx - lookback]).ln();
                features.push(ret);
            } else {
                features.push(0.0);
            }
        }

        // Simple moving average ratio
        if idx >= 20 {
            let sma20: f64 = self.prices[idx - 20..idx].iter().sum::<f64>() / 20.0;
            features.push(self.prices[idx] / sma20 - 1.0);
        } else {
            features.push(0.0);
        }

        // Volatility (std dev of recent returns)
        if idx >= 20 {
            let returns: Vec<f64> = (1..20)
                .map(|i| (self.prices[idx - i + 1] / self.prices[idx - i]).ln())
                .collect();
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
            features.push(var.sqrt());
        } else {
            features.push(0.0);
        }

        // Current position as feature
        features.push(self.position as f64);

        // Unrealized PnL if in position
        if self.position != 0 && self.entry_price > 0.0 {
            let pnl = self.position as f64 * (self.prices[idx] / self.entry_price - 1.0);
            features.push(pnl);
        } else {
            features.push(0.0);
        }

        Array1::from_vec(features)
    }

    /// Get the state size
    pub fn state_size(&self) -> usize {
        8
    }

    /// Take action and return (reward, done)
    pub fn step(&mut self, action: usize) -> (f64, bool) {
        let current_price = self.prices[self.current_step];
        let mut reward = 0.0;

        match action {
            0 => {
                // Buy
                if self.position <= 0 {
                    if self.position == -1 {
                        // Close short position
                        reward = -(current_price / self.entry_price - 1.0) - self.fee_rate;
                    }
                    self.position = 1;
                    self.entry_price = current_price;
                    reward -= self.fee_rate; // Entry fee
                }
            }
            1 => {
                // Sell
                if self.position >= 0 {
                    if self.position == 1 {
                        // Close long position
                        reward = (current_price / self.entry_price - 1.0) - self.fee_rate;
                    }
                    self.position = -1;
                    self.entry_price = current_price;
                    reward -= self.fee_rate; // Entry fee
                }
            }
            _ => {
                // Hold
                if self.position == 1 {
                    let next_idx = (self.current_step + 1).min(self.prices.len() - 1);
                    reward = (self.prices[next_idx] / current_price).ln();
                } else if self.position == -1 {
                    let next_idx = (self.current_step + 1).min(self.prices.len() - 1);
                    reward = -(self.prices[next_idx] / current_price).ln();
                }
            }
        }

        self.current_step += 1;
        let done = self.current_step >= self.prices.len() - 1;

        // Scale reward to match support range
        reward *= 100.0;

        (reward, done)
    }

    /// Reset the environment
    pub fn reset(&mut self) {
        self.current_step = self.window_size;
        self.position = 0;
        self.entry_price = 0.0;
    }
}

// =============================================================================
// Bybit API Client
// =============================================================================

/// Kline (candlestick) data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structures
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// Client for Bybit V5 API
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline data from Bybit API (blocking)
    pub fn fetch_klines_blocking(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::blocking::Client::new();
        let response: BybitResponse = client.get(&url).send()?.json()?;

        let mut klines: Vec<Kline> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Kline {
                        timestamp: item[0].parse().unwrap_or(0),
                        open: item[1].parse().unwrap_or(0.0),
                        high: item[2].parse().unwrap_or(0.0),
                        low: item[3].parse().unwrap_or(0.0),
                        close: item[4].parse().unwrap_or(0.0),
                        volume: item[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse to get chronological order
        klines.reverse();
        Ok(klines)
    }

    /// Fetch kline data from Bybit API (async)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::Client::new();
        let response: BybitResponse = client.get(&url).send().await?.json().await?;

        let mut klines: Vec<Kline> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Kline {
                        timestamp: item[0].parse().unwrap_or(0),
                        open: item[1].parse().unwrap_or(0.0),
                        high: item[2].parse().unwrap_or(0.0),
                        low: item[3].parse().unwrap_or(0.0),
                        close: item[4].parse().unwrap_or(0.0),
                        volume: item[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        klines.reverse();
        Ok(klines)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract close prices from klines
pub fn extract_close_prices(klines: &[Kline]) -> Vec<f64> {
    klines.iter().map(|k| k.close).collect()
}

// =============================================================================
// Risk Metrics
// =============================================================================

/// Compute Value at Risk from a distribution
pub fn compute_var_from_dist(dist: &Array1<f64>, support: &Array1<f64>, alpha: f64) -> f64 {
    let mut cumulative = 0.0;
    for i in 0..dist.len() {
        cumulative += dist[i];
        if cumulative >= alpha {
            return support[i];
        }
    }
    support[support.len() - 1]
}

/// Compute Conditional Value at Risk from a distribution
pub fn compute_cvar_from_dist(dist: &Array1<f64>, support: &Array1<f64>, alpha: f64) -> f64 {
    let mut cumulative = 0.0;
    let mut cvar = 0.0;
    for i in 0..dist.len() {
        if cumulative + dist[i] >= alpha {
            let remaining = alpha - cumulative;
            cvar += remaining * support[i];
            break;
        }
        cvar += dist[i] * support[i];
        cumulative += dist[i];
    }
    if alpha > 0.0 {
        cvar / alpha
    } else {
        0.0
    }
}

// =============================================================================
// Standard DQN (for comparison)
// =============================================================================

/// Simple DQN network for comparison with C51
pub struct DQNNetwork {
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
    pub output_layer: DenseLayer,
}

impl DQNNetwork {
    pub fn new(state_size: usize, num_actions: usize) -> Self {
        Self {
            layer1: DenseLayer::new(state_size, 128, true),
            layer2: DenseLayer::new(128, 128, true),
            output_layer: DenseLayer::new(128, num_actions, false),
        }
    }

    pub fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let h1 = self.layer1.forward(state);
        let h2 = self.layer2.forward(&h1);
        self.output_layer.forward(&h2)
    }

    pub fn best_action(&self, state: &Array1<f64>) -> usize {
        let q = self.forward(state);
        q.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_support_generation() {
        let support = generate_support();
        assert_eq!(support.len(), NUM_ATOMS);
        assert!((support[0] - V_MIN).abs() < 1e-10);
        assert!((support[NUM_ATOMS - 1] - V_MAX).abs() < 1e-10);
        // Check uniform spacing
        for i in 1..NUM_ATOMS {
            let diff = support[i] - support[i - 1];
            assert!((diff - DZ).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        // Probabilities sum to 1
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        // All probabilities positive
        assert!(probs.iter().all(|&p| p > 0.0));
        // Highest logit = highest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let logits = Array1::from_vec(vec![1000.0, 1001.0, 1002.0]);
        let probs = softmax(&logits);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_c51_network_output_shape() {
        let state_size = 8;
        let network = C51Network::new(state_size, NUM_ACTIONS, NUM_ATOMS);
        let state = Array1::zeros(state_size);
        let dists = network.forward(&state);

        assert_eq!(dists.shape(), &[NUM_ACTIONS, NUM_ATOMS]);

        // Each action's distribution should sum to 1
        for a in 0..NUM_ACTIONS {
            let sum: f64 = dists.row(a).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Action {} distribution sums to {}, expected 1.0",
                a,
                sum
            );
        }
    }

    #[test]
    fn test_c51_q_values() {
        let state_size = 8;
        let network = C51Network::new(state_size, NUM_ACTIONS, NUM_ATOMS);
        let state = Array1::zeros(state_size);
        let q = network.q_values(&state);

        assert_eq!(q.len(), NUM_ACTIONS);
        // Q-values should be finite
        assert!(q.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_categorical_projection() {
        let proj = CategoricalProjection::new();

        // Create a simple target distribution (uniform)
        let next_dist = Array1::from_elem(NUM_ATOMS, 1.0 / NUM_ATOMS as f64);

        let projected = proj.project(0.0, 0.99, false, &next_dist);

        // Projected distribution should sum to 1
        assert!(
            (projected.sum() - 1.0).abs() < 1e-6,
            "Projected distribution sums to {}, expected 1.0",
            projected.sum()
        );
        // All probabilities should be non-negative
        assert!(projected.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_categorical_projection_terminal() {
        let proj = CategoricalProjection::new();
        let next_dist = Array1::from_elem(NUM_ATOMS, 1.0 / NUM_ATOMS as f64);

        // Terminal state: gamma=0, only reward matters
        let reward = 1.0;
        let projected = proj.project(reward, 0.99, true, &next_dist);

        assert!((projected.sum() - 1.0).abs() < 1e-6);
        // All mass should be concentrated around the reward atom
        let reward_atom_idx = ((reward - V_MIN) / DZ).round() as usize;
        // Check that most mass is near the reward
        let nearby_mass: f64 = projected
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i as i64 - reward_atom_idx as i64).unsigned_abs() <= 1)
            .map(|(_, p)| p)
            .sum();
        assert!(nearby_mass > 0.99, "Terminal projection: mass near reward = {}", nearby_mass);
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Identical distributions should have minimal loss
        let p = Array1::from_vec(vec![0.2, 0.3, 0.5]);
        let loss = cross_entropy_loss(&p, &p);
        // Cross-entropy of p with itself equals entropy of p
        let entropy: f64 = -p.iter().map(|x| x * x.ln()).sum::<f64>();
        assert!((loss - entropy).abs() < 1e-6);
    }

    #[test]
    fn test_kl_divergence() {
        // KL divergence of distribution with itself should be ~0
        let p = Array1::from_vec(vec![0.2, 0.3, 0.5]);
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-6, "KL(p||p) should be ~0, got {}", kl);

        // KL divergence is non-negative
        let q = Array1::from_vec(vec![0.1, 0.4, 0.5]);
        let kl = kl_divergence(&p, &q);
        assert!(kl >= -1e-10, "KL divergence should be non-negative, got {}", kl);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        assert!(buffer.is_empty());

        // Add transitions
        for i in 0..50 {
            buffer.push(Transition {
                state: Array1::from_elem(8, i as f64),
                action: i % 3,
                reward: i as f64 * 0.1,
                next_state: Array1::from_elem(8, (i + 1) as f64),
                done: false,
            });
        }

        assert_eq!(buffer.len(), 50);

        // Sample batch
        let batch = buffer.sample(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_replay_buffer_overflow() {
        let mut buffer = ReplayBuffer::new(5);

        // Add more than capacity
        for i in 0..10 {
            buffer.push(Transition {
                state: Array1::from_elem(4, i as f64),
                action: 0,
                reward: i as f64,
                next_state: Array1::from_elem(4, 0.0),
                done: false,
            });
        }

        // Buffer should not exceed capacity
        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn test_trading_environment() {
        // Create simple price series: 100, 101, 102, ..., 150
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let mut env = TradingEnvironment::new(prices, 20);

        let state = env.get_state();
        assert_eq!(state.len(), env.state_size());

        // Take a buy action
        let (reward, done) = env.step(0);
        assert!(reward.is_finite());
        assert!(!done);
    }

    #[test]
    fn test_c51_agent_action_selection() {
        let agent = C51Agent::new(8);
        let state = Array1::zeros(8);
        let action = agent.select_action(&state);
        assert!(action < NUM_ACTIONS);
    }

    #[test]
    fn test_risk_metrics() {
        // Create a simple distribution
        let mut dist = Array1::zeros(NUM_ATOMS);
        // Put all mass at the middle atom
        dist[NUM_ATOMS / 2] = 1.0;
        let support = generate_support();

        let var = compute_var_from_dist(&dist, &support, 0.05);
        assert!((var - support[NUM_ATOMS / 2]).abs() < 1e-6);

        let cvar = compute_cvar_from_dist(&dist, &support, 0.05);
        assert!(cvar.is_finite());
    }

    #[test]
    fn test_dqn_network() {
        let network = DQNNetwork::new(8, NUM_ACTIONS);
        let state = Array1::zeros(8);
        let q = network.forward(&state);
        assert_eq!(q.len(), NUM_ACTIONS);
        assert!(q.iter().all(|v| v.is_finite()));

        let action = network.best_action(&state);
        assert!(action < NUM_ACTIONS);
    }
}
