//! Hierarchical Reinforcement Learning for Trading
//!
//! This library implements a two-level hierarchical RL system:
//! - High-level policy (Manager): Detects market regimes and sets allocation goals
//! - Low-level policy (Worker): Executes trades to achieve the manager's goals
//!
//! Architecture based on Feudal Networks (Vezhnevets et al., 2017) and HAC (Levy et al., 2019).

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================
// Market Regime
// ============================================================

/// Detected market regime from the high-level policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
}

impl MarketRegime {
    /// Convert regime to a numeric index.
    pub fn as_index(&self) -> usize {
        match self {
            MarketRegime::Bull => 0,
            MarketRegime::Bear => 1,
            MarketRegime::Sideways => 2,
        }
    }

    /// Create regime from index.
    pub fn from_index(idx: usize) -> Self {
        match idx % 3 {
            0 => MarketRegime::Bull,
            1 => MarketRegime::Bear,
            _ => MarketRegime::Sideways,
        }
    }
}

// ============================================================
// Linear Layer
// ============================================================

/// A simple linear layer: output = input * weights + bias, with optional ReLU.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub use_relu: bool,
}

impl LinearLayer {
    /// Create a new linear layer with random initialization.
    pub fn new(input_dim: usize, output_dim: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_dim as f64).sqrt();
        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);
        Self {
            weights,
            bias,
            use_relu,
        }
    }

    /// Create a linear layer with specific weights (for testing).
    pub fn with_weights(weights: Array2<f64>, bias: Array1<f64>, use_relu: bool) -> Self {
        Self {
            weights,
            bias,
            use_relu,
        }
    }

    /// Forward pass.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let out = input.dot(&self.weights) + &self.bias;
        if self.use_relu {
            out.mapv(|x| x.max(0.0))
        } else {
            out
        }
    }

    /// Update weights with simple gradient step.
    pub fn update(&mut self, weight_grads: &Array2<f64>, bias_grads: &Array1<f64>, lr: f64) {
        self.weights = &self.weights - &(weight_grads * lr);
        self.bias = &self.bias - &(bias_grads * lr);
    }
}

// ============================================================
// High-Level Policy (Manager)
// ============================================================

/// The Manager policy operates at a coarser timescale.
/// It observes aggregated market features and outputs:
/// 1. A regime classification (bull/bear/sideways)
/// 2. A goal vector for the worker (4D: direction, magnitude, vol_tolerance, time_horizon)
#[derive(Debug, Clone)]
pub struct HighLevelPolicy {
    pub layer1: LinearLayer,
    pub layer2: LinearLayer,
    pub regime_head: LinearLayer,
    pub goal_head: LinearLayer,
    pub decision_interval: usize,
    pub goal_dim: usize,
    pub learning_rate: f64,
}

impl HighLevelPolicy {
    /// Create a new high-level policy.
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of the aggregated state features
    /// * `hidden_dim` - Hidden layer size
    /// * `goal_dim` - Dimension of goal vectors (default 4)
    /// * `decision_interval` - How many low-level steps per high-level decision
    pub fn new(
        state_dim: usize,
        hidden_dim: usize,
        goal_dim: usize,
        decision_interval: usize,
    ) -> Self {
        Self {
            layer1: LinearLayer::new(state_dim, hidden_dim, true),
            layer2: LinearLayer::new(hidden_dim, hidden_dim, true),
            regime_head: LinearLayer::new(hidden_dim, 3, false),
            goal_head: LinearLayer::new(hidden_dim, goal_dim, false),
            decision_interval,
            goal_dim,
            learning_rate: 0.001,
        }
    }

    /// Forward pass: returns (regime_logits, goal_vector).
    pub fn forward(&self, state: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let h1 = self.layer1.forward(state);
        let h2 = self.layer2.forward(&h1);
        let regime_logits = self.regime_head.forward(&h2);
        let goal = self.goal_head.forward(&h2);
        // Normalize goal to unit vector (direction in goal space)
        let norm = goal.dot(&goal).sqrt().max(1e-8);
        let goal_normalized = goal / norm;
        (regime_logits, goal_normalized)
    }

    /// Get the detected regime from logits.
    pub fn detect_regime(&self, state: &Array1<f64>) -> MarketRegime {
        let (logits, _) = self.forward(state);
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(2);
        MarketRegime::from_index(max_idx)
    }

    /// Get goal vector for the worker.
    pub fn get_goal(&self, state: &Array1<f64>) -> Array1<f64> {
        let (_, goal) = self.forward(state);
        goal
    }

    /// Simple policy gradient update for the manager.
    pub fn update(&mut self, states: &[Array1<f64>], rewards: &[f64]) {
        if states.is_empty() || rewards.is_empty() {
            return;
        }
        let mut rng = rand::thread_rng();
        let mean_reward: f64 = rewards.iter().sum::<f64>() / rewards.len() as f64;
        // Simple perturbation-based update
        for reward in rewards {
            let advantage = reward - mean_reward;
            let perturbation = rng.gen_range(-0.01..0.01) * advantage;
            self.layer1.bias.mapv_inplace(|x| x + perturbation * self.learning_rate);
            self.layer2.bias.mapv_inplace(|x| x + perturbation * self.learning_rate);
        }
    }
}

// ============================================================
// Low-Level Policy (Worker)
// ============================================================

/// The Worker policy operates at every time step.
/// It takes the current state AND the manager's goal vector,
/// and outputs a trading action (position size in [-1, 1]).
#[derive(Debug, Clone)]
pub struct LowLevelPolicy {
    pub layer1: LinearLayer,
    pub layer2: LinearLayer,
    pub action_head: LinearLayer,
    pub learning_rate: f64,
}

impl LowLevelPolicy {
    /// Create a new low-level policy.
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of the per-step state
    /// * `goal_dim` - Dimension of goal vector from manager
    /// * `hidden_dim` - Hidden layer size
    pub fn new(state_dim: usize, goal_dim: usize, hidden_dim: usize) -> Self {
        let input_dim = state_dim + goal_dim;
        Self {
            layer1: LinearLayer::new(input_dim, hidden_dim, true),
            layer2: LinearLayer::new(hidden_dim, hidden_dim, true),
            action_head: LinearLayer::new(hidden_dim, 1, false),
            learning_rate: 0.001,
        }
    }

    /// Forward pass: returns action in [-1, 1].
    pub fn forward(&self, state: &Array1<f64>, goal: &Array1<f64>) -> f64 {
        let mut input = Array1::zeros(state.len() + goal.len());
        input.slice_mut(ndarray::s![..state.len()]).assign(state);
        input.slice_mut(ndarray::s![state.len()..]).assign(goal);
        let h1 = self.layer1.forward(&input);
        let h2 = self.layer2.forward(&h1);
        let action_raw = self.action_head.forward(&h2);
        // Tanh to bound action in [-1, 1]
        action_raw[0].tanh()
    }

    /// Compute intrinsic reward based on cosine similarity between
    /// state transition and the manager's goal vector.
    pub fn compute_intrinsic_reward(
        state_before: &Array1<f64>,
        state_after: &Array1<f64>,
        goal: &Array1<f64>,
    ) -> f64 {
        let transition = state_after - state_before;
        // Use only the first goal_dim dimensions of transition
        let t_len = transition.len().min(goal.len());
        let t_slice = transition.slice(ndarray::s![..t_len]);
        let g_slice = goal.slice(ndarray::s![..t_len]);

        let dot = t_slice.dot(&g_slice);
        let t_norm = t_slice.dot(&t_slice).sqrt().max(1e-8);
        let g_norm = g_slice.dot(&g_slice).sqrt().max(1e-8);
        dot / (t_norm * g_norm)
    }

    /// Simple policy gradient update for the worker.
    pub fn update(
        &mut self,
        _states: &[Array1<f64>],
        _goals: &[Array1<f64>],
        rewards: &[f64],
    ) {
        if rewards.is_empty() {
            return;
        }
        let mut rng = rand::thread_rng();
        let mean_reward: f64 = rewards.iter().sum::<f64>() / rewards.len() as f64;
        for reward in rewards {
            let advantage = reward - mean_reward;
            let perturbation = rng.gen_range(-0.01..0.01) * advantage;
            self.layer1.bias.mapv_inplace(|x| x + perturbation * self.learning_rate);
            self.layer2.bias.mapv_inplace(|x| x + perturbation * self.learning_rate);
        }
    }
}

// ============================================================
// Feudal Network
// ============================================================

/// The FeudalNetwork combines a Manager and Worker into a coherent
/// hierarchical architecture. The manager sets goals every `decision_interval`
/// steps; the worker acts at every step to achieve those goals.
#[derive(Debug, Clone)]
pub struct FeudalNetwork {
    pub manager: HighLevelPolicy,
    pub worker: LowLevelPolicy,
    pub current_goal: Option<Array1<f64>>,
    pub current_regime: MarketRegime,
    pub step_count: usize,
}

impl FeudalNetwork {
    /// Create a new feudal network.
    pub fn new(
        state_dim: usize,
        hidden_dim: usize,
        goal_dim: usize,
        decision_interval: usize,
    ) -> Self {
        Self {
            manager: HighLevelPolicy::new(state_dim, hidden_dim, goal_dim, decision_interval),
            worker: LowLevelPolicy::new(state_dim, goal_dim, hidden_dim),
            current_goal: None,
            current_regime: MarketRegime::Sideways,
            step_count: 0,
        }
    }

    /// Decide on an action given the current state.
    /// Updates the manager's goal if at a decision boundary.
    pub fn act(&mut self, state: &Array1<f64>) -> (f64, MarketRegime) {
        // Manager decides at intervals
        if self.step_count % self.manager.decision_interval == 0 || self.current_goal.is_none() {
            self.current_regime = self.manager.detect_regime(state);
            self.current_goal = Some(self.manager.get_goal(state));
        }
        let goal = self.current_goal.as_ref().unwrap();
        let action = self.worker.forward(state, goal);
        self.step_count += 1;
        (action, self.current_regime)
    }

    /// Reset the network state for a new episode.
    pub fn reset(&mut self) {
        self.current_goal = None;
        self.current_regime = MarketRegime::Sideways;
        self.step_count = 0;
    }
}

// ============================================================
// Hierarchical Agent
// ============================================================

/// Experience tuple for the replay buffer.
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Array1<f64>,
    pub action: f64,
    pub reward: f64,
    pub next_state: Array1<f64>,
    pub goal: Array1<f64>,
    pub intrinsic_reward: f64,
}

/// Top-level agent that orchestrates training and inference.
#[derive(Debug, Clone)]
pub struct HierarchicalAgent {
    pub network: FeudalNetwork,
    pub replay_buffer: Vec<Experience>,
    pub buffer_capacity: usize,
    pub batch_size: usize,
    pub gamma: f64,
    pub total_extrinsic_reward: f64,
    pub episode_count: usize,
}

impl HierarchicalAgent {
    /// Create a new hierarchical agent.
    pub fn new(
        state_dim: usize,
        hidden_dim: usize,
        goal_dim: usize,
        decision_interval: usize,
    ) -> Self {
        Self {
            network: FeudalNetwork::new(state_dim, hidden_dim, goal_dim, decision_interval),
            replay_buffer: Vec::new(),
            buffer_capacity: 10000,
            batch_size: 64,
            gamma: 0.99,
            total_extrinsic_reward: 0.0,
            episode_count: 0,
        }
    }

    /// Select an action for the given state.
    pub fn act(&mut self, state: &Array1<f64>) -> (f64, MarketRegime) {
        self.network.act(state)
    }

    /// Store an experience in the replay buffer.
    pub fn store_experience(&mut self, exp: Experience) {
        if self.replay_buffer.len() >= self.buffer_capacity {
            self.replay_buffer.remove(0);
        }
        self.replay_buffer.push(exp);
    }

    /// Train both manager and worker from the replay buffer.
    pub fn train(&mut self) -> (f64, f64) {
        if self.replay_buffer.len() < self.batch_size {
            return (0.0, 0.0);
        }

        let mut rng = rand::thread_rng();
        let mut manager_states = Vec::new();
        let mut manager_rewards = Vec::new();
        let mut worker_states = Vec::new();
        let mut worker_goals = Vec::new();
        let mut worker_rewards = Vec::new();

        // Sample a batch
        for _ in 0..self.batch_size {
            let idx = rng.gen_range(0..self.replay_buffer.len());
            let exp = &self.replay_buffer[idx];
            manager_states.push(exp.state.clone());
            manager_rewards.push(exp.reward);
            worker_states.push(exp.state.clone());
            worker_goals.push(exp.goal.clone());
            // Worker uses combined reward: extrinsic + intrinsic
            worker_rewards.push(exp.reward + exp.intrinsic_reward);
        }

        let manager_avg = manager_rewards.iter().sum::<f64>() / manager_rewards.len() as f64;
        let worker_avg = worker_rewards.iter().sum::<f64>() / worker_rewards.len() as f64;

        // Update both policies
        self.network
            .manager
            .update(&manager_states, &manager_rewards);
        self.network
            .worker
            .update(&worker_states, &worker_goals, &worker_rewards);

        (manager_avg, worker_avg)
    }

    /// Run one episode on the given price data.
    /// Returns (total_reward, Vec<(step, regime, action, price)>).
    pub fn run_episode(
        &mut self,
        features: &[Array1<f64>],
        prices: &[f64],
    ) -> (f64, Vec<(usize, MarketRegime, f64, f64)>) {
        self.network.reset();
        let mut total_reward = 0.0;
        let mut position = 0.0;
        let mut log = Vec::new();

        for i in 0..features.len().saturating_sub(1) {
            let state = &features[i];
            let (action, regime) = self.act(state);

            // Compute reward from position change
            let price_return = if prices[i] > 0.0 {
                (prices[i + 1] - prices[i]) / prices[i]
            } else {
                0.0
            };
            let reward = position * price_return;

            // Compute intrinsic reward
            let next_state = &features[i + 1];
            let goal = self
                .network
                .current_goal
                .clone()
                .unwrap_or_else(|| Array1::zeros(self.network.manager.goal_dim));
            let intrinsic =
                LowLevelPolicy::compute_intrinsic_reward(state, next_state, &goal);

            self.store_experience(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                goal: goal.clone(),
                intrinsic_reward: intrinsic,
            });

            total_reward += reward;
            position = action; // action is the new position
            log.push((i, regime, action, prices[i]));
        }

        self.total_extrinsic_reward += total_reward;
        self.episode_count += 1;
        (total_reward, log)
    }
}

// ============================================================
// Flat RL Agent (Baseline)
// ============================================================

/// A simple flat (non-hierarchical) RL agent for comparison.
#[derive(Debug, Clone)]
pub struct FlatAgent {
    pub layer1: LinearLayer,
    pub layer2: LinearLayer,
    pub action_head: LinearLayer,
    pub learning_rate: f64,
}

impl FlatAgent {
    /// Create a new flat agent.
    pub fn new(state_dim: usize, hidden_dim: usize) -> Self {
        Self {
            layer1: LinearLayer::new(state_dim, hidden_dim, true),
            layer2: LinearLayer::new(hidden_dim, hidden_dim, true),
            action_head: LinearLayer::new(hidden_dim, 1, false),
            learning_rate: 0.001,
        }
    }

    /// Forward pass: returns action in [-1, 1].
    pub fn act(&self, state: &Array1<f64>) -> f64 {
        let h1 = self.layer1.forward(state);
        let h2 = self.layer2.forward(&h1);
        let action_raw = self.action_head.forward(&h2);
        action_raw[0].tanh()
    }

    /// Run one episode and return total reward.
    pub fn run_episode(&self, features: &[Array1<f64>], prices: &[f64]) -> f64 {
        let mut total_reward = 0.0;
        let mut position = 0.0;
        for i in 0..features.len().saturating_sub(1) {
            let action = self.act(&features[i]);
            let price_return = if prices[i] > 0.0 {
                (prices[i + 1] - prices[i]) / prices[i]
            } else {
                0.0
            };
            total_reward += position * price_return;
            position = action;
        }
        total_reward
    }
}

// ============================================================
// Feature Extraction
// ============================================================

/// Extract trading features from OHLCV data.
/// Returns (feature_vectors, close_prices).
pub fn extract_features(candles: &[Candle]) -> (Vec<Array1<f64>>, Vec<f64>) {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    let mut features = Vec::new();
    let mut prices = Vec::new();
    let lookback = 20;

    for i in lookback..closes.len() {
        // Log return
        let ret = if closes[i - 1] > 0.0 {
            (closes[i] / closes[i - 1]).ln()
        } else {
            0.0
        };

        // Rolling volatility
        let returns: Vec<f64> = (i - lookback..i)
            .map(|j| {
                if closes[j] > 0.0 && j > 0 {
                    (closes[j] / closes[j - 1].max(1e-8)).ln()
                } else {
                    0.0
                }
            })
            .collect();
        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt();

        // Momentum: SMA(5) - SMA(20)
        let sma5: f64 = closes[i - 5..i].iter().sum::<f64>() / 5.0;
        let sma20: f64 = closes[i - lookback..i].iter().sum::<f64>() / lookback as f64;
        let momentum = if sma20 > 0.0 {
            (sma5 - sma20) / sma20
        } else {
            0.0
        };

        // Volume ratio
        let avg_vol = volumes[i - lookback..i].iter().sum::<f64>() / lookback as f64;
        let vol_ratio = if avg_vol > 0.0 {
            volumes[i] / avg_vol
        } else {
            1.0
        };

        features.push(Array1::from_vec(vec![ret, volatility, momentum, vol_ratio]));
        prices.push(closes[i]);
    }

    (features, prices)
}

// ============================================================
// Bybit API Client
// ============================================================

/// OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structures.
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

/// Client for fetching market data from Bybit v5 public API.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    /// Create a new Bybit client.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (OHLCV) data for a given symbol and interval.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair, e.g., "BTCUSDT"
    /// * `interval` - Kline interval, e.g., "60" for 1 hour
    /// * `limit` - Number of candles to fetch (max 200)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );
        let response: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut candles: Vec<Candle> = response
            .result
            .list
            .iter()
            .filter_map(|item| {
                if item.len() >= 6 {
                    Some(Candle {
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

        // Bybit returns newest first, reverse to chronological order
        candles.reverse();
        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_regime_roundtrip() {
        assert_eq!(MarketRegime::from_index(0), MarketRegime::Bull);
        assert_eq!(MarketRegime::from_index(1), MarketRegime::Bear);
        assert_eq!(MarketRegime::from_index(2), MarketRegime::Sideways);
        assert_eq!(MarketRegime::Bull.as_index(), 0);
        assert_eq!(MarketRegime::Bear.as_index(), 1);
        assert_eq!(MarketRegime::Sideways.as_index(), 2);
        // Wrapping
        assert_eq!(MarketRegime::from_index(3), MarketRegime::Bull);
    }

    #[test]
    fn test_linear_layer_forward() {
        let weights = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.5]).unwrap();
        let bias = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let layer = LinearLayer::with_weights(weights, bias, false);

        let input = Array1::from_vec(vec![1.0, 2.0]);
        let output = layer.forward(&input);
        // [1*1+2*0+0.1, 1*0+2*1+0.2, 1*(-1)+2*0.5+0.3] = [1.1, 2.2, 0.3]
        assert!((output[0] - 1.1).abs() < 1e-10);
        assert!((output[1] - 2.2).abs() < 1e-10);
        assert!((output[2] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_linear_layer_relu() {
        let weights =
            Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).unwrap();
        let bias = Array1::zeros(2);
        let layer = LinearLayer::with_weights(weights, bias, true);

        let input = Array1::from_vec(vec![1.0, 0.0]);
        let output = layer.forward(&input);
        // [1*1+0*(-1), 1*(-1)+0*1] = [1.0, -1.0] -> ReLU -> [1.0, 0.0]
        assert!((output[0] - 1.0).abs() < 1e-10);
        assert!((output[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_high_level_policy_output_shapes() {
        let policy = HighLevelPolicy::new(4, 16, 4, 5);
        let state = Array1::from_vec(vec![0.01, 0.02, 0.001, 1.1]);
        let (logits, goal) = policy.forward(&state);
        assert_eq!(logits.len(), 3); // 3 regimes
        assert_eq!(goal.len(), 4); // 4D goal

        // Goal should be approximately unit length
        let goal_norm = goal.dot(&goal).sqrt();
        assert!((goal_norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_low_level_policy_action_bounds() {
        let policy = LowLevelPolicy::new(4, 4, 16);
        let state = Array1::from_vec(vec![0.5, 0.1, -0.02, 1.5]);
        let goal = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let action = policy.forward(&state, &goal);
        assert!(action >= -1.0 && action <= 1.0, "Action {} out of bounds", action);
    }

    #[test]
    fn test_intrinsic_reward_cosine_similarity() {
        // State moves in the same direction as goal -> positive reward
        let s_before = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let s_after = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let goal = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let reward = LowLevelPolicy::compute_intrinsic_reward(&s_before, &s_after, &goal);
        assert!(reward > 0.9, "Expected positive intrinsic reward, got {}", reward);

        // State moves opposite to goal -> negative reward
        let s_after_neg = Array1::from_vec(vec![-1.0, 0.0, 0.0, 0.0]);
        let reward_neg =
            LowLevelPolicy::compute_intrinsic_reward(&s_before, &s_after_neg, &goal);
        assert!(reward_neg < -0.9, "Expected negative intrinsic reward, got {}", reward_neg);

        // Orthogonal movement -> near zero reward
        let s_after_orth = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        let reward_orth =
            LowLevelPolicy::compute_intrinsic_reward(&s_before, &s_after_orth, &goal);
        assert!(
            reward_orth.abs() < 0.1,
            "Expected near-zero intrinsic reward, got {}",
            reward_orth
        );
    }

    #[test]
    fn test_feudal_network_decision_interval() {
        let mut network = FeudalNetwork::new(4, 16, 4, 3);
        let state = Array1::from_vec(vec![0.01, 0.02, 0.001, 1.1]);

        // First call should set a goal
        let (_, regime1) = network.act(&state);
        let goal1 = network.current_goal.clone().unwrap();

        // Steps 1 and 2 should keep the same goal
        let (_, _) = network.act(&state);
        let goal_mid = network.current_goal.clone().unwrap();
        assert_eq!(goal1, goal_mid);

        // After 2 more steps (total 3), manager should update
        let (_, _) = network.act(&state);
        // This is step 3 (index 3), which triggers a new manager decision
        let _ = network.act(&state);
        // The regime may or may not change (deterministic network, same input)
        assert!(network.step_count == 4);

        // Verify we get valid regimes
        assert!(
            regime1 == MarketRegime::Bull
                || regime1 == MarketRegime::Bear
                || regime1 == MarketRegime::Sideways
        );
    }

    #[test]
    fn test_hierarchical_agent_episode() {
        let mut agent = HierarchicalAgent::new(4, 16, 4, 5);

        // Create synthetic price data
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let features: Vec<Array1<f64>> = (0..50)
            .map(|i| {
                Array1::from_vec(vec![
                    0.005 * (i as f64).sin(),
                    0.02,
                    0.001 * i as f64,
                    1.0 + 0.1 * (i as f64).cos(),
                ])
            })
            .collect();

        let (reward, log) = agent.run_episode(&features, &prices);
        assert!(!log.is_empty());
        assert_eq!(log.len(), 49); // features.len() - 1
        assert!(reward.is_finite());
        assert_eq!(agent.episode_count, 1);
        assert!(!agent.replay_buffer.is_empty());
    }

    #[test]
    fn test_feature_extraction() {
        // Create synthetic candle data
        let candles: Vec<Candle> = (0..50)
            .map(|i| Candle {
                timestamp: 1000000 + i * 60000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0 + (i as f64 * 10.0),
            })
            .collect();

        let (features, prices) = extract_features(&candles);
        assert!(!features.is_empty());
        assert_eq!(features.len(), prices.len());
        // Each feature vector has 4 dimensions
        assert_eq!(features[0].len(), 4);
        // All prices should be positive
        assert!(prices.iter().all(|p| *p > 0.0));
    }

    #[test]
    fn test_flat_agent_action_bounds() {
        let agent = FlatAgent::new(4, 16);
        let state = Array1::from_vec(vec![0.01, 0.02, -0.001, 1.2]);
        let action = agent.act(&state);
        assert!(
            action >= -1.0 && action <= 1.0,
            "Flat agent action {} out of bounds",
            action
        );
    }
}
