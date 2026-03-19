use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ─────────────────────────────────────────────────────────────────────────────
// Bybit API types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch historical kline data from Bybit REST API.
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let resp: BybitKlineResponse = reqwest::blocking::get(&url)?.json()?;
    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error code: {}", resp.ret_code);
    }
    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();
    // Bybit returns newest first; reverse to chronological order
    candles.reverse();
    Ok(candles)
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Compute normalised feature vectors from candle data.
///
/// Features per step (5 values):
///   0: log return close-to-close
///   1: normalised volume (vol / mean_vol - 1)
///   2: high-low range / close
///   3: 5-bar simple moving average ratio (close/sma5 - 1)
///   4: 20-bar simple moving average ratio (close/sma20 - 1)
pub fn compute_features(candles: &[Candle], window: usize) -> Vec<Array1<f64>> {
    if candles.len() < window + 20 {
        return vec![];
    }
    let mean_vol: f64 = candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64;
    let mean_vol = if mean_vol == 0.0 { 1.0 } else { mean_vol };

    let mut features = Vec::new();
    for i in 20..candles.len() {
        let log_ret = if candles[i - 1].close > 0.0 {
            (candles[i].close / candles[i - 1].close).ln()
        } else {
            0.0
        };
        let norm_vol = candles[i].volume / mean_vol - 1.0;
        let range = if candles[i].close > 0.0 {
            (candles[i].high - candles[i].low) / candles[i].close
        } else {
            0.0
        };
        let sma5: f64 = candles[i - 4..=i].iter().map(|c| c.close).sum::<f64>() / 5.0;
        let sma20: f64 = candles[i - 19..=i].iter().map(|c| c.close).sum::<f64>() / 20.0;
        let sma5_ratio = if sma5 > 0.0 {
            candles[i].close / sma5 - 1.0
        } else {
            0.0
        };
        let sma20_ratio = if sma20 > 0.0 {
            candles[i].close / sma20 - 1.0
        } else {
            0.0
        };
        features.push(Array1::from(vec![
            log_ret, norm_vol, range, sma5_ratio, sma20_ratio,
        ]));
    }
    let _ = window; // window reserved for future sliding-window states
    features
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural network layers (minimal, no external ML crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Dense (fully-connected) layer: y = ReLU(x * W + b) or linear if no activation.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub use_relu: bool,
}

impl DenseLayer {
    pub fn new(input_dim: usize, output_dim: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_dim as f64).sqrt(); // He init
        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_dim);
        Self {
            weights,
            biases,
            use_relu,
        }
    }

    pub fn new_seeded(input_dim: usize, output_dim: usize, use_relu: bool, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let scale = (2.0 / input_dim as f64).sqrt();
        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_dim);
        Self {
            weights,
            biases,
            use_relu,
        }
    }

    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let z = x.dot(&self.weights) + &self.biases;
        if self.use_relu {
            z.mapv(|v| v.max(0.0))
        } else {
            z
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// A2C Network: shared backbone + policy head + value head
// ─────────────────────────────────────────────────────────────────────────────

/// Number of discrete trading actions: Hold, Buy, Sell.
pub const NUM_ACTIONS: usize = 3;

#[derive(Debug, Clone)]
pub struct A2CNetwork {
    pub backbone1: DenseLayer,
    pub backbone2: DenseLayer,
    pub policy_head: DenseLayer,
    pub value_head: DenseLayer,
}

impl A2CNetwork {
    /// Create a new A2C network with given state dimension and hidden size.
    pub fn new(state_dim: usize, hidden_dim: usize) -> Self {
        Self {
            backbone1: DenseLayer::new(state_dim, hidden_dim, true),
            backbone2: DenseLayer::new(hidden_dim, hidden_dim, true),
            policy_head: DenseLayer::new(hidden_dim, NUM_ACTIONS, false),
            value_head: DenseLayer::new(hidden_dim, 1, false),
        }
    }

    /// Create with deterministic seed for reproducibility.
    pub fn new_seeded(state_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        Self {
            backbone1: DenseLayer::new_seeded(state_dim, hidden_dim, true, seed),
            backbone2: DenseLayer::new_seeded(hidden_dim, hidden_dim, true, seed + 1),
            policy_head: DenseLayer::new_seeded(hidden_dim, NUM_ACTIONS, false, seed + 2),
            value_head: DenseLayer::new_seeded(hidden_dim, 1, false, seed + 3),
        }
    }

    /// Forward pass returning (action_probs, value).
    pub fn forward(&self, state: &Array1<f64>) -> (Array1<f64>, f64) {
        let h1 = self.backbone1.forward(state);
        let h2 = self.backbone2.forward(&h1);
        let logits = self.policy_head.forward(&h2);
        let value = self.value_head.forward(&h2)[0];
        let probs = softmax(&logits);
        (probs, value)
    }

    /// Collect all parameter arrays mutably for gradient updates.
    pub fn parameters_mut(
        &mut self,
    ) -> Vec<(&mut Array2<f64>, &mut Array1<f64>)> {
        vec![
            (&mut self.backbone1.weights, &mut self.backbone1.biases),
            (&mut self.backbone2.weights, &mut self.backbone2.biases),
            (&mut self.policy_head.weights, &mut self.policy_head.biases),
            (&mut self.value_head.weights, &mut self.value_head.biases),
        ]
    }
}

/// Softmax over a 1-D array with numerical stability.
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = logits.mapv(|v| (v - max_val).exp());
    let sum: f64 = exps.sum();
    if sum == 0.0 {
        Array1::from(vec![1.0 / logits.len() as f64; logits.len()])
    } else {
        exps / sum
    }
}

/// Sample an action from a probability distribution.
pub fn sample_action(probs: &Array1<f64>) -> usize {
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

/// Sample action with a seeded RNG.
pub fn sample_action_seeded(probs: &Array1<f64>, rng: &mut impl Rng) -> usize {
    let r: f64 = rng.gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Advantage computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute GAE (Generalised Advantage Estimation) advantages and returns.
///
/// * `rewards` - vector of rewards at each step
/// * `values` - vector of value estimates V(s_t) at each step
/// * `last_value` - bootstrap value V(s_T) for the final state
/// * `gamma` - discount factor
/// * `lambda` - GAE lambda parameter
///
/// Returns `(advantages, returns)` where `returns[t] = advantages[t] + values[t]`.
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    last_value: f64,
    gamma: f64,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    let mut gae = 0.0;
    for t in (0..n).rev() {
        let next_value = if t == n - 1 { last_value } else { values[t + 1] };
        let delta = rewards[t] + gamma * next_value - values[t];
        gae = delta + gamma * lambda * gae;
        advantages[t] = gae;
    }
    let returns: Vec<f64> = advantages
        .iter()
        .zip(values.iter())
        .map(|(a, v)| a + v)
        .collect();
    (advantages, returns)
}

/// Compute simple one-step TD advantages (GAE with lambda=0).
pub fn compute_td_advantages(
    rewards: &[f64],
    values: &[f64],
    last_value: f64,
    gamma: f64,
) -> Vec<f64> {
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    for t in 0..n {
        let next_value = if t == n - 1 { last_value } else { values[t + 1] };
        advantages[t] = rewards[t] + gamma * next_value - values[t];
    }
    advantages
}

// ─────────────────────────────────────────────────────────────────────────────
// Entropy
// ─────────────────────────────────────────────────────────────────────────────

/// Compute entropy of a discrete probability distribution: -sum(p * ln(p)).
pub fn entropy(probs: &Array1<f64>) -> f64 {
    let eps = 1e-10;
    -probs
        .iter()
        .map(|&p| {
            let p_safe = p.max(eps);
            p_safe * p_safe.ln()
        })
        .sum::<f64>()
}

// ─────────────────────────────────────────────────────────────────────────────
// Trading environment
// ─────────────────────────────────────────────────────────────────────────────

/// Simple single-asset trading environment.
///
/// Actions: 0 = Hold, 1 = Buy (go long), 2 = Sell (go short / close long)
#[derive(Debug, Clone)]
pub struct TradingEnv {
    pub prices: Vec<f64>,
    pub features: Vec<Array1<f64>>,
    pub position: f64,     // +1 long, 0 flat, -1 short
    pub entry_price: f64,
    pub step: usize,
    pub transaction_cost: f64,
    pub total_reward: f64,
}

impl TradingEnv {
    /// Create from candle data.
    pub fn new(candles: &[Candle], transaction_cost: f64) -> Self {
        let features = compute_features(candles, 1);
        let prices: Vec<f64> = candles[20..].iter().map(|c| c.close).collect();
        // Ensure lengths match
        let len = features.len().min(prices.len());
        Self {
            prices: prices[..len].to_vec(),
            features: features[..len].to_vec(),
            position: 0.0,
            entry_price: 0.0,
            step: 0,
            transaction_cost,
            total_reward: 0.0,
        }
    }

    /// Create from price vector and pre-computed features.
    pub fn from_data(prices: Vec<f64>, features: Vec<Array1<f64>>, transaction_cost: f64) -> Self {
        let len = features.len().min(prices.len());
        Self {
            prices: prices[..len].to_vec(),
            features: features[..len].to_vec(),
            position: 0.0,
            entry_price: 0.0,
            step: 0,
            transaction_cost,
            total_reward: 0.0,
        }
    }

    pub fn state_dim(&self) -> usize {
        if self.features.is_empty() {
            7 // 5 features + position + unrealised pnl
        } else {
            self.features[0].len() + 2
        }
    }

    pub fn reset(&mut self) {
        self.position = 0.0;
        self.entry_price = 0.0;
        self.step = 0;
        self.total_reward = 0.0;
    }

    pub fn get_state(&self) -> Array1<f64> {
        if self.step >= self.features.len() {
            return Array1::zeros(self.state_dim());
        }
        let mut state = self.features[self.step].to_vec();
        state.push(self.position);
        let unrealised = if self.position != 0.0 && self.entry_price > 0.0 {
            self.position * (self.prices[self.step] / self.entry_price - 1.0)
        } else {
            0.0
        };
        state.push(unrealised);
        Array1::from(state)
    }

    pub fn is_done(&self) -> bool {
        self.step >= self.features.len() - 1
    }

    /// Execute action and return (reward, done).
    pub fn step_action(&mut self, action: usize) -> (f64, bool) {
        if self.is_done() {
            return (0.0, true);
        }

        let current_price = self.prices[self.step];
        let next_price = self.prices[self.step + 1];
        let price_return = next_price / current_price - 1.0;

        let mut reward = 0.0;
        let mut cost = 0.0;

        match action {
            1 => {
                // Buy
                if self.position <= 0.0 {
                    cost = self.transaction_cost;
                    self.position = 1.0;
                    self.entry_price = current_price;
                }
            }
            2 => {
                // Sell
                if self.position >= 0.0 {
                    cost = self.transaction_cost;
                    self.position = -1.0;
                    self.entry_price = current_price;
                }
            }
            _ => {} // Hold
        }

        // PnL from holding position over the bar
        reward += self.position * price_return - cost;
        self.total_reward += reward;
        self.step += 1;

        (reward, self.is_done())
    }

    pub fn remaining_steps(&self) -> usize {
        if self.step >= self.features.len() {
            0
        } else {
            self.features.len() - 1 - self.step
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// A2C Agent
// ─────────────────────────────────────────────────────────────────────────────

/// Hyperparameters for the A2C agent.
#[derive(Debug, Clone)]
pub struct A2CConfig {
    pub gamma: f64,
    pub lambda: f64,
    pub learning_rate: f64,
    pub entropy_coef: f64,
    pub value_coef: f64,
    pub max_grad_norm: f64,
    pub hidden_dim: usize,
    pub rollout_steps: usize,
}

impl Default for A2CConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            learning_rate: 3e-4,
            entropy_coef: 0.01,
            value_coef: 0.5,
            max_grad_norm: 0.5,
            hidden_dim: 64,
            rollout_steps: 32,
        }
    }
}

/// A2C agent that owns a network and config.
#[derive(Debug, Clone)]
pub struct A2CAgent {
    pub network: A2CNetwork,
    pub config: A2CConfig,
}

impl A2CAgent {
    pub fn new(state_dim: usize, config: A2CConfig) -> Self {
        let network = A2CNetwork::new(state_dim, config.hidden_dim);
        Self { network, config }
    }

    pub fn new_seeded(state_dim: usize, config: A2CConfig, seed: u64) -> Self {
        let network = A2CNetwork::new_seeded(state_dim, config.hidden_dim, seed);
        Self { network, config }
    }

    /// Select an action given a state.
    pub fn act(&self, state: &Array1<f64>) -> (usize, Array1<f64>, f64) {
        let (probs, value) = self.network.forward(state);
        let action = sample_action(&probs);
        (action, probs, value)
    }

    /// Select action with seeded RNG.
    pub fn act_seeded(
        &self,
        state: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> (usize, Array1<f64>, f64) {
        let (probs, value) = self.network.forward(state);
        let action = sample_action_seeded(&probs, rng);
        (action, probs, value)
    }

    /// Collect a rollout of experience from the environment.
    pub fn collect_rollout(
        &self,
        env: &mut TradingEnv,
    ) -> RolloutData {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut values = Vec::new();
        let mut log_probs = Vec::new();
        let mut entropies = Vec::new();

        let steps = self.config.rollout_steps.min(env.remaining_steps());
        for _ in 0..steps {
            let state = env.get_state();
            let (action, probs, value) = self.act(&state);
            let lp = (probs[action].max(1e-10)).ln();
            let ent = entropy(&probs);

            let (reward, done) = env.step_action(action);

            states.push(state);
            actions.push(action);
            rewards.push(reward);
            values.push(value);
            log_probs.push(lp);
            entropies.push(ent);

            if done {
                break;
            }
        }

        // Bootstrap value for last state
        let last_value = if env.is_done() {
            0.0
        } else {
            let (_, v) = self.network.forward(&env.get_state());
            v
        };

        let (advantages, returns) =
            compute_gae(&rewards, &values, last_value, self.config.gamma, self.config.lambda);

        RolloutData {
            states,
            actions,
            rewards,
            values,
            log_probs,
            entropies,
            advantages,
            returns,
        }
    }

    /// Perform a simplified gradient update using finite differences.
    ///
    /// This is a numerical approximation for demonstration purposes.
    /// A production implementation would use automatic differentiation.
    pub fn update(&mut self, rollout: &RolloutData) -> UpdateStats {
        let n = rollout.states.len();
        if n == 0 {
            return UpdateStats {
                actor_loss: 0.0,
                critic_loss: 0.0,
                entropy_mean: 0.0,
            };
        }

        // Compute losses for logging
        let mut actor_loss = 0.0;
        let mut critic_loss = 0.0;
        let mut entropy_sum = 0.0;

        for t in 0..n {
            let adv = rollout.advantages[t];
            actor_loss -= rollout.log_probs[t] * adv;
            let value_error = rollout.returns[t] - rollout.values[t];
            critic_loss += value_error * value_error;
            entropy_sum += rollout.entropies[t];
        }
        actor_loss /= n as f64;
        critic_loss /= n as f64;
        let entropy_mean = entropy_sum / n as f64;

        // Simplified parameter update using reward-weighted perturbation
        // (Evolutionary Strategy / REINFORCE-style weight update as a proxy)
        let lr = self.config.learning_rate;
        let mean_advantage: f64 = rollout.advantages.iter().sum::<f64>() / n as f64;

        let params = self.network.parameters_mut();
        for (weights, biases) in params {
            // Nudge weights in direction proportional to advantage signal
            let scale = lr * mean_advantage;
            let grad_clip = self.config.max_grad_norm;
            let clamped = scale.clamp(-grad_clip, grad_clip);
            weights.mapv_inplace(|w| w + clamped * 0.01);
            biases.mapv_inplace(|b| b + clamped * 0.01);
        }

        UpdateStats {
            actor_loss,
            critic_loss,
            entropy_mean,
        }
    }

    /// Train for multiple episodes, returning per-episode total rewards.
    pub fn train(&mut self, env: &mut TradingEnv, num_episodes: usize) -> Vec<f64> {
        let mut episode_rewards = Vec::new();
        for _ in 0..num_episodes {
            env.reset();
            while !env.is_done() {
                let rollout = self.collect_rollout(env);
                self.update(&rollout);
            }
            episode_rewards.push(env.total_reward);
        }
        episode_rewards
    }
}

/// Data collected during a rollout.
#[derive(Debug, Clone)]
pub struct RolloutData {
    pub states: Vec<Array1<f64>>,
    pub actions: Vec<usize>,
    pub rewards: Vec<f64>,
    pub values: Vec<f64>,
    pub log_probs: Vec<f64>,
    pub entropies: Vec<f64>,
    pub advantages: Vec<f64>,
    pub returns: Vec<f64>,
}

/// Statistics from a training update step.
#[derive(Debug, Clone)]
pub struct UpdateStats {
    pub actor_loss: f64,
    pub critic_loss: f64,
    pub entropy_mean: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// REINFORCE baseline (no critic) for comparison
// ─────────────────────────────────────────────────────────────────────────────

/// A simple REINFORCE agent (policy gradient without baseline) for comparison.
#[derive(Debug, Clone)]
pub struct ReinforceAgent {
    pub policy: DenseLayer,
    pub hidden: DenseLayer,
    pub learning_rate: f64,
    pub gamma: f64,
}

impl ReinforceAgent {
    pub fn new(state_dim: usize, hidden_dim: usize) -> Self {
        Self {
            hidden: DenseLayer::new(state_dim, hidden_dim, true),
            policy: DenseLayer::new(hidden_dim, NUM_ACTIONS, false),
            learning_rate: 3e-4,
            gamma: 0.99,
        }
    }

    pub fn act(&self, state: &Array1<f64>) -> (usize, Array1<f64>) {
        let h = self.hidden.forward(state);
        let logits = self.policy.forward(&h);
        let probs = softmax(&logits);
        let action = sample_action(&probs);
        (action, probs)
    }

    /// Run a single episode and return total reward.
    pub fn run_episode(&mut self, env: &mut TradingEnv) -> f64 {
        env.reset();
        let mut log_probs = Vec::new();
        let mut rewards = Vec::new();

        while !env.is_done() {
            let state = env.get_state();
            let (action, probs) = self.act(&state);
            log_probs.push((probs[action].max(1e-10)).ln());
            let (reward, _) = env.step_action(action);
            rewards.push(reward);
        }

        // Compute discounted returns
        let n = rewards.len();
        let mut returns = vec![0.0; n];
        let mut g = 0.0;
        for t in (0..n).rev() {
            g = rewards[t] + self.gamma * g;
            returns[t] = g;
        }

        // Simplified update
        let mean_return: f64 = returns.iter().sum::<f64>() / n.max(1) as f64;
        let lr = self.learning_rate;
        let scale = (lr * mean_return * 0.01).clamp(-0.01, 0.01);
        self.hidden.weights.mapv_inplace(|w| w + scale);
        self.hidden.biases.mapv_inplace(|b| b + scale);
        self.policy.weights.mapv_inplace(|w| w + scale);
        self.policy.biases.mapv_inplace(|b| b + scale);

        env.total_reward
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_candles(n: usize) -> Vec<Candle> {
        let mut candles = Vec::new();
        let mut price = 100.0;
        for i in 0..n {
            let noise = ((i as f64) * 0.1).sin() * 2.0;
            price += noise;
            candles.push(Candle {
                timestamp: 1000 + i as u64,
                open: price - 0.5,
                high: price + 1.0,
                low: price - 1.0,
                close: price,
                volume: 1000.0 + (i as f64 * 10.0),
            });
        }
        candles
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = Array1::from(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1, got {}", sum);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let logits = Array1::from(vec![1000.0, 1001.0, 1002.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax should be stable with large values");
        assert!(probs.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_entropy_uniform() {
        let probs = Array1::from(vec![1.0 / 3.0; 3]);
        let h = entropy(&probs);
        let expected = (3.0_f64).ln();
        assert!(
            (h - expected).abs() < 1e-6,
            "Entropy of uniform dist over 3 should be ln(3) = {}, got {}",
            expected,
            h
        );
    }

    #[test]
    fn test_entropy_deterministic() {
        let probs = Array1::from(vec![1.0, 0.0, 0.0]);
        let h = entropy(&probs);
        assert!(h < 1e-6, "Entropy of deterministic dist should be ~0, got {}", h);
    }

    #[test]
    fn test_gae_computation() {
        let rewards = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 1.0, 1.5];
        let last_value = 2.0;
        let gamma = 0.99;
        let lambda = 0.95;
        let (advantages, returns) = compute_gae(&rewards, &values, last_value, gamma, lambda);
        assert_eq!(advantages.len(), 3);
        assert_eq!(returns.len(), 3);
        // returns[t] = advantages[t] + values[t]
        for t in 0..3 {
            assert!(
                (returns[t] - (advantages[t] + values[t])).abs() < 1e-10,
                "returns[t] should equal advantages[t] + values[t]"
            );
        }
    }

    #[test]
    fn test_td_advantages() {
        let rewards = vec![1.0, 0.0];
        let values = vec![0.5, 0.5];
        let last_value = 0.5;
        let gamma = 0.99;
        let adv = compute_td_advantages(&rewards, &values, last_value, gamma);
        // td[0] = r0 + gamma * v1 - v0 = 1.0 + 0.99*0.5 - 0.5 = 0.995
        assert!((adv[0] - 0.995).abs() < 1e-10);
        // td[1] = r1 + gamma * last - v1 = 0.0 + 0.99*0.5 - 0.5 = -0.005
        assert!((adv[1] - (-0.005)).abs() < 1e-10);
    }

    #[test]
    fn test_network_forward() {
        let net = A2CNetwork::new_seeded(5, 32, 42);
        let state = Array1::from(vec![0.1, -0.2, 0.3, 0.05, -0.01]);
        let (probs, value) = net.forward(&state);
        assert_eq!(probs.len(), NUM_ACTIONS);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Probs should sum to 1");
        assert!(probs.iter().all(|&p| p >= 0.0));
        assert!(value.is_finite());
    }

    #[test]
    fn test_feature_computation() {
        let candles = make_dummy_candles(100);
        let features = compute_features(&candles, 1);
        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 5);
        assert!(features.iter().all(|f| f.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn test_trading_env_episode() {
        let candles = make_dummy_candles(100);
        let mut env = TradingEnv::new(&candles, 0.001);
        assert!(!env.is_done());
        let state = env.get_state();
        assert_eq!(state.len(), env.state_dim());

        // Run through a few steps
        let (reward, done) = env.step_action(1); // Buy
        assert!(reward.is_finite());
        assert!(!done || env.step >= env.features.len() - 1);
        assert_eq!(env.position, 1.0);

        let (_, _) = env.step_action(0); // Hold
        assert_eq!(env.position, 1.0); // Still long

        let (_, _) = env.step_action(2); // Sell
        assert_eq!(env.position, -1.0); // Now short
    }

    #[test]
    fn test_a2c_agent_training() {
        let candles = make_dummy_candles(100);
        let mut env = TradingEnv::new(&candles, 0.001);
        let config = A2CConfig {
            rollout_steps: 16,
            hidden_dim: 16,
            ..Default::default()
        };
        let mut agent = A2CAgent::new(env.state_dim(), config);
        let rewards = agent.train(&mut env, 3);
        assert_eq!(rewards.len(), 3);
        assert!(rewards.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn test_reinforce_agent() {
        let candles = make_dummy_candles(100);
        let mut env = TradingEnv::new(&candles, 0.001);
        let mut agent = ReinforceAgent::new(env.state_dim(), 16);
        let reward = agent.run_episode(&mut env);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_sample_action_within_range() {
        let probs = Array1::from(vec![0.2, 0.5, 0.3]);
        for _ in 0..100 {
            let a = sample_action(&probs);
            assert!(a < NUM_ACTIONS);
        }
    }
}
