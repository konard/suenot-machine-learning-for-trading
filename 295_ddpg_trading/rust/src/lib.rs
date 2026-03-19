use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Neural-network helpers
// ---------------------------------------------------------------------------

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_deriv(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn tanh_clamp(x: f64) -> f64 {
    x.tanh()
}

fn random_matrix(rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| {
        rng.gen_range(-scale..scale)
    })
}

fn _random_vector(size: usize, scale: f64) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    Array1::from_shape_fn(size, |_| rng.gen_range(-scale..scale))
}

// ---------------------------------------------------------------------------
// Actor Network: state -> continuous action in [-1, 1]
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ActorNetwork {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    pub w3: Array2<f64>,
    pub b3: Array1<f64>,
}

impl ActorNetwork {
    pub fn new(state_dim: usize, hidden1: usize, hidden2: usize, action_dim: usize) -> Self {
        let scale1 = (2.0 / state_dim as f64).sqrt();
        let scale2 = (2.0 / hidden1 as f64).sqrt();
        let scale3 = 0.003; // small init for output layer
        Self {
            w1: random_matrix(hidden1, state_dim, scale1),
            b1: Array1::zeros(hidden1),
            w2: random_matrix(hidden2, hidden1, scale2),
            b2: Array1::zeros(hidden2),
            w3: random_matrix(action_dim, hidden2, scale3),
            b3: Array1::zeros(action_dim),
        }
    }

    /// Forward pass returning (action, hidden1_pre, hidden1_out, hidden2_pre, hidden2_out)
    pub fn forward_full(&self, state: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let h1_pre = self.w1.dot(state) + &self.b1;
        let h1 = h1_pre.mapv(relu);
        let h2_pre = self.w2.dot(&h1) + &self.b2;
        let h2 = h2_pre.mapv(relu);
        let out_pre = self.w3.dot(&h2) + &self.b3;
        let action = out_pre.mapv(tanh_clamp);
        (action, h1_pre, h1, h2_pre, h2)
    }

    pub fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        self.forward_full(state).0
    }

    /// Compute gradient of action w.r.t. actor parameters given dJ/da (from critic).
    /// Updates weights using the provided learning rate.
    pub fn update(&mut self, state: &Array1<f64>, da: &Array1<f64>, lr: f64) {
        let (_action, h1_pre, h1, h2_pre, h2) = self.forward_full(state);
        let out_pre = self.w3.dot(&h2) + &self.b3;

        // d(tanh)/d(pre) = 1 - tanh^2
        let tanh_vals = out_pre.mapv(tanh_clamp);
        let dtanh = tanh_vals.mapv(|t| 1.0 - t * t);
        let d_out = da * &dtanh;

        // Layer 3 gradients
        let grad_w3 = d_out.clone().insert_axis(Axis(1)).dot(&h2.clone().insert_axis(Axis(0)));
        let grad_b3 = d_out.clone();

        // Backprop to layer 2
        let d_h2 = self.w3.t().dot(&d_out) * h2_pre.mapv(relu_deriv);
        let grad_w2 = d_h2.clone().insert_axis(Axis(1)).dot(&h1.clone().insert_axis(Axis(0)));
        let grad_b2 = d_h2.clone();

        // Backprop to layer 1
        let d_h1 = self.w2.t().dot(&d_h2) * h1_pre.mapv(relu_deriv);
        let grad_w1 = d_h1.clone().insert_axis(Axis(1)).dot(&state.clone().insert_axis(Axis(0)));
        let grad_b1 = d_h1;

        // Apply gradients (ascent for policy improvement)
        self.w3 = &self.w3 + &(lr * &grad_w3);
        self.b3 = &self.b3 + &(lr * &grad_b3);
        self.w2 = &self.w2 + &(lr * &grad_w2);
        self.b2 = &self.b2 + &(lr * &grad_b2);
        self.w1 = &self.w1 + &(lr * &grad_w1);
        self.b1 = &self.b1 + &(lr * &grad_b1);
    }

    pub fn soft_update(&mut self, source: &ActorNetwork, tau: f64) {
        self.w1 = &self.w1 * (1.0 - tau) + &source.w1 * tau;
        self.b1 = &self.b1 * (1.0 - tau) + &source.b1 * tau;
        self.w2 = &self.w2 * (1.0 - tau) + &source.w2 * tau;
        self.b2 = &self.b2 * (1.0 - tau) + &source.b2 * tau;
        self.w3 = &self.w3 * (1.0 - tau) + &source.w3 * tau;
        self.b3 = &self.b3 * (1.0 - tau) + &source.b3 * tau;
    }
}

// ---------------------------------------------------------------------------
// Critic Network: (state, action) -> Q-value
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct CriticNetwork {
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    pub w3: Array2<f64>,
    pub b3: Array1<f64>,
}

impl CriticNetwork {
    pub fn new(state_dim: usize, action_dim: usize, hidden1: usize, hidden2: usize) -> Self {
        let input_dim = state_dim + action_dim;
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden1 as f64).sqrt();
        let scale3 = 0.003;
        Self {
            w1: random_matrix(hidden1, input_dim, scale1),
            b1: Array1::zeros(hidden1),
            w2: random_matrix(hidden2, hidden1, scale2),
            b2: Array1::zeros(hidden2),
            w3: random_matrix(1, hidden2, scale3),
            b3: Array1::zeros(1),
        }
    }

    fn make_input(state: &Array1<f64>, action: &Array1<f64>) -> Array1<f64> {
        let mut input = Array1::zeros(state.len() + action.len());
        input.slice_mut(ndarray::s![..state.len()]).assign(state);
        input.slice_mut(ndarray::s![state.len()..]).assign(action);
        input
    }

    pub fn forward_full(&self, state: &Array1<f64>, action: &Array1<f64>) -> (f64, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let input = Self::make_input(state, action);
        let h1_pre = self.w1.dot(&input) + &self.b1;
        let h1 = h1_pre.mapv(relu);
        let h2_pre = self.w2.dot(&h1) + &self.b2;
        let h2 = h2_pre.mapv(relu);
        let q = (self.w3.dot(&h2) + &self.b3)[0];
        (q, input, h1_pre, h1, h2_pre, h2)
    }

    pub fn forward(&self, state: &Array1<f64>, action: &Array1<f64>) -> f64 {
        self.forward_full(state, action).0
    }

    /// Returns gradient of Q w.r.t. action (for actor update)
    pub fn action_gradient(&self, state: &Array1<f64>, action: &Array1<f64>) -> Array1<f64> {
        let (_q, _input, h1_pre, _h1, h2_pre, _h2) = self.forward_full(state, action);
        let action_dim = action.len();
        let state_dim = state.len();

        // dQ/d(out) = 1
        let d_out = Array1::from_elem(1, 1.0);
        // Backprop through layer 3
        let d_h2 = self.w3.t().dot(&d_out) * h2_pre.mapv(relu_deriv);
        // Backprop through layer 2
        let d_h1 = self.w2.t().dot(&d_h2) * h1_pre.mapv(relu_deriv);
        // Backprop through layer 1 to input
        let d_input = self.w1.t().dot(&d_h1);
        // Extract action part
        d_input.slice(ndarray::s![state_dim..state_dim + action_dim]).to_owned()
    }

    /// Update critic using TD error for a single sample.
    pub fn update(&mut self, state: &Array1<f64>, action: &Array1<f64>, target_q: f64, lr: f64) {
        let (q, input, h1_pre, h1, h2_pre, h2) = self.forward_full(state, action);
        let td_error = q - target_q;

        // dL/dQ = td_error (from MSE loss, without the 0.5 factor, single sample)
        let d_out = Array1::from_elem(1, td_error);

        // Layer 3
        let grad_w3 = d_out.clone().insert_axis(Axis(1)).dot(&h2.clone().insert_axis(Axis(0)));
        let grad_b3 = d_out.clone();
        let d_h2 = self.w3.t().dot(&d_out) * h2_pre.mapv(relu_deriv);

        // Layer 2
        let grad_w2 = d_h2.clone().insert_axis(Axis(1)).dot(&h1.clone().insert_axis(Axis(0)));
        let grad_b2 = d_h2.clone();
        let d_h1 = self.w2.t().dot(&d_h2) * h1_pre.mapv(relu_deriv);

        // Layer 1
        let grad_w1 = d_h1.clone().insert_axis(Axis(1)).dot(&input.clone().insert_axis(Axis(0)));
        let grad_b1 = d_h1;

        // Gradient descent (minimize loss)
        self.w3 = &self.w3 - &(lr * &grad_w3);
        self.b3 = &self.b3 - &(lr * &grad_b3);
        self.w2 = &self.w2 - &(lr * &grad_w2);
        self.b2 = &self.b2 - &(lr * &grad_b2);
        self.w1 = &self.w1 - &(lr * &grad_w1);
        self.b1 = &self.b1 - &(lr * &grad_b1);
    }

    pub fn soft_update(&mut self, source: &CriticNetwork, tau: f64) {
        self.w1 = &self.w1 * (1.0 - tau) + &source.w1 * tau;
        self.b1 = &self.b1 * (1.0 - tau) + &source.b1 * tau;
        self.w2 = &self.w2 * (1.0 - tau) + &source.w2 * tau;
        self.b2 = &self.b2 * (1.0 - tau) + &source.b2 * tau;
        self.w3 = &self.w3 * (1.0 - tau) + &source.w3 * tau;
        self.b3 = &self.b3 * (1.0 - tau) + &source.b3 * tau;
    }
}

// ---------------------------------------------------------------------------
// Ornstein-Uhlenbeck Noise
// ---------------------------------------------------------------------------

pub struct OUNoise {
    pub theta: f64,
    pub mu: f64,
    pub sigma: f64,
    pub state: f64,
    pub sigma_decay: f64,
    pub sigma_min: f64,
}

impl OUNoise {
    pub fn new(theta: f64, mu: f64, sigma: f64) -> Self {
        Self {
            theta,
            mu,
            sigma,
            state: mu,
            sigma_decay: 0.9999,
            sigma_min: 0.01,
        }
    }

    pub fn sample(&mut self) -> f64 {
        let mut rng = rand::thread_rng();
        let noise: f64 = rng.gen_range(-1.0..1.0) * 1.7320508; // approximate normal via uniform scaled
        self.state += self.theta * (self.mu - self.state) + self.sigma * noise;
        self.sigma = (self.sigma * self.sigma_decay).max(self.sigma_min);
        self.state
    }

    pub fn reset(&mut self) {
        self.state = self.mu;
    }
}

// ---------------------------------------------------------------------------
// Replay Buffer
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Transition {
    pub state: Array1<f64>,
    pub action: Array1<f64>,
    pub reward: f64,
    pub next_state: Array1<f64>,
    pub done: bool,
}

pub struct ReplayBuffer {
    pub buffer: VecDeque<Transition>,
    pub capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
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

// ---------------------------------------------------------------------------
// Trading Environment
// ---------------------------------------------------------------------------

pub struct TradingEnv {
    pub prices: Vec<f64>,
    pub volumes: Vec<f64>,
    pub current_step: usize,
    pub position: f64,
    pub entry_price: f64,
    pub pnl: f64,
    pub state_dim: usize,
    pub window: usize,
}

impl TradingEnv {
    pub fn new(prices: Vec<f64>, volumes: Vec<f64>) -> Self {
        Self {
            prices,
            volumes,
            current_step: 10,
            position: 0.0,
            entry_price: 0.0,
            pnl: 0.0,
            state_dim: 8,
            window: 10,
        }
    }

    pub fn reset(&mut self) -> Array1<f64> {
        self.current_step = self.window;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.pnl = 0.0;
        self.get_state()
    }

    pub fn get_state(&self) -> Array1<f64> {
        let i = self.current_step;
        let mut state = Array1::zeros(self.state_dim);

        if i >= 1 && i < self.prices.len() {
            // Return features at different horizons
            let ret1 = (self.prices[i] - self.prices[i - 1]) / self.prices[i - 1];
            let ret5 = if i >= 5 {
                (self.prices[i] - self.prices[i - 5]) / self.prices[i - 5]
            } else {
                0.0
            };
            let ret10 = if i >= 10 {
                (self.prices[i] - self.prices[i - 10]) / self.prices[i - 10]
            } else {
                0.0
            };

            // Volatility (std of recent returns)
            let mut returns = Vec::new();
            for j in (i.saturating_sub(9))..i {
                if j >= 1 {
                    returns.push((self.prices[j + 1] - self.prices[j]) / self.prices[j]);
                }
            }
            let vol = if returns.len() > 1 {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64).sqrt()
            } else {
                0.0
            };

            // Volume ratio
            let vol_mean: f64 = if i >= 5 && !self.volumes.is_empty() {
                self.volumes[i.saturating_sub(5)..=i.min(self.volumes.len() - 1)]
                    .iter()
                    .sum::<f64>()
                    / 5.0
            } else {
                1.0
            };
            let vol_ratio = if !self.volumes.is_empty() && vol_mean > 0.0 {
                self.volumes[i.min(self.volumes.len() - 1)] / vol_mean
            } else {
                1.0
            };

            state[0] = ret1 * 100.0; // scale returns
            state[1] = ret5 * 100.0;
            state[2] = ret10 * 100.0;
            state[3] = vol * 100.0;
            state[4] = vol_ratio - 1.0;
            state[5] = self.position;
            state[6] = if self.entry_price > 0.0 {
                (self.prices[i] - self.entry_price) / self.entry_price * 100.0
            } else {
                0.0
            };
            state[7] = (i as f64 / self.prices.len() as f64) * 2.0 - 1.0; // time progress
        }
        state
    }

    pub fn step(&mut self, action: f64) -> (Array1<f64>, f64, bool) {
        let action = action.clamp(-1.0, 1.0);
        let i = self.current_step;

        if i + 1 >= self.prices.len() {
            return (self.get_state(), 0.0, true);
        }

        let price_return = (self.prices[i + 1] - self.prices[i]) / self.prices[i];

        // Reward: position * return - transaction cost
        let position_change = (action - self.position).abs();
        let transaction_cost = position_change * 0.0004; // 4 bps cost
        let raw_reward = self.position * price_return;
        let risk_penalty = 0.5 * (self.position * price_return).powi(2); // variance penalty
        let reward = raw_reward - transaction_cost - risk_penalty;

        self.pnl += raw_reward - transaction_cost;

        // Update position
        if action.abs() > 0.01 && self.position.abs() < 0.01 {
            self.entry_price = self.prices[i];
        } else if action.abs() < 0.01 {
            self.entry_price = 0.0;
        }
        self.position = action;
        self.current_step += 1;

        let done = self.current_step + 1 >= self.prices.len();
        let next_state = self.get_state();

        (next_state, reward, done)
    }

    pub fn max_steps(&self) -> usize {
        if self.prices.len() > self.window + 1 {
            self.prices.len() - self.window - 1
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------------------
// DDPG Agent
// ---------------------------------------------------------------------------

pub struct DDPGAgent {
    pub actor: ActorNetwork,
    pub critic: CriticNetwork,
    pub target_actor: ActorNetwork,
    pub target_critic: CriticNetwork,
    pub replay_buffer: ReplayBuffer,
    pub noise: OUNoise,
    pub gamma: f64,
    pub tau: f64,
    pub actor_lr: f64,
    pub critic_lr: f64,
    pub batch_size: usize,
}

impl DDPGAgent {
    pub fn new(state_dim: usize, action_dim: usize) -> Self {
        let actor = ActorNetwork::new(state_dim, 64, 32, action_dim);
        let critic = CriticNetwork::new(state_dim, action_dim, 64, 32);
        let target_actor = actor.clone();
        let target_critic = critic.clone();

        Self {
            actor,
            critic,
            target_actor,
            target_critic,
            replay_buffer: ReplayBuffer::new(100_000),
            noise: OUNoise::new(0.15, 0.0, 0.2),
            gamma: 0.99,
            tau: 0.005,
            actor_lr: 1e-4,
            critic_lr: 1e-3,
            batch_size: 64,
        }
    }

    pub fn select_action(&mut self, state: &Array1<f64>, explore: bool) -> Array1<f64> {
        let mut action = self.actor.forward(state);
        if explore {
            let noise = self.noise.sample();
            action[0] = (action[0] + noise).clamp(-1.0, 1.0);
        }
        action
    }

    pub fn store_transition(&mut self, transition: Transition) {
        self.replay_buffer.push(transition);
    }

    pub fn train_step(&mut self) -> Option<f64> {
        if self.replay_buffer.len() < self.batch_size {
            return None;
        }

        let batch = self.replay_buffer.sample(self.batch_size);
        let mut total_critic_loss = 0.0;

        for t in &batch {
            // Compute target Q
            let target_action = self.target_actor.forward(&t.next_state);
            let target_q = self.target_critic.forward(&t.next_state, &target_action);
            let y = t.reward + if t.done { 0.0 } else { self.gamma * target_q };

            // Update critic
            let q = self.critic.forward(&t.state, &t.action);
            total_critic_loss += (q - y).powi(2);
            self.critic.update(&t.state, &t.action, y, self.critic_lr / self.batch_size as f64);
        }

        // Update actor
        for t in &batch {
            let action = self.actor.forward(&t.state);
            let da = self.critic.action_gradient(&t.state, &action);
            self.actor.update(&t.state, &da, self.actor_lr / self.batch_size as f64);
        }

        // Soft update targets
        self.target_actor.soft_update(&self.actor, self.tau);
        self.target_critic.soft_update(&self.critic, self.tau);

        Some(total_critic_loss / self.batch_size as f64)
    }
}

// ---------------------------------------------------------------------------
// Bybit API Client
// ---------------------------------------------------------------------------

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
    pub symbol: Option<String>,
    pub category: Option<String>,
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

pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch OHLCV kline data from Bybit API v5.
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} - {}", resp.ret_code, resp.ret_msg);
        }

        let mut candles: Vec<Candle> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Candle {
                        timestamp: row[0].parse().unwrap_or(0),
                        open: row[1].parse().unwrap_or(0.0),
                        high: row[2].parse().unwrap_or(0.0),
                        low: row[3].parse().unwrap_or(0.0),
                        close: row[4].parse().unwrap_or(0.0),
                        volume: row[5].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first; reverse to chronological order
        candles.reverse();
        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility: compute performance metrics
// ---------------------------------------------------------------------------

pub fn compute_sharpe(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = var.sqrt();
    if std < 1e-10 {
        0.0
    } else {
        mean / std * (252.0_f64).sqrt() // annualized
    }
}

pub fn compute_max_drawdown(cumulative_pnl: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0;
    for &val in cumulative_pnl {
        if val > peak {
            peak = val;
        }
        let dd = peak - val;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_output_range() {
        let actor = ActorNetwork::new(8, 64, 32, 1);
        let state = Array1::from_vec(vec![0.1, -0.2, 0.3, 0.05, 0.0, 0.5, -0.1, 0.0]);
        let action = actor.forward(&state);
        assert!(action[0] >= -1.0 && action[0] <= 1.0, "Actor output must be in [-1, 1]");
    }

    #[test]
    fn test_critic_forward() {
        let critic = CriticNetwork::new(8, 1, 64, 32);
        let state = Array1::from_vec(vec![0.1, -0.2, 0.3, 0.05, 0.0, 0.5, -0.1, 0.0]);
        let action = Array1::from_vec(vec![0.5]);
        let q = critic.forward(&state, &action);
        assert!(q.is_finite(), "Q-value must be finite");
    }

    #[test]
    fn test_ou_noise_mean_reverting() {
        let mut noise = OUNoise::new(0.5, 0.0, 0.1);
        let mut samples = Vec::new();
        for _ in 0..1000 {
            samples.push(noise.sample());
        }
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 1.0, "OU noise should be roughly mean-reverting to 0");
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        assert!(buffer.is_empty());
        for i in 0..50 {
            buffer.push(Transition {
                state: Array1::from_vec(vec![i as f64]),
                action: Array1::from_vec(vec![0.0]),
                reward: 1.0,
                next_state: Array1::from_vec(vec![(i + 1) as f64]),
                done: false,
            });
        }
        assert_eq!(buffer.len(), 50);
        let batch = buffer.sample(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_replay_buffer_overflow() {
        let mut buffer = ReplayBuffer::new(10);
        for i in 0..20 {
            buffer.push(Transition {
                state: Array1::from_vec(vec![i as f64]),
                action: Array1::from_vec(vec![0.0]),
                reward: 0.0,
                next_state: Array1::from_vec(vec![0.0]),
                done: false,
            });
        }
        assert_eq!(buffer.len(), 10, "Buffer should cap at capacity");
    }

    #[test]
    fn test_soft_update() {
        let mut target = ActorNetwork::new(4, 16, 8, 1);
        let source = ActorNetwork::new(4, 16, 8, 1);
        let old_w1 = target.w1.clone();
        target.soft_update(&source, 0.005);
        // Weights should have changed
        let diff = (&target.w1 - &old_w1).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0, "Soft update should change target weights");
    }

    #[test]
    fn test_trading_env() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let volumes: Vec<f64> = vec![1000.0; 100];
        let mut env = TradingEnv::new(prices, volumes);
        let state = env.reset();
        assert_eq!(state.len(), 8);

        let (next_state, reward, done) = env.step(0.5);
        assert_eq!(next_state.len(), 8);
        assert!(reward.is_finite());
        assert!(!done || env.current_step + 1 >= 100);
    }

    #[test]
    fn test_ddpg_agent_creation() {
        let agent = DDPGAgent::new(8, 1);
        assert_eq!(agent.gamma, 0.99);
        assert_eq!(agent.tau, 0.005);
        assert_eq!(agent.batch_size, 64);
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.005, 0.015, 0.01];
        let sharpe = compute_sharpe(&returns);
        assert!(sharpe.is_finite());
        assert!(sharpe > 0.0, "Positive returns should give positive Sharpe");
    }

    #[test]
    fn test_max_drawdown() {
        let cumulative = vec![0.0, 0.1, 0.2, 0.15, 0.05, 0.25];
        let dd = compute_max_drawdown(&cumulative);
        assert!((dd - 0.15).abs() < 1e-10, "Max drawdown should be 0.15");
    }
}
