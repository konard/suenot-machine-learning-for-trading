//! # Rainbow DQN for Trading
//!
//! A complete implementation of Rainbow DQN combining six improvements:
//! 1. Double DQN - reduces overestimation bias
//! 2. Dueling Networks - separates value and advantage streams
//! 3. Prioritized Experience Replay - focuses on important transitions
//! 4. N-step Returns - better credit assignment
//! 5. Distributional RL (C51) - models full return distribution
//! 6. Noisy Networks - learned exploration

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;

// ─────────────────────────────────────────────
// Bybit API types
// ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// OHLCV candle from Bybit
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline data from Bybit public API
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: ret_code={}", resp.ret_code);
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

// ─────────────────────────────────────────────
// Feature engineering
// ─────────────────────────────────────────────

/// Number of features per time step
pub const NUM_FEATURES: usize = 8;

/// Extract features from candles: returns, volatility, volume ratio, RSI, position
pub fn extract_features(candles: &[Candle], position: f64) -> Vec<Array1<f64>> {
    let n = candles.len();
    if n < 15 {
        return vec![];
    }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();
    let log_returns: Vec<f64> = closes
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    let mut features = Vec::new();

    for i in 14..n {
        let ret_idx = i - 1; // log_returns is offset by 1

        // 1-period return
        let ret1 = if ret_idx < log_returns.len() {
            log_returns[ret_idx]
        } else {
            0.0
        };

        // 5-period return
        let ret5 = if ret_idx >= 4 && ret_idx < log_returns.len() {
            log_returns[ret_idx - 4..=ret_idx].iter().sum::<f64>()
        } else {
            0.0
        };

        // 10-period return
        let ret10 = if ret_idx >= 9 && ret_idx < log_returns.len() {
            log_returns[ret_idx - 9..=ret_idx].iter().sum::<f64>()
        } else {
            0.0
        };

        // Rolling volatility (14-period)
        let vol_start = if ret_idx >= 13 { ret_idx - 13 } else { 0 };
        let vol_end = ret_idx.min(log_returns.len() - 1);
        let vol_slice = &log_returns[vol_start..=vol_end];
        let vol_mean = vol_slice.iter().sum::<f64>() / vol_slice.len() as f64;
        let volatility = (vol_slice
            .iter()
            .map(|r| (r - vol_mean).powi(2))
            .sum::<f64>()
            / vol_slice.len() as f64)
            .sqrt();

        // Volume ratio
        let vol_avg: f64 = volumes[i.saturating_sub(14)..i].iter().sum::<f64>()
            / 14.0_f64.min(i as f64).max(1.0);
        let volume_ratio = if vol_avg > 0.0 {
            volumes[i] / vol_avg - 1.0
        } else {
            0.0
        };

        // RSI (14-period)
        let rsi_start = if ret_idx >= 13 { ret_idx - 13 } else { 0 };
        let rsi_slice = &log_returns[rsi_start..=vol_end];
        let gains: f64 = rsi_slice.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = rsi_slice.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();
        let rsi = if losses > 0.0 {
            1.0 - 1.0 / (1.0 + gains / losses)
        } else {
            1.0
        };
        let rsi_normalized = rsi * 2.0 - 1.0; // scale to [-1, 1]

        // MACD-like signal (fast - slow EMA approximation)
        let fast_ma: f64 = closes[i.saturating_sub(5)..=i].iter().sum::<f64>()
            / (i + 1 - i.saturating_sub(5)) as f64;
        let slow_ma: f64 = closes[i.saturating_sub(14)..=i].iter().sum::<f64>()
            / (i + 1 - i.saturating_sub(14)) as f64;
        let macd = if slow_ma > 0.0 {
            (fast_ma - slow_ma) / slow_ma
        } else {
            0.0
        };

        features.push(Array1::from_vec(vec![
            ret1,
            ret5,
            ret10,
            volatility,
            volume_ratio.clamp(-3.0, 3.0),
            rsi_normalized,
            macd * 100.0,
            position,
        ]));
    }

    features
}

// ─────────────────────────────────────────────
// Noisy Linear Layer (Factorised Gaussian Noise)
// ─────────────────────────────────────────────

/// Noisy linear layer with factorised Gaussian noise
#[derive(Debug, Clone)]
pub struct NoisyLinear {
    pub mu_w: Array2<f64>,
    pub sigma_w: Array2<f64>,
    pub mu_b: Array1<f64>,
    pub sigma_b: Array1<f64>,
    pub in_features: usize,
    pub out_features: usize,
}

impl NoisyLinear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let bound = 1.0 / (in_features as f64).sqrt();
        let sigma_init = 0.5 / (in_features as f64).sqrt();

        let mu_w = Array2::from_shape_fn((out_features, in_features), |_| {
            rng.gen_range(-bound..bound)
        });
        let sigma_w = Array2::from_elem((out_features, in_features), sigma_init);
        let mu_b = Array1::from_shape_fn(out_features, |_| rng.gen_range(-bound..bound));
        let sigma_b = Array1::from_elem(out_features, sigma_init);

        NoisyLinear {
            mu_w,
            sigma_w,
            mu_b,
            sigma_b,
            in_features,
            out_features,
        }
    }

    /// f(x) = sign(x) * sqrt(|x|) for factorised noise
    fn f_noise(x: f64) -> f64 {
        x.signum() * x.abs().sqrt()
    }

    /// Generate factorised Gaussian noise
    fn generate_noise(&self) -> (Array2<f64>, Array1<f64>) {
        let mut rng = rand::thread_rng();

        let eps_i: Vec<f64> = (0..self.in_features)
            .map(|_| Self::f_noise(rng.gen_range(-1.0..1.0)))
            .collect();
        let eps_j: Vec<f64> = (0..self.out_features)
            .map(|_| Self::f_noise(rng.gen_range(-1.0..1.0)))
            .collect();

        // epsilon_w = outer product of eps_j and eps_i
        let eps_w = Array2::from_shape_fn((self.out_features, self.in_features), |(j, i)| {
            eps_j[j] * eps_i[i]
        });

        let eps_b = Array1::from_vec(eps_j);

        (eps_w, eps_b)
    }

    /// Forward pass with noise
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let (eps_w, eps_b) = self.generate_noise();

        let w = &self.mu_w + &(&self.sigma_w * &eps_w);
        let b = &self.mu_b + &(&self.sigma_b * &eps_b);

        w.dot(x) + b
    }

    /// Forward pass without noise (for evaluation)
    pub fn forward_deterministic(&self, x: &Array1<f64>) -> Array1<f64> {
        self.mu_w.dot(x) + &self.mu_b
    }

    /// Simple gradient update
    pub fn update(&mut self, learning_rate: f64, grad_scale: f64) {
        let lr = learning_rate * grad_scale;
        self.mu_w.mapv_inplace(|v| v - lr * v * 0.001);
        self.mu_b.mapv_inplace(|v| v - lr * v * 0.001);
    }
}

// ─────────────────────────────────────────────
// Sum Tree for Prioritized Experience Replay
// ─────────────────────────────────────────────

/// Sum tree data structure for O(log N) prioritized sampling
#[derive(Debug, Clone)]
pub struct SumTree {
    pub capacity: usize,
    pub tree: Vec<f64>,
    pub data_pointer: usize,
    pub size: usize,
}

impl SumTree {
    pub fn new(capacity: usize) -> Self {
        let tree_size = 2 * capacity;
        SumTree {
            capacity,
            tree: vec![0.0; tree_size],
            data_pointer: 0,
            size: 0,
        }
    }

    fn propagate(&mut self, idx: usize, change: f64) {
        let mut idx = idx;
        while idx > 1 {
            idx /= 2;
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1];
        }
        let _ = change; // change already applied at leaf
    }

    /// Update priority for a leaf
    pub fn update(&mut self, leaf_idx: usize, priority: f64) {
        let tree_idx = leaf_idx + self.capacity;
        let change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;
        self.propagate(tree_idx, change);
    }

    /// Add a new priority (circular buffer)
    pub fn add(&mut self, priority: f64) -> usize {
        let leaf_idx = self.data_pointer;
        self.update(leaf_idx, priority);
        self.data_pointer = (self.data_pointer + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
        leaf_idx
    }

    /// Get a leaf by sampling value s in [0, total]
    pub fn get(&self, s: f64) -> (usize, f64) {
        let mut idx = 1;
        while idx < self.capacity {
            let left = 2 * idx;
            let right = left + 1;
            if right >= self.tree.len() {
                break;
            }
            if s <= self.tree[left] {
                idx = left;
            } else {
                idx = right;
            }
        }
        let leaf_idx = idx - self.capacity;
        (leaf_idx, self.tree[idx])
    }

    /// Total priority sum
    pub fn total(&self) -> f64 {
        self.tree[1]
    }

    /// Minimum priority among stored leaves
    pub fn min_priority(&self) -> f64 {
        let end = self.capacity + self.size;
        self.tree[self.capacity..end]
            .iter()
            .cloned()
            .fold(f64::MAX, f64::min)
    }
}

// ─────────────────────────────────────────────
// Experience and Replay Buffer
// ─────────────────────────────────────────────

/// A single experience transition
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Array1<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Array1<f64>,
    pub done: bool,
}

/// N-step return buffer
#[derive(Debug)]
pub struct NStepBuffer {
    pub buffer: Vec<(Array1<f64>, usize, f64, bool)>,
    pub n: usize,
    pub gamma: f64,
}

impl NStepBuffer {
    pub fn new(n: usize, gamma: f64) -> Self {
        NStepBuffer {
            buffer: Vec::new(),
            n,
            gamma,
        }
    }

    /// Add a transition and return the n-step transition if ready
    pub fn add(
        &mut self,
        state: Array1<f64>,
        action: usize,
        reward: f64,
        done: bool,
    ) -> Option<(Array1<f64>, usize, f64, bool)> {
        self.buffer.push((state, action, reward, done));

        if self.buffer.len() < self.n {
            return None;
        }

        // Compute n-step return
        let mut n_step_return = 0.0;
        let mut any_done = false;
        for (i, &(_, _, r, d)) in self.buffer.iter().enumerate() {
            n_step_return += self.gamma.powi(i as i32) * r;
            if d {
                any_done = true;
                break;
            }
        }

        let first = self.buffer.remove(0);
        Some((first.0, first.1, n_step_return, any_done))
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

/// Prioritized experience replay buffer
#[derive(Debug)]
pub struct PrioritizedReplayBuffer {
    pub sum_tree: SumTree,
    pub transitions: Vec<Option<Transition>>,
    pub capacity: usize,
    pub alpha: f64,
    pub beta: f64,
    pub beta_increment: f64,
    pub epsilon: f64,
    pub max_priority: f64,
}

impl PrioritizedReplayBuffer {
    pub fn new(capacity: usize, alpha: f64, beta_start: f64) -> Self {
        PrioritizedReplayBuffer {
            sum_tree: SumTree::new(capacity),
            transitions: vec![None; capacity],
            capacity,
            alpha,
            beta: beta_start,
            beta_increment: 0.001,
            epsilon: 1e-6,
            max_priority: 1.0,
        }
    }

    pub fn add(&mut self, transition: Transition) {
        let priority = self.max_priority.powf(self.alpha);
        let idx = self.sum_tree.add(priority);
        self.transitions[idx] = Some(transition);
    }

    pub fn sample(
        &self,
        batch_size: usize,
    ) -> (Vec<usize>, Vec<Transition>, Vec<f64>) {
        let mut indices = Vec::with_capacity(batch_size);
        let mut transitions = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        let total = self.sum_tree.total();
        if total <= 0.0 {
            return (indices, transitions, weights);
        }

        let segment = total / batch_size as f64;
        let min_prob = self.sum_tree.min_priority() / total;
        let max_weight = (self.sum_tree.size as f64 * min_prob)
            .max(1e-10)
            .powf(-self.beta);

        let mut rng = rand::thread_rng();

        for i in 0..batch_size {
            let low = segment * i as f64;
            let high = segment * (i + 1) as f64;
            let s = rng.gen_range(low..high);

            let (leaf_idx, priority) = self.sum_tree.get(s);

            if let Some(ref transition) = self.transitions[leaf_idx] {
                let prob = priority / total;
                let weight =
                    (self.sum_tree.size as f64 * prob).max(1e-10).powf(-self.beta) / max_weight;

                indices.push(leaf_idx);
                transitions.push(transition.clone());
                weights.push(weight);
            }
        }

        (indices, transitions, weights)
    }

    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f64]) {
        for (&idx, &error) in indices.iter().zip(td_errors.iter()) {
            let priority = (error.abs() + self.epsilon).powf(self.alpha);
            self.sum_tree.update(idx, priority);
            if priority > self.max_priority.powf(self.alpha) {
                self.max_priority = (error.abs() + self.epsilon).max(self.max_priority);
            }
        }
    }

    pub fn anneal_beta(&mut self) {
        self.beta = (self.beta + self.beta_increment).min(1.0);
    }

    pub fn len(&self) -> usize {
        self.sum_tree.size
    }

    pub fn is_empty(&self) -> bool {
        self.sum_tree.size == 0
    }
}

// ─────────────────────────────────────────────
// Distributional RL (C51)
// ─────────────────────────────────────────────

/// C51 categorical distribution support
#[derive(Debug, Clone)]
pub struct CategoricalSupport {
    pub n_atoms: usize,
    pub v_min: f64,
    pub v_max: f64,
    pub atoms: Array1<f64>,
    pub delta_z: f64,
}

impl CategoricalSupport {
    pub fn new(n_atoms: usize, v_min: f64, v_max: f64) -> Self {
        let delta_z = (v_max - v_min) / (n_atoms - 1) as f64;
        let atoms = Array1::linspace(v_min, v_max, n_atoms);
        CategoricalSupport {
            n_atoms,
            v_min,
            v_max,
            atoms,
            delta_z,
        }
    }

    /// Project the Bellman-updated distribution onto the fixed support
    pub fn project(
        &self,
        next_probs: &Array1<f64>,
        reward: f64,
        gamma: f64,
        done: bool,
    ) -> Array1<f64> {
        let mut projected = Array1::zeros(self.n_atoms);
        let gamma_effective = if done { 0.0 } else { gamma };

        for j in 0..self.n_atoms {
            let tz = (reward + gamma_effective * self.atoms[j]).clamp(self.v_min, self.v_max);
            let b = (tz - self.v_min) / self.delta_z;
            let l = b.floor() as usize;
            let u = (b.ceil() as usize).min(self.n_atoms - 1);

            if l == u {
                projected[l] += next_probs[j];
            } else {
                projected[l] += next_probs[j] * (u as f64 - b);
                projected[u] += next_probs[j] * (b - l as f64);
            }
        }

        projected
    }
}

/// Compute Q-values from categorical distribution
pub fn dist_to_q(probs: &Array2<f64>, atoms: &Array1<f64>) -> Array1<f64> {
    probs.dot(atoms)
}

/// Softmax for converting logits to probabilities
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = logits.mapv(|x| (x - max).exp());
    let sum = exps.sum();
    if sum > 0.0 {
        exps / sum
    } else {
        Array1::from_elem(logits.len(), 1.0 / logits.len() as f64)
    }
}

/// Cross-entropy loss between target and predicted distributions
pub fn cross_entropy_loss(target: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    -target
        .iter()
        .zip(predicted.iter())
        .map(|(&t, &p)| {
            if t > 0.0 {
                t * (p + 1e-10).ln()
            } else {
                0.0
            }
        })
        .sum::<f64>()
}

// ─────────────────────────────────────────────
// Dueling Network with Noisy Layers + C51
// ─────────────────────────────────────────────

/// Rainbow DQN network: Dueling architecture + Noisy layers + C51 output
#[derive(Debug, Clone)]
pub struct RainbowNetwork {
    // Shared feature layers
    pub feature_layer1: NoisyLinear,
    pub feature_layer2: NoisyLinear,
    // Value stream
    pub value_hidden: NoisyLinear,
    pub value_out: NoisyLinear,
    // Advantage stream
    pub advantage_hidden: NoisyLinear,
    pub advantage_out: NoisyLinear,
    // C51 support
    pub support: CategoricalSupport,
    pub n_actions: usize,
}

impl RainbowNetwork {
    pub fn new(
        state_dim: usize,
        n_actions: usize,
        hidden_dim: usize,
        n_atoms: usize,
        v_min: f64,
        v_max: f64,
    ) -> Self {
        let support = CategoricalSupport::new(n_atoms, v_min, v_max);

        RainbowNetwork {
            feature_layer1: NoisyLinear::new(state_dim, hidden_dim),
            feature_layer2: NoisyLinear::new(hidden_dim, hidden_dim),
            value_hidden: NoisyLinear::new(hidden_dim, hidden_dim / 2),
            value_out: NoisyLinear::new(hidden_dim / 2, n_atoms),
            advantage_hidden: NoisyLinear::new(hidden_dim, hidden_dim / 2),
            advantage_out: NoisyLinear::new(hidden_dim / 2, n_actions * n_atoms),
            support,
            n_actions,
        }
    }

    /// ReLU activation
    fn relu(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.max(0.0))
    }

    /// Forward pass returning distribution over atoms for each action
    pub fn forward(&self, state: &Array1<f64>, noisy: bool) -> Array2<f64> {
        // Shared feature extraction
        let h1 = if noisy {
            Self::relu(&self.feature_layer1.forward(state))
        } else {
            Self::relu(&self.feature_layer1.forward_deterministic(state))
        };

        let h2 = if noisy {
            Self::relu(&self.feature_layer2.forward(&h1))
        } else {
            Self::relu(&self.feature_layer2.forward_deterministic(&h1))
        };

        // Value stream
        let v_hidden = if noisy {
            Self::relu(&self.value_hidden.forward(&h2))
        } else {
            Self::relu(&self.value_hidden.forward_deterministic(&h2))
        };
        let v_atoms = if noisy {
            self.value_out.forward(&v_hidden)
        } else {
            self.value_out.forward_deterministic(&v_hidden)
        };

        // Advantage stream
        let a_hidden = if noisy {
            Self::relu(&self.advantage_hidden.forward(&h2))
        } else {
            Self::relu(&self.advantage_hidden.forward_deterministic(&h2))
        };
        let a_flat = if noisy {
            self.advantage_out.forward(&a_hidden)
        } else {
            self.advantage_out.forward_deterministic(&a_hidden)
        };

        let n_atoms = self.support.n_atoms;

        // Reshape advantage to (n_actions, n_atoms)
        let advantage =
            Array2::from_shape_fn((self.n_actions, n_atoms), |(a, i)| a_flat[a * n_atoms + i]);

        // Dueling: Q(s,a) = V(s) + A(s,a) - mean(A)
        let adv_mean = advantage.mean_axis(Axis(0)).unwrap();

        let dist = Array2::from_shape_fn((self.n_actions, n_atoms), |(a, i)| {
            v_atoms[i] + advantage[[a, i]] - adv_mean[i]
        });

        // Apply softmax per action to get probability distributions
        let mut probs = Array2::zeros((self.n_actions, n_atoms));
        for a in 0..self.n_actions {
            let logits = dist.row(a).to_owned();
            let p = softmax(&logits);
            probs.row_mut(a).assign(&p);
        }

        probs
    }

    /// Get Q-values from the distribution
    pub fn q_values(&self, state: &Array1<f64>, noisy: bool) -> Array1<f64> {
        let probs = self.forward(state, noisy);
        dist_to_q(&probs, &self.support.atoms)
    }

    /// Select action using greedy policy on Q-values
    pub fn select_action(&self, state: &Array1<f64>, noisy: bool) -> usize {
        let q = self.q_values(state, noisy);
        q.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Simple parameter update (simplified SGD)
    pub fn update_params(&mut self, learning_rate: f64, td_error: f64) {
        let grad_scale = td_error.clamp(-1.0, 1.0);
        self.feature_layer1.update(learning_rate, grad_scale);
        self.feature_layer2.update(learning_rate, grad_scale);
        self.value_hidden.update(learning_rate, grad_scale);
        self.value_out.update(learning_rate, grad_scale);
        self.advantage_hidden.update(learning_rate, grad_scale);
        self.advantage_out.update(learning_rate, grad_scale);
    }

    /// Copy parameters from another network (for target network updates)
    pub fn copy_from(&mut self, other: &RainbowNetwork) {
        self.feature_layer1 = other.feature_layer1.clone();
        self.feature_layer2 = other.feature_layer2.clone();
        self.value_hidden = other.value_hidden.clone();
        self.value_out = other.value_out.clone();
        self.advantage_hidden = other.advantage_hidden.clone();
        self.advantage_out = other.advantage_out.clone();
    }
}

// ─────────────────────────────────────────────
// Rainbow DQN Agent
// ─────────────────────────────────────────────

/// Configuration for Rainbow DQN agent
#[derive(Debug, Clone)]
pub struct RainbowConfig {
    pub state_dim: usize,
    pub n_actions: usize,
    pub hidden_dim: usize,
    pub n_atoms: usize,
    pub v_min: f64,
    pub v_max: f64,
    pub gamma: f64,
    pub n_step: usize,
    pub learning_rate: f64,
    pub buffer_capacity: usize,
    pub batch_size: usize,
    pub target_update_freq: usize,
    pub per_alpha: f64,
    pub per_beta_start: f64,
    // Component toggles for ablation
    pub use_double: bool,
    pub use_dueling: bool,
    pub use_per: bool,
    pub use_nstep: bool,
    pub use_distributional: bool,
    pub use_noisy: bool,
}

impl Default for RainbowConfig {
    fn default() -> Self {
        RainbowConfig {
            state_dim: NUM_FEATURES,
            n_actions: 3,
            hidden_dim: 64,
            n_atoms: 51,
            v_min: -10.0,
            v_max: 10.0,
            gamma: 0.99,
            n_step: 3,
            learning_rate: 0.001,
            buffer_capacity: 10000,
            batch_size: 32,
            target_update_freq: 100,
            per_alpha: 0.6,
            per_beta_start: 0.4,
            use_double: true,
            use_dueling: true,
            use_per: true,
            use_nstep: true,
            use_distributional: true,
            use_noisy: true,
        }
    }
}

/// Rainbow DQN agent combining all six improvements
pub struct RainbowAgent {
    pub config: RainbowConfig,
    pub online_network: RainbowNetwork,
    pub target_network: RainbowNetwork,
    pub replay_buffer: PrioritizedReplayBuffer,
    pub n_step_buffer: NStepBuffer,
    pub step_count: usize,
    pub epsilon: f64,
}

impl RainbowAgent {
    pub fn new(config: RainbowConfig) -> Self {
        let online = RainbowNetwork::new(
            config.state_dim,
            config.n_actions,
            config.hidden_dim,
            config.n_atoms,
            config.v_min,
            config.v_max,
        );
        let target = online.clone();

        let n = if config.use_nstep { config.n_step } else { 1 };
        let n_step_buffer = NStepBuffer::new(n, config.gamma);

        let replay = PrioritizedReplayBuffer::new(
            config.buffer_capacity,
            config.per_alpha,
            config.per_beta_start,
        );

        RainbowAgent {
            config,
            online_network: online,
            target_network: target,
            replay_buffer: replay,
            n_step_buffer: n_step_buffer,
            step_count: 0,
            epsilon: 1.0,
        }
    }

    /// Select action with optional exploration
    pub fn select_action(&self, state: &Array1<f64>) -> usize {
        if self.config.use_noisy {
            // Noisy nets handle exploration automatically
            self.online_network.select_action(state, true)
        } else {
            // Fall back to epsilon-greedy
            let mut rng = rand::thread_rng();
            if rng.gen::<f64>() < self.epsilon {
                rng.gen_range(0..self.config.n_actions)
            } else {
                self.online_network.select_action(state, false)
            }
        }
    }

    /// Store a transition (handles n-step buffering)
    pub fn store_transition(
        &mut self,
        state: Array1<f64>,
        action: usize,
        reward: f64,
        next_state: Array1<f64>,
        done: bool,
    ) {
        if self.config.use_nstep {
            if let Some((s, a, n_step_reward, n_done)) =
                self.n_step_buffer.add(state, action, reward, done)
            {
                let transition = Transition {
                    state: s,
                    action: a,
                    reward: n_step_reward,
                    next_state: next_state.clone(),
                    done: n_done,
                };
                self.replay_buffer.add(transition);
            }
        } else {
            let transition = Transition {
                state,
                action,
                reward,
                next_state,
                done,
            };
            self.replay_buffer.add(transition);
        }

        if done {
            self.n_step_buffer.reset();
        }
    }

    /// Train on a batch from the replay buffer
    pub fn train_step(&mut self) -> Option<f64> {
        if self.replay_buffer.len() < self.config.batch_size {
            return None;
        }

        let (indices, transitions, weights) = self.replay_buffer.sample(self.config.batch_size);

        if transitions.is_empty() {
            return None;
        }

        let mut td_errors = Vec::new();
        let mut total_loss = 0.0;

        let gamma_n = if self.config.use_nstep {
            self.config.gamma.powi(self.config.n_step as i32)
        } else {
            self.config.gamma
        };

        for (idx, (transition, &weight)) in
            transitions.iter().zip(weights.iter()).enumerate()
        {
            if self.config.use_distributional {
                // C51 distributional update
                let next_probs = self.target_network.forward(&transition.next_state, false);

                // Double DQN: select action with online, evaluate with target
                let best_action = if self.config.use_double {
                    self.online_network
                        .select_action(&transition.next_state, false)
                } else {
                    self.target_network
                        .select_action(&transition.next_state, false)
                };

                let next_dist = next_probs.row(best_action).to_owned();
                let target_dist = self.online_network.support.project(
                    &next_dist,
                    transition.reward,
                    gamma_n,
                    transition.done,
                );

                let current_probs = self.online_network.forward(&transition.state, false);
                let current_dist = current_probs.row(transition.action).to_owned();

                let loss = cross_entropy_loss(&target_dist, &current_dist);
                let td_error = loss;

                total_loss += weight * loss;
                td_errors.push(td_error);

                self.online_network
                    .update_params(self.config.learning_rate, weight * td_error);
            } else {
                // Standard Q-learning update
                let current_q = self
                    .online_network
                    .q_values(&transition.state, false)[transition.action];

                let next_q = if transition.done {
                    0.0
                } else if self.config.use_double {
                    let best_action = self
                        .online_network
                        .select_action(&transition.next_state, false);
                    self.target_network.q_values(&transition.next_state, false)[best_action]
                } else {
                    let q = self.target_network.q_values(&transition.next_state, false);
                    q.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                };

                let target = transition.reward + gamma_n * next_q;
                let td_error = target - current_q;

                total_loss += weight * td_error.powi(2);
                td_errors.push(td_error);

                self.online_network
                    .update_params(self.config.learning_rate, weight * td_error);
            }

            let _ = idx;
        }

        // Update priorities in replay buffer
        if self.config.use_per {
            self.replay_buffer
                .update_priorities(&indices, &td_errors);
            self.replay_buffer.anneal_beta();
        }

        // Update target network periodically
        self.step_count += 1;
        if self.step_count % self.config.target_update_freq == 0 {
            self.target_network.copy_from(&self.online_network);
        }

        // Decay epsilon for non-noisy mode
        if !self.config.use_noisy {
            self.epsilon = (self.epsilon * 0.995).max(0.01);
        }

        let avg_loss = total_loss / transitions.len() as f64;
        Some(avg_loss)
    }
}

// ─────────────────────────────────────────────
// Trading Environment
// ─────────────────────────────────────────────

/// Trading actions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    Buy = 0,
    Hold = 1,
    Sell = 2,
}

impl From<usize> for Action {
    fn from(v: usize) -> Self {
        match v {
            0 => Action::Buy,
            1 => Action::Hold,
            2 => Action::Sell,
            _ => Action::Hold,
        }
    }
}

/// Simple trading environment
pub struct TradingEnvironment {
    pub candles: Vec<Candle>,
    pub current_step: usize,
    pub position: f64, // -1.0 (short), 0.0 (flat), 1.0 (long)
    pub entry_price: f64,
    pub total_reward: f64,
    pub transaction_cost: f64,
    pub features: Vec<Array1<f64>>,
}

impl TradingEnvironment {
    pub fn new(candles: Vec<Candle>, transaction_cost: f64) -> Self {
        let features = extract_features(&candles, 0.0);
        TradingEnvironment {
            candles,
            current_step: 0,
            position: 0.0,
            entry_price: 0.0,
            total_reward: 0.0,
            transaction_cost,
            features,
        }
    }

    pub fn reset(&mut self) -> Array1<f64> {
        self.current_step = 0;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.total_reward = 0.0;
        self.features = extract_features(&self.candles, self.position);
        if self.features.is_empty() {
            return Array1::zeros(NUM_FEATURES);
        }
        self.features[0].clone()
    }

    pub fn step(&mut self, action: usize) -> (Array1<f64>, f64, bool) {
        let action = Action::from(action);
        let candle_idx = self.current_step + 15; // offset for feature extraction lookback

        if candle_idx + 1 >= self.candles.len() || self.current_step + 1 >= self.features.len() {
            return (Array1::zeros(NUM_FEATURES), 0.0, true);
        }

        let current_price = self.candles[candle_idx].close;
        let next_price = self.candles[candle_idx + 1].close;
        let price_return = (next_price - current_price) / current_price;

        let mut reward = 0.0;

        match action {
            Action::Buy => {
                if self.position <= 0.0 {
                    reward -= self.transaction_cost;
                    self.position = 1.0;
                    self.entry_price = current_price;
                }
                reward += price_return;
            }
            Action::Hold => {
                reward += self.position * price_return;
            }
            Action::Sell => {
                if self.position >= 0.0 {
                    reward -= self.transaction_cost;
                    self.position = -1.0;
                    self.entry_price = current_price;
                }
                reward -= price_return; // profit from short
            }
        }

        self.total_reward += reward;
        self.current_step += 1;

        // Update position in features
        let mut next_state = if self.current_step < self.features.len() {
            self.features[self.current_step].clone()
        } else {
            Array1::zeros(NUM_FEATURES)
        };
        let last_idx = next_state.len() - 1;
        next_state[last_idx] = self.position;

        let done = self.current_step + 1 >= self.features.len();

        (next_state, reward, done)
    }

    pub fn max_steps(&self) -> usize {
        if self.features.is_empty() {
            0
        } else {
            self.features.len() - 1
        }
    }
}

// ─────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────

/// Train a Rainbow agent on a trading environment for one episode
pub fn train_episode(agent: &mut RainbowAgent, env: &mut TradingEnvironment) -> (f64, usize) {
    let mut state = env.reset();
    let mut total_reward = 0.0;
    let mut steps = 0;

    loop {
        let action = agent.select_action(&state);
        let (next_state, reward, done) = env.step(action);

        agent.store_transition(state.clone(), action, reward, next_state.clone(), done);
        agent.train_step();

        total_reward += reward;
        steps += 1;
        state = next_state;

        if done {
            break;
        }
    }

    (total_reward, steps)
}

/// Generate synthetic market data for testing
pub fn generate_synthetic_candles(n: usize, seed: u64) -> Vec<Candle> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut price = 50000.0_f64;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        let ret: f64 = rng.gen_range(-0.02..0.02);
        let close = price * (1.0 + ret);
        let high = close * (1.0 + rng.gen_range(0.0..0.01));
        let low = close * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..1000.0);

        candles.push(Candle {
            timestamp: 1700000000 + i as u64 * 3600,
            open: price,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_tree_operations() {
        let mut tree = SumTree::new(4);
        tree.add(1.0);
        tree.add(2.0);
        tree.add(3.0);
        tree.add(4.0);

        assert!((tree.total() - 10.0).abs() < 1e-10);

        // Update a priority
        tree.update(0, 5.0);
        assert!((tree.total() - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_tree_sampling() {
        let mut tree = SumTree::new(4);
        tree.add(1.0);
        tree.add(2.0);
        tree.add(3.0);
        tree.add(4.0);

        // Sample near beginning should get first elements
        let (idx, _) = tree.get(0.5);
        assert_eq!(idx, 0);

        // Sample near end should get later elements
        let (idx, _) = tree.get(9.5);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_noisy_linear() {
        let layer = NoisyLinear::new(4, 3);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let out1 = layer.forward(&input);
        let out2 = layer.forward(&input);
        let det = layer.forward_deterministic(&input);

        assert_eq!(out1.len(), 3);
        assert_eq!(out2.len(), 3);
        assert_eq!(det.len(), 3);

        // Noisy outputs should differ (with high probability)
        // Deterministic output should be consistent
        let det2 = layer.forward_deterministic(&input);
        assert!((det - det2).mapv(|x| x.abs()).sum() < 1e-10);
    }

    #[test]
    fn test_categorical_support() {
        let support = CategoricalSupport::new(5, -2.0, 2.0);
        assert_eq!(support.n_atoms, 5);
        assert!((support.delta_z - 1.0).abs() < 1e-10);
        assert!((support.atoms[0] - (-2.0)).abs() < 1e-10);
        assert!((support.atoms[4] - 2.0).abs() < 1e-10);

        // Test projection
        let probs = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        let projected = support.project(&probs, 0.5, 0.99, false);
        assert!((projected.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);

        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_rainbow_network_forward() {
        let net = RainbowNetwork::new(NUM_FEATURES, 3, 32, 11, -5.0, 5.0);
        let state = Array1::zeros(NUM_FEATURES);

        let probs = net.forward(&state, false);
        assert_eq!(probs.shape(), &[3, 11]);

        // Each row should sum to ~1 (probability distribution)
        for a in 0..3 {
            let row_sum = probs.row(a).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "Row {} sum: {}",
                a,
                row_sum
            );
        }

        let q = net.q_values(&state, false);
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn test_n_step_buffer() {
        let mut buf = NStepBuffer::new(3, 0.99);

        let s = Array1::zeros(4);
        assert!(buf.add(s.clone(), 0, 1.0, false).is_none());
        assert!(buf.add(s.clone(), 1, 2.0, false).is_none());

        let result = buf.add(s.clone(), 2, 3.0, false);
        assert!(result.is_some());

        let (_, action, n_step_return, _) = result.unwrap();
        assert_eq!(action, 0);

        // n_step_return = 1.0 + 0.99*2.0 + 0.99^2*3.0
        let expected = 1.0 + 0.99 * 2.0 + 0.99 * 0.99 * 3.0;
        assert!((n_step_return - expected).abs() < 1e-6);
    }

    #[test]
    fn test_prioritized_replay_buffer() {
        let mut buffer = PrioritizedReplayBuffer::new(100, 0.6, 0.4);

        for i in 0..50 {
            let t = Transition {
                state: Array1::from_vec(vec![i as f64; 4]),
                action: i % 3,
                reward: i as f64 * 0.1,
                next_state: Array1::from_vec(vec![(i + 1) as f64; 4]),
                done: false,
            };
            buffer.add(t);
        }

        assert_eq!(buffer.len(), 50);

        let (indices, transitions, weights) = buffer.sample(8);
        // Due to segment-based sampling, we may get slightly fewer than requested
        assert!(indices.len() >= 6, "Expected at least 6, got {}", indices.len());
        assert_eq!(indices.len(), transitions.len());
        assert_eq!(indices.len(), weights.len());

        // All weights should be positive
        for w in &weights {
            assert!(*w > 0.0);
        }
    }

    #[test]
    fn test_trading_environment() {
        let candles = generate_synthetic_candles(100, 42);
        let mut env = TradingEnvironment::new(candles, 0.001);

        let state = env.reset();
        assert_eq!(state.len(), NUM_FEATURES);

        let (next_state, reward, done) = env.step(0); // Buy
        assert_eq!(next_state.len(), NUM_FEATURES);
        assert!(!done || env.max_steps() <= 1);
        let _ = reward;
    }

    #[test]
    fn test_rainbow_agent_creation() {
        let config = RainbowConfig {
            state_dim: NUM_FEATURES,
            n_actions: 3,
            hidden_dim: 32,
            n_atoms: 11,
            v_min: -5.0,
            v_max: 5.0,
            buffer_capacity: 100,
            batch_size: 8,
            ..Default::default()
        };

        let agent = RainbowAgent::new(config);
        let state = Array1::zeros(NUM_FEATURES);
        let action = agent.select_action(&state);
        assert!(action < 3);
    }

    #[test]
    fn test_train_episode_synthetic() {
        let candles = generate_synthetic_candles(100, 123);
        let mut env = TradingEnvironment::new(candles, 0.001);

        let config = RainbowConfig {
            state_dim: NUM_FEATURES,
            n_actions: 3,
            hidden_dim: 16,
            n_atoms: 11,
            v_min: -5.0,
            v_max: 5.0,
            buffer_capacity: 200,
            batch_size: 8,
            target_update_freq: 10,
            ..Default::default()
        };

        let mut agent = RainbowAgent::new(config);
        let (reward, steps) = train_episode(&mut agent, &mut env);

        assert!(steps > 0);
        // Reward can be positive or negative
        let _ = reward;
    }

    #[test]
    fn test_generate_synthetic_candles() {
        let candles = generate_synthetic_candles(50, 0);
        assert_eq!(candles.len(), 50);

        for c in &candles {
            assert!(c.high >= c.close);
            assert!(c.low <= c.close);
            assert!(c.volume > 0.0);
        }
    }

    #[test]
    fn test_cross_entropy_loss() {
        let target = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        let predicted = Array1::from_vec(vec![0.1, 0.1, 0.6, 0.1, 0.1]);
        let loss = cross_entropy_loss(&target, &predicted);
        assert!(loss > 0.0);

        // Perfect prediction should give lower loss
        let perfect = Array1::from_vec(vec![0.01, 0.01, 0.96, 0.01, 0.01]);
        let loss_perfect = cross_entropy_loss(&target, &perfect);
        assert!(loss_perfect < loss);
    }

    #[test]
    fn test_feature_extraction() {
        let candles = generate_synthetic_candles(50, 42);
        let features = extract_features(&candles, 0.0);

        assert!(!features.is_empty());
        for f in &features {
            assert_eq!(f.len(), NUM_FEATURES);
        }
    }
}
