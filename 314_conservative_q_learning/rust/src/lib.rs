//! # Conservative Q-Learning (CQL) for Trading
//!
//! This crate implements Conservative Q-Learning for offline reinforcement learning
//! applied to cryptocurrency trading. CQL learns conservative trading policies from
//! historical data without requiring live market interaction.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Trading actions
// ---------------------------------------------------------------------------

/// Discrete trading actions available to the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingAction {
    StrongSell = 0,
    Sell = 1,
    Hold = 2,
    Buy = 3,
    StrongBuy = 4,
}

impl TradingAction {
    pub const NUM_ACTIONS: usize = 5;

    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::StrongSell,
            1 => Self::Sell,
            2 => Self::Hold,
            3 => Self::Buy,
            4 => Self::StrongBuy,
            _ => Self::Hold,
        }
    }

    /// Position delta implied by this action (fraction of capital).
    pub fn position_delta(&self) -> f64 {
        match self {
            Self::StrongSell => -1.0,
            Self::Sell => -0.5,
            Self::Hold => 0.0,
            Self::Buy => 0.5,
            Self::StrongBuy => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Simple feedforward Q-network
// ---------------------------------------------------------------------------

/// A simple feedforward neural network that approximates the Q-function.
///
/// Architecture: input -> hidden1 (ReLU) -> hidden2 (ReLU) -> output (NUM_ACTIONS)
#[derive(Debug, Clone)]
pub struct QNetwork {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
    w3: Array2<f64>,
    b3: Array1<f64>,
    input_size: usize,
    hidden1_size: usize,
    hidden2_size: usize,
    output_size: usize,
}

impl QNetwork {
    /// Create a new Q-network with Xavier initialization.
    pub fn new(input_size: usize, hidden1: usize, hidden2: usize) -> Self {
        let mut rng = rand::thread_rng();
        let output_size = TradingAction::NUM_ACTIONS;

        let mut xavier = |fan_in: usize, fan_out: usize| -> f64 {
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
            rng.gen_range(-limit..limit)
        };

        let w1 = Array2::from_shape_fn((input_size, hidden1), |_| xavier(input_size, hidden1));
        let b1 = Array1::zeros(hidden1);
        let w2 = Array2::from_shape_fn((hidden1, hidden2), |_| xavier(hidden1, hidden2));
        let b2 = Array1::zeros(hidden2);
        let w3 = Array2::from_shape_fn((hidden2, output_size), |_| xavier(hidden2, output_size));
        let b3 = Array1::zeros(output_size);

        Self {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            input_size,
            hidden1_size: hidden1,
            hidden2_size: hidden2,
            output_size,
        }
    }

    /// Forward pass: returns Q-values for all actions given state.
    pub fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        // Layer 1
        let z1 = state.dot(&self.w1) + &self.b1;
        let h1 = z1.mapv(|x| x.max(0.0)); // ReLU

        // Layer 2
        let z2 = h1.dot(&self.w2) + &self.b2;
        let h2 = z2.mapv(|x| x.max(0.0)); // ReLU

        // Output
        h2.dot(&self.w3) + &self.b3
    }

    /// Forward pass for a batch of states.
    pub fn forward_batch(&self, states: &Array2<f64>) -> Array2<f64> {
        let z1 = states.dot(&self.w1) + &self.b1;
        let h1 = z1.mapv(|x| x.max(0.0));
        let z2 = h1.dot(&self.w2) + &self.b2;
        let h2 = z2.mapv(|x| x.max(0.0));
        h2.dot(&self.w3) + &self.b3
    }

    /// Apply a simple gradient update: param -= lr * gradient.
    /// This performs a numerical-gradient-free update using the CQL loss components.
    pub fn update_from_loss_components(
        &mut self,
        states: &Array2<f64>,
        actions: &[usize],
        td_targets: &Array1<f64>,
        alpha: f64,
        learning_rate: f64,
        _num_ood_samples: usize,
    ) {
        let batch_size = states.nrows();

        // For each sample, compute gradients via finite differences style update
        // We use a simplified gradient: push Q(s,a_dataset) toward td_target,
        // push down Q for random (OOD) actions, push up Q for dataset actions.
        for i in 0..batch_size {
            let state = states.row(i).to_owned();
            let action = actions[i];
            let target = td_targets[i];

            // --- Forward pass with intermediate activations ---
            let z1 = state.dot(&self.w1) + &self.b1;
            let h1 = z1.mapv(|x| x.max(0.0));
            let mask1 = z1.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

            let z2 = h1.dot(&self.w2) + &self.b2;
            let h2 = z2.mapv(|x| x.max(0.0));
            let mask2 = z2.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

            let q_values = h2.dot(&self.w3) + &self.b3;
            let q_action = q_values[action];

            // --- TD gradient: dL/dQ = 2*(Q - target) for the chosen action ---
            let td_error = q_action - target;
            let mut dq = Array1::zeros(self.output_size);
            dq[action] = 2.0 * td_error;

            // --- CQL gradient: push down OOD actions, push up dataset action ---
            // logsumexp approximation gradient
            let q_max = q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_q: Array1<f64> = q_values.mapv(|q| (q - q_max).exp());
            let sum_exp: f64 = exp_q.sum();
            // d(logsumexp)/dQ_a = exp(Q_a) / sum(exp(Q))  (softmax)
            let softmax = exp_q.mapv(|e| e / sum_exp);

            // CQL penalty gradient: alpha * (softmax - one_hot(dataset_action))
            let mut cql_grad = softmax.clone();
            cql_grad[action] -= 1.0;
            dq = dq + alpha * &cql_grad;

            // Scale
            let scale = learning_rate / batch_size as f64;

            // --- Backprop through layer 3 ---
            let dw3_col = &h2 * scale;
            for a in 0..self.output_size {
                let g = dq[a];
                for j in 0..self.hidden2_size {
                    self.w3[[j, a]] -= g * dw3_col[j];
                }
                self.b3[a] -= g * scale;
            }

            // dh2
            let dh2 = dq.dot(&self.w3.t()) * &mask2;

            // --- Backprop through layer 2 ---
            let dw2_col = &h1 * scale;
            for j in 0..self.hidden2_size {
                let g = dh2[j];
                for k in 0..self.hidden1_size {
                    self.w2[[k, j]] -= g * dw2_col[k];
                }
                self.b2[j] -= g * scale;
            }

            // dh1
            let dh1 = dh2.dot(&self.w2.t()) * &mask1;

            // --- Backprop through layer 1 ---
            for j in 0..self.hidden1_size {
                let g = dh1[j];
                for k in 0..self.input_size {
                    self.w1[[k, j]] -= g * state[k] * scale;
                }
                self.b1[j] -= g * scale;
            }
        }
    }

    /// Soft-update target network parameters: target = tau*online + (1-tau)*target.
    pub fn soft_update_from(&mut self, online: &QNetwork, tau: f64) {
        self.w1 = &online.w1 * tau + &self.w1 * (1.0 - tau);
        self.b1 = &online.b1 * tau + &self.b1 * (1.0 - tau);
        self.w2 = &online.w2 * tau + &self.w2 * (1.0 - tau);
        self.b2 = &online.b2 * tau + &self.b2 * (1.0 - tau);
        self.w3 = &online.w3 * tau + &self.w3 * (1.0 - tau);
        self.b3 = &online.b3 * tau + &self.b3 * (1.0 - tau);
    }

    /// Return the number of parameters in this network.
    pub fn num_parameters(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len() + self.w3.len() + self.b3.len()
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }
}

// ---------------------------------------------------------------------------
// Transition & Replay Buffer
// ---------------------------------------------------------------------------

/// A single transition (s, a, r, s', done).
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Array1<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Array1<f64>,
    pub done: bool,
}

/// Replay buffer for storing offline trading experiences.
#[derive(Debug)]
pub struct ReplayBuffer {
    transitions: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            transitions: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.transitions.len() >= self.capacity {
            self.transitions.pop_front();
        }
        self.transitions.push_back(transition);
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Sample a random batch of transitions.
    pub fn sample(&self, batch_size: usize) -> Vec<Transition> {
        let mut rng = rand::thread_rng();
        let len = self.transitions.len();
        (0..batch_size)
            .map(|_| {
                let idx = rng.gen_range(0..len);
                self.transitions[idx].clone()
            })
            .collect()
    }

    /// Return all transitions as vectors of components.
    pub fn all_transitions(&self) -> Vec<Transition> {
        self.transitions.iter().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// CQL Agent
// ---------------------------------------------------------------------------

/// Configuration for the CQL agent.
#[derive(Debug, Clone)]
pub struct CQLConfig {
    pub state_dim: usize,
    pub hidden1: usize,
    pub hidden2: usize,
    pub gamma: f64,
    pub alpha: f64,
    pub learning_rate: f64,
    pub tau: f64,
    pub epsilon: f64,
    pub num_ood_samples: usize,
}

impl Default for CQLConfig {
    fn default() -> Self {
        Self {
            state_dim: 7,
            hidden1: 64,
            hidden2: 32,
            gamma: 0.99,
            alpha: 1.0,
            learning_rate: 1e-3,
            tau: 0.005,
            epsilon: 0.1,
            num_ood_samples: 10,
        }
    }
}

/// Conservative Q-Learning agent for offline trading.
pub struct CQLAgent {
    pub q_network: QNetwork,
    pub target_network: QNetwork,
    pub config: CQLConfig,
    pub training_losses: Vec<f64>,
}

impl CQLAgent {
    pub fn new(config: CQLConfig) -> Self {
        let q_network = QNetwork::new(config.state_dim, config.hidden1, config.hidden2);
        let target_network = q_network.clone();
        Self {
            q_network,
            target_network,
            config,
            training_losses: Vec::new(),
        }
    }

    /// Select an action using epsilon-greedy policy.
    pub fn select_action(&self, state: &Array1<f64>, explore: bool) -> usize {
        let mut rng = rand::thread_rng();
        if explore && rng.gen::<f64>() < self.config.epsilon {
            rng.gen_range(0..TradingAction::NUM_ACTIONS)
        } else {
            let q_values = self.q_network.forward(state);
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        }
    }

    /// Compute the CQL loss on a batch (returns total_loss, td_loss, cql_penalty).
    pub fn compute_cql_loss(&self, batch: &[Transition]) -> (f64, f64, f64) {
        let batch_size = batch.len() as f64;
        let mut td_loss = 0.0;
        let mut cql_penalty = 0.0;
        for t in batch {
            let q_values = self.q_network.forward(&t.state);
            let q_action = q_values[t.action];

            // TD target
            let next_q = self.target_network.forward(&t.next_state);
            let max_next_q = next_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let td_target = if t.done {
                t.reward
            } else {
                t.reward + self.config.gamma * max_next_q
            };
            td_loss += (q_action - td_target).powi(2);

            // CQL penalty: logsumexp(Q(s, .)) - Q(s, a_dataset)
            let q_max = q_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let logsumexp: f64 =
                q_max + q_values.iter().map(|q| (q - q_max).exp()).sum::<f64>().ln();

            // Also sample random OOD actions to improve logsumexp estimate
            // (in this discrete case, we already sum over all actions above)
            cql_penalty += logsumexp - q_action;
        }

        td_loss /= batch_size;
        cql_penalty /= batch_size;
        let total_loss = td_loss + self.config.alpha * cql_penalty;

        (total_loss, td_loss, cql_penalty)
    }

    /// Perform a single training step on the given batch.
    pub fn train_step(&mut self, batch: &[Transition]) -> f64 {
        let batch_size = batch.len();
        let state_dim = self.config.state_dim;

        // Build batch arrays
        let mut states = Array2::zeros((batch_size, state_dim));
        let mut actions = Vec::with_capacity(batch_size);
        let mut td_targets = Array1::zeros(batch_size);

        for (i, t) in batch.iter().enumerate() {
            states.row_mut(i).assign(&t.state);
            actions.push(t.action);

            // Compute TD target using target network
            let next_q = self.target_network.forward(&t.next_state);
            let max_next_q = next_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            td_targets[i] = if t.done {
                t.reward
            } else {
                t.reward + self.config.gamma * max_next_q
            };
        }

        // Compute loss before update for logging
        let (total_loss, _, _) = self.compute_cql_loss(batch);

        // Update Q-network
        self.q_network.update_from_loss_components(
            &states,
            &actions,
            &td_targets,
            self.config.alpha,
            self.config.learning_rate,
            self.config.num_ood_samples,
        );

        // Soft-update target network
        let tau = self.config.tau;
        let online = self.q_network.clone();
        self.target_network.soft_update_from(&online, tau);

        self.training_losses.push(total_loss);
        total_loss
    }

    /// Run offline training for a given number of epochs over the replay buffer.
    pub fn train_offline(
        &mut self,
        buffer: &ReplayBuffer,
        epochs: usize,
        batch_size: usize,
    ) -> Vec<f64> {
        let mut epoch_losses = Vec::with_capacity(epochs);
        let steps_per_epoch = (buffer.len() / batch_size).max(1);

        for _epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            for _step in 0..steps_per_epoch {
                let batch = buffer.sample(batch_size);
                let loss = self.train_step(&batch);
                epoch_loss += loss;
            }
            epoch_loss /= steps_per_epoch as f64;
            epoch_losses.push(epoch_loss);
        }
        epoch_losses
    }
}

// ---------------------------------------------------------------------------
// Standard Q-Learning Agent (for comparison)
// ---------------------------------------------------------------------------

/// Standard Q-learning agent without conservative regularization.
pub struct StandardQLAgent {
    pub q_network: QNetwork,
    pub target_network: QNetwork,
    pub gamma: f64,
    pub learning_rate: f64,
    pub tau: f64,
    pub epsilon: f64,
    pub training_losses: Vec<f64>,
}

impl StandardQLAgent {
    pub fn new(state_dim: usize, hidden1: usize, hidden2: usize) -> Self {
        let q_network = QNetwork::new(state_dim, hidden1, hidden2);
        let target_network = q_network.clone();
        Self {
            q_network,
            target_network,
            gamma: 0.99,
            learning_rate: 1e-3,
            tau: 0.005,
            epsilon: 0.1,
            training_losses: Vec::new(),
        }
    }

    pub fn select_action(&self, state: &Array1<f64>, explore: bool) -> usize {
        let mut rng = rand::thread_rng();
        if explore && rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..TradingAction::NUM_ACTIONS)
        } else {
            let q_values = self.q_network.forward(state);
            q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        }
    }

    pub fn train_step(&mut self, batch: &[Transition]) -> f64 {
        let batch_size = batch.len();
        let state_dim = self.q_network.input_size();

        let mut states = Array2::zeros((batch_size, state_dim));
        let mut actions = Vec::with_capacity(batch_size);
        let mut td_targets = Array1::zeros(batch_size);

        for (i, t) in batch.iter().enumerate() {
            states.row_mut(i).assign(&t.state);
            actions.push(t.action);

            let next_q = self.target_network.forward(&t.next_state);
            let max_next_q = next_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            td_targets[i] = if t.done {
                t.reward
            } else {
                t.reward + self.gamma * max_next_q
            };
        }

        // Compute TD loss
        let mut td_loss = 0.0;
        for (i, t) in batch.iter().enumerate() {
            let q_values = self.q_network.forward(&t.state);
            td_loss += (q_values[t.action] - td_targets[i]).powi(2);
        }
        td_loss /= batch_size as f64;

        // Update with alpha=0 (no CQL penalty)
        self.q_network.update_from_loss_components(
            &states,
            &actions,
            &td_targets,
            0.0, // no CQL regularization
            self.learning_rate,
            0,
        );

        let tau = self.tau;
        let online = self.q_network.clone();
        self.target_network.soft_update_from(&online, tau);

        self.training_losses.push(td_loss);
        td_loss
    }

    pub fn train_offline(
        &mut self,
        buffer: &ReplayBuffer,
        epochs: usize,
        batch_size: usize,
    ) -> Vec<f64> {
        let mut epoch_losses = Vec::with_capacity(epochs);
        let steps_per_epoch = (buffer.len() / batch_size).max(1);

        for _epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            for _step in 0..steps_per_epoch {
                let batch = buffer.sample(batch_size);
                let loss = self.train_step(&batch);
                epoch_loss += loss;
            }
            epoch_loss /= steps_per_epoch as f64;
            epoch_losses.push(epoch_loss);
        }
        epoch_losses
    }
}

// ---------------------------------------------------------------------------
// Bybit API client
// ---------------------------------------------------------------------------

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

/// A single OHLCV candle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Client for fetching historical data from Bybit.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch historical kline data from Bybit.
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

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", resp.ret_msg);
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
// Feature engineering
// ---------------------------------------------------------------------------

/// Extract trading features from a sequence of candles.
/// Returns (features_per_candle, state_dim).
pub fn extract_features(candles: &[Candle]) -> Vec<Array1<f64>> {
    if candles.is_empty() {
        return vec![];
    }

    let max_volume = candles
        .iter()
        .map(|c| c.volume)
        .fold(0.0f64, f64::max)
        .max(1.0);

    candles
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let price_return = if c.open > 0.0 {
                (c.close - c.open) / c.open
            } else {
                0.0
            };
            let hl_range = if c.open > 0.0 {
                (c.high - c.low) / c.open
            } else {
                0.0
            };
            let volume_ratio = c.volume / max_volume;
            let body_ratio = if (c.high - c.low).abs() > 1e-10 {
                (c.close - c.open).abs() / (c.high - c.low)
            } else {
                0.0
            };

            // Simple trend: compare to 5-period SMA
            let lookback = 5.min(i + 1);
            let sma: f64 = candles[i + 1 - lookback..=i]
                .iter()
                .map(|c2| c2.close)
                .sum::<f64>()
                / lookback as f64;
            let trend = if sma > 0.0 {
                (c.close - sma) / sma
            } else {
                0.0
            };

            // Position and P&L are added during dataset construction
            // Here we return raw market features (5 dims)
            // + 2 placeholder dims for position and pnl = 7 total
            Array1::from_vec(vec![
                price_return,
                hl_range,
                volume_ratio,
                body_ratio,
                trend,
                0.0, // position placeholder
                0.0, // pnl placeholder
            ])
        })
        .collect()
}

/// Build an offline dataset from candles using a simple momentum behavior policy.
pub fn build_offline_dataset(candles: &[Candle], transaction_cost: f64) -> ReplayBuffer {
    let features = extract_features(candles);
    let mut buffer = ReplayBuffer::new(features.len());

    if features.len() < 2 {
        return buffer;
    }

    let mut position = 0.0f64;
    let mut cumulative_pnl = 0.0f64;

    for i in 0..features.len() - 1 {
        let mut state = features[i].clone();
        state[5] = position;
        state[6] = cumulative_pnl;

        // Behavior policy: simple momentum
        let price_return = state[0];
        let action = if price_return > 0.02 {
            TradingAction::StrongBuy as usize
        } else if price_return > 0.005 {
            TradingAction::Buy as usize
        } else if price_return < -0.02 {
            TradingAction::StrongSell as usize
        } else if price_return < -0.005 {
            TradingAction::Sell as usize
        } else {
            TradingAction::Hold as usize
        };

        let action_obj = TradingAction::from_index(action);
        let new_position = (position + action_obj.position_delta()).clamp(-1.0, 1.0);

        // Reward: P&L from position - transaction costs
        let next_return = if candles[i].close > 0.0 {
            (candles[i + 1].close - candles[i].close) / candles[i].close
        } else {
            0.0
        };
        let pnl = position * next_return;
        let tc = transaction_cost * (new_position - position).abs();
        let reward = pnl - tc;

        cumulative_pnl += reward;

        let mut next_state = features[i + 1].clone();
        next_state[5] = new_position;
        next_state[6] = cumulative_pnl;

        let done = i == features.len() - 2;

        buffer.push(Transition {
            state,
            action,
            reward,
            next_state,
            done,
        });

        position = new_position;
    }

    buffer
}

/// Evaluate a policy on the offline dataset, returning total reward.
pub fn evaluate_policy<F>(buffer: &ReplayBuffer, select_action: F) -> f64
where
    F: Fn(&Array1<f64>) -> usize,
{
    let transitions = buffer.all_transitions();
    let mut total_reward = 0.0;

    for t in &transitions {
        let _action_idx = select_action(&t.state);
        // Use the actual reward as a proxy for evaluation
        // In a full implementation, we would re-simulate with the new actions
        total_reward += t.reward;
    }

    total_reward
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_trading_action_from_index() {
        assert_eq!(TradingAction::from_index(0), TradingAction::StrongSell);
        assert_eq!(TradingAction::from_index(1), TradingAction::Sell);
        assert_eq!(TradingAction::from_index(2), TradingAction::Hold);
        assert_eq!(TradingAction::from_index(3), TradingAction::Buy);
        assert_eq!(TradingAction::from_index(4), TradingAction::StrongBuy);
        assert_eq!(TradingAction::from_index(99), TradingAction::Hold); // out of range
    }

    #[test]
    fn test_q_network_forward() {
        let net = QNetwork::new(7, 32, 16);
        let state = Array1::from_vec(vec![0.01, 0.02, 0.5, 0.3, 0.1, 0.0, 0.0]);
        let q_values = net.forward(&state);
        assert_eq!(q_values.len(), TradingAction::NUM_ACTIONS);
        // Q-values should be finite
        for q in q_values.iter() {
            assert!(q.is_finite(), "Q-value should be finite");
        }
    }

    #[test]
    fn test_q_network_batch_forward() {
        let net = QNetwork::new(7, 32, 16);
        let states = Array2::from_shape_fn((5, 7), |(i, j)| (i * 7 + j) as f64 * 0.01);
        let q_batch = net.forward_batch(&states);
        assert_eq!(q_batch.nrows(), 5);
        assert_eq!(q_batch.ncols(), TradingAction::NUM_ACTIONS);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        assert!(buffer.is_empty());

        for i in 0..10 {
            buffer.push(Transition {
                state: Array1::zeros(7),
                action: i % 5,
                reward: 0.01,
                next_state: Array1::zeros(7),
                done: false,
            });
        }
        assert_eq!(buffer.len(), 10);

        let sample = buffer.sample(3);
        assert_eq!(sample.len(), 3);
    }

    #[test]
    fn test_replay_buffer_capacity() {
        let mut buffer = ReplayBuffer::new(5);
        for i in 0..10 {
            buffer.push(Transition {
                state: Array1::from_vec(vec![i as f64; 7]),
                action: 0,
                reward: i as f64,
                next_state: Array1::zeros(7),
                done: false,
            });
        }
        // Should only keep the last 5
        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn test_cql_agent_creation() {
        let config = CQLConfig::default();
        let agent = CQLAgent::new(config);
        assert_eq!(agent.q_network.num_parameters(), agent.target_network.num_parameters());
        assert!(agent.training_losses.is_empty());
    }

    #[test]
    fn test_cql_agent_select_action() {
        let config = CQLConfig {
            epsilon: 0.0, // always greedy
            ..CQLConfig::default()
        };
        let agent = CQLAgent::new(config);
        let state = Array1::zeros(7);
        let action = agent.select_action(&state, false);
        assert!(action < TradingAction::NUM_ACTIONS);
    }

    #[test]
    fn test_cql_loss_computation() {
        let config = CQLConfig::default();
        let agent = CQLAgent::new(config);

        let batch: Vec<Transition> = (0..5)
            .map(|i| Transition {
                state: Array1::from_vec(vec![0.01 * i as f64; 7]),
                action: i % 5,
                reward: 0.01,
                next_state: Array1::from_vec(vec![0.01 * (i + 1) as f64; 7]),
                done: false,
            })
            .collect();

        let (total, td, cql) = agent.compute_cql_loss(&batch);
        assert!(total.is_finite());
        assert!(td.is_finite());
        assert!(cql.is_finite());
        // CQL penalty should be >= 0 (logsumexp >= any individual Q-value)
        assert!(cql >= -1e-10, "CQL penalty should be non-negative, got {}", cql);
    }

    #[test]
    fn test_cql_training_step() {
        let config = CQLConfig::default();
        let mut agent = CQLAgent::new(config);

        let batch: Vec<Transition> = (0..8)
            .map(|i| Transition {
                state: Array1::from_vec(vec![0.01 * i as f64; 7]),
                action: i % 5,
                reward: 0.01,
                next_state: Array1::from_vec(vec![0.01 * (i + 1) as f64; 7]),
                done: i == 7,
            })
            .collect();

        let loss = agent.train_step(&batch);
        assert!(loss.is_finite());
        assert_eq!(agent.training_losses.len(), 1);
    }

    #[test]
    fn test_feature_extraction() {
        let candles = vec![
            Candle { timestamp: 1, open: 100.0, high: 105.0, low: 98.0, close: 103.0, volume: 1000.0 },
            Candle { timestamp: 2, open: 103.0, high: 107.0, low: 101.0, close: 106.0, volume: 1200.0 },
            Candle { timestamp: 3, open: 106.0, high: 108.0, low: 104.0, close: 105.0, volume: 800.0 },
        ];
        let features = extract_features(&candles);
        assert_eq!(features.len(), 3);
        assert_eq!(features[0].len(), 7);
        // Price return for first candle: (103 - 100) / 100 = 0.03
        assert!((features[0][0] - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_build_offline_dataset() {
        let candles: Vec<Candle> = (0..20)
            .map(|i| Candle {
                timestamp: i as u64,
                open: 100.0 + i as f64,
                high: 102.0 + i as f64,
                low: 99.0 + i as f64,
                close: 101.0 + i as f64,
                volume: 1000.0 + 10.0 * i as f64,
            })
            .collect();

        let buffer = build_offline_dataset(&candles, 0.001);
        assert_eq!(buffer.len(), 19); // n-1 transitions
    }

    #[test]
    fn test_soft_update() {
        let mut target = QNetwork::new(7, 16, 8);
        let online = QNetwork::new(7, 16, 8);

        let before_w1 = target.w1.clone();
        target.soft_update_from(&online, 0.01);

        // Parameters should have changed
        let changed = target
            .w1
            .iter()
            .zip(before_w1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(changed, "Target network params should change after soft update");
    }

    #[test]
    fn test_offline_training() {
        let candles: Vec<Candle> = (0..50)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.1).sin() * 5.0;
                Candle {
                    timestamp: i as u64,
                    open: base,
                    high: base + 2.0,
                    low: base - 1.0,
                    close: base + 1.0,
                    volume: 1000.0,
                }
            })
            .collect();

        let buffer = build_offline_dataset(&candles, 0.001);
        let config = CQLConfig {
            learning_rate: 1e-4,
            ..CQLConfig::default()
        };
        let mut agent = CQLAgent::new(config);

        let losses = agent.train_offline(&buffer, 3, 16);
        assert_eq!(losses.len(), 3);
        for loss in &losses {
            assert!(loss.is_finite());
        }
    }
}
