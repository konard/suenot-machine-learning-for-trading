//! Decision Transformer for Trading
//!
//! Implements a Decision Transformer that reframes reinforcement learning as sequence modeling.
//! The model generates trading actions conditioned on desired returns-to-go (RTG),
//! past states, and past actions using a causal transformer architecture.

use anyhow::{Result, Context};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================
// Data Structures
// ============================================================

/// OHLCV candle data from exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// A single transition in a trajectory
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Vec<f64>,
    pub action: i32,    // -1 = sell, 0 = hold, 1 = buy
    pub reward: f64,
    pub rtg: f64,       // return-to-go
    pub timestep: usize,
}

/// A complete trajectory (episode) of trading
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub transitions: Vec<Transition>,
}

/// Offline dataset of trajectories
#[derive(Debug, Clone)]
pub struct TrajectoryDataset {
    pub trajectories: Vec<Trajectory>,
    pub state_dim: usize,
    pub max_episode_len: usize,
}

/// Trading action enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingAction {
    Sell = -1,
    Hold = 0,
    Buy = 1,
}

impl TradingAction {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => TradingAction::Sell,
            1 => TradingAction::Hold,
            2 => TradingAction::Buy,
            _ => TradingAction::Hold,
        }
    }

    pub fn to_index(self) -> usize {
        match self {
            TradingAction::Sell => 0,
            TradingAction::Hold => 1,
            TradingAction::Buy => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TradingAction::Sell => "SELL",
            TradingAction::Hold => "HOLD",
            TradingAction::Buy => "BUY",
        }
    }
}

// ============================================================
// Return-to-Go Calculation
// ============================================================

/// Compute return-to-go from a sequence of rewards.
/// RTG at timestep t = sum of rewards from t to T.
/// Uses reverse cumulative sum for O(n) computation.
pub fn compute_rtg(rewards: &[f64]) -> Vec<f64> {
    let n = rewards.len();
    if n == 0 {
        return vec![];
    }
    let mut rtg = vec![0.0; n];
    rtg[n - 1] = rewards[n - 1];
    for i in (0..n - 1).rev() {
        rtg[i] = rewards[i] + rtg[i + 1];
    }
    rtg
}

/// Compute discounted return-to-go with discount factor gamma.
pub fn compute_discounted_rtg(rewards: &[f64], gamma: f64) -> Vec<f64> {
    let n = rewards.len();
    if n == 0 {
        return vec![];
    }
    let mut rtg = vec![0.0; n];
    rtg[n - 1] = rewards[n - 1];
    for i in (0..n - 1).rev() {
        rtg[i] = rewards[i] + gamma * rtg[i + 1];
    }
    rtg
}

// ============================================================
// Causal Self-Attention
// ============================================================

/// Causal (masked) self-attention layer.
/// Ensures each position can only attend to previous positions.
#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_k: usize,
    // Weight matrices: [d_model x d_model]
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub w_o: Array2<f64>,
}

impl CausalSelfAttention {
    /// Create a new causal self-attention layer with random initialization.
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let d_k = d_model / n_heads;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / d_model as f64).sqrt();

        let mut init_matrix = |rows: usize, cols: usize| -> Array2<f64> {
            Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
        };

        CausalSelfAttention {
            d_model,
            n_heads,
            d_k,
            w_q: init_matrix(d_model, d_model),
            w_k: init_matrix(d_model, d_model),
            w_v: init_matrix(d_model, d_model),
            w_o: init_matrix(d_model, d_model),
        }
    }

    /// Forward pass with causal masking.
    /// Input: [seq_len x d_model], Output: [seq_len x d_model]
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Compute Q, K, V: [seq_len x d_model]
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        // Scaled dot-product attention with causal mask
        // scores: [seq_len x seq_len]
        let scale = (self.d_k as f64).sqrt();
        let mut scores = q.dot(&k.t()) / scale;

        // Apply causal mask: positions can only attend to earlier positions
        apply_causal_mask(&mut scores);

        // Softmax over last axis
        let attn = softmax_rows(&scores);

        // Weighted sum of values
        let context = attn.dot(&v);

        // Output projection
        context.dot(&self.w_o)
    }
}

/// Apply causal mask to attention scores.
/// Sets future positions (j > i) to a large negative value.
pub fn apply_causal_mask(scores: &mut Array2<f64>) {
    let n = scores.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            scores[[i, j]] = f64::NEG_INFINITY;
        }
    }
}

/// Row-wise softmax for attention scores.
fn softmax_rows(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    for mut row in result.rows_mut() {
        let max_val = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max_val).exp());
        let sum: f64 = row.sum();
        if sum > 0.0 {
            row.mapv_inplace(|v| v / sum);
        }
    }
    result
}

// ============================================================
// Transformer Block
// ============================================================

/// A single transformer block: attention + FFN with residual connections.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub attention: CausalSelfAttention,
    pub w_ff1: Array2<f64>,
    pub w_ff2: Array2<f64>,
    pub b_ff1: Array1<f64>,
    pub b_ff2: Array1<f64>,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / d_model as f64).sqrt();
        let scale2 = (2.0 / d_ff as f64).sqrt();

        TransformerBlock {
            attention: CausalSelfAttention::new(d_model, n_heads),
            w_ff1: Array2::from_shape_fn((d_model, d_ff), |_| rng.gen_range(-scale1..scale1)),
            w_ff2: Array2::from_shape_fn((d_ff, d_model), |_| rng.gen_range(-scale2..scale2)),
            b_ff1: Array1::zeros(d_ff),
            b_ff2: Array1::zeros(d_model),
        }
    }

    /// Forward pass: Attention + Residual + FFN + Residual
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Self-attention with residual
        let attn_out = self.attention.forward(x);
        let x_residual = x + &attn_out;
        let x_norm = layer_norm(&x_residual);

        // FFN with residual
        let ff_hidden = x_norm.dot(&self.w_ff1) + &self.b_ff1;
        let ff_relu = ff_hidden.mapv(|v| v.max(0.0)); // ReLU
        let ff_out = ff_relu.dot(&self.w_ff2) + &self.b_ff2;
        let out = &x_norm + &ff_out;

        layer_norm(&out)
    }
}

/// Layer normalization over the last axis.
fn layer_norm(x: &Array2<f64>) -> Array2<f64> {
    let eps = 1e-5;
    let mut result = x.clone();
    for mut row in result.rows_mut() {
        let mean = row.mean().unwrap_or(0.0);
        let var = row.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
        let std = (var + eps).sqrt();
        row.mapv_inplace(|v| (v - mean) / std);
    }
    result
}

// ============================================================
// Decision Transformer
// ============================================================

/// Decision Transformer model for trading.
///
/// Processes sequences of (return-to-go, state, action) triplets
/// and predicts the next action conditioned on desired returns.
#[derive(Debug)]
pub struct DecisionTransformer {
    pub d_model: usize,
    pub state_dim: usize,
    pub n_actions: usize,
    pub max_seq_len: usize,

    // Embedding layers
    pub rtg_embed: Array2<f64>,     // [1 x d_model]
    pub state_embed: Array2<f64>,   // [state_dim x d_model]
    pub action_embed: Array2<f64>,  // [n_actions x d_model]
    pub timestep_embed: Array2<f64>, // [max_seq_len x d_model]

    // Transformer blocks
    pub blocks: Vec<TransformerBlock>,

    // Action prediction head
    pub action_head: Array2<f64>,   // [d_model x n_actions]

    // Learning rate
    pub lr: f64,
}

impl DecisionTransformer {
    /// Create a new Decision Transformer.
    pub fn new(
        state_dim: usize,
        n_actions: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        max_seq_len: usize,
        lr: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / d_model as f64).sqrt();

        let mut init = |rows: usize, cols: usize| -> Array2<f64> {
            Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
        };

        let d_ff = d_model * 4;
        let blocks = (0..n_layers)
            .map(|_| TransformerBlock::new(d_model, n_heads, d_ff))
            .collect();

        DecisionTransformer {
            d_model,
            state_dim,
            n_actions,
            max_seq_len,
            rtg_embed: init(1, d_model),
            state_embed: init(state_dim, d_model),
            action_embed: init(n_actions, d_model),
            timestep_embed: init(max_seq_len, d_model),
            blocks,
            action_head: init(d_model, n_actions),
            lr,
        }
    }

    /// Embed a return-to-go scalar into d_model dimensions.
    fn embed_rtg(&self, rtg: f64) -> Array1<f64> {
        self.rtg_embed.row(0).mapv(|w| w * rtg)
    }

    /// Embed a state vector into d_model dimensions.
    fn embed_state(&self, state: &[f64]) -> Array1<f64> {
        let state_arr = Array1::from_vec(state.to_vec());
        // Pad or truncate state to match state_dim
        let mut padded = Array1::zeros(self.state_dim);
        for i in 0..state.len().min(self.state_dim) {
            padded[i] = state_arr[i];
        }
        padded.dot(&self.state_embed)
    }

    /// Embed an action index into d_model dimensions.
    fn embed_action(&self, action_idx: usize) -> Array1<f64> {
        let idx = action_idx.min(self.n_actions - 1);
        self.action_embed.row(idx).to_owned()
    }

    /// Get timestep embedding.
    fn get_timestep_embed(&self, t: usize) -> Array1<f64> {
        let idx = t.min(self.max_seq_len - 1);
        self.timestep_embed.row(idx).to_owned()
    }

    /// Forward pass: given a sequence of (RTG, state, action) triplets,
    /// predict action logits for each timestep.
    ///
    /// Returns action logits of shape [n_timesteps x n_actions].
    pub fn forward(
        &self,
        rtgs: &[f64],
        states: &[Vec<f64>],
        actions: &[i32],
    ) -> Array2<f64> {
        let n = rtgs.len();
        // Build the interleaved sequence: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        let seq_len = n * 3;
        let mut sequence = Array2::zeros((seq_len, self.d_model));

        for t in 0..n {
            let te = self.get_timestep_embed(t);

            // RTG token
            let rtg_tok = self.embed_rtg(rtgs[t]) + &te;
            sequence.row_mut(t * 3).assign(&rtg_tok);

            // State token
            let state_tok = self.embed_state(&states[t]) + &te;
            sequence.row_mut(t * 3 + 1).assign(&state_tok);

            // Action token
            let action_idx = (actions[t] + 1) as usize; // map -1,0,1 -> 0,1,2
            let action_tok = self.embed_action(action_idx) + &te;
            sequence.row_mut(t * 3 + 2).assign(&action_tok);
        }

        // Pass through transformer blocks
        let mut hidden = sequence;
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }

        // Extract action prediction positions (every 3rd position starting at index 1 = state positions)
        // The action is predicted from the state token position
        let mut action_logits = Array2::zeros((n, self.n_actions));
        for t in 0..n {
            let state_hidden = hidden.row(t * 3 + 1); // state position
            let logits = state_hidden.dot(&self.action_head);
            action_logits.row_mut(t).assign(&logits);
        }

        action_logits
    }

    /// Predict the next action given context.
    /// Returns (action_index, action_probabilities).
    pub fn predict_action(
        &self,
        rtgs: &[f64],
        states: &[Vec<f64>],
        actions: &[i32],
    ) -> (usize, Vec<f64>) {
        let logits = self.forward(rtgs, states, actions);
        let last_logits = logits.row(logits.nrows() - 1);

        // Softmax to get probabilities
        let max_val = last_logits.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f64> = last_logits.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();
        let probs: Vec<f64> = exp_vals.iter().map(|&v| v / sum).collect();

        // Argmax for greedy action selection
        let action_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1);

        (action_idx, probs)
    }

    /// Train on a batch of trajectories for one epoch.
    /// Uses simplified gradient update (finite differences for demonstration).
    /// Returns average loss.
    pub fn train_epoch(&mut self, dataset: &TrajectoryDataset) -> f64 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for trajectory in &dataset.trajectories {
            if trajectory.transitions.len() < 2 {
                continue;
            }

            let rtgs: Vec<f64> = trajectory.transitions.iter().map(|t| t.rtg).collect();
            let states: Vec<Vec<f64>> = trajectory.transitions.iter().map(|t| t.state.clone()).collect();
            let actions: Vec<i32> = trajectory.transitions.iter().map(|t| t.action).collect();

            // Forward pass
            let logits = self.forward(&rtgs, &states, &actions);

            // Compute cross-entropy loss
            let mut epoch_loss = 0.0;
            for t in 0..actions.len() {
                let target_idx = (actions[t] + 1) as usize; // -1,0,1 -> 0,1,2
                let row = logits.row(t);

                // Softmax
                let max_val = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_vals: Vec<f64> = row.iter().map(|&v| (v - max_val).exp()).collect();
                let sum: f64 = exp_vals.iter().sum();
                let log_prob = (exp_vals[target_idx] / sum).max(1e-10).ln();
                epoch_loss -= log_prob;
            }
            epoch_loss /= actions.len() as f64;
            total_loss += epoch_loss;
            count += 1;

            // Simplified gradient update: perturb action head weights
            // This is a demonstration; a production system would use backpropagation
            self.update_action_head(&logits, &actions);
        }

        if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        }
    }

    /// Simplified weight update for the action prediction head.
    fn update_action_head(&mut self, logits: &Array2<f64>, target_actions: &[i32]) {
        let n = target_actions.len().min(logits.nrows());

        for t in 0..n {
            let target_idx = (target_actions[t] + 1) as usize;
            let row = logits.row(t);

            // Softmax
            let max_val = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_vals: Vec<f64> = row.iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f64 = exp_vals.iter().sum();
            let probs: Vec<f64> = exp_vals.iter().map(|&v| v / sum).collect();

            // Gradient of cross-entropy w.r.t. logits: prob - one_hot(target)
            for a in 0..self.n_actions {
                let grad = probs[a] - if a == target_idx { 1.0 } else { 0.0 };
                // Update action head weights
                for d in 0..self.d_model {
                    self.action_head[[d, a]] -= self.lr * grad * 0.01;
                }
            }
        }
    }

    /// Autoregressive inference: generate a sequence of actions
    /// conditioned on a target RTG.
    pub fn generate_actions(
        &self,
        initial_state: &[f64],
        target_rtg: f64,
        states_sequence: &[Vec<f64>],
        n_steps: usize,
    ) -> Vec<(TradingAction, f64)> {
        let mut rtgs = vec![target_rtg];
        let mut states = vec![initial_state.to_vec()];
        let mut actions: Vec<i32> = vec![0]; // start with hold
        let mut results = Vec::new();

        let steps = n_steps.min(states_sequence.len());

        for step in 0..steps {
            let (action_idx, probs) = self.predict_action(&rtgs, &states, &actions);
            let action = TradingAction::from_index(action_idx);
            let confidence = probs[action_idx];

            results.push((action, confidence));

            // Simulate reward (price change based)
            let reward = if step + 1 < states_sequence.len() {
                let current_close = states_sequence[step].get(3).copied().unwrap_or(0.0);
                let next_close = states_sequence[step + 1].get(3).copied().unwrap_or(0.0);
                if current_close > 0.0 {
                    let pct_change = (next_close - current_close) / current_close;
                    match action {
                        TradingAction::Buy => pct_change,
                        TradingAction::Sell => -pct_change,
                        TradingAction::Hold => 0.0,
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Update RTG
            let new_rtg = rtgs.last().unwrap() - reward;
            rtgs.push(new_rtg);

            // Add next state
            if step + 1 < states_sequence.len() {
                states.push(states_sequence[step + 1].clone());
            } else {
                states.push(states_sequence[step].clone());
            }

            // Record action taken
            actions.push(action as i32 - 1); // Map back: Sell=-1, Hold=0, Buy=1
            // Fix: use the correct mapping
            let action_val = match action {
                TradingAction::Sell => -1,
                TradingAction::Hold => 0,
                TradingAction::Buy => 1,
            };
            *actions.last_mut().unwrap() = action_val;
        }

        results
    }
}

// ============================================================
// Offline Dataset Construction
// ============================================================

impl TrajectoryDataset {
    /// Build an offline dataset from OHLCV candles.
    ///
    /// Segments the data into episodes of `episode_len` candles,
    /// assigns actions based on price movement heuristics,
    /// and computes RTG for each transition.
    pub fn from_candles(candles: &[Candle], episode_len: usize) -> Self {
        let state_dim = 5; // OHLCV features
        let mut trajectories = Vec::new();

        if candles.len() < 2 {
            return TrajectoryDataset {
                trajectories,
                state_dim,
                max_episode_len: episode_len,
            };
        }

        // Normalize candle data
        let normalized = normalize_candles(candles);

        // Segment into episodes
        let n_episodes = normalized.len().saturating_sub(1) / episode_len;

        for ep in 0..n_episodes.max(1) {
            let start = ep * episode_len;
            let end = (start + episode_len).min(normalized.len() - 1);
            if start >= end {
                continue;
            }

            // Compute rewards and heuristic actions
            let mut rewards = Vec::new();
            let mut actions = Vec::new();
            let mut states = Vec::new();

            for i in start..end {
                let state = vec![
                    normalized[i].open,
                    normalized[i].high,
                    normalized[i].low,
                    normalized[i].close,
                    normalized[i].volume,
                ];
                states.push(state);

                // Heuristic action based on next candle's price change
                let pct_change = if candles[i].close > 0.0 {
                    (candles[i + 1].close - candles[i].close) / candles[i].close
                } else {
                    0.0
                };

                let action = if pct_change > 0.001 {
                    1 // Buy
                } else if pct_change < -0.001 {
                    -1 // Sell
                } else {
                    0 // Hold
                };
                actions.push(action);

                // Reward = PnL from the heuristic action
                let reward = action as f64 * pct_change;
                rewards.push(reward);
            }

            // Compute RTG
            let rtg_values = compute_rtg(&rewards);

            // Build trajectory
            let transitions: Vec<Transition> = (0..states.len())
                .map(|i| Transition {
                    state: states[i].clone(),
                    action: actions[i],
                    reward: rewards[i],
                    rtg: rtg_values[i],
                    timestep: i,
                })
                .collect();

            if !transitions.is_empty() {
                trajectories.push(Trajectory { transitions });
            }
        }

        TrajectoryDataset {
            trajectories,
            state_dim,
            max_episode_len: episode_len,
        }
    }

    /// Get all states as a flat vector of vectors (for inference).
    pub fn get_all_states(&self) -> Vec<Vec<f64>> {
        self.trajectories
            .iter()
            .flat_map(|t| t.transitions.iter().map(|tr| tr.state.clone()))
            .collect()
    }

    /// Get statistics about the dataset.
    pub fn stats(&self) -> DatasetStats {
        let total_transitions: usize = self.trajectories.iter().map(|t| t.transitions.len()).sum();
        let avg_return: f64 = if !self.trajectories.is_empty() {
            self.trajectories
                .iter()
                .map(|t| t.transitions.first().map(|tr| tr.rtg).unwrap_or(0.0))
                .sum::<f64>()
                / self.trajectories.len() as f64
        } else {
            0.0
        };
        let max_return = self.trajectories
            .iter()
            .map(|t| t.transitions.first().map(|tr| tr.rtg).unwrap_or(0.0))
            .fold(f64::NEG_INFINITY, f64::max);

        DatasetStats {
            n_trajectories: self.trajectories.len(),
            total_transitions,
            avg_return,
            max_return,
        }
    }
}

/// Dataset statistics.
#[derive(Debug)]
pub struct DatasetStats {
    pub n_trajectories: usize,
    pub total_transitions: usize,
    pub avg_return: f64,
    pub max_return: f64,
}

// ============================================================
// Data Normalization
// ============================================================

/// Normalize candle data to [0, 1] range using min-max normalization.
pub fn normalize_candles(candles: &[Candle]) -> Vec<Candle> {
    if candles.is_empty() {
        return vec![];
    }

    let min_price = candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    let max_price = candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
    let min_vol = candles.iter().map(|c| c.volume).fold(f64::INFINITY, f64::min);
    let max_vol = candles.iter().map(|c| c.volume).fold(f64::NEG_INFINITY, f64::max);

    let price_range = (max_price - min_price).max(1e-10);
    let vol_range = (max_vol - min_vol).max(1e-10);

    candles
        .iter()
        .map(|c| Candle {
            timestamp: c.timestamp,
            open: (c.open - min_price) / price_range,
            high: (c.high - min_price) / price_range,
            low: (c.low - min_price) / price_range,
            close: (c.close - min_price) / price_range,
            volume: (c.volume - min_vol) / vol_range,
        })
        .collect()
}

// ============================================================
// Bybit API Integration
// ============================================================

/// Response from Bybit V5 kline API.
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

/// Fetch OHLCV candles from Bybit V5 API.
///
/// # Arguments
/// * `symbol` - Trading pair (e.g., "BTCUSDT")
/// * `interval` - Candle interval (e.g., "60" for 1h)
/// * `limit` - Number of candles to fetch (max 200)
pub fn fetch_bybit_candles(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .context("Failed to build HTTP client")?;

    let response: BybitResponse = client
        .get(&url)
        .send()
        .context("Failed to send request to Bybit")?
        .json()
        .context("Failed to parse Bybit response")?;

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

/// Generate synthetic candle data for testing (when API is unavailable).
pub fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut price = 50000.0;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02);
        let open: f64 = price;
        let close: f64 = price * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: 1700000000 + (i as u64 * 3600),
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rtg() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0];
        let rtg = compute_rtg(&rewards);
        assert_eq!(rtg.len(), 4);
        assert!((rtg[0] - 10.0).abs() < 1e-10); // 1+2+3+4
        assert!((rtg[1] - 9.0).abs() < 1e-10);  // 2+3+4
        assert!((rtg[2] - 7.0).abs() < 1e-10);  // 3+4
        assert!((rtg[3] - 4.0).abs() < 1e-10);  // 4
    }

    #[test]
    fn test_compute_rtg_empty() {
        let rewards: Vec<f64> = vec![];
        let rtg = compute_rtg(&rewards);
        assert!(rtg.is_empty());
    }

    #[test]
    fn test_compute_discounted_rtg() {
        let rewards = vec![1.0, 1.0, 1.0];
        let rtg = compute_discounted_rtg(&rewards, 0.9);
        assert!((rtg[0] - (1.0 + 0.9 + 0.81)).abs() < 1e-10);
        assert!((rtg[1] - (1.0 + 0.9)).abs() < 1e-10);
        assert!((rtg[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_causal_mask() {
        let mut scores = Array2::zeros((4, 4));
        apply_causal_mask(&mut scores);

        // Lower triangle and diagonal should be 0
        for i in 0..4 {
            for j in 0..=i {
                assert!((scores[[i, j]] - 0.0).abs() < 1e-10);
            }
        }
        // Upper triangle should be -inf
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert!(scores[[i, j]].is_infinite() && scores[[i, j]] < 0.0);
            }
        }
    }

    #[test]
    fn test_causal_attention_output_shape() {
        let attn = CausalSelfAttention::new(16, 2);
        let input = Array2::from_shape_fn((5, 16), |_| rand::thread_rng().gen_range(-1.0..1.0));
        let output = attn.forward(&input);
        assert_eq!(output.shape(), &[5, 16]);
    }

    #[test]
    fn test_transformer_block_output_shape() {
        let block = TransformerBlock::new(16, 2, 32);
        let input = Array2::from_shape_fn((6, 16), |_| rand::thread_rng().gen_range(-1.0..1.0));
        let output = block.forward(&input);
        assert_eq!(output.shape(), &[6, 16]);
    }

    #[test]
    fn test_decision_transformer_forward() {
        let dt = DecisionTransformer::new(5, 3, 16, 2, 1, 20, 0.001);

        let rtgs = vec![0.05, 0.03, 0.01];
        let states = vec![
            vec![0.5, 0.6, 0.4, 0.55, 0.3],
            vec![0.52, 0.62, 0.42, 0.57, 0.35],
            vec![0.48, 0.58, 0.38, 0.50, 0.25],
        ];
        let actions = vec![1, 0, -1];

        let logits = dt.forward(&rtgs, &states, &actions);
        assert_eq!(logits.shape(), &[3, 3]); // 3 timesteps, 3 actions
    }

    #[test]
    fn test_decision_transformer_predict_action() {
        let dt = DecisionTransformer::new(5, 3, 16, 2, 1, 20, 0.001);

        let rtgs = vec![0.05];
        let states = vec![vec![0.5, 0.6, 0.4, 0.55, 0.3]];
        let actions = vec![0];

        let (action_idx, probs) = dt.predict_action(&rtgs, &states, &actions);
        assert!(action_idx < 3);
        assert_eq!(probs.len(), 3);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dataset_from_synthetic_candles() {
        let candles = generate_synthetic_candles(50);
        let dataset = TrajectoryDataset::from_candles(&candles, 20);

        assert!(!dataset.trajectories.is_empty());
        assert_eq!(dataset.state_dim, 5);

        let stats = dataset.stats();
        assert!(stats.total_transitions > 0);
    }

    #[test]
    fn test_normalize_candles() {
        let candles = generate_synthetic_candles(10);
        let normalized = normalize_candles(&candles);

        assert_eq!(normalized.len(), 10);
        for c in &normalized {
            assert!(c.open >= 0.0 && c.open <= 1.0);
            assert!(c.close >= 0.0 && c.close <= 1.0);
            assert!(c.volume >= 0.0 && c.volume <= 1.0);
        }
    }

    #[test]
    fn test_trading_action_mapping() {
        assert_eq!(TradingAction::from_index(0), TradingAction::Sell);
        assert_eq!(TradingAction::from_index(1), TradingAction::Hold);
        assert_eq!(TradingAction::from_index(2), TradingAction::Buy);
        assert_eq!(TradingAction::Buy.to_index(), 2);
        assert_eq!(TradingAction::Sell.label(), "SELL");
    }

    #[test]
    fn test_generate_actions() {
        let dt = DecisionTransformer::new(5, 3, 16, 2, 1, 20, 0.001);
        let candles = generate_synthetic_candles(20);
        let normalized = normalize_candles(&candles);
        let states: Vec<Vec<f64>> = normalized
            .iter()
            .map(|c| vec![c.open, c.high, c.low, c.close, c.volume])
            .collect();

        let results = dt.generate_actions(&states[0], 0.05, &states, 5);
        assert_eq!(results.len(), 5);
        for (action, confidence) in &results {
            assert!(confidence >= &0.0 && confidence <= &1.0);
            let _ = action.label(); // Should not panic
        }
    }
}
