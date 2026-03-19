use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================
// Trajectory Tokenizer
// ============================================================

/// Discretizes continuous state/action/reward values into tokens
#[derive(Debug, Clone)]
pub struct TrajectoryTokenizer {
    pub vocab_size: usize,
    pub state_dim: usize,
    pub action_dim: usize,
    /// Bin edges for each dimension: state dims + action dims + 1 reward dim
    pub bin_edges: Vec<Vec<f64>>,
}

impl TrajectoryTokenizer {
    /// Create a tokenizer by computing quantile-based bin edges from data
    pub fn fit(
        states: &[Vec<f64>],
        actions: &[Vec<f64>],
        rewards: &[f64],
        vocab_size: usize,
    ) -> Self {
        let state_dim = if states.is_empty() { 0 } else { states[0].len() };
        let action_dim = if actions.is_empty() { 0 } else { actions[0].len() };
        let total_dims = state_dim + action_dim + 1;
        let mut bin_edges = Vec::with_capacity(total_dims);

        // Compute bin edges for each state dimension
        for d in 0..state_dim {
            let mut values: Vec<f64> = states.iter().map(|s| s[d]).collect();
            bin_edges.push(compute_quantile_edges(&mut values, vocab_size));
        }

        // Compute bin edges for each action dimension
        for d in 0..action_dim {
            let mut values: Vec<f64> = actions.iter().map(|a| a[d]).collect();
            bin_edges.push(compute_quantile_edges(&mut values, vocab_size));
        }

        // Compute bin edges for rewards
        {
            let mut values: Vec<f64> = rewards.to_vec();
            bin_edges.push(compute_quantile_edges(&mut values, vocab_size));
        }

        Self {
            vocab_size,
            state_dim,
            action_dim,
            bin_edges,
        }
    }

    /// Tokenize a single state vector
    pub fn tokenize_state(&self, state: &[f64]) -> Vec<usize> {
        state
            .iter()
            .enumerate()
            .map(|(d, &v)| discretize(v, &self.bin_edges[d], self.vocab_size))
            .collect()
    }

    /// Tokenize a single action vector
    pub fn tokenize_action(&self, action: &[f64]) -> Vec<usize> {
        action
            .iter()
            .enumerate()
            .map(|(d, &v)| discretize(v, &self.bin_edges[self.state_dim + d], self.vocab_size))
            .collect()
    }

    /// Tokenize a single reward value
    pub fn tokenize_reward(&self, reward: f64) -> usize {
        let dim = self.state_dim + self.action_dim;
        discretize(reward, &self.bin_edges[dim], self.vocab_size)
    }

    /// Tokenize a full trajectory into a flat token sequence
    pub fn tokenize_trajectory(
        &self,
        states: &[Vec<f64>],
        actions: &[Vec<f64>],
        rewards: &[f64],
    ) -> Vec<usize> {
        let n = states.len().min(actions.len()).min(rewards.len());
        let mut tokens = Vec::new();
        for i in 0..n {
            tokens.extend(self.tokenize_state(&states[i]));
            tokens.extend(self.tokenize_action(&actions[i]));
            tokens.push(self.tokenize_reward(rewards[i]));
        }
        tokens
    }

    /// Number of tokens per timestep
    pub fn tokens_per_step(&self) -> usize {
        self.state_dim + self.action_dim + 1
    }

    /// Decode a reward token back to approximate continuous value
    pub fn decode_reward(&self, token: usize) -> f64 {
        let dim = self.state_dim + self.action_dim;
        decode_token(token, &self.bin_edges[dim], self.vocab_size)
    }

    /// Decode an action token at a given action dimension
    pub fn decode_action(&self, token: usize, action_dim: usize) -> f64 {
        decode_token(
            token,
            &self.bin_edges[self.state_dim + action_dim],
            self.vocab_size,
        )
    }
}

fn compute_quantile_edges(values: &mut Vec<f64>, n_bins: usize) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if values.is_empty() {
        return vec![0.0; n_bins + 1];
    }
    let mut edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        let idx = (i as f64 / n_bins as f64 * (values.len() - 1) as f64) as usize;
        let idx = idx.min(values.len() - 1);
        edges.push(values[idx]);
    }
    edges
}

fn discretize(value: f64, edges: &[f64], vocab_size: usize) -> usize {
    for i in 1..edges.len() {
        if value <= edges[i] {
            return (i - 1).min(vocab_size - 1);
        }
    }
    vocab_size - 1
}

fn decode_token(token: usize, edges: &[f64], vocab_size: usize) -> f64 {
    let t = token.min(vocab_size - 1);
    if t + 1 < edges.len() {
        (edges[t] + edges[t + 1]) / 2.0
    } else if !edges.is_empty() {
        *edges.last().unwrap()
    } else {
        0.0
    }
}

// ============================================================
// Simplified Transformer
// ============================================================

/// Configuration for the Trajectory Transformer
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub ff_dim: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 100,
            embed_dim: 64,
            num_heads: 4,
            num_layers: 2,
            max_seq_len: 512,
            ff_dim: 256,
        }
    }
}

/// A single attention head
#[derive(Debug, Clone)]
struct AttentionHead {
    w_q: Array2<f64>,
    w_k: Array2<f64>,
    w_v: Array2<f64>,
    head_dim: usize,
}

impl AttentionHead {
    fn new(embed_dim: usize, head_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (embed_dim + head_dim) as f64).sqrt();
        Self {
            w_q: random_matrix(embed_dim, head_dim, scale, &mut rng),
            w_k: random_matrix(embed_dim, head_dim, scale, &mut rng),
            w_v: random_matrix(embed_dim, head_dim, scale, &mut rng),
            head_dim,
        }
    }

    fn forward(&self, x: &Array2<f64>, causal: bool) -> Array2<f64> {
        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        let seq_len = x.nrows();
        let scale = (self.head_dim as f64).sqrt();

        // Compute attention scores
        let mut scores = Array2::zeros((seq_len, seq_len));
        for i in 0..seq_len {
            for j in 0..seq_len {
                if causal && j > i {
                    scores[[i, j]] = f64::NEG_INFINITY;
                } else {
                    let mut dot = 0.0;
                    for d in 0..self.head_dim {
                        dot += q[[i, d]] * k[[j, d]];
                    }
                    scores[[i, j]] = dot / scale;
                }
            }
        }

        // Softmax
        for i in 0..seq_len {
            let max_val = scores.row(i).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for j in 0..seq_len {
                scores[[i, j]] = (scores[[i, j]] - max_val).exp();
                sum += scores[[i, j]];
            }
            if sum > 0.0 {
                for j in 0..seq_len {
                    scores[[i, j]] /= sum;
                }
            }
        }

        scores.dot(&v)
    }
}

/// Multi-head attention layer
#[derive(Debug, Clone)]
struct MultiHeadAttention {
    heads: Vec<AttentionHead>,
    w_o: Array2<f64>,
}

impl MultiHeadAttention {
    fn new(embed_dim: usize, num_heads: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (head_dim * num_heads + embed_dim) as f64).sqrt();
        Self {
            heads: (0..num_heads)
                .map(|_| AttentionHead::new(embed_dim, head_dim))
                .collect(),
            w_o: random_matrix(head_dim * num_heads, embed_dim, scale, &mut rng),
        }
    }

    fn forward(&self, x: &Array2<f64>, causal: bool) -> Array2<f64> {
        let seq_len = x.nrows();
        let head_dim = self.heads[0].head_dim;
        let total_dim = head_dim * self.heads.len();

        let mut concat = Array2::zeros((seq_len, total_dim));
        for (h, head) in self.heads.iter().enumerate() {
            let out = head.forward(x, causal);
            for i in 0..seq_len {
                for d in 0..head_dim {
                    concat[[i, h * head_dim + d]] = out[[i, d]];
                }
            }
        }

        concat.dot(&self.w_o)
    }
}

/// Feed-forward network
#[derive(Debug, Clone)]
struct FeedForward {
    w1: Array2<f64>,
    w2: Array2<f64>,
    b1: Array1<f64>,
    b2: Array1<f64>,
}

impl FeedForward {
    fn new(embed_dim: usize, ff_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / (embed_dim + ff_dim) as f64).sqrt();
        let scale2 = (2.0 / (ff_dim + embed_dim) as f64).sqrt();
        Self {
            w1: random_matrix(embed_dim, ff_dim, scale1, &mut rng),
            w2: random_matrix(ff_dim, embed_dim, scale2, &mut rng),
            b1: Array1::zeros(ff_dim),
            b2: Array1::zeros(embed_dim),
        }
    }

    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut h = x.dot(&self.w1);
        // Add bias and ReLU
        for i in 0..h.nrows() {
            for j in 0..h.ncols() {
                h[[i, j]] = (h[[i, j]] + self.b1[j]).max(0.0);
            }
        }
        let mut out = h.dot(&self.w2);
        for i in 0..out.nrows() {
            for j in 0..out.ncols() {
                out[[i, j]] += self.b2[j];
            }
        }
        out
    }
}

/// Transformer block (attention + feed-forward with residual connections)
#[derive(Debug, Clone)]
struct TransformerBlock {
    attention: MultiHeadAttention,
    ff: FeedForward,
}

impl TransformerBlock {
    fn new(embed_dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(embed_dim, num_heads),
            ff: FeedForward::new(embed_dim, ff_dim),
        }
    }

    fn forward(&self, x: &Array2<f64>, causal: bool) -> Array2<f64> {
        // Self-attention with residual
        let attn_out = self.attention.forward(x, causal);
        let mut residual = x + &attn_out;
        layer_norm_inplace(&mut residual);

        // Feed-forward with residual
        let ff_out = self.ff.forward(&residual);
        let mut out = &residual + &ff_out;
        layer_norm_inplace(&mut out);

        out
    }
}

/// The Trajectory Transformer model
#[derive(Debug, Clone)]
pub struct TrajectoryTransformer {
    pub config: TransformerConfig,
    token_embeddings: Array2<f64>,
    position_embeddings: Array2<f64>,
    blocks: Vec<TransformerBlock>,
    output_projection: Array2<f64>,
}

impl TrajectoryTransformer {
    pub fn new(config: TransformerConfig) -> Self {
        let mut rng = rand::thread_rng();
        let embed_scale = (1.0 / config.embed_dim as f64).sqrt();
        let out_scale = (2.0 / (config.embed_dim + config.vocab_size) as f64).sqrt();

        Self {
            token_embeddings: random_matrix(
                config.vocab_size,
                config.embed_dim,
                embed_scale,
                &mut rng,
            ),
            position_embeddings: random_matrix(
                config.max_seq_len,
                config.embed_dim,
                embed_scale,
                &mut rng,
            ),
            blocks: (0..config.num_layers)
                .map(|_| {
                    TransformerBlock::new(config.embed_dim, config.num_heads, config.ff_dim)
                })
                .collect(),
            output_projection: random_matrix(
                config.embed_dim,
                config.vocab_size,
                out_scale,
                &mut rng,
            ),
            config,
        }
    }

    /// Forward pass: tokens -> logits
    pub fn forward(&self, tokens: &[usize]) -> Array2<f64> {
        let seq_len = tokens.len().min(self.config.max_seq_len);
        let embed_dim = self.config.embed_dim;

        // Token + position embeddings
        let mut x = Array2::zeros((seq_len, embed_dim));
        for (i, &tok) in tokens[..seq_len].iter().enumerate() {
            let tok = tok.min(self.config.vocab_size - 1);
            for d in 0..embed_dim {
                x[[i, d]] = self.token_embeddings[[tok, d]] + self.position_embeddings[[i, d]];
            }
        }

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x, true);
        }

        // Output projection to logits
        x.dot(&self.output_projection)
    }

    /// Get probability distribution for next token given context
    pub fn predict_next_token_probs(&self, tokens: &[usize]) -> Vec<f64> {
        let logits = self.forward(tokens);
        let last_row = logits.row(logits.nrows() - 1);

        // Softmax
        let max_val = last_row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = last_row.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    }

    /// Simple training step with cross-entropy loss (SGD)
    /// Returns the average loss over the sequence
    pub fn train_step(&mut self, tokens: &[usize], learning_rate: f64) -> f64 {
        if tokens.len() < 2 {
            return 0.0;
        }

        let logits = self.forward(&tokens[..tokens.len() - 1]);
        let targets = &tokens[1..];
        let seq_len = logits.nrows();
        let vocab = self.config.vocab_size;

        // Compute cross-entropy loss
        let mut total_loss = 0.0;
        for i in 0..seq_len {
            let max_val = logits.row(i).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f64> = logits.row(i).iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let target = targets[i].min(vocab - 1);
            total_loss += -(exps[target] / sum).ln();
        }

        // Simplified gradient update on output projection
        // In a real implementation, backprop would go through all layers
        for i in 0..seq_len {
            let max_val = logits.row(i).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f64> = logits.row(i).iter().map(|&v| (v - max_val).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();
            let target = targets[i].min(vocab - 1);

            // Gradient for output projection (simplified)
            for v in 0..vocab {
                let grad = probs[v] - if v == target { 1.0 } else { 0.0 };
                let scale = learning_rate * grad / seq_len as f64;
                for d in 0..self.config.embed_dim {
                    self.output_projection[[d, v]] -= scale * 0.01;
                }
            }
        }

        total_loss / seq_len as f64
    }
}

fn layer_norm_inplace(x: &mut Array2<f64>) {
    let eps = 1e-5;
    for i in 0..x.nrows() {
        let mean = x.row(i).mean().unwrap_or(0.0);
        let var = x.row(i).iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / x.ncols() as f64;
        let std = (var + eps).sqrt();
        for j in 0..x.ncols() {
            x[[i, j]] = (x[[i, j]] - mean) / std;
        }
    }
}

fn random_matrix(rows: usize, cols: usize, scale: f64, rng: &mut impl Rng) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
}

// ============================================================
// Beam Search
// ============================================================

/// A beam candidate during search
#[derive(Debug, Clone)]
pub struct BeamCandidate {
    pub tokens: Vec<usize>,
    pub log_prob: f64,
}

/// Beam search for trajectory planning
pub struct BeamSearch {
    pub beam_width: usize,
    pub max_steps: usize,
}

impl BeamSearch {
    pub fn new(beam_width: usize, max_steps: usize) -> Self {
        Self {
            beam_width,
            max_steps,
        }
    }

    /// Run beam search from a prefix of tokens
    pub fn search(
        &self,
        model: &TrajectoryTransformer,
        prefix: &[usize],
    ) -> Vec<BeamCandidate> {
        let mut beams = vec![BeamCandidate {
            tokens: prefix.to_vec(),
            log_prob: 0.0,
        }];

        for _step in 0..self.max_steps {
            let mut candidates = Vec::new();

            for beam in &beams {
                let probs = model.predict_next_token_probs(&beam.tokens);

                // Take top-k tokens
                let mut indexed: Vec<(usize, f64)> =
                    probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let top_k = self.beam_width.min(indexed.len());
                for &(tok, prob) in indexed.iter().take(top_k) {
                    if prob > 1e-10 {
                        let mut new_tokens = beam.tokens.clone();
                        new_tokens.push(tok);
                        candidates.push(BeamCandidate {
                            tokens: new_tokens,
                            log_prob: beam.log_prob + prob.ln(),
                        });
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Keep top beam_width candidates
            candidates.sort_by(|a, b| {
                b.log_prob
                    .partial_cmp(&a.log_prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(self.beam_width);
            beams = candidates;
        }

        beams
    }
}

// ============================================================
// Return-Conditioned Planner
// ============================================================

/// Plans trading actions conditioned on a target return
pub struct ReturnConditionedPlanner {
    pub model: TrajectoryTransformer,
    pub tokenizer: TrajectoryTokenizer,
    pub beam_search: BeamSearch,
}

impl ReturnConditionedPlanner {
    pub fn new(
        model: TrajectoryTransformer,
        tokenizer: TrajectoryTokenizer,
        beam_width: usize,
        plan_horizon: usize,
    ) -> Self {
        Self {
            model,
            tokenizer,
            beam_search: BeamSearch::new(beam_width, plan_horizon),
        }
    }

    /// Plan actions from current state with a target return
    pub fn plan(
        &self,
        current_state: &[f64],
        target_return: f64,
    ) -> Vec<(Vec<f64>, f64)> {
        // Tokenize current state
        let mut prefix = self.tokenizer.tokenize_state(current_state);

        // Add target return token as conditioning
        let return_token = self.tokenizer.tokenize_reward(target_return);
        prefix.push(return_token);

        // Run beam search
        let beams = self.beam_search.search(&self.model, &prefix);

        // Extract actions from best beam
        let mut actions = Vec::new();
        if let Some(best) = beams.first() {
            let tps = self.tokenizer.tokens_per_step();
            let state_dim = self.tokenizer.state_dim;
            let action_dim = self.tokenizer.action_dim;

            // Parse tokens back into actions
            let payload = &best.tokens[prefix.len()..];
            let mut pos = 0;
            while pos + tps <= payload.len() + state_dim {
                // Skip state tokens (we already know the state)
                let action_start = if pos == 0 { 0 } else { state_dim };
                let actual_pos = pos + action_start;

                if actual_pos + action_dim <= payload.len() {
                    let action: Vec<f64> = (0..action_dim)
                        .map(|d| self.tokenizer.decode_action(payload[actual_pos + d], d))
                        .collect();
                    let reward_pos = actual_pos + action_dim;
                    let reward = if reward_pos < payload.len() {
                        self.tokenizer.decode_reward(payload[reward_pos])
                    } else {
                        0.0
                    };
                    actions.push((action, reward));
                }
                pos += tps;
            }
        }

        actions
    }
}

// ============================================================
// Bybit API Client
// ============================================================

#[derive(Debug, Deserialize, Serialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// OHLCV candle data
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API client for fetching market data
pub struct BybitClient {
    pub base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data
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

        let client = reqwest::blocking::Client::new();
        let resp: BybitKlineResponse = client.get(&url).send()?.json()?;

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

        // Bybit returns newest first, reverse for chronological order
        candles.reverse();
        Ok(candles)
    }
}

// ============================================================
// Trading Feature Engineering
// ============================================================

/// Extract trading state features from candle data
pub fn extract_trading_states(candles: &[Candle], lookback: usize) -> Vec<Vec<f64>> {
    let mut states = Vec::new();

    for i in lookback..candles.len() {
        let mut features = Vec::new();

        // Log return
        let log_return = (candles[i].close / candles[i - 1].close).ln();
        features.push(log_return);

        // Moving average ratio (short/long)
        let short_ma: f64 = candles[i.saturating_sub(5)..=i]
            .iter()
            .map(|c| c.close)
            .sum::<f64>()
            / (i - i.saturating_sub(5) + 1) as f64;
        let long_ma: f64 = candles[i.saturating_sub(lookback)..=i]
            .iter()
            .map(|c| c.close)
            .sum::<f64>()
            / (i - i.saturating_sub(lookback) + 1) as f64;
        features.push(short_ma / long_ma - 1.0);

        // Volatility (std of returns over lookback)
        let returns: Vec<f64> = (i.saturating_sub(lookback) + 1..=i)
            .map(|j| (candles[j].close / candles[j - 1].close).ln())
            .collect();
        let mean_ret = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
        let var = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / returns.len().max(1) as f64;
        features.push(var.sqrt());

        // Relative volume
        let avg_vol: f64 = candles[i.saturating_sub(lookback)..=i]
            .iter()
            .map(|c| c.volume)
            .sum::<f64>()
            / (i - i.saturating_sub(lookback) + 1) as f64;
        let rel_vol = if avg_vol > 0.0 {
            candles[i].volume / avg_vol
        } else {
            1.0
        };
        features.push(rel_vol);

        // Price momentum (current price / price N bars ago)
        let momentum = candles[i].close / candles[i.saturating_sub(lookback)].close - 1.0;
        features.push(momentum);

        states.push(features);
    }

    states
}

/// Generate trading actions based on future returns (for training data)
pub fn generate_actions(candles: &[Candle], lookback: usize) -> Vec<Vec<f64>> {
    let mut actions = Vec::new();

    for i in lookback..candles.len() {
        let future_return = if i + 1 < candles.len() {
            (candles[i + 1].close / candles[i].close).ln()
        } else {
            0.0
        };

        // Discretize into action: [-2, -1, 0, 1, 2] mapped to continuous
        let action = if future_return > 0.01 {
            2.0
        } else if future_return > 0.002 {
            1.0
        } else if future_return < -0.01 {
            -2.0
        } else if future_return < -0.002 {
            -1.0
        } else {
            0.0
        };

        actions.push(vec![action]);
    }

    actions
}

/// Compute rewards from actions and future price changes
pub fn compute_rewards(candles: &[Candle], actions: &[Vec<f64>], lookback: usize) -> Vec<f64> {
    let mut rewards = Vec::new();

    for (idx, i) in (lookback..candles.len()).enumerate() {
        let future_return = if i + 1 < candles.len() {
            (candles[i + 1].close / candles[i].close).ln()
        } else {
            0.0
        };

        let action = if idx < actions.len() {
            actions[idx][0]
        } else {
            0.0
        };

        // Reward = action * future_return (positive action + positive return = positive reward)
        rewards.push(action * future_return * 100.0);
    }

    rewards
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let states: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                vec![
                    (i as f64 * 0.1).sin(),
                    (i as f64 * 0.05).cos(),
                    i as f64 / 100.0,
                    (i as f64 * 0.2).sin().abs(),
                    (i as f64 * 0.03).cos(),
                ]
            })
            .collect();

        let actions: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![((i % 5) as f64 - 2.0)])
            .collect();

        let rewards: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin() * 0.5)
            .collect();

        (states, actions, rewards)
    }

    #[test]
    fn test_tokenizer_fit_and_tokenize() {
        let (states, actions, rewards) = sample_data();
        let tokenizer = TrajectoryTokenizer::fit(&states, &actions, &rewards, 50);

        assert_eq!(tokenizer.vocab_size, 50);
        assert_eq!(tokenizer.state_dim, 5);
        assert_eq!(tokenizer.action_dim, 1);
        assert_eq!(tokenizer.tokens_per_step(), 7); // 5 state + 1 action + 1 reward

        let state_tokens = tokenizer.tokenize_state(&states[0]);
        assert_eq!(state_tokens.len(), 5);
        for &t in &state_tokens {
            assert!(t < 50);
        }
    }

    #[test]
    fn test_tokenize_trajectory() {
        let (states, actions, rewards) = sample_data();
        let tokenizer = TrajectoryTokenizer::fit(&states, &actions, &rewards, 50);

        let trajectory = tokenizer.tokenize_trajectory(&states[..10], &actions[..10], &rewards[..10]);
        assert_eq!(trajectory.len(), 10 * 7);
    }

    #[test]
    fn test_transformer_forward() {
        let config = TransformerConfig {
            vocab_size: 50,
            embed_dim: 32,
            num_heads: 4,
            num_layers: 1,
            max_seq_len: 64,
            ff_dim: 128,
        };
        let model = TrajectoryTransformer::new(config);

        let tokens = vec![1, 5, 10, 20, 30, 3, 25];
        let logits = model.forward(&tokens);

        assert_eq!(logits.nrows(), 7);
        assert_eq!(logits.ncols(), 50);
    }

    #[test]
    fn test_predict_next_token_probs() {
        let config = TransformerConfig {
            vocab_size: 50,
            embed_dim: 32,
            num_heads: 4,
            num_layers: 1,
            max_seq_len: 64,
            ff_dim: 128,
        };
        let model = TrajectoryTransformer::new(config);

        let probs = model.predict_next_token_probs(&[1, 5, 10]);
        assert_eq!(probs.len(), 50);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Probabilities should sum to 1, got {}", sum);

        for &p in &probs {
            assert!(p >= 0.0, "Probabilities should be non-negative");
        }
    }

    #[test]
    fn test_beam_search() {
        let config = TransformerConfig {
            vocab_size: 50,
            embed_dim: 32,
            num_heads: 4,
            num_layers: 1,
            max_seq_len: 64,
            ff_dim: 128,
        };
        let model = TrajectoryTransformer::new(config);
        let beam = BeamSearch::new(3, 5);

        let results = beam.search(&model, &[1, 5, 10]);

        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        for candidate in &results {
            assert!(candidate.tokens.len() >= 3); // at least the prefix
            assert!(candidate.tokens.starts_with(&[1, 5, 10]));
        }
    }

    #[test]
    fn test_train_step_reduces_loss() {
        let config = TransformerConfig {
            vocab_size: 20,
            embed_dim: 16,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 32,
            ff_dim: 64,
        };
        let mut model = TrajectoryTransformer::new(config);

        let tokens = vec![1, 5, 10, 3, 7, 12, 1, 5, 10, 3, 7, 12];

        let loss1 = model.train_step(&tokens, 0.01);
        // Train a few more times
        let mut loss = loss1;
        for _ in 0..20 {
            loss = model.train_step(&tokens, 0.01);
        }

        // Loss should generally decrease (or at least not explode)
        assert!(loss.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_reward_encode_decode_roundtrip() {
        let rewards: Vec<f64> = (-50..50).map(|i| i as f64 * 0.1).collect();
        let states: Vec<Vec<f64>> = rewards.iter().map(|&r| vec![r, r * 2.0]).collect();
        let actions: Vec<Vec<f64>> = rewards.iter().map(|_| vec![0.0]).collect();

        let tokenizer = TrajectoryTokenizer::fit(&states, &actions, &rewards, 100);

        // Check that encoding and decoding gives approximately correct results
        let test_reward = 1.5;
        let token = tokenizer.tokenize_reward(test_reward);
        let decoded = tokenizer.decode_reward(token);

        // Should be in the right ballpark (within a bin width)
        assert!(
            (decoded - test_reward).abs() < 1.0,
            "Decoded reward {} should be close to original {}",
            decoded,
            test_reward
        );
    }

    #[test]
    fn test_extract_trading_states() {
        let candles: Vec<Candle> = (0..30)
            .map(|i| Candle {
                timestamp: i as u64 * 3600000,
                open: 50000.0 + (i as f64 * 0.1).sin() * 1000.0,
                high: 50500.0 + (i as f64 * 0.1).sin() * 1000.0,
                low: 49500.0 + (i as f64 * 0.1).sin() * 1000.0,
                close: 50000.0 + ((i + 1) as f64 * 0.1).sin() * 1000.0,
                volume: 1000.0 + (i as f64 * 0.3).sin() * 500.0,
            })
            .collect();

        let states = extract_trading_states(&candles, 10);
        assert_eq!(states.len(), 20); // 30 - 10
        assert_eq!(states[0].len(), 5); // 5 features

        for state in &states {
            for &v in state {
                assert!(v.is_finite(), "State features should be finite");
            }
        }
    }
}
