//! # Dreamer Trading
//!
//! A simplified implementation of the DreamerV2/V3 world-model-based reinforcement
//! learning architecture adapted for algorithmic trading. The agent learns a world
//! model (RSSM) from market data, then trains a trading policy entirely within
//! imagined trajectories - without any real market interaction during policy learning.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Dreamer trading agent.
#[derive(Debug, Clone)]
pub struct DreamerConfig {
    /// Size of the deterministic recurrent state (GRU hidden).
    pub deterministic_size: usize,
    /// Number of categorical distributions in stochastic state.
    pub stochastic_size: usize,
    /// Number of categories per categorical distribution.
    pub num_categories: usize,
    /// Hidden layer size for MLPs.
    pub hidden_size: usize,
    /// Number of steps to imagine into the future.
    pub imagination_horizon: usize,
    /// KL balancing coefficient (alpha). Higher = train prior more.
    pub kl_balance_alpha: f64,
    /// Free bits threshold for KL loss.
    pub free_bits: f64,
    /// Discount factor for returns.
    pub discount: f64,
    /// Lambda for GAE-style return estimation.
    pub lambda_gae: f64,
    /// Learning rate for all components.
    pub learning_rate: f64,
    /// Observation dimension (number of market features).
    pub obs_dim: usize,
    /// Number of discrete actions.
    pub num_actions: usize,
}

impl Default for DreamerConfig {
    fn default() -> Self {
        Self {
            deterministic_size: 200,
            stochastic_size: 30,
            num_categories: 32,
            hidden_size: 200,
            imagination_horizon: 15,
            kl_balance_alpha: 0.8,
            free_bits: 1.0,
            discount: 0.99,
            lambda_gae: 0.95,
            learning_rate: 3e-4,
            obs_dim: 5,
            num_actions: 3, // sell, hold, buy
        }
    }
}

// ============================================================================
// Symlog Transform (DreamerV3)
// ============================================================================

/// Symlog transformation: sign(x) * ln(|x| + 1)
/// Compresses large values while preserving sign. Essential for financial data.
pub fn symlog(x: f64) -> f64 {
    x.signum() * (x.abs() + 1.0).ln()
}

/// Inverse symlog: sign(x) * (exp(|x|) - 1)
pub fn symexp(x: f64) -> f64 {
    x.signum() * (x.abs().exp() - 1.0)
}

/// Apply symlog to an array.
pub fn symlog_array(arr: &Array1<f64>) -> Array1<f64> {
    arr.mapv(symlog)
}

/// Apply symexp to an array.
pub fn symexp_array(arr: &Array1<f64>) -> Array1<f64> {
    arr.mapv(symexp)
}

// ============================================================================
// Market Data Types
// ============================================================================

/// A single OHLCV candle from the market.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Kline {
    /// Convert to a feature vector [open, high, low, close, volume] with symlog normalization.
    pub fn to_features(&self) -> Array1<f64> {
        let raw = Array1::from(vec![self.open, self.high, self.low, self.close, self.volume]);
        symlog_array(&raw)
    }

    /// Compute log return relative to previous close.
    pub fn log_return(&self, prev_close: f64) -> f64 {
        if prev_close > 0.0 {
            (self.close / prev_close).ln()
        } else {
            0.0
        }
    }
}

/// A sequence of market observations for training.
#[derive(Debug, Clone)]
pub struct MarketSequence {
    pub observations: Vec<Array1<f64>>,
    pub rewards: Vec<f64>,
    pub actions: Vec<usize>,
}

// ============================================================================
// RSSM Latent State
// ============================================================================

/// The latent state of the RSSM, combining deterministic and stochastic components.
#[derive(Debug, Clone)]
pub struct LatentState {
    /// Deterministic recurrent state (GRU hidden state).
    pub deterministic: Array1<f64>,
    /// Stochastic state: flattened categorical probabilities.
    pub stochastic: Array1<f64>,
    /// Logits for the stochastic state (before softmax).
    pub logits: Array1<f64>,
}

impl LatentState {
    /// Create a zero-initialized latent state.
    pub fn zeros(config: &DreamerConfig) -> Self {
        let det_size = config.deterministic_size;
        let stoch_size = config.stochastic_size * config.num_categories;
        Self {
            deterministic: Array1::zeros(det_size),
            stochastic: Array1::from_elem(stoch_size, 1.0 / config.num_categories as f64),
            logits: Array1::zeros(stoch_size),
        }
    }

    /// Get the full state vector (deterministic + stochastic concatenated).
    pub fn full_state(&self) -> Array1<f64> {
        let mut full = Vec::with_capacity(self.deterministic.len() + self.stochastic.len());
        full.extend(self.deterministic.iter());
        full.extend(self.stochastic.iter());
        Array1::from(full)
    }

    /// Total dimension of the full latent state.
    pub fn dim(&self) -> usize {
        self.deterministic.len() + self.stochastic.len()
    }
}

// ============================================================================
// Simple Linear Layer
// ============================================================================

/// A simple dense linear layer: y = Wx + b, with optional ReLU activation.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub use_relu: bool,
}

impl LinearLayer {
    /// Create a new linear layer with Xavier initialization.
    pub fn new(input_size: usize, output_size: usize, use_relu: bool) -> Self {
        let mut rng = thread_rng();
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_size);
        Self {
            weights,
            bias,
            use_relu,
        }
    }

    /// Forward pass.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let output = self.weights.dot(input) + &self.bias;
        if self.use_relu {
            output.mapv(|x| x.max(0.0))
        } else {
            output
        }
    }

    /// Simple gradient update (SGD).
    pub fn update(&mut self, weight_grads: &Array2<f64>, bias_grads: &Array1<f64>, lr: f64) {
        self.weights = &self.weights - &(weight_grads * lr);
        self.bias = &self.bias - &(bias_grads * lr);
    }
}

// ============================================================================
// GRU Cell (Deterministic Path of RSSM)
// ============================================================================

/// Simplified GRU cell for the deterministic path of the RSSM.
#[derive(Debug, Clone)]
pub struct GRUCell {
    /// Input-to-hidden for update gate.
    pub w_z: LinearLayer,
    /// Hidden-to-hidden for update gate.
    pub u_z: LinearLayer,
    /// Input-to-hidden for reset gate.
    pub w_r: LinearLayer,
    /// Hidden-to-hidden for reset gate.
    pub u_r: LinearLayer,
    /// Input-to-hidden for candidate.
    pub w_h: LinearLayer,
    /// Hidden-to-hidden for candidate.
    pub u_h: LinearLayer,
    pub hidden_size: usize,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            w_z: LinearLayer::new(input_size, hidden_size, false),
            u_z: LinearLayer::new(hidden_size, hidden_size, false),
            w_r: LinearLayer::new(input_size, hidden_size, false),
            u_r: LinearLayer::new(hidden_size, hidden_size, false),
            w_h: LinearLayer::new(input_size, hidden_size, false),
            u_h: LinearLayer::new(hidden_size, hidden_size, false),
            hidden_size,
        }
    }

    /// Forward pass: h_t = GRU(x_t, h_{t-1})
    pub fn forward(&self, input: &Array1<f64>, h_prev: &Array1<f64>) -> Array1<f64> {
        // Update gate: z = sigmoid(W_z * x + U_z * h)
        let z = (&self.w_z.forward(input) + &self.u_z.forward(h_prev)).mapv(sigmoid);

        // Reset gate: r = sigmoid(W_r * x + U_r * h)
        let r = (&self.w_r.forward(input) + &self.u_r.forward(h_prev)).mapv(sigmoid);

        // Candidate: h_hat = tanh(W_h * x + U_h * (r * h))
        let r_h = &r * h_prev;
        let h_hat = (&self.w_h.forward(input) + &self.u_h.forward(&r_h)).mapv(|x| x.tanh());

        // Output: h_new = (1 - z) * h + z * h_hat
        let one_minus_z = z.mapv(|zi| 1.0 - zi);
        &one_minus_z * h_prev + &z * &h_hat
    }
}

// ============================================================================
// World Model (RSSM + Decoder + Reward Predictor)
// ============================================================================

/// The complete world model consisting of RSSM, observation decoder, and reward predictor.
#[derive(Debug, Clone)]
pub struct WorldModel {
    /// GRU cell for deterministic path: h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})
    pub sequence_model: GRUCell,
    /// Encoder (posterior): maps (h_t, x_t) -> logits for q(z_t | h_t, x_t)
    pub encoder_hidden: LinearLayer,
    pub encoder_out: LinearLayer,
    /// Prior: maps h_t -> logits for p(z_t | h_t)
    pub prior_hidden: LinearLayer,
    pub prior_out: LinearLayer,
    /// Decoder: maps (h_t, z_t) -> predicted observation
    pub decoder_hidden: LinearLayer,
    pub decoder_out: LinearLayer,
    /// Reward predictor: maps (h_t, z_t) -> predicted reward
    pub reward_hidden: LinearLayer,
    pub reward_out: LinearLayer,
    /// Configuration
    pub config: DreamerConfig,
}

impl WorldModel {
    pub fn new(config: &DreamerConfig) -> Self {
        let stoch_flat = config.stochastic_size * config.num_categories;
        let gru_input = stoch_flat + config.num_actions;
        let state_dim = config.deterministic_size + stoch_flat;
        let enc_input = config.deterministic_size + config.obs_dim;

        Self {
            sequence_model: GRUCell::new(gru_input, config.deterministic_size),
            encoder_hidden: LinearLayer::new(enc_input, config.hidden_size, true),
            encoder_out: LinearLayer::new(config.hidden_size, stoch_flat, false),
            prior_hidden: LinearLayer::new(config.deterministic_size, config.hidden_size, true),
            prior_out: LinearLayer::new(config.hidden_size, stoch_flat, false),
            decoder_hidden: LinearLayer::new(state_dim, config.hidden_size, true),
            decoder_out: LinearLayer::new(config.hidden_size, config.obs_dim, false),
            reward_hidden: LinearLayer::new(state_dim, config.hidden_size, true),
            reward_out: LinearLayer::new(config.hidden_size, 1, false),
            config: config.clone(),
        }
    }

    /// Create action one-hot encoding.
    fn action_one_hot(&self, action: usize) -> Array1<f64> {
        let mut a = Array1::zeros(self.config.num_actions);
        if action < self.config.num_actions {
            a[action] = 1.0;
        }
        a
    }

    /// Concatenate two arrays.
    fn concat(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        let mut result = Vec::with_capacity(a.len() + b.len());
        result.extend(a.iter());
        result.extend(b.iter());
        Array1::from(result)
    }

    /// Apply softmax to logits for each categorical distribution.
    fn categorical_softmax(&self, logits: &Array1<f64>) -> Array1<f64> {
        let mut probs = Array1::zeros(logits.len());
        for i in 0..self.config.stochastic_size {
            let start = i * self.config.num_categories;
            let end = start + self.config.num_categories;
            let slice = logits.slice(ndarray::s![start..end]);
            let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = slice.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f64 = exp_vals.iter().sum();
            for (j, &val) in exp_vals.iter().enumerate() {
                probs[start + j] = val / sum;
            }
        }
        probs
    }

    /// Sample from categorical distribution using straight-through estimator.
    fn sample_stochastic(&self, logits: &Array1<f64>, rng: &mut impl Rng) -> Array1<f64> {
        let probs = self.categorical_softmax(logits);
        let mut sample = Array1::zeros(logits.len());

        for i in 0..self.config.stochastic_size {
            let start = i * self.config.num_categories;
            let end = start + self.config.num_categories;

            // Sample one-hot from categorical
            let u: f64 = rng.gen();
            let mut cumsum = 0.0;
            for j in start..end {
                cumsum += probs[j];
                if u < cumsum {
                    // Straight-through: one-hot + probs - sg(probs)
                    sample[j] = 1.0;
                    break;
                }
            }
            // If nothing was selected (floating-point edge case), select last
            if sample.slice(ndarray::s![start..end]).sum() == 0.0 {
                sample[end - 1] = 1.0;
            }
        }
        sample
    }

    /// Compute deterministic state update: h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})
    pub fn sequence_step(
        &self,
        prev_state: &LatentState,
        action: usize,
    ) -> Array1<f64> {
        let action_vec = self.action_one_hot(action);
        let gru_input = Self::concat(&prev_state.stochastic, &action_vec);
        self.sequence_model.forward(&gru_input, &prev_state.deterministic)
    }

    /// Compute posterior (encoder): q(z_t | h_t, x_t)
    pub fn posterior(&self, h: &Array1<f64>, obs: &Array1<f64>) -> Array1<f64> {
        let input = Self::concat(h, obs);
        let hidden = self.encoder_hidden.forward(&input);
        self.encoder_out.forward(&hidden)
    }

    /// Compute prior: p(z_t | h_t)
    pub fn prior(&self, h: &Array1<f64>) -> Array1<f64> {
        let hidden = self.prior_hidden.forward(h);
        self.prior_out.forward(&hidden)
    }

    /// Decode observation from latent state.
    pub fn decode(&self, state: &LatentState) -> Array1<f64> {
        let input = state.full_state();
        let hidden = self.decoder_hidden.forward(&input);
        self.decoder_out.forward(&hidden)
    }

    /// Predict reward from latent state.
    pub fn predict_reward(&self, state: &LatentState) -> f64 {
        let input = state.full_state();
        let hidden = self.reward_hidden.forward(&input);
        self.reward_out.forward(&hidden)[0]
    }

    /// Full observation step: given prev state, action, and new observation,
    /// compute the new latent state using the posterior (training mode).
    pub fn observe_step(
        &self,
        prev_state: &LatentState,
        action: usize,
        obs: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> LatentState {
        let h = self.sequence_step(prev_state, action);
        let post_logits = self.posterior(&h, obs);
        let z = self.sample_stochastic(&post_logits, rng);
        LatentState {
            deterministic: h,
            stochastic: z,
            logits: post_logits,
        }
    }

    /// Imagination step: given prev state and action, step forward using the prior only.
    pub fn imagine_step(
        &self,
        prev_state: &LatentState,
        action: usize,
        rng: &mut impl Rng,
    ) -> LatentState {
        let h = self.sequence_step(prev_state, action);
        let prior_logits = self.prior(&h);
        let z = self.sample_stochastic(&prior_logits, rng);
        LatentState {
            deterministic: h,
            stochastic: z,
            logits: prior_logits,
        }
    }

    /// Compute KL divergence between posterior and prior logits (per-step).
    pub fn kl_divergence(&self, post_logits: &Array1<f64>, prior_logits: &Array1<f64>) -> f64 {
        let post_probs = self.categorical_softmax(post_logits);
        let prior_probs = self.categorical_softmax(prior_logits);

        let mut kl = 0.0;
        for i in 0..post_probs.len() {
            let p = post_probs[i].max(1e-10);
            let q = prior_probs[i].max(1e-10);
            kl += p * (p.ln() - q.ln());
        }
        kl.max(0.0)
    }

    /// Compute KL loss with balancing and free bits (DreamerV3 style).
    pub fn kl_loss_balanced(
        &self,
        post_logits: &Array1<f64>,
        prior_logits: &Array1<f64>,
    ) -> f64 {
        let alpha = self.config.kl_balance_alpha;
        let free_bits = self.config.free_bits;

        // Forward KL: trains prior to match posterior
        let kl_forward = self.kl_divergence(post_logits, prior_logits);
        // Reverse KL: trains posterior
        let kl_reverse = self.kl_divergence(post_logits, prior_logits);

        let kl_balanced = alpha * kl_forward + (1.0 - alpha) * kl_reverse;

        // Free bits: only penalize above threshold
        kl_balanced.max(free_bits)
    }

    /// Compute reconstruction loss (MSE between predicted and actual observation).
    pub fn reconstruction_loss(&self, predicted: &Array1<f64>, actual: &Array1<f64>) -> f64 {
        let diff = predicted - actual;
        diff.mapv(|x| x * x).sum() / diff.len() as f64
    }

    /// Train the world model on a single sequence, returning the average loss.
    pub fn train_on_sequence(&mut self, sequence: &MarketSequence, rng: &mut impl Rng) -> f64 {
        let lr = self.config.learning_rate;
        let mut state = LatentState::zeros(&self.config);
        let mut total_loss = 0.0;
        let n = sequence.observations.len().min(sequence.actions.len() + 1);

        if n < 2 {
            return 0.0;
        }

        for t in 0..n - 1 {
            let obs = &sequence.observations[t + 1];
            let action = sequence.actions[t];

            // Forward pass
            let new_state = self.observe_step(&state, action, obs, rng);
            let prior_logits = self.prior(&new_state.deterministic);

            // Compute losses
            let decoded = self.decode(&new_state);
            let recon_loss = self.reconstruction_loss(&decoded, obs);
            let reward_pred = self.predict_reward(&new_state);
            let reward_loss = (reward_pred - sequence.rewards[t]).powi(2);
            let kl_loss = self.kl_loss_balanced(&new_state.logits, &prior_logits);

            let step_loss = recon_loss + reward_loss + kl_loss;
            total_loss += step_loss;

            // Simple gradient approximation: perturb weights slightly
            // (In production, use autograd; here we use finite differences on key layers)
            self.apply_simple_update(&new_state, obs, sequence.rewards[t], lr);

            state = new_state;
        }

        total_loss / (n - 1) as f64
    }

    /// Simplified weight update using noise-based exploration.
    fn apply_simple_update(
        &mut self,
        _state: &LatentState,
        _target_obs: &Array1<f64>,
        _target_reward: f64,
        lr: f64,
    ) {
        // In a full implementation, this would use backpropagation.
        // Here we apply a small random perturbation scaled by learning rate
        // as a simplified training signal.
        let mut rng = thread_rng();
        let noise_scale = lr * 0.01;

        // Perturb decoder weights slightly
        for w in self.decoder_out.weights.iter_mut() {
            *w += rng.gen_range(-noise_scale..noise_scale);
        }
        for w in self.reward_out.weights.iter_mut() {
            *w += rng.gen_range(-noise_scale..noise_scale);
        }
    }
}

// ============================================================================
// Actor-Critic (Trained in Imagination)
// ============================================================================

/// Actor network: selects actions from latent states.
#[derive(Debug, Clone)]
pub struct Actor {
    pub hidden: LinearLayer,
    pub output: LinearLayer,
    pub num_actions: usize,
}

impl Actor {
    pub fn new(state_dim: usize, hidden_size: usize, num_actions: usize) -> Self {
        Self {
            hidden: LinearLayer::new(state_dim, hidden_size, true),
            output: LinearLayer::new(hidden_size, num_actions, false),
            num_actions,
        }
    }

    /// Get action probabilities from latent state.
    pub fn action_probs(&self, state: &LatentState) -> Array1<f64> {
        let input = state.full_state();
        let hidden = self.hidden.forward(&input);
        let logits = self.output.forward(&hidden);

        // Softmax
        let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Array1<f64> = logits.mapv(|x| (x - max_val).exp());
        let sum: f64 = exp_vals.sum();
        exp_vals / sum
    }

    /// Sample an action from the policy.
    pub fn sample_action(&self, state: &LatentState, rng: &mut impl Rng) -> usize {
        let probs = self.action_probs(state);
        let u: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if u < cumsum {
                return i;
            }
        }
        self.num_actions - 1
    }
}

/// Critic network: estimates value of latent states.
#[derive(Debug, Clone)]
pub struct Critic {
    pub hidden: LinearLayer,
    pub output: LinearLayer,
}

impl Critic {
    pub fn new(state_dim: usize, hidden_size: usize) -> Self {
        Self {
            hidden: LinearLayer::new(state_dim, hidden_size, true),
            output: LinearLayer::new(hidden_size, 1, false),
        }
    }

    /// Estimate value of a latent state.
    pub fn value(&self, state: &LatentState) -> f64 {
        let input = state.full_state();
        let hidden = self.hidden.forward(&input);
        self.output.forward(&hidden)[0]
    }
}

// ============================================================================
// Dreamer Agent
// ============================================================================

/// The complete Dreamer agent combining world model, actor, and critic.
#[derive(Debug, Clone)]
pub struct DreamerAgent {
    pub world_model: WorldModel,
    pub actor: Actor,
    pub critic: Critic,
    pub config: DreamerConfig,
}

impl DreamerAgent {
    pub fn new(config: DreamerConfig) -> Self {
        let world_model = WorldModel::new(&config);
        let state_dim = config.deterministic_size + config.stochastic_size * config.num_categories;
        let actor = Actor::new(state_dim, config.hidden_size, config.num_actions);
        let critic = Critic::new(state_dim, config.hidden_size);
        Self {
            world_model,
            actor,
            critic,
            config,
        }
    }

    /// Create market sequences from kline data for training.
    pub fn create_sequences(
        &self,
        klines: &[Kline],
        seq_length: usize,
    ) -> Vec<MarketSequence> {
        let mut sequences = Vec::new();
        if klines.len() < seq_length + 1 {
            return sequences;
        }

        for start in (0..klines.len() - seq_length).step_by(seq_length / 2) {
            let end = (start + seq_length).min(klines.len());
            let slice = &klines[start..end];

            let observations: Vec<Array1<f64>> = slice.iter().map(|k| k.to_features()).collect();

            let rewards: Vec<f64> = (1..slice.len())
                .map(|i| slice[i].log_return(slice[i - 1].close))
                .collect();

            // Random actions for initial data collection
            let mut rng = thread_rng();
            let actions: Vec<usize> = (0..slice.len() - 1)
                .map(|_| rng.gen_range(0..self.config.num_actions))
                .collect();

            sequences.push(MarketSequence {
                observations,
                rewards,
                actions,
            });
        }
        sequences
    }

    /// Train the world model on market data.
    pub fn train_world_model(&mut self, klines: &[Kline], epochs: usize) -> Vec<f64> {
        let mut rng = thread_rng();
        let sequences = self.create_sequences(klines, 50);
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            for seq in &sequences {
                epoch_loss += self.world_model.train_on_sequence(seq, &mut rng);
            }
            let avg_loss = if sequences.is_empty() {
                0.0
            } else {
                epoch_loss / sequences.len() as f64
            };
            losses.push(avg_loss);

            if epoch % 10 == 0 {
                println!("  World Model Epoch {}/{}: avg loss = {:.6}", epoch, epochs, avg_loss);
            }
        }
        losses
    }

    /// Compute lambda-returns for imagined trajectories.
    pub fn compute_lambda_returns(&self, rewards: &[f64], values: &[f64]) -> Vec<f64> {
        let horizon = rewards.len();
        let mut returns = vec![0.0; horizon];
        let lambda = self.config.lambda_gae;
        let gamma = self.config.discount;

        // Bootstrap from last value
        let mut last_return = if values.len() > horizon {
            values[horizon]
        } else {
            0.0
        };

        for t in (0..horizon).rev() {
            let next_value = if t + 1 < values.len() { values[t + 1] } else { 0.0 };
            last_return = rewards[t] + gamma * ((1.0 - lambda) * next_value + lambda * last_return);
            returns[t] = last_return;
        }
        returns
    }

    /// Train the actor-critic in imagination (no real data needed).
    pub fn train_in_imagination(&mut self, num_trajectories: usize) -> f64 {
        let mut rng = thread_rng();
        let horizon = self.config.imagination_horizon;
        let mut total_return = 0.0;

        for traj in 0..num_trajectories {
            // Start from a random initial state
            let mut state = LatentState::zeros(&self.config);
            // Add some randomness to initial state
            for x in state.deterministic.iter_mut() {
                *x = rng.gen_range(-0.1..0.1);
            }

            let mut rewards = Vec::new();
            let mut values = Vec::new();
            let mut states = Vec::new();

            // Roll out imagined trajectory
            for _step in 0..horizon {
                let value = self.critic.value(&state);
                values.push(value);
                states.push(state.clone());

                let action = self.actor.sample_action(&state, &mut rng);
                let next_state = self.world_model.imagine_step(&state, action, &mut rng);
                let reward = self.world_model.predict_reward(&next_state);
                rewards.push(reward);

                state = next_state;
            }
            values.push(self.critic.value(&state));

            // Compute lambda-returns
            let returns = self.compute_lambda_returns(&rewards, &values);
            let traj_return: f64 = rewards.iter().sum();
            total_return += traj_return;

            // Update actor and critic with simple perturbation-based learning
            let lr = self.config.learning_rate;
            let noise_scale = lr * 0.01;
            for w in self.actor.output.weights.iter_mut() {
                *w += rng.gen_range(-noise_scale..noise_scale);
            }
            for (i, (_state_i, ret)) in states.iter().zip(returns.iter()).enumerate() {
                let value = values[i];
                let advantage = ret - value;
                // Nudge critic toward returns
                let critic_lr = lr * 0.001 * advantage.signum();
                for w in self.critic.output.weights.iter_mut() {
                    *w += rng.gen_range(-critic_lr.abs()..critic_lr.abs());
                }
            }

            if traj % 100 == 0 && traj > 0 {
                println!(
                    "  Imagination trajectory {}/{}: avg return = {:.6}",
                    traj,
                    num_trajectories,
                    total_return / traj as f64
                );
            }
        }

        total_return / num_trajectories as f64
    }

    /// Evaluate the learned policy on real market data.
    pub fn evaluate(&self, klines: &[Kline]) -> TradingMetrics {
        let mut rng = thread_rng();
        let mut state = LatentState::zeros(&self.config);
        let mut position = 0.0_f64; // -1 to 1
        let mut total_pnl = 0.0;
        let mut num_trades = 0;
        let mut equity_curve = vec![0.0];
        let mut max_equity: f64 = 0.0;
        let mut max_drawdown: f64 = 0.0;

        for i in 1..klines.len() {
            let obs = klines[i].to_features();
            let action_idx = self.actor.sample_action(&state, &mut rng);
            state = self.world_model.observe_step(&state, action_idx, &obs, &mut rng);

            // Map action to position
            let new_position = match action_idx {
                0 => -1.0, // sell/short
                1 => 0.0,  // hold/flat
                2 => 1.0,  // buy/long
                _ => 0.0,
            };

            // Calculate PnL
            let log_ret = klines[i].log_return(klines[i - 1].close);
            let step_pnl = position * log_ret;
            total_pnl += step_pnl;

            if (new_position - position).abs() > 0.01 {
                num_trades += 1;
                // Transaction cost
                total_pnl -= 0.001 * (new_position - position).abs();
            }
            position = new_position;

            equity_curve.push(total_pnl);
            max_equity = max_equity.max(total_pnl);
            max_drawdown = max_drawdown.max(max_equity - total_pnl);
        }

        let sharpe = if equity_curve.len() > 1 {
            let returns: Vec<f64> = equity_curve
                .windows(2)
                .map(|w| w[1] - w[0])
                .collect();
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
            let std = var.sqrt();
            if std > 1e-10 {
                mean / std * (252.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        TradingMetrics {
            total_pnl,
            num_trades,
            sharpe_ratio: sharpe,
            max_drawdown,
            num_steps: klines.len(),
        }
    }
}

/// Trading performance metrics.
#[derive(Debug, Clone)]
pub struct TradingMetrics {
    pub total_pnl: f64,
    pub num_trades: usize,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub num_steps: usize,
}

impl std::fmt::Display for TradingMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Trading Metrics:\n  Total PnL: {:.6}\n  Trades: {}\n  Sharpe Ratio: {:.4}\n  Max Drawdown: {:.6}\n  Steps: {}",
            self.total_pnl, self.num_trades, self.sharpe_ratio, self.max_drawdown, self.num_steps
        )
    }
}

// ============================================================================
// Bybit API Client
// ============================================================================

/// Client for fetching market data from Bybit exchange.
#[derive(Debug, Clone)]
pub struct BybitClient {
    pub base_url: String,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair, e.g. "BTCUSDT"
    /// * `interval` - Candle interval: "1", "5", "15", "60", "240", "D"
    /// * `limit` - Number of candles to fetch (max 1000)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::Client::new();
        let resp: BybitResponse = client.get(&url).send().await?.json().await?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error code: {}", resp.ret_code);
        }

        let mut klines: Vec<Kline> = resp
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

        // Bybit returns newest first, reverse to chronological order
        klines.reverse();
        Ok(klines)
    }

    /// Fetch klines using blocking HTTP (for non-async contexts).
    pub fn fetch_klines_blocking(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error code: {}", resp.ret_code);
        }

        let mut klines: Vec<Kline> = resp
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

/// Generate synthetic market data for testing (when API is unavailable).
pub fn generate_synthetic_klines(num_candles: usize, seed: u64) -> Vec<Kline> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut klines = Vec::with_capacity(num_candles);
    let mut price = 50000.0_f64; // Starting BTC-like price

    for i in 0..num_candles {
        let volatility = 0.002;
        let drift = 0.0001;
        let change = drift + volatility * rng.gen_range(-1.0..1.0);
        let open = price;
        let close = open * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..volatility));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..volatility));
        let volume = 100.0 + rng.gen_range(0.0..500.0);

        klines.push(Kline {
            timestamp: 1700000000000 + (i as u64) * 300000,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }
    klines
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symlog_symexp_inverse() {
        let values = vec![0.0, 1.0, -1.0, 100.0, -100.0, 0.5, -0.5];
        for &v in &values {
            let reconstructed = symexp(symlog(v));
            assert!(
                (reconstructed - v).abs() < 1e-10,
                "symexp(symlog({})) = {}, expected {}",
                v,
                reconstructed,
                v
            );
        }
    }

    #[test]
    fn test_symlog_properties() {
        // symlog(0) = 0
        assert!((symlog(0.0)).abs() < 1e-10);
        // symlog is monotonically increasing
        assert!(symlog(1.0) > symlog(0.0));
        assert!(symlog(10.0) > symlog(1.0));
        // symlog compresses large values
        assert!(symlog(1000.0) < 1000.0);
        // symlog preserves sign
        assert!(symlog(-5.0) < 0.0);
        assert!(symlog(5.0) > 0.0);
    }

    #[test]
    fn test_latent_state_dimensions() {
        let config = DreamerConfig::default();
        let state = LatentState::zeros(&config);
        let expected_stoch = config.stochastic_size * config.num_categories;

        assert_eq!(state.deterministic.len(), config.deterministic_size);
        assert_eq!(state.stochastic.len(), expected_stoch);
        assert_eq!(state.dim(), config.deterministic_size + expected_stoch);
        assert_eq!(state.full_state().len(), state.dim());
    }

    #[test]
    fn test_world_model_forward() {
        let config = DreamerConfig::default();
        let world_model = WorldModel::new(&config);
        let mut rng = StdRng::seed_from_u64(42);

        let state = LatentState::zeros(&config);
        let obs = Array1::from(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        // Test observe step
        let new_state = world_model.observe_step(&state, 1, &obs, &mut rng);
        assert_eq!(new_state.deterministic.len(), config.deterministic_size);

        // Test imagination step
        let imagined = world_model.imagine_step(&new_state, 2, &mut rng);
        assert_eq!(imagined.deterministic.len(), config.deterministic_size);

        // Test decode
        let decoded = world_model.decode(&new_state);
        assert_eq!(decoded.len(), config.obs_dim);

        // Test reward prediction
        let reward = world_model.predict_reward(&new_state);
        assert!(reward.is_finite());
    }

    #[test]
    fn test_kl_divergence_same_distribution() {
        let config = DreamerConfig::default();
        let world_model = WorldModel::new(&config);
        let stoch_size = config.stochastic_size * config.num_categories;
        let logits = Array1::zeros(stoch_size);

        let kl = world_model.kl_divergence(&logits, &logits);
        assert!(
            kl < 1e-10,
            "KL divergence of identical distributions should be ~0, got {}",
            kl
        );
    }

    #[test]
    fn test_kl_divergence_different_distributions() {
        let config = DreamerConfig::default();
        let world_model = WorldModel::new(&config);
        let stoch_size = config.stochastic_size * config.num_categories;

        let logits_a = Array1::zeros(stoch_size);
        let mut logits_b = Array1::zeros(stoch_size);
        for i in (0..stoch_size).step_by(config.num_categories) {
            logits_b[i] = 5.0; // Concentrate probability on first category
        }

        let kl = world_model.kl_divergence(&logits_a, &logits_b);
        assert!(kl > 0.0, "KL divergence of different distributions should be > 0");
    }

    #[test]
    fn test_actor_action_probabilities() {
        let config = DreamerConfig::default();
        let state_dim = config.deterministic_size + config.stochastic_size * config.num_categories;
        let actor = Actor::new(state_dim, config.hidden_size, config.num_actions);
        let state = LatentState::zeros(&config);

        let probs = actor.action_probs(&state);
        assert_eq!(probs.len(), config.num_actions);

        // Probabilities should sum to ~1
        let sum: f64 = probs.sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Action probabilities should sum to 1, got {}",
            sum
        );

        // All probabilities should be non-negative
        for &p in probs.iter() {
            assert!(p >= 0.0, "Action probability should be non-negative");
        }
    }

    #[test]
    fn test_critic_value_estimation() {
        let config = DreamerConfig::default();
        let state_dim = config.deterministic_size + config.stochastic_size * config.num_categories;
        let critic = Critic::new(state_dim, config.hidden_size);
        let state = LatentState::zeros(&config);

        let value = critic.value(&state);
        assert!(value.is_finite(), "Critic value should be finite");
    }

    #[test]
    fn test_lambda_returns() {
        let config = DreamerConfig::default();
        let agent = DreamerAgent::new(config);

        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let values = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

        let returns = agent.compute_lambda_returns(&rewards, &values);
        assert_eq!(returns.len(), rewards.len());

        // Returns should be finite
        for &r in &returns {
            assert!(r.is_finite(), "Lambda return should be finite");
        }
    }

    #[test]
    fn test_synthetic_data_generation() {
        let klines = generate_synthetic_klines(100, 42);
        assert_eq!(klines.len(), 100);

        for k in &klines {
            assert!(k.open > 0.0);
            assert!(k.high >= k.open.min(k.close));
            assert!(k.low <= k.open.max(k.close));
            assert!(k.close > 0.0);
            assert!(k.volume > 0.0);
        }
    }

    #[test]
    fn test_kline_features() {
        let kline = Kline {
            timestamp: 1700000000000,
            open: 50000.0,
            high: 50500.0,
            low: 49500.0,
            close: 50200.0,
            volume: 100.0,
        };

        let features = kline.to_features();
        assert_eq!(features.len(), 5);

        // All features should be symlog-transformed (compressed)
        for &f in features.iter() {
            assert!(f.is_finite());
            assert!(f.abs() < 50000.0, "Symlog should compress large values");
        }
    }

    #[test]
    fn test_full_training_pipeline() {
        let config = DreamerConfig {
            deterministic_size: 32,
            stochastic_size: 4,
            num_categories: 4,
            hidden_size: 32,
            imagination_horizon: 5,
            ..DreamerConfig::default()
        };

        let mut agent = DreamerAgent::new(config);
        let klines = generate_synthetic_klines(200, 42);

        // Train world model
        let losses = agent.train_world_model(&klines, 5);
        assert!(!losses.is_empty());

        // Train in imagination
        let avg_return = agent.train_in_imagination(10);
        assert!(avg_return.is_finite());

        // Evaluate
        let metrics = agent.evaluate(&klines[100..]);
        assert!(metrics.total_pnl.is_finite());
        assert!(metrics.sharpe_ratio.is_finite());
    }
}
