//! # Curiosity-Driven Trading
//!
//! Implementation of Intrinsic Curiosity Module (ICM) and Random Network Distillation (RND)
//! for exploration-driven trading agents. These modules provide intrinsic reward signals
//! that encourage the agent to explore novel market states.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Linear Layer
// ---------------------------------------------------------------------------

/// A simple fully-connected linear layer: y = xW + b with optional ReLU activation.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub use_relu: bool,
}

impl LinearLayer {
    /// Create a new linear layer with Xavier-initialized weights.
    pub fn new(input_dim: usize, output_dim: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
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

    /// Create a layer with fixed random weights (for RND target network).
    pub fn new_fixed_random(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::from_shape_fn(output_dim, |_| rng.gen_range(-0.01..0.01));
        Self {
            weights,
            bias,
            use_relu: true,
        }
    }

    /// Forward pass through the layer.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let output = input.dot(&self.weights) + &self.bias;
        if self.use_relu {
            output.mapv(|x| x.max(0.0))
        } else {
            output
        }
    }

    /// Update weights using gradient descent.
    /// `input` is the layer input, `grad_output` is the gradient of loss w.r.t. output.
    /// Returns gradient w.r.t. input for backpropagation.
    pub fn backward(
        &mut self,
        input: &Array1<f64>,
        output: &Array1<f64>,
        grad_output: &Array1<f64>,
        lr: f64,
    ) -> Array1<f64> {
        // Apply ReLU derivative if needed
        let grad = if self.use_relu {
            let relu_mask = output.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            grad_output * &relu_mask
        } else {
            grad_output.clone()
        };

        // Compute gradients
        let grad_weights =
            input
                .view()
                .insert_axis(ndarray::Axis(1))
                .dot(&grad.view().insert_axis(ndarray::Axis(0)));
        let grad_input = self.weights.dot(&grad);

        // Update parameters
        self.weights = &self.weights - &(grad_weights * lr);
        self.bias = &self.bias - &(&grad * lr);

        grad_input
    }
}

// ---------------------------------------------------------------------------
// Simple MLP (2-layer network)
// ---------------------------------------------------------------------------

/// A two-layer MLP used as a building block for models.
#[derive(Debug, Clone)]
pub struct MLP {
    pub layer1: LinearLayer,
    pub layer2: LinearLayer,
}

impl MLP {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, output_relu: bool) -> Self {
        Self {
            layer1: LinearLayer::new(input_dim, hidden_dim, true),
            layer2: LinearLayer::new(hidden_dim, output_dim, output_relu),
        }
    }

    pub fn new_fixed_random(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            layer1: LinearLayer::new_fixed_random(input_dim, hidden_dim),
            layer2: LinearLayer::new_fixed_random(hidden_dim, output_dim),
        }
    }

    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let h = self.layer1.forward(input);
        self.layer2.forward(&h)
    }

    /// Train the MLP on a single sample, returning the loss.
    pub fn train_step(&mut self, input: &Array1<f64>, target: &Array1<f64>, lr: f64) -> f64 {
        // Forward
        let h = self.layer1.forward(input);
        let output = self.layer2.forward(&h);

        // Compute MSE loss and gradient
        let diff = &output - target;
        let loss = diff.mapv(|x| x * x).sum() / diff.len() as f64;
        let grad_output = &diff * (2.0 / diff.len() as f64);

        // Backward
        let grad_h = self.layer2.backward(&h, &output, &grad_output, lr);
        let _grad_input = self.layer1.backward(input, &h, &grad_h, lr);

        loss
    }
}

// ---------------------------------------------------------------------------
// Feature Encoder
// ---------------------------------------------------------------------------

/// Encodes raw market state observations into a compact feature representation.
#[derive(Debug, Clone)]
pub struct FeatureEncoder {
    pub network: MLP,
    pub input_dim: usize,
    pub feature_dim: usize,
}

impl FeatureEncoder {
    pub fn new(input_dim: usize, hidden_dim: usize, feature_dim: usize) -> Self {
        Self {
            network: MLP::new(input_dim, hidden_dim, feature_dim, true),
            input_dim,
            feature_dim,
        }
    }

    /// Encode a raw state into feature space.
    pub fn encode(&self, state: &Array1<f64>) -> Array1<f64> {
        self.network.forward(state)
    }
}

// ---------------------------------------------------------------------------
// Forward Model
// ---------------------------------------------------------------------------

/// Predicts the next state features given current state features and action.
/// Core component of the ICM that generates prediction error as curiosity signal.
#[derive(Debug, Clone)]
pub struct ForwardModel {
    pub network: MLP,
    pub feature_dim: usize,
    pub num_actions: usize,
}

impl ForwardModel {
    pub fn new(feature_dim: usize, hidden_dim: usize, num_actions: usize) -> Self {
        let input_dim = feature_dim + num_actions;
        Self {
            network: MLP::new(input_dim, hidden_dim, feature_dim, false),
            feature_dim,
            num_actions,
        }
    }

    /// Predict the next state features from current features and one-hot action.
    pub fn predict(&self, state_features: &Array1<f64>, action: usize) -> Array1<f64> {
        let input = self.build_input(state_features, action);
        self.network.forward(&input)
    }

    /// Train on a single transition, returning prediction error (intrinsic reward).
    pub fn train_step(
        &mut self,
        state_features: &Array1<f64>,
        action: usize,
        next_state_features: &Array1<f64>,
        lr: f64,
    ) -> f64 {
        let input = self.build_input(state_features, action);
        self.network.train_step(&input, next_state_features, lr)
    }

    fn build_input(&self, state_features: &Array1<f64>, action: usize) -> Array1<f64> {
        let mut input = Array1::zeros(self.feature_dim + self.num_actions);
        input
            .slice_mut(ndarray::s![..self.feature_dim])
            .assign(state_features);
        if action < self.num_actions {
            input[self.feature_dim + action] = 1.0;
        }
        input
    }
}

// ---------------------------------------------------------------------------
// Inverse Model
// ---------------------------------------------------------------------------

/// Predicts the action taken given two consecutive state feature vectors.
/// Forces the feature encoder to capture action-relevant state aspects.
#[derive(Debug, Clone)]
pub struct InverseModel {
    pub network: MLP,
    pub feature_dim: usize,
    pub num_actions: usize,
}

impl InverseModel {
    pub fn new(feature_dim: usize, hidden_dim: usize, num_actions: usize) -> Self {
        let input_dim = feature_dim * 2;
        Self {
            network: MLP::new(input_dim, hidden_dim, num_actions, false),
            feature_dim,
            num_actions,
        }
    }

    /// Predict action probabilities from two consecutive state features.
    pub fn predict(
        &self,
        state_features: &Array1<f64>,
        next_state_features: &Array1<f64>,
    ) -> Array1<f64> {
        let input = self.build_input(state_features, next_state_features);
        let logits = self.network.forward(&input);
        softmax(&logits)
    }

    /// Train on a single transition, returning the cross-entropy loss.
    pub fn train_step(
        &mut self,
        state_features: &Array1<f64>,
        next_state_features: &Array1<f64>,
        action: usize,
        lr: f64,
    ) -> f64 {
        let input = self.build_input(state_features, next_state_features);
        let logits = self.network.forward(&input);
        let probs = softmax(&logits);

        // Cross-entropy loss
        let loss = -(probs[action].max(1e-10).ln());

        // Gradient of cross-entropy w.r.t. logits (softmax + CE simplification)
        let mut grad = probs.clone();
        grad[action] -= 1.0;
        let grad_scaled = &grad * (1.0 / self.num_actions as f64);

        // Backward pass
        let h = self.network.layer1.forward(&input);
        let output = self.network.layer2.forward(&h);
        let grad_h = self.network.layer2.backward(&h, &output, &grad_scaled, lr);
        let _grad_input = self.network.layer1.backward(&input, &h, &grad_h, lr);

        loss
    }

    fn build_input(
        &self,
        state_features: &Array1<f64>,
        next_state_features: &Array1<f64>,
    ) -> Array1<f64> {
        let mut input = Array1::zeros(self.feature_dim * 2);
        input
            .slice_mut(ndarray::s![..self.feature_dim])
            .assign(state_features);
        input
            .slice_mut(ndarray::s![self.feature_dim..])
            .assign(next_state_features);
        input
    }
}

// ---------------------------------------------------------------------------
// ICM Module
// ---------------------------------------------------------------------------

/// Intrinsic Curiosity Module (ICM) combining forward and inverse models.
/// Provides intrinsic reward based on forward model prediction error.
#[derive(Debug, Clone)]
pub struct ICMModule {
    pub encoder: FeatureEncoder,
    pub forward_model: ForwardModel,
    pub inverse_model: InverseModel,
    pub eta: f64,   // Intrinsic reward scaling factor
    pub alpha: f64, // Balance between forward and inverse loss
}

impl ICMModule {
    pub fn new(
        state_dim: usize,
        feature_dim: usize,
        hidden_dim: usize,
        num_actions: usize,
        eta: f64,
        alpha: f64,
    ) -> Self {
        Self {
            encoder: FeatureEncoder::new(state_dim, hidden_dim, feature_dim),
            forward_model: ForwardModel::new(feature_dim, hidden_dim, num_actions),
            inverse_model: InverseModel::new(feature_dim, hidden_dim, num_actions),
            eta,
            alpha,
        }
    }

    /// Compute the intrinsic reward for a transition.
    pub fn intrinsic_reward(
        &self,
        state: &Array1<f64>,
        action: usize,
        next_state: &Array1<f64>,
    ) -> f64 {
        let phi_s = self.encoder.encode(state);
        let phi_s_next = self.encoder.encode(next_state);
        let predicted_next = self.forward_model.predict(&phi_s, action);

        let error = &predicted_next - &phi_s_next;
        let prediction_error = error.mapv(|x| x * x).sum();

        self.eta * prediction_error
    }

    /// Train on a single transition, returning (intrinsic_reward, forward_loss, inverse_loss).
    pub fn train_step(
        &mut self,
        state: &Array1<f64>,
        action: usize,
        next_state: &Array1<f64>,
        lr: f64,
    ) -> (f64, f64, f64) {
        let phi_s = self.encoder.encode(state);
        let phi_s_next = self.encoder.encode(next_state);

        // Forward model prediction error = intrinsic reward
        let fwd_loss = self
            .forward_model
            .train_step(&phi_s, action, &phi_s_next, lr * self.alpha);

        // Inverse model loss
        let inv_loss =
            self.inverse_model
                .train_step(&phi_s, &phi_s_next, action, lr * (1.0 - self.alpha));

        let intrinsic_reward = self.eta * fwd_loss;

        (intrinsic_reward, fwd_loss, inv_loss)
    }
}

// ---------------------------------------------------------------------------
// RND Module
// ---------------------------------------------------------------------------

/// Random Network Distillation (RND) module.
/// Uses a fixed random target network and a trainable predictor network.
/// Prediction error serves as a novelty / intrinsic reward signal.
#[derive(Debug, Clone)]
pub struct RNDModule {
    pub target_network: MLP,    // Fixed random weights
    pub predictor_network: MLP, // Trainable
    pub output_dim: usize,
}

impl RNDModule {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            target_network: MLP::new_fixed_random(input_dim, hidden_dim, output_dim),
            predictor_network: MLP::new(input_dim, hidden_dim, output_dim, false),
            output_dim,
        }
    }

    /// Compute the intrinsic reward (prediction error) for a state.
    pub fn intrinsic_reward(&self, state: &Array1<f64>) -> f64 {
        let target = self.target_network.forward(state);
        let prediction = self.predictor_network.forward(state);
        let diff = &prediction - &target;
        diff.mapv(|x| x * x).sum() / self.output_dim as f64
    }

    /// Train the predictor on a state, returning the prediction error (intrinsic reward).
    pub fn train_step(&mut self, state: &Array1<f64>, lr: f64) -> f64 {
        let target = self.target_network.forward(state);
        let loss = self.predictor_network.train_step(state, &target, lr);
        loss
    }
}

// ---------------------------------------------------------------------------
// Running Statistics for Reward Normalization
// ---------------------------------------------------------------------------

/// Maintains running mean and variance for normalizing intrinsic rewards.
#[derive(Debug, Clone)]
pub struct RunningStats {
    pub mean: f64,
    pub var: f64,
    pub count: f64,
    pub epsilon: f64,
}

impl RunningStats {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            var: 1.0,
            count: 0.0,
            epsilon: 1e-8,
        }
    }

    /// Update running statistics with a new value.
    pub fn update(&mut self, value: f64) {
        self.count += 1.0;
        let delta = value - self.mean;
        self.mean += delta / self.count;
        let delta2 = value - self.mean;
        self.var += (delta * delta2 - self.var) / self.count;
    }

    /// Normalize a value using the running statistics.
    pub fn normalize(&self, value: f64) -> f64 {
        if self.count < 2.0 {
            return value;
        }
        (value - self.mean) / (self.var.sqrt() + self.epsilon)
    }
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Curiosity-Driven Agent
// ---------------------------------------------------------------------------

/// Actions available to the trading agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingAction {
    Buy = 0,
    Sell = 1,
    Hold = 2,
}

impl TradingAction {
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => TradingAction::Buy,
            1 => TradingAction::Sell,
            _ => TradingAction::Hold,
        }
    }

    pub fn to_index(self) -> usize {
        self as usize
    }
}

/// Configuration for the curiosity-driven trading agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub state_dim: usize,
    pub feature_dim: usize,
    pub hidden_dim: usize,
    pub num_actions: usize,
    pub beta: f64,          // Weight of intrinsic reward
    pub beta_decay: f64,    // Decay factor for beta over time
    pub learning_rate: f64,
    pub icm_eta: f64,
    pub icm_alpha: f64,
    pub use_icm: bool,
    pub use_rnd: bool,
    pub rnd_output_dim: usize,
    pub epsilon: f64,       // Epsilon-greedy exploration
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            state_dim: 8,
            feature_dim: 16,
            hidden_dim: 32,
            num_actions: 3,
            beta: 0.5,
            beta_decay: 0.999,
            learning_rate: 0.001,
            icm_eta: 0.1,
            icm_alpha: 0.5,
            use_icm: true,
            use_rnd: true,
            rnd_output_dim: 16,
            epsilon: 0.1,
        }
    }
}

/// A curiosity-driven trading agent combining extrinsic and intrinsic rewards.
#[derive(Debug, Clone)]
pub struct CuriosityDrivenAgent {
    pub config: AgentConfig,
    pub icm: Option<ICMModule>,
    pub rnd: Option<RNDModule>,
    pub q_network: MLP,
    pub reward_stats: RunningStats,
    pub total_steps: usize,
    pub cumulative_intrinsic_reward: f64,
    pub cumulative_extrinsic_reward: f64,
}

impl CuriosityDrivenAgent {
    pub fn new(config: AgentConfig) -> Self {
        let icm = if config.use_icm {
            Some(ICMModule::new(
                config.state_dim,
                config.feature_dim,
                config.hidden_dim,
                config.num_actions,
                config.icm_eta,
                config.icm_alpha,
            ))
        } else {
            None
        };

        let rnd = if config.use_rnd {
            Some(RNDModule::new(
                config.state_dim,
                config.hidden_dim,
                config.rnd_output_dim,
            ))
        } else {
            None
        };

        let q_network = MLP::new(config.state_dim, config.hidden_dim, config.num_actions, false);

        Self {
            config,
            icm,
            rnd,
            q_network,
            reward_stats: RunningStats::new(),
            total_steps: 0,
            cumulative_intrinsic_reward: 0.0,
            cumulative_extrinsic_reward: 0.0,
        }
    }

    /// Select an action using epsilon-greedy policy with Q-values.
    pub fn select_action(&self, state: &Array1<f64>) -> TradingAction {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.config.epsilon {
            TradingAction::from_index(rng.gen_range(0..self.config.num_actions))
        } else {
            let q_values = self.q_network.forward(state);
            let best_action = q_values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(2);
            TradingAction::from_index(best_action)
        }
    }

    /// Compute the combined intrinsic reward from ICM and RND.
    pub fn compute_intrinsic_reward(
        &self,
        state: &Array1<f64>,
        action: usize,
        next_state: &Array1<f64>,
    ) -> f64 {
        let mut intrinsic = 0.0;
        if let Some(ref icm) = self.icm {
            intrinsic += icm.intrinsic_reward(state, action, next_state);
        }
        if let Some(ref rnd) = self.rnd {
            intrinsic += rnd.intrinsic_reward(state);
        }
        intrinsic
    }

    /// Current beta value (decayed over time).
    pub fn current_beta(&self) -> f64 {
        self.config.beta * self.config.beta_decay.powi(self.total_steps as i32)
    }

    /// Train on a single transition (state, action, reward, next_state).
    /// Returns (total_reward, intrinsic_reward, extrinsic_reward).
    pub fn train_step(
        &mut self,
        state: &Array1<f64>,
        action: TradingAction,
        extrinsic_reward: f64,
        next_state: &Array1<f64>,
    ) -> (f64, f64, f64) {
        let action_idx = action.to_index();
        let lr = self.config.learning_rate;

        // Train ICM and get intrinsic reward
        let mut intrinsic_reward = 0.0;
        if let Some(ref mut icm) = self.icm {
            let (icm_reward, _fwd_loss, _inv_loss) =
                icm.train_step(state, action_idx, next_state, lr);
            intrinsic_reward += icm_reward;
        }

        // Train RND and get intrinsic reward
        if let Some(ref mut rnd) = self.rnd {
            let rnd_reward = rnd.train_step(state, lr);
            intrinsic_reward += rnd_reward;
        }

        // Normalize intrinsic reward
        self.reward_stats.update(intrinsic_reward);
        let normalized_intrinsic = self.reward_stats.normalize(intrinsic_reward);

        // Combine rewards
        let beta = self.current_beta();
        let total_reward = extrinsic_reward + beta * normalized_intrinsic;

        // Update Q-network using simple Q-learning update
        let q_values = self.q_network.forward(state);
        let next_q_values = self.q_network.forward(next_state);
        let max_next_q = next_q_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let gamma = 0.99;
        let mut target_q = q_values.clone();
        target_q[action_idx] = total_reward + gamma * max_next_q;

        self.q_network.train_step(state, &target_q, lr);

        // Update counters
        self.total_steps += 1;
        self.cumulative_intrinsic_reward += intrinsic_reward;
        self.cumulative_extrinsic_reward += extrinsic_reward;

        (total_reward, intrinsic_reward, extrinsic_reward)
    }
}

// ---------------------------------------------------------------------------
// Market State Construction
// ---------------------------------------------------------------------------

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

/// Convert a sequence of candles into a feature vector for the agent.
/// Features: [normalized_return, volatility, volume_ratio, high_low_range,
///            close_position, momentum_3, momentum_5, rsi_approx]
pub fn candles_to_state(candles: &[Candle], index: usize) -> Option<Array1<f64>> {
    if index < 5 || index >= candles.len() {
        return None;
    }

    let current = &candles[index];
    let prev = &candles[index - 1];

    // 1. Normalized return
    let ret = if prev.close > 0.0 {
        (current.close / prev.close).ln()
    } else {
        0.0
    };

    // 2. Volatility (std of last 5 returns)
    let returns: Vec<f64> = (1..=5)
        .filter_map(|i| {
            if index >= i && candles[index - i].close > 0.0 {
                Some((candles[index - i + 1].close / candles[index - i].close).ln())
            } else {
                None
            }
        })
        .collect();
    let vol = if returns.len() > 1 {
        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
        var.sqrt()
    } else {
        0.0
    };

    // 3. Volume ratio (current / average of last 5)
    let avg_vol: f64 = (1..=5)
        .map(|i| if index >= i { candles[index - i].volume } else { 0.0 })
        .sum::<f64>()
        / 5.0;
    let vol_ratio = if avg_vol > 0.0 {
        current.volume / avg_vol
    } else {
        1.0
    };

    // 4. High-low range normalized by close
    let hl_range = if current.close > 0.0 {
        (current.high - current.low) / current.close
    } else {
        0.0
    };

    // 5. Close position within candle range
    let close_pos = if (current.high - current.low).abs() > 1e-10 {
        (current.close - current.low) / (current.high - current.low)
    } else {
        0.5
    };

    // 6. Momentum (3-period return)
    let mom3 = if index >= 3 && candles[index - 3].close > 0.0 {
        (current.close / candles[index - 3].close).ln()
    } else {
        0.0
    };

    // 7. Momentum (5-period return)
    let mom5 = if index >= 5 && candles[index - 5].close > 0.0 {
        (current.close / candles[index - 5].close).ln()
    } else {
        0.0
    };

    // 8. Approximate RSI (ratio of up moves to total moves over 5 periods)
    let up_count = returns.iter().filter(|r| **r > 0.0).count();
    let rsi_approx = if !returns.is_empty() {
        up_count as f64 / returns.len() as f64
    } else {
        0.5
    };

    Some(Array1::from(vec![
        ret * 100.0,       // Scale return
        vol * 100.0,       // Scale volatility
        vol_ratio - 1.0,   // Center volume ratio
        hl_range * 100.0,  // Scale range
        close_pos - 0.5,   // Center close position
        mom3 * 100.0,      // Scale 3-period momentum
        mom5 * 100.0,      // Scale 5-period momentum
        rsi_approx - 0.5,  // Center RSI
    ]))
}

// ---------------------------------------------------------------------------
// Bybit API Client
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

/// Client for fetching market data from Bybit V5 API.
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit.
    /// `symbol`: e.g. "BTCUSDT"
    /// `interval`: e.g. "5" (minutes), "60", "240", "D"
    /// `limit`: number of candles (max 200)
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
            anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
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

// ---------------------------------------------------------------------------
// Simulation Environment
// ---------------------------------------------------------------------------

/// A simple trading simulation environment for backtesting.
pub struct TradingEnvironment {
    pub candles: Vec<Candle>,
    pub current_index: usize,
    pub position: f64,       // +1 long, -1 short, 0 flat
    pub entry_price: f64,
    pub total_pnl: f64,
    pub trade_count: usize,
}

impl TradingEnvironment {
    pub fn new(candles: Vec<Candle>) -> Self {
        Self {
            candles,
            current_index: 5, // Start after enough history for features
            position: 0.0,
            entry_price: 0.0,
            total_pnl: 0.0,
            trade_count: 0,
        }
    }

    /// Reset the environment to the beginning.
    pub fn reset(&mut self) -> Option<Array1<f64>> {
        self.current_index = 5;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.total_pnl = 0.0;
        self.trade_count = 0;
        candles_to_state(&self.candles, self.current_index)
    }

    /// Take an action and return (next_state, extrinsic_reward, done).
    pub fn step(&mut self, action: TradingAction) -> (Option<Array1<f64>>, f64, bool) {
        let current_price = self.candles[self.current_index].close;
        let mut reward = 0.0;

        match action {
            TradingAction::Buy => {
                if self.position <= 0.0 {
                    // Close short if any
                    if self.position < 0.0 {
                        reward = (self.entry_price - current_price) / self.entry_price;
                        self.trade_count += 1;
                    }
                    // Open long
                    self.position = 1.0;
                    self.entry_price = current_price;
                }
            }
            TradingAction::Sell => {
                if self.position >= 0.0 {
                    // Close long if any
                    if self.position > 0.0 {
                        reward = (current_price - self.entry_price) / self.entry_price;
                        self.trade_count += 1;
                    }
                    // Open short
                    self.position = -1.0;
                    self.entry_price = current_price;
                }
            }
            TradingAction::Hold => {
                // Unrealized PnL as small reward signal
                if self.position > 0.0 {
                    reward = 0.01 * (current_price - self.entry_price) / self.entry_price;
                } else if self.position < 0.0 {
                    reward = 0.01 * (self.entry_price - current_price) / self.entry_price;
                }
            }
        }

        self.total_pnl += reward;
        self.current_index += 1;

        let done = self.current_index >= self.candles.len() - 1;
        let next_state = if !done {
            candles_to_state(&self.candles, self.current_index)
        } else {
            None
        };

        (next_state, reward, done)
    }

    /// Check if the environment is done.
    pub fn is_done(&self) -> bool {
        self.current_index >= self.candles.len() - 1
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Softmax function for converting logits to probabilities.
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals = logits.mapv(|x| (x - max_val).exp());
    let sum = exp_vals.sum();
    if sum > 0.0 {
        exp_vals / sum
    } else {
        Array1::from_elem(logits.len(), 1.0 / logits.len() as f64)
    }
}

/// Generate synthetic candle data for testing.
pub fn generate_synthetic_candles(n: usize, regime_changes: usize) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64;
    let regime_length = n / (regime_changes + 1);

    for i in 0..n {
        let regime = i / regime_length;
        let (drift, volatility) = match regime % 4 {
            0 => (0.001, 0.01),   // Bull trend
            1 => (-0.001, 0.02),  // Bear trend with higher vol
            2 => (0.0, 0.005),    // Low vol sideways
            _ => (0.0, 0.04),     // High vol sideways (crisis)
        };

        let ret = drift + volatility * rng.gen_range(-1.0..1.0);
        let new_price = price * (1.0 + ret);
        let high = new_price * (1.0 + rng.gen_range(0.0..0.005));
        let low = new_price * (1.0 - rng.gen_range(0.0..0.005));
        let volume = 100.0 + rng.gen_range(0.0..200.0) * (1.0 + volatility * 10.0);

        candles.push(Candle {
            timestamp: 1000000 + (i as u64) * 60000,
            open: price,
            high,
            low,
            close: new_price,
            volume,
        });
        price = new_price;
    }

    candles
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer_forward() {
        let layer = LinearLayer::new(4, 3, false);
        let input = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let output = layer.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_forward_model_predict() {
        let model = ForwardModel::new(8, 16, 3);
        let features = Array1::from(vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2]);
        let prediction = model.predict(&features, 0);
        assert_eq!(prediction.len(), 8);
    }

    #[test]
    fn test_inverse_model_predict() {
        let model = InverseModel::new(8, 16, 3);
        let s1 = Array1::from(vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2]);
        let s2 = Array1::from(vec![0.2, -0.1, 0.4, 0.3, 0.0, 0.6, -0.2, 0.3]);
        let probs = model.predict(&s1, &s2);
        assert_eq!(probs.len(), 3);
        // Probabilities should sum to ~1.0
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax probs sum = {}", sum);
    }

    #[test]
    fn test_icm_intrinsic_reward() {
        let icm = ICMModule::new(8, 16, 32, 3, 0.1, 0.5);
        let s1 = Array1::from(vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2]);
        let s2 = Array1::from(vec![0.2, -0.1, 0.4, 0.3, 0.0, 0.6, -0.2, 0.3]);
        let reward = icm.intrinsic_reward(&s1, 1, &s2);
        assert!(reward >= 0.0, "Intrinsic reward should be non-negative");
    }

    #[test]
    fn test_rnd_intrinsic_reward() {
        let rnd = RNDModule::new(8, 32, 16);
        let state = Array1::from(vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2]);
        let reward = rnd.intrinsic_reward(&state);
        assert!(reward >= 0.0, "RND reward should be non-negative");
    }

    #[test]
    fn test_rnd_training_reduces_error() {
        let mut rnd = RNDModule::new(8, 32, 16);
        let state = Array1::from(vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2]);

        let initial_error = rnd.intrinsic_reward(&state);
        // Train on the same state multiple times
        for _ in 0..100 {
            rnd.train_step(&state, 0.01);
        }
        let final_error = rnd.intrinsic_reward(&state);

        assert!(
            final_error < initial_error,
            "RND error should decrease with training: {} -> {}",
            initial_error,
            final_error
        );
    }

    #[test]
    fn test_candles_to_state() {
        let candles = generate_synthetic_candles(20, 1);
        let state = candles_to_state(&candles, 10);
        assert!(state.is_some());
        let s = state.unwrap();
        assert_eq!(s.len(), 8);
        // All values should be finite
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_agent_select_action() {
        let config = AgentConfig::default();
        let agent = CuriosityDrivenAgent::new(config);
        let state = Array1::from(vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2]);
        let action = agent.select_action(&state);
        let idx = action.to_index();
        assert!(idx < 3, "Action index should be 0, 1, or 2");
    }

    #[test]
    fn test_agent_train_step() {
        let config = AgentConfig::default();
        let mut agent = CuriosityDrivenAgent::new(config);
        let state = Array1::from(vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2]);
        let next_state = Array1::from(vec![0.2, -0.1, 0.4, 0.3, 0.0, 0.6, -0.2, 0.3]);

        let (total, intrinsic, extrinsic) =
            agent.train_step(&state, TradingAction::Buy, 0.01, &next_state);

        assert!(total.is_finite());
        assert!(intrinsic >= 0.0);
        assert!((extrinsic - 0.01).abs() < 1e-10);
        assert_eq!(agent.total_steps, 1);
    }

    #[test]
    fn test_trading_environment() {
        let candles = generate_synthetic_candles(50, 2);
        let mut env = TradingEnvironment::new(candles);
        let state = env.reset();
        assert!(state.is_some());

        let (next_state, reward, done) = env.step(TradingAction::Buy);
        assert!(next_state.is_some());
        assert!(reward.is_finite());
        assert!(!done);
    }

    #[test]
    fn test_running_stats() {
        let mut stats = RunningStats::new();
        for i in 0..100 {
            stats.update(i as f64);
        }
        // Mean should be close to 49.5
        assert!((stats.mean - 49.5).abs() < 1.0);
        // Normalized value should be roughly zero-mean
        let normalized = stats.normalize(49.5);
        assert!(normalized.abs() < 0.5);
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Largest logit should have largest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_beta_decay() {
        let config = AgentConfig {
            beta: 1.0,
            beta_decay: 0.99,
            ..AgentConfig::default()
        };
        let mut agent = CuriosityDrivenAgent::new(config);
        let initial_beta = agent.current_beta();
        assert!((initial_beta - 1.0).abs() < 1e-10);

        // Simulate some steps
        let state = Array1::from(vec![0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2]);
        let next_state = Array1::from(vec![0.2, -0.1, 0.4, 0.3, 0.0, 0.6, -0.2, 0.3]);
        for _ in 0..10 {
            agent.train_step(&state, TradingAction::Hold, 0.0, &next_state);
        }

        let decayed_beta = agent.current_beta();
        assert!(decayed_beta < initial_beta);
        assert!((decayed_beta - 0.99_f64.powi(10)).abs() < 1e-6);
    }
}
