use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// =============================================================================
// Constants
// =============================================================================

/// Number of LOB depth levels on each side (bid/ask)
pub const NUM_LEVELS: usize = 10;
/// Features per level: bid_price, bid_volume, ask_price, ask_volume
pub const FEATURES_PER_LEVEL: usize = 4;
/// Total raw LOB features: 10 levels * 4 features
pub const LOB_FEATURE_SIZE: usize = NUM_LEVELS * FEATURES_PER_LEVEL;
/// Additional derived features: volume imbalances (10) + spread + OFI + cumulative delta
pub const DERIVED_FEATURES: usize = NUM_LEVELS + 3;
/// Total state size
pub const STATE_SIZE: usize = LOB_FEATURE_SIZE + DERIVED_FEATURES;

/// Number of actions: Buy(0), Sell(1), Hold(2)
pub const NUM_ACTIONS: usize = 3;
/// Number of prediction classes: Down(0), Stationary(1), Up(2)
pub const NUM_CLASSES: usize = 3;

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
    /// Create a new dense layer with He initialization
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
// LOB Snapshot
// =============================================================================

/// A single snapshot of the limit order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LOBSnapshot {
    pub timestamp: u64,
    /// Bid prices at each level (best to worst)
    pub bid_prices: Vec<f64>,
    /// Bid volumes at each level
    pub bid_volumes: Vec<f64>,
    /// Ask prices at each level (best to worst)
    pub ask_prices: Vec<f64>,
    /// Ask volumes at each level
    pub ask_volumes: Vec<f64>,
}

impl LOBSnapshot {
    /// Compute the mid-price
    pub fn mid_price(&self) -> f64 {
        if self.bid_prices.is_empty() || self.ask_prices.is_empty() {
            return 0.0;
        }
        (self.bid_prices[0] + self.ask_prices[0]) / 2.0
    }

    /// Compute the bid-ask spread
    pub fn spread(&self) -> f64 {
        if self.bid_prices.is_empty() || self.ask_prices.is_empty() {
            return 0.0;
        }
        self.ask_prices[0] - self.bid_prices[0]
    }

    /// Compute volume imbalance at each level
    pub fn volume_imbalances(&self) -> Vec<f64> {
        self.bid_volumes
            .iter()
            .zip(self.ask_volumes.iter())
            .map(|(bv, av)| {
                let total = bv + av;
                if total > 0.0 {
                    (bv - av) / total
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Convert to a flat feature vector
    pub fn to_features(&self) -> Array1<f64> {
        let mut features = Vec::with_capacity(LOB_FEATURE_SIZE);
        let mid = self.mid_price();
        let normalizer = if mid > 0.0 { mid } else { 1.0 };

        for i in 0..NUM_LEVELS {
            let bp = if i < self.bid_prices.len() {
                self.bid_prices[i] / normalizer - 1.0
            } else {
                0.0
            };
            let bv = if i < self.bid_volumes.len() {
                self.bid_volumes[i]
            } else {
                0.0
            };
            let ap = if i < self.ask_prices.len() {
                self.ask_prices[i] / normalizer - 1.0
            } else {
                0.0
            };
            let av = if i < self.ask_volumes.len() {
                self.ask_volumes[i]
            } else {
                0.0
            };
            features.push(bp);
            features.push(bv);
            features.push(ap);
            features.push(av);
        }

        Array1::from_vec(features)
    }

    /// Get full state vector including derived features
    pub fn to_state(&self, prev: Option<&LOBSnapshot>) -> Array1<f64> {
        let mut state = self.to_features().to_vec();

        // Volume imbalances at each level
        let imbalances = self.volume_imbalances();
        for i in 0..NUM_LEVELS {
            if i < imbalances.len() {
                state.push(imbalances[i]);
            } else {
                state.push(0.0);
            }
        }

        // Normalized spread
        let mid = self.mid_price();
        let spread = if mid > 0.0 {
            self.spread() / mid
        } else {
            0.0
        };
        state.push(spread);

        // Order flow imbalance (OFI)
        let ofi = if let Some(prev_snap) = prev {
            compute_ofi(prev_snap, self)
        } else {
            0.0
        };
        state.push(ofi);

        // Cumulative volume delta (total bid volume - total ask volume)
        let total_bid: f64 = self.bid_volumes.iter().sum();
        let total_ask: f64 = self.ask_volumes.iter().sum();
        let cum_delta = if (total_bid + total_ask) > 0.0 {
            (total_bid - total_ask) / (total_bid + total_ask)
        } else {
            0.0
        };
        state.push(cum_delta);

        Array1::from_vec(state)
    }
}

/// Compute Order Flow Imbalance between two consecutive snapshots
pub fn compute_ofi(prev: &LOBSnapshot, curr: &LOBSnapshot) -> f64 {
    let mut ofi = 0.0;

    // Bid side contribution
    if !prev.bid_prices.is_empty() && !curr.bid_prices.is_empty() {
        if curr.bid_prices[0] >= prev.bid_prices[0] {
            let delta_bv = if !curr.bid_volumes.is_empty() && !prev.bid_volumes.is_empty() {
                curr.bid_volumes[0] - prev.bid_volumes[0]
            } else {
                0.0
            };
            ofi += delta_bv;
        }
    }

    // Ask side contribution
    if !prev.ask_prices.is_empty() && !curr.ask_prices.is_empty() {
        if curr.ask_prices[0] <= prev.ask_prices[0] {
            let delta_av = if !curr.ask_volumes.is_empty() && !prev.ask_volumes.is_empty() {
                curr.ask_volumes[0] - prev.ask_volumes[0]
            } else {
                0.0
            };
            ofi -= delta_av;
        }
    }

    ofi
}

// =============================================================================
// LOB Network
// =============================================================================

/// Neural network for LOB-based mid-price direction prediction
#[derive(Clone)]
pub struct LOBNetwork {
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
    pub layer3: DenseLayer,
    pub output_layer: DenseLayer,
    pub learning_rate: f64,
}

impl LOBNetwork {
    /// Create a new LOB network
    pub fn new(state_size: usize) -> Self {
        Self {
            layer1: DenseLayer::new(state_size, 256, true),
            layer2: DenseLayer::new(256, 128, true),
            layer3: DenseLayer::new(128, 64, true),
            output_layer: DenseLayer::new(64, NUM_CLASSES, false),
            learning_rate: 0.001,
        }
    }

    /// Forward pass: returns class probabilities [Down, Stationary, Up]
    pub fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let h1 = self.layer1.forward(state);
        let h2 = self.layer2.forward(&h1);
        let h3 = self.layer3.forward(&h2);
        let logits = self.output_layer.forward(&h3);
        softmax(&logits)
    }

    /// Get Q-values for trading actions (mapped from class probabilities)
    pub fn q_values(&self, state: &Array1<f64>) -> Array1<f64> {
        let probs = self.forward(state);
        // Map class probabilities to action values:
        // Buy is good when Up is predicted, Sell when Down, Hold when Stationary
        let q_buy = probs[2] - probs[0]; // Up prob - Down prob
        let q_sell = probs[0] - probs[2]; // Down prob - Up prob
        let q_hold = probs[1] * 0.5; // Small reward for predicting stationary
        Array1::from_vec(vec![q_buy, q_sell, q_hold])
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

    /// Get predicted direction: 0=Down, 1=Stationary, 2=Up
    pub fn predict_direction(&self, state: &Array1<f64>) -> usize {
        let probs = self.forward(state);
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }
}

// =============================================================================
// Cross-Entropy Loss
// =============================================================================

/// Compute cross-entropy loss for multi-class classification
pub fn cross_entropy_loss(target: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let epsilon = 1e-8;
    -target
        .iter()
        .zip(predicted.iter())
        .map(|(t, p)| t * (p + epsilon).ln())
        .sum::<f64>()
}

/// Create one-hot encoded target
pub fn one_hot(class: usize, num_classes: usize) -> Array1<f64> {
    let mut target = Array1::zeros(num_classes);
    if class < num_classes {
        target[class] = 1.0;
    }
    target
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
// LOB Agent
// =============================================================================

/// LOB agent with epsilon-greedy exploration
pub struct LOBAgent {
    pub network: LOBNetwork,
    pub target_network: LOBNetwork,
    pub replay_buffer: ReplayBuffer,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_min: f64,
    pub epsilon_decay: f64,
    pub batch_size: usize,
    pub target_update_freq: usize,
    pub step_count: usize,
}

impl LOBAgent {
    pub fn new(state_size: usize) -> Self {
        let network = LOBNetwork::new(state_size);
        let target_network = network.clone();
        Self {
            network,
            target_network,
            replay_buffer: ReplayBuffer::new(10000),
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

        for transition in &batch {
            // Get current predictions
            let current_probs = self.network.forward(&transition.state);

            // Compute target using Bellman equation
            let next_q = self.target_network.q_values(&transition.next_state);
            let best_next_q = next_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let target_value = transition.reward
                + if transition.done {
                    0.0
                } else {
                    self.gamma * best_next_q
                };

            // Create target distribution based on action taken
            let mut target = current_probs.clone();
            target[transition.action] = target_value.clamp(0.0, 1.0);

            // Compute loss
            let loss = cross_entropy_loss(&target, &current_probs);
            total_loss += loss;

            // Simple weight perturbation for training
            self.perturb_weights(&transition.state, transition.action, &target);
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
        state: &Array1<f64>,
        action: usize,
        target: &Array1<f64>,
    ) {
        let mut rng = rand::thread_rng();
        let scale = 0.001;

        // Get current loss
        let current_pred = self.network.forward(state);
        let current_loss = cross_entropy_loss(target, &current_pred);

        // Perturb output layer biases
        for i in 0..self.network.output_layer.biases.len() {
            let original = self.network.output_layer.biases[i];
            let delta: f64 = rng.gen_range(-scale..scale);
            self.network.output_layer.biases[i] = original + delta;

            let new_pred = self.network.forward(state);
            let new_loss = cross_entropy_loss(target, &new_pred);

            if new_loss >= current_loss {
                self.network.output_layer.biases[i] = original;
            }
        }

        // Perturb layer3 biases for deeper learning
        let layer3_len = self.network.layer3.biases.len();
        let perturb_count = (layer3_len / 4).max(1);
        for _ in 0..perturb_count {
            let idx = rng.gen_range(0..layer3_len);
            let original = self.network.layer3.biases[idx];
            let delta: f64 = rng.gen_range(-scale..scale);
            self.network.layer3.biases[idx] = original + delta;

            let new_pred = self.network.forward(state);
            let new_loss = cross_entropy_loss(target, &new_pred);

            if new_loss >= current_loss {
                self.network.layer3.biases[idx] = original;
            }
        }

        let _ = action; // action already incorporated into target
    }

    /// Get class probabilities for a given state
    pub fn predict(&self, state: &Array1<f64>) -> Array1<f64> {
        self.network.forward(state)
    }

    /// Compute volume imbalance signal from current predictions
    pub fn imbalance_signal(&self, state: &Array1<f64>) -> f64 {
        let probs = self.network.forward(state);
        // Up probability minus Down probability: positive means buy signal
        probs[2] - probs[0]
    }
}

// =============================================================================
// LOB Trading Environment
// =============================================================================

/// Trading environment driven by LOB snapshots
pub struct LOBEnvironment {
    pub snapshots: Vec<LOBSnapshot>,
    pub current_step: usize,
    pub position: i32, // -1: short, 0: flat, 1: long
    pub entry_price: f64,
    pub fee_rate: f64,
    pub total_pnl: f64,
    pub num_trades: usize,
}

impl LOBEnvironment {
    pub fn new(snapshots: Vec<LOBSnapshot>) -> Self {
        Self {
            snapshots,
            current_step: 1, // Start at 1 to have a previous snapshot for OFI
            position: 0,
            entry_price: 0.0,
            fee_rate: 0.00075, // 0.075% Bybit futures fee
            total_pnl: 0.0,
            num_trades: 0,
        }
    }

    /// Get current state features
    pub fn get_state(&self) -> Array1<f64> {
        let curr = &self.snapshots[self.current_step];
        let prev = if self.current_step > 0 {
            Some(&self.snapshots[self.current_step - 1])
        } else {
            None
        };
        curr.to_state(prev)
    }

    /// Get state size
    pub fn state_size(&self) -> usize {
        STATE_SIZE
    }

    /// Take action and return (reward, done)
    pub fn step(&mut self, action: usize) -> (f64, bool) {
        let current_mid = self.snapshots[self.current_step].mid_price();
        let spread = self.snapshots[self.current_step].spread();
        let half_spread = spread / 2.0;
        let mut reward = 0.0;

        match action {
            0 => {
                // Buy
                if self.position <= 0 {
                    if self.position == -1 {
                        // Close short
                        let buy_price = current_mid + half_spread;
                        reward = (self.entry_price - buy_price) / self.entry_price
                            - self.fee_rate;
                    }
                    self.position = 1;
                    self.entry_price = current_mid + half_spread; // Buy at ask side
                    reward -= self.fee_rate;
                    self.num_trades += 1;
                }
            }
            1 => {
                // Sell
                if self.position >= 0 {
                    if self.position == 1 {
                        // Close long
                        let sell_price = current_mid - half_spread;
                        reward = (sell_price - self.entry_price) / self.entry_price
                            - self.fee_rate;
                    }
                    self.position = -1;
                    self.entry_price = current_mid - half_spread; // Sell at bid side
                    reward -= self.fee_rate;
                    self.num_trades += 1;
                }
            }
            _ => {
                // Hold
                if self.position != 0 && self.current_step + 1 < self.snapshots.len() {
                    let next_mid = self.snapshots[self.current_step + 1].mid_price();
                    let price_change = (next_mid - current_mid) / current_mid;
                    reward = self.position as f64 * price_change;
                }
            }
        }

        self.total_pnl += reward;
        self.current_step += 1;
        let done = self.current_step >= self.snapshots.len() - 1;

        // Scale reward for better learning
        reward *= 100.0;

        (reward, done)
    }

    /// Reset the environment
    pub fn reset(&mut self) {
        self.current_step = 1;
        self.position = 0;
        self.entry_price = 0.0;
        self.total_pnl = 0.0;
        self.num_trades = 0;
    }
}

// =============================================================================
// LOB Synthesis from OHLCV
// =============================================================================

/// Synthesize LOB snapshots from OHLCV kline data
/// Since full order book data requires direct exchange feeds,
/// we create realistic LOB-style features from candlestick data.
pub fn synthesize_lob_from_klines(klines: &[Kline]) -> Vec<LOBSnapshot> {
    let mut rng = rand::thread_rng();
    let mut snapshots = Vec::with_capacity(klines.len());

    for kline in klines {
        let mid = kline.close;
        let range = kline.high - kline.low;
        let tick_size = range / (2.0 * NUM_LEVELS as f64);
        let base_spread = (range * 0.01).max(mid * 0.0001);

        let mut bid_prices = Vec::with_capacity(NUM_LEVELS);
        let mut bid_volumes = Vec::with_capacity(NUM_LEVELS);
        let mut ask_prices = Vec::with_capacity(NUM_LEVELS);
        let mut ask_volumes = Vec::with_capacity(NUM_LEVELS);

        for i in 0..NUM_LEVELS {
            // Prices spread out from mid
            let offset = base_spread / 2.0 + tick_size * i as f64;
            bid_prices.push(mid - offset);
            ask_prices.push(mid + offset);

            // Volume decreases with distance from mid, with noise
            let base_vol = kline.volume / (NUM_LEVELS as f64 * 2.0);
            let decay = 1.0 / (1.0 + i as f64 * 0.3);
            let noise: f64 = rng.gen_range(0.7..1.3);
            bid_volumes.push(base_vol * decay * noise);

            let noise2: f64 = rng.gen_range(0.7..1.3);
            ask_volumes.push(base_vol * decay * noise2);
        }

        snapshots.push(LOBSnapshot {
            timestamp: kline.timestamp,
            bid_prices,
            bid_volumes,
            ask_prices,
            ask_volumes,
        });
    }

    snapshots
}

// =============================================================================
// Label Generation
// =============================================================================

/// Generate labels from LOB snapshots using smoothed mid-price movement
/// Returns: Vec of (label, smoothed_return) where label is 0=Down, 1=Stationary, 2=Up
pub fn generate_labels(
    snapshots: &[LOBSnapshot],
    horizon: usize,
    threshold: f64,
) -> Vec<(usize, f64)> {
    let mut labels = Vec::new();

    for i in 0..snapshots.len() {
        if i + horizon >= snapshots.len() {
            labels.push((1, 0.0)); // Default to stationary for end of data
            continue;
        }

        let current_mid = snapshots[i].mid_price();
        if current_mid == 0.0 {
            labels.push((1, 0.0));
            continue;
        }

        // Smoothed future mid-price
        let future_avg: f64 = (1..=horizon)
            .map(|j| snapshots[i + j].mid_price())
            .sum::<f64>()
            / horizon as f64;

        let movement = (future_avg - current_mid) / current_mid;

        let label = if movement > threshold {
            2 // Up
        } else if movement < -threshold {
            0 // Down
        } else {
            1 // Stationary
        };

        labels.push((label, movement));
    }

    labels
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
// Metrics
// =============================================================================

/// Compute accuracy of predictions
pub fn compute_accuracy(predictions: &[usize], labels: &[usize]) -> f64 {
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| p == l)
        .count();
    correct as f64 / predictions.len() as f64
}

/// Compute F1 score for a specific class
pub fn compute_f1(predictions: &[usize], labels: &[usize], target_class: usize) -> f64 {
    let true_positives = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(&p, &l)| p == target_class && l == target_class)
        .count() as f64;

    let predicted_positives = predictions
        .iter()
        .filter(|&&p| p == target_class)
        .count() as f64;

    let actual_positives = labels
        .iter()
        .filter(|&&l| l == target_class)
        .count() as f64;

    let precision = if predicted_positives > 0.0 {
        true_positives / predicted_positives
    } else {
        0.0
    };

    let recall = if actual_positives > 0.0 {
        true_positives / actual_positives
    } else {
        0.0
    };

    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

/// Compute Sharpe ratio from a series of returns
pub fn compute_sharpe(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
    let std = var.sqrt();
    if std > 0.0 {
        mean / std * (252.0_f64).sqrt() // Annualized
    } else {
        0.0
    }
}

/// Compute maximum drawdown from cumulative returns
pub fn compute_max_drawdown(returns: &[f64]) -> f64 {
    let mut cumulative = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;

    for r in returns {
        cumulative *= 1.0 + r;
        if cumulative > peak {
            peak = cumulative;
        }
        let dd = (peak - cumulative) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|&p| p > 0.0));
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
    fn test_lob_snapshot_mid_price() {
        let snap = LOBSnapshot {
            timestamp: 0,
            bid_prices: vec![100.0, 99.0],
            bid_volumes: vec![10.0, 20.0],
            ask_prices: vec![101.0, 102.0],
            ask_volumes: vec![15.0, 25.0],
        };
        assert!((snap.mid_price() - 100.5).abs() < 1e-10);
        assert!((snap.spread() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_imbalance() {
        let snap = LOBSnapshot {
            timestamp: 0,
            bid_prices: vec![100.0],
            bid_volumes: vec![30.0],
            ask_prices: vec![101.0],
            ask_volumes: vec![10.0],
        };
        let imb = snap.volume_imbalances();
        assert_eq!(imb.len(), 1);
        // (30 - 10) / (30 + 10) = 0.5
        assert!((imb[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_lob_snapshot_to_features() {
        let snap = LOBSnapshot {
            timestamp: 0,
            bid_prices: vec![99.0; NUM_LEVELS],
            bid_volumes: vec![10.0; NUM_LEVELS],
            ask_prices: vec![101.0; NUM_LEVELS],
            ask_volumes: vec![10.0; NUM_LEVELS],
        };
        let features = snap.to_features();
        assert_eq!(features.len(), LOB_FEATURE_SIZE);
        assert!(features.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_lob_snapshot_to_state() {
        let snap = LOBSnapshot {
            timestamp: 0,
            bid_prices: vec![99.0; NUM_LEVELS],
            bid_volumes: vec![10.0; NUM_LEVELS],
            ask_prices: vec![101.0; NUM_LEVELS],
            ask_volumes: vec![10.0; NUM_LEVELS],
        };
        let state = snap.to_state(None);
        assert_eq!(state.len(), STATE_SIZE);
        assert!(state.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ofi_computation() {
        let prev = LOBSnapshot {
            timestamp: 0,
            bid_prices: vec![100.0],
            bid_volumes: vec![10.0],
            ask_prices: vec![101.0],
            ask_volumes: vec![15.0],
        };
        let curr = LOBSnapshot {
            timestamp: 1,
            bid_prices: vec![100.0],
            bid_volumes: vec![20.0],
            ask_prices: vec![101.0],
            ask_volumes: vec![10.0],
        };
        let ofi = compute_ofi(&prev, &curr);
        // Bid volume increased by 10 (bid price stayed), ask volume decreased by 5 (ask price stayed)
        // OFI = +10 - (-5) = +15
        assert!((ofi - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_lob_network_output() {
        let network = LOBNetwork::new(STATE_SIZE);
        let state = Array1::zeros(STATE_SIZE);
        let probs = network.forward(&state);

        assert_eq!(probs.len(), NUM_CLASSES);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_lob_network_q_values() {
        let network = LOBNetwork::new(STATE_SIZE);
        let state = Array1::zeros(STATE_SIZE);
        let q = network.q_values(&state);
        assert_eq!(q.len(), NUM_ACTIONS);
        assert!(q.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_lob_network_predict_direction() {
        let network = LOBNetwork::new(STATE_SIZE);
        let state = Array1::zeros(STATE_SIZE);
        let direction = network.predict_direction(&state);
        assert!(direction < NUM_CLASSES);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let p = Array1::from_vec(vec![0.2, 0.3, 0.5]);
        let loss = cross_entropy_loss(&p, &p);
        let entropy: f64 = -p.iter().map(|x| x * x.ln()).sum::<f64>();
        assert!((loss - entropy).abs() < 1e-6);
    }

    #[test]
    fn test_one_hot() {
        let oh = one_hot(2, 3);
        assert_eq!(oh.len(), 3);
        assert!((oh[0]).abs() < 1e-10);
        assert!((oh[1]).abs() < 1e-10);
        assert!((oh[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        assert!(buffer.is_empty());

        for i in 0..50 {
            buffer.push(Transition {
                state: Array1::from_elem(STATE_SIZE, i as f64),
                action: i % 3,
                reward: i as f64 * 0.1,
                next_state: Array1::from_elem(STATE_SIZE, (i + 1) as f64),
                done: false,
            });
        }

        assert_eq!(buffer.len(), 50);
        let batch = buffer.sample(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_replay_buffer_overflow() {
        let mut buffer = ReplayBuffer::new(5);
        for i in 0..10 {
            buffer.push(Transition {
                state: Array1::from_elem(4, i as f64),
                action: 0,
                reward: i as f64,
                next_state: Array1::from_elem(4, 0.0),
                done: false,
            });
        }
        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn test_lob_agent_action_selection() {
        let agent = LOBAgent::new(STATE_SIZE);
        let state = Array1::zeros(STATE_SIZE);
        let action = agent.select_action(&state);
        assert!(action < NUM_ACTIONS);
    }

    #[test]
    fn test_generate_labels() {
        let snapshots: Vec<LOBSnapshot> = (0..20)
            .map(|i| LOBSnapshot {
                timestamp: i as u64,
                bid_prices: vec![100.0 + i as f64 * 0.1; NUM_LEVELS],
                bid_volumes: vec![10.0; NUM_LEVELS],
                ask_prices: vec![101.0 + i as f64 * 0.1; NUM_LEVELS],
                ask_volumes: vec![10.0; NUM_LEVELS],
            })
            .collect();

        let labels = generate_labels(&snapshots, 5, 0.001);
        assert_eq!(labels.len(), snapshots.len());
        // With steadily increasing prices, most labels should be Up
        let up_count = labels.iter().filter(|(l, _)| *l == 2).count();
        assert!(up_count > 0);
    }

    #[test]
    fn test_synthesize_lob_from_klines() {
        let klines = vec![
            Kline {
                timestamp: 0,
                open: 50000.0,
                high: 50200.0,
                low: 49800.0,
                close: 50100.0,
                volume: 1000.0,
            },
            Kline {
                timestamp: 1,
                open: 50100.0,
                high: 50300.0,
                low: 49900.0,
                close: 50200.0,
                volume: 1200.0,
            },
        ];
        let snapshots = synthesize_lob_from_klines(&klines);
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].bid_prices.len(), NUM_LEVELS);
        assert_eq!(snapshots[0].ask_prices.len(), NUM_LEVELS);
        // Bid prices should be below mid, ask prices above
        let mid = snapshots[0].mid_price();
        assert!(snapshots[0].bid_prices[0] < mid);
        assert!(snapshots[0].ask_prices[0] > mid);
    }

    #[test]
    fn test_lob_environment() {
        let snapshots: Vec<LOBSnapshot> = (0..50)
            .map(|i| LOBSnapshot {
                timestamp: i as u64,
                bid_prices: vec![100.0 + i as f64 * 0.01; NUM_LEVELS],
                bid_volumes: vec![10.0; NUM_LEVELS],
                ask_prices: vec![101.0 + i as f64 * 0.01; NUM_LEVELS],
                ask_volumes: vec![10.0; NUM_LEVELS],
            })
            .collect();

        let mut env = LOBEnvironment::new(snapshots);
        let state = env.get_state();
        assert_eq!(state.len(), STATE_SIZE);

        let (reward, done) = env.step(0); // Buy
        assert!(reward.is_finite());
        assert!(!done);
    }

    #[test]
    fn test_accuracy() {
        let preds = vec![0, 1, 2, 0, 1];
        let labels = vec![0, 1, 2, 1, 1];
        let acc = compute_accuracy(&preds, &labels);
        assert!((acc - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score() {
        let preds = vec![0, 0, 1, 1, 0];
        let labels = vec![0, 1, 1, 0, 0];
        let f1_class0 = compute_f1(&preds, &labels, 0);
        assert!(f1_class0 > 0.0 && f1_class0 <= 1.0);
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005];
        let sharpe = compute_sharpe(&returns);
        assert!(sharpe.is_finite());
        assert!(sharpe > 0.0); // Positive average return
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, 0.05, -0.2, -0.1, 0.15];
        let mdd = compute_max_drawdown(&returns);
        assert!(mdd > 0.0);
        assert!(mdd <= 1.0);
    }
}
