use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ============================================================================
// Cosine Embedding for Quantile τ
// ============================================================================

/// Cosine basis embedding that maps a scalar τ ∈ [0,1] to a high-dimensional vector.
/// φ_j(τ) = ReLU(Σ_i cos(π·i·τ) · w_ij + b_j)
#[derive(Debug, Clone)]
pub struct CosineEmbedding {
    pub n_cosines: usize,
    pub embedding_dim: usize,
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl CosineEmbedding {
    pub fn new(n_cosines: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / n_cosines as f64).sqrt();
        let weights = Array2::from_shape_fn((n_cosines, embedding_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(embedding_dim);
        Self {
            n_cosines,
            embedding_dim,
            weights,
            bias,
        }
    }

    /// Compute cosine embedding for a single τ value
    pub fn forward(&self, tau: f64) -> Array1<f64> {
        // cos_features[i] = cos(π * i * τ) for i = 0..n_cosines
        let cos_features = Array1::from_shape_fn(self.n_cosines, |i| (PI * i as f64 * tau).cos());

        // output = ReLU(cos_features · weights + bias)
        let linear = cos_features.dot(&self.weights) + &self.bias;
        linear.mapv(|x| x.max(0.0))
    }

    /// Compute cosine embeddings for multiple τ values
    pub fn forward_batch(&self, taus: &[f64]) -> Array2<f64> {
        let n = taus.len();
        let mut result = Array2::zeros((n, self.embedding_dim));
        for (i, &tau) in taus.iter().enumerate() {
            let embedding = self.forward(tau);
            result.row_mut(i).assign(&embedding);
        }
        result
    }
}

// ============================================================================
// IQN Network
// ============================================================================

/// IQN Network: maps (state, τ) → Q-values for each action.
/// Architecture: Z_τ(s, a) = f(g(s) ⊙ φ(τ))_a
#[derive(Debug, Clone)]
pub struct IQNNetwork {
    pub state_dim: usize,
    pub action_dim: usize,
    pub hidden_dim: usize,
    pub cosine_embedding: CosineEmbedding,
    // State encoder: g(s)
    pub state_w1: Array2<f64>,
    pub state_b1: Array1<f64>,
    // Value network: f(·) → Q-values
    pub value_w1: Array2<f64>,
    pub value_b1: Array1<f64>,
    pub value_w2: Array2<f64>,
    pub value_b2: Array1<f64>,
}

impl IQNNetwork {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize, n_cosines: usize) -> Self {
        let mut rng = rand::thread_rng();

        let he_state = (2.0 / state_dim as f64).sqrt();
        let he_hidden = (2.0 / hidden_dim as f64).sqrt();

        let state_w1 = Array2::from_shape_fn((state_dim, hidden_dim), |_| {
            rng.gen_range(-he_state..he_state)
        });
        let state_b1 = Array1::zeros(hidden_dim);

        let value_w1 = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            rng.gen_range(-he_hidden..he_hidden)
        });
        let value_b1 = Array1::zeros(hidden_dim);

        let value_w2 = Array2::from_shape_fn((hidden_dim, action_dim), |_| {
            rng.gen_range(-he_hidden..he_hidden)
        });
        let value_b2 = Array1::zeros(action_dim);

        Self {
            state_dim,
            action_dim,
            hidden_dim,
            cosine_embedding: CosineEmbedding::new(n_cosines, hidden_dim),
            state_w1,
            state_b1,
            value_w1,
            value_b1,
            value_w2,
            value_b2,
        }
    }

    /// Encode state through the state encoder: g(s) = ReLU(s · W + b)
    pub fn encode_state(&self, state: &Array1<f64>) -> Array1<f64> {
        let h = state.dot(&self.state_w1) + &self.state_b1;
        h.mapv(|x| x.max(0.0))
    }

    /// Forward pass: compute Q-value for a given state, action, and quantile τ
    /// Z_τ(s, a) = f(g(s) ⊙ φ(τ))_a
    pub fn forward(&self, state: &Array1<f64>, tau: f64) -> Array1<f64> {
        let state_emb = self.encode_state(state);
        let tau_emb = self.cosine_embedding.forward(tau);

        // Element-wise multiplication: g(s) ⊙ φ(τ)
        let combined = &state_emb * &tau_emb;

        // Value network
        let h = combined.dot(&self.value_w1) + &self.value_b1;
        let h = h.mapv(|x| x.max(0.0));
        h.dot(&self.value_w2) + &self.value_b2
    }

    /// Compute quantile values for multiple τ samples
    pub fn forward_batch(
        &self,
        state: &Array1<f64>,
        taus: &[f64],
    ) -> Array2<f64> {
        let n = taus.len();
        let mut result = Array2::zeros((n, self.action_dim));
        for (i, &tau) in taus.iter().enumerate() {
            let q_values = self.forward(state, tau);
            result.row_mut(i).assign(&q_values);
        }
        result
    }

    /// Soft update: θ_target ← τ·θ + (1-τ)·θ_target
    pub fn soft_update(&mut self, source: &IQNNetwork, polyak: f64) {
        let blend = |target: &mut Array2<f64>, src: &Array2<f64>| {
            target.zip_mut_with(src, |t, s| *t = polyak * s + (1.0 - polyak) * *t);
        };
        let blend1 = |target: &mut Array1<f64>, src: &Array1<f64>| {
            target.zip_mut_with(src, |t, s| *t = polyak * s + (1.0 - polyak) * *t);
        };

        blend(&mut self.state_w1, &source.state_w1);
        blend1(&mut self.state_b1, &source.state_b1);
        blend(&mut self.value_w1, &source.value_w1);
        blend1(&mut self.value_b1, &source.value_b1);
        blend(&mut self.value_w2, &source.value_w2);
        blend1(&mut self.value_b2, &source.value_b2);
        blend(
            &mut self.cosine_embedding.weights,
            &source.cosine_embedding.weights,
        );
        blend1(
            &mut self.cosine_embedding.bias,
            &source.cosine_embedding.bias,
        );
    }
}

// ============================================================================
// Quantile Huber Loss
// ============================================================================

/// Huber loss with threshold κ
pub fn huber_loss(u: f64, kappa: f64) -> f64 {
    if u.abs() <= kappa {
        0.5 * u * u
    } else {
        kappa * (u.abs() - 0.5 * kappa)
    }
}

/// Quantile Huber loss: |τ - 1(u < 0)| * L_κ(u) / κ
pub fn quantile_huber_loss(u: f64, tau: f64, kappa: f64) -> f64 {
    let indicator = if u < 0.0 { 1.0 } else { 0.0 };
    let weight = (tau - indicator).abs();
    weight * huber_loss(u, kappa) / kappa
}

/// Compute total quantile Huber loss for a batch of quantile predictions
pub fn compute_iqn_loss(
    predicted_quantiles: &Array2<f64>, // (N_τ, actions)
    target_quantiles: &Array2<f64>,    // (N_τ', actions)
    taus: &[f64],                      // N_τ quantile fractions
    action: usize,
    kappa: f64,
) -> f64 {
    let n_tau = taus.len();
    let n_target = target_quantiles.nrows();
    let mut total_loss = 0.0;

    for i in 0..n_tau {
        for j in 0..n_target {
            let u = target_quantiles[[j, action]] - predicted_quantiles[[i, action]];
            total_loss += quantile_huber_loss(u, taus[i], kappa);
        }
    }

    total_loss / (n_tau * n_target) as f64
}

// ============================================================================
// Risk Measures for Action Selection
// ============================================================================

/// Risk measure for distorting quantile sampling
#[derive(Debug, Clone)]
pub enum RiskMeasure {
    /// Risk-neutral: τ ~ U[0, 1]
    Neutral,
    /// CVaR at level α: τ ~ U[0, α], focuses on worst α fraction
    CVaR { alpha: f64 },
    /// Wang distortion: g_η(τ) = Φ(Φ^{-1}(τ) + η)
    Wang { eta: f64 },
}

impl RiskMeasure {
    /// Sample quantile fractions according to the risk measure
    pub fn sample_taus(&self, n: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        match self {
            RiskMeasure::Neutral => (0..n).map(|_| rng.gen_range(0.0..1.0)).collect(),
            RiskMeasure::CVaR { alpha } => {
                (0..n).map(|_| rng.gen_range(0.0..*alpha)).collect()
            }
            RiskMeasure::Wang { eta } => {
                (0..n)
                    .map(|_| {
                        let u: f64 = rng.gen_range(0.01..0.99);
                        let z = probit(u) + eta;
                        normal_cdf(z).clamp(0.01, 0.99)
                    })
                    .collect()
            }
        }
    }
}

/// Select action using risk-sensitive policy
pub fn select_action_risk_sensitive(
    network: &IQNNetwork,
    state: &Array1<f64>,
    risk_measure: &RiskMeasure,
    n_samples: usize,
) -> usize {
    let taus = risk_measure.sample_taus(n_samples);
    let quantile_values = network.forward_batch(state, &taus);

    // Average over quantile samples for each action
    let mean_q = quantile_values.mean_axis(Axis(0)).unwrap();

    // Select action with highest (risk-adjusted) value
    mean_q
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// Compute the return distribution for a given state and action
pub fn compute_return_distribution(
    network: &IQNNetwork,
    state: &Array1<f64>,
    action: usize,
    n_quantiles: usize,
) -> Vec<(f64, f64)> {
    let taus: Vec<f64> = (0..n_quantiles)
        .map(|i| (i as f64 + 0.5) / n_quantiles as f64)
        .collect();

    let mut distribution = Vec::with_capacity(n_quantiles);
    for &tau in &taus {
        let q_values = network.forward(state, tau);
        distribution.push((tau, q_values[action]));
    }
    distribution
}

// ============================================================================
// Replay Buffer
// ============================================================================

/// Experience tuple for replay
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: Array1<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Array1<f64>,
    pub done: bool,
}

/// Simple replay buffer with uniform sampling
#[derive(Debug)]
pub struct ReplayBuffer {
    pub buffer: Vec<Experience>,
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

    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(experience);
        } else {
            self.buffer[self.position] = experience;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Experience> {
        let mut rng = rand::thread_rng();
        let len = self.buffer.len();
        (0..batch_size)
            .map(|_| {
                let idx = rng.gen_range(0..len);
                &self.buffer[idx]
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

// ============================================================================
// IQN Agent
// ============================================================================

/// IQN Agent with target network and replay buffer
pub struct IQNAgent {
    pub network: IQNNetwork,
    pub target_network: IQNNetwork,
    pub replay_buffer: ReplayBuffer,
    pub risk_measure: RiskMeasure,
    pub gamma: f64,
    pub kappa: f64,
    pub n_quantile_samples: usize,
    pub learning_rate: f64,
    pub polyak_tau: f64,
    pub epsilon: f64,
}

impl IQNAgent {
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        n_cosines: usize,
        buffer_capacity: usize,
        risk_measure: RiskMeasure,
    ) -> Self {
        let network = IQNNetwork::new(state_dim, action_dim, hidden_dim, n_cosines);
        let target_network = network.clone();
        Self {
            network,
            target_network,
            replay_buffer: ReplayBuffer::new(buffer_capacity),
            risk_measure,
            gamma: 0.99,
            kappa: 1.0,
            n_quantile_samples: 8,
            learning_rate: 0.001,
            polyak_tau: 0.005,
            epsilon: 0.1,
        }
    }

    /// Select action with epsilon-greedy exploration
    pub fn select_action(&self, state: &Array1<f64>) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.network.action_dim)
        } else {
            select_action_risk_sensitive(
                &self.network,
                state,
                &self.risk_measure,
                self.n_quantile_samples,
            )
        }
    }

    /// Store experience in replay buffer
    pub fn store_experience(&mut self, experience: Experience) {
        self.replay_buffer.push(experience);
    }

    /// Perform one training step (simplified gradient-free update)
    pub fn train_step(&mut self, batch_size: usize) -> f64 {
        if self.replay_buffer.len() < batch_size {
            return 0.0;
        }

        let batch: Vec<Experience> = self
            .replay_buffer
            .sample(batch_size)
            .into_iter()
            .cloned()
            .collect();
        let mut total_loss = 0.0;

        for exp in &batch {
            let taus = self.risk_measure.sample_taus(self.n_quantile_samples);
            let target_taus: Vec<f64> = RiskMeasure::Neutral
                .sample_taus(self.n_quantile_samples);

            // Current quantile values
            let current_q = self.network.forward_batch(&exp.state, &taus);

            // Target quantile values
            let next_q = self.target_network.forward_batch(&exp.next_state, &target_taus);
            let best_next_action = next_q
                .mean_axis(Axis(0))
                .unwrap()
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            // Compute target: r + γ * Z(s', a*)
            let mut target_quantiles = Array2::zeros((self.n_quantile_samples, self.network.action_dim));
            for i in 0..self.n_quantile_samples {
                let target_val = if exp.done {
                    exp.reward
                } else {
                    exp.reward + self.gamma * next_q[[i, best_next_action]]
                };
                for a in 0..self.network.action_dim {
                    target_quantiles[[i, a]] = target_val;
                }
            }

            let loss = compute_iqn_loss(
                &current_q,
                &target_quantiles,
                &taus,
                exp.action,
                self.kappa,
            );
            total_loss += loss;

            // Simplified parameter update (gradient approximation)
            self.apply_simple_update(&exp.state, &taus, &target_quantiles, exp.action);
        }

        // Soft update target network
        self.target_network
            .soft_update(&self.network, self.polyak_tau);

        total_loss / batch_size as f64
    }

    /// Simplified parameter update using finite differences
    fn apply_simple_update(
        &mut self,
        state: &Array1<f64>,
        taus: &[f64],
        targets: &Array2<f64>,
        action: usize,
    ) {
        let lr = self.learning_rate;
        let kappa = self.kappa;

        for t_idx in 0..taus.len() {
            let tau = taus[t_idx];
            let predicted = self.network.forward(state, tau);
            let target = targets[[t_idx, action]];
            let error = target - predicted[action];

            // Update output layer weights for the action
            let state_emb = self.network.encode_state(state);
            let tau_emb = self.network.cosine_embedding.forward(tau);
            let combined = &state_emb * &tau_emb;
            let h = combined.dot(&self.network.value_w1) + &self.network.value_b1;
            let h = h.mapv(|x| x.max(0.0));

            let grad_sign = if error.abs() <= kappa {
                error
            } else {
                kappa * error.signum()
            };
            let weight = if error < 0.0 {
                (tau - 1.0).abs()
            } else {
                tau
            };
            let scaled_grad = lr * weight * grad_sign / (taus.len() as f64);

            for j in 0..self.network.hidden_dim {
                self.network.value_w2[[j, action]] += scaled_grad * h[j];
            }
            self.network.value_b2[action] += scaled_grad;
        }
    }

    /// Get return distribution for analysis
    pub fn get_return_distribution(
        &self,
        state: &Array1<f64>,
        action: usize,
        n_quantiles: usize,
    ) -> Vec<(f64, f64)> {
        compute_return_distribution(&self.network, state, action, n_quantiles)
    }
}

// ============================================================================
// Trading Environment
// ============================================================================

/// Simple trading environment for crypto
#[derive(Debug, Clone)]
pub struct TradingEnvironment {
    pub prices: Vec<f64>,
    pub current_step: usize,
    pub position: f64, // -1.0 (short), 0.0 (flat), 1.0 (long)
    pub entry_price: f64,
    pub total_pnl: f64,
    pub transaction_cost: f64,
    pub lookback: usize,
}

impl TradingEnvironment {
    pub fn new(prices: Vec<f64>, transaction_cost: f64, lookback: usize) -> Self {
        Self {
            prices,
            current_step: lookback,
            position: 0.0,
            entry_price: 0.0,
            total_pnl: 0.0,
            transaction_cost,
            lookback,
        }
    }

    pub fn reset(&mut self) -> Array1<f64> {
        self.current_step = self.lookback;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.total_pnl = 0.0;
        self.get_state()
    }

    pub fn get_state(&self) -> Array1<f64> {
        let mut features = Vec::new();

        // Log returns for lookback period
        for i in 1..=self.lookback.min(5) {
            if self.current_step >= i {
                let ret = (self.prices[self.current_step] / self.prices[self.current_step - i]).ln();
                features.push(ret);
            } else {
                features.push(0.0);
            }
        }

        // Volatility (std of recent returns)
        let returns: Vec<f64> = (1..=self.lookback.min(10))
            .filter(|&i| self.current_step >= i)
            .map(|i| (self.prices[self.current_step] / self.prices[self.current_step - i]).ln())
            .collect();
        let vol = if returns.len() > 1 {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64).sqrt()
        } else {
            0.0
        };
        features.push(vol);

        // Current position
        features.push(self.position);

        // Unrealized PnL
        let unrealized = if self.position != 0.0 {
            self.position * (self.prices[self.current_step] - self.entry_price) / self.entry_price
        } else {
            0.0
        };
        features.push(unrealized);

        Array1::from_vec(features)
    }

    /// Step the environment. Actions: 0=sell/short, 1=hold, 2=buy/long
    pub fn step(&mut self, action: usize) -> (Array1<f64>, f64, bool) {
        let new_position = match action {
            0 => -1.0,
            1 => 0.0,
            2 => 1.0,
            _ => 0.0,
        };

        let current_price = self.prices[self.current_step];

        // Calculate reward
        let mut reward = 0.0;

        // Position change cost
        let position_change = (new_position - self.position).abs();
        if position_change > 0.0 {
            reward -= self.transaction_cost * position_change;

            // Close existing position
            if self.position != 0.0 {
                let pnl =
                    self.position * (current_price - self.entry_price) / self.entry_price;
                reward += pnl;
                self.total_pnl += pnl;
            }

            // Open new position
            if new_position != 0.0 {
                self.entry_price = current_price;
            }
        }

        self.position = new_position;
        self.current_step += 1;

        // Mark-to-market for held position
        if self.position != 0.0 && self.current_step < self.prices.len() {
            let next_price = self.prices[self.current_step];
            let mtm = self.position * (next_price - current_price) / current_price;
            reward += mtm;
        }

        let done = self.current_step >= self.prices.len() - 1;
        let next_state = if done {
            Array1::zeros(self.get_state().len())
        } else {
            self.get_state()
        };

        (next_state, reward, done)
    }

    pub fn state_dim(&self) -> usize {
        // 5 returns + 1 vol + 1 position + 1 unrealized pnl = 8
        8
    }
}

// ============================================================================
// Bybit API Client
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Client for fetching market data from Bybit
pub struct BybitClient {
    pub base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data from Bybit
    /// Returns Vec of (timestamp, open, high, low, close, volume)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(i64, f64, f64, f64, f64, f64)>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitKlineResponse = reqwest::blocking::get(&url)?.json()?;

        if response.ret_code != 0 {
            anyhow::bail!("Bybit API error: {}", response.ret_msg);
        }

        let mut klines = Vec::new();
        for item in &response.result.list {
            if item.len() >= 6 {
                let timestamp = item[0].parse::<i64>().unwrap_or(0);
                let open = item[1].parse::<f64>().unwrap_or(0.0);
                let high = item[2].parse::<f64>().unwrap_or(0.0);
                let low = item[3].parse::<f64>().unwrap_or(0.0);
                let close = item[4].parse::<f64>().unwrap_or(0.0);
                let volume = item[5].parse::<f64>().unwrap_or(0.0);
                klines.push((timestamp, open, high, low, close, volume));
            }
        }

        // Reverse to chronological order (Bybit returns newest first)
        klines.reverse();
        Ok(klines)
    }

    /// Extract close prices from kline data
    pub fn get_close_prices(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<f64>> {
        let klines = self.fetch_klines(symbol, interval, limit)?;
        Ok(klines.iter().map(|(_, _, _, _, close, _)| *close).collect())
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Approximate probit (inverse normal CDF) using rational approximation
pub fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Rational approximation (Abramowitz & Stegun)
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 {
        -result
    } else {
        result
    }
}

/// Normal CDF approximation
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Horner form)
pub fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Generate synthetic price data for testing (geometric Brownian motion)
pub fn generate_synthetic_prices(n: usize, mu: f64, sigma: f64, initial_price: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(n);
    prices.push(initial_price);

    for _ in 1..n {
        let z: f64 = probit(rng.gen_range(0.01..0.99));
        let dt = 1.0 / 252.0;
        let log_return = (mu - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * z;
        let new_price = prices.last().unwrap() * log_return.exp();
        prices.push(new_price);
    }

    prices
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_embedding_output_shape() {
        let emb = CosineEmbedding::new(64, 32);
        let result = emb.forward(0.5);
        assert_eq!(result.len(), 32);

        let batch = emb.forward_batch(&[0.1, 0.5, 0.9]);
        assert_eq!(batch.shape(), &[3, 32]);
    }

    #[test]
    fn test_cosine_embedding_boundary_values() {
        let emb = CosineEmbedding::new(64, 32);
        let at_zero = emb.forward(0.0);
        let at_one = emb.forward(1.0);
        // Embeddings at 0 and 1 should be different
        let diff: f64 = (&at_zero - &at_one).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0, "Embeddings at τ=0 and τ=1 should differ");
    }

    #[test]
    fn test_iqn_network_forward() {
        let net = IQNNetwork::new(8, 3, 32, 64);
        let state = Array1::from_vec(vec![0.01, -0.02, 0.005, 0.0, -0.01, 0.02, 0.0, 0.0]);
        let q_values = net.forward(&state, 0.5);
        assert_eq!(q_values.len(), 3); // 3 actions
    }

    #[test]
    fn test_iqn_network_batch_forward() {
        let net = IQNNetwork::new(8, 3, 32, 64);
        let state = Array1::from_vec(vec![0.01, -0.02, 0.005, 0.0, -0.01, 0.02, 0.0, 0.0]);
        let taus = vec![0.1, 0.25, 0.5, 0.75, 0.9];
        let result = net.forward_batch(&state, &taus);
        assert_eq!(result.shape(), &[5, 3]);
    }

    #[test]
    fn test_quantile_huber_loss_symmetry() {
        // At τ=0.5, positive and negative errors should have same loss
        let loss_pos = quantile_huber_loss(1.0, 0.5, 1.0);
        let loss_neg = quantile_huber_loss(-1.0, 0.5, 1.0);
        assert!(
            (loss_pos - loss_neg).abs() < 1e-10,
            "At τ=0.5, loss should be symmetric"
        );
    }

    #[test]
    fn test_quantile_huber_loss_asymmetry() {
        // At τ=0.9: if u>0 (target > predicted, i.e. underprediction), weight = τ = 0.9
        // At τ=0.9: if u<0 (target < predicted, i.e. overprediction), weight = 1-τ = 0.1
        let loss_pos = quantile_huber_loss(1.0, 0.9, 1.0); // underprediction, weight=0.9
        let loss_neg = quantile_huber_loss(-1.0, 0.9, 1.0); // overprediction, weight=0.1
        assert!(
            loss_pos > loss_neg,
            "At τ=0.9, underprediction (u>0) should be penalized more"
        );
    }

    #[test]
    fn test_huber_loss_quadratic_region() {
        // For |u| <= κ, Huber loss = 0.5 * u^2
        let loss = huber_loss(0.5, 1.0);
        assert!((loss - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_huber_loss_linear_region() {
        // For |u| > κ, Huber loss = κ * (|u| - 0.5*κ)
        let loss = huber_loss(2.0, 1.0);
        let expected = 1.0 * (2.0 - 0.5);
        assert!((loss - expected).abs() < 1e-10);
    }

    #[test]
    fn test_risk_measure_cvar_sampling() {
        let cvar = RiskMeasure::CVaR { alpha: 0.25 };
        let taus = cvar.sample_taus(1000);
        assert!(taus.iter().all(|&t| t >= 0.0 && t <= 0.25));
    }

    #[test]
    fn test_risk_measure_neutral_sampling() {
        let neutral = RiskMeasure::Neutral;
        let taus = neutral.sample_taus(1000);
        assert!(taus.iter().all(|&t| t >= 0.0 && t <= 1.0));
        // Mean should be approximately 0.5
        let mean: f64 = taus.iter().sum::<f64>() / taus.len() as f64;
        assert!(
            (mean - 0.5).abs() < 0.05,
            "Neutral mean should be ~0.5, got {}",
            mean
        );
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        assert!(buffer.is_empty());

        for i in 0..10 {
            buffer.push(Experience {
                state: Array1::from_vec(vec![i as f64; 4]),
                action: i % 3,
                reward: i as f64 * 0.1,
                next_state: Array1::from_vec(vec![(i + 1) as f64; 4]),
                done: false,
            });
        }

        assert_eq!(buffer.len(), 10);
        let batch = buffer.sample(5);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_trading_environment() {
        let prices = generate_synthetic_prices(100, 0.05, 0.2, 100.0);
        let mut env = TradingEnvironment::new(prices, 0.001, 10);
        let state = env.reset();
        assert_eq!(state.len(), 8);

        let (next_state, _reward, done) = env.step(2); // Buy
        assert!(!done);
        assert_eq!(next_state.len(), 8);
    }

    #[test]
    fn test_return_distribution() {
        let net = IQNNetwork::new(8, 3, 32, 64);
        let state = Array1::from_vec(vec![0.01, -0.02, 0.005, 0.0, -0.01, 0.02, 0.0, 0.0]);
        let dist = compute_return_distribution(&net, &state, 0, 10);
        assert_eq!(dist.len(), 10);
        // τ values should be evenly spaced
        for (i, (tau, _)) in dist.iter().enumerate() {
            let expected = (i as f64 + 0.5) / 10.0;
            assert!((tau - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        let cdf_pos = normal_cdf(1.0);
        let cdf_neg = normal_cdf(-1.0);
        assert!(
            (cdf_pos + cdf_neg - 1.0).abs() < 1e-6,
            "CDF should be symmetric: Φ(x) + Φ(-x) = 1"
        );
    }

    #[test]
    fn test_probit_inverse() {
        // probit(Φ(x)) ≈ x
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let p = normal_cdf(x);
            let recovered = probit(p);
            assert!(
                (recovered - x).abs() < 0.01,
                "probit(Φ({})) = {}, expected {}",
                x,
                recovered,
                x
            );
        }
    }

    #[test]
    fn test_agent_action_selection() {
        let agent = IQNAgent::new(8, 3, 32, 64, 1000, RiskMeasure::Neutral);
        let state = Array1::from_vec(vec![0.01, -0.02, 0.005, 0.0, -0.01, 0.02, 0.0, 0.0]);
        let action = agent.select_action(&state);
        assert!(action < 3);
    }

    #[test]
    fn test_synthetic_prices() {
        let prices = generate_synthetic_prices(100, 0.05, 0.2, 100.0);
        assert_eq!(prices.len(), 100);
        assert!((prices[0] - 100.0).abs() < 1e-10);
        assert!(prices.iter().all(|&p| p > 0.0));
    }
}
