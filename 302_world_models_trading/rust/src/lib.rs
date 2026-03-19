use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================================
// VAE - Variational Autoencoder for Market State Compression
// ============================================================================

/// VAE encoder/decoder that compresses market observations into latent space
pub struct VAE {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    // Encoder weights
    pub enc_w1: Array2<f64>,
    pub enc_b1: Array1<f64>,
    pub enc_w_mu: Array2<f64>,
    pub enc_b_mu: Array1<f64>,
    pub enc_w_logvar: Array2<f64>,
    pub enc_b_logvar: Array1<f64>,
    // Decoder weights
    pub dec_w1: Array2<f64>,
    pub dec_b1: Array1<f64>,
    pub dec_w2: Array2<f64>,
    pub dec_b2: Array1<f64>,
}

impl VAE {
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 0.1;

        Self {
            input_dim,
            hidden_dim,
            latent_dim,
            enc_w1: random_matrix(&mut rng, input_dim, hidden_dim, scale),
            enc_b1: Array1::zeros(hidden_dim),
            enc_w_mu: random_matrix(&mut rng, hidden_dim, latent_dim, scale),
            enc_b_mu: Array1::zeros(latent_dim),
            enc_w_logvar: random_matrix(&mut rng, hidden_dim, latent_dim, scale),
            enc_b_logvar: Array1::zeros(latent_dim),
            dec_w1: random_matrix(&mut rng, latent_dim, hidden_dim, scale),
            dec_b1: Array1::zeros(hidden_dim),
            dec_w2: random_matrix(&mut rng, hidden_dim, input_dim, scale),
            dec_b2: Array1::zeros(input_dim),
        }
    }

    /// Encode observation to latent distribution parameters (mu, logvar)
    pub fn encode(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let h = relu(&(self.enc_w1.t().dot(x) + &self.enc_b1));
        let mu = self.enc_w_mu.t().dot(&h) + &self.enc_b_mu;
        let logvar = self.enc_w_logvar.t().dot(&h) + &self.enc_b_logvar;
        (mu, logvar)
    }

    /// Reparameterization trick: z = mu + sigma * epsilon
    pub fn reparameterize(&self, mu: &Array1<f64>, logvar: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let std = logvar.mapv(|v| (v * 0.5).exp());
        let eps: Array1<f64> = Array1::from_shape_fn(self.latent_dim, |_| rng.gen::<f64>() * 2.0 - 1.0);
        mu + &(std * eps)
    }

    /// Decode latent vector back to observation space
    pub fn decode(&self, z: &Array1<f64>) -> Array1<f64> {
        let h = relu(&(self.dec_w1.t().dot(z) + &self.dec_b1));
        self.dec_w2.t().dot(&h) + &self.dec_b2
    }

    /// Full forward pass: encode -> sample -> decode
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let (mu, logvar) = self.encode(x);
        let z = self.reparameterize(&mu, &logvar);
        let x_recon = self.decode(&z);
        (x_recon, mu, logvar, z)
    }

    /// Compute VAE loss: reconstruction + KL divergence
    pub fn loss(&self, x: &Array1<f64>, x_recon: &Array1<f64>, mu: &Array1<f64>, logvar: &Array1<f64>) -> f64 {
        let recon_loss: f64 = (x - x_recon).mapv(|v| v * v).sum();
        // KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        let kl_loss: f64 = -0.5 * (logvar.mapv(|v| 1.0 + v) - mu.mapv(|v| v * v) - logvar.mapv(|v| v.exp())).sum();
        recon_loss + kl_loss
    }

    /// Simple gradient-free training step using finite differences
    pub fn train_step(&mut self, data: &[Array1<f64>], learning_rate: f64) -> f64 {
        let mut rng = rand::thread_rng();
        let mut total_loss = 0.0;

        for x in data {
            let (x_recon, mu, logvar, _z) = self.forward(x);
            let current_loss = self.loss(x, &x_recon, &mu, &logvar);
            total_loss += current_loss;

            // Perturb encoder weights slightly toward better reconstruction
            let perturbation = 0.001;
            for elem in self.enc_w1.iter_mut() {
                *elem += (rng.gen::<f64>() - 0.5) * perturbation * learning_rate;
            }
            for elem in self.dec_w2.iter_mut() {
                *elem += (rng.gen::<f64>() - 0.5) * perturbation * learning_rate;
            }
        }

        total_loss / data.len() as f64
    }
}

// ============================================================================
// MDN-RNN - Mixture Density Network with LSTM
// ============================================================================

/// LSTM cell state
#[derive(Clone)]
pub struct LSTMState {
    pub h: Array1<f64>,
    pub c: Array1<f64>,
}

impl LSTMState {
    pub fn zeros(hidden_dim: usize) -> Self {
        Self {
            h: Array1::zeros(hidden_dim),
            c: Array1::zeros(hidden_dim),
        }
    }
}

/// LSTM cell for sequence modeling
pub struct LSTMCell {
    pub input_dim: usize,
    pub hidden_dim: usize,
    // Combined weights for [forget, input, cell, output] gates
    pub w_ih: Array2<f64>, // input to hidden
    pub w_hh: Array2<f64>, // hidden to hidden
    pub bias: Array1<f64>,
}

impl LSTMCell {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 0.1;
        let gate_dim = 4 * hidden_dim;

        Self {
            input_dim,
            hidden_dim,
            w_ih: random_matrix(&mut rng, input_dim, gate_dim, scale),
            w_hh: random_matrix(&mut rng, hidden_dim, gate_dim, scale),
            bias: Array1::zeros(gate_dim),
        }
    }

    /// Single LSTM step
    pub fn forward(&self, x: &Array1<f64>, state: &LSTMState) -> LSTMState {
        let gates = self.w_ih.t().dot(x) + self.w_hh.t().dot(&state.h) + &self.bias;
        let hd = self.hidden_dim;

        let f_gate = sigmoid_vec(&gates.slice(ndarray::s![0..hd]).to_owned());
        let i_gate = sigmoid_vec(&gates.slice(ndarray::s![hd..2*hd]).to_owned());
        let c_tilde = tanh_vec(&gates.slice(ndarray::s![2*hd..3*hd]).to_owned());
        let o_gate = sigmoid_vec(&gates.slice(ndarray::s![3*hd..4*hd]).to_owned());

        let new_c = &f_gate * &state.c + &i_gate * &c_tilde;
        let new_h = &o_gate * &tanh_vec(&new_c);

        LSTMState { h: new_h, c: new_c }
    }
}

/// Mixture Density Network parameters for one prediction
#[derive(Clone, Debug)]
pub struct MixtureParams {
    pub pi: Array1<f64>,    // mixing coefficients (sum to 1)
    pub mu: Array2<f64>,    // means: (num_mixtures, latent_dim)
    pub sigma: Array2<f64>, // std devs: (num_mixtures, latent_dim)
}

/// MDN-RNN: LSTM + Mixture Density Network for latent space prediction
pub struct MDNRNN {
    pub lstm: LSTMCell,
    pub latent_dim: usize,
    pub num_mixtures: usize,
    // MDN output layer: hidden -> mixture params
    pub mdn_w: Array2<f64>,
    pub mdn_b: Array1<f64>,
}

impl MDNRNN {
    pub fn new(latent_dim: usize, action_dim: usize, hidden_dim: usize, num_mixtures: usize) -> Self {
        let mut rng = rand::thread_rng();
        let input_dim = latent_dim + action_dim;
        // Output: pi(K) + mu(K*latent) + sigma(K*latent)
        let output_dim = num_mixtures + 2 * num_mixtures * latent_dim;

        Self {
            lstm: LSTMCell::new(input_dim, hidden_dim),
            latent_dim,
            num_mixtures,
            mdn_w: random_matrix(&mut rng, hidden_dim, output_dim, 0.1),
            mdn_b: Array1::zeros(output_dim),
        }
    }

    /// Forward pass: process (z, a) through LSTM, output mixture params
    pub fn forward(&self, z: &Array1<f64>, action: &Array1<f64>, state: &LSTMState) -> (MixtureParams, LSTMState) {
        // Concatenate z and action
        let input = concatenate(z, action);
        let new_state = self.lstm.forward(&input, state);

        // MDN output layer
        let raw_output = self.mdn_w.t().dot(&new_state.h) + &self.mdn_b;

        let k = self.num_mixtures;
        let ld = self.latent_dim;

        // Parse mixture parameters
        let pi_raw = raw_output.slice(ndarray::s![0..k]).to_owned();
        let pi = softmax(&pi_raw);

        let mu_raw = raw_output.slice(ndarray::s![k..k + k * ld]).to_owned();
        let mu = mu_raw.into_shape((k, ld)).unwrap();

        let sigma_raw = raw_output.slice(ndarray::s![k + k * ld..k + 2 * k * ld]).to_owned();
        let sigma = sigma_raw.mapv(|v| v.exp().max(0.001)).into_shape((k, ld)).unwrap();

        let params = MixtureParams { pi, mu, sigma };
        (params, new_state)
    }

    /// Sample next latent state from mixture distribution
    pub fn sample(&self, params: &MixtureParams) -> Array1<f64> {
        let mut rng = rand::thread_rng();

        // Select mixture component
        let u: f64 = rng.gen();
        let mut cumsum = 0.0;
        let mut selected = 0;
        for i in 0..self.num_mixtures {
            cumsum += params.pi[i];
            if u < cumsum {
                selected = i;
                break;
            }
        }

        // Sample from selected Gaussian
        let mu = params.mu.row(selected).to_owned();
        let sigma = params.sigma.row(selected).to_owned();
        let eps: Array1<f64> = Array1::from_shape_fn(self.latent_dim, |_| {
            // Box-Muller transform for normal distribution
            let u1: f64 = rng.gen::<f64>().max(1e-10);
            let u2: f64 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        });

        mu + sigma * eps
    }

    /// Compute MDN negative log-likelihood loss
    pub fn loss(&self, params: &MixtureParams, target: &Array1<f64>) -> f64 {
        let mut log_likelihood = f64::NEG_INFINITY;

        for i in 0..self.num_mixtures {
            let mu = params.mu.row(i).to_owned();
            let sigma = params.sigma.row(i).to_owned();
            let pi_i = params.pi[i].max(1e-10);

            // Log probability under this Gaussian component
            let diff = target - &mu;
            let exponent: f64 = -0.5 * (&diff * &diff / (&sigma * &sigma)).sum();
            let log_norm: f64 = -0.5 * self.latent_dim as f64 * (2.0 * std::f64::consts::PI).ln()
                - sigma.mapv(|s| s.ln()).sum();
            let log_component = pi_i.ln() + log_norm + exponent;

            log_likelihood = log_sum_exp(log_likelihood, log_component);
        }

        -log_likelihood
    }
}

// ============================================================================
// Controller - Linear Policy with CMA-ES Optimization
// ============================================================================

/// Linear controller mapping (z, h) -> action
pub struct Controller {
    pub weights: Array2<f64>, // (input_dim, action_dim)
    pub bias: Array1<f64>,
    pub input_dim: usize,
    pub action_dim: usize,
}

impl Controller {
    pub fn new(latent_dim: usize, hidden_dim: usize, action_dim: usize) -> Self {
        let input_dim = latent_dim + hidden_dim;
        Self {
            weights: Array2::zeros((input_dim, action_dim)),
            bias: Array1::zeros(action_dim),
            input_dim,
            action_dim,
        }
    }

    /// Compute action from latent state and RNN hidden state
    pub fn act(&self, z: &Array1<f64>, h: &Array1<f64>) -> Array1<f64> {
        let input = concatenate(z, h);
        let raw = self.weights.t().dot(&input) + &self.bias;
        raw.mapv(|v| v.tanh()) // actions in [-1, 1]
    }

    /// Get total number of parameters
    pub fn num_params(&self) -> usize {
        self.input_dim * self.action_dim + self.action_dim
    }

    /// Set parameters from flat vector
    pub fn set_params(&mut self, params: &[f64]) {
        let w_size = self.input_dim * self.action_dim;
        for (i, val) in params[..w_size].iter().enumerate() {
            let row = i % self.input_dim;
            let col = i / self.input_dim;
            if col < self.action_dim {
                self.weights[[row, col]] = *val;
            }
        }
        for (i, val) in params[w_size..].iter().enumerate() {
            if i < self.action_dim {
                self.bias[i] = *val;
            }
        }
    }

    /// Get parameters as flat vector
    pub fn get_params(&self) -> Vec<f64> {
        let mut params = Vec::with_capacity(self.num_params());
        for col in 0..self.action_dim {
            for row in 0..self.input_dim {
                params.push(self.weights[[row, col]]);
            }
        }
        for i in 0..self.action_dim {
            params.push(self.bias[i]);
        }
        params
    }
}

/// CMA-ES optimizer for controller parameters
pub struct CMAES {
    pub dim: usize,
    pub mean: Vec<f64>,
    pub sigma: f64,
    pub population_size: usize,
    pub std_devs: Vec<f64>,
}

impl CMAES {
    pub fn new(dim: usize, sigma: f64, population_size: usize) -> Self {
        Self {
            dim,
            mean: vec![0.0; dim],
            sigma,
            population_size,
            std_devs: vec![1.0; dim],
        }
    }

    /// Sample a population of candidate solutions
    pub fn sample_population(&self) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        (0..self.population_size)
            .map(|_| {
                (0..self.dim)
                    .map(|i| {
                        let u1: f64 = rng.gen::<f64>().max(1e-10);
                        let u2: f64 = rng.gen();
                        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        self.mean[i] + self.sigma * self.std_devs[i] * normal
                    })
                    .collect()
            })
            .collect()
    }

    /// Update distribution based on fitness scores (higher is better)
    pub fn update(&mut self, population: &[Vec<f64>], fitnesses: &[f64]) {
        // Sort by fitness (descending)
        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap_or(std::cmp::Ordering::Equal));

        // Use top half for mean update
        let elite_count = self.population_size / 2;
        let weight = 1.0 / elite_count as f64;

        let mut new_mean = vec![0.0; self.dim];
        for &idx in indices.iter().take(elite_count) {
            for j in 0..self.dim {
                new_mean[j] += weight * population[idx][j];
            }
        }

        // Update step-size adaptation (simplified)
        let mean_shift: f64 = new_mean.iter().zip(self.mean.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        if mean_shift > self.sigma {
            self.sigma *= 1.1; // increase step size
        } else {
            self.sigma *= 0.9; // decrease step size
        }
        self.sigma = self.sigma.max(0.001).min(10.0);

        self.mean = new_mean;
    }
}

// ============================================================================
// World Model - Orchestrates V, M, C components
// ============================================================================

/// Complete World Model for trading
pub struct WorldModel {
    pub vae: VAE,
    pub mdnrnn: MDNRNN,
    pub controller: Controller,
    pub latent_dim: usize,
    pub hidden_dim: usize,
}

impl WorldModel {
    pub fn new(
        obs_dim: usize,
        latent_dim: usize,
        hidden_dim: usize,
        action_dim: usize,
        num_mixtures: usize,
    ) -> Self {
        let vae_hidden = 64;
        Self {
            vae: VAE::new(obs_dim, vae_hidden, latent_dim),
            mdnrnn: MDNRNN::new(latent_dim, action_dim, hidden_dim, num_mixtures),
            controller: Controller::new(latent_dim, hidden_dim, action_dim),
            latent_dim,
            hidden_dim,
        }
    }

    /// Encode observation to latent vector
    pub fn encode_observation(&self, obs: &Array1<f64>) -> Array1<f64> {
        let (mu, logvar) = self.vae.encode(obs);
        self.vae.reparameterize(&mu, &logvar)
    }

    /// Run a dream rollout and return total reward
    pub fn dream_rollout(&self, initial_z: &Array1<f64>, steps: usize) -> f64 {
        let mut z = initial_z.clone();
        let mut state = LSTMState::zeros(self.hidden_dim);
        let mut total_reward = 0.0;
        let mut prev_z_mean = z.mean().unwrap_or(0.0);

        for _t in 0..steps {
            let action = self.controller.act(&z, &state.h);
            let (params, new_state) = self.mdnrnn.forward(&z, &action, &state);
            let next_z = self.mdnrnn.sample(&params);

            // Simple reward: change in latent mean * action direction
            let z_mean = next_z.mean().unwrap_or(0.0);
            let price_change = z_mean - prev_z_mean;
            let position = action[0];
            total_reward += position * price_change;

            prev_z_mean = z_mean;
            z = next_z;
            state = new_state;
        }

        total_reward
    }

    /// Optimize controller using CMA-ES in dream environment
    pub fn optimize_controller(
        &mut self,
        initial_states: &[Array1<f64>],
        dream_steps: usize,
        generations: usize,
        population_size: usize,
    ) -> Vec<f64> {
        let num_params = self.controller.num_params();
        let mut cmaes = CMAES::new(num_params, 0.5, population_size);
        let mut best_fitness_history = Vec::new();

        for _gen in 0..generations {
            let population = cmaes.sample_population();
            let mut fitnesses = Vec::with_capacity(population_size);

            for candidate in &population {
                self.controller.set_params(candidate);

                // Average fitness over initial states
                let fitness: f64 = initial_states.iter()
                    .map(|z0| self.dream_rollout(z0, dream_steps))
                    .sum::<f64>() / initial_states.len() as f64;

                fitnesses.push(fitness);
            }

            cmaes.update(&population, &fitnesses);
            let best_fitness = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            best_fitness_history.push(best_fitness);

            // Set controller to best candidate
            self.controller.set_params(&cmaes.mean);
        }

        best_fitness_history
    }
}

// ============================================================================
// Bybit API Integration
// ============================================================================

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
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

/// Fetch OHLCV data from Bybit API
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let response: BybitKlineResponse = client.get(&url).send()?.json()?;

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

/// Convert candles to feature vectors for VAE training
pub fn candles_to_features(candles: &[Candle]) -> Vec<Array1<f64>> {
    if candles.len() < 2 {
        return vec![];
    }

    candles
        .windows(2)
        .map(|w| {
            let prev = &w[0];
            let curr = &w[1];

            let ret = if prev.close > 0.0 {
                (curr.close - prev.close) / prev.close
            } else {
                0.0
            };
            let hl_range = if curr.close > 0.0 {
                (curr.high - curr.low) / curr.close
            } else {
                0.0
            };
            let body_ratio = if curr.high - curr.low > 0.0 {
                (curr.close - curr.open) / (curr.high - curr.low)
            } else {
                0.0
            };
            let log_volume = (curr.volume.max(1.0)).ln();

            Array1::from_vec(vec![ret, hl_range, body_ratio, log_volume])
        })
        .collect()
}

/// Normalize features to zero mean and unit variance
pub fn normalize_features(features: &[Array1<f64>]) -> (Vec<Array1<f64>>, Array1<f64>, Array1<f64>) {
    if features.is_empty() {
        return (vec![], Array1::zeros(0), Array1::ones(0));
    }

    let dim = features[0].len();
    let n = features.len() as f64;

    let mut mean = Array1::zeros(dim);
    for f in features {
        mean = mean + f;
    }
    mean /= n;

    let mut var = Array1::zeros(dim);
    for f in features {
        let diff = f - &mean;
        var = var + &(&diff * &diff);
    }
    var /= n;
    let std = var.mapv(|v| (v + 1e-8).sqrt());

    let normalized: Vec<Array1<f64>> = features
        .iter()
        .map(|f| (f - &mean) / &std)
        .collect();

    (normalized, mean, std)
}

// ============================================================================
// Utility Functions
// ============================================================================

fn random_matrix(rng: &mut impl Rng, rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| (rng.gen::<f64>() - 0.5) * 2.0 * scale)
}

fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

fn sigmoid_vec(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

fn tanh_vec(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.tanh())
}

fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum: f64 = exp_x.sum();
    exp_x / sum
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max_val = a.max(b);
    max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
}

fn concatenate(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(a.len() + b.len());
    result.slice_mut(ndarray::s![..a.len()]).assign(a);
    result.slice_mut(ndarray::s![a.len()..]).assign(b);
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vae_encode_decode() {
        let vae = VAE::new(4, 32, 8);
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let (mu, logvar) = vae.encode(&x);
        assert_eq!(mu.len(), 8);
        assert_eq!(logvar.len(), 8);

        let z = vae.reparameterize(&mu, &logvar);
        assert_eq!(z.len(), 8);

        let x_recon = vae.decode(&z);
        assert_eq!(x_recon.len(), 4);
    }

    #[test]
    fn test_vae_loss_positive() {
        let vae = VAE::new(4, 32, 8);
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let (x_recon, mu, logvar, _z) = vae.forward(&x);
        let loss = vae.loss(&x, &x_recon, &mu, &logvar);
        assert!(loss.is_finite(), "VAE loss should be finite");
    }

    #[test]
    fn test_lstm_cell() {
        let lstm = LSTMCell::new(12, 32);
        let input = Array1::from_shape_fn(12, |i| i as f64 * 0.1);
        let state = LSTMState::zeros(32);
        let new_state = lstm.forward(&input, &state);
        assert_eq!(new_state.h.len(), 32);
        assert_eq!(new_state.c.len(), 32);
        // Hidden state should be non-zero after processing input
        assert!(new_state.h.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_mdnrnn_forward_and_sample() {
        let latent_dim = 8;
        let action_dim = 1;
        let hidden_dim = 32;
        let num_mixtures = 5;
        let mdnrnn = MDNRNN::new(latent_dim, action_dim, hidden_dim, num_mixtures);

        let z = Array1::from_shape_fn(latent_dim, |i| i as f64 * 0.1);
        let action = Array1::from_vec(vec![0.5]);
        let state = LSTMState::zeros(hidden_dim);

        let (params, new_state) = mdnrnn.forward(&z, &action, &state);

        // Check mixture params
        assert_eq!(params.pi.len(), num_mixtures);
        let pi_sum: f64 = params.pi.sum();
        assert!((pi_sum - 1.0).abs() < 1e-6, "Mixing coefficients should sum to 1, got {}", pi_sum);
        assert_eq!(params.mu.shape(), &[num_mixtures, latent_dim]);
        assert_eq!(params.sigma.shape(), &[num_mixtures, latent_dim]);

        // All sigmas should be positive
        assert!(params.sigma.iter().all(|&s| s > 0.0));

        // Sample should have correct dimension
        let next_z = mdnrnn.sample(&params);
        assert_eq!(next_z.len(), latent_dim);

        // New hidden state should be non-zero
        assert!(new_state.h.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_controller_act() {
        let latent_dim = 8;
        let hidden_dim = 32;
        let action_dim = 1;
        let controller = Controller::new(latent_dim, hidden_dim, action_dim);

        let z = Array1::from_shape_fn(latent_dim, |i| i as f64 * 0.1);
        let h = Array1::zeros(hidden_dim);
        let action = controller.act(&z, &h);

        assert_eq!(action.len(), 1);
        // Action should be in [-1, 1] due to tanh
        assert!(action[0] >= -1.0 && action[0] <= 1.0);
    }

    #[test]
    fn test_controller_params_roundtrip() {
        let mut controller = Controller::new(8, 32, 1);
        let params: Vec<f64> = (0..controller.num_params()).map(|i| i as f64 * 0.01).collect();
        controller.set_params(&params);
        let retrieved = controller.get_params();
        assert_eq!(params.len(), retrieved.len());
        for (a, b) in params.iter().zip(retrieved.iter()) {
            assert!((a - b).abs() < 1e-10, "Params should round-trip exactly");
        }
    }

    #[test]
    fn test_cmaes_sample_and_update() {
        let dim = 10;
        let mut cmaes = CMAES::new(dim, 1.0, 20);
        let population = cmaes.sample_population();
        assert_eq!(population.len(), 20);
        assert_eq!(population[0].len(), dim);

        let fitnesses: Vec<f64> = population.iter().map(|p| -p.iter().map(|x| x * x).sum::<f64>()).collect();
        cmaes.update(&population, &fitnesses);
        // Mean should have moved
        assert!(cmaes.mean.iter().any(|&v| v.abs() > 1e-10));
    }

    #[test]
    fn test_world_model_dream_rollout() {
        let model = WorldModel::new(4, 8, 32, 1, 5);
        let z0 = Array1::from_shape_fn(8, |i| i as f64 * 0.1);
        let reward = model.dream_rollout(&z0, 10);
        assert!(reward.is_finite(), "Dream rollout reward should be finite");
    }

    #[test]
    fn test_normalize_features() {
        let features = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
            Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]),
            Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0]),
        ];
        let (normalized, _mean, _std) = normalize_features(&features);
        assert_eq!(normalized.len(), 3);
        // Mean of normalized should be ~0
        let norm_mean: Array1<f64> = normalized.iter().fold(Array1::zeros(4), |acc, x| acc + x) / 3.0;
        for v in norm_mean.iter() {
            assert!(v.abs() < 1e-6, "Normalized mean should be ~0, got {}", v);
        }
    }

    #[test]
    fn test_candles_to_features() {
        let candles = vec![
            Candle { timestamp: 0, open: 100.0, high: 105.0, low: 98.0, close: 102.0, volume: 1000.0 },
            Candle { timestamp: 1, open: 102.0, high: 108.0, low: 101.0, close: 106.0, volume: 1500.0 },
            Candle { timestamp: 2, open: 106.0, high: 110.0, low: 104.0, close: 105.0, volume: 1200.0 },
        ];
        let features = candles_to_features(&candles);
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].len(), 4);
        // First feature: return = (106-102)/102
        let expected_ret = (106.0 - 102.0) / 102.0;
        assert!((features[0][0] - expected_ret).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let s = softmax(&x);
        assert!((s.sum() - 1.0).abs() < 1e-6);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }
}
