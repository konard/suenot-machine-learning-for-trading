use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// =============================================================================
// Trading Action Enum
// =============================================================================

/// Discrete trading actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    Buy = 0,
    Hold = 1,
    Sell = 2,
}

impl Action {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Action::Buy,
            1 => Action::Hold,
            _ => Action::Sell,
        }
    }

    pub fn num_actions() -> usize {
        3
    }
}

// =============================================================================
// Expert Dataset
// =============================================================================

/// A single state-action pair from expert demonstrations
#[derive(Debug, Clone)]
pub struct ExpertSample {
    pub state: Vec<f64>,
    pub action: Action,
}

/// Expert dataset: collection of state-action pairs
#[derive(Debug, Clone)]
pub struct ExpertDataset {
    pub samples: Vec<ExpertSample>,
    pub state_dim: usize,
}

/// Expert strategy type for generating demonstrations
#[derive(Debug, Clone, Copy)]
pub enum ExpertStrategy {
    /// Buy when short MA > long MA, sell when short MA < long MA
    TrendFollowing {
        short_window: usize,
        long_window: usize,
    },
    /// Buy when price is below lower band, sell when above upper band
    MeanReversion { window: usize, num_std: f64 },
}

impl ExpertDataset {
    /// Construct expert dataset from price data using a specified strategy
    pub fn from_prices(prices: &[f64], volumes: &[f64], strategy: ExpertStrategy) -> Self {
        let lookback = match strategy {
            ExpertStrategy::TrendFollowing { long_window, .. } => long_window,
            ExpertStrategy::MeanReversion { window, .. } => window,
        };

        let mut samples = Vec::new();
        let state_dim = 8; // features per state

        for i in lookback..prices.len() {
            let state = Self::extract_features(prices, volumes, i, lookback);
            let action = Self::expert_action(prices, i, &strategy);
            samples.push(ExpertSample { state, action });
        }

        ExpertDataset { samples, state_dim }
    }

    /// Extract feature vector from price/volume data at index i
    pub fn extract_features(prices: &[f64], volumes: &[f64], i: usize, lookback: usize) -> Vec<f64> {
        let mut features = Vec::with_capacity(8);

        // Feature 1: normalized return (1-step)
        let ret_1 = if prices[i - 1] != 0.0 {
            (prices[i] - prices[i - 1]) / prices[i - 1]
        } else {
            0.0
        };
        features.push(ret_1);

        // Feature 2: normalized return (5-step)
        let ret_5 = if i >= 5 && prices[i - 5] != 0.0 {
            (prices[i] - prices[i - 5]) / prices[i - 5]
        } else {
            0.0
        };
        features.push(ret_5);

        // Feature 3: short MA / price ratio
        let short_w = lookback.min(5);
        let short_ma: f64 =
            prices[i + 1 - short_w..=i].iter().sum::<f64>() / short_w as f64;
        features.push(if prices[i] != 0.0 {
            short_ma / prices[i] - 1.0
        } else {
            0.0
        });

        // Feature 4: long MA / price ratio
        let long_ma: f64 =
            prices[i + 1 - lookback..=i].iter().sum::<f64>() / lookback as f64;
        features.push(if prices[i] != 0.0 {
            long_ma / prices[i] - 1.0
        } else {
            0.0
        });

        // Feature 5: volatility (std of returns over lookback)
        let returns: Vec<f64> = (i + 1 - lookback..i)
            .map(|j| {
                if prices[j] != 0.0 {
                    (prices[j + 1] - prices[j]) / prices[j]
                } else {
                    0.0
                }
            })
            .collect();
        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
        features.push(var.sqrt());

        // Feature 6: volume ratio
        let vol_ma: f64 = if volumes.len() > i && i >= lookback {
            let v_slice = &volumes[i + 1 - lookback..=i];
            v_slice.iter().sum::<f64>() / v_slice.len() as f64
        } else {
            1.0
        };
        let cur_vol = if volumes.len() > i { volumes[i] } else { 1.0 };
        features.push(if vol_ma != 0.0 {
            cur_vol / vol_ma - 1.0
        } else {
            0.0
        });

        // Feature 7: momentum (RSI-like)
        let gains: f64 = returns.iter().filter(|r| **r > 0.0).sum();
        let losses: f64 = returns.iter().filter(|r| **r < 0.0).map(|r| r.abs()).sum();
        let rsi = if gains + losses > 0.0 {
            gains / (gains + losses) * 2.0 - 1.0 // normalize to [-1, 1]
        } else {
            0.0
        };
        features.push(rsi);

        // Feature 8: price position in recent range
        let recent_high = prices[i + 1 - lookback..=i]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let recent_low = prices[i + 1 - lookback..=i]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let range = recent_high - recent_low;
        features.push(if range > 0.0 {
            (prices[i] - recent_low) / range * 2.0 - 1.0
        } else {
            0.0
        });

        features
    }

    /// Determine expert action based on strategy
    pub fn expert_action(prices: &[f64], i: usize, strategy: &ExpertStrategy) -> Action {
        match strategy {
            ExpertStrategy::TrendFollowing {
                short_window,
                long_window,
            } => {
                let short_ma: f64 = prices[i + 1 - short_window..=i].iter().sum::<f64>()
                    / *short_window as f64;
                let long_ma: f64 = prices[i + 1 - long_window..=i].iter().sum::<f64>()
                    / *long_window as f64;
                let threshold = long_ma * 0.001; // small threshold to avoid noise
                if short_ma > long_ma + threshold {
                    Action::Buy
                } else if short_ma < long_ma - threshold {
                    Action::Sell
                } else {
                    Action::Hold
                }
            }
            ExpertStrategy::MeanReversion { window, num_std } => {
                let ma: f64 =
                    prices[i + 1 - window..=i].iter().sum::<f64>() / *window as f64;
                let var: f64 = prices[i + 1 - window..=i]
                    .iter()
                    .map(|p| (p - ma).powi(2))
                    .sum::<f64>()
                    / *window as f64;
                let std = var.sqrt();
                let upper = ma + num_std * std;
                let lower = ma - num_std * std;
                if prices[i] < lower {
                    Action::Buy
                } else if prices[i] > upper {
                    Action::Sell
                } else {
                    Action::Hold
                }
            }
        }
    }

    /// Get states as ndarray matrix
    pub fn states_matrix(&self) -> Array2<f64> {
        let n = self.samples.len();
        let mut mat = Array2::zeros((n, self.state_dim));
        for (i, sample) in self.samples.iter().enumerate() {
            for (j, &val) in sample.state.iter().enumerate() {
                mat[[i, j]] = val;
            }
        }
        mat
    }

    /// Get actions as vector of indices
    pub fn action_indices(&self) -> Vec<usize> {
        self.samples.iter().map(|s| s.action as usize).collect()
    }

    /// Merge another dataset into this one
    pub fn merge(&mut self, other: &ExpertDataset) {
        self.samples.extend(other.samples.iter().cloned());
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

// =============================================================================
// BC Policy (Feedforward Neural Network)
// =============================================================================

/// A simple feedforward neural network for behavioral cloning
#[derive(Debug, Clone)]
pub struct BCPolicy {
    /// Weights for layer 1: (hidden_dim, state_dim)
    w1: Array2<f64>,
    /// Biases for layer 1
    b1: Array1<f64>,
    /// Weights for layer 2: (hidden_dim2, hidden_dim)
    w2: Array2<f64>,
    /// Biases for layer 2
    b2: Array1<f64>,
    /// Weights for output layer: (num_actions, hidden_dim2)
    w3: Array2<f64>,
    /// Biases for output layer
    b3: Array1<f64>,
    /// State dimension
    pub state_dim: usize,
    /// Number of actions
    pub num_actions: usize,
}

impl BCPolicy {
    /// Create a new BC policy with random initialization
    pub fn new(state_dim: usize, hidden_dim: usize, hidden_dim2: usize, num_actions: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / state_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();
        let scale3 = (2.0 / hidden_dim2 as f64).sqrt();

        let w1 = Array2::from_shape_fn((hidden_dim, state_dim), |_| {
            rng.gen_range(-scale1..scale1)
        });
        let b1 = Array1::zeros(hidden_dim);

        let w2 = Array2::from_shape_fn((hidden_dim2, hidden_dim), |_| {
            rng.gen_range(-scale2..scale2)
        });
        let b2 = Array1::zeros(hidden_dim2);

        let w3 = Array2::from_shape_fn((num_actions, hidden_dim2), |_| {
            rng.gen_range(-scale3..scale3)
        });
        let b3 = Array1::zeros(num_actions);

        BCPolicy {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            state_dim,
            num_actions,
        }
    }

    /// Forward pass: state -> logits
    pub fn forward(&self, state: &[f64]) -> Vec<f64> {
        let x = Array1::from_vec(state.to_vec());

        // Layer 1: linear + ReLU
        let h1 = self.w1.dot(&x) + &self.b1;
        let h1 = h1.mapv(|v| v.max(0.0));

        // Layer 2: linear + ReLU
        let h2 = self.w2.dot(&h1) + &self.b2;
        let h2 = h2.mapv(|v| v.max(0.0));

        // Output layer: linear (logits)
        let logits = self.w3.dot(&h2) + &self.b3;
        logits.to_vec()
    }

    /// Forward pass returning softmax probabilities
    pub fn predict_probs(&self, state: &[f64]) -> Vec<f64> {
        let logits = self.forward(state);
        softmax(&logits)
    }

    /// Predict the most likely action
    pub fn predict_action(&self, state: &[f64]) -> Action {
        let probs = self.predict_probs(state);
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        Action::from_index(best)
    }

    /// Train on a dataset using SGD with cross-entropy loss
    pub fn train(
        &mut self,
        dataset: &ExpertDataset,
        num_epochs: usize,
        learning_rate: f64,
        batch_size: usize,
    ) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut losses = Vec::new();

        for _epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            // Shuffle indices
            let mut indices: Vec<usize> = (0..dataset.len()).collect();
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }

            for batch_start in (0..dataset.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(dataset.len());
                let batch_indices = &indices[batch_start..batch_end];

                // Accumulate gradients
                let mut dw1: Array2<f64> = Array2::zeros(self.w1.dim());
                let mut db1: Array1<f64> = Array1::zeros(self.b1.dim());
                let mut dw2: Array2<f64> = Array2::zeros(self.w2.dim());
                let mut db2: Array1<f64> = Array1::zeros(self.b2.dim());
                let mut dw3: Array2<f64> = Array2::zeros(self.w3.dim());
                let mut db3: Array1<f64> = Array1::zeros(self.b3.dim());
                let mut batch_loss = 0.0;

                for &idx in batch_indices {
                    let sample = &dataset.samples[idx];
                    let x = Array1::from_vec(sample.state.clone());
                    let target = sample.action as usize;

                    // Forward pass with intermediate activations
                    let z1 = self.w1.dot(&x) + &self.b1;
                    let h1 = z1.mapv(|v| v.max(0.0));

                    let z2 = self.w2.dot(&h1) + &self.b2;
                    let h2 = z2.mapv(|v| v.max(0.0));

                    let logits = self.w3.dot(&h2) + &self.b3;
                    let probs = softmax(&logits.to_vec());

                    // Cross-entropy loss
                    let p = probs[target].max(1e-10);
                    batch_loss -= p.ln();

                    // Backprop through softmax + cross-entropy
                    let mut d_logits = Array1::from_vec(probs.clone());
                    d_logits[target] -= 1.0;

                    // Gradients for output layer
                    for a in 0..self.num_actions {
                        for j in 0..h2.len() {
                            dw3[[a, j]] += d_logits[a] * h2[j];
                        }
                        db3[a] += d_logits[a];
                    }

                    // Backprop through layer 2
                    let d_h2 = self.w3.t().dot(&d_logits);
                    let d_z2 = &d_h2 * &z2.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

                    for i in 0..d_z2.len() {
                        for j in 0..h1.len() {
                            dw2[[i, j]] += d_z2[i] * h1[j];
                        }
                        db2[i] += d_z2[i];
                    }

                    // Backprop through layer 1
                    let d_h1 = self.w2.t().dot(&d_z2);
                    let d_z1 = &d_h1 * &z1.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

                    for i in 0..d_z1.len() {
                        for j in 0..x.len() {
                            dw1[[i, j]] += d_z1[i] * x[j];
                        }
                        db1[i] += d_z1[i];
                    }
                }

                let bs = batch_indices.len() as f64;
                epoch_loss += batch_loss / bs;
                num_batches += 1;

                // Update weights
                let lr = learning_rate / bs;
                self.w1 = &self.w1 - &(&dw1 * lr);
                self.b1 = &self.b1 - &(&db1 * lr);
                self.w2 = &self.w2 - &(&dw2 * lr);
                self.b2 = &self.b2 - &(&db2 * lr);
                self.w3 = &self.w3 - &(&dw3 * lr);
                self.b3 = &self.b3 - &(&db3 * lr);
            }

            if num_batches > 0 {
                losses.push(epoch_loss / num_batches as f64);
            }
        }

        losses
    }

    /// Evaluate accuracy on a dataset
    pub fn evaluate(&self, dataset: &ExpertDataset) -> f64 {
        if dataset.is_empty() {
            return 0.0;
        }
        let correct: usize = dataset
            .samples
            .iter()
            .filter(|s| self.predict_action(&s.state) == s.action)
            .count();
        correct as f64 / dataset.len() as f64
    }
}

// =============================================================================
// DAgger Trainer
// =============================================================================

/// Configuration for DAgger training
#[derive(Debug, Clone)]
pub struct DAggerConfig {
    /// Number of DAgger iterations
    pub num_iterations: usize,
    /// Initial mixing parameter (1.0 = pure expert)
    pub initial_beta: f64,
    /// Beta decay factor per iteration
    pub beta_decay: f64,
    /// Training epochs per iteration
    pub epochs_per_iter: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of rollout steps for data collection
    pub rollout_steps: usize,
}

impl Default for DAggerConfig {
    fn default() -> Self {
        DAggerConfig {
            num_iterations: 5,
            initial_beta: 0.7,
            beta_decay: 0.6,
            epochs_per_iter: 30,
            learning_rate: 0.01,
            batch_size: 32,
            rollout_steps: 100,
        }
    }
}

/// DAgger trainer for iterative dataset aggregation
pub struct DAggerTrainer {
    pub config: DAggerConfig,
    pub policy: BCPolicy,
    pub dataset: ExpertDataset,
    pub iteration_accuracies: Vec<f64>,
}

impl DAggerTrainer {
    /// Create a new DAgger trainer
    pub fn new(policy: BCPolicy, initial_dataset: ExpertDataset, config: DAggerConfig) -> Self {
        DAggerTrainer {
            config,
            policy,
            dataset: initial_dataset,
            iteration_accuracies: Vec::new(),
        }
    }

    /// Run the DAgger loop
    pub fn run(
        &mut self,
        prices: &[f64],
        volumes: &[f64],
        strategy: ExpertStrategy,
    ) -> Vec<f64> {
        let mut all_losses = Vec::new();
        let mut beta = self.config.initial_beta;

        for iter in 0..self.config.num_iterations {
            // Train policy on current aggregated dataset
            let losses = self.policy.train(
                &self.dataset,
                self.config.epochs_per_iter,
                self.config.learning_rate,
                self.config.batch_size,
            );
            all_losses.extend(losses);

            // Evaluate current policy
            let accuracy = self.policy.evaluate(&self.dataset);
            self.iteration_accuracies.push(accuracy);

            // Collect new data using mixing policy
            let new_samples = self.collect_rollout(prices, volumes, &strategy, beta);

            if !new_samples.is_empty() {
                let new_dataset = ExpertDataset {
                    samples: new_samples,
                    state_dim: self.dataset.state_dim,
                };
                self.dataset.merge(&new_dataset);
            }

            beta *= self.config.beta_decay;

            if iter < self.config.num_iterations - 1 {
                let _ = iter; // suppress unused variable
            }
        }

        all_losses
    }

    /// Collect rollout data using the mixing policy
    fn collect_rollout(
        &self,
        prices: &[f64],
        volumes: &[f64],
        strategy: &ExpertStrategy,
        beta: f64,
    ) -> Vec<ExpertSample> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::new();

        let lookback = match strategy {
            ExpertStrategy::TrendFollowing { long_window, .. } => *long_window,
            ExpertStrategy::MeanReversion { window, .. } => *window,
        };

        let max_steps = self.config.rollout_steps.min(prices.len().saturating_sub(lookback));

        for step in 0..max_steps {
            let i = lookback + step;
            if i >= prices.len() {
                break;
            }

            let state = ExpertDataset::extract_features(prices, volumes, i, lookback);

            // Mixing policy: with probability beta, use expert; otherwise use learned policy
            let _policy_action = if rng.gen::<f64>() < beta {
                ExpertDataset::expert_action(prices, i, strategy)
            } else {
                self.policy.predict_action(&state)
            };

            // Always label with expert action (key insight of DAgger)
            let expert_action = ExpertDataset::expert_action(prices, i, strategy);

            samples.push(ExpertSample {
                state,
                action: expert_action,
            });
        }

        samples
    }
}

// =============================================================================
// Covariate Shift Analysis
// =============================================================================

/// Analyzes covariate shift between expert and policy state distributions
pub struct CovariateShiftAnalyzer;

impl CovariateShiftAnalyzer {
    /// Compute simple distribution statistics for comparison
    pub fn compute_distribution_stats(states: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
        if states.is_empty() {
            return (vec![], vec![]);
        }
        let dim = states[0].len();
        let n = states.len() as f64;

        let mut means = vec![0.0; dim];
        let mut vars = vec![0.0; dim];

        for state in states {
            for (j, &val) in state.iter().enumerate() {
                means[j] += val / n;
            }
        }

        for state in states {
            for (j, &val) in state.iter().enumerate() {
                vars[j] += (val - means[j]).powi(2) / n;
            }
        }

        (means, vars)
    }

    /// Estimate KL divergence between two distributions (Gaussian approximation)
    pub fn estimate_kl_divergence(
        expert_states: &[Vec<f64>],
        policy_states: &[Vec<f64>],
    ) -> f64 {
        let (mu_e, var_e) = Self::compute_distribution_stats(expert_states);
        let (mu_p, var_p) = Self::compute_distribution_stats(policy_states);

        if mu_e.is_empty() || mu_p.is_empty() {
            return 0.0;
        }

        let dim = mu_e.len();
        let mut kl = 0.0;

        for j in 0..dim {
            let sigma_e = var_e[j].max(1e-10);
            let sigma_p = var_p[j].max(1e-10);

            // KL(P_expert || P_policy) for univariate Gaussians, summed over dimensions
            kl += (sigma_p / sigma_e).ln()
                + (sigma_e + (mu_e[j] - mu_p[j]).powi(2)) / (2.0 * sigma_p)
                - 0.5;
        }

        kl / dim as f64
    }

    /// Measure feature-wise drift between expert and policy distributions
    pub fn feature_drift(
        expert_states: &[Vec<f64>],
        policy_states: &[Vec<f64>],
    ) -> Vec<f64> {
        let (mu_e, var_e) = Self::compute_distribution_stats(expert_states);
        let (mu_p, _var_p) = Self::compute_distribution_stats(policy_states);

        mu_e.iter()
            .zip(mu_p.iter())
            .zip(var_e.iter())
            .map(|((&me, &mp), &ve)| {
                let std = ve.sqrt().max(1e-10);
                ((me - mp) / std).abs()
            })
            .collect()
    }
}

// =============================================================================
// Bybit API Client
// =============================================================================

/// OHLCV candle data
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
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Bybit REST API client for fetching market data
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        BybitClient {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (OHLCV) data from Bybit
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
            anyhow::bail!("Bybit API error: ret_code={}", resp.ret_code);
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

        // Bybit returns newest first, reverse for chronological order
        klines.reverse();
        Ok(klines)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Compute softmax of a vector
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Generate synthetic price data (geometric Brownian motion)
pub fn generate_synthetic_prices(
    initial_price: f64,
    num_steps: usize,
    drift: f64,
    volatility: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut prices = Vec::with_capacity(num_steps);
    let mut volumes = Vec::with_capacity(num_steps);

    prices.push(initial_price);
    volumes.push(1000.0);

    for _ in 1..num_steps {
        let z: f64 = rng.gen::<f64>() * 2.0 - 1.0; // uniform approx for normal
        let ret = drift + volatility * z;
        let new_price = prices.last().unwrap() * (1.0 + ret);
        prices.push(new_price.max(0.01));
        volumes.push(1000.0 * (1.0 + rng.gen::<f64>() * 0.5));
    }

    (prices, volumes)
}

/// Simulate a random policy and return total return
pub fn simulate_random_policy(prices: &[f64], num_trades: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let mut cash = 10000.0;
    let mut position = 0.0;
    let lookback = 20;

    for i in lookback..prices.len().min(lookback + num_trades) {
        let action_idx: usize = rng.gen_range(0..3);
        let action = Action::from_index(action_idx);

        match action {
            Action::Buy => {
                if cash > 0.0 {
                    position = cash / prices[i];
                    cash = 0.0;
                }
            }
            Action::Sell => {
                if position > 0.0 {
                    cash = position * prices[i];
                    position = 0.0;
                }
            }
            Action::Hold => {}
        }
    }

    // Final value
    let final_value = cash + position * prices[prices.len() - 1];
    (final_value - 10000.0) / 10000.0
}

/// Simulate a policy on price data and return total return
pub fn simulate_policy(policy: &BCPolicy, prices: &[f64], volumes: &[f64], lookback: usize) -> f64 {
    let mut cash = 10000.0;
    let mut position = 0.0;

    for i in lookback..prices.len() {
        let state = ExpertDataset::extract_features(prices, volumes, i, lookback);
        let action = policy.predict_action(&state);

        match action {
            Action::Buy => {
                if cash > 0.0 {
                    position = cash / prices[i];
                    cash = 0.0;
                }
            }
            Action::Sell => {
                if position > 0.0 {
                    cash = position * prices[i];
                    position = 0.0;
                }
            }
            Action::Hold => {}
        }
    }

    let final_value = cash + position * prices[prices.len() - 1];
    (final_value - 10000.0) / 10000.0
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_prices() -> (Vec<f64>, Vec<f64>) {
        generate_synthetic_prices(100.0, 200, 0.001, 0.02)
    }

    #[test]
    fn test_expert_dataset_construction() {
        let (prices, volumes) = sample_prices();
        let strategy = ExpertStrategy::TrendFollowing {
            short_window: 5,
            long_window: 20,
        };
        let dataset = ExpertDataset::from_prices(&prices, &volumes, strategy);

        assert!(!dataset.is_empty());
        assert_eq!(dataset.state_dim, 8);
        assert_eq!(dataset.samples[0].state.len(), 8);
        // Should have prices.len() - lookback samples
        assert_eq!(dataset.len(), prices.len() - 20);
    }

    #[test]
    fn test_bc_policy_forward() {
        let policy = BCPolicy::new(8, 32, 16, 3);
        let state = vec![0.01, -0.02, 0.005, -0.001, 0.03, 0.1, 0.5, 0.2];

        let probs = policy.predict_probs(&state);
        assert_eq!(probs.len(), 3);

        // Probabilities should sum to ~1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // All probabilities should be positive
        for p in &probs {
            assert!(*p > 0.0);
        }
    }

    #[test]
    fn test_bc_policy_training() {
        let (prices, volumes) = sample_prices();
        let strategy = ExpertStrategy::TrendFollowing {
            short_window: 5,
            long_window: 20,
        };
        let dataset = ExpertDataset::from_prices(&prices, &volumes, strategy);

        let mut policy = BCPolicy::new(8, 32, 16, 3);
        let losses = policy.train(&dataset, 10, 0.01, 32);

        assert!(!losses.is_empty());
        // Loss should generally decrease (first > last, though not guaranteed for every run)
        // Just check that training produced output
        assert_eq!(losses.len(), 10);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher logit should give higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_covariate_shift_analysis() {
        let expert_states = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.15, 0.25, 0.35],
            vec![0.12, 0.22, 0.32],
        ];
        let policy_states = vec![
            vec![0.5, 0.6, 0.7],
            vec![0.55, 0.65, 0.75],
            vec![0.52, 0.62, 0.72],
        ];

        let kl = CovariateShiftAnalyzer::estimate_kl_divergence(&expert_states, &policy_states);
        assert!(kl > 0.0, "KL divergence should be positive for different distributions");

        let drift = CovariateShiftAnalyzer::feature_drift(&expert_states, &policy_states);
        assert_eq!(drift.len(), 3);
        for d in &drift {
            assert!(*d > 0.0, "Feature drift should be positive");
        }
    }

    #[test]
    fn test_dagger_training() {
        let (prices, volumes) = sample_prices();
        let strategy = ExpertStrategy::TrendFollowing {
            short_window: 5,
            long_window: 20,
        };
        let dataset = ExpertDataset::from_prices(&prices, &volumes, strategy);
        let initial_size = dataset.len();

        let policy = BCPolicy::new(8, 32, 16, 3);
        let config = DAggerConfig {
            num_iterations: 3,
            initial_beta: 0.7,
            beta_decay: 0.5,
            epochs_per_iter: 5,
            learning_rate: 0.01,
            batch_size: 32,
            rollout_steps: 50,
        };

        let mut trainer = DAggerTrainer::new(policy, dataset, config);
        let losses = trainer.run(&prices, &volumes, strategy);

        assert!(!losses.is_empty());
        // Dataset should have grown from aggregation
        assert!(trainer.dataset.len() > initial_size);
        // Should have accuracy for each iteration
        assert_eq!(trainer.iteration_accuracies.len(), 3);
    }

    #[test]
    fn test_mean_reversion_strategy() {
        let (prices, volumes) = sample_prices();
        let strategy = ExpertStrategy::MeanReversion {
            window: 20,
            num_std: 1.5,
        };
        let dataset = ExpertDataset::from_prices(&prices, &volumes, strategy);

        assert!(!dataset.is_empty());
        // Should have all three action types (in most random price series)
        let actions: Vec<Action> = dataset.samples.iter().map(|s| s.action).collect();
        let has_buy = actions.iter().any(|a| *a == Action::Buy);
        let has_sell = actions.iter().any(|a| *a == Action::Sell);
        let has_hold = actions.iter().any(|a| *a == Action::Hold);
        // At least two of three actions should be present
        assert!(
            (has_buy as u8 + has_sell as u8 + has_hold as u8) >= 2,
            "Strategy should produce at least 2 different action types"
        );
    }

    #[test]
    fn test_dataset_merge() {
        let (prices, volumes) = sample_prices();
        let strategy = ExpertStrategy::TrendFollowing {
            short_window: 5,
            long_window: 20,
        };

        let mut dataset1 = ExpertDataset::from_prices(&prices, &volumes, strategy);
        let dataset2 = ExpertDataset::from_prices(&prices, &volumes, strategy);

        let size1 = dataset1.len();
        let size2 = dataset2.len();

        dataset1.merge(&dataset2);
        assert_eq!(dataset1.len(), size1 + size2);
    }
}
