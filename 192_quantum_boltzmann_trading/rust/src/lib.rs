//! # Quantum Boltzmann Trading
//!
//! A Quantum Boltzmann Machine (QBM) simulator for learning joint probability
//! distributions of asset returns. Uses the Suzuki-Trotter decomposition to
//! simulate quantum tunneling effects on classical hardware, enabling better
//! exploration of multimodal financial distributions.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ─── Classical Restricted Boltzmann Machine ────────────────────────────────

/// A classical Restricted Boltzmann Machine serving as the baseline model.
/// Uses binary visible and hidden units with contrastive divergence training.
pub struct ClassicalRBM {
    pub n_visible: usize,
    pub n_hidden: usize,
    pub weights: Array2<f64>,
    pub visible_bias: Array1<f64>,
    pub hidden_bias: Array1<f64>,
    pub learning_rate: f64,
}

impl ClassicalRBM {
    /// Creates a new classical RBM with random small weights.
    pub fn new(n_visible: usize, n_hidden: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((n_visible, n_hidden), |_| {
            rng.gen_range(-0.01..0.01)
        });
        let visible_bias = Array1::zeros(n_visible);
        let hidden_bias = Array1::zeros(n_hidden);

        Self {
            n_visible,
            n_hidden,
            weights,
            visible_bias,
            hidden_bias,
            learning_rate,
        }
    }

    /// Computes the energy of a visible-hidden configuration.
    /// E(v, h) = -sum_i a_i*v_i - sum_j b_j*h_j - sum_{i,j} v_i*W_{ij}*h_j
    pub fn compute_energy(&self, visible: &Array1<f64>, hidden: &Array1<f64>) -> f64 {
        let vis_term = self.visible_bias.dot(visible);
        let hid_term = self.hidden_bias.dot(hidden);
        let interaction: f64 = visible
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                hidden
                    .iter()
                    .enumerate()
                    .map(|(j, &h)| v * self.weights[[i, j]] * h)
                    .sum::<f64>()
            })
            .sum();
        -(vis_term + hid_term + interaction)
    }

    /// Sigmoid activation function.
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Samples hidden units given visible units.
    pub fn sample_hidden(&self, visible: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let activations = self.hidden_bias.clone() + &visible.dot(&self.weights);
        Array1::from_shape_fn(self.n_hidden, |j| {
            let prob = Self::sigmoid(activations[j]);
            if rng.gen::<f64>() < prob { 1.0 } else { 0.0 }
        })
    }

    /// Computes hidden unit probabilities given visible units (no sampling).
    pub fn hidden_probabilities(&self, visible: &Array1<f64>) -> Array1<f64> {
        let activations = self.hidden_bias.clone() + &visible.dot(&self.weights);
        activations.mapv(Self::sigmoid)
    }

    /// Samples visible units given hidden units.
    pub fn sample_visible(&self, hidden: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let activations = self.visible_bias.clone() + &self.weights.dot(hidden);
        Array1::from_shape_fn(self.n_visible, |i| {
            let prob = Self::sigmoid(activations[i]);
            if rng.gen::<f64>() < prob { 1.0 } else { 0.0 }
        })
    }

    /// Performs one Gibbs sampling step: v -> h -> v' -> h'.
    pub fn gibbs_step(
        &self,
        visible: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let h_probs = self.hidden_probabilities(visible);
        let h_sample = self.sample_hidden(visible);
        let v_recon = self.sample_visible(&h_sample);
        let h_recon_probs = self.hidden_probabilities(&v_recon);
        (h_probs, h_sample, v_recon, h_recon_probs)
    }

    /// Trains the RBM using CD-k (contrastive divergence with k Gibbs steps).
    pub fn train_cd_k(&mut self, data: &[Array1<f64>], k: usize, epochs: usize) {
        for _epoch in 0..epochs {
            for sample in data {
                // Positive phase
                let h_pos_probs = self.hidden_probabilities(sample);

                // Negative phase: k steps of Gibbs sampling
                let mut v_neg = sample.clone();
                let mut h_neg_probs = h_pos_probs.clone();
                for _ in 0..k {
                    let h_sample = {
                        let mut rng = rand::thread_rng();
                        Array1::from_shape_fn(self.n_hidden, |j| {
                            if rng.gen::<f64>() < h_neg_probs[j] { 1.0 } else { 0.0 }
                        })
                    };
                    v_neg = self.sample_visible(&h_sample);
                    h_neg_probs = self.hidden_probabilities(&v_neg);
                }

                // Update weights: W += lr * (v_pos * h_pos^T - v_neg * h_neg^T)
                for i in 0..self.n_visible {
                    for j in 0..self.n_hidden {
                        let pos_grad = sample[i] * h_pos_probs[j];
                        let neg_grad = v_neg[i] * h_neg_probs[j];
                        self.weights[[i, j]] += self.learning_rate * (pos_grad - neg_grad);
                    }
                }

                // Update biases
                for i in 0..self.n_visible {
                    self.visible_bias[i] += self.learning_rate * (sample[i] - v_neg[i]);
                }
                for j in 0..self.n_hidden {
                    self.hidden_bias[j] += self.learning_rate * (h_pos_probs[j] - h_neg_probs[j]);
                }
            }
        }
    }

    /// Generates samples from the trained model using Gibbs sampling.
    pub fn generate_samples(&self, n_samples: usize, gibbs_steps: usize) -> Vec<Array1<f64>> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            // Random initialization
            let mut v = Array1::from_shape_fn(self.n_visible, |_| {
                if rng.gen::<f64>() < 0.5 { 1.0 } else { 0.0 }
            });

            // Gibbs sampling
            for _ in 0..gibbs_steps {
                let h = self.sample_hidden(&v);
                v = self.sample_visible(&h);
            }

            samples.push(v);
        }
        samples
    }
}

// ─── Quantum Boltzmann Machine ─────────────────────────────────────────────

/// A Quantum Boltzmann Machine that extends the classical RBM with
/// transverse-field terms, simulated via Suzuki-Trotter decomposition.
pub struct QuantumBoltzmannMachine {
    pub n_visible: usize,
    pub n_hidden: usize,
    pub weights: Array2<f64>,
    pub visible_bias: Array1<f64>,
    pub hidden_bias: Array1<f64>,
    /// Transverse field strength controlling quantum tunneling.
    pub gamma: f64,
    /// Number of Trotter slices for quantum simulation.
    pub n_trotter: usize,
    pub learning_rate: f64,
    pub temperature: f64,
}

impl QuantumBoltzmannMachine {
    /// Creates a new QBM with the specified architecture.
    pub fn new(
        n_visible: usize,
        n_hidden: usize,
        gamma: f64,
        n_trotter: usize,
        learning_rate: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((n_visible, n_hidden), |_| {
            rng.gen_range(-0.01..0.01)
        });
        let visible_bias = Array1::zeros(n_visible);
        let hidden_bias = Array1::zeros(n_hidden);

        Self {
            n_visible,
            n_hidden,
            weights,
            visible_bias,
            hidden_bias,
            gamma,
            n_trotter,
            learning_rate,
            temperature: 1.0,
        }
    }

    /// Computes the classical energy for a visible-hidden configuration.
    pub fn compute_energy(&self, visible: &Array1<f64>, hidden: &Array1<f64>) -> f64 {
        let vis_term = self.visible_bias.dot(visible);
        let hid_term = self.hidden_bias.dot(hidden);
        let interaction: f64 = visible
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                hidden
                    .iter()
                    .enumerate()
                    .map(|(j, &h)| v * self.weights[[i, j]] * h)
                    .sum::<f64>()
            })
            .sum();
        -(vis_term + hid_term + interaction)
    }

    /// Computes the inter-replica coupling strength from the Suzuki-Trotter
    /// decomposition: J_perp = -(T/2) * ln(tanh(Gamma / (M * T)))
    fn inter_replica_coupling(&self) -> f64 {
        let arg = self.gamma / (self.n_trotter as f64 * self.temperature);
        if arg < 1e-10 {
            return 0.0;
        }
        -(self.temperature / 2.0) * (arg.tanh().ln())
    }

    /// Sigmoid activation function.
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Performs quantum-enhanced Gibbs sampling using the Suzuki-Trotter
    /// decomposition. Maintains M replicas coupled along the Trotter dimension.
    pub fn quantum_gibbs_sample(
        &self,
        initial_visible: &Array1<f64>,
        steps: usize,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut rng = rand::thread_rng();
        let n_total = self.n_visible + self.n_hidden;
        let j_perp = self.inter_replica_coupling();

        // Initialize Trotter replicas
        let mut replicas: Vec<Array1<f64>> = (0..self.n_trotter)
            .map(|_| {
                let mut state = Array1::zeros(n_total);
                for i in 0..self.n_visible {
                    state[i] = initial_visible[i];
                }
                for j in 0..self.n_hidden {
                    state[self.n_visible + j] = if rng.gen::<f64>() < 0.5 { 1.0 } else { 0.0 };
                }
                state
            })
            .collect();

        let scaled_beta = 1.0 / (self.n_trotter as f64 * self.temperature);

        for _step in 0..steps {
            for m in 0..self.n_trotter {
                // Within-replica updates (classical Gibbs sampling with scaled temperature)
                // Update hidden units given visible
                for j in 0..self.n_hidden {
                    let visible_part: Array1<f64> =
                        Array1::from_iter(replicas[m].iter().take(self.n_visible).cloned());
                    let activation = self.hidden_bias[j]
                        + visible_part
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| v * self.weights[[i, j]])
                            .sum::<f64>();

                    // Inter-replica coupling for this unit
                    let prev = if m == 0 { self.n_trotter - 1 } else { m - 1 };
                    let next = if m == self.n_trotter - 1 { 0 } else { m + 1 };
                    let spin_current = 2.0 * replicas[m][self.n_visible + j] - 1.0;
                    let spin_prev = 2.0 * replicas[prev][self.n_visible + j] - 1.0;
                    let spin_next = 2.0 * replicas[next][self.n_visible + j] - 1.0;
                    let coupling = j_perp * spin_current * (spin_prev + spin_next);

                    let prob = Self::sigmoid(scaled_beta * activation + coupling * scaled_beta);
                    replicas[m][self.n_visible + j] =
                        if rng.gen::<f64>() < prob { 1.0 } else { 0.0 };
                }

                // Update visible units given hidden
                for i in 0..self.n_visible {
                    let hidden_part: Array1<f64> = Array1::from_iter(
                        replicas[m]
                            .iter()
                            .skip(self.n_visible)
                            .take(self.n_hidden)
                            .cloned(),
                    );
                    let activation = self.visible_bias[i]
                        + hidden_part
                            .iter()
                            .enumerate()
                            .map(|(j, &h)| self.weights[[i, j]] * h)
                            .sum::<f64>();

                    let prev = if m == 0 { self.n_trotter - 1 } else { m - 1 };
                    let next = if m == self.n_trotter - 1 { 0 } else { m + 1 };
                    let spin_current = 2.0 * replicas[m][i] - 1.0;
                    let spin_prev = 2.0 * replicas[prev][i] - 1.0;
                    let spin_next = 2.0 * replicas[next][i] - 1.0;
                    let coupling = j_perp * spin_current * (spin_prev + spin_next);

                    let prob = Self::sigmoid(scaled_beta * activation + coupling * scaled_beta);
                    replicas[m][i] = if rng.gen::<f64>() < prob { 1.0 } else { 0.0 };
                }
            }
        }

        // Average over Trotter replicas to get the final state
        let mut avg_visible = Array1::zeros(self.n_visible);
        let mut avg_hidden = Array1::zeros(self.n_hidden);
        for replica in &replicas {
            for i in 0..self.n_visible {
                avg_visible[i] += replica[i];
            }
            for j in 0..self.n_hidden {
                avg_hidden[j] += replica[self.n_visible + j];
            }
        }
        let m_f64 = self.n_trotter as f64;
        avg_visible /= m_f64;
        avg_hidden /= m_f64;

        // Threshold to binary
        let visible_out = avg_visible.mapv(|x| if x >= 0.5 { 1.0 } else { 0.0 });
        let hidden_out = avg_hidden.mapv(|x| if x >= 0.5 { 1.0 } else { 0.0 });

        (visible_out, hidden_out)
    }

    /// Computes hidden unit probabilities given visible units (classical part).
    pub fn hidden_probabilities(&self, visible: &Array1<f64>) -> Array1<f64> {
        let activations = self.hidden_bias.clone() + &visible.dot(&self.weights);
        activations.mapv(Self::sigmoid)
    }

    /// Trains the QBM using quantum contrastive divergence.
    ///
    /// # Arguments
    /// * `data` - Training data as binary vectors
    /// * `cd_k` - Number of Gibbs sampling steps per CD iteration
    /// * `epochs` - Number of training epochs
    /// * `gamma_decay` - Factor to multiply gamma by each epoch (annealing)
    pub fn train(
        &mut self,
        data: &[Array1<f64>],
        cd_k: usize,
        epochs: usize,
        gamma_decay: f64,
    ) {
        for _epoch in 0..epochs {
            for sample in data {
                // Positive phase: compute expectations under data distribution
                let h_pos_probs = self.hidden_probabilities(sample);

                // Negative phase: quantum-enhanced Gibbs sampling
                let (v_neg, _h_neg) = self.quantum_gibbs_sample(sample, cd_k);
                let h_neg_probs = self.hidden_probabilities(&v_neg);

                // Update weights
                for i in 0..self.n_visible {
                    for j in 0..self.n_hidden {
                        let pos_grad = sample[i] * h_pos_probs[j];
                        let neg_grad = v_neg[i] * h_neg_probs[j];
                        self.weights[[i, j]] +=
                            self.learning_rate * (pos_grad - neg_grad);
                    }
                }

                // Update biases
                for i in 0..self.n_visible {
                    self.visible_bias[i] += self.learning_rate * (sample[i] - v_neg[i]);
                }
                for j in 0..self.n_hidden {
                    self.hidden_bias[j] +=
                        self.learning_rate * (h_pos_probs[j] - h_neg_probs[j]);
                }
            }

            // Anneal the transverse field
            self.gamma *= gamma_decay;
        }
    }

    /// Generates samples from the trained QBM.
    pub fn generate_samples(&self, n_samples: usize, gibbs_steps: usize) -> Vec<Array1<f64>> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let v_init = Array1::from_shape_fn(self.n_visible, |_| {
                if rng.gen::<f64>() < 0.5 { 1.0 } else { 0.0 }
            });
            let (v_out, _) = self.quantum_gibbs_sample(&v_init, gibbs_steps);
            samples.push(v_out);
        }
        samples
    }

    /// Computes the free energy of a visible configuration (classical
    /// approximation). Useful for anomaly detection.
    pub fn free_energy(&self, visible: &Array1<f64>) -> f64 {
        let vis_term = self.visible_bias.dot(visible);
        let mut hidden_term = 0.0;
        for j in 0..self.n_hidden {
            let activation = self.hidden_bias[j]
                + visible
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * self.weights[[i, j]])
                    .sum::<f64>();
            hidden_term += (1.0 + activation.exp()).ln();
        }
        -vis_term - hidden_term
    }
}

// ─── Binary Encoding of Returns ────────────────────────────────────────────

/// Computes quantile-based bin edges for discretizing returns.
pub fn compute_bin_edges(returns: &[f64], n_bins: usize) -> Vec<f64> {
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(f64::NEG_INFINITY);
    for i in 1..n_bins {
        let idx = (i as f64 / n_bins as f64 * sorted.len() as f64) as usize;
        let idx = idx.min(sorted.len() - 1);
        edges.push(sorted[idx]);
    }
    edges.push(f64::INFINITY);
    edges
}

/// Discretizes a single return value into a bin index.
pub fn discretize_return(value: f64, edges: &[f64]) -> usize {
    for i in 1..edges.len() {
        if value < edges[i] {
            return i - 1;
        }
    }
    edges.len() - 2
}

/// Encodes a bin index using thermometer encoding.
/// Bin index k out of n_bins produces a vector [1,1,...,1,0,...,0]
/// with k ones followed by (n_bins-1-k) zeros.
pub fn thermometer_encode(bin_index: usize, n_bins: usize) -> Vec<f64> {
    (0..n_bins - 1)
        .map(|i| if i < bin_index + 1 { 1.0 } else { 0.0 })
        .collect()
}

/// Decodes a thermometer-encoded vector back to a bin index.
pub fn thermometer_decode(encoded: &[f64]) -> usize {
    let ones = encoded.iter().filter(|&&x| x > 0.5).count();
    if ones == 0 { 0 } else { ones - 1 }
}

/// Encodes multiple assets' returns into a single binary feature vector.
/// Each asset's return is discretized and thermometer-encoded, then concatenated.
pub fn encode_returns(
    returns: &[f64],
    bin_edges: &[Vec<f64>],
    n_bins: usize,
) -> Array1<f64> {
    let mut encoded = Vec::new();
    for (asset_idx, &ret) in returns.iter().enumerate() {
        let bin = discretize_return(ret, &bin_edges[asset_idx]);
        let therm = thermometer_encode(bin, n_bins);
        encoded.extend(therm);
    }
    Array1::from_vec(encoded)
}

/// Decodes a binary feature vector back to bin indices for each asset.
pub fn decode_to_bins(encoded: &Array1<f64>, n_assets: usize, n_bins: usize) -> Vec<usize> {
    let bits_per_asset = n_bins - 1;
    (0..n_assets)
        .map(|a| {
            let start = a * bits_per_asset;
            let end = start + bits_per_asset;
            let slice: Vec<f64> = encoded.iter().skip(start).take(end - start).cloned().collect();
            thermometer_decode(&slice)
        })
        .collect()
}

// ─── Bybit API Client ──────────────────────────────────────────────────────

/// Kline (candlestick) data from Bybit.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Client for fetching market data from the Bybit V5 API.
pub struct BybitClient {
    base_url: String,
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

    /// Fetches kline data for a given symbol and interval.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair, e.g., "BTCUSDT"
    /// * `interval` - Kline interval: "1", "5", "15", "60", "240", "D", "W"
    /// * `limit` - Number of candles (max 200)
    pub fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let client = reqwest::blocking::Client::new();
        let resp: BybitResponse = client.get(&url).send()?.json()?;

        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
        }

        let mut klines: Vec<Kline> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
                Some(Kline {
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
        klines.reverse();
        Ok(klines)
    }

    /// Computes log returns from kline close prices.
    pub fn compute_log_returns(klines: &[Kline]) -> Vec<f64> {
        klines
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect()
    }
}

// ─── Scenario Generation Utilities ─────────────────────────────────────────

/// Statistics for comparing real vs generated distributions.
#[derive(Debug, Clone)]
pub struct DistributionStats {
    pub mean: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub min: f64,
    pub max: f64,
}

impl DistributionStats {
    /// Computes summary statistics for a slice of values.
    pub fn from_data(data: &[f64]) -> Self {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let skewness = if std_dev > 1e-10 {
            data.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n
        } else {
            0.0
        };
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self { mean, std_dev, skewness, min, max }
    }
}

/// Computes the bin-level distribution (histogram) from encoded samples.
pub fn compute_bin_distribution(
    samples: &[Array1<f64>],
    n_assets: usize,
    n_bins: usize,
) -> Vec<Vec<f64>> {
    let n_samples = samples.len() as f64;
    let mut distributions = vec![vec![0.0; n_bins]; n_assets];

    for sample in samples {
        let bins = decode_to_bins(sample, n_assets, n_bins);
        for (asset, bin) in bins.iter().enumerate() {
            if *bin < n_bins {
                distributions[asset][*bin] += 1.0;
            }
        }
    }

    // Normalize to probabilities
    for dist in &mut distributions {
        for count in dist.iter_mut() {
            *count /= n_samples;
        }
    }
    distributions
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_rbm_energy() {
        let rbm = ClassicalRBM::new(4, 3, 0.1);
        let v = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let h = Array1::from_vec(vec![1.0, 1.0, 0.0]);
        let energy = rbm.compute_energy(&v, &h);
        // Energy should be finite
        assert!(energy.is_finite());
    }

    #[test]
    fn test_classical_rbm_sample_hidden() {
        let rbm = ClassicalRBM::new(4, 3, 0.1);
        let v = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let h = rbm.sample_hidden(&v);
        assert_eq!(h.len(), 3);
        // All values should be 0 or 1
        for &val in h.iter() {
            assert!(val == 0.0 || val == 1.0);
        }
    }

    #[test]
    fn test_classical_rbm_training() {
        let mut rbm = ClassicalRBM::new(4, 2, 0.1);
        let data = vec![
            Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]),
            Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]),
            Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]),
        ];
        rbm.train_cd_k(&data, 1, 10);
        // Weights should have changed from initial near-zero values
        let max_weight = rbm.weights.iter().cloned().fold(0.0_f64, f64::max);
        let min_weight = rbm.weights.iter().cloned().fold(0.0_f64, f64::min);
        assert!(max_weight > 0.01 || min_weight < -0.01);
    }

    #[test]
    fn test_qbm_energy() {
        let qbm = QuantumBoltzmannMachine::new(4, 3, 1.0, 4, 0.01);
        let v = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let h = Array1::from_vec(vec![1.0, 1.0, 0.0]);
        let energy = qbm.compute_energy(&v, &h);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_qbm_quantum_gibbs_sample() {
        let qbm = QuantumBoltzmannMachine::new(4, 3, 1.0, 4, 0.01);
        let v = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let (v_out, h_out) = qbm.quantum_gibbs_sample(&v, 5);
        assert_eq!(v_out.len(), 4);
        assert_eq!(h_out.len(), 3);
        for &val in v_out.iter().chain(h_out.iter()) {
            assert!(val == 0.0 || val == 1.0);
        }
    }

    #[test]
    fn test_qbm_inter_replica_coupling() {
        let qbm = QuantumBoltzmannMachine::new(4, 3, 1.0, 4, 0.01);
        let j = qbm.inter_replica_coupling();
        assert!(j.is_finite());
        // With gamma > 0 and T > 0, coupling should be positive
        assert!(j > 0.0);
    }

    #[test]
    fn test_qbm_free_energy() {
        let qbm = QuantumBoltzmannMachine::new(4, 3, 1.0, 4, 0.01);
        let v1 = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let v2 = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let fe1 = qbm.free_energy(&v1);
        let fe2 = qbm.free_energy(&v2);
        assert!(fe1.is_finite());
        assert!(fe2.is_finite());
    }

    #[test]
    fn test_qbm_training() {
        let mut qbm = QuantumBoltzmannMachine::new(4, 2, 1.0, 4, 0.1);
        let data = vec![
            Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]),
            Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]),
            Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]),
        ];
        let initial_gamma = qbm.gamma;
        qbm.train(&data, 1, 5, 0.95);
        // Gamma should have decayed
        assert!(qbm.gamma < initial_gamma);
        // Weights should have changed
        let max_weight = qbm.weights.iter().cloned().fold(0.0_f64, f64::max);
        let min_weight = qbm.weights.iter().cloned().fold(0.0_f64, f64::min);
        assert!(max_weight > 0.01 || min_weight < -0.01);
    }

    #[test]
    fn test_qbm_generate_samples() {
        let qbm = QuantumBoltzmannMachine::new(4, 3, 0.5, 4, 0.01);
        let samples = qbm.generate_samples(10, 3);
        assert_eq!(samples.len(), 10);
        for s in &samples {
            assert_eq!(s.len(), 4);
            for &val in s.iter() {
                assert!(val == 0.0 || val == 1.0);
            }
        }
    }

    #[test]
    fn test_thermometer_encoding() {
        // Bin 0 out of 4 bins -> [1, 0, 0]
        assert_eq!(thermometer_encode(0, 4), vec![1.0, 0.0, 0.0]);
        // Bin 1 out of 4 bins -> [1, 1, 0]
        assert_eq!(thermometer_encode(1, 4), vec![1.0, 1.0, 0.0]);
        // Bin 2 out of 4 bins -> [1, 1, 1]
        assert_eq!(thermometer_encode(2, 4), vec![1.0, 1.0, 1.0]);
        // Bin 3 out of 4 bins -> [1, 1, 1] (saturated)
        assert_eq!(thermometer_encode(3, 4), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_thermometer_decode() {
        assert_eq!(thermometer_decode(&[1.0, 0.0, 0.0]), 0);
        assert_eq!(thermometer_decode(&[1.0, 1.0, 0.0]), 1);
        assert_eq!(thermometer_decode(&[1.0, 1.0, 1.0]), 2);
        assert_eq!(thermometer_decode(&[0.0, 0.0, 0.0]), 0);
    }

    #[test]
    fn test_compute_bin_edges() {
        let returns = vec![-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05];
        let edges = compute_bin_edges(&returns, 4);
        assert_eq!(edges.len(), 5); // n_bins + 1
        assert_eq!(edges[0], f64::NEG_INFINITY);
        assert_eq!(edges[4], f64::INFINITY);
        // Middle edges should be sorted
        assert!(edges[1] < edges[2]);
        assert!(edges[2] < edges[3]);
    }

    #[test]
    fn test_discretize_return() {
        let edges = vec![f64::NEG_INFINITY, -0.01, 0.0, 0.01, f64::INFINITY];
        assert_eq!(discretize_return(-0.05, &edges), 0);
        assert_eq!(discretize_return(-0.005, &edges), 1);
        assert_eq!(discretize_return(0.005, &edges), 2);
        assert_eq!(discretize_return(0.05, &edges), 3);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let n_bins = 4;
        let returns = vec![0.01, -0.02];
        let edges = vec![
            vec![f64::NEG_INFINITY, -0.01, 0.0, 0.01, f64::INFINITY],
            vec![f64::NEG_INFINITY, -0.01, 0.0, 0.01, f64::INFINITY],
        ];

        let encoded = encode_returns(&returns, &edges, n_bins);
        assert_eq!(encoded.len(), 2 * (n_bins - 1)); // 2 assets * 3 bits each

        let bins = decode_to_bins(&encoded, 2, n_bins);
        assert_eq!(bins.len(), 2);
        // First return 0.01 should be in bin 2 or 3
        assert!(bins[0] >= 2);
        // Second return -0.02 should be in bin 0
        assert_eq!(bins[1], 0);
    }

    #[test]
    fn test_distribution_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = DistributionStats::from_data(&data);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_compute_bin_distribution() {
        let n_bins = 3;
        let samples = vec![
            Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]), // asset0: bin0, asset1: bin0
            Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]), // asset0: bin2, asset1: bin2
        ];
        let dist = compute_bin_distribution(&samples, 2, n_bins);
        assert_eq!(dist.len(), 2);
        assert_eq!(dist[0].len(), n_bins);
        // Sum of each distribution should be ~1.0
        let sum0: f64 = dist[0].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-10);
    }
}
