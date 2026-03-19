//! Factor VQ-VAE: Vector Quantized Variational Autoencoder for discrete factor models in finance.
//!
//! This crate implements a VQ-VAE that learns discrete market regimes from financial data.
//! Each codebook entry represents a prototypical market state (regime), and the model
//! learns to assign each observation to the nearest regime.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

/// ReLU activation applied element-wise.
fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

/// Euclidean distance squared between two vectors.
fn distance_sq(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    (a - b).mapv(|v| v * v).sum()
}

/// Mean squared error between two vectors.
fn mse(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let diff = a - b;
    diff.mapv(|v| v * v).mean().unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Dense (linear) layer
// ---------------------------------------------------------------------------

/// A simple fully-connected layer: y = x * W + b
#[derive(Clone, Debug)]
pub struct Linear {
    pub weights: Array2<f64>, // (in_dim, out_dim)
    pub bias: Array1<f64>,    // (out_dim,)
}

impl Linear {
    /// Xavier-uniform initialisation.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (6.0 / (in_dim + out_dim) as f64).sqrt();
        let weights = Array2::from_shape_fn((in_dim, out_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(out_dim);
        Self { weights, bias }
    }

    /// Forward pass for a single sample.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.dot(&self.weights) + &self.bias
    }

    /// Simple SGD update: W -= lr * dW, b -= lr * db.
    pub fn update(&mut self, d_weights: &Array2<f64>, d_bias: &Array1<f64>, lr: f64) {
        self.weights.scaled_add(-lr, d_weights);
        self.bias.scaled_add(-lr, d_bias);
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Two-layer encoder: input -> hidden (ReLU) -> embedding.
#[derive(Clone, Debug)]
pub struct Encoder {
    pub layer1: Linear,
    pub layer2: Linear,
}

impl Encoder {
    pub fn new(input_dim: usize, hidden_dim: usize, embedding_dim: usize) -> Self {
        Self {
            layer1: Linear::new(input_dim, hidden_dim),
            layer2: Linear::new(hidden_dim, embedding_dim),
        }
    }

    /// Forward pass returning (z_e, hidden) for use in backprop.
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let hidden = relu(&self.layer1.forward(x));
        let z_e = self.layer2.forward(&hidden);
        (z_e, hidden)
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Two-layer decoder: embedding -> hidden (ReLU) -> reconstruction.
#[derive(Clone, Debug)]
pub struct Decoder {
    pub layer1: Linear,
    pub layer2: Linear,
}

impl Decoder {
    pub fn new(embedding_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            layer1: Linear::new(embedding_dim, hidden_dim),
            layer2: Linear::new(hidden_dim, output_dim),
        }
    }

    /// Forward pass returning (x_hat, hidden).
    pub fn forward(&self, z_q: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let hidden = relu(&self.layer1.forward(z_q));
        let x_hat = self.layer2.forward(&hidden);
        (x_hat, hidden)
    }
}

// ---------------------------------------------------------------------------
// Codebook with EMA updates
// ---------------------------------------------------------------------------

/// Vector quantization codebook with Exponential Moving Average updates.
#[derive(Clone, Debug)]
pub struct Codebook {
    pub num_entries: usize,
    pub embedding_dim: usize,
    pub entries: Vec<Array1<f64>>,
    pub ema_count: Vec<f64>,
    pub ema_sum: Vec<Array1<f64>>,
    pub gamma: f64,
    pub epsilon: f64,
}

impl Codebook {
    /// Create a new codebook with `K` entries of dimension `D`, initialised randomly.
    pub fn new(num_entries: usize, embedding_dim: usize, gamma: f64) -> Self {
        let mut rng = rand::thread_rng();
        let entries: Vec<Array1<f64>> = (0..num_entries)
            .map(|_| Array1::from_shape_fn(embedding_dim, |_| rng.gen_range(-1.0..1.0)))
            .collect();
        let ema_count = vec![1.0; num_entries];
        let ema_sum = entries.clone();
        Self {
            num_entries,
            embedding_dim,
            entries,
            ema_count,
            ema_sum,
            gamma,
            epsilon: 1e-5,
        }
    }

    /// Initialise codebook from a batch of encoder outputs (k-means++ style seeding).
    pub fn init_from_data(&mut self, data: &[Array1<f64>]) {
        if data.is_empty() {
            return;
        }
        let mut rng = rand::thread_rng();
        let mut centres: Vec<Array1<f64>> = Vec::with_capacity(self.num_entries);

        // Pick first centre uniformly at random.
        centres.push(data[rng.gen_range(0..data.len())].clone());

        // Pick remaining centres proportional to squared distance.
        for _ in 1..self.num_entries {
            let dists: Vec<f64> = data
                .iter()
                .map(|x| {
                    centres
                        .iter()
                        .map(|c| distance_sq(x, c))
                        .fold(f64::MAX, f64::min)
                })
                .collect();
            let total: f64 = dists.iter().sum();
            if total < 1e-12 {
                centres.push(data[rng.gen_range(0..data.len())].clone());
                continue;
            }
            let threshold = rng.gen_range(0.0..total);
            let mut cumsum = 0.0;
            let mut chosen = data.len() - 1;
            for (i, &d) in dists.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    chosen = i;
                    break;
                }
            }
            centres.push(data[chosen].clone());
        }

        self.entries = centres.clone();
        self.ema_sum = centres;
        self.ema_count = vec![1.0; self.num_entries];
    }

    /// Find the nearest codebook entry index for a given vector.
    pub fn quantize_index(&self, z_e: &Array1<f64>) -> usize {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, distance_sq(z_e, e)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }

    /// Quantize: returns (z_q, index).
    /// z_q is set to the codebook entry (forward) but gradients flow through z_e (straight-through).
    pub fn quantize(&self, z_e: &Array1<f64>) -> (Array1<f64>, usize) {
        let idx = self.quantize_index(z_e);
        (self.entries[idx].clone(), idx)
    }

    /// EMA update given a batch of (encoder_output, assigned_index) pairs.
    pub fn ema_update(&mut self, assignments: &[(Array1<f64>, usize)]) {
        // Count and sum per entry.
        let mut counts = vec![0.0f64; self.num_entries];
        let mut sums: Vec<Array1<f64>> = (0..self.num_entries)
            .map(|_| Array1::zeros(self.embedding_dim))
            .collect();

        for (z_e, idx) in assignments {
            counts[*idx] += 1.0;
            sums[*idx] = &sums[*idx] + z_e;
        }

        for k in 0..self.num_entries {
            self.ema_count[k] = self.gamma * self.ema_count[k] + (1.0 - self.gamma) * counts[k];
            self.ema_sum[k] =
                &self.ema_sum[k] * self.gamma + &sums[k] * (1.0 - self.gamma);

            // Laplace smoothing to avoid division by zero.
            let n = self.ema_count[k] + self.epsilon;
            self.entries[k] = &self.ema_sum[k] / n;
        }
    }
}

// ---------------------------------------------------------------------------
// VQ-VAE model
// ---------------------------------------------------------------------------

/// Hyper-parameters for VQ-VAE training.
#[derive(Clone, Debug)]
pub struct VqVaeConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub embedding_dim: usize,
    pub num_codes: usize,
    pub beta: f64,       // commitment loss weight
    pub gamma: f64,      // EMA decay
    pub learning_rate: f64,
    pub num_epochs: usize,
}

impl Default for VqVaeConfig {
    fn default() -> Self {
        Self {
            input_dim: 5,
            hidden_dim: 32,
            embedding_dim: 8,
            num_codes: 8,
            beta: 0.25,
            gamma: 0.99,
            learning_rate: 1e-3,
            num_epochs: 200,
        }
    }
}

/// The full VQ-VAE model.
#[derive(Clone, Debug)]
pub struct VqVae {
    pub encoder: Encoder,
    pub decoder: Decoder,
    pub codebook: Codebook,
    pub config: VqVaeConfig,
}

/// Per-sample output of a forward pass.
#[derive(Debug)]
pub struct VqVaeOutput {
    pub x_hat: Array1<f64>,
    pub z_e: Array1<f64>,
    pub z_q: Array1<f64>,
    pub code_index: usize,
    pub recon_loss: f64,
    pub commit_loss: f64,
}

impl VqVae {
    pub fn new(config: VqVaeConfig) -> Self {
        let encoder = Encoder::new(config.input_dim, config.hidden_dim, config.embedding_dim);
        let decoder = Decoder::new(config.embedding_dim, config.hidden_dim, config.input_dim);
        let codebook = Codebook::new(config.num_codes, config.embedding_dim, config.gamma);
        Self {
            encoder,
            decoder,
            codebook,
            config,
        }
    }

    /// Forward pass for a single sample.
    pub fn forward(&self, x: &Array1<f64>) -> VqVaeOutput {
        let (z_e, _enc_hidden) = self.encoder.forward(x);
        let (z_q, code_index) = self.codebook.quantize(&z_e);

        // Straight-through: decoder sees z_q, but we track z_e for commitment loss.
        let (x_hat, _dec_hidden) = self.decoder.forward(&z_q);

        let recon_loss = mse(x, &x_hat);
        let commit_loss = mse(&z_e, &z_q);

        VqVaeOutput {
            x_hat,
            z_e,
            z_q,
            code_index,
            recon_loss,
            commit_loss,
        }
    }

    /// Train the VQ-VAE on a dataset (Vec of feature vectors).
    /// Returns (epoch_losses, final_assignments).
    pub fn train(&mut self, data: &[Array1<f64>]) -> (Vec<f64>, Vec<usize>) {
        // Initialise codebook from data.
        let z_es: Vec<Array1<f64>> = data
            .iter()
            .map(|x| self.encoder.forward(x).0)
            .collect();
        self.codebook.init_from_data(&z_es);

        let mut epoch_losses = Vec::with_capacity(self.config.num_epochs);

        for _epoch in 0..self.config.num_epochs {
            let mut total_loss = 0.0;
            let mut assignments: Vec<(Array1<f64>, usize)> = Vec::with_capacity(data.len());

            for x in data {
                // ---- Forward ----
                let (z_e, enc_hidden) = self.encoder.forward(x);
                let (z_q, code_idx) = self.codebook.quantize(&z_e);
                let (x_hat, dec_hidden) = self.decoder.forward(&z_q);

                let recon_loss = mse(x, &x_hat);
                let commit_loss = mse(&z_e, &z_q);
                let loss = recon_loss + self.config.beta * commit_loss;
                total_loss += loss;

                assignments.push((z_e.clone(), code_idx));

                // ---- Backward (manual gradient computation) ----
                let n = x.len() as f64;
                let lr = self.config.learning_rate;

                // d(recon_loss)/d(x_hat) = 2*(x_hat - x) / n
                let d_x_hat = (&x_hat - x) * (2.0 / n);

                // Decoder layer2: x_hat = dec_hidden * W4 + b4
                let d_dec_w2 = outer(&dec_hidden, &d_x_hat);
                let d_dec_b2 = d_x_hat.clone();
                let d_dec_hidden_raw = d_x_hat.dot(&self.decoder.layer2.weights.t());
                // ReLU derivative
                let d_dec_hidden =
                    &d_dec_hidden_raw * &dec_hidden.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

                // Decoder layer1: dec_hidden_pre = z_q * W3 + b3
                let d_dec_w1 = outer(&z_q, &d_dec_hidden);
                let d_dec_b1 = d_dec_hidden.clone();

                // Straight-through: gradient w.r.t. z_q is passed to z_e.
                let d_z_e_recon = d_dec_hidden.dot(&self.decoder.layer1.weights.t());

                // Commitment loss gradient w.r.t. z_e: 2*beta*(z_e - z_q) / D
                let d_embed = z_e.len() as f64;
                let d_z_e_commit =
                    (&z_e - &z_q) * (2.0 * self.config.beta / d_embed);

                let d_z_e = &d_z_e_recon + &d_z_e_commit;

                // Encoder layer2: z_e = enc_hidden * W2 + b2
                let d_enc_w2 = outer(&enc_hidden, &d_z_e);
                let d_enc_b2 = d_z_e.clone();
                let d_enc_hidden_raw = d_z_e.dot(&self.encoder.layer2.weights.t());
                let d_enc_hidden =
                    &d_enc_hidden_raw * &enc_hidden.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

                // Encoder layer1: enc_hidden_pre = x * W1 + b1
                let d_enc_w1 = outer(x, &d_enc_hidden);
                let d_enc_b1 = d_enc_hidden.clone();

                // ---- Parameter updates (SGD) ----
                self.decoder.layer2.update(&d_dec_w2, &d_dec_b2, lr);
                self.decoder.layer1.update(&d_dec_w1, &d_dec_b1, lr);
                self.encoder.layer2.update(&d_enc_w2, &d_enc_b2, lr);
                self.encoder.layer1.update(&d_enc_w1, &d_enc_b1, lr);
            }

            // EMA codebook update (per epoch).
            self.codebook.ema_update(&assignments);

            epoch_losses.push(total_loss / data.len() as f64);
        }

        // Final assignments.
        let final_assignments: Vec<usize> = data
            .iter()
            .map(|x| {
                let (z_e, _) = self.encoder.forward(x);
                self.codebook.quantize_index(&z_e)
            })
            .collect();

        (epoch_losses, final_assignments)
    }

    /// Decode a codebook entry (for regime-conditional generation).
    pub fn decode_codebook_entry(&self, k: usize) -> Array1<f64> {
        let z_q = &self.codebook.entries[k];
        self.decoder.forward(z_q).0
    }

    /// Generate regime-conditional synthetic sample by perturbing a codebook entry.
    pub fn generate_sample(&self, regime: usize, noise_scale: f64) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let z_q = &self.codebook.entries[regime];
        let noise = Array1::from_shape_fn(self.config.embedding_dim, |_| {
            rng.gen_range(-noise_scale..noise_scale)
        });
        let z_perturbed = z_q + &noise;
        self.decoder.forward(&z_perturbed).0
    }
}

/// Outer product of two vectors.
fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Market regime analysis
// ---------------------------------------------------------------------------

/// Statistics for a single market regime.
#[derive(Debug, Clone)]
pub struct RegimeStats {
    pub regime_id: usize,
    pub count: usize,
    pub mean_return: f64,
    pub mean_volatility: f64,
    pub mean_volume_ratio: f64,
    pub mean_range: f64,
    pub mean_oc_spread: f64,
}

/// Analyse market regimes from codebook assignments and original features.
/// Features are assumed to be [log_return, volatility, volume_ratio, range, oc_spread].
pub fn analyse_regimes(
    data: &[Array1<f64>],
    assignments: &[usize],
    num_regimes: usize,
) -> Vec<RegimeStats> {
    let mut stats = Vec::with_capacity(num_regimes);

    for k in 0..num_regimes {
        let members: Vec<&Array1<f64>> = data
            .iter()
            .zip(assignments.iter())
            .filter(|(_, &a)| a == k)
            .map(|(x, _)| x)
            .collect();

        let count = members.len();
        if count == 0 {
            stats.push(RegimeStats {
                regime_id: k,
                count: 0,
                mean_return: 0.0,
                mean_volatility: 0.0,
                mean_volume_ratio: 0.0,
                mean_range: 0.0,
                mean_oc_spread: 0.0,
            });
            continue;
        }

        let n = count as f64;
        let mean_return = members.iter().map(|x| x[0]).sum::<f64>() / n;
        let mean_volatility = members.iter().map(|x| x[1]).sum::<f64>() / n;
        let mean_volume_ratio = members.iter().map(|x| x[2]).sum::<f64>() / n;
        let mean_range = members.iter().map(|x| x[3]).sum::<f64>() / n;
        let mean_oc_spread = members.iter().map(|x| x[4]).sum::<f64>() / n;

        stats.push(RegimeStats {
            regime_id: k,
            count,
            mean_return,
            mean_volatility,
            mean_volume_ratio,
            mean_range,
            mean_oc_spread,
        });
    }

    stats
}

// ---------------------------------------------------------------------------
// Bybit API data fetching
// ---------------------------------------------------------------------------

/// Raw kline (candlestick) data from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub open_time: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API v5 response structures.
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

/// Fetch daily kline data from Bybit for a given symbol.
/// Returns klines in chronological order (oldest first).
pub fn fetch_bybit_klines(symbol: &str, limit: usize) -> anyhow::Result<Vec<Kline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval=D&limit={}",
        symbol, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
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
                open_time: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order.
    klines.reverse();
    Ok(klines)
}

/// Compute feature vectors from raw kline data.
/// Features: [log_return, realised_vol, volume_ratio, normalised_range, oc_spread]
/// Uses a rolling window for volatility and volume average.
pub fn compute_features(klines: &[Kline], window: usize) -> Vec<Array1<f64>> {
    if klines.len() < window + 1 {
        return Vec::new();
    }

    // Log returns.
    let returns: Vec<f64> = klines
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();

    let mut features = Vec::new();

    for i in window..returns.len() {
        let log_return = returns[i];

        // Realised volatility over the window.
        let window_returns = &returns[(i - window)..i];
        let mean_r: f64 = window_returns.iter().sum::<f64>() / window as f64;
        let var: f64 =
            window_returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / window as f64;
        let volatility = var.sqrt();

        // Volume ratio (current vs window average).
        let kline_idx = i + 1; // offset by 1 because returns start from index 1
        let vol_avg: f64 = klines[(kline_idx - window)..kline_idx]
            .iter()
            .map(|k| k.volume)
            .sum::<f64>()
            / window as f64;
        let volume_ratio = if vol_avg > 0.0 {
            klines[kline_idx].volume / vol_avg
        } else {
            1.0
        };

        // Normalised range.
        let k = &klines[kline_idx];
        let range = if k.close > 0.0 {
            (k.high - k.low) / k.close
        } else {
            0.0
        };

        // Open-close spread.
        let oc_spread = if k.open > 0.0 {
            (k.close - k.open) / k.open
        } else {
            0.0
        };

        features.push(Array1::from_vec(vec![
            log_return,
            volatility,
            volume_ratio,
            range,
            oc_spread,
        ]));
    }

    features
}

/// Normalise features to zero mean and unit variance. Returns (normalised, means, stds).
pub fn normalise_features(
    data: &[Array1<f64>],
) -> (Vec<Array1<f64>>, Array1<f64>, Array1<f64>) {
    if data.is_empty() {
        return (Vec::new(), Array1::zeros(0), Array1::ones(0));
    }
    let dim = data[0].len();
    let n = data.len() as f64;

    let mut means = Array1::zeros(dim);
    for x in data {
        means = &means + x;
    }
    means /= n;

    let mut vars = Array1::zeros(dim);
    for x in data {
        let diff = x - &means;
        vars = &vars + &diff.mapv(|v| v * v);
    }
    vars /= n;
    let stds = vars.mapv(|v| v.sqrt().max(1e-8));

    let normalised: Vec<Array1<f64>> = data.iter().map(|x| (x - &means) / &stds).collect();

    (normalised, means, stds)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_linear_forward() {
        let layer = Linear::new(3, 2);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = layer.forward(&x);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_encoder_decoder_dimensions() {
        let encoder = Encoder::new(5, 16, 8);
        let decoder = Decoder::new(8, 16, 5);
        let x = Array1::from_vec(vec![0.1, -0.2, 0.3, 0.05, -0.1]);
        let (z_e, _) = encoder.forward(&x);
        assert_eq!(z_e.len(), 8);
        let (x_hat, _) = decoder.forward(&z_e);
        assert_eq!(x_hat.len(), 5);
    }

    #[test]
    fn test_codebook_quantize() {
        let mut codebook = Codebook::new(4, 3, 0.99);
        // Set known entries.
        codebook.entries[0] = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        codebook.entries[1] = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        codebook.entries[2] = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        codebook.entries[3] = Array1::from_vec(vec![-1.0, 0.0, 0.0]);

        let z = Array1::from_vec(vec![0.9, 0.1, 0.0]);
        let (_, idx) = codebook.quantize(&z);
        assert_eq!(idx, 0);

        let z2 = Array1::from_vec(vec![-0.8, 0.1, 0.2]);
        let (_, idx2) = codebook.quantize(&z2);
        assert_eq!(idx2, 3);
    }

    #[test]
    fn test_codebook_ema_update() {
        let mut codebook = Codebook::new(2, 2, 0.9);
        codebook.entries[0] = Array1::from_vec(vec![0.0, 0.0]);
        codebook.entries[1] = Array1::from_vec(vec![1.0, 1.0]);
        codebook.ema_sum = codebook.entries.clone();
        codebook.ema_count = vec![1.0, 1.0];

        let assignments = vec![
            (Array1::from_vec(vec![0.1, 0.1]), 0usize),
            (Array1::from_vec(vec![0.2, 0.2]), 0),
            (Array1::from_vec(vec![1.1, 1.2]), 1),
        ];
        codebook.ema_update(&assignments);

        // Entry 0 should have moved toward ~(0.15, 0.15).
        assert!(codebook.entries[0][0] > 0.0);
        assert!(codebook.entries[0][1] > 0.0);
    }

    #[test]
    fn test_vqvae_forward() {
        let config = VqVaeConfig {
            input_dim: 5,
            hidden_dim: 16,
            embedding_dim: 4,
            num_codes: 4,
            beta: 0.25,
            gamma: 0.99,
            learning_rate: 1e-3,
            num_epochs: 10,
        };
        let model = VqVae::new(config);
        let x = Array1::from_vec(vec![0.1, 0.02, 1.5, 0.03, -0.01]);
        let output = model.forward(&x);

        assert_eq!(output.x_hat.len(), 5);
        assert!(output.code_index < 4);
        assert!(output.recon_loss >= 0.0);
        assert!(output.commit_loss >= 0.0);
    }

    #[test]
    fn test_vqvae_training_reduces_loss() {
        let config = VqVaeConfig {
            input_dim: 3,
            hidden_dim: 16,
            embedding_dim: 4,
            num_codes: 3,
            beta: 0.25,
            gamma: 0.99,
            learning_rate: 1e-3,
            num_epochs: 100,
        };
        let mut model = VqVae::new(config);

        // Create simple synthetic data with 3 clusters.
        let mut data = Vec::new();
        for _ in 0..20 {
            data.push(Array1::from_vec(vec![1.0, 0.0, 0.0]));
            data.push(Array1::from_vec(vec![0.0, 1.0, 0.0]));
            data.push(Array1::from_vec(vec![0.0, 0.0, 1.0]));
        }

        let (losses, assignments) = model.train(&data);

        // Loss should decrease.
        let first_losses: f64 = losses[..5].iter().sum::<f64>() / 5.0;
        let last_losses: f64 = losses[(losses.len() - 5)..].iter().sum::<f64>() / 5.0;
        assert!(
            last_losses < first_losses,
            "Loss should decrease: first={}, last={}",
            first_losses,
            last_losses
        );

        // All codes should be used.
        assert_eq!(assignments.len(), data.len());
    }

    #[test]
    fn test_regime_analysis() {
        let data = vec![
            Array1::from_vec(vec![0.01, 0.005, 1.0, 0.02, 0.005]),
            Array1::from_vec(vec![0.02, 0.006, 1.1, 0.03, 0.01]),
            Array1::from_vec(vec![-0.05, 0.03, 2.0, 0.08, -0.04]),
        ];
        let assignments = vec![0, 0, 1];
        let stats = analyse_regimes(&data, &assignments, 2);

        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].count, 2);
        assert_eq!(stats[1].count, 1);
        assert!((stats[0].mean_return - 0.015).abs() < 1e-9);
        assert!((stats[1].mean_return - (-0.05)).abs() < 1e-9);
    }

    #[test]
    fn test_generate_sample() {
        let config = VqVaeConfig {
            input_dim: 5,
            hidden_dim: 16,
            embedding_dim: 4,
            num_codes: 4,
            beta: 0.25,
            gamma: 0.99,
            learning_rate: 1e-3,
            num_epochs: 1,
        };
        let model = VqVae::new(config);
        let sample = model.generate_sample(0, 0.1);
        assert_eq!(sample.len(), 5);
    }

    #[test]
    fn test_compute_features() {
        // Create minimal kline data.
        let klines: Vec<Kline> = (0..25)
            .map(|i| {
                let price = 100.0 + i as f64;
                Kline {
                    open_time: 1000 + i * 86400,
                    open: price,
                    high: price + 2.0,
                    low: price - 1.0,
                    close: price + 0.5,
                    volume: 1000.0 + (i as f64) * 10.0,
                }
            })
            .collect();

        let features = compute_features(&klines, 10);
        assert!(!features.is_empty());
        assert_eq!(features[0].len(), 5);
    }

    #[test]
    fn test_normalise_features() {
        let data = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0, 4.0]),
            Array1::from_vec(vec![5.0, 6.0]),
        ];
        let (normed, means, stds) = normalise_features(&data);

        assert_eq!(normed.len(), 3);
        // Mean of normalised data should be ~0.
        let mean_normed: f64 = normed.iter().map(|x| x[0]).sum::<f64>() / 3.0;
        assert!(mean_normed.abs() < 1e-10);
        assert!((means[0] - 3.0).abs() < 1e-10);
    }
}
