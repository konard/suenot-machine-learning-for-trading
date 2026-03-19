use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ============================================================================
// Encoder: maps input features to continuous embedding
// ============================================================================

pub struct Encoder {
    pub weights1: Array2<f64>,
    pub biases1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub biases2: Array1<f64>,
}

impl Encoder {
    pub fn new(input_dim: usize, hidden_dim: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();

        let weights1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            rng.gen_range(-scale1..scale1)
        });
        let biases1 = Array1::zeros(hidden_dim);
        let weights2 = Array2::from_shape_fn((hidden_dim, embedding_dim), |_| {
            rng.gen_range(-scale2..scale2)
        });
        let biases2 = Array1::zeros(embedding_dim);

        Self {
            weights1,
            biases1,
            weights2,
            biases2,
        }
    }

    /// Forward pass: input -> hidden (ReLU) -> embedding
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let hidden = x.dot(&self.weights1) + &self.biases1;
        let hidden = hidden.mapv(|v| v.max(0.0)); // ReLU
        hidden.dot(&self.weights2) + &self.biases2
    }
}

// ============================================================================
// Codebook: K embedding vectors with EMA updates
// ============================================================================

pub struct Codebook {
    pub embeddings: Vec<Array1<f64>>,
    pub ema_count: Vec<f64>,
    pub ema_weight: Vec<Array1<f64>>,
    pub decay: f64,
    pub epsilon: f64,
}

impl Codebook {
    pub fn new(num_embeddings: usize, embedding_dim: usize, decay: f64) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (num_embeddings as f64).sqrt();

        let embeddings: Vec<Array1<f64>> = (0..num_embeddings)
            .map(|_| Array1::from_shape_fn(embedding_dim, |_| rng.gen_range(-scale..scale)))
            .collect();

        let ema_count = vec![0.0; num_embeddings];
        let ema_weight: Vec<Array1<f64>> = embeddings.clone();

        Self {
            embeddings,
            ema_count,
            ema_weight,
            decay,
            epsilon: 1e-5,
        }
    }

    /// Find nearest codebook entry. Returns (quantized vector, index).
    pub fn quantize(&self, z_e: &Array1<f64>) -> (Array1<f64>, usize) {
        let (_, idx, _) = self.quantize_with_distance(z_e);
        (self.embeddings[idx].clone(), idx)
    }

    /// Find nearest codebook entry. Returns (quantized vector, index, distance).
    pub fn quantize_with_distance(&self, z_e: &Array1<f64>) -> (Array1<f64>, usize, f64) {
        let mut min_dist = f64::MAX;
        let mut min_idx = 0;

        for (i, entry) in self.embeddings.iter().enumerate() {
            let diff = z_e - entry;
            let dist = diff.mapv(|x| x * x).sum();
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        (self.embeddings[min_idx].clone(), min_idx, min_dist)
    }

    /// EMA update of codebook entries given encoder outputs and their assignments.
    pub fn ema_update(&mut self, z_e_batch: &[Array1<f64>], assignments: &[usize]) {
        let k = self.embeddings.len();
        let dim = self.embeddings[0].len();

        // Count assignments and accumulate embeddings per code
        let mut counts = vec![0.0_f64; k];
        let mut sums: Vec<Array1<f64>> = (0..k).map(|_| Array1::zeros(dim)).collect();

        for (z_e, &idx) in z_e_batch.iter().zip(assignments.iter()) {
            counts[idx] += 1.0;
            sums[idx] = &sums[idx] + z_e;
        }

        // EMA update
        for i in 0..k {
            self.ema_count[i] = self.decay * self.ema_count[i] + (1.0 - self.decay) * counts[i];
            self.ema_weight[i] =
                &self.ema_weight[i] * self.decay + &sums[i] * (1.0 - self.decay);

            // Laplace smoothing
            let n = self.ema_count[i] + self.epsilon;
            self.embeddings[i] = &self.ema_weight[i] / n;
        }
    }

    /// Number of codebook entries
    pub fn num_embeddings(&self) -> usize {
        self.embeddings.len()
    }
}

// ============================================================================
// Decoder: maps quantized embedding back to input space
// ============================================================================

pub struct Decoder {
    pub weights1: Array2<f64>,
    pub biases1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub biases2: Array1<f64>,
}

impl Decoder {
    pub fn new(embedding_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / embedding_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();

        let weights1 = Array2::from_shape_fn((embedding_dim, hidden_dim), |_| {
            rng.gen_range(-scale1..scale1)
        });
        let biases1 = Array1::zeros(hidden_dim);
        let weights2 = Array2::from_shape_fn((hidden_dim, output_dim), |_| {
            rng.gen_range(-scale2..scale2)
        });
        let biases2 = Array1::zeros(output_dim);

        Self {
            weights1,
            biases1,
            weights2,
            biases2,
        }
    }

    /// Forward pass: embedding -> hidden (ReLU) -> reconstruction
    pub fn forward(&self, z_q: &Array1<f64>) -> Array1<f64> {
        let hidden = z_q.dot(&self.weights1) + &self.biases1;
        let hidden = hidden.mapv(|v| v.max(0.0)); // ReLU
        hidden.dot(&self.weights2) + &self.biases2
    }
}

// ============================================================================
// VQ-VAE: combines encoder, codebook, and decoder
// ============================================================================

pub struct VQVAE {
    pub encoder: Encoder,
    pub codebook: Codebook,
    pub decoder: Decoder,
    pub beta: f64,
    pub learning_rate: f64,
}

impl VQVAE {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        embedding_dim: usize,
        num_embeddings: usize,
        beta: f64,
        learning_rate: f64,
    ) -> Self {
        Self {
            encoder: Encoder::new(input_dim, hidden_dim, embedding_dim),
            codebook: Codebook::new(num_embeddings, embedding_dim, 0.99),
            decoder: Decoder::new(embedding_dim, hidden_dim, input_dim),
            beta,
            learning_rate,
        }
    }

    /// Full forward pass: encode -> quantize -> decode.
    /// Returns (reconstruction, z_e, z_q, codebook_index).
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>, usize) {
        let z_e = self.encoder.forward(x);
        let (z_q, idx) = self.codebook.quantize(&z_e);
        // Straight-through: use z_q for forward, but gradient flows to z_e
        let x_hat = self.decoder.forward(&z_q);
        (x_hat, z_e, z_q, idx)
    }

    /// Compute VQ-VAE loss components.
    /// Returns (reconstruction_loss, codebook_loss, commitment_loss).
    pub fn compute_loss(
        &self,
        x: &Array1<f64>,
        x_hat: &Array1<f64>,
        z_e: &Array1<f64>,
        z_q: &Array1<f64>,
    ) -> (f64, f64, f64) {
        let recon_loss = (x - x_hat).mapv(|v| v * v).sum();
        let codebook_loss = (z_q - z_e).mapv(|v| v * v).sum();
        let commitment_loss = self.beta * (z_e - z_q).mapv(|v| v * v).sum();
        (recon_loss, codebook_loss, commitment_loss)
    }

    /// Train one epoch on a batch of data using simple gradient descent
    /// with numerical gradients for encoder/decoder, and EMA for codebook.
    pub fn train_epoch(&mut self, data: &[Array1<f64>]) -> f64 {
        let n = data.len();
        if n == 0 {
            return 0.0;
        }

        // Forward pass on all data to get assignments for EMA update
        let mut z_e_list = Vec::with_capacity(n);
        let mut assignments = Vec::with_capacity(n);
        let mut total_loss = 0.0;

        for x in data.iter() {
            let z_e = self.encoder.forward(x);
            let (z_q, idx) = self.codebook.quantize(&z_e);
            let x_hat = self.decoder.forward(&z_q);

            let (recon, cb, commit) = self.compute_loss(x, &x_hat, &z_e, &z_q);
            total_loss += recon + cb + commit;

            z_e_list.push(z_e);
            assignments.push(idx);
        }

        // EMA update for codebook
        self.codebook.ema_update(&z_e_list, &assignments);

        // Simple gradient update for encoder and decoder weights
        // Using numerical gradient approximation for simplicity
        let eps = 1e-4;
        let lr = self.learning_rate;

        // Update encoder weights1
        self.update_encoder_weights1(data, eps, lr);
        // Update encoder weights2
        self.update_encoder_weights2(data, eps, lr);
        // Update decoder weights1
        self.update_decoder_weights1(data, eps, lr);
        // Update decoder weights2
        self.update_decoder_weights2(data, eps, lr);

        total_loss / n as f64
    }

    /// Simplified training using stochastic updates with finite differences.
    /// For each sample, perturb random weight elements and estimate gradient.
    pub fn train_epoch_stochastic(&mut self, data: &[Array1<f64>], perturbations_per_sample: usize) -> f64 {
        let n = data.len();
        if n == 0 {
            return 0.0;
        }

        let mut rng = rand::thread_rng();
        let mut total_loss = 0.0;

        // Collect encoder outputs and assignments for EMA
        let mut z_e_list = Vec::with_capacity(n);
        let mut assignments = Vec::with_capacity(n);

        for x in data.iter() {
            let (x_hat, z_e, z_q, idx) = self.forward(x);
            let (recon, cb, commit) = self.compute_loss(x, &x_hat, &z_e, &z_q);
            total_loss += recon + cb + commit;
            z_e_list.push(z_e);
            assignments.push(idx);
        }

        // EMA update for codebook
        self.codebook.ema_update(&z_e_list, &assignments);

        // Stochastic perturbation of encoder and decoder weights
        let eps = 1e-4;
        let lr = self.learning_rate;

        for _ in 0..perturbations_per_sample {
            let sample = &data[rng.gen_range(0..n)];

            // Perturb random encoder weight1 element
            {
                let (r, c) = (
                    rng.gen_range(0..self.encoder.weights1.nrows()),
                    rng.gen_range(0..self.encoder.weights1.ncols()),
                );
                let grad = self.numerical_gradient_encoder_w1(sample, r, c, eps);
                self.encoder.weights1[[r, c]] -= lr * grad;
            }

            // Perturb random encoder weight2 element
            {
                let (r, c) = (
                    rng.gen_range(0..self.encoder.weights2.nrows()),
                    rng.gen_range(0..self.encoder.weights2.ncols()),
                );
                let grad = self.numerical_gradient_encoder_w2(sample, r, c, eps);
                self.encoder.weights2[[r, c]] -= lr * grad;
            }

            // Perturb random decoder weight1 element
            {
                let (r, c) = (
                    rng.gen_range(0..self.decoder.weights1.nrows()),
                    rng.gen_range(0..self.decoder.weights1.ncols()),
                );
                let grad = self.numerical_gradient_decoder_w1(sample, r, c, eps);
                self.decoder.weights1[[r, c]] -= lr * grad;
            }

            // Perturb random decoder weight2 element
            {
                let (r, c) = (
                    rng.gen_range(0..self.decoder.weights2.nrows()),
                    rng.gen_range(0..self.decoder.weights2.ncols()),
                );
                let grad = self.numerical_gradient_decoder_w2(sample, r, c, eps);
                self.decoder.weights2[[r, c]] -= lr * grad;
            }
        }

        total_loss / n as f64
    }

    fn sample_loss(&self, x: &Array1<f64>) -> f64 {
        let (x_hat, z_e, z_q, _) = self.forward(x);
        let (recon, cb, commit) = self.compute_loss(x, &x_hat, &z_e, &z_q);
        recon + cb + commit
    }

    fn numerical_gradient_encoder_w1(&mut self, x: &Array1<f64>, r: usize, c: usize, eps: f64) -> f64 {
        let orig = self.encoder.weights1[[r, c]];
        self.encoder.weights1[[r, c]] = orig + eps;
        let loss_plus = self.sample_loss(x);
        self.encoder.weights1[[r, c]] = orig - eps;
        let loss_minus = self.sample_loss(x);
        self.encoder.weights1[[r, c]] = orig;
        (loss_plus - loss_minus) / (2.0 * eps)
    }

    fn numerical_gradient_encoder_w2(&mut self, x: &Array1<f64>, r: usize, c: usize, eps: f64) -> f64 {
        let orig = self.encoder.weights2[[r, c]];
        self.encoder.weights2[[r, c]] = orig + eps;
        let loss_plus = self.sample_loss(x);
        self.encoder.weights2[[r, c]] = orig - eps;
        let loss_minus = self.sample_loss(x);
        self.encoder.weights2[[r, c]] = orig;
        (loss_plus - loss_minus) / (2.0 * eps)
    }

    fn numerical_gradient_decoder_w1(&mut self, x: &Array1<f64>, r: usize, c: usize, eps: f64) -> f64 {
        let orig = self.decoder.weights1[[r, c]];
        self.decoder.weights1[[r, c]] = orig + eps;
        let loss_plus = self.sample_loss(x);
        self.decoder.weights1[[r, c]] = orig - eps;
        let loss_minus = self.sample_loss(x);
        self.decoder.weights1[[r, c]] = orig;
        (loss_plus - loss_minus) / (2.0 * eps)
    }

    fn numerical_gradient_decoder_w2(&mut self, x: &Array1<f64>, r: usize, c: usize, eps: f64) -> f64 {
        let orig = self.decoder.weights2[[r, c]];
        self.decoder.weights2[[r, c]] = orig + eps;
        let loss_plus = self.sample_loss(x);
        self.decoder.weights2[[r, c]] = orig - eps;
        let loss_minus = self.sample_loss(x);
        self.decoder.weights2[[r, c]] = orig;
        (loss_plus - loss_minus) / (2.0 * eps)
    }

    // Full numerical gradient updates (used in train_epoch)
    fn update_encoder_weights1(&mut self, data: &[Array1<f64>], eps: f64, lr: f64) {
        let (rows, cols) = (self.encoder.weights1.nrows(), self.encoder.weights1.ncols());
        let mut rng = rand::thread_rng();
        // Update a subset of weights for efficiency
        let num_updates = (rows * cols).min(50);
        for _ in 0..num_updates {
            let r = rng.gen_range(0..rows);
            let c = rng.gen_range(0..cols);
            let mut grad = 0.0;
            for x in data.iter() {
                grad += self.numerical_gradient_encoder_w1(x, r, c, eps);
            }
            grad /= data.len() as f64;
            self.encoder.weights1[[r, c]] -= lr * grad;
        }
    }

    fn update_encoder_weights2(&mut self, data: &[Array1<f64>], eps: f64, lr: f64) {
        let (rows, cols) = (self.encoder.weights2.nrows(), self.encoder.weights2.ncols());
        let mut rng = rand::thread_rng();
        let num_updates = (rows * cols).min(50);
        for _ in 0..num_updates {
            let r = rng.gen_range(0..rows);
            let c = rng.gen_range(0..cols);
            let mut grad = 0.0;
            for x in data.iter() {
                grad += self.numerical_gradient_encoder_w2(x, r, c, eps);
            }
            grad /= data.len() as f64;
            self.encoder.weights2[[r, c]] -= lr * grad;
        }
    }

    fn update_decoder_weights1(&mut self, data: &[Array1<f64>], eps: f64, lr: f64) {
        let (rows, cols) = (self.decoder.weights1.nrows(), self.decoder.weights1.ncols());
        let mut rng = rand::thread_rng();
        let num_updates = (rows * cols).min(50);
        for _ in 0..num_updates {
            let r = rng.gen_range(0..rows);
            let c = rng.gen_range(0..cols);
            let mut grad = 0.0;
            for x in data.iter() {
                grad += self.numerical_gradient_decoder_w1(x, r, c, eps);
            }
            grad /= data.len() as f64;
            self.decoder.weights1[[r, c]] -= lr * grad;
        }
    }

    fn update_decoder_weights2(&mut self, data: &[Array1<f64>], eps: f64, lr: f64) {
        let (rows, cols) = (self.decoder.weights2.nrows(), self.decoder.weights2.ncols());
        let mut rng = rand::thread_rng();
        let num_updates = (rows * cols).min(50);
        for _ in 0..num_updates {
            let r = rng.gen_range(0..rows);
            let c = rng.gen_range(0..cols);
            let mut grad = 0.0;
            for x in data.iter() {
                grad += self.numerical_gradient_decoder_w2(x, r, c, eps);
            }
            grad /= data.len() as f64;
            self.decoder.weights2[[r, c]] -= lr * grad;
        }
    }

    /// Compute anomaly score for a single input (min codebook distance).
    pub fn anomaly_score(&self, x: &Array1<f64>) -> f64 {
        let z_e = self.encoder.forward(x);
        let (_, _, dist) = self.codebook.quantize_with_distance(&z_e);
        dist.sqrt()
    }

    /// Encode a batch of data and return codebook indices.
    pub fn encode_to_tokens(&self, data: &[Array1<f64>]) -> Vec<usize> {
        data.iter()
            .map(|x| {
                let z_e = self.encoder.forward(x);
                let (_, idx) = self.codebook.quantize(&z_e);
                idx
            })
            .collect()
    }
}

// ============================================================================
// Codebook utilization analysis
// ============================================================================

/// Compute codebook usage counts from token assignments.
pub fn codebook_usage(tokens: &[usize], num_embeddings: usize) -> Vec<usize> {
    let mut counts = vec![0usize; num_embeddings];
    for &t in tokens {
        if t < num_embeddings {
            counts[t] += 1;
        }
    }
    counts
}

/// Compute perplexity of codebook usage distribution.
/// Measures how many codes are effectively used.
/// Perplexity = K means all codes equally used; perplexity = 1 means only one code used.
pub fn codebook_perplexity(tokens: &[usize], num_embeddings: usize) -> f64 {
    let counts = codebook_usage(tokens, num_embeddings);
    let total = tokens.len() as f64;
    if total == 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}

/// Detect anomalous data points based on codebook distance.
/// Returns indices of data points whose anomaly score exceeds the given percentile.
pub fn detect_anomalies(
    vqvae: &VQVAE,
    data: &[Array1<f64>],
    percentile: f64,
) -> Vec<(usize, f64)> {
    let mut scores: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, x)| (i, vqvae.anomaly_score(x)))
        .collect();

    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let threshold_idx = ((percentile / 100.0) * scores.len() as f64) as usize;
    let threshold_idx = threshold_idx.min(scores.len().saturating_sub(1));
    let threshold = scores[threshold_idx].1;

    scores
        .into_iter()
        .filter(|(_, score)| *score >= threshold)
        .collect()
}

// ============================================================================
// Data normalization utilities
// ============================================================================

/// Normalize OHLCV data: compute percentage changes for prices, log-transform volume.
pub fn normalize_ohlcv(data: &[OhlcvCandle]) -> Vec<Vec<f64>> {
    if data.len() < 2 {
        return vec![];
    }

    let mut normalized = Vec::with_capacity(data.len() - 1);

    for i in 1..data.len() {
        let prev_close = data[i - 1].close;
        if prev_close == 0.0 {
            continue;
        }

        let open_pct = (data[i].open - prev_close) / prev_close;
        let high_pct = (data[i].high - prev_close) / prev_close;
        let low_pct = (data[i].low - prev_close) / prev_close;
        let close_pct = (data[i].close - prev_close) / prev_close;
        let vol_log = (data[i].volume + 1.0).ln();

        normalized.push(vec![open_pct, high_pct, low_pct, close_pct, vol_log]);
    }

    normalized
}

/// Create sliding windows from normalized data.
pub fn create_windows(normalized: &[Vec<f64>], window_size: usize) -> Vec<Array1<f64>> {
    if normalized.len() < window_size {
        return vec![];
    }

    let feature_dim = normalized[0].len();
    let mut windows = Vec::new();

    for i in 0..=(normalized.len() - window_size) {
        let mut flat = Vec::with_capacity(window_size * feature_dim);
        for j in 0..window_size {
            flat.extend_from_slice(&normalized[i + j]);
        }
        windows.push(Array1::from(flat));
    }

    windows
}

/// Z-score normalization of windows.
pub fn zscore_normalize(windows: &mut [Array1<f64>]) {
    if windows.is_empty() {
        return;
    }

    let dim = windows[0].len();
    let n = windows.len() as f64;

    for d in 0..dim {
        let mean: f64 = windows.iter().map(|w| w[d]).sum::<f64>() / n;
        let var: f64 = windows.iter().map(|w| (w[d] - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt().max(1e-8);

        for w in windows.iter_mut() {
            w[d] = (w[d] - mean) / std;
        }
    }
}

// ============================================================================
// Bybit API integration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvCandle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

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

/// Fetch OHLCV candles from Bybit API.
/// interval: "D" for daily, "60" for hourly, etc.
/// limit: max number of candles (up to 200).
pub fn fetch_bybit_ohlcv(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<OhlcvCandle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("User-Agent", "vq-vae-trading/0.1.0")
        .send()?
        .json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut candles: Vec<OhlcvCandle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(OhlcvCandle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first, reverse to chronological order
    candles.reverse();
    Ok(candles)
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_forward() {
        let encoder = Encoder::new(10, 16, 8);
        let input = Array1::from(vec![1.0; 10]);
        let output = encoder.forward(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_decoder_forward() {
        let decoder = Decoder::new(8, 16, 10);
        let input = Array1::from(vec![1.0; 8]);
        let output = decoder.forward(&input);
        assert_eq!(output.len(), 10);
    }

    #[test]
    fn test_codebook_quantize() {
        let codebook = Codebook::new(32, 8, 0.99);
        let z_e = Array1::from(vec![0.5; 8]);
        let (z_q, idx) = codebook.quantize(&z_e);
        assert_eq!(z_q.len(), 8);
        assert!(idx < 32);
    }

    #[test]
    fn test_codebook_quantize_with_distance() {
        let codebook = Codebook::new(32, 8, 0.99);
        let z_e = Array1::from(vec![0.5; 8]);
        let (z_q, idx, dist) = codebook.quantize_with_distance(&z_e);
        assert_eq!(z_q.len(), 8);
        assert!(idx < 32);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_codebook_ema_update() {
        let mut codebook = Codebook::new(4, 3, 0.99);
        let z_e_batch = vec![
            Array1::from(vec![1.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 1.0, 0.0]),
            Array1::from(vec![0.0, 0.0, 1.0]),
        ];
        let assignments = vec![0, 1, 2];
        codebook.ema_update(&z_e_batch, &assignments);
        // After update, embeddings should have shifted slightly
        // toward assigned vectors
        assert_eq!(codebook.embeddings.len(), 4);
    }

    #[test]
    fn test_vqvae_forward() {
        let vqvae = VQVAE::new(10, 16, 8, 32, 0.25, 0.001);
        let x = Array1::from(vec![0.1; 10]);
        let (x_hat, z_e, z_q, idx) = vqvae.forward(&x);
        assert_eq!(x_hat.len(), 10);
        assert_eq!(z_e.len(), 8);
        assert_eq!(z_q.len(), 8);
        assert!(idx < 32);
    }

    #[test]
    fn test_vqvae_loss() {
        let vqvae = VQVAE::new(10, 16, 8, 32, 0.25, 0.001);
        let x = Array1::from(vec![0.1; 10]);
        let (x_hat, z_e, z_q, _) = vqvae.forward(&x);
        let (recon, cb, commit) = vqvae.compute_loss(&x, &x_hat, &z_e, &z_q);
        assert!(recon >= 0.0);
        assert!(cb >= 0.0);
        assert!(commit >= 0.0);
    }

    #[test]
    fn test_anomaly_score() {
        let vqvae = VQVAE::new(10, 16, 8, 32, 0.25, 0.001);
        let x = Array1::from(vec![0.1; 10]);
        let score = vqvae.anomaly_score(&x);
        assert!(score >= 0.0);
    }

    #[test]
    fn test_encode_to_tokens() {
        let vqvae = VQVAE::new(10, 16, 8, 32, 0.25, 0.001);
        let data = vec![
            Array1::from(vec![0.1; 10]),
            Array1::from(vec![0.2; 10]),
            Array1::from(vec![0.3; 10]),
        ];
        let tokens = vqvae.encode_to_tokens(&data);
        assert_eq!(tokens.len(), 3);
        for &t in &tokens {
            assert!(t < 32);
        }
    }

    #[test]
    fn test_codebook_usage() {
        let tokens = vec![0, 1, 2, 0, 1, 0, 3];
        let usage = codebook_usage(&tokens, 5);
        assert_eq!(usage, vec![3, 2, 1, 1, 0]);
    }

    #[test]
    fn test_codebook_perplexity_uniform() {
        // Uniform usage of 4 codes -> perplexity should be ~4
        let tokens: Vec<usize> = (0..400).map(|i| i % 4).collect();
        let perplexity = codebook_perplexity(&tokens, 4);
        assert!((perplexity - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_codebook_perplexity_single() {
        // All same code -> perplexity should be 1
        let tokens = vec![0usize; 100];
        let perplexity = codebook_perplexity(&tokens, 4);
        assert!((perplexity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_ohlcv() {
        let candles = vec![
            OhlcvCandle { timestamp: 0, open: 100.0, high: 110.0, low: 90.0, close: 105.0, volume: 1000.0 },
            OhlcvCandle { timestamp: 1, open: 106.0, high: 112.0, low: 103.0, close: 110.0, volume: 1200.0 },
        ];
        let normalized = normalize_ohlcv(&candles);
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].len(), 5);
        // open_pct = (106 - 105) / 105 ≈ 0.00952
        assert!((normalized[0][0] - 1.0 / 105.0).abs() < 1e-6);
    }

    #[test]
    fn test_create_windows() {
        let normalized = vec![
            vec![0.01, 0.02, -0.01, 0.015, 7.0],
            vec![0.02, 0.03, 0.0, 0.025, 7.1],
            vec![-0.01, 0.01, -0.02, -0.005, 6.9],
            vec![0.03, 0.04, 0.01, 0.035, 7.2],
        ];
        let windows = create_windows(&normalized, 2);
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].len(), 10); // 2 * 5 features
    }

    #[test]
    fn test_detect_anomalies() {
        let vqvae = VQVAE::new(10, 16, 8, 32, 0.25, 0.001);
        let data: Vec<Array1<f64>> = (0..20)
            .map(|i| Array1::from(vec![i as f64 * 0.1; 10]))
            .collect();
        let anomalies = detect_anomalies(&vqvae, &data, 90.0);
        // Should detect ~10% as anomalies
        assert!(!anomalies.is_empty());
        assert!(anomalies.len() <= 4); // at most ~20% due to ties
    }

    #[test]
    fn test_zscore_normalize() {
        let mut windows = vec![
            Array1::from(vec![1.0, 2.0, 3.0]),
            Array1::from(vec![4.0, 5.0, 6.0]),
            Array1::from(vec![7.0, 8.0, 9.0]),
        ];
        zscore_normalize(&mut windows);
        // After z-score, mean should be ~0 and std ~1 for each dimension
        let mean: f64 = windows.iter().map(|w| w[0]).sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_train_epoch_stochastic() {
        let mut vqvae = VQVAE::new(5, 8, 4, 8, 0.25, 0.001);
        let data = vec![
            Array1::from(vec![0.1, 0.2, -0.1, 0.15, 7.0]),
            Array1::from(vec![0.2, 0.3, 0.0, 0.25, 7.1]),
            Array1::from(vec![-0.1, 0.1, -0.2, -0.05, 6.9]),
        ];
        let loss = vqvae.train_epoch_stochastic(&data, 5);
        assert!(loss > 0.0);
    }
}
