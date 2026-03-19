use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh_act(x: f64) -> f64 {
    x.tanh()
}

// ---------------------------------------------------------------------------
// Xavier-style weight initialization
// ---------------------------------------------------------------------------

fn xavier_init(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
}

fn zeros1(n: usize) -> Array1<f64> {
    Array1::zeros(n)
}

fn _zeros2(r: usize, c: usize) -> Array2<f64> {
    Array2::zeros((r, c))
}

// ---------------------------------------------------------------------------
// GRU Cell
// ---------------------------------------------------------------------------

/// A single GRU cell: input_size -> hidden_size.
#[derive(Clone)]
pub struct GRUCell {
    pub hidden_size: usize,
    pub input_size: usize,
    // Reset gate weights
    pub w_r: Array2<f64>,
    pub u_r: Array2<f64>,
    pub b_r: Array1<f64>,
    // Update gate weights
    pub w_u: Array2<f64>,
    pub u_u: Array2<f64>,
    pub b_u: Array1<f64>,
    // Candidate weights
    pub w_h: Array2<f64>,
    pub u_h: Array2<f64>,
    pub b_h: Array1<f64>,
}

impl GRUCell {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            hidden_size,
            input_size,
            w_r: xavier_init(hidden_size, input_size),
            u_r: xavier_init(hidden_size, hidden_size),
            b_r: zeros1(hidden_size),
            w_u: xavier_init(hidden_size, input_size),
            u_u: xavier_init(hidden_size, hidden_size),
            b_u: zeros1(hidden_size),
            w_h: xavier_init(hidden_size, input_size),
            u_h: xavier_init(hidden_size, hidden_size),
            b_h: zeros1(hidden_size),
        }
    }

    /// Forward pass: (x_t, h_{t-1}) -> h_t
    pub fn forward(&self, x: &Array1<f64>, h_prev: &Array1<f64>) -> Array1<f64> {
        let r = self.gate_sigmoid(&self.w_r, &self.u_r, &self.b_r, x, h_prev);
        let u = self.gate_sigmoid(&self.w_u, &self.u_u, &self.b_u, x, h_prev);

        // candidate: tanh(W_h * x + U_h * (r * h_prev) + b_h)
        let rh = &r * h_prev;
        let h_candidate = self.gate_tanh(&self.w_h, &self.u_h, &self.b_h, x, &rh);

        // h_t = (1 - u) * h_prev + u * h_candidate
        let one_minus_u = Array1::ones(self.hidden_size) - &u;
        &one_minus_u * h_prev + &u * &h_candidate
    }

    fn gate_sigmoid(
        &self,
        w: &Array2<f64>,
        u: &Array2<f64>,
        b: &Array1<f64>,
        x: &Array1<f64>,
        h: &Array1<f64>,
    ) -> Array1<f64> {
        let wx = w.dot(x);
        let uh = u.dot(h);
        let raw = &wx + &uh + b;
        raw.mapv(sigmoid)
    }

    fn gate_tanh(
        &self,
        w: &Array2<f64>,
        u: &Array2<f64>,
        b: &Array1<f64>,
        x: &Array1<f64>,
        h: &Array1<f64>,
    ) -> Array1<f64> {
        let wx = w.dot(x);
        let uh = u.dot(h);
        let raw = &wx + &uh + b;
        raw.mapv(tanh_act)
    }

    /// Collect all parameters as mutable slices for SGD updates.
    pub fn params_mut(&mut self) -> Vec<&mut Array2<f64>> {
        vec![
            &mut self.w_r,
            &mut self.u_r,
            &mut self.w_u,
            &mut self.u_u,
            &mut self.w_h,
            &mut self.u_h,
        ]
    }

    pub fn biases_mut(&mut self) -> Vec<&mut Array1<f64>> {
        vec![&mut self.b_r, &mut self.b_u, &mut self.b_h]
    }
}

// ---------------------------------------------------------------------------
// Linear Layer
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Linear {
    pub weight: Array2<f64>,
    pub bias: Array1<f64>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: xavier_init(out_features, in_features),
            bias: zeros1(out_features),
        }
    }

    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        self.weight.dot(x) + &self.bias
    }
}

// ---------------------------------------------------------------------------
// Recurrent Encoder: sequence -> (mu_t, log_var_t) per timestep
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RecurrentEncoder {
    pub gru: GRUCell,
    pub fc_mu: Linear,
    pub fc_logvar: Linear,
    pub hidden_size: usize,
}

impl RecurrentEncoder {
    pub fn new(input_size: usize, hidden_size: usize, latent_size: usize) -> Self {
        Self {
            gru: GRUCell::new(input_size, hidden_size),
            fc_mu: Linear::new(hidden_size, latent_size),
            fc_logvar: Linear::new(hidden_size, latent_size),
            hidden_size,
        }
    }

    /// Encode a full sequence.  Returns (mus, log_vars, hidden_states).
    pub fn forward(
        &self,
        sequence: &[Array1<f64>],
    ) -> (Vec<Array1<f64>>, Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut h = zeros1(self.hidden_size);
        let mut mus = Vec::with_capacity(sequence.len());
        let mut logvars = Vec::with_capacity(sequence.len());
        let mut hiddens = Vec::with_capacity(sequence.len());

        for x_t in sequence {
            h = self.gru.forward(x_t, &h);
            mus.push(self.fc_mu.forward(&h));
            logvars.push(self.fc_logvar.forward(&h));
            hiddens.push(h.clone());
        }
        (mus, logvars, hiddens)
    }
}

// ---------------------------------------------------------------------------
// Recurrent Decoder: z_t sequence -> reconstructed sequence
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RecurrentDecoder {
    pub gru: GRUCell,
    pub fc_out: Linear,
    pub hidden_size: usize,
}

impl RecurrentDecoder {
    pub fn new(latent_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            gru: GRUCell::new(latent_size, hidden_size),
            fc_out: Linear::new(hidden_size, output_size),
            hidden_size,
        }
    }

    /// Decode a sequence of latent vectors.
    pub fn forward(&self, z_seq: &[Array1<f64>]) -> Vec<Array1<f64>> {
        let mut h = zeros1(self.hidden_size);
        let mut outputs = Vec::with_capacity(z_seq.len());

        for z_t in z_seq {
            h = self.gru.forward(z_t, &h);
            outputs.push(self.fc_out.forward(&h));
        }
        outputs
    }
}

// ---------------------------------------------------------------------------
// Prior Network: h_{t-1} -> (mu_prior, logvar_prior)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct PriorNetwork {
    pub fc_mu: Linear,
    pub fc_logvar: Linear,
}

impl PriorNetwork {
    pub fn new(hidden_size: usize, latent_size: usize) -> Self {
        Self {
            fc_mu: Linear::new(hidden_size, latent_size),
            fc_logvar: Linear::new(hidden_size, latent_size),
        }
    }

    pub fn forward(&self, h: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        (self.fc_mu.forward(h), self.fc_logvar.forward(h))
    }
}

// ---------------------------------------------------------------------------
// Reparameterization trick
// ---------------------------------------------------------------------------

pub fn reparameterize(mu: &Array1<f64>, logvar: &Array1<f64>) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    let std = logvar.mapv(|v| (v * 0.5).exp());
    let eps = Array1::from_shape_fn(mu.len(), |_| rng.gen::<f64>() * 2.0 - 1.0);
    mu + &(&std * &eps)
}

// ---------------------------------------------------------------------------
// KL divergence between two diagonal Gaussians
// ---------------------------------------------------------------------------

pub fn kl_divergence(
    mu_q: &Array1<f64>,
    logvar_q: &Array1<f64>,
    mu_p: &Array1<f64>,
    logvar_p: &Array1<f64>,
) -> f64 {
    let d = mu_q.len();
    let mut kl = 0.0;
    for j in 0..d {
        let var_q = logvar_q[j].exp();
        let var_p = logvar_p[j].exp();
        kl += (var_p.ln() - var_q.ln()) + (var_q + (mu_q[j] - mu_p[j]).powi(2)) / var_p - 1.0;
    }
    0.5 * kl
}

// ---------------------------------------------------------------------------
// RVRAE Model
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RVRAE {
    pub encoder: RecurrentEncoder,
    pub decoder: RecurrentDecoder,
    pub prior: PriorNetwork,
    pub latent_size: usize,
    pub hidden_size: usize,
}

impl RVRAE {
    pub fn new(input_size: usize, hidden_size: usize, latent_size: usize) -> Self {
        Self {
            encoder: RecurrentEncoder::new(input_size, hidden_size, latent_size),
            decoder: RecurrentDecoder::new(latent_size, hidden_size, input_size),
            prior: PriorNetwork::new(hidden_size, latent_size),
            latent_size,
            hidden_size,
        }
    }

    /// Full forward pass. Returns (reconstructed, z_seq, total_loss).
    pub fn forward(&self, sequence: &[Array1<f64>], beta: f64) -> ForwardResult {
        let (mus, logvars, hiddens) = self.encoder.forward(sequence);

        let mut z_seq = Vec::with_capacity(sequence.len());
        let mut kl_total = 0.0;

        for t in 0..sequence.len() {
            let z_t = reparameterize(&mus[t], &logvars[t]);

            // Prior: conditioned on previous hidden state
            let h_prev = if t == 0 {
                zeros1(self.hidden_size)
            } else {
                hiddens[t - 1].clone()
            };
            let (mu_p, logvar_p) = self.prior.forward(&h_prev);

            kl_total += kl_divergence(&mus[t], &logvars[t], &mu_p, &logvar_p);
            z_seq.push(z_t);
        }

        let reconstructed = self.decoder.forward(&z_seq);

        // Reconstruction loss (MSE)
        let mut recon_loss = 0.0;
        for t in 0..sequence.len() {
            let diff = &sequence[t] - &reconstructed[t];
            recon_loss += diff.mapv(|v| v * v).sum();
        }
        recon_loss /= sequence.len() as f64;

        let total_loss = recon_loss + beta * kl_total / sequence.len() as f64;

        ForwardResult {
            reconstructed,
            z_seq,
            recon_loss,
            kl_loss: kl_total / sequence.len() as f64,
            total_loss,
        }
    }

    /// Extract dynamic latent factors from a sequence.
    pub fn extract_factors(&self, sequence: &[Array1<f64>]) -> Vec<Array1<f64>> {
        let (mus, _logvars, _hiddens) = self.encoder.forward(sequence);
        // Use the mean (no sampling) for deterministic factor extraction
        mus
    }

    /// Simple SGD training loop with KL annealing.
    pub fn train(
        &mut self,
        data: &[Array1<f64>],
        epochs: usize,
        learning_rate: f64,
        annealing_epochs: usize,
    ) -> Vec<f64> {
        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let beta = (epoch as f64 / annealing_epochs as f64).min(1.0);

            let result = self.forward(data, beta);
            losses.push(result.total_loss);

            // Numerical gradient descent (perturbation-based).
            // For a production system one would use autograd; here we use
            // finite-difference approximation on the encoder/decoder parameters
            // to keep the implementation self-contained.
            self.numerical_sgd_step(data, beta, learning_rate);

            if epoch % 50 == 0 || epoch == epochs - 1 {
                println!(
                    "Epoch {}/{}: loss={:.6}, recon={:.6}, kl={:.6}, beta={:.4}",
                    epoch + 1,
                    epochs,
                    result.total_loss,
                    result.recon_loss,
                    result.kl_loss,
                    beta
                );
            }
        }
        losses
    }

    /// Perturbation-based SGD step for all linear projection weights.
    fn numerical_sgd_step(
        &mut self,
        data: &[Array1<f64>],
        beta: f64,
        lr: f64,
    ) {
        let eps = 1e-4;

        // Helper: perturb a single element in a Linear layer's weight matrix
        // and compute the loss gradient via finite differences.
        // We update the encoder projection layers and decoder projection layer.
        self.update_linear_weights(&self.encoder.fc_mu.clone(), data, beta, lr, eps, |m, l| {
            m.encoder.fc_mu = l;
        });
        self.update_linear_weights(&self.encoder.fc_logvar.clone(), data, beta, lr, eps, |m, l| {
            m.encoder.fc_logvar = l;
        });
        self.update_linear_weights(&self.decoder.fc_out.clone(), data, beta, lr, eps, |m, l| {
            m.decoder.fc_out = l;
        });
    }

    fn update_linear_weights<F>(
        &mut self,
        layer: &Linear,
        data: &[Array1<f64>],
        beta: f64,
        lr: f64,
        eps: f64,
        set_layer: F,
    ) where
        F: Fn(&mut Self, Linear),
    {
        let _base_loss = self.forward(data, beta).total_loss;
        let (rows, cols) = layer.weight.dim();

        let mut new_layer = layer.clone();

        // Update a subset of weights for efficiency (stochastic coordinate descent)
        let mut rng = rand::thread_rng();
        let num_updates = (rows * cols).min(8);
        for _ in 0..num_updates {
            let i = rng.gen_range(0..rows);
            let j = rng.gen_range(0..cols);

            new_layer.weight[[i, j]] += eps;
            set_layer(self, new_layer.clone());
            let loss_plus = self.forward(data, beta).total_loss;

            new_layer.weight[[i, j]] -= 2.0 * eps;
            set_layer(self, new_layer.clone());
            let loss_minus = self.forward(data, beta).total_loss;

            let grad = (loss_plus - loss_minus) / (2.0 * eps);
            new_layer.weight[[i, j]] += eps; // restore
            new_layer.weight[[i, j]] -= lr * grad;
        }

        // Update biases
        let num_bias_updates = new_layer.bias.len().min(4);
        for _ in 0..num_bias_updates {
            let i = rng.gen_range(0..new_layer.bias.len());

            new_layer.bias[i] += eps;
            set_layer(self, new_layer.clone());
            let loss_plus = self.forward(data, beta).total_loss;

            new_layer.bias[i] -= 2.0 * eps;
            set_layer(self, new_layer.clone());
            let loss_minus = self.forward(data, beta).total_loss;

            let grad = (loss_plus - loss_minus) / (2.0 * eps);
            new_layer.bias[i] += eps;
            new_layer.bias[i] -= lr * grad;
        }

        set_layer(self, new_layer);
    }
}

pub struct ForwardResult {
    pub reconstructed: Vec<Array1<f64>>,
    pub z_seq: Vec<Array1<f64>>,
    pub recon_loss: f64,
    pub kl_loss: f64,
    pub total_loss: f64,
}

// ---------------------------------------------------------------------------
// Factor Dynamics Analysis
// ---------------------------------------------------------------------------

pub struct FactorDynamics;

impl FactorDynamics {
    /// Compute autocorrelation of a factor trajectory at a given lag.
    pub fn autocorrelation(factor_values: &[f64], lag: usize) -> f64 {
        let n = factor_values.len();
        if lag >= n {
            return 0.0;
        }
        let mean: f64 = factor_values.iter().sum::<f64>() / n as f64;
        let var: f64 = factor_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        if var < 1e-12 {
            return 0.0;
        }
        let cov: f64 = (0..n - lag)
            .map(|t| (factor_values[t] - mean) * (factor_values[t + lag] - mean))
            .sum::<f64>()
            / (n - lag) as f64;
        cov / var
    }

    /// Compute factor velocity: ||z_t - z_{t-1}|| for each t.
    pub fn factor_velocity(z_seq: &[Array1<f64>]) -> Vec<f64> {
        let mut velocities = Vec::with_capacity(z_seq.len().saturating_sub(1));
        for t in 1..z_seq.len() {
            let diff = &z_seq[t] - &z_seq[t - 1];
            let norm = diff.mapv(|v| v * v).sum().sqrt();
            velocities.push(norm);
        }
        velocities
    }

    /// Detect regime shifts as timesteps where velocity exceeds threshold.
    pub fn detect_regime_shifts(z_seq: &[Array1<f64>], threshold_sigma: f64) -> Vec<usize> {
        let velocities = Self::factor_velocity(z_seq);
        if velocities.is_empty() {
            return vec![];
        }
        let mean_v: f64 = velocities.iter().sum::<f64>() / velocities.len() as f64;
        let var_v: f64 = velocities.iter().map(|v| (v - mean_v).powi(2)).sum::<f64>()
            / velocities.len() as f64;
        let std_v = var_v.sqrt();
        let threshold = mean_v + threshold_sigma * std_v;

        velocities
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > threshold)
            .map(|(t, _)| t + 1) // +1 because velocity[0] corresponds to t=1
            .collect()
    }

    /// Compute pairwise correlation between two factor trajectories.
    pub fn factor_correlation(factor_a: &[f64], factor_b: &[f64]) -> f64 {
        let n = factor_a.len().min(factor_b.len());
        if n == 0 {
            return 0.0;
        }
        let mean_a: f64 = factor_a[..n].iter().sum::<f64>() / n as f64;
        let mean_b: f64 = factor_b[..n].iter().sum::<f64>() / n as f64;

        let cov: f64 = (0..n)
            .map(|i| (factor_a[i] - mean_a) * (factor_b[i] - mean_b))
            .sum::<f64>()
            / n as f64;
        let std_a = (factor_a[..n]
            .iter()
            .map(|v| (v - mean_a).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        let std_b = (factor_b[..n]
            .iter()
            .map(|v| (v - mean_b).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();

        if std_a < 1e-12 || std_b < 1e-12 {
            return 0.0;
        }
        cov / (std_a * std_b)
    }

    /// Rolling factor correlation with a given window size.
    pub fn rolling_correlation(
        factor_a: &[f64],
        factor_b: &[f64],
        window: usize,
    ) -> Vec<f64> {
        let n = factor_a.len().min(factor_b.len());
        if n < window {
            return vec![];
        }
        (0..=n - window)
            .map(|start| {
                Self::factor_correlation(
                    &factor_a[start..start + window],
                    &factor_b[start..start + window],
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Bybit API Data Fetching
// ---------------------------------------------------------------------------

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

/// Fetch kline data from Bybit API for a given symbol.
/// Returns a vector of closing prices (newest first from API, reversed to chronological).
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<f64>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    // Each entry: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    let mut closes: Vec<f64> = resp
        .result
        .list
        .iter()
        .filter_map(|row| row.get(4)?.parse::<f64>().ok())
        .collect();

    // API returns newest first; reverse to chronological order
    closes.reverse();
    Ok(closes)
}

/// Compute log returns from a price series.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Normalize a series to zero mean and unit variance.
pub fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt().max(1e-12);
    let normalized: Vec<f64> = data.iter().map(|v| (v - mean) / std).collect();
    (normalized, mean, std)
}

/// Combine multiple asset return series into a multivariate time-series
/// represented as Vec<Array1<f64>> where each entry is one timestep across all assets.
pub fn combine_assets(asset_returns: &[Vec<f64>]) -> Vec<Array1<f64>> {
    if asset_returns.is_empty() {
        return vec![];
    }
    let min_len = asset_returns.iter().map(|a| a.len()).min().unwrap_or(0);
    let num_assets = asset_returns.len();

    (0..min_len)
        .map(|t| {
            let vals: Vec<f64> = (0..num_assets).map(|a| asset_returns[a][t]).collect();
            Array1::from_vec(vals)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gru_cell_output_shape() {
        let gru = GRUCell::new(3, 5);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let h = zeros1(5);
        let h_new = gru.forward(&x, &h);
        assert_eq!(h_new.len(), 5);
    }

    #[test]
    fn test_gru_cell_deterministic() {
        let gru = GRUCell::new(2, 3);
        let x = Array1::from_vec(vec![0.5, -0.5]);
        let h = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let h1 = gru.forward(&x, &h);
        let h2 = gru.forward(&x, &h);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_encoder_output_shapes() {
        let enc = RecurrentEncoder::new(3, 8, 2);
        let seq: Vec<Array1<f64>> = (0..5)
            .map(|_| Array1::from_vec(vec![0.1, 0.2, 0.3]))
            .collect();
        let (mus, logvars, hiddens) = enc.forward(&seq);
        assert_eq!(mus.len(), 5);
        assert_eq!(logvars.len(), 5);
        assert_eq!(hiddens.len(), 5);
        assert_eq!(mus[0].len(), 2);
        assert_eq!(logvars[0].len(), 2);
        assert_eq!(hiddens[0].len(), 8);
    }

    #[test]
    fn test_decoder_output_shapes() {
        let dec = RecurrentDecoder::new(2, 8, 3);
        let z_seq: Vec<Array1<f64>> = (0..5)
            .map(|_| Array1::from_vec(vec![0.1, 0.2]))
            .collect();
        let outputs = dec.forward(&z_seq);
        assert_eq!(outputs.len(), 5);
        assert_eq!(outputs[0].len(), 3);
    }

    #[test]
    fn test_reparameterize_shape() {
        let mu = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let logvar = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let z = reparameterize(&mu, &logvar);
        assert_eq!(z.len(), 3);
    }

    #[test]
    fn test_kl_divergence_same_distribution() {
        let mu = Array1::from_vec(vec![0.0, 0.0]);
        let logvar = Array1::from_vec(vec![0.0, 0.0]);
        let kl = kl_divergence(&mu, &logvar, &mu, &logvar);
        assert!(kl.abs() < 1e-10, "KL divergence of identical dists should be ~0");
    }

    #[test]
    fn test_kl_divergence_different_distributions() {
        let mu_q = Array1::from_vec(vec![1.0, 0.0]);
        let logvar_q = Array1::from_vec(vec![0.0, 0.0]);
        let mu_p = Array1::from_vec(vec![0.0, 0.0]);
        let logvar_p = Array1::from_vec(vec![0.0, 0.0]);
        let kl = kl_divergence(&mu_q, &logvar_q, &mu_p, &logvar_p);
        assert!(kl > 0.0, "KL should be positive for different dists");
        // KL(N(1,1)||N(0,1)) = 0.5 * (0 + 1 + 1 - 1) = 0.5 per dim with shift
        assert!((kl - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rvrae_forward() {
        let model = RVRAE::new(3, 8, 2);
        let seq: Vec<Array1<f64>> = (0..10)
            .map(|i| Array1::from_vec(vec![i as f64 * 0.1, 0.2, -0.1]))
            .collect();
        let result = model.forward(&seq, 1.0);
        assert_eq!(result.reconstructed.len(), 10);
        assert_eq!(result.z_seq.len(), 10);
        assert!(result.total_loss.is_finite());
    }

    #[test]
    fn test_extract_factors() {
        let model = RVRAE::new(3, 8, 2);
        let seq: Vec<Array1<f64>> = (0..10)
            .map(|i| Array1::from_vec(vec![i as f64 * 0.1, 0.2, -0.1]))
            .collect();
        let factors = model.extract_factors(&seq);
        assert_eq!(factors.len(), 10);
        assert_eq!(factors[0].len(), 2);
    }

    #[test]
    fn test_autocorrelation() {
        // Constant series => autocorrelation = 0 (variance ~0)
        let constant = vec![1.0; 20];
        let ac = FactorDynamics::autocorrelation(&constant, 1);
        assert!(ac.abs() < 1e-6);

        // Sine wave should have positive autocorrelation at small lag
        let sine: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin())
            .collect();
        let ac1 = FactorDynamics::autocorrelation(&sine, 1);
        assert!(ac1 > 0.5, "Sine wave should have high lag-1 autocorrelation");
    }

    #[test]
    fn test_factor_velocity() {
        let z_seq = vec![
            Array1::from_vec(vec![0.0, 0.0]),
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![1.0, 1.0]),
        ];
        let velocities = FactorDynamics::factor_velocity(&z_seq);
        assert_eq!(velocities.len(), 2);
        assert!((velocities[0] - 1.0).abs() < 1e-10);
        assert!((velocities[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_regime_shift_detection() {
        let mut z_seq: Vec<Array1<f64>> = (0..20)
            .map(|i| Array1::from_vec(vec![i as f64 * 0.01, 0.0]))
            .collect();
        // Insert a big jump at t=10
        z_seq[10] = Array1::from_vec(vec![5.0, 5.0]);
        let shifts = FactorDynamics::detect_regime_shifts(&z_seq, 2.0);
        assert!(!shifts.is_empty(), "Should detect the regime shift");
        assert!(shifts.contains(&10) || shifts.contains(&11));
    }

    #[test]
    fn test_factor_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = FactorDynamics::factor_correlation(&a, &b);
        assert!((corr - 1.0).abs() < 1e-10, "Perfectly correlated");

        let c = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = FactorDynamics::factor_correlation(&a, &c);
        assert!((corr_neg + 1.0).abs() < 1e-10, "Perfectly anti-correlated");
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 105.0, 103.0];
        let rets = log_returns(&prices);
        assert_eq!(rets.len(), 2);
        assert!((rets[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (normed, mean, std) = normalize(&data);
        assert!((mean - 3.0).abs() < 1e-10);
        assert!(std > 0.0);
        let normed_mean: f64 = normed.iter().sum::<f64>() / normed.len() as f64;
        assert!(normed_mean.abs() < 1e-10);
    }

    #[test]
    fn test_combine_assets() {
        let a = vec![0.01, 0.02, 0.03];
        let b = vec![0.04, 0.05, 0.06];
        let combined = combine_assets(&[a, b]);
        assert_eq!(combined.len(), 3);
        assert_eq!(combined[0].len(), 2);
        assert!((combined[0][0] - 0.01).abs() < 1e-10);
        assert!((combined[0][1] - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let rolling = FactorDynamics::rolling_correlation(&a, &b, 5);
        assert_eq!(rolling.len(), 6);
        for corr in &rolling {
            assert!((*corr - 1.0).abs() < 1e-10);
        }
    }
}
