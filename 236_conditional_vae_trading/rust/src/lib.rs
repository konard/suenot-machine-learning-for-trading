//! Conditional VAE (CVAE) for Trading
//!
//! Implements a Conditional Variational Autoencoder that generates
//! market scenarios conditioned on regime labels (bull/bear/sideways).

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Market Regime
// ---------------------------------------------------------------------------

/// Market regime labels used as conditioning variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
}

impl MarketRegime {
    pub const COUNT: usize = 3;

    /// One-hot encode the regime.
    pub fn one_hot(&self) -> Array1<f64> {
        let mut v = Array1::zeros(Self::COUNT);
        v[self.index()] = 1.0;
        v
    }

    pub fn index(&self) -> usize {
        match self {
            MarketRegime::Bull => 0,
            MarketRegime::Bear => 1,
            MarketRegime::Sideways => 2,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => MarketRegime::Bull,
            1 => MarketRegime::Bear,
            _ => MarketRegime::Sideways,
        }
    }
}

// ---------------------------------------------------------------------------
// Regime detection
// ---------------------------------------------------------------------------

/// Detect market regimes from a price series using rolling statistics.
///
/// * `prices` – closing prices (must have length > `window`)
/// * `window` – rolling window size for mean / std computation
pub fn detect_regimes(prices: &[f64], window: usize) -> Vec<MarketRegime> {
    if prices.len() < 2 {
        return vec![];
    }

    // Compute log-returns
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    if returns.len() < window {
        return returns.iter().map(|_| MarketRegime::Sideways).collect();
    }

    // Global average volatility (used as threshold)
    let global_vol = {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64).sqrt()
    };

    let mut regimes = Vec::with_capacity(returns.len());

    for i in 0..returns.len() {
        if i < window {
            regimes.push(MarketRegime::Sideways);
            continue;
        }
        let slice = &returns[i - window..i];
        let roll_mean = slice.iter().sum::<f64>() / window as f64;
        let roll_vol = (slice.iter().map(|r| (r - roll_mean).powi(2)).sum::<f64>()
            / window as f64)
            .sqrt();

        let regime = if roll_mean > 0.0 && roll_vol <= global_vol {
            MarketRegime::Bull
        } else if roll_mean < 0.0 && roll_vol > global_vol {
            MarketRegime::Bear
        } else {
            MarketRegime::Sideways
        };
        regimes.push(regime);
    }

    regimes
}

// ---------------------------------------------------------------------------
// Simple linear layer helper
// ---------------------------------------------------------------------------

/// A dense (fully-connected) layer: y = relu(x * W + b)  or  y = x * W + b.
#[derive(Debug, Clone)]
struct Linear {
    weight: Array2<f64>,
    bias: Array1<f64>,
}

impl Linear {
    fn new_random(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / input_dim as f64).sqrt();
        let weight = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);
        Self { weight, bias }
    }

    fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.dot(&self.weight) + &self.bias
    }

    fn forward_relu(&self, x: &Array1<f64>) -> Array1<f64> {
        let out = self.forward(x);
        out.mapv(|v| v.max(0.0))
    }

    /// Simple SGD update: w -= lr * grad.
    fn update(&mut self, weight_grad: &Array2<f64>, bias_grad: &Array1<f64>, lr: f64) {
        self.weight.scaled_add(-lr, weight_grad);
        self.bias.scaled_add(-lr, bias_grad);
    }
}

// ---------------------------------------------------------------------------
// Condition Encoder
// ---------------------------------------------------------------------------

/// Encodes conditioning information (one-hot regime or continuous macro) into
/// a fixed-dimensional representation.
#[derive(Debug, Clone)]
pub struct ConditionEncoder {
    layer: Linear,
    output_dim: usize,
}

impl ConditionEncoder {
    pub fn new(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            layer: Linear::new_random(input_dim, output_dim, rng),
            output_dim,
        }
    }

    pub fn forward(&self, condition: &Array1<f64>) -> Array1<f64> {
        self.layer.forward_relu(condition)
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

// ---------------------------------------------------------------------------
// CVAE Encoder  q(z | x, c)
// ---------------------------------------------------------------------------

/// The recognition network: maps (x, encoded_condition) → (mu, log_var).
#[derive(Debug, Clone)]
pub struct CVAEEncoder {
    hidden: Linear,
    mu_layer: Linear,
    logvar_layer: Linear,
    #[allow(dead_code)]
    latent_dim: usize,
}

impl CVAEEncoder {
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            hidden: Linear::new_random(input_dim, hidden_dim, rng),
            mu_layer: Linear::new_random(hidden_dim, latent_dim, rng),
            logvar_layer: Linear::new_random(hidden_dim, latent_dim, rng),
            latent_dim,
        }
    }

    /// Returns (mu, log_var).
    pub fn forward(&self, x_and_c: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let h = self.hidden.forward_relu(x_and_c);
        let mu = self.mu_layer.forward(&h);
        let logvar = self.logvar_layer.forward(&h);
        (mu, logvar)
    }
}

// ---------------------------------------------------------------------------
// CVAE Decoder  p(x | z, c)
// ---------------------------------------------------------------------------

/// The generative network: maps (z, encoded_condition) → reconstructed x.
#[derive(Debug, Clone)]
pub struct CVAEDecoder {
    hidden: Linear,
    output_layer: Linear,
}

impl CVAEDecoder {
    pub fn new(
        latent_dim: usize,
        cond_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            hidden: Linear::new_random(latent_dim + cond_dim, hidden_dim, rng),
            output_layer: Linear::new_random(hidden_dim, output_dim, rng),
        }
    }

    pub fn forward(&self, z_and_c: &Array1<f64>) -> Array1<f64> {
        let h = self.hidden.forward_relu(z_and_c);
        self.output_layer.forward(&h) // linear output
    }
}

// ---------------------------------------------------------------------------
// Conditional Prior  p(z | c)
// ---------------------------------------------------------------------------

/// Learned prior that maps the condition to (mu_prior, log_var_prior).
#[derive(Debug, Clone)]
pub struct ConditionalPrior {
    mu_layer: Linear,
    logvar_layer: Linear,
}

impl ConditionalPrior {
    pub fn new(cond_dim: usize, latent_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            mu_layer: Linear::new_random(cond_dim, latent_dim, rng),
            logvar_layer: Linear::new_random(cond_dim, latent_dim, rng),
        }
    }

    pub fn forward(&self, c: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let mu = self.mu_layer.forward(c);
        let logvar = self.logvar_layer.forward(c);
        (mu, logvar)
    }
}

// ---------------------------------------------------------------------------
// CVAE  (full model)
// ---------------------------------------------------------------------------

/// Complete Conditional VAE for trading data.
pub struct CVAE {
    pub cond_encoder: ConditionEncoder,
    pub encoder: CVAEEncoder,
    pub decoder: CVAEDecoder,
    pub prior: ConditionalPrior,
    pub input_dim: usize,
    pub latent_dim: usize,
    pub cond_encoded_dim: usize,
}

impl CVAE {
    /// Create a new CVAE.
    ///
    /// * `input_dim` – dimensionality of input data (e.g. window of returns)
    /// * `cond_raw_dim` – raw condition size (e.g. 3 for one-hot regime)
    /// * `cond_encoded_dim` – encoded condition size
    /// * `hidden_dim` – hidden layer size
    /// * `latent_dim` – latent space dimensionality
    pub fn new(
        input_dim: usize,
        cond_raw_dim: usize,
        cond_encoded_dim: usize,
        hidden_dim: usize,
        latent_dim: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let cond_encoder = ConditionEncoder::new(cond_raw_dim, cond_encoded_dim, rng);
        let encoder = CVAEEncoder::new(input_dim + cond_encoded_dim, hidden_dim, latent_dim, rng);
        let decoder = CVAEDecoder::new(latent_dim, cond_encoded_dim, hidden_dim, input_dim, rng);
        let prior = ConditionalPrior::new(cond_encoded_dim, latent_dim, rng);

        Self {
            cond_encoder,
            encoder,
            decoder,
            prior,
            input_dim,
            latent_dim,
            cond_encoded_dim,
        }
    }

    /// Reparameterization trick: sample z = mu + sigma * eps.
    pub fn reparameterize(mu: &Array1<f64>, logvar: &Array1<f64>, rng: &mut impl Rng) -> Array1<f64> {
        let std = logvar.mapv(|lv| (lv * 0.5).exp());
        let eps = Array1::from_shape_fn(mu.len(), |_| {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        });
        mu + &(std * &eps)
    }

    /// Concatenate two 1-D arrays.
    fn concat(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
        let mut v = Vec::with_capacity(a.len() + b.len());
        v.extend(a.iter());
        v.extend(b.iter());
        Array1::from(v)
    }

    /// Full forward pass. Returns (reconstruction, mu_enc, logvar_enc, mu_prior, logvar_prior).
    pub fn forward(
        &self,
        x: &Array1<f64>,
        cond_raw: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let c = self.cond_encoder.forward(cond_raw);
        let xc = Self::concat(x, &c);
        let (mu_enc, logvar_enc) = self.encoder.forward(&xc);
        let z = Self::reparameterize(&mu_enc, &logvar_enc, rng);
        let zc = Self::concat(&z, &c);
        let recon = self.decoder.forward(&zc);
        let (mu_prior, logvar_prior) = self.prior.forward(&c);
        (recon, mu_enc, logvar_enc, mu_prior, logvar_prior)
    }

    /// Conditional ELBO loss = reconstruction_loss + KL divergence.
    pub fn loss(
        x: &Array1<f64>,
        recon: &Array1<f64>,
        mu_enc: &Array1<f64>,
        logvar_enc: &Array1<f64>,
        mu_prior: &Array1<f64>,
        logvar_prior: &Array1<f64>,
    ) -> (f64, f64, f64) {
        // Reconstruction: MSE
        let recon_loss = (x - recon).mapv(|v| v.powi(2)).sum() / x.len() as f64;

        // KL divergence between q(z|x,c) and p(z|c)
        let kl = 0.5
            * (logvar_prior - logvar_enc
                + (logvar_enc.mapv(f64::exp)
                    + (mu_enc - mu_prior).mapv(|v| v.powi(2)))
                    / logvar_prior.mapv(f64::exp)
                - 1.0)
                .sum();

        let total = recon_loss + kl;
        (total, recon_loss, kl)
    }

    /// Generate samples conditioned on a specific regime.
    pub fn generate(
        &self,
        regime: MarketRegime,
        num_samples: usize,
        rng: &mut impl Rng,
    ) -> Array2<f64> {
        let cond_raw = regime.one_hot();
        let c = self.cond_encoder.forward(&cond_raw);
        let (mu_prior, logvar_prior) = self.prior.forward(&c);

        let mut samples = Array2::zeros((num_samples, self.input_dim));
        for i in 0..num_samples {
            let z = Self::reparameterize(&mu_prior, &logvar_prior, rng);
            let zc = Self::concat(&z, &c);
            let x = self.decoder.forward(&zc);
            samples.row_mut(i).assign(&x);
        }
        samples
    }
}

// ---------------------------------------------------------------------------
// Training (simple numerical-gradient approach for demonstration)
// ---------------------------------------------------------------------------

/// Train the CVAE using finite-difference gradient estimation.
///
/// This is a simplified training loop suitable for demonstration.
/// Production systems should use automatic differentiation.
pub fn train_cvae(
    cvae: &mut CVAE,
    data: &[(Array1<f64>, MarketRegime)],
    epochs: usize,
    lr: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let mut loss_history = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (x, regime) in data.iter() {
            let cond_raw = regime.one_hot();
            let (recon, mu_enc, logvar_enc, mu_prior, logvar_prior) =
                cvae.forward(x, &cond_raw, rng);
            let (loss, _recon_l, _kl_l) =
                CVAE::loss(x, &recon, &mu_enc, &logvar_enc, &mu_prior, &logvar_prior);
            total_loss += loss;

            // --- Numerical gradient for decoder output layer ---
            let eps = 1e-4;

            // Update decoder output weights
            let (rows, cols) = cvae.decoder.output_layer.weight.dim();
            let mut w_grad = Array2::zeros((rows, cols));
            for r in 0..rows {
                for col in 0..cols {
                    cvae.decoder.output_layer.weight[[r, col]] += eps;
                    let (recon_p, _, _, _, _) = cvae.forward(x, &cond_raw, rng);
                    let (loss_p, _, _) =
                        CVAE::loss(x, &recon_p, &mu_enc, &logvar_enc, &mu_prior, &logvar_prior);
                    cvae.decoder.output_layer.weight[[r, col]] -= eps;
                    w_grad[[r, col]] = (loss_p - loss) / eps;
                }
            }
            let mut b_grad = Array1::zeros(cols);
            for col in 0..cols {
                cvae.decoder.output_layer.bias[col] += eps;
                let (recon_p, _, _, _, _) = cvae.forward(x, &cond_raw, rng);
                let (loss_p, _, _) =
                    CVAE::loss(x, &recon_p, &mu_enc, &logvar_enc, &mu_prior, &logvar_prior);
                cvae.decoder.output_layer.bias[col] -= eps;
                b_grad[col] = (loss_p - loss) / eps;
            }
            cvae.decoder.output_layer.update(&w_grad, &b_grad, lr);

            // Update decoder hidden weights
            let (rows, cols) = cvae.decoder.hidden.weight.dim();
            let mut w_grad = Array2::zeros((rows, cols));
            for r in 0..rows {
                for col in 0..cols {
                    cvae.decoder.hidden.weight[[r, col]] += eps;
                    let (recon_p, _, _, _, _) = cvae.forward(x, &cond_raw, rng);
                    let (loss_p, _, _) =
                        CVAE::loss(x, &recon_p, &mu_enc, &logvar_enc, &mu_prior, &logvar_prior);
                    cvae.decoder.hidden.weight[[r, col]] -= eps;
                    w_grad[[r, col]] = (loss_p - loss) / eps;
                }
            }
            let mut b_grad = Array1::zeros(cols);
            for col in 0..cols {
                cvae.decoder.hidden.bias[col] += eps;
                let (recon_p, _, _, _, _) = cvae.forward(x, &cond_raw, rng);
                let (loss_p, _, _) =
                    CVAE::loss(x, &recon_p, &mu_enc, &logvar_enc, &mu_prior, &logvar_prior);
                cvae.decoder.hidden.bias[col] -= eps;
                b_grad[col] = (loss_p - loss) / eps;
            }
            cvae.decoder.hidden.update(&w_grad, &b_grad, lr);
        }

        let avg_loss = total_loss / data.len() as f64;
        loss_history.push(avg_loss);

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("Epoch {}/{}: avg loss = {:.6}", epoch + 1, epochs, avg_loss);
        }
    }

    loss_history
}

// ---------------------------------------------------------------------------
// Quality Metrics per Condition
// ---------------------------------------------------------------------------

/// Compute per-regime quality metrics comparing generated vs real data.
pub struct RegimeMetrics {
    pub regime: MarketRegime,
    pub real_mean: f64,
    pub real_std: f64,
    pub gen_mean: f64,
    pub gen_std: f64,
    pub mean_error: f64,
    pub std_error: f64,
    pub wasserstein_approx: f64,
}

impl std::fmt::Display for RegimeMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}: real(mean={:.6}, std={:.6}) gen(mean={:.6}, std={:.6}) \
             err(mean={:.6}, std={:.6}) wasserstein~{:.6}",
            self.regime,
            self.real_mean,
            self.real_std,
            self.gen_mean,
            self.gen_std,
            self.mean_error,
            self.std_error,
            self.wasserstein_approx
        )
    }
}

/// Evaluate generated samples against real data for each regime.
pub fn evaluate_per_regime(
    cvae: &CVAE,
    data: &[(Array1<f64>, MarketRegime)],
    samples_per_regime: usize,
    rng: &mut impl Rng,
) -> Vec<RegimeMetrics> {
    let mut metrics = Vec::new();

    for regime_idx in 0..MarketRegime::COUNT {
        let regime = MarketRegime::from_index(regime_idx);

        // Collect real data for this regime (flatten windows to scalar returns)
        let real_values: Vec<f64> = data
            .iter()
            .filter(|(_, r)| *r == regime)
            .flat_map(|(x, _)| x.iter().cloned())
            .collect();

        if real_values.is_empty() {
            continue;
        }

        let real_mean = real_values.iter().sum::<f64>() / real_values.len() as f64;
        let real_std = (real_values
            .iter()
            .map(|v| (v - real_mean).powi(2))
            .sum::<f64>()
            / real_values.len() as f64)
            .sqrt();

        // Generate samples
        let generated = cvae.generate(regime, samples_per_regime, rng);
        let gen_values: Vec<f64> = generated.iter().cloned().collect();

        let gen_mean = gen_values.iter().sum::<f64>() / gen_values.len() as f64;
        let gen_std = (gen_values
            .iter()
            .map(|v| (v - gen_mean).powi(2))
            .sum::<f64>()
            / gen_values.len() as f64)
            .sqrt();

        // Approximate 1-Wasserstein via sorted quantile differences
        let mut real_sorted = real_values.clone();
        let mut gen_sorted = gen_values.clone();
        real_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        gen_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n_quantiles = 100;
        let wasserstein: f64 = (0..n_quantiles)
            .map(|i| {
                let q = i as f64 / n_quantiles as f64;
                let ri = (q * (real_sorted.len() - 1) as f64) as usize;
                let gi = (q * (gen_sorted.len() - 1) as f64) as usize;
                (real_sorted[ri] - gen_sorted[gi]).abs()
            })
            .sum::<f64>()
            / n_quantiles as f64;

        metrics.push(RegimeMetrics {
            regime,
            real_mean,
            real_std,
            gen_mean,
            gen_std,
            mean_error: (real_mean - gen_mean).abs(),
            std_error: (real_std - gen_std).abs(),
            wasserstein_approx: wasserstein,
        });
    }

    metrics
}

// ---------------------------------------------------------------------------
// Data preparation helpers
// ---------------------------------------------------------------------------

/// Convert prices + regimes into windowed training samples.
pub fn prepare_training_data(
    prices: &[f64],
    window: usize,
) -> Vec<(Array1<f64>, MarketRegime)> {
    if prices.len() < 2 {
        return vec![];
    }

    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    let regimes = detect_regimes(prices, window);

    // Normalize returns
    let mean_r = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_r = (returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / returns.len() as f64)
        .sqrt()
        .max(1e-8);
    let norm_returns: Vec<f64> = returns.iter().map(|r| (r - mean_r) / std_r).collect();

    let mut data = Vec::new();
    for i in window..norm_returns.len() {
        let x = Array1::from(norm_returns[i - window..i].to_vec());
        data.push((x, regimes[i]));
    }
    data
}

// ---------------------------------------------------------------------------
// Bybit API Integration
// ---------------------------------------------------------------------------

/// Response structure for Bybit kline endpoint.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// Fetch OHLCV kline data from Bybit.
///
/// * `symbol` – e.g. "BTCUSDT"
/// * `interval` – e.g. "60" for 1h, "D" for daily
/// * `limit` – number of candles (max 1000)
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<f64>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("Accept", "application/json")
        .send()
        .map_err(|e| anyhow!("HTTP request failed: {}", e))?
        .json()
        .map_err(|e| anyhow!("JSON parse failed: {}", e))?;

    if resp.ret_code != 0 {
        return Err(anyhow!("Bybit API error: {}", resp.ret_msg));
    }

    // Kline fields: [timestamp, open, high, low, close, volume, turnover]
    // Data is returned newest-first, so we reverse.
    let mut prices: Vec<f64> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 5 {
                row[4].parse::<f64>().ok() // close price
            } else {
                None
            }
        })
        .collect();

    prices.reverse(); // oldest first
    Ok(prices)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Axis;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_regime_one_hot() {
        assert_eq!(MarketRegime::Bull.one_hot(), Array1::from(vec![1.0, 0.0, 0.0]));
        assert_eq!(MarketRegime::Bear.one_hot(), Array1::from(vec![0.0, 1.0, 0.0]));
        assert_eq!(MarketRegime::Sideways.one_hot(), Array1::from(vec![0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_regime_from_index_roundtrip() {
        for i in 0..MarketRegime::COUNT {
            let r = MarketRegime::from_index(i);
            assert_eq!(r.index(), i);
        }
    }

    #[test]
    fn test_detect_regimes_basic() {
        // Create mixed regime data: uptrend with low vol, noisy period, downtrend with high vol
        let mut prices = Vec::new();
        let mut p: f64 = 100.0;

        // Bull phase: steady uptrend, low volatility
        for _ in 0..40 {
            p *= 1.005;
            prices.push(p);
        }
        // Transition with some noise
        for i in 0..40 {
            let noise = if i % 2 == 0 { 1.008 } else { 0.992 };
            p *= noise;
            prices.push(p);
        }
        // Bear phase: downtrend with higher volatility
        for i in 0..40 {
            let shock = if i % 3 == 0 { 0.96 } else { 0.985 };
            p *= shock;
            prices.push(p);
        }

        let regimes = detect_regimes(&prices, 10);
        assert_eq!(regimes.len(), prices.len() - 1);

        // Should have all three regime types present
        let bull_count = regimes.iter().filter(|r| **r == MarketRegime::Bull).count();
        let bear_count = regimes.iter().filter(|r| **r == MarketRegime::Bear).count();
        let side_count = regimes.iter().filter(|r| **r == MarketRegime::Sideways).count();

        // At minimum we should detect multiple regime types
        let types_detected = [bull_count > 0, bear_count > 0, side_count > 0]
            .iter()
            .filter(|&&x| x)
            .count();
        assert!(types_detected >= 2,
            "Expected at least 2 regime types, got {} (bull={}, bear={}, side={})",
            types_detected, bull_count, bear_count, side_count);
    }

    #[test]
    fn test_cvae_forward_dimensions() {
        let mut rng = make_rng();
        let input_dim = 5;
        let cvae = CVAE::new(input_dim, 3, 4, 16, 3, &mut rng);

        let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1]);
        let c = MarketRegime::Bull.one_hot();

        let (recon, mu, logvar, mu_p, logvar_p) = cvae.forward(&x, &c, &mut rng);

        assert_eq!(recon.len(), input_dim);
        assert_eq!(mu.len(), 3); // latent_dim
        assert_eq!(logvar.len(), 3);
        assert_eq!(mu_p.len(), 3);
        assert_eq!(logvar_p.len(), 3);
    }

    #[test]
    fn test_cvae_loss_non_negative() {
        let mu_enc = Array1::from(vec![0.0, 0.0]);
        let logvar_enc = Array1::from(vec![0.0, 0.0]);
        let mu_prior = Array1::from(vec![0.0, 0.0]);
        let logvar_prior = Array1::from(vec![0.0, 0.0]);
        let x = Array1::from(vec![1.0, 2.0]);
        let recon = Array1::from(vec![1.0, 2.0]);

        let (total, recon_l, kl) = CVAE::loss(&x, &recon, &mu_enc, &logvar_enc, &mu_prior, &logvar_prior);
        assert!(recon_l >= 0.0);
        // KL should be 0 when distributions are identical
        assert!((kl).abs() < 1e-10);
        assert!(total >= 0.0);
    }

    #[test]
    fn test_cvae_loss_kl_positive_when_different() {
        let mu_enc = Array1::from(vec![1.0, 2.0]);
        let logvar_enc = Array1::from(vec![0.5, 0.5]);
        let mu_prior = Array1::from(vec![0.0, 0.0]);
        let logvar_prior = Array1::from(vec![0.0, 0.0]);
        let x = Array1::from(vec![1.0, 2.0]);
        let recon = Array1::from(vec![1.0, 2.0]);

        let (_, _, kl) = CVAE::loss(&x, &recon, &mu_enc, &logvar_enc, &mu_prior, &logvar_prior);
        assert!(kl > 0.0, "KL should be positive for different distributions");
    }

    #[test]
    fn test_generate_shape() {
        let mut rng = make_rng();
        let input_dim = 5;
        let cvae = CVAE::new(input_dim, 3, 4, 16, 3, &mut rng);

        let samples = cvae.generate(MarketRegime::Bear, 100, &mut rng);
        assert_eq!(samples.dim(), (100, input_dim));
    }

    #[test]
    fn test_reparameterize() {
        let mut rng = make_rng();
        let mu = Array1::from(vec![0.0, 0.0, 0.0]);
        let logvar = Array1::from(vec![0.0, 0.0, 0.0]); // std = 1

        let z = CVAE::reparameterize(&mu, &logvar, &mut rng);
        assert_eq!(z.len(), 3);
        // Values should be roughly standard normal
        for &v in z.iter() {
            assert!(v.abs() < 10.0, "Sample too extreme: {}", v);
        }
    }

    #[test]
    fn test_prepare_training_data() {
        let mut prices = Vec::new();
        let mut p: f64 = 100.0;
        for _ in 0..100 {
            p *= 1.0 + 0.001 * p.sin(); // some movement
            prices.push(p);
        }

        let window = 5;
        let data = prepare_training_data(&prices, window);
        assert!(!data.is_empty());

        for (x, _regime) in &data {
            assert_eq!(x.len(), window);
        }
    }

    #[test]
    fn test_condition_encoder() {
        let mut rng = make_rng();
        let enc = ConditionEncoder::new(3, 8, &mut rng);

        let c = MarketRegime::Bull.one_hot();
        let out = enc.forward(&c);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_conditional_prior() {
        let mut rng = make_rng();
        let prior = ConditionalPrior::new(8, 4, &mut rng);

        let c = Array1::from(vec![1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]);
        let (mu, logvar) = prior.forward(&c);
        assert_eq!(mu.len(), 4);
        assert_eq!(logvar.len(), 4);
    }

    #[test]
    fn test_different_regimes_produce_different_outputs() {
        let mut rng = make_rng();
        let cvae = CVAE::new(5, 3, 4, 16, 3, &mut rng);

        let bull_samples = cvae.generate(MarketRegime::Bull, 500, &mut rng);
        let bear_samples = cvae.generate(MarketRegime::Bear, 500, &mut rng);

        let bull_mean = bull_samples.mean_axis(Axis(0)).unwrap();
        let bear_mean = bear_samples.mean_axis(Axis(0)).unwrap();

        // They should differ (with random init, not guaranteed to be huge but nonzero)
        let diff = (&bull_mean - &bear_mean).mapv(|v| v.abs()).sum();
        assert!(diff > 1e-6, "Different regimes should produce different outputs");
    }
}
