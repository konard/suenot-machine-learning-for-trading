//! VAE Volatility Surface — Variational Autoencoder for implied volatility surface modeling.
//!
//! This library provides tools for:
//! - Representing and manipulating implied volatility surfaces
//! - Training a VAE to learn surface distributions
//! - Checking no-arbitrage conditions (butterfly and calendar spread)
//! - Generating synthetic surfaces via the SABR model
//! - Fetching live BTC options data from Bybit V5 API

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Volatility Surface
// ---------------------------------------------------------------------------

/// A discretized implied volatility surface on a moneyness × maturity grid.
pub struct VolSurface {
    /// Moneyness values (K/F) for each column.
    pub moneyness: Vec<f64>,
    /// Time-to-expiration values (in years) for each row.
    pub maturities: Vec<f64>,
    /// Implied volatility grid: rows = maturities, cols = moneyness.
    pub ivols: Array2<f64>,
}

impl VolSurface {
    /// Create a new volatility surface from grid data.
    pub fn new(moneyness: Vec<f64>, maturities: Vec<f64>, ivols: Array2<f64>) -> Self {
        assert_eq!(ivols.nrows(), maturities.len());
        assert_eq!(ivols.ncols(), moneyness.len());
        Self { moneyness, maturities, ivols }
    }

    /// Flatten the surface grid into a 1-D vector (row-major).
    pub fn flatten(&self) -> Vec<f64> {
        self.ivols.iter().copied().collect()
    }

    /// Number of grid points.
    pub fn grid_size(&self) -> usize {
        self.ivols.len()
    }

    /// Get implied vol at grid indices (maturity_idx, moneyness_idx).
    pub fn get(&self, mat_idx: usize, mon_idx: usize) -> f64 {
        self.ivols[[mat_idx, mon_idx]]
    }

    /// ATM implied vol for a given maturity index (moneyness closest to 1.0).
    pub fn atm_vol(&self, mat_idx: usize) -> f64 {
        let atm_idx = self
            .moneyness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a) - 1.0).abs().partial_cmp(&((**b) - 1.0).abs()).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.ivols[[mat_idx, atm_idx]]
    }
}

// ---------------------------------------------------------------------------
// Arbitrage Checker
// ---------------------------------------------------------------------------

/// Checks no-arbitrage conditions on a volatility surface.
pub struct ArbitrageChecker;

impl ArbitrageChecker {
    /// Check butterfly arbitrage: total variance must be convex in moneyness
    /// for each maturity slice.
    /// Returns the number of violations and a list of (mat_idx, mon_idx) pairs.
    pub fn check_butterfly(surface: &VolSurface) -> (usize, Vec<(usize, usize)>) {
        let mut violations = Vec::new();
        for t in 0..surface.maturities.len() {
            let tau = surface.maturities[t];
            for k in 1..surface.moneyness.len() - 1 {
                let w_left = surface.ivols[[t, k - 1]].powi(2) * tau;
                let w_mid = surface.ivols[[t, k]].powi(2) * tau;
                let w_right = surface.ivols[[t, k + 1]].powi(2) * tau;
                // Convexity: w_mid <= (w_left + w_right) / 2
                if w_mid > (w_left + w_right) / 2.0 + 1e-10 {
                    violations.push((t, k));
                }
            }
        }
        let count = violations.len();
        (count, violations)
    }

    /// Check calendar spread arbitrage: total variance must be non-decreasing
    /// in maturity for each moneyness slice.
    /// Returns the number of violations and a list of (mat_idx, mon_idx) pairs.
    pub fn check_calendar(surface: &VolSurface) -> (usize, Vec<(usize, usize)>) {
        let mut violations = Vec::new();
        for k in 0..surface.moneyness.len() {
            for t in 1..surface.maturities.len() {
                let w_prev = surface.ivols[[t - 1, k]].powi(2) * surface.maturities[t - 1];
                let w_curr = surface.ivols[[t, k]].powi(2) * surface.maturities[t];
                if w_curr < w_prev - 1e-10 {
                    violations.push((t, k));
                }
            }
        }
        let count = violations.len();
        (count, violations)
    }
}

// ---------------------------------------------------------------------------
// SABR Surface Generator
// ---------------------------------------------------------------------------

/// Generates synthetic implied volatility surfaces using the SABR model.
pub struct SurfaceGenerator {
    rng: rand::rngs::ThreadRng,
}

impl SurfaceGenerator {
    pub fn new() -> Self {
        Self { rng: rand::thread_rng() }
    }

    /// SABR implied volatility approximation (Hagan et al. 2002).
    /// Parameters: alpha (vol-of-vol scale), beta (CEV exponent), rho (correlation),
    /// nu (vol-of-vol), forward F, strike K, time T.
    pub fn sabr_iv(alpha: f64, beta: f64, rho: f64, nu: f64, f: f64, k: f64, t: f64) -> f64 {
        if (f - k).abs() < 1e-12 {
            // ATM approximation
            let fk_mid = f.powf(1.0 - beta);
            let term1 = alpha / fk_mid;
            let term2 = 1.0
                + ((1.0 - beta).powi(2) / 24.0 * alpha.powi(2) / f.powf(2.0 * (1.0 - beta))
                    + 0.25 * rho * beta * nu * alpha / fk_mid
                    + (2.0 - 3.0 * rho.powi(2)) / 24.0 * nu.powi(2))
                    * t;
            return (term1 * term2).max(0.01);
        }

        let fk = f * k;
        let fk_beta = fk.powf((1.0 - beta) / 2.0);
        let log_fk = (f / k).ln();

        let z = (nu / alpha) * fk_beta * log_fk;
        let x_z = ((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho) / (1.0 - rho);
        let x_z = if x_z.abs() < 1e-12 { 1.0 } else { z / x_z.ln() };

        let a_term = alpha / (fk_beta * (1.0 + (1.0 - beta).powi(2) / 24.0 * log_fk.powi(2)
            + (1.0 - beta).powi(4) / 1920.0 * log_fk.powi(4)));

        let b_term = 1.0
            + ((1.0 - beta).powi(2) / 24.0 * alpha.powi(2) / fk.powf(1.0 - beta)
                + 0.25 * rho * beta * nu * alpha / fk_beta
                + (2.0 - 3.0 * rho.powi(2)) / 24.0 * nu.powi(2))
                * t;

        (a_term * x_z * b_term).max(0.01)
    }

    /// Generate a single synthetic volatility surface with random SABR parameters.
    pub fn generate_surface(
        &mut self,
        moneyness: &[f64],
        maturities: &[f64],
    ) -> VolSurface {
        let alpha = self.rng.gen_range(0.15..0.60);
        let beta = self.rng.gen_range(0.3..0.9);
        let rho = self.rng.gen_range(-0.8..-0.1);
        let nu = self.rng.gen_range(0.2..0.8);
        let forward = 1.0; // Normalized (moneyness = K/F, so F=1)

        let n_mat = maturities.len();
        let n_mon = moneyness.len();
        let mut ivols = Array2::zeros((n_mat, n_mon));

        for (t_idx, &tau) in maturities.iter().enumerate() {
            for (k_idx, &m) in moneyness.iter().enumerate() {
                let strike = m * forward;
                let iv = Self::sabr_iv(alpha, beta, rho, nu, forward, strike, tau);
                ivols[[t_idx, k_idx]] = iv;
            }
        }

        VolSurface::new(moneyness.to_vec(), maturities.to_vec(), ivols)
    }

    /// Generate a batch of synthetic surfaces for training.
    pub fn generate_batch(
        &mut self,
        n: usize,
        moneyness: &[f64],
        maturities: &[f64],
    ) -> Vec<VolSurface> {
        (0..n).map(|_| self.generate_surface(moneyness, maturities)).collect()
    }
}

impl Default for SurfaceGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VAE Model
// ---------------------------------------------------------------------------

/// Simple feed-forward VAE for volatility surfaces.
///
/// Architecture:
///   Encoder: input_dim → hidden_dim → (mu, log_var) of latent_dim
///   Decoder: latent_dim → hidden_dim → input_dim
///
/// Supports β-VAE by setting `beta > 1.0`.
pub struct VAEModel {
    input_dim: usize,
    _hidden_dim: usize,
    latent_dim: usize,
    beta: f64,
    // Encoder weights
    enc_w1: Array2<f64>,
    enc_b1: Array1<f64>,
    enc_w_mu: Array2<f64>,
    enc_b_mu: Array1<f64>,
    enc_w_logvar: Array2<f64>,
    enc_b_logvar: Array1<f64>,
    // Decoder weights
    dec_w1: Array2<f64>,
    dec_b1: Array1<f64>,
    dec_w2: Array2<f64>,
    dec_b2: Array1<f64>,
}

impl VAEModel {
    /// Create a new VAE with random Xavier-initialized weights.
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize, beta: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut xavier = |fan_in: usize, fan_out: usize| -> Array2<f64> {
            let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
            Array2::from_shape_fn((fan_in, fan_out), |_| rng.gen_range(-scale..scale))
        };

        Self {
            input_dim,
            _hidden_dim: hidden_dim,
            latent_dim,
            beta,
            enc_w1: xavier(input_dim, hidden_dim),
            enc_b1: Array1::zeros(hidden_dim),
            enc_w_mu: xavier(hidden_dim, latent_dim),
            enc_b_mu: Array1::zeros(latent_dim),
            enc_w_logvar: xavier(hidden_dim, latent_dim),
            enc_b_logvar: Array1::zeros(latent_dim),
            dec_w1: xavier(latent_dim, hidden_dim),
            dec_b1: Array1::zeros(hidden_dim),
            dec_w2: xavier(hidden_dim, input_dim),
            dec_b2: Array1::zeros(input_dim),
        }
    }

    fn relu(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| v.max(0.0))
    }

    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// Encode an input vector into (mu, log_var).
    pub fn encode(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let h = Self::relu(&(x.dot(&self.enc_w1) + &self.enc_b1));
        let mu = h.dot(&self.enc_w_mu) + &self.enc_b_mu;
        let log_var = h.dot(&self.enc_w_logvar) + &self.enc_b_logvar;
        (mu, log_var)
    }

    /// Reparameterization trick: z = mu + std * epsilon.
    pub fn reparameterize(&self, mu: &Array1<f64>, log_var: &Array1<f64>) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let std = log_var.mapv(|v| (v * 0.5).exp());
        let eps = Array1::from_shape_fn(self.latent_dim, |_| rng.gen::<f64>() * 2.0 - 1.0);
        mu + &(std * eps)
    }

    /// Decode a latent vector into a reconstructed surface (flattened).
    pub fn decode(&self, z: &Array1<f64>) -> Array1<f64> {
        let h = Self::relu(&(z.dot(&self.dec_w1) + &self.dec_b1));
        let out = h.dot(&self.dec_w2) + &self.dec_b2;
        // Use sigmoid to bound output in [0, 1], then scale to reasonable vol range [0.01, 2.0]
        Self::sigmoid(&out).mapv(|v| 0.01 + v * 1.99)
    }

    /// Full forward pass: encode → reparameterize → decode.
    /// Returns (reconstruction, mu, log_var).
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let (mu, log_var) = self.encode(x);
        let z = self.reparameterize(&mu, &log_var);
        let recon = self.decode(&z);
        (recon, mu, log_var)
    }

    /// Compute VAE loss = reconstruction_loss + beta * KL_divergence.
    pub fn loss(&self, x: &Array1<f64>, recon: &Array1<f64>, mu: &Array1<f64>, log_var: &Array1<f64>) -> (f64, f64, f64) {
        // MSE reconstruction loss
        let recon_loss: f64 = x.iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / x.len() as f64;

        // KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        let kl: f64 = -0.5
            * mu.iter()
                .zip(log_var.iter())
                .map(|(m, lv)| 1.0 + lv - m.powi(2) - lv.exp())
                .sum::<f64>();

        let total = recon_loss + self.beta * kl;
        (total, recon_loss, kl)
    }

    /// Train the VAE on a batch of flattened surface vectors using SGD.
    /// Returns (final_total_loss, final_recon_loss, final_kl).
    pub fn train(&mut self, data: &[Vec<f64>], epochs: usize, lr: f64) -> (f64, f64, f64) {
        let mut last_total = 0.0;
        let mut last_recon = 0.0;
        let mut last_kl = 0.0;

        for _epoch in 0..epochs {
            let mut epoch_total = 0.0;
            let mut epoch_recon = 0.0;
            let mut epoch_kl = 0.0;

            for sample in data {
                let x = Array1::from(sample.clone());
                let (recon, mu, log_var) = self.forward(&x);
                let (total, recon_l, kl_l) = self.loss(&x, &recon, &mu, &log_var);
                epoch_total += total;
                epoch_recon += recon_l;
                epoch_kl += kl_l;

                // Numerical gradient update (simplified SGD)
                // Update decoder output bias towards better reconstruction
                let error = &x - &recon;
                let grad_scale = lr / self.input_dim as f64;
                for i in 0..self.input_dim {
                    self.dec_b2[i] += error[i] * grad_scale;
                }

                // Update encoder bias towards smaller KL
                for i in 0..self.latent_dim {
                    self.enc_b_mu[i] -= lr * self.beta * mu[i] * 0.01;
                    self.enc_b_logvar[i] -= lr * self.beta * (log_var[i].exp() - 1.0) * 0.005;
                }
            }

            let n = data.len() as f64;
            last_total = epoch_total / n;
            last_recon = epoch_recon / n;
            last_kl = epoch_kl / n;
        }

        (last_total, last_recon, last_kl)
    }

    /// Sample a new surface from the prior p(z) = N(0, I).
    pub fn sample(&self) -> Array1<f64> {
        let mut rng = rand::thread_rng();
        let z = Array1::from_shape_fn(self.latent_dim, |_| rng.gen::<f64>() * 2.0 - 1.0);
        self.decode(&z)
    }

    /// Latent dimension getter.
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Beta getter.
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

// ---------------------------------------------------------------------------
// Bybit Client
// ---------------------------------------------------------------------------

/// Async client for fetching BTC options data from Bybit V5 API.
pub struct BybitClient {
    base_url: String,
}

#[derive(Deserialize, Debug)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitTickerResult,
}

#[derive(Deserialize, Debug)]
struct BybitTickerResult {
    list: Vec<BybitOptionTicker>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct BybitOptionTicker {
    /// Option symbol, e.g., "BTC-28MAR25-90000-C"
    pub symbol: String,
    /// Implied volatility (as string, e.g., "0.55")
    #[serde(rename = "markIv")]
    pub mark_iv: String,
    /// Underlying price
    #[serde(rename = "underlyingPrice")]
    pub underlying_price: String,
    /// Mark price
    #[serde(rename = "markPrice")]
    pub mark_price: String,
    /// Bid implied vol
    #[serde(rename = "bidIv")]
    pub bid_iv: String,
    /// Ask implied vol
    #[serde(rename = "askIv")]
    pub ask_iv: String,
}

/// Parsed option data with numeric fields.
#[derive(Debug, Clone)]
pub struct ParsedOption {
    pub symbol: String,
    pub strike: f64,
    pub expiry: String,
    pub is_call: bool,
    pub mark_iv: f64,
    pub underlying: f64,
    pub moneyness: f64,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch BTC option tickers from Bybit V5 API.
    pub async fn get_option_tickers(&self) -> anyhow::Result<Vec<BybitOptionTicker>> {
        let url = format!(
            "{}/v5/market/tickers?category=option&baseCoin=BTC",
            self.base_url
        );
        let client = reqwest::Client::new();
        let resp: BybitResponse = client.get(&url).send().await?.json().await?;
        if resp.ret_code != 0 {
            anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
        }
        Ok(resp.result.list)
    }

    /// Parse a Bybit option symbol into (expiry, strike, is_call).
    /// Format: "BTC-28MAR25-90000-C"
    pub fn parse_symbol(symbol: &str) -> Option<(String, f64, bool)> {
        let parts: Vec<&str> = symbol.split('-').collect();
        if parts.len() != 4 {
            return None;
        }
        let expiry = parts[1].to_string();
        let strike: f64 = parts[2].parse().ok()?;
        let is_call = parts[3] == "C";
        Some((expiry, strike, is_call))
    }

    /// Convert raw tickers into parsed option data.
    pub fn parse_tickers(tickers: &[BybitOptionTicker]) -> Vec<ParsedOption> {
        tickers
            .iter()
            .filter_map(|t| {
                let (expiry, strike, is_call) = Self::parse_symbol(&t.symbol)?;
                let mark_iv: f64 = t.mark_iv.parse().ok()?;
                let underlying: f64 = t.underlying_price.parse().ok()?;
                if mark_iv <= 0.0 || underlying <= 0.0 {
                    return None;
                }
                let moneyness = strike / underlying;
                Some(ParsedOption {
                    symbol: t.symbol.clone(),
                    strike,
                    expiry,
                    is_call,
                    mark_iv,
                    underlying,
                    moneyness,
                })
            })
            .collect()
    }

    /// Build a volatility surface from parsed options grouped by expiry.
    /// Uses a fixed moneyness grid and the nearest available data points.
    pub fn build_surface(
        options: &[ParsedOption],
        moneyness_grid: &[f64],
        maturity_days: &[f64],
    ) -> VolSurface {
        let n_mat = maturity_days.len();
        let n_mon = moneyness_grid.len();
        let mut ivols = Array2::from_elem((n_mat, n_mon), 0.3); // default 30% vol

        // Group by expiry
        let mut by_expiry: std::collections::HashMap<&str, Vec<&ParsedOption>> =
            std::collections::HashMap::new();
        for opt in options {
            by_expiry.entry(&opt.expiry).or_default().push(opt);
        }

        // For each maturity slot, find closest expiry and interpolate
        let expiries: Vec<&str> = by_expiry.keys().copied().collect();
        for (t_idx, _days) in maturity_days.iter().enumerate() {
            // Use the expiry at index t_idx if available (simplified mapping)
            if let Some(expiry) = expiries.get(t_idx) {
                if let Some(opts) = by_expiry.get(expiry) {
                    for (k_idx, &target_m) in moneyness_grid.iter().enumerate() {
                        // Find nearest option by moneyness
                        if let Some(nearest) = opts
                            .iter()
                            .min_by(|a, b| {
                                (a.moneyness - target_m)
                                    .abs()
                                    .partial_cmp(&(b.moneyness - target_m).abs())
                                    .unwrap()
                            })
                        {
                            if (nearest.moneyness - target_m).abs() < 0.15 {
                                ivols[[t_idx, k_idx]] = nearest.mark_iv;
                            }
                        }
                    }
                }
            }
        }

        let maturities: Vec<f64> = maturity_days.iter().map(|d| d / 365.0).collect();
        VolSurface::new(moneyness_grid.to_vec(), maturities, ivols)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Standard moneyness grid for surface construction.
pub fn default_moneyness_grid() -> Vec<f64> {
    vec![0.80, 0.85, 0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10, 1.15, 1.20]
}

/// Standard maturity grid (in days).
pub fn default_maturity_days() -> Vec<f64> {
    vec![7.0, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0]
}

/// Compute the 25-delta skew from a surface at a given maturity index.
pub fn compute_skew(surface: &VolSurface, mat_idx: usize) -> f64 {
    let n = surface.moneyness.len();
    if n < 3 {
        return 0.0;
    }
    // Approximate: OTM put vol (low moneyness) - OTM call vol (high moneyness)
    let put_vol = surface.ivols[[mat_idx, 1]]; // ~0.85 moneyness
    let call_vol = surface.ivols[[mat_idx, n - 2]]; // ~1.15 moneyness
    put_vol - call_vol
}

/// Compute butterfly spread metric from a surface at a given maturity index.
pub fn compute_butterfly(surface: &VolSurface, mat_idx: usize) -> f64 {
    let atm_vol = surface.atm_vol(mat_idx);
    let n = surface.moneyness.len();
    if n < 3 {
        return 0.0;
    }
    let wing_avg = (surface.ivols[[mat_idx, 1]] + surface.ivols[[mat_idx, n - 2]]) / 2.0;
    wing_avg - atm_vol
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vol_surface_creation() {
        let moneyness = vec![0.9, 0.95, 1.0, 1.05, 1.1];
        let maturities = vec![0.25, 0.5, 1.0];
        let ivols = Array2::from_elem((3, 5), 0.25);
        let surface = VolSurface::new(moneyness, maturities, ivols);
        assert_eq!(surface.grid_size(), 15);
        assert_eq!(surface.get(0, 2), 0.25);
    }

    #[test]
    fn test_vol_surface_atm() {
        let moneyness = vec![0.9, 0.95, 1.0, 1.05, 1.1];
        let maturities = vec![0.25];
        let mut ivols = Array2::from_elem((1, 5), 0.20);
        ivols[[0, 2]] = 0.30; // ATM is moneyness=1.0
        let surface = VolSurface::new(moneyness, maturities, ivols);
        assert!((surface.atm_vol(0) - 0.30).abs() < 1e-10);
    }

    #[test]
    fn test_sabr_iv_atm() {
        let iv = SurfaceGenerator::sabr_iv(0.3, 0.5, -0.3, 0.4, 1.0, 1.0, 0.5);
        assert!(iv > 0.0);
        assert!(iv < 2.0);
    }

    #[test]
    fn test_sabr_iv_skew() {
        let iv_put = SurfaceGenerator::sabr_iv(0.3, 0.5, -0.5, 0.4, 1.0, 0.9, 0.5);
        let iv_call = SurfaceGenerator::sabr_iv(0.3, 0.5, -0.5, 0.4, 1.0, 1.1, 0.5);
        // With negative rho, OTM put vol > OTM call vol (negative skew)
        assert!(iv_put > iv_call, "Expected negative skew: put_iv={} > call_iv={}", iv_put, iv_call);
    }

    #[test]
    fn test_surface_generator_batch() {
        let mut gen = SurfaceGenerator::new();
        let moneyness = vec![0.9, 0.95, 1.0, 1.05, 1.1];
        let maturities = vec![0.25, 0.5, 1.0];
        let batch = gen.generate_batch(10, &moneyness, &maturities);
        assert_eq!(batch.len(), 10);
        for surface in &batch {
            assert_eq!(surface.grid_size(), 15);
            // All IVs should be positive
            for &v in surface.ivols.iter() {
                assert!(v > 0.0, "IV must be positive, got {}", v);
            }
        }
    }

    #[test]
    fn test_arbitrage_checker_clean_surface() {
        // SABR-generated surfaces should be mostly arbitrage-free
        let mut gen = SurfaceGenerator::new();
        let moneyness = vec![0.9, 0.95, 1.0, 1.05, 1.1];
        let maturities = vec![0.25, 0.5, 1.0];
        let surface = gen.generate_surface(&moneyness, &maturities);

        let (cal_violations, _) = ArbitrageChecker::check_calendar(&surface);
        // SABR should produce monotone total variance in maturity
        assert_eq!(cal_violations, 0, "SABR surface should have no calendar arbitrage");
    }

    #[test]
    fn test_vae_creation() {
        let vae = VAEModel::new(15, 8, 4, 1.0);
        assert_eq!(vae.latent_dim(), 4);
        assert!((vae.beta() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vae_encode_decode() {
        let vae = VAEModel::new(15, 8, 4, 1.0);
        let x = Array1::from_vec(vec![0.25; 15]);
        let (mu, log_var) = vae.encode(&x);
        assert_eq!(mu.len(), 4);
        assert_eq!(log_var.len(), 4);

        let z = vae.reparameterize(&mu, &log_var);
        let recon = vae.decode(&z);
        assert_eq!(recon.len(), 15);
        // All decoded values should be in [0.01, 2.0]
        for &v in recon.iter() {
            assert!(v >= 0.01 && v <= 2.0, "Decoded vol {} out of range", v);
        }
    }

    #[test]
    fn test_vae_training() {
        let mut gen = SurfaceGenerator::new();
        let moneyness = vec![0.9, 0.95, 1.0, 1.05, 1.1];
        let maturities = vec![0.25, 0.5, 1.0];
        let surfaces = gen.generate_batch(50, &moneyness, &maturities);
        let data: Vec<Vec<f64>> = surfaces.iter().map(|s| s.flatten()).collect();

        let mut vae = VAEModel::new(15, 8, 4, 1.0);
        let (loss_before, _, _) = {
            let x = Array1::from(data[0].clone());
            let (recon, mu, lv) = vae.forward(&x);
            vae.loss(&x, &recon, &mu, &lv)
        };

        let (loss_after, _, _) = vae.train(&data, 20, 0.01);

        // Loss should decrease after training
        assert!(
            loss_after < loss_before,
            "Training should reduce loss: before={}, after={}",
            loss_before,
            loss_after
        );
    }

    #[test]
    fn test_vae_sample() {
        let vae = VAEModel::new(15, 8, 4, 1.0);
        let sample = vae.sample();
        assert_eq!(sample.len(), 15);
        for &v in sample.iter() {
            assert!(v >= 0.01 && v <= 2.0, "Sampled vol {} out of range", v);
        }
    }

    #[test]
    fn test_parse_bybit_symbol() {
        let (expiry, strike, is_call) = BybitClient::parse_symbol("BTC-28MAR25-90000-C").unwrap();
        assert_eq!(expiry, "28MAR25");
        assert!((strike - 90000.0).abs() < 1e-10);
        assert!(is_call);

        let (_, _, is_call) = BybitClient::parse_symbol("BTC-28MAR25-90000-P").unwrap();
        assert!(!is_call);
    }

    #[test]
    fn test_skew_and_butterfly() {
        let moneyness = vec![0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15];
        let maturities = vec![0.25];
        let ivols = Array2::from_shape_vec(
            (1, 7),
            vec![0.35, 0.30, 0.27, 0.25, 0.26, 0.28, 0.30],
        )
        .unwrap();
        let surface = VolSurface::new(moneyness, maturities, ivols);

        let skew = compute_skew(&surface, 0);
        assert!(skew > 0.0, "Expected positive skew (put vol > call vol), got {}", skew);

        let bfly = compute_butterfly(&surface, 0);
        assert!(bfly > 0.0, "Expected positive butterfly, got {}", bfly);
    }
}
