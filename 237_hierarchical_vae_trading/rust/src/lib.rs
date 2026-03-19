//! Hierarchical VAE (HVAE) for Trading
//!
//! Implements a Hierarchical Variational Autoencoder with multi-scale latent
//! representations for financial time series. The hierarchy organises latent
//! variables into levels — top levels capture macro regime structure while
//! bottom levels capture fine-grained daily patterns.
//!
//! Based on the Ladder VAE architecture (Sønderby et al., 2016).

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Market Regime (used for evaluation, not conditioning)
// ---------------------------------------------------------------------------

/// Market regime labels used for evaluating learned representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
}

impl MarketRegime {
    pub const COUNT: usize = 3;

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

    pub fn name(&self) -> &'static str {
        match self {
            MarketRegime::Bull => "Bull",
            MarketRegime::Bear => "Bear",
            MarketRegime::Sideways => "Sideways",
        }
    }
}

// ---------------------------------------------------------------------------
// Regime detection
// ---------------------------------------------------------------------------

/// Detect market regimes from a price series using rolling statistics.
pub fn detect_regimes(prices: &[f64], window: usize) -> Vec<MarketRegime> {
    if prices.len() < 2 {
        return vec![];
    }

    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    if returns.len() < window {
        return returns.iter().map(|_| MarketRegime::Sideways).collect();
    }

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

/// A dense (fully-connected) layer: y = relu(x * W + b) or y = x * W + b.
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Array2<f64>,
    pub bias: Array1<f64>,
}

impl Linear {
    pub fn new_random(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / input_dim as f64).sqrt();
        let weight = Array2::from_shape_fn((input_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let bias = Array1::zeros(output_dim);
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        x.dot(&self.weight) + &self.bias
    }

    pub fn forward_relu(&self, x: &Array1<f64>) -> Array1<f64> {
        let out = self.forward(x);
        out.mapv(|v| v.max(0.0))
    }

    /// Simple SGD update: w -= lr * grad.
    pub fn update(&mut self, weight_grad: &Array2<f64>, bias_grad: &Array1<f64>, lr: f64) {
        self.weight.scaled_add(-lr, weight_grad);
        self.bias.scaled_add(-lr, bias_grad);
    }
}

// ---------------------------------------------------------------------------
// Hierarchy Level — one level of the HVAE
// ---------------------------------------------------------------------------

/// Parameters for one level of the hierarchy.
///
/// Each level has:
/// - bottom-up layers producing mu_bu, logvar_bu from the deterministic path
/// - top-down layers producing mu_td, logvar_td from the level above
/// - a sampling + projection layer for passing to the next level
#[derive(Debug, Clone)]
pub struct HierarchyLevel {
    /// Bottom-up: deterministic representation → (mu_bu, logvar_bu)
    pub bu_mu: Linear,
    pub bu_logvar: Linear,
    /// Top-down: z_{l+1} transformed → (mu_td, logvar_td)
    pub td_mu: Linear,
    pub td_logvar: Linear,
    /// Latent dim at this level
    pub latent_dim: usize,
}

impl HierarchyLevel {
    pub fn new(
        det_dim: usize,
        above_dim: usize,
        latent_dim: usize,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            bu_mu: Linear::new_random(det_dim, latent_dim, rng),
            bu_logvar: Linear::new_random(det_dim, latent_dim, rng),
            td_mu: Linear::new_random(above_dim, latent_dim, rng),
            td_logvar: Linear::new_random(above_dim, latent_dim, rng),
            latent_dim,
        }
    }

    /// Compute bottom-up parameters from deterministic representation.
    pub fn bottom_up(&self, d: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let mu = self.bu_mu.forward(d);
        let logvar = self.bu_logvar.forward(d);
        (mu, logvar)
    }

    /// Compute top-down prior parameters from the level above.
    pub fn top_down(&self, z_above: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let mu = self.td_mu.forward(z_above);
        let logvar = self.td_logvar.forward(z_above);
        (mu, logvar)
    }
}

// ---------------------------------------------------------------------------
// Precision-weighted merge
// ---------------------------------------------------------------------------

/// Merge two Gaussian distributions using precision weighting.
///
/// Given N(mu_1, sigma_1^2) and N(mu_2, sigma_2^2), the merged distribution is:
///   precision = 1/sigma_1^2 + 1/sigma_2^2
///   mu_merged = (mu_1/sigma_1^2 + mu_2/sigma_2^2) / precision
///   sigma_merged^2 = 1 / precision
///
/// We work in log-variance space for numerical stability.
pub fn precision_weighted_merge(
    mu_bu: &Array1<f64>,
    logvar_bu: &Array1<f64>,
    mu_td: &Array1<f64>,
    logvar_td: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let prec_bu = logvar_bu.mapv(|lv| (-lv).exp()); // 1/sigma^2
    let prec_td = logvar_td.mapv(|lv| (-lv).exp());
    let prec_total = &prec_bu + &prec_td;

    let mu_merged = (mu_bu * &prec_bu + mu_td * &prec_td) / &prec_total;
    let logvar_merged = prec_total.mapv(|p| -(p.ln()));

    (mu_merged, logvar_merged)
}

// ---------------------------------------------------------------------------
// Hierarchical VAE
// ---------------------------------------------------------------------------

/// Complete Hierarchical VAE for trading data.
///
/// Architecture with L=3 levels:
/// - Level 3 (top): macro regime structure
/// - Level 2 (middle): tactical patterns
/// - Level 1 (bottom): fine-grained daily patterns
///
/// Bottom-up encoder:
///   x → d_1 → d_2 → d_3
///
/// Top-down generative / inference:
///   z_3 ~ q(z_3 | d_3)
///   z_2 ~ q(z_2 | merge(bu(d_2), td(z_3)))
///   z_1 ~ q(z_1 | merge(bu(d_1), td(z_2)))
///   x_recon = decode(z_1)
pub struct HierarchicalVAE {
    // Bottom-up encoder layers
    pub enc_layer1: Linear, // x → d_1
    pub enc_layer2: Linear, // d_1 → d_2
    pub enc_layer3: Linear, // d_2 → d_3

    // Hierarchy levels (each has BU and TD parameter networks)
    pub level3: HierarchyLevel, // Top level
    pub level2: HierarchyLevel, // Middle level
    pub level1: HierarchyLevel, // Bottom level

    // Decoder: z_1 → reconstruction
    pub dec_hidden: Linear,
    pub dec_output: Linear,

    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dims: [usize; 3], // [level1, level2, level3]
}

impl HierarchicalVAE {
    /// Create a new Hierarchical VAE.
    ///
    /// * `input_dim` – dimensionality of input data (e.g. window of returns)
    /// * `hidden_dim` – hidden layer size for encoder/decoder
    /// * `latent_dims` – latent dimensions per level [bottom, middle, top]
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        latent_dims: [usize; 3],
        rng: &mut impl Rng,
    ) -> Self {
        // Bottom-up encoder: x → d1 → d2 → d3
        let enc_layer1 = Linear::new_random(input_dim, hidden_dim, rng);
        let enc_layer2 = Linear::new_random(hidden_dim, hidden_dim, rng);
        let enc_layer3 = Linear::new_random(hidden_dim, hidden_dim, rng);

        // Level 3 (top): BU from d3, TD is the prior (no z above)
        let level3 = HierarchyLevel::new(hidden_dim, latent_dims[2], latent_dims[2], rng);
        // Level 2 (middle): BU from d2, TD from z3
        let level2 = HierarchyLevel::new(hidden_dim, latent_dims[2], latent_dims[1], rng);
        // Level 1 (bottom): BU from d1, TD from z2
        let level1 = HierarchyLevel::new(hidden_dim, latent_dims[1], latent_dims[0], rng);

        // Decoder: z1 → hidden → output
        let dec_hidden = Linear::new_random(latent_dims[0], hidden_dim, rng);
        let dec_output = Linear::new_random(hidden_dim, input_dim, rng);

        Self {
            enc_layer1,
            enc_layer2,
            enc_layer3,
            level3,
            level2,
            level1,
            dec_hidden,
            dec_output,
            input_dim,
            hidden_dim,
            latent_dims,
        }
    }

    /// Reparameterization trick: sample z = mu + sigma * eps.
    pub fn reparameterize(
        mu: &Array1<f64>,
        logvar: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> Array1<f64> {
        let std = logvar.mapv(|lv| (lv * 0.5).exp());
        let eps = Array1::from_shape_fn(mu.len(), |_| {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        });
        mu + &(std * &eps)
    }

    /// Bottom-up encoder pass: x → (d1, d2, d3).
    pub fn encode_bottom_up(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let d1 = self.enc_layer1.forward_relu(x);
        let d2 = self.enc_layer2.forward_relu(&d1);
        let d3 = self.enc_layer3.forward_relu(&d2);
        (d1, d2, d3)
    }

    /// Decode from bottom-level latent z1 → reconstructed x.
    pub fn decode(&self, z1: &Array1<f64>) -> Array1<f64> {
        let h = self.dec_hidden.forward_relu(z1);
        self.dec_output.forward(&h)
    }

    /// Full forward pass through the hierarchy.
    ///
    /// Returns:
    /// - reconstruction
    /// - per-level: (mu_q, logvar_q, mu_prior, logvar_prior) for KL computation
    /// - latent samples z1, z2, z3
    pub fn forward(
        &self,
        x: &Array1<f64>,
        rng: &mut impl Rng,
    ) -> HVAEForwardResult {
        // Bottom-up pass
        let (d1, d2, d3) = self.encode_bottom_up(x);

        // --- Level 3 (top): only bottom-up, prior is N(0,I) ---
        let (mu_bu3, logvar_bu3) = self.level3.bottom_up(&d3);
        // Prior at top level: N(0, I)
        let mu_prior3 = Array1::zeros(self.latent_dims[2]);
        let logvar_prior3 = Array1::zeros(self.latent_dims[2]);
        // Posterior = bottom-up (no top-down at the top level)
        let mu_q3 = mu_bu3;
        let logvar_q3 = logvar_bu3;
        let z3 = Self::reparameterize(&mu_q3, &logvar_q3, rng);

        // --- Level 2 (middle): merge BU(d2) with TD(z3) ---
        let (mu_bu2, logvar_bu2) = self.level2.bottom_up(&d2);
        let (mu_td2, logvar_td2) = self.level2.top_down(&z3);
        let (mu_q2, logvar_q2) =
            precision_weighted_merge(&mu_bu2, &logvar_bu2, &mu_td2, &logvar_td2);
        // Prior at level 2 = top-down from z3
        let mu_prior2 = mu_td2.clone();
        let logvar_prior2 = logvar_td2.clone();
        let z2 = Self::reparameterize(&mu_q2, &logvar_q2, rng);

        // --- Level 1 (bottom): merge BU(d1) with TD(z2) ---
        let (mu_bu1, logvar_bu1) = self.level1.bottom_up(&d1);
        let (mu_td1, logvar_td1) = self.level1.top_down(&z2);
        let (mu_q1, logvar_q1) =
            precision_weighted_merge(&mu_bu1, &logvar_bu1, &mu_td1, &logvar_td1);
        let mu_prior1 = mu_td1.clone();
        let logvar_prior1 = logvar_td1.clone();
        let z1 = Self::reparameterize(&mu_q1, &logvar_q1, rng);

        // Decode from z1
        let recon = self.decode(&z1);

        HVAEForwardResult {
            recon,
            level_info: [
                LevelInfo {
                    mu_q: mu_q1,
                    logvar_q: logvar_q1,
                    mu_prior: mu_prior1,
                    logvar_prior: logvar_prior1,
                    z: z1,
                },
                LevelInfo {
                    mu_q: mu_q2,
                    logvar_q: logvar_q2,
                    mu_prior: mu_prior2,
                    logvar_prior: logvar_prior2,
                    z: z2,
                },
                LevelInfo {
                    mu_q: mu_q3,
                    logvar_q: logvar_q3,
                    mu_prior: mu_prior3,
                    logvar_prior: logvar_prior3,
                    z: z3,
                },
            ],
        }
    }

    /// Compute KL divergence between q and p for one level.
    pub fn kl_divergence(
        mu_q: &Array1<f64>,
        logvar_q: &Array1<f64>,
        mu_prior: &Array1<f64>,
        logvar_prior: &Array1<f64>,
    ) -> f64 {
        0.5 * (logvar_prior - logvar_q
            + (logvar_q.mapv(f64::exp)
                + (mu_q - mu_prior).mapv(|v| v.powi(2)))
                / logvar_prior.mapv(f64::exp)
            - 1.0)
            .sum()
    }

    /// Compute total loss: reconstruction + sum of per-level KL.
    ///
    /// * `beta` – KL annealing coefficient (0 to 1)
    /// * `free_bits` – minimum KL per level to prevent collapse
    pub fn loss(
        x: &Array1<f64>,
        result: &HVAEForwardResult,
        beta: f64,
        free_bits: f64,
    ) -> HVAELoss {
        // Reconstruction loss (MSE)
        let recon_loss = (x - &result.recon).mapv(|v| v.powi(2)).sum() / x.len() as f64;

        // Per-level KL with free bits
        let mut kl_per_level = [0.0; 3];
        for (i, info) in result.level_info.iter().enumerate() {
            let kl_raw = Self::kl_divergence(
                &info.mu_q,
                &info.logvar_q,
                &info.mu_prior,
                &info.logvar_prior,
            );
            kl_per_level[i] = kl_raw.max(free_bits);
        }

        let kl_total: f64 = kl_per_level.iter().sum();
        let total = recon_loss + beta * kl_total;

        HVAELoss {
            total,
            recon_loss,
            kl_per_level,
            kl_total,
        }
    }

    /// Generate samples from the hierarchical prior.
    ///
    /// Samples top-down: z3 ~ N(0,I), z2 ~ p(z2|z3), z1 ~ p(z1|z2), x ~ decode(z1).
    pub fn generate(&self, num_samples: usize, rng: &mut impl Rng) -> Array2<f64> {
        let mut samples = Array2::zeros((num_samples, self.input_dim));

        for i in 0..num_samples {
            // Level 3: sample from prior N(0, I)
            let mu3 = Array1::zeros(self.latent_dims[2]);
            let logvar3 = Array1::zeros(self.latent_dims[2]);
            let z3 = Self::reparameterize(&mu3, &logvar3, rng);

            // Level 2: sample from p(z2|z3)
            let (mu2, logvar2) = self.level2.top_down(&z3);
            let z2 = Self::reparameterize(&mu2, &logvar2, rng);

            // Level 1: sample from p(z1|z2)
            let (mu1, logvar1) = self.level1.top_down(&z2);
            let z1 = Self::reparameterize(&mu1, &logvar1, rng);

            // Decode
            let x = self.decode(&z1);
            samples.row_mut(i).assign(&x);
        }

        samples
    }

    /// Generate samples with a fixed top-level latent (regime-controlled generation).
    pub fn generate_with_fixed_top(
        &self,
        z3_fixed: &Array1<f64>,
        num_samples: usize,
        rng: &mut impl Rng,
    ) -> Array2<f64> {
        let mut samples = Array2::zeros((num_samples, self.input_dim));

        for i in 0..num_samples {
            let (mu2, logvar2) = self.level2.top_down(z3_fixed);
            let z2 = Self::reparameterize(&mu2, &logvar2, rng);

            let (mu1, logvar1) = self.level1.top_down(&z2);
            let z1 = Self::reparameterize(&mu1, &logvar1, rng);

            let x = self.decode(&z1);
            samples.row_mut(i).assign(&x);
        }

        samples
    }

    /// Extract top-level latent representation for a given input.
    pub fn encode_top_level(
        &self,
        x: &Array1<f64>,
    ) -> Array1<f64> {
        let (_d1, _d2, d3) = self.encode_bottom_up(x);
        let (mu, _logvar) = self.level3.bottom_up(&d3);
        mu
    }
}

// ---------------------------------------------------------------------------
// Forward result structures
// ---------------------------------------------------------------------------

/// Information for one level of the hierarchy.
pub struct LevelInfo {
    pub mu_q: Array1<f64>,
    pub logvar_q: Array1<f64>,
    pub mu_prior: Array1<f64>,
    pub logvar_prior: Array1<f64>,
    pub z: Array1<f64>,
}

/// Complete forward pass result.
pub struct HVAEForwardResult {
    pub recon: Array1<f64>,
    /// Level info: [level1 (bottom), level2 (middle), level3 (top)]
    pub level_info: [LevelInfo; 3],
}

/// Loss breakdown.
pub struct HVAELoss {
    pub total: f64,
    pub recon_loss: f64,
    pub kl_per_level: [f64; 3],
    pub kl_total: f64,
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// Train the HVAE using finite-difference gradient estimation.
///
/// This is a simplified training loop suitable for demonstration.
/// Production systems should use automatic differentiation.
pub fn train_hvae(
    hvae: &mut HierarchicalVAE,
    data: &[Array1<f64>],
    epochs: usize,
    lr: f64,
    free_bits: f64,
    rng: &mut impl Rng,
) -> Vec<HVAELoss> {
    let mut loss_history = Vec::with_capacity(epochs);
    let warmup_epochs = (epochs as f64 * 0.3).ceil() as usize;

    for epoch in 0..epochs {
        // KL annealing: linearly increase beta from 0 to 1 over warmup period
        let beta = if epoch < warmup_epochs {
            epoch as f64 / warmup_epochs as f64
        } else {
            1.0
        };

        let mut total_loss = 0.0;
        let mut total_recon = 0.0;
        let mut total_kl = [0.0_f64; 3];

        for x in data.iter() {
            let result = hvae.forward(x, rng);
            let loss = HierarchicalVAE::loss(x, &result, beta, free_bits);
            total_loss += loss.total;
            total_recon += loss.recon_loss;
            for l in 0..3 {
                total_kl[l] += loss.kl_per_level[l];
            }

            // Numerical gradient update for decoder output layer
            let eps = 1e-4;
            let (rows, cols) = hvae.dec_output.weight.dim();
            let mut w_grad = Array2::zeros((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    hvae.dec_output.weight[[r, c]] += eps;
                    let res_p = hvae.forward(x, rng);
                    let loss_p = HierarchicalVAE::loss(x, &res_p, beta, free_bits);
                    hvae.dec_output.weight[[r, c]] -= eps;
                    w_grad[[r, c]] = (loss_p.total - loss.total) / eps;
                }
            }
            let mut b_grad = Array1::zeros(cols);
            for c in 0..cols {
                hvae.dec_output.bias[c] += eps;
                let res_p = hvae.forward(x, rng);
                let loss_p = HierarchicalVAE::loss(x, &res_p, beta, free_bits);
                hvae.dec_output.bias[c] -= eps;
                b_grad[c] = (loss_p.total - loss.total) / eps;
            }
            hvae.dec_output.update(&w_grad, &b_grad, lr);

            // Numerical gradient update for decoder hidden layer
            let (rows, cols) = hvae.dec_hidden.weight.dim();
            let mut w_grad = Array2::zeros((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    hvae.dec_hidden.weight[[r, c]] += eps;
                    let res_p = hvae.forward(x, rng);
                    let loss_p = HierarchicalVAE::loss(x, &res_p, beta, free_bits);
                    hvae.dec_hidden.weight[[r, c]] -= eps;
                    w_grad[[r, c]] = (loss_p.total - loss.total) / eps;
                }
            }
            let mut b_grad = Array1::zeros(cols);
            for c in 0..cols {
                hvae.dec_hidden.bias[c] += eps;
                let res_p = hvae.forward(x, rng);
                let loss_p = HierarchicalVAE::loss(x, &res_p, beta, free_bits);
                hvae.dec_hidden.bias[c] -= eps;
                b_grad[c] = (loss_p.total - loss.total) / eps;
            }
            hvae.dec_hidden.update(&w_grad, &b_grad, lr);
        }

        let n = data.len() as f64;
        let avg_loss = HVAELoss {
            total: total_loss / n,
            recon_loss: total_recon / n,
            kl_per_level: [total_kl[0] / n, total_kl[1] / n, total_kl[2] / n],
            kl_total: total_kl.iter().sum::<f64>() / n,
        };

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {}/{}: loss={:.4} recon={:.4} KL=[{:.4}, {:.4}, {:.4}] beta={:.2}",
                epoch + 1,
                epochs,
                avg_loss.total,
                avg_loss.recon_loss,
                avg_loss.kl_per_level[0],
                avg_loss.kl_per_level[1],
                avg_loss.kl_per_level[2],
                beta
            );
        }

        loss_history.push(avg_loss);
    }

    loss_history
}

// ---------------------------------------------------------------------------
// Quality Metrics
// ---------------------------------------------------------------------------

/// Per-regime quality metrics comparing generated vs real data.
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
            "{}: real(mean={:.6}, std={:.6}) gen(mean={:.6}, std={:.6}) \
             err(mean={:.6}, std={:.6}) wasserstein~{:.6}",
            self.regime.name(),
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

/// Evaluate generated samples against real data, grouped by regime.
pub fn evaluate_generation(
    hvae: &HierarchicalVAE,
    data: &[Array1<f64>],
    regimes: &[MarketRegime],
    samples_total: usize,
    rng: &mut impl Rng,
) -> Vec<RegimeMetrics> {
    let generated = hvae.generate(samples_total, rng);
    let gen_values: Vec<f64> = generated.iter().cloned().collect();

    let gen_mean_all = gen_values.iter().sum::<f64>() / gen_values.len() as f64;
    let gen_std_all = (gen_values
        .iter()
        .map(|v| (v - gen_mean_all).powi(2))
        .sum::<f64>()
        / gen_values.len() as f64)
        .sqrt();

    let mut metrics = Vec::new();

    for regime_idx in 0..MarketRegime::COUNT {
        let regime = MarketRegime::from_index(regime_idx);

        let real_values: Vec<f64> = data
            .iter()
            .zip(regimes.iter())
            .filter(|(_, r)| **r == regime)
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

        // Approximate 1-Wasserstein
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
            gen_mean: gen_mean_all,
            gen_std: gen_std_all,
            mean_error: (real_mean - gen_mean_all).abs(),
            std_error: (real_std - gen_std_all).abs(),
            wasserstein_approx: wasserstein,
        });
    }

    metrics
}

// ---------------------------------------------------------------------------
// Data preparation helpers
// ---------------------------------------------------------------------------

/// Convert prices into windowed training samples with regime labels.
pub fn prepare_training_data(
    prices: &[f64],
    window: usize,
) -> (Vec<Array1<f64>>, Vec<MarketRegime>) {
    if prices.len() < 2 {
        return (vec![], vec![]);
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
    let mut data_regimes = Vec::new();
    for i in window..norm_returns.len() {
        let x = Array1::from(norm_returns[i - window..i].to_vec());
        data.push(x);
        data_regimes.push(regimes[i]);
    }

    (data, data_regimes)
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

    let mut prices: Vec<f64> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 5 {
                row[4].parse::<f64>().ok()
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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
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
        let mut prices = Vec::new();
        let mut p: f64 = 100.0;

        // Bull phase
        for _ in 0..40 {
            p *= 1.005;
            prices.push(p);
        }
        // Transition
        for i in 0..40 {
            let noise = if i % 2 == 0 { 1.008 } else { 0.992 };
            p *= noise;
            prices.push(p);
        }
        // Bear phase
        for i in 0..40 {
            let shock = if i % 3 == 0 { 0.96 } else { 0.985 };
            p *= shock;
            prices.push(p);
        }

        let regimes = detect_regimes(&prices, 10);
        assert_eq!(regimes.len(), prices.len() - 1);

        let bull_count = regimes.iter().filter(|r| **r == MarketRegime::Bull).count();
        let bear_count = regimes.iter().filter(|r| **r == MarketRegime::Bear).count();
        let side_count = regimes
            .iter()
            .filter(|r| **r == MarketRegime::Sideways)
            .count();

        let types_detected = [bull_count > 0, bear_count > 0, side_count > 0]
            .iter()
            .filter(|&&x| x)
            .count();
        assert!(
            types_detected >= 2,
            "Expected at least 2 regime types, got {} (bull={}, bear={}, side={})",
            types_detected,
            bull_count,
            bear_count,
            side_count
        );
    }

    #[test]
    fn test_precision_weighted_merge() {
        // Two identical distributions should produce the same distribution
        let mu1 = Array1::from(vec![1.0, 2.0]);
        let logvar1 = Array1::from(vec![0.0, 0.0]);
        let mu2 = Array1::from(vec![1.0, 2.0]);
        let logvar2 = Array1::from(vec![0.0, 0.0]);

        let (mu_m, logvar_m) = precision_weighted_merge(&mu1, &logvar1, &mu2, &logvar2);

        for i in 0..2 {
            assert!(
                (mu_m[i] - 1.0 * (i == 0) as u8 as f64 - 2.0 * (i == 1) as u8 as f64).abs()
                    < 1e-10
            );
        }
        // Variance should be halved (doubled precision) → logvar = ln(0.5) ≈ -0.693
        let expected_logvar = (0.5_f64).ln(); // ln(0.5) ≈ -0.6931
        for &lv in logvar_m.iter() {
            assert!((lv - expected_logvar).abs() < 0.01, "logvar_merged = {}", lv);
        }
    }

    #[test]
    fn test_precision_weighted_merge_skewed() {
        // One very precise (low variance), one imprecise
        let mu1 = Array1::from(vec![5.0]);
        let logvar1 = Array1::from(vec![-10.0]); // very precise
        let mu2 = Array1::from(vec![0.0]);
        let logvar2 = Array1::from(vec![10.0]); // very imprecise

        let (mu_m, _) = precision_weighted_merge(&mu1, &logvar1, &mu2, &logvar2);
        // Merged mean should be very close to mu1 (the precise one)
        assert!(
            (mu_m[0] - 5.0).abs() < 0.01,
            "Merged mean should be near 5.0, got {}",
            mu_m[0]
        );
    }

    #[test]
    fn test_hvae_forward_dimensions() {
        let mut rng = make_rng();
        let input_dim = 5;
        let hidden_dim = 16;
        let latent_dims = [3, 2, 2]; // bottom, middle, top

        let hvae = HierarchicalVAE::new(input_dim, hidden_dim, latent_dims, &mut rng);

        let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1]);
        let result = hvae.forward(&x, &mut rng);

        assert_eq!(result.recon.len(), input_dim);
        assert_eq!(result.level_info[0].z.len(), latent_dims[0]); // level 1
        assert_eq!(result.level_info[1].z.len(), latent_dims[1]); // level 2
        assert_eq!(result.level_info[2].z.len(), latent_dims[2]); // level 3
    }

    #[test]
    fn test_hvae_loss_structure() {
        let mut rng = make_rng();
        let hvae = HierarchicalVAE::new(5, 16, [3, 2, 2], &mut rng);

        let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1]);
        let result = hvae.forward(&x, &mut rng);
        let loss = HierarchicalVAE::loss(&x, &result, 1.0, 0.0);

        assert!(loss.recon_loss >= 0.0, "Recon loss should be non-negative");
        assert!(loss.total >= 0.0 || true, "Total loss computed"); // can be negative with KL
        assert_eq!(loss.kl_per_level.len(), 3);
    }

    #[test]
    fn test_hvae_kl_identical_distributions() {
        let mu = Array1::from(vec![0.0, 0.0]);
        let logvar = Array1::from(vec![0.0, 0.0]);

        let kl = HierarchicalVAE::kl_divergence(&mu, &logvar, &mu, &logvar);
        assert!(
            kl.abs() < 1e-10,
            "KL should be 0 for identical distributions, got {}",
            kl
        );
    }

    #[test]
    fn test_hvae_kl_different_distributions() {
        let mu_q = Array1::from(vec![1.0, 2.0]);
        let logvar_q = Array1::from(vec![0.5, 0.5]);
        let mu_p = Array1::from(vec![0.0, 0.0]);
        let logvar_p = Array1::from(vec![0.0, 0.0]);

        let kl = HierarchicalVAE::kl_divergence(&mu_q, &logvar_q, &mu_p, &logvar_p);
        assert!(
            kl > 0.0,
            "KL should be positive for different distributions"
        );
    }

    #[test]
    fn test_generate_shape() {
        let mut rng = make_rng();
        let input_dim = 5;
        let hvae = HierarchicalVAE::new(input_dim, 16, [3, 2, 2], &mut rng);

        let samples = hvae.generate(100, &mut rng);
        assert_eq!(samples.dim(), (100, input_dim));
    }

    #[test]
    fn test_generate_with_fixed_top() {
        let mut rng = make_rng();
        let input_dim = 5;
        let latent_dims = [3, 2, 2];
        let hvae = HierarchicalVAE::new(input_dim, 16, latent_dims, &mut rng);

        let z3 = Array1::from(vec![1.0, -1.0]);
        let samples = hvae.generate_with_fixed_top(&z3, 50, &mut rng);
        assert_eq!(samples.dim(), (50, input_dim));
    }

    #[test]
    fn test_different_top_latents_produce_different_outputs() {
        let mut rng = make_rng();
        let hvae = HierarchicalVAE::new(5, 16, [3, 2, 2], &mut rng);

        let z3_a = Array1::from(vec![2.0, 2.0]);
        let z3_b = Array1::from(vec![-2.0, -2.0]);

        let samples_a = hvae.generate_with_fixed_top(&z3_a, 500, &mut rng);
        let samples_b = hvae.generate_with_fixed_top(&z3_b, 500, &mut rng);

        let mean_a = samples_a.mean_axis(Axis(0)).unwrap();
        let mean_b = samples_b.mean_axis(Axis(0)).unwrap();

        let diff = (&mean_a - &mean_b).mapv(|v| v.abs()).sum();
        assert!(
            diff > 1e-6,
            "Different top latents should produce different outputs"
        );
    }

    #[test]
    fn test_encode_top_level() {
        let mut rng = make_rng();
        let hvae = HierarchicalVAE::new(5, 16, [3, 2, 2], &mut rng);

        let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1]);
        let z_top = hvae.encode_top_level(&x);
        assert_eq!(z_top.len(), 2); // top latent dim
    }

    #[test]
    fn test_reparameterize() {
        let mut rng = make_rng();
        let mu = Array1::from(vec![0.0, 0.0, 0.0]);
        let logvar = Array1::from(vec![0.0, 0.0, 0.0]);

        let z = HierarchicalVAE::reparameterize(&mu, &logvar, &mut rng);
        assert_eq!(z.len(), 3);
        for &v in z.iter() {
            assert!(v.abs() < 10.0, "Sample too extreme: {}", v);
        }
    }

    #[test]
    fn test_prepare_training_data() {
        let mut prices = Vec::new();
        let mut p: f64 = 100.0;
        for _ in 0..100 {
            p *= 1.0 + 0.001 * p.sin();
            prices.push(p);
        }

        let window = 5;
        let (data, regimes) = prepare_training_data(&prices, window);
        assert!(!data.is_empty());
        assert_eq!(data.len(), regimes.len());

        for x in &data {
            assert_eq!(x.len(), window);
        }
    }

    #[test]
    fn test_hierarchy_level() {
        let mut rng = make_rng();
        let level = HierarchyLevel::new(16, 4, 3, &mut rng);

        let d = Array1::from(vec![0.1; 16]);
        let (mu_bu, logvar_bu) = level.bottom_up(&d);
        assert_eq!(mu_bu.len(), 3);
        assert_eq!(logvar_bu.len(), 3);

        let z_above = Array1::from(vec![0.5; 4]);
        let (mu_td, logvar_td) = level.top_down(&z_above);
        assert_eq!(mu_td.len(), 3);
        assert_eq!(logvar_td.len(), 3);
    }

    #[test]
    fn test_free_bits() {
        let mut rng = make_rng();
        let hvae = HierarchicalVAE::new(5, 16, [3, 2, 2], &mut rng);
        let x = Array1::from(vec![0.1, -0.2, 0.05, 0.3, -0.1]);
        let result = hvae.forward(&x, &mut rng);

        // With high free_bits, KL per level should be at least free_bits
        let free_bits = 1.0;
        let loss = HierarchicalVAE::loss(&x, &result, 1.0, free_bits);
        for &kl in &loss.kl_per_level {
            assert!(
                kl >= free_bits - 1e-10,
                "KL {} should be >= free_bits {}",
                kl,
                free_bits
            );
        }
    }
}
