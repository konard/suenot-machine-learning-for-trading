use rand::Rng;
use serde::Deserialize;

// ─── Portfolio Representation ─────────────────────────────────────

/// A portfolio represented as a vector of asset weights.
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Asset names (e.g., "BTCUSDT", "ETHUSDT").
    pub assets: Vec<String>,
    /// Weights for each asset (sum to 1.0).
    pub weights: Vec<f64>,
}

impl Portfolio {
    /// Create a new portfolio from assets and weights.
    /// Weights are normalized to sum to 1.0.
    pub fn new(assets: Vec<String>, weights: Vec<f64>) -> Self {
        let total: f64 = weights.iter().sum();
        let normalized = if total > 0.0 {
            weights.iter().map(|w| w / total).collect()
        } else {
            vec![1.0 / weights.len() as f64; weights.len()]
        };
        Self {
            assets,
            weights: normalized,
        }
    }

    /// Number of assets.
    pub fn n_assets(&self) -> usize {
        self.assets.len()
    }

    /// Display the portfolio allocation.
    pub fn display(&self) -> String {
        self.assets
            .iter()
            .zip(self.weights.iter())
            .map(|(a, w)| format!("  {}: {:.2}%", a, w * 100.0))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ─── Dense Layer ──────────────────────────────────────────────────

/// A single dense (fully connected) layer with ReLU or linear activation.
#[derive(Debug, Clone)]
struct DenseLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    use_relu: bool,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization.
    fn new(input_size: usize, output_size: usize, use_relu: bool) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();
        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();
        let biases = vec![0.0; output_size];
        Self {
            weights,
            biases,
            use_relu,
        }
    }

    /// Forward pass through the layer.
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w, b)| {
                let z: f64 = w.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum::<f64>() + b;
                if self.use_relu {
                    z.max(0.0)
                } else {
                    z
                }
            })
            .collect()
    }

    /// Backward pass: compute gradients and return input gradients.
    fn backward(
        &self,
        input: &[f64],
        output: &[f64],
        grad_output: &[f64],
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let mut grad_weights = vec![vec![0.0; input.len()]; self.weights.len()];
        let mut grad_biases = vec![0.0; self.biases.len()];
        let mut grad_input = vec![0.0; input.len()];

        for i in 0..self.weights.len() {
            let grad = if self.use_relu && output[i] <= 0.0 {
                0.0
            } else {
                grad_output[i]
            };
            grad_biases[i] = grad;
            for j in 0..input.len() {
                grad_weights[i][j] = grad * input[j];
                grad_input[j] += grad * self.weights[i][j];
            }
        }
        (grad_weights, grad_biases, grad_input)
    }

    /// Update weights using SGD.
    fn update(&mut self, grad_weights: &[Vec<f64>], grad_biases: &[f64], lr: f64) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] -= lr * grad_weights[i][j];
            }
            self.biases[i] -= lr * grad_biases[i];
        }
    }
}

// ─── Portfolio VAE ────────────────────────────────────────────────

/// Variational Autoencoder for portfolio weight generation.
///
/// Architecture:
///   Encoder: input(n) -> hidden1(64) -> hidden2(32) -> mu(latent_dim), logvar(latent_dim)
///   Decoder: latent(latent_dim) -> hidden3(32) -> hidden4(64) -> output(n)
///   Output is passed through softmax to produce valid portfolio weights.
#[derive(Debug)]
pub struct PortfolioVAE {
    n_assets: usize,
    latent_dim: usize,
    beta: f64,
    // Encoder layers
    enc_layer1: DenseLayer,
    enc_layer2: DenseLayer,
    enc_mu: DenseLayer,
    enc_logvar: DenseLayer,
    // Decoder layers
    dec_layer1: DenseLayer,
    dec_layer2: DenseLayer,
    dec_output: DenseLayer,
}

impl PortfolioVAE {
    /// Create a new VAE with the given number of assets and latent dimensions.
    pub fn new(n_assets: usize, latent_dim: usize, beta: f64) -> Self {
        Self {
            n_assets,
            latent_dim,
            beta,
            enc_layer1: DenseLayer::new(n_assets, 64, true),
            enc_layer2: DenseLayer::new(64, 32, true),
            enc_mu: DenseLayer::new(32, latent_dim, false),
            enc_logvar: DenseLayer::new(32, latent_dim, false),
            dec_layer1: DenseLayer::new(latent_dim, 32, true),
            dec_layer2: DenseLayer::new(32, 64, true),
            dec_output: DenseLayer::new(64, n_assets, false),
        }
    }

    /// Encode a portfolio into latent space parameters (mu, logvar).
    pub fn encode(&self, weights: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let h1 = self.enc_layer1.forward(weights);
        let h2 = self.enc_layer2.forward(&h1);
        let mu = self.enc_mu.forward(&h2);
        let logvar = self.enc_logvar.forward(&h2);
        (mu, logvar)
    }

    /// Sample from the latent distribution using the reparameterization trick.
    pub fn reparameterize(&self, mu: &[f64], logvar: &[f64]) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        mu.iter()
            .zip(logvar.iter())
            .map(|(m, lv)| {
                let std = (lv / 2.0).exp();
                let eps: f64 = rng.gen_range(-1.0..1.0);
                m + std * eps
            })
            .collect()
    }

    /// Decode a latent vector into portfolio weights (pre-softmax).
    pub fn decode_raw(&self, z: &[f64]) -> Vec<f64> {
        let h3 = self.dec_layer1.forward(z);
        let h4 = self.dec_layer2.forward(&h3);
        self.dec_output.forward(&h4)
    }

    /// Decode a latent vector into valid portfolio weights (post-softmax).
    pub fn decode(&self, z: &[f64]) -> Vec<f64> {
        let raw = self.decode_raw(z);
        softmax(&raw)
    }

    /// Full forward pass: encode, sample, decode.
    /// Returns (reconstructed_weights, mu, logvar).
    pub fn forward(&self, weights: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mu, logvar) = self.encode(weights);
        let z = self.reparameterize(&mu, &logvar);
        let recon = self.decode(&z);
        (recon, mu, logvar)
    }

    /// Compute VAE loss: reconstruction loss + beta * KL divergence.
    pub fn loss(
        &self,
        input: &[f64],
        recon: &[f64],
        mu: &[f64],
        logvar: &[f64],
    ) -> (f64, f64, f64) {
        let recon_loss: f64 = input
            .iter()
            .zip(recon.iter())
            .map(|(x, r)| (x - r).powi(2))
            .sum::<f64>()
            / input.len() as f64;

        let kl_div: f64 = -0.5
            * mu.iter()
                .zip(logvar.iter())
                .map(|(m, lv)| 1.0 + lv - m.powi(2) - lv.exp())
                .sum::<f64>()
            / mu.len() as f64;

        let total = recon_loss + self.beta * kl_div;
        (total, recon_loss, kl_div)
    }

    /// Train the VAE on a set of portfolio weight vectors.
    pub fn train(&mut self, portfolios: &[Vec<f64>], epochs: usize, lr: f64) -> Vec<f64> {
        let mut losses = Vec::with_capacity(epochs);

        for _epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for portfolio in portfolios {
                // Forward pass through encoder
                let h1 = self.enc_layer1.forward(portfolio);
                let h2 = self.enc_layer2.forward(&h1);
                let mu = self.enc_mu.forward(&h2);
                let logvar = self.enc_logvar.forward(&h2);

                // Reparameterize
                let z = self.reparameterize(&mu, &logvar);

                // Forward pass through decoder
                let h3 = self.dec_layer1.forward(&z);
                let h4 = self.dec_layer2.forward(&h3);
                let raw_output = self.dec_output.forward(&h4);
                let recon = softmax(&raw_output);

                // Compute loss
                let (total_loss, _, _) = self.loss(portfolio, &recon, &mu, &logvar);
                epoch_loss += total_loss;

                // Compute reconstruction gradient (d(MSE)/d(recon))
                let grad_recon: Vec<f64> = portfolio
                    .iter()
                    .zip(recon.iter())
                    .map(|(x, r)| -2.0 * (x - r) / portfolio.len() as f64)
                    .collect();

                // Softmax Jacobian (simplified: use identity approximation for stability)
                let grad_raw = grad_recon.clone();

                // Backward through decoder
                let (gw_out, gb_out, gi_out) =
                    self.dec_output.backward(&h4, &raw_output, &grad_raw);
                let (gw_d2, gb_d2, gi_d2) =
                    self.dec_layer2.backward(&h3, &h4, &gi_out);
                let (gw_d1, gb_d1, _gi_d1) =
                    self.dec_layer1.backward(&z, &h3, &gi_d2);

                // KL gradient for mu and logvar
                let grad_mu: Vec<f64> = mu
                    .iter()
                    .map(|m| self.beta * m / mu.len() as f64)
                    .collect();
                let grad_logvar: Vec<f64> = logvar
                    .iter()
                    .map(|lv| self.beta * 0.5 * (lv.exp() - 1.0) / logvar.len() as f64)
                    .collect();

                // Backward through encoder (mu and logvar branches)
                let (gw_mu, gb_mu, gi_mu) =
                    self.enc_mu.backward(&h2, &mu, &grad_mu);
                let (gw_lv, gb_lv, gi_lv) =
                    self.enc_logvar.backward(&h2, &logvar, &grad_logvar);

                // Combine encoder gradients
                let gi_h2: Vec<f64> = gi_mu
                    .iter()
                    .zip(gi_lv.iter())
                    .map(|(a, b)| a + b)
                    .collect();

                let (gw_e2, gb_e2, gi_e2) =
                    self.enc_layer2.backward(&h1, &h2, &gi_h2);
                let (gw_e1, gb_e1, _gi_e1) =
                    self.enc_layer1.backward(portfolio, &h1, &gi_e2);

                // Update all layers
                self.enc_layer1.update(&gw_e1, &gb_e1, lr);
                self.enc_layer2.update(&gw_e2, &gb_e2, lr);
                self.enc_mu.update(&gw_mu, &gb_mu, lr);
                self.enc_logvar.update(&gw_lv, &gb_lv, lr);
                self.dec_layer1.update(&gw_d1, &gb_d1, lr);
                self.dec_layer2.update(&gw_d2, &gb_d2, lr);
                self.dec_output.update(&gw_out, &gb_out, lr);
            }

            epoch_loss /= portfolios.len() as f64;
            losses.push(epoch_loss);
        }

        losses
    }

    /// Generate a random portfolio by sampling from the prior.
    pub fn generate(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let z: Vec<f64> = (0..self.latent_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        self.decode(&z)
    }

    /// Generate multiple random portfolios.
    pub fn generate_many(&self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.generate()).collect()
    }

    /// Interpolate between two portfolios in latent space.
    pub fn interpolate(
        &self,
        weights_a: &[f64],
        weights_b: &[f64],
        steps: usize,
    ) -> Vec<Vec<f64>> {
        let (mu_a, _) = self.encode(weights_a);
        let (mu_b, _) = self.encode(weights_b);

        (0..=steps)
            .map(|i| {
                let alpha = i as f64 / steps as f64;
                let z: Vec<f64> = mu_a
                    .iter()
                    .zip(mu_b.iter())
                    .map(|(a, b)| (1.0 - alpha) * a + alpha * b)
                    .collect();
                self.decode(&z)
            })
            .collect()
    }

    /// Return the number of assets and latent dimension.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.n_assets, self.latent_dim)
    }
}

// ─── Portfolio Evaluator ──────────────────────────────────────────

/// Evaluates portfolio performance using historical return data.
#[derive(Debug)]
pub struct PortfolioEvaluator {
    /// Return series for each asset (rows = time, cols = assets).
    returns: Vec<Vec<f64>>,
    n_periods: usize,
}

impl PortfolioEvaluator {
    /// Create from a matrix of returns (one Vec per asset).
    pub fn new(returns: Vec<Vec<f64>>) -> Self {
        let n_periods = returns.first().map_or(0, |r| r.len());
        Self {
            returns,
            n_periods,
        }
    }

    /// Compute portfolio returns given weights.
    pub fn portfolio_returns(&self, weights: &[f64]) -> Vec<f64> {
        (0..self.n_periods)
            .map(|t| {
                self.returns
                    .iter()
                    .zip(weights.iter())
                    .map(|(asset_returns, w)| {
                        if t < asset_returns.len() {
                            w * asset_returns[t]
                        } else {
                            0.0
                        }
                    })
                    .sum()
            })
            .collect()
    }

    /// Compute annualized return (assuming daily returns, 365 days for crypto).
    pub fn annualized_return(&self, weights: &[f64]) -> f64 {
        let rets = self.portfolio_returns(weights);
        if rets.is_empty() {
            return 0.0;
        }
        let mean_daily: f64 = rets.iter().sum::<f64>() / rets.len() as f64;
        mean_daily * 365.0
    }

    /// Compute annualized volatility.
    pub fn annualized_volatility(&self, weights: &[f64]) -> f64 {
        let rets = self.portfolio_returns(weights);
        if rets.len() < 2 {
            return 0.0;
        }
        let mean: f64 = rets.iter().sum::<f64>() / rets.len() as f64;
        let variance: f64 =
            rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (rets.len() - 1) as f64;
        variance.sqrt() * (365.0_f64).sqrt()
    }

    /// Compute Sharpe ratio (risk-free rate assumed 0).
    pub fn sharpe_ratio(&self, weights: &[f64]) -> f64 {
        let ret = self.annualized_return(weights);
        let vol = self.annualized_volatility(weights);
        if vol == 0.0 {
            return 0.0;
        }
        ret / vol
    }

    /// Compute Sortino ratio (downside deviation only).
    pub fn sortino_ratio(&self, weights: &[f64]) -> f64 {
        let rets = self.portfolio_returns(weights);
        if rets.len() < 2 {
            return 0.0;
        }
        let mean: f64 = rets.iter().sum::<f64>() / rets.len() as f64;
        let downside_var: f64 =
            rets.iter().map(|r| if *r < 0.0 { r.powi(2) } else { 0.0 }).sum::<f64>()
                / rets.len() as f64;
        let downside_dev = downside_var.sqrt() * (365.0_f64).sqrt();
        if downside_dev == 0.0 {
            return 0.0;
        }
        (mean * 365.0) / downside_dev
    }

    /// Compute maximum drawdown.
    pub fn max_drawdown(&self, weights: &[f64]) -> f64 {
        let rets = self.portfolio_returns(weights);
        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for r in &rets {
            cumulative *= 1.0 + r;
            if cumulative > peak {
                peak = cumulative;
            }
            let dd = (peak - cumulative) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        max_dd
    }

    /// Compute diversification ratio: weighted average vol / portfolio vol.
    pub fn diversification_ratio(&self, weights: &[f64]) -> f64 {
        let port_vol = self.annualized_volatility(weights);
        if port_vol == 0.0 {
            return 1.0;
        }

        let weighted_avg_vol: f64 = self
            .returns
            .iter()
            .zip(weights.iter())
            .map(|(asset_rets, w)| {
                if asset_rets.len() < 2 {
                    return 0.0;
                }
                let mean: f64 = asset_rets.iter().sum::<f64>() / asset_rets.len() as f64;
                let var: f64 = asset_rets
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / (asset_rets.len() - 1) as f64;
                w.abs() * var.sqrt() * (365.0_f64).sqrt()
            })
            .sum();

        weighted_avg_vol / port_vol
    }

    /// Full evaluation summary.
    pub fn evaluate(&self, weights: &[f64]) -> PortfolioMetrics {
        PortfolioMetrics {
            annualized_return: self.annualized_return(weights),
            annualized_volatility: self.annualized_volatility(weights),
            sharpe_ratio: self.sharpe_ratio(weights),
            sortino_ratio: self.sortino_ratio(weights),
            max_drawdown: self.max_drawdown(weights),
            diversification_ratio: self.diversification_ratio(weights),
        }
    }
}

/// Summary of portfolio performance metrics.
#[derive(Debug, Clone)]
pub struct PortfolioMetrics {
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub diversification_ratio: f64,
}

impl PortfolioMetrics {
    /// Display metrics in a formatted string.
    pub fn display(&self) -> String {
        format!(
            "  Annualized Return:  {:.2}%\n  \
             Annualized Vol:     {:.2}%\n  \
             Sharpe Ratio:       {:.4}\n  \
             Sortino Ratio:      {:.4}\n  \
             Max Drawdown:       {:.2}%\n  \
             Diversification:    {:.4}",
            self.annualized_return * 100.0,
            self.annualized_volatility * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.diversification_ratio,
        )
    }
}

// ─── Dirichlet Sampler ────────────────────────────────────────────

/// Generates random portfolios using the Dirichlet distribution.
///
/// The Dirichlet distribution produces vectors of non-negative numbers
/// that sum to 1.0, making it ideal for portfolio weight generation.
#[derive(Debug)]
pub struct DirichletSampler {
    n_assets: usize,
    alpha: Vec<f64>,
}

impl DirichletSampler {
    /// Create a sampler with uniform concentration parameter.
    pub fn new(n_assets: usize, alpha: f64) -> Self {
        Self {
            n_assets,
            alpha: vec![alpha; n_assets],
        }
    }

    /// Create a sampler with per-asset concentration parameters.
    pub fn with_alphas(alphas: Vec<f64>) -> Self {
        Self {
            n_assets: alphas.len(),
            alpha: alphas,
        }
    }

    /// Sample a single portfolio from the Dirichlet distribution.
    ///
    /// Uses the gamma distribution method: sample x_i ~ Gamma(alpha_i, 1)
    /// and normalize: w_i = x_i / sum(x).
    pub fn sample(&self) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let gammas: Vec<f64> = self
            .alpha
            .iter()
            .map(|&a| gamma_sample(&mut rng, a))
            .collect();
        let total: f64 = gammas.iter().sum();
        if total > 0.0 {
            gammas.iter().map(|g| g / total).collect()
        } else {
            vec![1.0 / self.n_assets as f64; self.n_assets]
        }
    }

    /// Sample multiple portfolios.
    pub fn sample_many(&self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.sample()).collect()
    }
}

/// Simple gamma distribution sampler using Marsaglia and Tsang's method.
fn gamma_sample(rng: &mut impl Rng, alpha: f64) -> f64 {
    if alpha < 1.0 {
        // For alpha < 1, use the relation: Gamma(a) = Gamma(a+1) * U^(1/a)
        let u: f64 = rng.gen_range(0.0001..1.0);
        return gamma_sample(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }

    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x: f64 = standard_normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.gen_range(0.0001..1.0);
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Standard normal sample using Box-Muller transform.
fn standard_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen_range(0.0001..1.0);
    let u2: f64 = rng.gen_range(0.0001..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ─── Portfolio Generator ──────────────────────────────────────────

/// High-level portfolio generation interface wrapping the VAE.
#[derive(Debug)]
pub struct PortfolioGenerator {
    vae: PortfolioVAE,
    asset_names: Vec<String>,
    max_weight: f64,
    min_weight: f64,
}

impl PortfolioGenerator {
    /// Create a new generator with the given assets and VAE parameters.
    pub fn new(
        asset_names: Vec<String>,
        latent_dim: usize,
        beta: f64,
        max_weight: f64,
        min_weight: f64,
    ) -> Self {
        let n = asset_names.len();
        Self {
            vae: PortfolioVAE::new(n, latent_dim, beta),
            asset_names,
            max_weight,
            min_weight,
        }
    }

    /// Train the generator on a set of portfolio weight vectors.
    pub fn train(&mut self, portfolios: &[Vec<f64>], epochs: usize, lr: f64) -> Vec<f64> {
        self.vae.train(portfolios, epochs, lr)
    }

    /// Generate a single portfolio with constraint enforcement.
    pub fn generate(&self) -> Portfolio {
        let weights = self.enforce_constraints(&self.vae.generate());
        Portfolio::new(self.asset_names.clone(), weights)
    }

    /// Generate multiple portfolios.
    pub fn generate_many(&self, n: usize) -> Vec<Portfolio> {
        (0..n).map(|_| self.generate()).collect()
    }

    /// Interpolate between two portfolios.
    pub fn interpolate(
        &self,
        weights_a: &[f64],
        weights_b: &[f64],
        steps: usize,
    ) -> Vec<Portfolio> {
        self.vae
            .interpolate(weights_a, weights_b, steps)
            .into_iter()
            .map(|w| {
                let constrained = self.enforce_constraints(&w);
                Portfolio::new(self.asset_names.clone(), constrained)
            })
            .collect()
    }

    /// Enforce position limits and minimum allocation.
    fn enforce_constraints(&self, weights: &[f64]) -> Vec<f64> {
        let mut w: Vec<f64> = weights
            .iter()
            .map(|&x| {
                let clamped = x.max(0.0).min(self.max_weight);
                if clamped < self.min_weight {
                    0.0
                } else {
                    clamped
                }
            })
            .collect();

        // Re-normalize to sum to 1.0
        let total: f64 = w.iter().sum();
        if total > 0.0 {
            for wi in &mut w {
                *wi /= total;
            }
        } else {
            // Fallback to equal weight
            let n = w.len() as f64;
            for wi in &mut w {
                *wi = 1.0 / n;
            }
        }
        w
    }

    /// Access the underlying VAE.
    pub fn vae(&self) -> &PortfolioVAE {
        &self.vae
    }
}

// ─── Bybit API Client ────────────────────────────────────────────

/// HTTP client for the Bybit V5 REST API.
#[derive(Debug)]
pub struct BybitClient {
    client: reqwest::Client,
    base_url: String,
}

#[derive(Debug, Deserialize)]
struct BybitResponse<T> {
    result: T,
}

#[derive(Debug, Deserialize)]
struct KlineResult {
    list: Vec<Vec<String>>,
}

/// A single candlestick (kline) bar.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl BybitClient {
    /// Create a new client pointing at the Bybit mainnet API.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data for a symbol.
    ///
    /// `interval` is the bar size: "1" = 1 min, "5" = 5 min, "60" = 1 hour, "D" = 1 day.
    /// `limit` is the number of bars to fetch (max 200).
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );
        let resp: BybitResponse<KlineResult> =
            self.client.get(&url).send().await?.json().await?;

        let mut klines = Vec::new();
        for item in &resp.result.list {
            if item.len() >= 6 {
                klines.push(Kline {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                });
            }
        }
        klines.reverse(); // Bybit returns newest first
        Ok(klines)
    }

    /// Fetch kline data for multiple symbols and compute return series.
    pub async fn get_multi_asset_returns(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<Vec<f64>>> {
        let mut all_returns = Vec::new();

        for symbol in symbols {
            let klines = self.get_klines(symbol, interval, limit).await?;
            let returns: Vec<f64> = klines
                .windows(2)
                .map(|w| {
                    if w[0].close > 0.0 {
                        (w[1].close - w[0].close) / w[0].close
                    } else {
                        0.0
                    }
                })
                .collect();
            all_returns.push(returns);
        }

        Ok(all_returns)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Utility Functions ───────────────────────────────────────────

/// Softmax function: converts raw values to probabilities summing to 1.
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Generate synthetic return data for testing.
pub fn generate_synthetic_returns(n_assets: usize, n_periods: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    (0..n_assets)
        .map(|_| {
            let drift = rng.gen_range(-0.001..0.001);
            let vol = rng.gen_range(0.01..0.05);
            (0..n_periods)
                .map(|_| drift + vol * standard_normal(&mut rng))
                .collect()
        })
        .collect()
}

/// Generate training portfolios using a combination of methods.
pub fn generate_training_portfolios(n_assets: usize, n_portfolios: usize) -> Vec<Vec<f64>> {
    let mut portfolios = Vec::with_capacity(n_portfolios);

    // 1/3 concentrated portfolios (small alpha)
    let concentrated = DirichletSampler::new(n_assets, 0.3);
    for _ in 0..n_portfolios / 3 {
        portfolios.push(concentrated.sample());
    }

    // 1/3 balanced portfolios (large alpha)
    let balanced = DirichletSampler::new(n_assets, 5.0);
    for _ in 0..n_portfolios / 3 {
        portfolios.push(balanced.sample());
    }

    // 1/3 moderate portfolios
    let moderate = DirichletSampler::new(n_assets, 1.0);
    for _ in portfolios.len()..n_portfolios {
        portfolios.push(moderate.sample());
    }

    portfolios
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let s = softmax(&x);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn test_softmax_equal() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let s = softmax(&x);
        for w in &s {
            assert!((*w - 0.25).abs() < 1e-9);
        }
    }

    #[test]
    fn test_portfolio_creation() {
        let p = Portfolio::new(
            vec!["BTC".into(), "ETH".into(), "SOL".into()],
            vec![4.0, 3.0, 3.0],
        );
        assert_eq!(p.n_assets(), 3);
        let sum: f64 = p.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert!((p.weights[0] - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_dirichlet_sampler() {
        let sampler = DirichletSampler::new(5, 1.0);
        let portfolio = sampler.sample();
        assert_eq!(portfolio.len(), 5);
        let sum: f64 = portfolio.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for w in &portfolio {
            assert!(*w >= 0.0);
        }
    }

    #[test]
    fn test_dirichlet_many() {
        let sampler = DirichletSampler::new(4, 2.0);
        let portfolios = sampler.sample_many(100);
        assert_eq!(portfolios.len(), 100);
        for p in &portfolios {
            assert_eq!(p.len(), 4);
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dirichlet_concentrated() {
        let sampler = DirichletSampler::new(5, 0.1);
        let portfolios = sampler.sample_many(50);
        // With low alpha, max weight should often be high
        let mut max_weights: Vec<f64> = portfolios
            .iter()
            .map(|p| p.iter().cloned().fold(0.0, f64::max))
            .collect();
        max_weights.sort_by(|a, b| b.partial_cmp(a).unwrap());
        // At least some should have concentrated weights (> 0.5)
        assert!(max_weights[0] > 0.3);
    }

    #[test]
    fn test_vae_encode_decode() {
        let vae = PortfolioVAE::new(4, 2, 0.1);
        let input = vec![0.25, 0.25, 0.25, 0.25];
        let (mu, logvar) = vae.encode(&input);
        assert_eq!(mu.len(), 2);
        assert_eq!(logvar.len(), 2);

        let recon = vae.decode(&mu);
        assert_eq!(recon.len(), 4);
        let sum: f64 = recon.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vae_forward() {
        let vae = PortfolioVAE::new(5, 3, 0.5);
        let input = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let (recon, mu, logvar) = vae.forward(&input);
        assert_eq!(recon.len(), 5);
        assert_eq!(mu.len(), 3);
        assert_eq!(logvar.len(), 3);
        let sum: f64 = recon.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vae_loss() {
        let vae = PortfolioVAE::new(4, 2, 1.0);
        let input = vec![0.25, 0.25, 0.25, 0.25];
        let (recon, mu, logvar) = vae.forward(&input);
        let (total, recon_loss, kl) = vae.loss(&input, &recon, &mu, &logvar);
        assert!(total >= 0.0);
        assert!(recon_loss >= 0.0);
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_vae_generate() {
        let vae = PortfolioVAE::new(4, 2, 0.1);
        let portfolio = vae.generate();
        assert_eq!(portfolio.len(), 4);
        let sum: f64 = portfolio.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        for w in &portfolio {
            assert!(*w >= 0.0);
        }
    }

    #[test]
    fn test_vae_generate_many() {
        let vae = PortfolioVAE::new(5, 3, 0.1);
        let portfolios = vae.generate_many(10);
        assert_eq!(portfolios.len(), 10);
        for p in &portfolios {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_vae_interpolation() {
        let vae = PortfolioVAE::new(4, 2, 0.1);
        let a = vec![0.7, 0.1, 0.1, 0.1];
        let b = vec![0.1, 0.1, 0.1, 0.7];
        let interp = vae.interpolate(&a, &b, 5);
        assert_eq!(interp.len(), 6); // 0..=5
        for p in &interp {
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_vae_training() {
        let sampler = DirichletSampler::new(4, 1.0);
        let data = sampler.sample_many(100);

        let mut vae = PortfolioVAE::new(4, 2, 0.1);
        let losses = vae.train(&data, 10, 0.001);
        assert_eq!(losses.len(), 10);
        // Loss should be finite
        for l in &losses {
            assert!(l.is_finite(), "loss was not finite: {}", l);
        }
    }

    #[test]
    fn test_portfolio_evaluator_returns() {
        let returns = vec![
            vec![0.01, -0.02, 0.03, 0.01, -0.01],
            vec![0.02, 0.01, -0.01, 0.02, 0.01],
        ];
        let eval = PortfolioEvaluator::new(returns);
        let weights = vec![0.5, 0.5];
        let port_rets = eval.portfolio_returns(&weights);
        assert_eq!(port_rets.len(), 5);
        // First period: 0.5*0.01 + 0.5*0.02 = 0.015
        assert!((port_rets[0] - 0.015).abs() < 1e-9);
    }

    #[test]
    fn test_portfolio_evaluator_metrics() {
        let returns = generate_synthetic_returns(4, 100);
        let eval = PortfolioEvaluator::new(returns);
        let weights = vec![0.25, 0.25, 0.25, 0.25];
        let metrics = eval.evaluate(&weights);
        assert!(metrics.annualized_volatility >= 0.0);
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.max_drawdown <= 1.0);
        assert!(metrics.diversification_ratio >= 0.0);
    }

    #[test]
    fn test_portfolio_evaluator_sharpe() {
        // Use slightly varying positive returns so volatility is nonzero
        let returns = vec![
            vec![0.01, 0.012, 0.009, 0.011, 0.01],
            vec![0.008, 0.011, 0.01, 0.012, 0.009],
        ];
        let eval = PortfolioEvaluator::new(returns);
        let weights = vec![0.5, 0.5];
        let sharpe = eval.sharpe_ratio(&weights);
        // Positive returns with low vol should give positive Sharpe
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_portfolio_generator() {
        let assets = vec![
            "BTC".into(),
            "ETH".into(),
            "SOL".into(),
            "ADA".into(),
        ];
        let gen = PortfolioGenerator::new(assets, 2, 0.1, 0.5, 0.05);
        let portfolio = gen.generate();
        assert_eq!(portfolio.n_assets(), 4);
        let sum: f64 = portfolio.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        for w in &portfolio.weights {
            assert!(*w <= 0.5 + 1e-9 || *w == 0.0);
        }
    }

    #[test]
    fn test_training_data_generation() {
        let data = generate_training_portfolios(4, 90);
        assert_eq!(data.len(), 90);
        for p in &data {
            assert_eq!(p.len(), 4);
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_synthetic_returns() {
        let returns = generate_synthetic_returns(3, 50);
        assert_eq!(returns.len(), 3);
        for asset in &returns {
            assert_eq!(asset.len(), 50);
        }
    }

    #[test]
    fn test_end_to_end() {
        // Generate training data
        let n_assets = 4;
        let training_data = generate_training_portfolios(n_assets, 200);

        // Train VAE
        let mut vae = PortfolioVAE::new(n_assets, 2, 0.1);
        let losses = vae.train(&training_data, 20, 0.001);
        assert!(!losses.is_empty());

        // Generate new portfolios
        let generated = vae.generate_many(10);
        for p in &generated {
            assert_eq!(p.len(), n_assets);
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
            for w in p {
                assert!(*w >= 0.0);
            }
        }

        // Evaluate with synthetic returns
        let returns = generate_synthetic_returns(n_assets, 100);
        let evaluator = PortfolioEvaluator::new(returns);
        for p in &generated {
            let metrics = evaluator.evaluate(p);
            assert!(metrics.max_drawdown >= 0.0);
            assert!(metrics.annualized_volatility >= 0.0);
        }
    }
}
