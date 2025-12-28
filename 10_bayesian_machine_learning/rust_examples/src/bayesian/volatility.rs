//! Stochastic Volatility Models.
//!
//! Implements a simple stochastic volatility model where:
//! - Returns: r_t = exp(h_t/2) * epsilon_t, epsilon_t ~ N(0, 1)
//! - Log-volatility: h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t
//!
//! Parameters:
//! - mu: mean log-volatility
//! - phi: persistence parameter (|phi| < 1 for stationarity)
//! - sigma_eta: volatility of volatility

use super::inference::{MCMCConfig, MCMCSamples};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Stochastic Volatility Model
#[derive(Debug, Clone)]
pub struct StochasticVolatility {
    /// Mean log-volatility
    pub mu: f64,
    /// Persistence parameter
    pub phi: f64,
    /// Volatility of volatility
    pub sigma_eta: f64,
}

impl Default for StochasticVolatility {
    fn default() -> Self {
        Self {
            mu: -5.0,  // exp(-5/2) ~ 0.08 = 8% volatility
            phi: 0.95, // High persistence
            sigma_eta: 0.2,
        }
    }
}

impl StochasticVolatility {
    /// Create a new stochastic volatility model
    pub fn new(mu: f64, phi: f64, sigma_eta: f64) -> Self {
        assert!(phi.abs() < 1.0, "phi must be in (-1, 1) for stationarity");
        assert!(sigma_eta > 0.0, "sigma_eta must be positive");
        Self { mu, phi, sigma_eta }
    }

    /// Simulate returns from the model
    pub fn simulate<R: Rng>(&self, n: usize, rng: &mut R) -> SimulatedData {
        let mut h = Vec::with_capacity(n);
        let mut returns = Vec::with_capacity(n);

        // Initial log-volatility from stationary distribution
        let h_var = self.sigma_eta.powi(2) / (1.0 - self.phi.powi(2));
        let h_0 = self.mu + h_var.sqrt() * rng.sample::<f64, _>(StandardNormal);
        h.push(h_0);

        for t in 0..n {
            // Update log-volatility
            if t > 0 {
                let eta: f64 = rng.sample(StandardNormal);
                let h_t = self.mu + self.phi * (h[t - 1] - self.mu) + self.sigma_eta * eta;
                h.push(h_t);
            }

            // Generate return
            let epsilon: f64 = rng.sample(StandardNormal);
            let vol = (h[t] / 2.0).exp();
            returns.push(vol * epsilon);
        }

        // Convert log-volatility to volatility
        let volatility: Vec<f64> = h.iter().map(|&ht| (ht / 2.0).exp()).collect();

        SimulatedData {
            returns,
            log_volatility: h,
            volatility,
        }
    }

    /// Fit the model to returns using MCMC
    ///
    /// This uses a simplified particle MCMC approach
    pub fn fit(&self, returns: &[f64], config: &MCMCConfig) -> StochasticVolatilityResult {
        let n = returns.len();
        let returns = returns.to_vec();

        // Estimate initial volatility from returns
        let squared_returns: Vec<f64> = returns.iter().map(|r| r.powi(2)).collect();
        let log_sq_returns: Vec<f64> = squared_returns
            .iter()
            .map(|&sr| (sr + 1e-10).ln())
            .collect();

        // Simple estimate of log-volatility using smoothing
        let mut h_init: Vec<f64> = log_sq_returns.clone();

        // Simple moving average smoothing
        let window = 5.min(n / 3);
        if window > 1 {
            let mut smoothed = Vec::with_capacity(n);
            for i in 0..n {
                let start = i.saturating_sub(window / 2);
                let end = (i + window / 2 + 1).min(n);
                let avg: f64 = h_init[start..end].iter().sum::<f64>() / (end - start) as f64;
                smoothed.push(avg);
            }
            h_init = smoothed;
        }

        // MCMC for parameters given h
        // We use a Gibbs-like approach alternating between h and parameters
        let prior_mu_std = 2.0;
        let prior_phi_shape = 20.0; // Beta(20, 1.5) prior for (phi+1)/2
        let prior_phi_rate = 1.5;
        let prior_sigma_shape = 2.0;
        let prior_sigma_rate = 10.0;

        // Log posterior for parameters given h
        let log_posterior = move |params: &[f64], h: &[f64]| -> f64 {
            let mu = params[0];
            let phi_raw = params[1]; // logit scale
            let phi = 2.0 / (1.0 + (-phi_raw).exp()) - 1.0; // Transform to (-1, 1)
            let log_sigma = params[2];
            let sigma_eta = log_sigma.exp();

            if phi.abs() >= 0.9999 || sigma_eta <= 0.0 {
                return f64::NEG_INFINITY;
            }

            // Prior on mu: Normal(0, prior_mu_std)
            let log_prior_mu = -0.5 * (mu / prior_mu_std).powi(2);

            // Prior on phi: Beta prior on (phi+1)/2
            let phi_transformed = (phi + 1.0) / 2.0;
            let log_prior_phi = (prior_phi_shape - 1.0) * phi_transformed.ln()
                + (prior_phi_rate - 1.0) * (1.0 - phi_transformed).ln();

            // Prior on sigma_eta: InverseGamma
            let log_prior_sigma =
                -(prior_sigma_shape + 1.0) * log_sigma - prior_sigma_rate / sigma_eta + log_sigma;

            // Likelihood for h given parameters (AR(1) process)
            let h_var = sigma_eta.powi(2) / (1.0 - phi.powi(2));
            let mut log_lik_h = -0.5 * ((h[0] - mu).powi(2) / h_var + h_var.ln());

            for t in 1..h.len() {
                let h_mean = mu + phi * (h[t - 1] - mu);
                log_lik_h += -0.5 * ((h[t] - h_mean).powi(2) / sigma_eta.powi(2) + log_sigma * 2.0);
            }

            log_prior_mu + log_prior_phi + log_prior_sigma + log_lik_h
        };

        // Run MCMC
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng: StdRng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let n_total = config.n_warmup + config.n_samples;
        let mut samples = Vec::with_capacity(config.n_samples);
        let mut h_samples = Vec::with_capacity(config.n_samples);

        // Initial parameters
        let h_mean: f64 = h_init.iter().sum::<f64>() / n as f64;
        let mut current_params = vec![h_mean, 2.0, (-2.0_f64).ln()]; // mu, logit(phi), log(sigma)
        let mut current_h = h_init.clone();
        let mut current_log_prob = log_posterior(&current_params, &current_h);

        let proposal_std = vec![0.1, 0.1, 0.1];
        let h_proposal_std = 0.2;

        let mut accepted = 0u64;
        let mut total = 0u64;

        for i in 0..n_total {
            // Update parameters
            let mut proposal_params = current_params.clone();
            for (j, std) in proposal_std.iter().enumerate() {
                proposal_params[j] += std * rng.sample::<f64, _>(StandardNormal);
            }

            let proposal_log_prob = log_posterior(&proposal_params, &current_h);
            let log_alpha = proposal_log_prob - current_log_prob;

            let u: f64 = rng.gen();
            if log_alpha > 0.0 || u.ln() < log_alpha {
                current_params = proposal_params;
                current_log_prob = proposal_log_prob;
                accepted += 1;
            }
            total += 1;

            // Update h (single-site updates for a few positions)
            for _ in 0..10 {
                let t = rng.gen_range(0..n);
                let mut proposal_h = current_h.clone();
                proposal_h[t] += h_proposal_std * rng.sample::<f64, _>(StandardNormal);

                // Log-likelihood ratio for h[t]
                let phi_raw = current_params[1];
                let phi = 2.0 / (1.0 + (-phi_raw).exp()) - 1.0;
                let mu = current_params[0];
                let sigma_eta = current_params[2].exp();

                // Returns likelihood
                let old_vol = (current_h[t] / 2.0).exp();
                let new_vol = (proposal_h[t] / 2.0).exp();

                let log_lik_old = -0.5 * (current_h[t] + returns[t].powi(2) / old_vol.powi(2));
                let log_lik_new = -0.5 * (proposal_h[t] + returns[t].powi(2) / new_vol.powi(2));

                // AR(1) transition
                let mut log_prior_old = 0.0;
                let mut log_prior_new = 0.0;

                if t > 0 {
                    let h_mean_old = mu + phi * (current_h[t - 1] - mu);
                    log_prior_old +=
                        -0.5 * (current_h[t] - h_mean_old).powi(2) / sigma_eta.powi(2);
                    log_prior_new +=
                        -0.5 * (proposal_h[t] - h_mean_old).powi(2) / sigma_eta.powi(2);
                }

                if t < n - 1 {
                    let h_mean_old_next = mu + phi * (current_h[t] - mu);
                    let h_mean_new_next = mu + phi * (proposal_h[t] - mu);
                    log_prior_old +=
                        -0.5 * (current_h[t + 1] - h_mean_old_next).powi(2) / sigma_eta.powi(2);
                    log_prior_new +=
                        -0.5 * (current_h[t + 1] - h_mean_new_next).powi(2) / sigma_eta.powi(2);
                }

                let log_alpha_h =
                    (log_lik_new + log_prior_new) - (log_lik_old + log_prior_old);

                let u: f64 = rng.gen();
                if log_alpha_h > 0.0 || u.ln() < log_alpha_h {
                    current_h = proposal_h;
                }
            }

            // Store samples after warmup
            if i >= config.n_warmup {
                // Transform parameters back
                let mu = current_params[0];
                let phi = 2.0 / (1.0 + (-current_params[1]).exp()) - 1.0;
                let sigma_eta = current_params[2].exp();

                samples.push(vec![mu, phi, sigma_eta]);
                h_samples.push(current_h.clone());
            }

            current_log_prob = log_posterior(&current_params, &current_h);
        }

        // Calculate posterior mean of volatility
        let n_h_samples = h_samples.len();
        let mut mean_volatility = vec![0.0; n];
        for h_sample in &h_samples {
            for (t, &ht) in h_sample.iter().enumerate() {
                mean_volatility[t] += (ht / 2.0).exp();
            }
        }
        for v in &mut mean_volatility {
            *v /= n_h_samples as f64;
        }

        // Credible intervals for volatility
        let mut vol_low = vec![0.0; n];
        let mut vol_high = vec![0.0; n];
        for t in 0..n {
            let mut vols: Vec<f64> = h_samples.iter().map(|h| (h[t] / 2.0).exp()).collect();
            vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let low_idx = (0.025 * vols.len() as f64) as usize;
            let high_idx = (0.975 * vols.len() as f64) as usize;
            vol_low[t] = vols[low_idx];
            vol_high[t] = vols[high_idx.min(vols.len() - 1)];
        }

        // Parameter summaries
        let mu_samples: Vec<f64> = samples.iter().map(|s| s[0]).collect();
        let phi_samples: Vec<f64> = samples.iter().map(|s| s[1]).collect();
        let sigma_samples: Vec<f64> = samples.iter().map(|s| s[2]).collect();

        StochasticVolatilityResult {
            mu_samples,
            phi_samples,
            sigma_eta_samples: sigma_samples,
            mean_volatility,
            volatility_ci_low: vol_low,
            volatility_ci_high: vol_high,
            acceptance_rate: accepted as f64 / total as f64,
        }
    }
}

/// Simulated data from SV model
#[derive(Debug, Clone)]
pub struct SimulatedData {
    /// Simulated returns
    pub returns: Vec<f64>,
    /// True log-volatility path
    pub log_volatility: Vec<f64>,
    /// True volatility path
    pub volatility: Vec<f64>,
}

/// Result of SV model estimation
#[derive(Debug, Clone)]
pub struct StochasticVolatilityResult {
    /// Posterior samples for mu
    pub mu_samples: Vec<f64>,
    /// Posterior samples for phi
    pub phi_samples: Vec<f64>,
    /// Posterior samples for sigma_eta
    pub sigma_eta_samples: Vec<f64>,
    /// Posterior mean of volatility over time
    pub mean_volatility: Vec<f64>,
    /// Lower 95% CI for volatility
    pub volatility_ci_low: Vec<f64>,
    /// Upper 95% CI for volatility
    pub volatility_ci_high: Vec<f64>,
    /// MCMC acceptance rate
    pub acceptance_rate: f64,
}

impl StochasticVolatilityResult {
    /// Get posterior mean of mu
    pub fn mu_mean(&self) -> f64 {
        self.mu_samples.iter().sum::<f64>() / self.mu_samples.len() as f64
    }

    /// Get posterior mean of phi
    pub fn phi_mean(&self) -> f64 {
        self.phi_samples.iter().sum::<f64>() / self.phi_samples.len() as f64
    }

    /// Get posterior mean of sigma_eta
    pub fn sigma_eta_mean(&self) -> f64 {
        self.sigma_eta_samples.iter().sum::<f64>() / self.sigma_eta_samples.len() as f64
    }

    /// Calculate credible interval for a parameter
    fn ci(samples: &[f64], level: f64) -> (f64, f64) {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha_half = (1.0 - level) / 2.0;
        let low_idx = (alpha_half * sorted.len() as f64) as usize;
        let high_idx = ((1.0 - alpha_half) * sorted.len() as f64) as usize;
        (sorted[low_idx], sorted[high_idx.min(sorted.len() - 1)])
    }

    /// Print summary
    pub fn summary(&self) {
        println!("\nStochastic Volatility Model Summary");
        println!("{:-<50}", "");
        println!("MCMC acceptance rate: {:.1}%", self.acceptance_rate * 100.0);
        println!();
        println!("Parameters:");
        println!(
            "  mu (mean log-vol):     {:>8.4} (95% CI: [{:.4}, {:.4}])",
            self.mu_mean(),
            Self::ci(&self.mu_samples, 0.95).0,
            Self::ci(&self.mu_samples, 0.95).1
        );
        println!(
            "  phi (persistence):     {:>8.4} (95% CI: [{:.4}, {:.4}])",
            self.phi_mean(),
            Self::ci(&self.phi_samples, 0.95).0,
            Self::ci(&self.phi_samples, 0.95).1
        );
        println!(
            "  sigma_eta (vol of vol):{:>8.4} (95% CI: [{:.4}, {:.4}])",
            self.sigma_eta_mean(),
            Self::ci(&self.sigma_eta_samples, 0.95).0,
            Self::ci(&self.sigma_eta_samples, 0.95).1
        );
        println!();
        println!("Volatility (mean across time): {:.4}",
            self.mean_volatility.iter().sum::<f64>() / self.mean_volatility.len() as f64
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_sv_simulation() {
        let model = StochasticVolatility::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data = model.simulate(100, &mut rng);

        assert_eq!(data.returns.len(), 100);
        assert_eq!(data.volatility.len(), 100);

        // Volatility should be positive
        assert!(data.volatility.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_sv_estimation() {
        // Generate data
        let true_model = StochasticVolatility::new(-4.0, 0.9, 0.3);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data = true_model.simulate(200, &mut rng);

        // Fit model
        let model = StochasticVolatility::default();
        let config = MCMCConfig::new(1000).with_warmup(500).with_seed(123);
        let result = model.fit(&data.returns, &config);

        // Check that estimated phi is positive (high persistence)
        assert!(result.phi_mean() > 0.0, "phi = {}", result.phi_mean());
    }
}
