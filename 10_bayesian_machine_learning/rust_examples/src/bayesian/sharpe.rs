//! Bayesian Sharpe Ratio estimation.
//!
//! Implements Bayesian estimation of the Sharpe ratio using a robust
//! Student-t likelihood, following Kruschke's BEST approach.
//!
//! This provides:
//! - Full posterior distribution of Sharpe ratio
//! - Credible intervals
//! - Probability of outperformance comparisons

use super::distributions::Distribution;
use super::inference::{AdaptiveMetropolisHastings, MCMCConfig, MCMCSamples};

/// Bayesian Sharpe Ratio estimator
///
/// Uses a Student-t likelihood for robustness to outliers:
/// returns ~ StudentT(nu, mu, sigma)
///
/// Priors:
/// - mu ~ Normal(0, prior_mu_std)
/// - sigma ~ HalfCauchy(prior_sigma_scale)
/// - nu ~ Exponential(1/29) + 1 (shifted to ensure nu > 1)
#[derive(Debug, Clone)]
pub struct BayesianSharpe {
    /// Annualization factor (e.g., 252 for daily, 52 for weekly)
    pub annualization: f64,
    /// Prior std for mean
    pub prior_mu_std: f64,
    /// Prior scale for sigma
    pub prior_sigma_scale: f64,
}

impl Default for BayesianSharpe {
    fn default() -> Self {
        Self {
            annualization: 252.0, // Daily returns
            prior_mu_std: 0.1,
            prior_sigma_scale: 0.1,
        }
    }
}

impl BayesianSharpe {
    /// Create a new Bayesian Sharpe estimator
    pub fn new(annualization: f64) -> Self {
        Self {
            annualization,
            ..Default::default()
        }
    }

    /// Set prior for mean
    pub fn with_prior_mu_std(mut self, std: f64) -> Self {
        self.prior_mu_std = std;
        self
    }

    /// Set prior for sigma
    pub fn with_prior_sigma_scale(mut self, scale: f64) -> Self {
        self.prior_sigma_scale = scale;
        self
    }

    /// Estimate Sharpe ratio from returns
    pub fn estimate(&self, returns: &[f64], config: &MCMCConfig) -> BayesianSharpeResult {
        let returns = returns.to_vec();
        let prior_mu_std = self.prior_mu_std;
        let prior_sigma_scale = self.prior_sigma_scale;

        // Log posterior function
        // params: [mu, log_sigma, log_nu_minus_1]
        let log_posterior = move |params: &[f64]| {
            let mu = params[0];
            let log_sigma = params[1];
            let sigma = log_sigma.exp();
            let log_nu_minus_1 = params[2];
            let nu = log_nu_minus_1.exp() + 1.0; // nu > 1

            // Prior on mu: Normal(0, prior_mu_std)
            let log_prior_mu = -0.5 * (mu / prior_mu_std).powi(2);

            // Prior on sigma: Half-Cauchy(prior_sigma_scale)
            let log_prior_sigma = -(1.0 + (sigma / prior_sigma_scale).powi(2)).ln() + log_sigma;

            // Prior on nu: Exponential(1/29) shifted by 1
            let log_prior_nu = -(nu - 1.0) / 29.0 + log_nu_minus_1;

            // Likelihood: Student-t
            let log_likelihood: f64 = returns
                .iter()
                .map(|&r| student_t_log_pdf(r, mu, sigma, nu))
                .sum();

            if log_likelihood.is_finite() {
                log_prior_mu + log_prior_sigma + log_prior_nu + log_likelihood
            } else {
                f64::NEG_INFINITY
            }
        };

        // Initial values from data
        let data_mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let data_var: f64 = returns.iter().map(|r| (r - data_mean).powi(2)).sum::<f64>()
            / (returns.len() - 1) as f64;
        let data_std = data_var.sqrt();

        let initial = vec![
            data_mean,
            data_std.ln(),
            (30.0_f64 - 1.0).ln(), // nu = 30 initially
        ];

        let param_names = vec![
            "mu".to_string(),
            "log_sigma".to_string(),
            "log_nu_minus_1".to_string(),
        ];

        let sampler = AdaptiveMetropolisHastings::new(log_posterior, initial, param_names)
            .with_initial_std(vec![data_std * 0.1, 0.5, 0.5]);

        let samples = sampler.run(config);

        // Transform samples
        let mu_samples = samples.get_param(0);
        let sigma_samples: Vec<f64> = samples.get_param(1).iter().map(|&x| x.exp()).collect();
        let nu_samples: Vec<f64> = samples
            .get_param(2)
            .iter()
            .map(|&x| x.exp() + 1.0)
            .collect();

        // Calculate Sharpe ratio samples
        let annualization = self.annualization;
        let sharpe_samples: Vec<f64> = mu_samples
            .iter()
            .zip(sigma_samples.iter())
            .map(|(&mu, &sigma)| {
                if sigma > 0.0 {
                    (mu * annualization) / (sigma * annualization.sqrt())
                } else {
                    0.0
                }
            })
            .collect();

        BayesianSharpeResult {
            mu_samples,
            sigma_samples,
            nu_samples,
            sharpe_samples,
            annualization,
            acceptance_rate: samples.acceptance_rate,
        }
    }

    /// Compare two return series and compute probability of outperformance
    pub fn compare(
        &self,
        returns1: &[f64],
        returns2: &[f64],
        config: &MCMCConfig,
    ) -> BayesianSharpeComparison {
        let result1 = self.estimate(returns1, config);
        let result2 = self.estimate(returns2, config);

        // Difference in Sharpe ratios
        let n = result1.sharpe_samples.len().min(result2.sharpe_samples.len());
        let sharpe_diff: Vec<f64> = result1.sharpe_samples[..n]
            .iter()
            .zip(result2.sharpe_samples[..n].iter())
            .map(|(&s1, &s2)| s1 - s2)
            .collect();

        // Probability that strategy 1 has higher Sharpe
        let prob_1_better = sharpe_diff.iter().filter(|&&d| d > 0.0).count() as f64 / n as f64;

        // Effect size (standardized difference)
        let mean_diff: f64 = sharpe_diff.iter().sum::<f64>() / n as f64;
        let std_diff: f64 = {
            let var: f64 = sharpe_diff.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / (n - 1) as f64;
            var.sqrt()
        };
        let effect_size = if std_diff > 0.0 {
            mean_diff / std_diff
        } else {
            0.0
        };

        BayesianSharpeComparison {
            result1,
            result2,
            sharpe_diff,
            prob_1_better,
            effect_size,
        }
    }
}

/// Result of Bayesian Sharpe ratio estimation
#[derive(Debug, Clone)]
pub struct BayesianSharpeResult {
    /// Posterior samples for mean return
    pub mu_samples: Vec<f64>,
    /// Posterior samples for standard deviation
    pub sigma_samples: Vec<f64>,
    /// Posterior samples for degrees of freedom
    pub nu_samples: Vec<f64>,
    /// Posterior samples for Sharpe ratio
    pub sharpe_samples: Vec<f64>,
    /// Annualization factor used
    pub annualization: f64,
    /// MCMC acceptance rate
    pub acceptance_rate: f64,
}

impl BayesianSharpeResult {
    /// Get posterior mean of Sharpe ratio
    pub fn sharpe_mean(&self) -> f64 {
        self.sharpe_samples.iter().sum::<f64>() / self.sharpe_samples.len() as f64
    }

    /// Get posterior std of Sharpe ratio
    pub fn sharpe_std(&self) -> f64 {
        let mean = self.sharpe_mean();
        let var: f64 = self
            .sharpe_samples
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>()
            / (self.sharpe_samples.len() - 1) as f64;
        var.sqrt()
    }

    /// Get credible interval for Sharpe ratio
    pub fn sharpe_ci(&self, level: f64) -> (f64, f64) {
        let mut sorted = self.sharpe_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha_half = (1.0 - level) / 2.0;
        let low_idx = (alpha_half * sorted.len() as f64) as usize;
        let high_idx = ((1.0 - alpha_half) * sorted.len() as f64) as usize;

        (sorted[low_idx], sorted[high_idx.min(sorted.len() - 1)])
    }

    /// Probability that Sharpe ratio is positive
    pub fn prob_positive(&self) -> f64 {
        self.sharpe_samples.iter().filter(|&&s| s > 0.0).count() as f64
            / self.sharpe_samples.len() as f64
    }

    /// Probability that Sharpe ratio exceeds threshold
    pub fn prob_exceeds(&self, threshold: f64) -> f64 {
        self.sharpe_samples
            .iter()
            .filter(|&&s| s > threshold)
            .count() as f64
            / self.sharpe_samples.len() as f64
    }

    /// Print summary
    pub fn summary(&self) {
        let ci = self.sharpe_ci(0.95);
        println!("\nBayesian Sharpe Ratio Summary");
        println!("{:-<50}", "");
        println!("Annualization factor: {}", self.annualization);
        println!("MCMC acceptance rate: {:.1}%", self.acceptance_rate * 100.0);
        println!();
        println!("Sharpe Ratio:");
        println!("  Mean:           {:>10.4}", self.sharpe_mean());
        println!("  Std:            {:>10.4}", self.sharpe_std());
        println!("  95% CI:         [{:.4}, {:.4}]", ci.0, ci.1);
        println!("  P(SR > 0):      {:>10.1}%", self.prob_positive() * 100.0);
        println!();
        println!("Mean Return (mu):");
        let mu_mean: f64 = self.mu_samples.iter().sum::<f64>() / self.mu_samples.len() as f64;
        println!("  Mean:           {:>10.6}", mu_mean);
        println!();
        println!("Volatility (sigma):");
        let sigma_mean: f64 =
            self.sigma_samples.iter().sum::<f64>() / self.sigma_samples.len() as f64;
        println!("  Mean:           {:>10.6}", sigma_mean);
    }
}

/// Result of comparing two strategies
#[derive(Debug, Clone)]
pub struct BayesianSharpeComparison {
    /// Results for first strategy
    pub result1: BayesianSharpeResult,
    /// Results for second strategy
    pub result2: BayesianSharpeResult,
    /// Posterior samples of Sharpe difference (strategy 1 - strategy 2)
    pub sharpe_diff: Vec<f64>,
    /// Probability that strategy 1 has higher Sharpe ratio
    pub prob_1_better: f64,
    /// Effect size (Cohen's d of the difference)
    pub effect_size: f64,
}

impl BayesianSharpeComparison {
    /// Get credible interval for difference
    pub fn diff_ci(&self, level: f64) -> (f64, f64) {
        let mut sorted = self.sharpe_diff.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha_half = (1.0 - level) / 2.0;
        let low_idx = (alpha_half * sorted.len() as f64) as usize;
        let high_idx = ((1.0 - alpha_half) * sorted.len() as f64) as usize;

        (sorted[low_idx], sorted[high_idx.min(sorted.len() - 1)])
    }

    /// Print comparison summary
    pub fn summary(&self) {
        let ci = self.diff_ci(0.95);
        let mean_diff: f64 = self.sharpe_diff.iter().sum::<f64>() / self.sharpe_diff.len() as f64;

        println!("\nBayesian Sharpe Ratio Comparison");
        println!("{:-<50}", "");
        println!("Strategy 1 Sharpe: {:.4} (95% CI: [{:.4}, {:.4}])",
            self.result1.sharpe_mean(),
            self.result1.sharpe_ci(0.95).0,
            self.result1.sharpe_ci(0.95).1
        );
        println!("Strategy 2 Sharpe: {:.4} (95% CI: [{:.4}, {:.4}])",
            self.result2.sharpe_mean(),
            self.result2.sharpe_ci(0.95).0,
            self.result2.sharpe_ci(0.95).1
        );
        println!();
        println!("Difference (Strategy 1 - Strategy 2):");
        println!("  Mean:            {:>10.4}", mean_diff);
        println!("  95% CI:          [{:.4}, {:.4}]", ci.0, ci.1);
        println!("  P(S1 > S2):      {:>10.1}%", self.prob_1_better * 100.0);
        println!("  Effect Size:     {:>10.4}", self.effect_size);
    }
}

/// Log PDF of Student-t distribution
fn student_t_log_pdf(x: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;

    let z = (x - mu) / sigma;
    ln_gamma((nu + 1.0) / 2.0)
        - ln_gamma(nu / 2.0)
        - 0.5 * (nu * std::f64::consts::PI).ln()
        - sigma.ln()
        - ((nu + 1.0) / 2.0) * (1.0 + z.powi(2) / nu).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::Distribution;

    #[test]
    fn test_bayesian_sharpe() {
        // Generate returns with known mean and std
        let mut rng = StdRng::seed_from_u64(42);
        let normal = rand_distr::Normal::new(0.001, 0.02).unwrap();
        let returns: Vec<f64> = (0..500).map(|_| normal.sample(&mut rng)).collect();

        let estimator = BayesianSharpe::new(252.0);
        let config = MCMCConfig::new(2000).with_warmup(500).with_seed(42);
        let result = estimator.estimate(&returns, &config);

        // Expected Sharpe ~ 0.001 * 252 / (0.02 * sqrt(252)) ~ 0.79
        let expected_sharpe = 0.001 * 252.0 / (0.02 * 252.0_f64.sqrt());
        let estimated_sharpe = result.sharpe_mean();

        // Allow some tolerance due to MCMC sampling
        assert!(
            (estimated_sharpe - expected_sharpe).abs() < 0.5,
            "Expected ~{:.2}, got {:.2}",
            expected_sharpe,
            estimated_sharpe
        );
    }
}
