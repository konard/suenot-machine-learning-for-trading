//! Bayesian inference methods.
//!
//! Implements:
//! - Conjugate prior updates
//! - Metropolis-Hastings MCMC sampling
//! - Summary statistics for posterior samples

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Trait for conjugate prior updates
pub trait ConjugatePrior {
    /// Type of observed data
    type Observation;

    /// Update the prior with a single observation
    fn update(&self, observation: &Self::Observation) -> Self;

    /// Update the prior with multiple observations
    fn update_batch(&self, observations: &[Self::Observation]) -> Self
    where
        Self: Sized + Clone,
    {
        let mut result = self.clone();
        for obs in observations {
            result = result.update(obs);
        }
        result
    }
}

/// MCMC sampling configuration
#[derive(Debug, Clone)]
pub struct MCMCConfig {
    /// Number of samples to generate
    pub n_samples: usize,
    /// Number of warmup (burn-in) samples to discard
    pub n_warmup: usize,
    /// Thinning factor (keep every nth sample)
    pub thin: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for MCMCConfig {
    fn default() -> Self {
        Self {
            n_samples: 10000,
            n_warmup: 1000,
            thin: 1,
            seed: None,
        }
    }
}

impl MCMCConfig {
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            ..Default::default()
        }
    }

    pub fn with_warmup(mut self, n_warmup: usize) -> Self {
        self.n_warmup = n_warmup;
        self
    }

    pub fn with_thin(mut self, thin: usize) -> Self {
        self.thin = thin;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// MCMC sampling results
#[derive(Debug, Clone)]
pub struct MCMCSamples {
    /// Parameter names
    pub param_names: Vec<String>,
    /// Samples matrix: rows = samples, cols = parameters
    pub samples: Vec<Vec<f64>>,
    /// Acceptance rate
    pub acceptance_rate: f64,
}

impl MCMCSamples {
    /// Get samples for a specific parameter by index
    pub fn get_param(&self, idx: usize) -> Vec<f64> {
        self.samples.iter().map(|s| s[idx]).collect()
    }

    /// Get samples for a specific parameter by name
    pub fn get_param_by_name(&self, name: &str) -> Option<Vec<f64>> {
        self.param_names
            .iter()
            .position(|n| n == name)
            .map(|idx| self.get_param(idx))
    }

    /// Calculate mean for each parameter
    pub fn means(&self) -> Vec<f64> {
        let n = self.samples.len() as f64;
        let n_params = self.param_names.len();
        let mut means = vec![0.0; n_params];

        for sample in &self.samples {
            for (i, &val) in sample.iter().enumerate() {
                means[i] += val;
            }
        }

        for mean in &mut means {
            *mean /= n;
        }

        means
    }

    /// Calculate standard deviation for each parameter
    pub fn stds(&self) -> Vec<f64> {
        let means = self.means();
        let n = self.samples.len() as f64;
        let n_params = self.param_names.len();
        let mut vars = vec![0.0; n_params];

        for sample in &self.samples {
            for (i, &val) in sample.iter().enumerate() {
                vars[i] += (val - means[i]).powi(2);
            }
        }

        vars.iter().map(|v| (v / (n - 1.0)).sqrt()).collect()
    }

    /// Calculate credible intervals for each parameter
    pub fn credible_intervals(&self, level: f64) -> Vec<(f64, f64)> {
        let n_params = self.param_names.len();
        let alpha_half = (1.0 - level) / 2.0;
        let low_idx = (alpha_half * self.samples.len() as f64) as usize;
        let high_idx = ((1.0 - alpha_half) * self.samples.len() as f64) as usize;

        let mut intervals = Vec::with_capacity(n_params);

        for i in 0..n_params {
            let mut param_samples: Vec<f64> = self.get_param(i);
            param_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            intervals.push((param_samples[low_idx], param_samples[high_idx.min(param_samples.len() - 1)]));
        }

        intervals
    }

    /// Calculate effective sample size (simple autocorrelation-based estimate)
    pub fn effective_sample_size(&self, param_idx: usize) -> f64 {
        let samples = self.get_param(param_idx);
        let n = samples.len();
        if n < 10 {
            return n as f64;
        }

        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let var: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        if var == 0.0 {
            return n as f64;
        }

        // Calculate autocorrelation at lag 1
        let mut autocorr = 0.0;
        for i in 0..(n - 1) {
            autocorr += (samples[i] - mean) * (samples[i + 1] - mean);
        }
        autocorr /= (n - 1) as f64 * var;

        // Approximate ESS
        let ess = n as f64 / (1.0 + 2.0 * autocorr.abs());
        ess.max(1.0)
    }

    /// Print summary statistics
    pub fn summary(&self) {
        let means = self.means();
        let stds = self.stds();
        let cis = self.credible_intervals(0.95);

        println!("\nMCMC Summary (n = {}, acceptance = {:.1}%)",
            self.samples.len(),
            self.acceptance_rate * 100.0
        );
        println!("{:>15} {:>12} {:>12} {:>20}", "Parameter", "Mean", "Std", "95% CI");
        println!("{:-<60}", "");

        for (i, name) in self.param_names.iter().enumerate() {
            println!(
                "{:>15} {:>12.4} {:>12.4} [{:>8.4}, {:>8.4}]",
                name, means[i], stds[i], cis[i].0, cis[i].1
            );
        }
    }
}

/// Trait for MCMC samplers
pub trait MCMC {
    /// Sample from the posterior
    fn sample(&self, config: &MCMCConfig) -> MCMCSamples;
}

/// Metropolis-Hastings sampler
pub struct MetropolisHastings<F>
where
    F: Fn(&[f64]) -> f64,
{
    /// Log-posterior function
    log_posterior: F,
    /// Initial parameter values
    initial: Vec<f64>,
    /// Proposal standard deviations
    proposal_std: Vec<f64>,
    /// Parameter names
    param_names: Vec<String>,
}

impl<F> MetropolisHastings<F>
where
    F: Fn(&[f64]) -> f64,
{
    /// Create a new Metropolis-Hastings sampler
    pub fn new(
        log_posterior: F,
        initial: Vec<f64>,
        proposal_std: Vec<f64>,
        param_names: Vec<String>,
    ) -> Self {
        assert_eq!(initial.len(), proposal_std.len());
        assert_eq!(initial.len(), param_names.len());
        Self {
            log_posterior,
            initial,
            proposal_std,
            param_names,
        }
    }

    /// Run the sampler
    pub fn run(&self, config: &MCMCConfig) -> MCMCSamples {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng: StdRng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let n_total = config.n_warmup + config.n_samples * config.thin;
        let n_params = self.initial.len();

        let mut current = self.initial.clone();
        let mut current_log_prob = (self.log_posterior)(&current);
        let mut samples = Vec::with_capacity(config.n_samples);
        let mut accepted = 0u64;
        let mut total = 0u64;

        for i in 0..n_total {
            // Propose new state
            let mut proposal = current.clone();
            for (j, std) in self.proposal_std.iter().enumerate() {
                let noise: f64 = rng.sample(StandardNormal);
                proposal[j] += std * noise;
            }

            // Calculate acceptance probability
            let proposal_log_prob = (self.log_posterior)(&proposal);
            let log_alpha = proposal_log_prob - current_log_prob;

            // Accept or reject
            let u: f64 = rng.gen();
            if log_alpha > 0.0 || u.ln() < log_alpha {
                current = proposal;
                current_log_prob = proposal_log_prob;
                accepted += 1;
            }
            total += 1;

            // Store sample (after warmup, with thinning)
            if i >= config.n_warmup && (i - config.n_warmup) % config.thin == 0 {
                samples.push(current.clone());
            }
        }

        MCMCSamples {
            param_names: self.param_names.clone(),
            samples,
            acceptance_rate: accepted as f64 / total as f64,
        }
    }
}

/// Adaptive Metropolis-Hastings with automatic proposal scaling
pub struct AdaptiveMetropolisHastings<F>
where
    F: Fn(&[f64]) -> f64,
{
    /// Log-posterior function
    log_posterior: F,
    /// Initial parameter values
    initial: Vec<f64>,
    /// Initial proposal standard deviations
    initial_proposal_std: Vec<f64>,
    /// Parameter names
    param_names: Vec<String>,
    /// Target acceptance rate
    target_acceptance: f64,
}

impl<F> AdaptiveMetropolisHastings<F>
where
    F: Fn(&[f64]) -> f64,
{
    pub fn new(
        log_posterior: F,
        initial: Vec<f64>,
        param_names: Vec<String>,
    ) -> Self {
        let n_params = initial.len();
        Self {
            log_posterior,
            initial: initial.clone(),
            initial_proposal_std: vec![1.0; n_params],
            param_names,
            target_acceptance: 0.234, // Optimal for multivariate
        }
    }

    pub fn with_initial_std(mut self, std: Vec<f64>) -> Self {
        self.initial_proposal_std = std;
        self
    }

    pub fn run(&self, config: &MCMCConfig) -> MCMCSamples {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng: StdRng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let n_total = config.n_warmup + config.n_samples * config.thin;
        let mut proposal_std = self.initial_proposal_std.clone();

        let mut current = self.initial.clone();
        let mut current_log_prob = (self.log_posterior)(&current);
        let mut samples = Vec::with_capacity(config.n_samples);
        let mut accepted = 0u64;
        let mut total = 0u64;

        // Adaptation windows
        let adapt_window = 100;
        let mut window_accepted = 0;

        for i in 0..n_total {
            // Propose new state
            let mut proposal = current.clone();
            for (j, std) in proposal_std.iter().enumerate() {
                let noise: f64 = rng.sample(StandardNormal);
                proposal[j] += std * noise;
            }

            // Calculate acceptance probability
            let proposal_log_prob = (self.log_posterior)(&proposal);
            let log_alpha = proposal_log_prob - current_log_prob;

            // Accept or reject
            let u: f64 = rng.gen();
            if log_alpha > 0.0 || u.ln() < log_alpha {
                current = proposal;
                current_log_prob = proposal_log_prob;
                accepted += 1;
                window_accepted += 1;
            }
            total += 1;

            // Adapt proposal during warmup
            if i < config.n_warmup && (i + 1) % adapt_window == 0 {
                let window_rate = window_accepted as f64 / adapt_window as f64;
                let scale = if window_rate > self.target_acceptance {
                    1.1 // Increase step size
                } else {
                    0.9 // Decrease step size
                };
                for std in &mut proposal_std {
                    *std *= scale;
                    *std = std.clamp(0.001, 100.0);
                }
                window_accepted = 0;
            }

            // Store sample (after warmup, with thinning)
            if i >= config.n_warmup && (i - config.n_warmup) % config.thin == 0 {
                samples.push(current.clone());
            }
        }

        MCMCSamples {
            param_names: self.param_names.clone(),
            samples,
            acceptance_rate: accepted as f64 / total as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metropolis_hastings_normal() {
        // Sample from N(2, 1)
        let log_posterior = |params: &[f64]| {
            let mu = params[0];
            -0.5 * (mu - 2.0).powi(2)
        };

        let mh = MetropolisHastings::new(
            log_posterior,
            vec![0.0],
            vec![1.0],
            vec!["mu".to_string()],
        );

        let config = MCMCConfig::new(5000).with_warmup(1000).with_seed(42);
        let samples = mh.run(&config);

        let mean = samples.means()[0];
        assert!((mean - 2.0).abs() < 0.1, "Mean was {}", mean);
    }

    #[test]
    fn test_mcmc_samples_summary() {
        let samples = MCMCSamples {
            param_names: vec!["a".to_string(), "b".to_string()],
            samples: vec![
                vec![1.0, 2.0],
                vec![1.1, 2.1],
                vec![0.9, 1.9],
                vec![1.0, 2.0],
            ],
            acceptance_rate: 0.5,
        };

        let means = samples.means();
        assert!((means[0] - 1.0).abs() < 0.1);
        assert!((means[1] - 2.0).abs() < 0.1);
    }
}
