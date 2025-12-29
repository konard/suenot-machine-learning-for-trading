//! Probability distributions for Bayesian inference.
//!
//! Implements Beta, Normal, and Student-t distributions with
//! PDF, sampling, and statistical moments.

use rand::Rng;
use rand_distr::{Distribution as RandDistribution, StandardNormal};
use std::f64::consts::PI;

/// Trait for probability distributions
pub trait Distribution {
    /// Probability density function
    fn pdf(&self, x: f64) -> f64;

    /// Log probability density function
    fn log_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }

    /// Mean of the distribution
    fn mean(&self) -> f64;

    /// Variance of the distribution
    fn variance(&self) -> f64;

    /// Standard deviation
    fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sample from the distribution
    fn sample<R: Rng>(&self, rng: &mut R) -> f64;

    /// Sample multiple values
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

/// Beta distribution for probabilities
#[derive(Debug, Clone, Copy)]
pub struct Beta {
    /// Alpha parameter (successes + prior)
    pub alpha: f64,
    /// Beta parameter (failures + prior)
    pub beta: f64,
}

impl Beta {
    /// Create a new Beta distribution
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0, "Alpha must be positive");
        assert!(beta > 0.0, "Beta must be positive");
        Self { alpha, beta }
    }

    /// Create a uniform prior (Beta(1, 1))
    pub fn uniform() -> Self {
        Self::new(1.0, 1.0)
    }

    /// Create Jeffreys prior (Beta(0.5, 0.5))
    pub fn jeffreys() -> Self {
        Self::new(0.5, 0.5)
    }

    /// Update with observed successes and failures (conjugate update)
    pub fn update(&self, successes: u64, failures: u64) -> Self {
        Self {
            alpha: self.alpha + successes as f64,
            beta: self.beta + failures as f64,
        }
    }

    /// Mode of the distribution (most likely value)
    pub fn mode(&self) -> f64 {
        if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        } else if self.alpha <= 1.0 && self.beta > 1.0 {
            0.0
        } else if self.alpha > 1.0 && self.beta <= 1.0 {
            1.0
        } else {
            0.5 // Undefined, return midpoint
        }
    }

    /// Credible interval at given level (e.g., 0.95 for 95%)
    pub fn credible_interval(&self, level: f64) -> (f64, f64) {
        let alpha_half = (1.0 - level) / 2.0;
        let low = self.quantile(alpha_half);
        let high = self.quantile(1.0 - alpha_half);
        (low, high)
    }

    /// Quantile function (inverse CDF) - approximate using Newton's method
    pub fn quantile(&self, p: f64) -> f64 {
        use statrs::distribution::{Beta as StatrsBeta, ContinuousCDF};
        let dist = StatrsBeta::new(self.alpha, self.beta).unwrap();
        dist.inverse_cdf(p)
    }
}

impl Distribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x > 1.0 {
            return 0.0;
        }
        use statrs::function::gamma::ln_gamma;
        let log_beta = ln_gamma(self.alpha) + ln_gamma(self.beta) - ln_gamma(self.alpha + self.beta);
        let log_pdf = (self.alpha - 1.0) * x.ln() + (self.beta - 1.0) * (1.0 - x).ln() - log_beta;
        log_pdf.exp()
    }

    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        (self.alpha * self.beta) / (ab.powi(2) * (ab + 1.0))
    }

    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        use rand_distr::Beta as RandBeta;
        let dist = RandBeta::new(self.alpha, self.beta).unwrap();
        dist.sample(rng)
    }
}

/// Normal (Gaussian) distribution
#[derive(Debug, Clone, Copy)]
pub struct Normal {
    /// Mean
    pub mu: f64,
    /// Standard deviation
    pub sigma: f64,
}

impl Normal {
    /// Create a new Normal distribution
    pub fn new(mu: f64, sigma: f64) -> Self {
        assert!(sigma > 0.0, "Sigma must be positive");
        Self { mu, sigma }
    }

    /// Standard normal distribution N(0, 1)
    pub fn standard() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Credible interval
    pub fn credible_interval(&self, level: f64) -> (f64, f64) {
        use statrs::distribution::{ContinuousCDF, Normal as StatrsNormal};
        let dist = StatrsNormal::new(self.mu, self.sigma).unwrap();
        let alpha_half = (1.0 - level) / 2.0;
        (dist.inverse_cdf(alpha_half), dist.inverse_cdf(1.0 - alpha_half))
    }

    /// CDF - probability that X <= x
    pub fn cdf(&self, x: f64) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal as StatrsNormal};
        let dist = StatrsNormal::new(self.mu, self.sigma).unwrap();
        dist.cdf(x)
    }
}

impl Distribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        (1.0 / (self.sigma * (2.0 * PI).sqrt())) * (-0.5 * z.powi(2)).exp()
    }

    fn mean(&self) -> f64 {
        self.mu
    }

    fn variance(&self) -> f64 {
        self.sigma.powi(2)
    }

    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        let z: f64 = rng.sample(StandardNormal);
        self.mu + self.sigma * z
    }
}

/// Student-t distribution (useful for robust inference)
#[derive(Debug, Clone, Copy)]
pub struct StudentT {
    /// Degrees of freedom
    pub nu: f64,
    /// Location parameter
    pub mu: f64,
    /// Scale parameter
    pub sigma: f64,
}

impl StudentT {
    /// Create a new Student-t distribution
    pub fn new(nu: f64, mu: f64, sigma: f64) -> Self {
        assert!(nu > 0.0, "Degrees of freedom must be positive");
        assert!(sigma > 0.0, "Scale must be positive");
        Self { nu, mu, sigma }
    }

    /// Standard Student-t with given degrees of freedom
    pub fn standard(nu: f64) -> Self {
        Self::new(nu, 0.0, 1.0)
    }
}

impl Distribution for StudentT {
    fn pdf(&self, x: f64) -> f64 {
        use statrs::function::gamma::ln_gamma;
        let z = (x - self.mu) / self.sigma;
        let log_coef = ln_gamma((self.nu + 1.0) / 2.0)
            - ln_gamma(self.nu / 2.0)
            - 0.5 * (self.nu * PI).ln()
            - self.sigma.ln();
        let log_kernel = -((self.nu + 1.0) / 2.0) * (1.0 + z.powi(2) / self.nu).ln();
        (log_coef + log_kernel).exp()
    }

    fn mean(&self) -> f64 {
        if self.nu > 1.0 {
            self.mu
        } else {
            f64::NAN
        }
    }

    fn variance(&self) -> f64 {
        if self.nu > 2.0 {
            self.sigma.powi(2) * self.nu / (self.nu - 2.0)
        } else {
            f64::INFINITY
        }
    }

    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        use rand_distr::StudentT as RandStudentT;
        let dist = RandStudentT::new(self.nu).unwrap();
        self.mu + self.sigma * dist.sample(rng)
    }
}

/// Inverse Gamma distribution (conjugate prior for variance)
#[derive(Debug, Clone, Copy)]
pub struct InverseGamma {
    /// Shape parameter
    pub alpha: f64,
    /// Scale parameter
    pub beta: f64,
}

impl InverseGamma {
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0, "Alpha must be positive");
        assert!(beta > 0.0, "Beta must be positive");
        Self { alpha, beta }
    }

    /// Weakly informative prior
    pub fn weakly_informative() -> Self {
        Self::new(0.001, 0.001)
    }
}

impl Distribution for InverseGamma {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        use statrs::function::gamma::ln_gamma;
        let log_pdf = self.alpha * self.beta.ln()
            - ln_gamma(self.alpha)
            - (self.alpha + 1.0) * x.ln()
            - self.beta / x;
        log_pdf.exp()
    }

    fn mean(&self) -> f64 {
        if self.alpha > 1.0 {
            self.beta / (self.alpha - 1.0)
        } else {
            f64::INFINITY
        }
    }

    fn variance(&self) -> f64 {
        if self.alpha > 2.0 {
            self.beta.powi(2) / ((self.alpha - 1.0).powi(2) * (self.alpha - 2.0))
        } else {
            f64::INFINITY
        }
    }

    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        use rand_distr::Gamma;
        let gamma = Gamma::new(self.alpha, 1.0 / self.beta).unwrap();
        1.0 / gamma.sample(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_beta_distribution() {
        let beta = Beta::new(2.0, 5.0);
        assert!((beta.mean() - 2.0 / 7.0).abs() < 1e-10);

        let mut rng = StdRng::seed_from_u64(42);
        let samples = beta.sample_n(&mut rng, 1000);
        assert!(samples.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_beta_conjugate_update() {
        let prior = Beta::uniform();
        let posterior = prior.update(10, 5);
        assert_eq!(posterior.alpha, 11.0);
        assert_eq!(posterior.beta, 6.0);
    }

    #[test]
    fn test_normal_distribution() {
        let normal = Normal::new(5.0, 2.0);
        assert_eq!(normal.mean(), 5.0);
        assert_eq!(normal.variance(), 4.0);
    }

    #[test]
    fn test_student_t() {
        let t = StudentT::new(30.0, 0.0, 1.0);
        // With high df, should be close to normal
        let normal = Normal::standard();

        let mut rng = StdRng::seed_from_u64(42);
        let t_samples: Vec<f64> = (0..10000).map(|_| t.sample(&mut rng)).collect();
        let mean: f64 = t_samples.iter().sum::<f64>() / 10000.0;
        assert!(mean.abs() < 0.1);
    }
}
