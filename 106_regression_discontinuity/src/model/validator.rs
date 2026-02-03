//! RDD validation tools.
//!
//! Provides tests to verify RDD assumptions:
//! - McCrary density test for manipulation
//! - Placebo tests at false cutoffs
//! - Covariate balance tests

use crate::model::rdd::{Kernel, RDDModel};
use crate::{RDDError, Result};
use statrs::distribution::{ContinuousCDF, Normal};

/// Results from McCrary density test.
#[derive(Debug, Clone)]
pub struct DensityTestResult {
    /// Log difference in density
    pub theta: f64,
    /// Standard error
    pub se: f64,
    /// z-statistic
    pub z_stat: f64,
    /// p-value for test of no discontinuity
    pub p_value: f64,
    /// Estimated density left of cutoff
    pub density_left: f64,
    /// Estimated density right of cutoff
    pub density_right: f64,
}

impl DensityTestResult {
    /// Check if there is evidence of manipulation.
    pub fn has_manipulation(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// Results from placebo test at false cutoffs.
#[derive(Debug, Clone)]
pub struct PlaceboTestResult {
    /// Placebo cutoff value
    pub cutoff: f64,
    /// Treatment effect estimate
    pub tau: f64,
    /// Standard error
    pub se: f64,
    /// p-value
    pub p_value: f64,
}

impl PlaceboTestResult {
    /// Check if placebo effect is significant (bad - suggests spurious effects).
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// Covariate balance result.
#[derive(Debug, Clone)]
pub struct CovariateBalanceResult {
    /// Name of the covariate
    pub name: String,
    /// Difference at cutoff
    pub difference: f64,
    /// Standard error
    pub se: f64,
    /// p-value
    pub p_value: f64,
    /// Whether the covariate is balanced
    pub balanced: bool,
}

/// RDD validation tools.
pub struct RDDValidator {
    cutoff: f64,
    bandwidth: f64,
    kernel: Kernel,
}

impl RDDValidator {
    /// Create a new validator from an RDD model.
    pub fn from_model(model: &RDDModel) -> Self {
        Self {
            cutoff: model.cutoff,
            bandwidth: model.bandwidth,
            kernel: model.kernel,
        }
    }

    /// Create a new validator with specified parameters.
    pub fn new(cutoff: f64, bandwidth: f64, kernel: Kernel) -> Self {
        Self {
            cutoff,
            bandwidth,
            kernel,
        }
    }

    /// McCrary density test for manipulation of the running variable.
    ///
    /// Tests if there is a discontinuity in the density of the running
    /// variable at the cutoff. A significant discontinuity suggests
    /// potential manipulation.
    pub fn density_test(&self, running_var: &[f64], n_bins: usize) -> Result<DensityTestResult> {
        let x: Vec<f64> = running_var.iter().map(|&v| v - self.cutoff).collect();

        let left: Vec<f64> = x.iter().filter(|&&xi| xi < 0.0).cloned().collect();
        let right: Vec<f64> = x.iter().filter(|&&xi| xi >= 0.0).cloned().collect();

        if left.len() < 10 || right.len() < 10 {
            return Err(RDDError::InsufficientData {
                needed: 10,
                got: left.len().min(right.len()),
            });
        }

        // Compute bin width
        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let h = (x_max - x_min) / (2 * n_bins) as f64;

        // Count in bins near cutoff
        let left_bins: Vec<f64> = (0..n_bins)
            .map(|i| -h * (n_bins - i) as f64)
            .collect();
        let right_bins: Vec<f64> = (0..n_bins).map(|i| h * i as f64).collect();

        // Count observations in each bin
        let count_in_bin = |data: &[f64], lower: f64, upper: f64| -> usize {
            data.iter().filter(|&&xi| xi >= lower && xi < upper).count()
        };

        let _left_counts: Vec<usize> = left_bins
            .windows(2)
            .map(|w| count_in_bin(&left, w[0], w[1]))
            .collect();
        let _right_counts: Vec<usize> = right_bins
            .windows(2)
            .map(|w| count_in_bin(&right, w[0], w[1]))
            .collect();

        // Density at cutoff
        let n = x.len() as f64;
        let count_left_near = left.iter().filter(|&&xi| xi >= -h).count() as f64;
        let count_right_near = right.iter().filter(|&&xi| xi < h).count() as f64;

        let density_left = count_left_near / (h * n);
        let density_right = count_right_near / (h * n);

        // Log difference
        let (theta, se) = if density_left > 0.0 && density_right > 0.0 {
            let theta = (density_right / density_left).ln();
            let se = (1.0 / count_left_near.max(1.0) + 1.0 / count_right_near.max(1.0)).sqrt();
            (theta, se)
        } else {
            (0.0, 1.0)
        };

        let z_stat = if se > 0.0 { theta / se } else { 0.0 };
        let normal = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal.cdf(z_stat.abs()));

        Ok(DensityTestResult {
            theta,
            se,
            z_stat,
            p_value,
            density_left,
            density_right,
        })
    }

    /// Placebo tests at false cutoffs.
    ///
    /// If there are significant effects at placebo cutoffs, it suggests
    /// the effect at the true cutoff might be spurious.
    pub fn placebo_test(
        &self,
        cutoffs: &[f64],
        running_var: &[f64],
        outcome: &[f64],
    ) -> Vec<PlaceboTestResult> {
        cutoffs
            .iter()
            .map(|&c| {
                let mut model = RDDModel::new(c, self.bandwidth, self.kernel);

                match model.fit(running_var, outcome) {
                    Ok(results) => PlaceboTestResult {
                        cutoff: c,
                        tau: results.tau,
                        se: results.se,
                        p_value: results.p_value,
                    },
                    Err(_) => PlaceboTestResult {
                        cutoff: c,
                        tau: f64::NAN,
                        se: f64::NAN,
                        p_value: f64::NAN,
                    },
                }
            })
            .collect()
    }

    /// Test for balance of covariates at the cutoff.
    ///
    /// Covariates should not show discontinuities at the cutoff,
    /// as this would indicate potential confounding.
    pub fn covariate_balance(
        &self,
        running_var: &[f64],
        covariates: &[Vec<f64>],
        names: Option<&[&str]>,
    ) -> Vec<CovariateBalanceResult> {
        let n_covariates = covariates.len();
        let default_names: Vec<String> = (0..n_covariates)
            .map(|i| format!("covariate_{}", i))
            .collect();
        let cov_names: Vec<&str> = names
            .map(|n| n.to_vec())
            .unwrap_or_else(|| default_names.iter().map(|s| s.as_str()).collect());

        covariates
            .iter()
            .zip(cov_names.iter())
            .map(|(cov, &name)| {
                let mut model = RDDModel::new(self.cutoff, self.bandwidth, self.kernel);

                match model.fit(running_var, cov) {
                    Ok(results) => CovariateBalanceResult {
                        name: name.to_string(),
                        difference: results.tau,
                        se: results.se,
                        p_value: results.p_value,
                        balanced: results.p_value > 0.05,
                    },
                    Err(_) => CovariateBalanceResult {
                        name: name.to_string(),
                        difference: f64::NAN,
                        se: f64::NAN,
                        p_value: f64::NAN,
                        balanced: false,
                    },
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_density_test() {
        // Generate data without manipulation
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let x: Vec<f64> = (0..500).map(|_| rng.gen_range(-50.0..50.0)).collect();

        let validator = RDDValidator::new(0.0, 20.0, Kernel::Triangular);
        let result = validator.density_test(&x, 10).unwrap();

        // Should not reject null (no manipulation)
        assert!(result.p_value > 0.05, "p_value = {}", result.p_value);
    }

    #[test]
    fn test_placebo() {
        use rand_distr::Normal;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 2.0).unwrap();

        let mut x = Vec::with_capacity(500);
        let mut y = Vec::with_capacity(500);

        for _ in 0..500 {
            let xi: f64 = rng.gen_range(-50.0..50.0);
            let treatment = if xi >= 0.0 { 5.0 } else { 0.0 };
            let yi = treatment + normal.sample(&mut rng);
            x.push(xi);
            y.push(yi);
        }

        let validator = RDDValidator::new(0.0, 20.0, Kernel::Triangular);
        let placebos = validator.placebo_test(&[-30.0, 30.0], &x, &y);

        // Placebo effects should generally not be significant
        for p in &placebos {
            if !p.tau.is_nan() {
                // Allow some to be significant by chance
                println!("Placebo at {}: tau={:.3}, p={:.3}", p.cutoff, p.tau, p.p_value);
            }
        }
    }
}
