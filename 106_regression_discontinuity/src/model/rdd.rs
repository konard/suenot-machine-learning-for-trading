//! Regression Discontinuity Design model implementation.
//!
//! Implements local linear regression with kernel weighting for both
//! sharp and fuzzy RDD designs.

use crate::{RDDError, Result};
use statrs::distribution::{ContinuousCDF, Normal};

/// Kernel functions for local linear regression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    /// K(u) = (1 - |u|) for |u| <= 1
    Triangular,
    /// K(u) = 0.5 for |u| <= 1
    Uniform,
    /// K(u) = 0.75(1 - u^2) for |u| <= 1
    Epanechnikov,
}

impl Kernel {
    /// Compute kernel weight for a given distance.
    pub fn weight(&self, u: f64) -> f64 {
        if u.abs() > 1.0 {
            return 0.0;
        }
        match self {
            Kernel::Triangular => (1.0 - u.abs()).max(0.0),
            Kernel::Uniform => 0.5,
            Kernel::Epanechnikov => 0.75 * (1.0 - u * u),
        }
    }

    /// Compute weights for a vector of distances.
    pub fn weights(&self, u: &[f64]) -> Vec<f64> {
        u.iter().map(|&x| self.weight(x)).collect()
    }
}

/// Results from RDD estimation.
#[derive(Debug, Clone)]
pub struct RDDResults {
    /// Treatment effect estimate
    pub tau: f64,
    /// Standard error
    pub se: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// t-statistic
    pub t_stat: f64,
    /// p-value for test that tau = 0
    pub p_value: f64,
    /// Bandwidth used
    pub bandwidth: f64,
    /// Number of observations left of cutoff
    pub n_left: usize,
    /// Number of observations right of cutoff
    pub n_right: usize,
    /// Intercept estimate left of cutoff
    pub alpha_left: f64,
    /// Intercept estimate right of cutoff
    pub alpha_right: f64,
    /// Slope estimate left of cutoff
    pub beta_left: f64,
    /// Slope estimate right of cutoff
    pub beta_right: f64,
}

impl RDDResults {
    /// Check if the treatment effect is statistically significant.
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// Regression Discontinuity Design model.
#[derive(Debug, Clone)]
pub struct RDDModel {
    /// Cutoff/threshold value
    pub cutoff: f64,
    /// Bandwidth for local regression
    pub bandwidth: f64,
    /// Kernel function
    pub kernel: Kernel,
    /// Significance level
    pub alpha: f64,
    /// Whether to use fuzzy RDD
    pub fuzzy: bool,
    /// Fitted results
    results: Option<RDDResults>,
}

impl RDDModel {
    /// Create a new RDD model.
    ///
    /// # Arguments
    /// * `cutoff` - The threshold value where treatment changes
    /// * `bandwidth` - Bandwidth for local regression
    /// * `kernel` - Kernel function for weighting
    pub fn new(cutoff: f64, bandwidth: f64, kernel: Kernel) -> Self {
        Self {
            cutoff,
            bandwidth,
            kernel,
            alpha: 0.05,
            fuzzy: false,
            results: None,
        }
    }

    /// Create a new RDD model with optimal bandwidth (estimated from data).
    pub fn with_optimal_bandwidth(cutoff: f64, kernel: Kernel) -> Self {
        Self {
            cutoff,
            bandwidth: 0.0, // Will be computed during fit
            kernel,
            alpha: 0.05,
            fuzzy: false,
            results: None,
        }
    }

    /// Set the significance level.
    pub fn set_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Enable fuzzy RDD.
    pub fn set_fuzzy(mut self, fuzzy: bool) -> Self {
        self.fuzzy = fuzzy;
        self
    }

    /// Fit the RDD model.
    ///
    /// # Arguments
    /// * `running_var` - The running variable (e.g., RSI values)
    /// * `outcome` - The outcome variable (e.g., returns)
    pub fn fit(&mut self, running_var: &[f64], outcome: &[f64]) -> Result<RDDResults> {
        if running_var.len() != outcome.len() {
            return Err(RDDError::InvalidParameter(
                "running_var and outcome must have same length".to_string(),
            ));
        }

        if running_var.len() < 20 {
            return Err(RDDError::InsufficientData {
                needed: 20,
                got: running_var.len(),
            });
        }

        // Center running variable at cutoff
        let x: Vec<f64> = running_var.iter().map(|&v| v - self.cutoff).collect();
        let y: Vec<f64> = outcome.to_vec();

        // Compute bandwidth if not set
        let h = if self.bandwidth <= 0.0 {
            self.compute_ik_bandwidth(&x, &y)
        } else {
            self.bandwidth
        };
        self.bandwidth = h;

        // Select observations within bandwidth
        let mut x_w = Vec::new();
        let mut y_w = Vec::new();
        let mut w_w = Vec::new();

        for (i, &xi) in x.iter().enumerate() {
            if xi.abs() <= h {
                let u = xi / h;
                let weight = self.kernel.weight(u);
                if weight > 0.0 {
                    x_w.push(xi);
                    y_w.push(y[i]);
                    w_w.push(weight);
                }
            }
        }

        // Separate left and right
        let left_indices: Vec<usize> = (0..x_w.len()).filter(|&i| x_w[i] < 0.0).collect();
        let right_indices: Vec<usize> = (0..x_w.len()).filter(|&i| x_w[i] >= 0.0).collect();

        let n_left = left_indices.len();
        let n_right = right_indices.len();

        if n_left < 5 || n_right < 5 {
            return Err(RDDError::InsufficientData {
                needed: 5,
                got: n_left.min(n_right),
            });
        }

        // Extract left and right data
        let x_left: Vec<f64> = left_indices.iter().map(|&i| x_w[i]).collect();
        let y_left: Vec<f64> = left_indices.iter().map(|&i| y_w[i]).collect();
        let w_left: Vec<f64> = left_indices.iter().map(|&i| w_w[i]).collect();

        let x_right: Vec<f64> = right_indices.iter().map(|&i| x_w[i]).collect();
        let y_right: Vec<f64> = right_indices.iter().map(|&i| y_w[i]).collect();
        let w_right: Vec<f64> = right_indices.iter().map(|&i| w_w[i]).collect();

        // Fit weighted OLS on each side
        let (alpha_left, beta_left, var_left) = self.weighted_ols(&x_left, &y_left, &w_left)?;
        let (alpha_right, beta_right, var_right) = self.weighted_ols(&x_right, &y_right, &w_right)?;

        // Treatment effect is difference in intercepts
        let tau = alpha_right - alpha_left;
        let se = (var_left + var_right).sqrt();

        // Confidence interval
        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = normal.inverse_cdf(1.0 - self.alpha / 2.0);
        let ci_lower = tau - z * se;
        let ci_upper = tau + z * se;

        // Test statistic and p-value
        let t_stat = if se > 0.0 { tau / se } else { 0.0 };
        let p_value = 2.0 * (1.0 - normal.cdf(t_stat.abs()));

        let results = RDDResults {
            tau,
            se,
            ci_lower,
            ci_upper,
            t_stat,
            p_value,
            bandwidth: h,
            n_left,
            n_right,
            alpha_left,
            alpha_right,
            beta_left,
            beta_right,
        };

        self.results = Some(results.clone());
        Ok(results)
    }

    /// Fit fuzzy RDD with treatment indicator.
    pub fn fit_fuzzy(
        &mut self,
        running_var: &[f64],
        outcome: &[f64],
        treatment: &[f64],
    ) -> Result<RDDResults> {
        if running_var.len() != outcome.len() || running_var.len() != treatment.len() {
            return Err(RDDError::InvalidParameter(
                "All inputs must have same length".to_string(),
            ));
        }

        // Center running variable
        let x: Vec<f64> = running_var.iter().map(|&v| v - self.cutoff).collect();
        let y: Vec<f64> = outcome.to_vec();
        let d: Vec<f64> = treatment.to_vec();

        let h = if self.bandwidth <= 0.0 {
            self.compute_ik_bandwidth(&x, &y)
        } else {
            self.bandwidth
        };
        self.bandwidth = h;

        // Select observations within bandwidth
        let mut x_w = Vec::new();
        let mut y_w = Vec::new();
        let mut d_w = Vec::new();
        let mut w_w = Vec::new();

        for (i, &xi) in x.iter().enumerate() {
            if xi.abs() <= h {
                let u = xi / h;
                let weight = self.kernel.weight(u);
                if weight > 0.0 {
                    x_w.push(xi);
                    y_w.push(y[i]);
                    d_w.push(d[i]);
                    w_w.push(weight);
                }
            }
        }

        // Separate left and right
        let left_indices: Vec<usize> = (0..x_w.len()).filter(|&i| x_w[i] < 0.0).collect();
        let right_indices: Vec<usize> = (0..x_w.len()).filter(|&i| x_w[i] >= 0.0).collect();

        let n_left = left_indices.len();
        let n_right = right_indices.len();

        if n_left < 5 || n_right < 5 {
            return Err(RDDError::InsufficientData {
                needed: 5,
                got: n_left.min(n_right),
            });
        }

        // First stage: treatment on running variable
        let d_left: Vec<f64> = left_indices.iter().map(|&i| d_w[i]).collect();
        let d_right: Vec<f64> = right_indices.iter().map(|&i| d_w[i]).collect();
        let x_left: Vec<f64> = left_indices.iter().map(|&i| x_w[i]).collect();
        let x_right: Vec<f64> = right_indices.iter().map(|&i| x_w[i]).collect();
        let w_left: Vec<f64> = left_indices.iter().map(|&i| w_w[i]).collect();
        let w_right: Vec<f64> = right_indices.iter().map(|&i| w_w[i]).collect();

        let (alpha_d_left, _, _) = self.weighted_ols(&x_left, &d_left, &w_left)?;
        let (alpha_d_right, _, _) = self.weighted_ols(&x_right, &d_right, &w_right)?;

        let first_stage = alpha_d_right - alpha_d_left;

        if first_stage.abs() < 0.01 {
            return Err(RDDError::ComputationError(format!(
                "First stage too weak: {:.4}",
                first_stage
            )));
        }

        // Reduced form: outcome on running variable
        let y_left: Vec<f64> = left_indices.iter().map(|&i| y_w[i]).collect();
        let y_right: Vec<f64> = right_indices.iter().map(|&i| y_w[i]).collect();

        let (alpha_left, beta_left, var_left) = self.weighted_ols(&x_left, &y_left, &w_left)?;
        let (alpha_right, beta_right, var_right) = self.weighted_ols(&x_right, &y_right, &w_right)?;

        let reduced_form = alpha_right - alpha_left;

        // IV estimate
        let tau = reduced_form / first_stage;
        let se_rf = (var_left + var_right).sqrt();
        let se = se_rf / first_stage.abs();

        let normal = Normal::new(0.0, 1.0).unwrap();
        let z = normal.inverse_cdf(1.0 - self.alpha / 2.0);
        let ci_lower = tau - z * se;
        let ci_upper = tau + z * se;

        let t_stat = if se > 0.0 { tau / se } else { 0.0 };
        let p_value = 2.0 * (1.0 - normal.cdf(t_stat.abs()));

        let results = RDDResults {
            tau,
            se,
            ci_lower,
            ci_upper,
            t_stat,
            p_value,
            bandwidth: h,
            n_left,
            n_right,
            alpha_left,
            alpha_right,
            beta_left,
            beta_right,
        };

        self.results = Some(results.clone());
        Ok(results)
    }

    /// Predict outcome for new running variable values.
    pub fn predict(&self, running_var: &[f64]) -> Result<Vec<f64>> {
        let results = self.results.as_ref().ok_or(RDDError::NotFitted)?;

        let predictions: Vec<f64> = running_var
            .iter()
            .map(|&v| {
                let x = v - self.cutoff;
                if x < 0.0 {
                    results.alpha_left + results.beta_left * x
                } else {
                    results.alpha_right + results.beta_right * x
                }
            })
            .collect();

        Ok(predictions)
    }

    /// Get fitted results.
    pub fn results(&self) -> Option<&RDDResults> {
        self.results.as_ref()
    }

    /// Weighted OLS estimation.
    fn weighted_ols(&self, x: &[f64], y: &[f64], w: &[f64]) -> Result<(f64, f64, f64)> {
        let n = x.len();
        if n < 2 {
            let mean_y = if n > 0 { y[0] } else { 0.0 };
            return Ok((mean_y, 0.0, 0.0));
        }

        // Weighted design matrix computations
        let sum_w: f64 = w.iter().sum();
        let sum_wx: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
        let sum_wx2: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi * xi).sum();
        let sum_wy: f64 = w.iter().zip(y.iter()).map(|(wi, yi)| wi * yi).sum();
        let sum_wxy: f64 = w
            .iter()
            .zip(x.iter())
            .zip(y.iter())
            .map(|((wi, xi), yi)| wi * xi * yi)
            .sum();

        // Solve normal equations with regularization
        let det = sum_w * sum_wx2 - sum_wx * sum_wx;
        let reg = 1e-10;
        let det_reg = det + reg;

        let beta = (sum_w * sum_wxy - sum_wx * sum_wy) / det_reg;
        let alpha = (sum_wy - beta * sum_wx) / sum_w;

        // Compute residual variance
        let mut ss_resid = 0.0;
        for i in 0..n {
            let pred = alpha + beta * x[i];
            let resid = y[i] - pred;
            ss_resid += w[i] * resid * resid;
        }
        let sigma2 = ss_resid / (sum_w - 2.0).max(1.0);

        // Variance of alpha estimate
        let var_alpha = sigma2 * sum_wx2 / det_reg.abs().max(1e-10);

        Ok((alpha, beta, var_alpha.max(0.0)))
    }

    /// Compute Imbens-Kalyanaraman optimal bandwidth.
    fn compute_ik_bandwidth(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();

        // Pilot bandwidth using Silverman's rule
        let std_x = std_dev(x);
        let h_pilot = 2.84 * std_x * (n as f64).powf(-0.2);

        // Estimate variance on each side
        let left: Vec<(f64, f64)> = x
            .iter()
            .zip(y.iter())
            .filter(|(&xi, _)| xi < 0.0)
            .map(|(&xi, &yi)| (xi, yi))
            .collect();
        let right: Vec<(f64, f64)> = x
            .iter()
            .zip(y.iter())
            .filter(|(&xi, _)| xi >= 0.0)
            .map(|(&xi, &yi)| (xi, yi))
            .collect();

        let y_left: Vec<f64> = left.iter().map(|(_, yi)| *yi).collect();
        let y_right: Vec<f64> = right.iter().map(|(_, yi)| *yi).collect();

        let sigma2_left = if y_left.len() > 1 {
            variance(&y_left)
        } else {
            variance(y)
        };
        let sigma2_right = if y_right.len() > 1 {
            variance(&y_right)
        } else {
            variance(y)
        };
        let sigma2 = (sigma2_left + sigma2_right) / 2.0;

        // Estimate curvature using polynomial fit
        let m2 = self.estimate_curvature(x, y, &left, &right);

        // IK optimal bandwidth
        let n_eff = n as f64;
        let h_opt = 2.84 * (sigma2 / (n_eff * (m2.abs() + 0.001).powi(2))).powf(0.2);

        // Bound bandwidth
        let data_range = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - x.iter().cloned().fold(f64::INFINITY, f64::min);
        let h_bounded = h_opt.min(data_range / 2.0).max(data_range / 20.0);

        h_bounded.max(h_pilot * 0.5)
    }

    /// Estimate curvature for bandwidth selection.
    fn estimate_curvature(
        &self,
        _x: &[f64],
        _y: &[f64],
        left: &[(f64, f64)],
        right: &[(f64, f64)],
    ) -> f64 {
        // Simple polynomial fit for curvature estimation
        let estimate_m2 = |data: &[(f64, f64)]| -> f64 {
            if data.len() < 3 {
                return 0.1;
            }

            let x_vals: Vec<f64> = data.iter().map(|(xi, _)| *xi).collect();
            let y_vals: Vec<f64> = data.iter().map(|(_, yi)| *yi).collect();

            // Quadratic fit: y = a + b*x + c*x^2
            // Second derivative = 2c
            let n = x_vals.len() as f64;
            let sum_x: f64 = x_vals.iter().sum();
            let sum_x2: f64 = x_vals.iter().map(|x| x * x).sum();
            let sum_x3: f64 = x_vals.iter().map(|x| x.powi(3)).sum();
            let sum_x4: f64 = x_vals.iter().map(|x| x.powi(4)).sum();
            let sum_y: f64 = y_vals.iter().sum();
            let sum_xy: f64 = x_vals.iter().zip(y_vals.iter()).map(|(x, y)| x * y).sum();
            let sum_x2y: f64 = x_vals.iter().zip(y_vals.iter()).map(|(x, y)| x * x * y).sum();

            // Solve 3x3 system (simplified, using determinant method)
            let det = n * (sum_x2 * sum_x4 - sum_x3 * sum_x3)
                - sum_x * (sum_x * sum_x4 - sum_x2 * sum_x3)
                + sum_x2 * (sum_x * sum_x3 - sum_x2 * sum_x2);

            if det.abs() < 1e-10 {
                return 0.1;
            }

            // Coefficient c (quadratic term)
            let c = (n * (sum_x2 * sum_x2y - sum_xy * sum_x3)
                - sum_x * (sum_x * sum_x2y - sum_y * sum_x3)
                + sum_x2 * (sum_x * sum_xy - sum_y * sum_x2))
                / det;

            2.0 * c // Second derivative
        };

        let m2_left = estimate_m2(left);
        let m2_right = estimate_m2(right);

        (m2_right - m2_left).abs().max(0.001)
    }
}

/// Compute standard deviation.
fn std_dev(x: &[f64]) -> f64 {
    variance(x).sqrt()
}

/// Compute variance.
fn variance(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n <= 1.0 {
        return 0.0;
    }
    let mean: f64 = x.iter().sum::<f64>() / n;
    let ss: f64 = x.iter().map(|&xi| (xi - mean).powi(2)).sum();
    ss / (n - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_weights() {
        let k = Kernel::Triangular;
        assert!((k.weight(0.0) - 1.0).abs() < 1e-10);
        assert!((k.weight(0.5) - 0.5).abs() < 1e-10);
        assert!((k.weight(1.0) - 0.0).abs() < 1e-10);
        assert_eq!(k.weight(1.5), 0.0);
    }

    #[test]
    fn test_rdd_fit() {
        // Generate synthetic data with treatment effect at cutoff
        use rand::prelude::*;
        use rand_distr::Normal;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let n = 500;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            let xi: f64 = rng.gen_range(-50.0..50.0);
            let treatment = if xi >= 0.0 { 5.0 } else { 0.0 };
            let yi = 2.0 + 0.1 * xi + treatment + normal.sample(&mut rng) * 2.0;
            x.push(xi);
            y.push(yi);
        }

        let mut model = RDDModel::new(0.0, 20.0, Kernel::Triangular);
        let results = model.fit(&x, &y).unwrap();

        // Treatment effect should be close to 5.0
        assert!((results.tau - 5.0).abs() < 1.5, "tau = {}", results.tau);
        assert!(results.p_value < 0.05);
    }
}
