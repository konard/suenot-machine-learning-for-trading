//! Bayesian Linear Regression for pairs trading and rolling regression.
//!
//! Implements:
//! - Bayesian linear regression with conjugate priors
//! - Rolling Bayesian regression for time-varying parameters

use super::inference::{AdaptiveMetropolisHastings, MCMCConfig, MCMCSamples};
use nalgebra::{DMatrix, DVector};

/// Bayesian Linear Regression model
///
/// y = X * beta + epsilon, where epsilon ~ N(0, sigma^2)
///
/// Uses Normal-Inverse-Gamma conjugate prior:
/// - beta | sigma^2 ~ N(mu_0, sigma^2 * V_0)
/// - sigma^2 ~ InvGamma(a_0, b_0)
#[derive(Debug, Clone)]
pub struct BayesianLinearRegression {
    /// Prior mean for coefficients
    pub prior_mean: DVector<f64>,
    /// Prior precision matrix (inverse of covariance)
    pub prior_precision: DMatrix<f64>,
    /// Prior shape for variance (a_0)
    pub prior_a: f64,
    /// Prior rate for variance (b_0)
    pub prior_b: f64,
    /// Posterior mean for coefficients
    pub posterior_mean: Option<DVector<f64>>,
    /// Posterior precision matrix
    pub posterior_precision: Option<DMatrix<f64>>,
    /// Posterior shape for variance
    pub posterior_a: Option<f64>,
    /// Posterior rate for variance
    pub posterior_b: Option<f64>,
    /// Number of features
    n_features: usize,
}

impl BayesianLinearRegression {
    /// Create a new Bayesian Linear Regression model
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of features (including intercept if desired)
    /// * `prior_precision` - Prior precision (inverse variance) for coefficients
    pub fn new(n_features: usize, prior_precision: f64) -> Self {
        Self {
            prior_mean: DVector::zeros(n_features),
            prior_precision: DMatrix::identity(n_features, n_features) * prior_precision,
            prior_a: 0.001,
            prior_b: 0.001,
            posterior_mean: None,
            posterior_precision: None,
            posterior_a: None,
            posterior_b: None,
            n_features,
        }
    }

    /// Set prior mean for coefficients
    pub fn with_prior_mean(mut self, mean: DVector<f64>) -> Self {
        assert_eq!(mean.len(), self.n_features);
        self.prior_mean = mean;
        self
    }

    /// Fit the model using conjugate posterior update
    ///
    /// # Arguments
    ///
    /// * `x` - Design matrix (n_samples x n_features)
    /// * `y` - Target values (n_samples)
    pub fn fit(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) {
        let n = y.len() as f64;

        // Posterior precision: Lambda_n = Lambda_0 + X'X
        let xtx = x.transpose() * x;
        let posterior_precision = &self.prior_precision + &xtx;

        // Posterior mean: mu_n = Lambda_n^{-1} (Lambda_0 * mu_0 + X'y)
        let xty = x.transpose() * y;
        let prior_term = &self.prior_precision * &self.prior_mean;
        let posterior_mean = posterior_precision
            .clone()
            .try_inverse()
            .expect("Posterior precision matrix is not invertible")
            * (prior_term + xty);

        // Posterior parameters for sigma^2
        let posterior_a = self.prior_a + n / 2.0;

        let residuals = y - x * &posterior_mean;
        let sse = residuals.dot(&residuals);
        let diff = &posterior_mean - &self.prior_mean;
        let prior_contribution = diff.transpose() * &self.prior_precision * &diff;
        let posterior_b = self.prior_b + 0.5 * (sse + prior_contribution[(0, 0)]);

        self.posterior_mean = Some(posterior_mean);
        self.posterior_precision = Some(posterior_precision);
        self.posterior_a = Some(posterior_a);
        self.posterior_b = Some(posterior_b);
    }

    /// Get posterior mean of coefficients
    pub fn coefficients(&self) -> Option<&DVector<f64>> {
        self.posterior_mean.as_ref()
    }

    /// Get posterior variance estimate for noise
    pub fn noise_variance(&self) -> Option<f64> {
        match (self.posterior_a, self.posterior_b) {
            (Some(a), Some(b)) if a > 1.0 => Some(b / (a - 1.0)),
            _ => None,
        }
    }

    /// Get coefficient standard errors
    pub fn coefficient_std(&self) -> Option<DVector<f64>> {
        let precision = self.posterior_precision.as_ref()?;
        let sigma2 = self.noise_variance()?;

        let cov = precision.clone().try_inverse()?;
        let stds: Vec<f64> = (0..self.n_features)
            .map(|i| (sigma2 * cov[(i, i)]).sqrt())
            .collect();

        Some(DVector::from_vec(stds))
    }

    /// Predict for new data
    pub fn predict(&self, x: &DMatrix<f64>) -> Option<DVector<f64>> {
        let beta = self.posterior_mean.as_ref()?;
        Some(x * beta)
    }

    /// Get 95% credible intervals for coefficients
    pub fn credible_intervals(&self) -> Option<Vec<(f64, f64)>> {
        let mean = self.posterior_mean.as_ref()?;
        let std = self.coefficient_std()?;

        // Approximate using normal (valid for large samples)
        let z = 1.96;
        Some(
            (0..self.n_features)
                .map(|i| (mean[i] - z * std[i], mean[i] + z * std[i]))
                .collect(),
        )
    }

    /// Sample from the posterior using MCMC
    pub fn sample_posterior(&self, config: &MCMCConfig) -> Option<MCMCSamples> {
        let post_mean = self.posterior_mean.as_ref()?.clone();
        let post_prec = self.posterior_precision.as_ref()?.clone();
        let post_a = self.posterior_a?;
        let post_b = self.posterior_b?;
        let n_features = self.n_features;

        // Log posterior for MCMC
        let log_posterior = move |params: &[f64]| {
            // params[0..n_features] = beta, params[n_features] = log(sigma^2)
            let log_sigma2 = params[n_features];
            let sigma2 = log_sigma2.exp();

            // Beta prior: N(post_mean, sigma^2 * post_prec^{-1})
            let beta = DVector::from_column_slice(&params[..n_features]);
            let diff = &beta - &post_mean;
            let quad_form = (diff.transpose() * &post_prec * &diff)[(0, 0)];
            let log_prior_beta = -0.5 * quad_form / sigma2 - 0.5 * n_features as f64 * log_sigma2;

            // Sigma^2 prior: InvGamma(post_a, post_b)
            let log_prior_sigma = -(post_a + 1.0) * log_sigma2 - post_b / sigma2;

            // Jacobian for log transform
            let log_jacobian = log_sigma2;

            log_prior_beta + log_prior_sigma + log_jacobian
        };

        let mut initial: Vec<f64> = post_mean.iter().copied().collect();
        initial.push(self.noise_variance().unwrap_or(1.0).ln());

        let mut param_names: Vec<String> = (0..n_features)
            .map(|i| format!("beta_{}", i))
            .collect();
        param_names.push("log_sigma2".to_string());

        let sampler = AdaptiveMetropolisHastings::new(log_posterior, initial, param_names);
        Some(sampler.run(config))
    }
}

/// Rolling Bayesian Linear Regression
///
/// Performs Bayesian regression over rolling windows, tracking
/// how coefficients change over time.
#[derive(Debug)]
pub struct RollingBayesianRegression {
    /// Window size for rolling regression
    pub window_size: usize,
    /// Prior precision for coefficients
    pub prior_precision: f64,
    /// Results: (timestamp, coefficients, std_errors)
    pub results: Vec<RollingRegressionResult>,
}

/// Result for a single rolling window
#[derive(Debug, Clone)]
pub struct RollingRegressionResult {
    /// Timestamp (or index) for this window
    pub timestamp: i64,
    /// Coefficient estimates
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Noise variance estimate
    pub noise_variance: f64,
}

impl RollingBayesianRegression {
    /// Create a new rolling Bayesian regression
    pub fn new(window_size: usize, prior_precision: f64) -> Self {
        Self {
            window_size,
            prior_precision,
            results: Vec::new(),
        }
    }

    /// Fit rolling regression
    ///
    /// # Arguments
    ///
    /// * `x` - Independent variable(s)
    /// * `y` - Dependent variable
    /// * `timestamps` - Timestamps for each observation
    /// * `include_intercept` - Whether to add intercept column
    pub fn fit(
        &mut self,
        x: &[f64],
        y: &[f64],
        timestamps: &[i64],
        include_intercept: bool,
    ) {
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), timestamps.len());

        let n = x.len();
        if n < self.window_size {
            return;
        }

        self.results.clear();
        let n_features = if include_intercept { 2 } else { 1 };

        for i in self.window_size..=n {
            let start = i - self.window_size;
            let x_window = &x[start..i];
            let y_window = &y[start..i];

            // Build design matrix
            let design_matrix = if include_intercept {
                let mut data = Vec::with_capacity(self.window_size * 2);
                for &xi in x_window {
                    data.push(1.0); // Intercept
                    data.push(xi);  // Slope
                }
                DMatrix::from_row_slice(self.window_size, 2, &data)
            } else {
                DMatrix::from_column_slice(self.window_size, 1, x_window)
            };

            let y_vec = DVector::from_column_slice(y_window);

            // Fit Bayesian regression
            let mut model = BayesianLinearRegression::new(n_features, self.prior_precision);
            model.fit(&design_matrix, &y_vec);

            if let (Some(coefs), Some(stds), Some(var)) = (
                model.coefficients(),
                model.coefficient_std(),
                model.noise_variance(),
            ) {
                self.results.push(RollingRegressionResult {
                    timestamp: timestamps[i - 1],
                    coefficients: coefs.iter().copied().collect(),
                    std_errors: stds.iter().copied().collect(),
                    noise_variance: var,
                });
            }
        }
    }

    /// Get coefficient time series (for a specific coefficient index)
    pub fn coefficient_series(&self, coef_idx: usize) -> Vec<(i64, f64, f64)> {
        self.results
            .iter()
            .filter_map(|r| {
                if coef_idx < r.coefficients.len() {
                    Some((r.timestamp, r.coefficients[coef_idx], r.std_errors[coef_idx]))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get slope series (assuming intercept + slope model)
    pub fn slope_series(&self) -> Vec<(i64, f64, f64)> {
        self.coefficient_series(1)
    }

    /// Get intercept series
    pub fn intercept_series(&self) -> Vec<(i64, f64, f64)> {
        self.coefficient_series(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_linear_regression() {
        // Generate simple linear data: y = 2 + 3*x + noise
        let n = 100;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 + 3.0 * x + 0.1).collect();

        let x = DMatrix::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { x_data[i] });
        let y = DVector::from_column_slice(&y_data);

        let mut model = BayesianLinearRegression::new(2, 0.01);
        model.fit(&x, &y);

        let coefs = model.coefficients().unwrap();
        assert!((coefs[0] - 2.0).abs() < 0.5, "Intercept was {}", coefs[0]);
        assert!((coefs[1] - 3.0).abs() < 0.5, "Slope was {}", coefs[1]);
    }

    #[test]
    fn test_rolling_regression() {
        let n = 50;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 0.5 * xi).collect();
        let timestamps: Vec<i64> = (0..n as i64).collect();

        let mut rolling = RollingBayesianRegression::new(20, 0.01);
        rolling.fit(&x, &y, &timestamps, true);

        assert!(!rolling.results.is_empty());

        let slopes = rolling.slope_series();
        for (_, slope, _) in &slopes {
            assert!((slope - 0.5).abs() < 0.2, "Slope was {}", slope);
        }
    }
}
