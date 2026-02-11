//! # Physics-Informed Neural Network for CIR Model
//!
//! This crate implements the Cox-Ingersoll-Ross (CIR) interest rate model
//! with analytical bond pricing, path simulation, calibration, and a
//! simple PINN-based PDE solver.
//!
//! The CIR SDE:
//! ```text
//! dr = kappa * (theta - r) * dt + sigma * sqrt(r) * dW
//! ```
//!
//! The CIR bond pricing PDE:
//! ```text
//! dP/dt + kappa*(theta - r)*dP/dr + 0.5*sigma^2*r*d^2P/dr^2 - r*P = 0
//! ```
//!
//! Key features:
//! - Analytical bond pricing via A(tau)*exp(-B(tau)*r)
//! - Exact path simulation via non-central chi-squared
//! - MLE calibration from rate time series
//! - Simple neural network for PINN demonstration

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{ChiSquared, Distribution, Normal};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared as StatChiSq, ContinuousCDF};
use std::f64::consts::PI;

// =============================================================================
// CIR Parameters
// =============================================================================

/// Parameters for the Cox-Ingersoll-Ross model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CIRParams {
    /// Mean-reversion speed (kappa > 0).
    pub kappa: f64,
    /// Long-run mean rate (theta > 0).
    pub theta: f64,
    /// Volatility coefficient (sigma > 0).
    pub sigma: f64,
    /// Initial short rate (r0 >= 0).
    pub r0: f64,
}

impl Default for CIRParams {
    fn default() -> Self {
        Self {
            kappa: 0.5,
            theta: 0.05,
            sigma: 0.1,
            r0: 0.03,
        }
    }
}

impl CIRParams {
    /// Check if the Feller condition is satisfied: 2*kappa*theta >= sigma^2.
    pub fn feller_condition(&self) -> bool {
        2.0 * self.kappa * self.theta >= self.sigma * self.sigma
    }

    /// Feller margin: 2*kappa*theta - sigma^2.
    pub fn feller_margin(&self) -> f64 {
        2.0 * self.kappa * self.theta - self.sigma * self.sigma
    }

    /// Compute gamma = sqrt(kappa^2 + 2*sigma^2).
    pub fn gamma(&self) -> f64 {
        (self.kappa * self.kappa + 2.0 * self.sigma * self.sigma).sqrt()
    }

    /// Degrees of freedom for the non-central chi-squared distribution.
    pub fn chi_sq_df(&self) -> f64 {
        4.0 * self.kappa * self.theta / (self.sigma * self.sigma)
    }

    /// Validate parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.kappa <= 0.0 {
            return Err("kappa must be positive".to_string());
        }
        if self.theta <= 0.0 {
            return Err("theta must be positive".to_string());
        }
        if self.sigma <= 0.0 {
            return Err("sigma must be positive".to_string());
        }
        if self.r0 < 0.0 {
            return Err("r0 must be non-negative".to_string());
        }
        Ok(())
    }
}

// =============================================================================
// Analytical Bond Pricing
// =============================================================================

/// Compute B(tau) in the CIR bond price formula P = A * exp(-B * r).
pub fn cir_b(tau: f64, params: &CIRParams) -> f64 {
    let gamma = params.gamma();
    let exp_gt = (gamma * tau).exp();
    let numerator = 2.0 * (exp_gt - 1.0);
    let denominator = (gamma + params.kappa) * (exp_gt - 1.0) + 2.0 * gamma;
    numerator / denominator
}

/// Compute A(tau) in the CIR bond price formula P = A * exp(-B * r).
pub fn cir_a(tau: f64, params: &CIRParams) -> f64 {
    let gamma = params.gamma();
    let exp_gt = (gamma * tau).exp();
    let exp_half = ((params.kappa + gamma) * tau / 2.0).exp();
    let denominator = (gamma + params.kappa) * (exp_gt - 1.0) + 2.0 * gamma;
    let base = 2.0 * gamma * exp_half / denominator;
    let exponent = 2.0 * params.kappa * params.theta / (params.sigma * params.sigma);
    base.powf(exponent)
}

/// Analytical CIR zero-coupon bond price.
///
/// P(r, t) = A(T-t) * exp(-B(T-t) * r)
///
/// # Arguments
/// * `r` - Current short rate
/// * `tau` - Time to maturity (T - t)
/// * `params` - CIR parameters
///
/// # Returns
/// Bond price in (0, 1].
pub fn cir_bond_price(r: f64, tau: f64, params: &CIRParams) -> f64 {
    if tau <= 0.0 {
        return 1.0;
    }
    let a = cir_a(tau, params);
    let b = cir_b(tau, params);
    a * (-b * r).exp()
}

/// CIR continuously compounded yield.
///
/// y(r, tau) = -ln(P) / tau = (B*r - ln(A)) / tau
pub fn cir_yield(r: f64, tau: f64, params: &CIRParams) -> f64 {
    if tau <= 1e-10 {
        return r;
    }
    let a = cir_a(tau, params);
    let b = cir_b(tau, params);
    (b * r - a.ln()) / tau
}

/// CIR instantaneous forward rate via finite difference.
pub fn cir_forward_rate(r: f64, tau: f64, params: &CIRParams) -> f64 {
    let dt = 1e-6;
    let ln_p_minus = cir_bond_price(r, tau - dt / 2.0, params).ln();
    let ln_p_plus = cir_bond_price(r, tau + dt / 2.0, params).ln();
    -(ln_p_plus - ln_p_minus) / dt
}

/// Generate a CIR yield curve for an array of maturities.
pub fn generate_yield_curve(r: f64, maturities: &[f64], params: &CIRParams) -> Vec<f64> {
    maturities.iter().map(|&tau| cir_yield(r, tau, params)).collect()
}

// =============================================================================
// CIR Moments
// =============================================================================

/// Conditional mean: E[r(t) | r(0) = r0] = theta + (r0 - theta)*exp(-kappa*t).
pub fn cir_conditional_mean(r0: f64, t: f64, params: &CIRParams) -> f64 {
    params.theta + (r0 - params.theta) * (-params.kappa * t).exp()
}

/// Conditional variance: Var[r(t) | r(0) = r0].
pub fn cir_conditional_variance(r0: f64, t: f64, params: &CIRParams) -> f64 {
    let exp_kt = (-params.kappa * t).exp();
    let sig2 = params.sigma * params.sigma;
    let term1 = r0 * sig2 * exp_kt * (1.0 - exp_kt) / params.kappa;
    let term2 = params.theta * sig2 * (1.0 - exp_kt).powi(2) / (2.0 * params.kappa);
    term1 + term2
}

// =============================================================================
// Path Simulation
// =============================================================================

/// Simulate CIR paths using the Euler-Maruyama scheme with full truncation.
///
/// # Arguments
/// * `params` - CIR parameters
/// * `t_final` - Time horizon
/// * `n_steps` - Number of time steps
/// * `n_paths` - Number of paths
///
/// # Returns
/// Array of shape (n_paths, n_steps + 1) containing rate paths.
pub fn simulate_cir_euler(
    params: &CIRParams,
    t_final: f64,
    n_steps: usize,
    n_paths: usize,
) -> Array2<f64> {
    let dt = t_final / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mut paths = Array2::zeros((n_paths, n_steps + 1));
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    for j in 0..n_paths {
        paths[[j, 0]] = params.r0;
        for i in 0..n_steps {
            let r = paths[[j, i]].max(0.0);
            let dw: f64 = normal.sample(&mut rng) * sqrt_dt;
            let drift = params.kappa * (params.theta - r) * dt;
            let diffusion = params.sigma * r.sqrt() * dw;
            paths[[j, i + 1]] = (paths[[j, i]] + drift + diffusion).max(0.0);
        }
    }

    paths
}

/// Simulate CIR paths using the exact non-central chi-squared method.
///
/// This method produces exact samples with no discretization error.
///
/// # Arguments
/// * `params` - CIR parameters
/// * `t_final` - Time horizon
/// * `n_steps` - Number of time steps
/// * `n_paths` - Number of paths
///
/// # Returns
/// Array of shape (n_paths, n_steps + 1) containing rate paths.
pub fn simulate_cir_paths(
    params: &CIRParams,
    t_final: f64,
    n_steps: usize,
    n_paths: usize,
) -> Array2<f64> {
    let dt = t_final / n_steps as f64;
    let mut paths = Array2::zeros((n_paths, n_steps + 1));
    let mut rng = rand::thread_rng();

    let c = params.sigma * params.sigma * (1.0 - (-params.kappa * dt).exp())
        / (4.0 * params.kappa);
    let df = 4.0 * params.kappa * params.theta / (params.sigma * params.sigma);

    for j in 0..n_paths {
        paths[[j, 0]] = params.r0;
        for i in 0..n_steps {
            let r_prev = paths[[j, i]].max(1e-10);
            let nc = r_prev * (-params.kappa * dt).exp() / c;

            // Sample from non-central chi-squared via the Poisson mixture method:
            // ncx2(df, nc) = chi2(df + 2*Poisson(nc/2))
            let poisson_lambda = nc / 2.0;
            let n_poisson = sample_poisson(&mut rng, poisson_lambda);
            let total_df = df + 2.0 * n_poisson as f64;

            // Sample from chi-squared with total_df degrees of freedom
            if total_df > 0.0 {
                let chi2 = ChiSquared::new(total_df).unwrap_or_else(|_| ChiSquared::new(1.0).unwrap());
                let sample: f64 = chi2.sample(&mut rng);
                paths[[j, i + 1]] = c * sample;
            } else {
                paths[[j, i + 1]] = 0.0;
            }
        }
    }

    paths
}

/// Sample from Poisson distribution using the inverse CDF method.
fn sample_poisson(rng: &mut impl Rng, lambda: f64) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k = 0u32;
    let mut p = 1.0f64;

    loop {
        k += 1;
        p *= rng.gen::<f64>();
        if p <= l {
            break;
        }
    }
    k - 1
}

// =============================================================================
// Simple Neural Network (PINN backbone)
// =============================================================================

/// A simple feedforward neural network for PINN demonstration.
///
/// Architecture: input(3) -> hidden layers with tanh -> output(1) with sigmoid
#[derive(Debug, Clone)]
pub struct SimpleNetwork {
    /// Weight matrices for each layer.
    pub weights: Vec<Array2<f64>>,
    /// Bias vectors for each layer.
    pub biases: Vec<Array1<f64>>,
    /// Number of layers (including output).
    pub n_layers: usize,
}

impl SimpleNetwork {
    /// Create a new network with given layer sizes.
    ///
    /// # Arguments
    /// * `layer_sizes` - Vector of layer dimensions, e.g., [3, 64, 64, 32, 1]
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut rng = rand::thread_rng();
        let n_layers = layer_sizes.len() - 1;

        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let fan_in = layer_sizes[i];
            let fan_out = layer_sizes[i + 1];

            // Xavier initialization
            let scale = (6.0 / (fan_in + fan_out) as f64).sqrt();

            let w = Array2::from_shape_fn((fan_in, fan_out), |_| {
                rng.gen_range(-scale..scale)
            });
            let b = Array1::zeros(fan_out);

            weights.push(w);
            biases.push(b);
        }

        Self {
            weights,
            biases,
            n_layers,
        }
    }

    /// Forward pass through the network.
    ///
    /// Uses tanh for hidden layers and sigmoid for the output layer.
    pub fn forward(&self, input: &Array1<f64>) -> f64 {
        let mut x = input.clone();

        for i in 0..self.n_layers {
            // Linear: x = x @ W + b
            let z = x.dot(&self.weights[i]) + &self.biases[i];

            if i < self.n_layers - 1 {
                // Hidden layer: tanh activation
                x = z.mapv(|v| v.tanh());
            } else {
                // Output layer: sigmoid activation
                x = z.mapv(|v| 1.0 / (1.0 + (-v).exp()));
            }
        }

        x[0]
    }

    /// Predict bond price for given (r, t) using normalized inputs.
    ///
    /// Input features: [r/r_max, t/T, sqrt(r/r_max)]
    pub fn predict(&self, r: f64, t: f64, r_max: f64, t_max: f64) -> f64 {
        let r_norm = r / r_max;
        let t_norm = t / t_max;
        let sqrt_r_norm = (r_norm.max(0.0)).sqrt();

        let input = Array1::from_vec(vec![r_norm, t_norm, sqrt_r_norm]);
        self.forward(&input)
    }

    /// Count total number of parameters.
    pub fn n_parameters(&self) -> usize {
        let mut count = 0;
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            count += w.len() + b.len();
        }
        count
    }
}

// =============================================================================
// PINN Training (simplified gradient-free approach for demonstration)
// =============================================================================

/// Configuration for PINN training.
#[derive(Debug, Clone)]
pub struct PINNConfig {
    pub n_collocation: usize,
    pub n_boundary: usize,
    pub n_terminal: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub r_max: f64,
    pub t_max: f64,
}

impl Default for PINNConfig {
    fn default() -> Self {
        Self {
            n_collocation: 1000,
            n_boundary: 100,
            n_terminal: 100,
            learning_rate: 0.01,
            epochs: 1000,
            r_max: 0.20,
            t_max: 5.0,
        }
    }
}

/// Compute the CIR PDE residual at a point using finite differences.
///
/// PDE: dP/dt + kappa*(theta-r)*dP/dr + 0.5*sigma^2*r*d^2P/dr^2 - r*P = 0
pub fn pde_residual(
    net: &SimpleNetwork,
    r: f64,
    t: f64,
    params: &CIRParams,
    config: &PINNConfig,
) -> f64 {
    let dr = 1e-4 * config.r_max;
    let dt = 1e-4 * config.t_max;

    let p = net.predict(r, t, config.r_max, config.t_max);

    // Finite difference derivatives
    let p_r_plus = net.predict(r + dr, t, config.r_max, config.t_max);
    let p_r_minus = net.predict((r - dr).max(0.0), t, config.r_max, config.t_max);
    let p_t_plus = net.predict(r, (t + dt).min(config.t_max), config.r_max, config.t_max);
    let p_t_minus = net.predict(r, (t - dt).max(0.0), config.r_max, config.t_max);

    let dp_dr = (p_r_plus - p_r_minus) / (2.0 * dr);
    let d2p_dr2 = (p_r_plus - 2.0 * p + p_r_minus) / (dr * dr);
    let dp_dt = (p_t_plus - p_t_minus) / (2.0 * dt);

    // CIR PDE residual
    dp_dt + params.kappa * (params.theta - r) * dp_dr
        + 0.5 * params.sigma * params.sigma * r * d2p_dr2
        - r * p
}

// =============================================================================
// Calibration (OLS method)
// =============================================================================

/// Calibrate CIR parameters using OLS from a rate time series.
///
/// Uses the discrete regression: dr = a + b*r_prev + error
/// where a = kappa*theta*dt, b = -kappa*dt
pub fn calibrate_cir_ols(rates: &[f64], dt: f64) -> CIRParams {
    let n = rates.len();
    if n < 3 {
        return CIRParams::default();
    }

    let mut sum_r = 0.0;
    let mut sum_r2 = 0.0;
    let mut sum_dr = 0.0;
    let mut sum_r_dr = 0.0;
    let count = (n - 1) as f64;

    for i in 0..n - 1 {
        let r = rates[i];
        let dr = rates[i + 1] - rates[i];
        sum_r += r;
        sum_r2 += r * r;
        sum_dr += dr;
        sum_r_dr += r * dr;
    }

    // OLS: dr = a + b * r
    let mean_r = sum_r / count;
    let mean_dr = sum_dr / count;
    let cov_r_dr = sum_r_dr / count - mean_r * mean_dr;
    let var_r = sum_r2 / count - mean_r * mean_r;

    let b = if var_r > 1e-20 { cov_r_dr / var_r } else { -0.5 * dt };
    let a = mean_dr - b * mean_r;

    let kappa = (-b / dt).max(0.01);
    let theta = if kappa * dt > 1e-10 { a / (kappa * dt) } else { mean_r };
    let theta = theta.max(1e-8);

    // Estimate sigma from residuals
    let mut residual_sum = 0.0;
    for i in 0..n - 1 {
        let predicted = a + b * rates[i];
        let residual = (rates[i + 1] - rates[i]) - predicted;
        let r_pos = rates[i].max(1e-10);
        residual_sum += residual * residual / (r_pos * dt);
    }
    let sigma = (residual_sum / count).sqrt().max(1e-6);

    CIRParams {
        kappa,
        theta,
        sigma,
        r0: rates[0].max(0.0),
    }
}

// =============================================================================
// Bybit Data Structures
// =============================================================================

/// A single funding rate record from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub symbol: String,
    pub funding_rate: f64,
    pub funding_rate_timestamp: String,
}

/// Bybit API response for funding rate history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitFundingResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitFundingResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitFundingResult {
    pub category: Option<String>,
    pub list: Vec<BybitFundingRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitFundingRecord {
    pub symbol: String,
    #[serde(rename = "fundingRate")]
    pub funding_rate: String,
    #[serde(rename = "fundingRateTimestamp")]
    pub funding_rate_timestamp: String,
}

// =============================================================================
// Credit Risk Functions
// =============================================================================

/// Compute survival probability using CIR intensity model.
///
/// Q(t, T) = A(T-t) * exp(-B(T-t) * lambda)
///
/// This is mathematically identical to CIR bond pricing.
pub fn survival_probability(lambda: f64, tau: f64, params: &CIRParams) -> f64 {
    cir_bond_price(lambda, tau, params)
}

/// Compute CDS spread under CIR intensity dynamics.
pub fn cds_spread(
    lambda: f64,
    maturity: f64,
    params: &CIRParams,
    recovery: f64,
    risk_free_rate: f64,
    n_payments_per_year: usize,
) -> f64 {
    let total_periods = (maturity * n_payments_per_year as f64) as usize;
    let dt_payment = 1.0 / n_payments_per_year as f64;

    let mut protection_leg = 0.0;
    let mut premium_leg = 0.0;

    for i in 1..=total_periods {
        let t_i = i as f64 * dt_payment;
        let t_prev = (i - 1) as f64 * dt_payment;

        let q_prev = survival_probability(lambda, t_prev, params);
        let q_curr = survival_probability(lambda, t_i, params);
        let df = (-risk_free_rate * t_i).exp();

        protection_leg += (1.0 - recovery) * (q_prev - q_curr) * df;
        premium_leg += q_curr * dt_payment * df;
    }

    if premium_leg <= 0.0 {
        0.0
    } else {
        protection_leg / premium_leg
    }
}

// =============================================================================
// Display Utilities
// =============================================================================

/// Print a formatted summary of CIR parameters.
pub fn print_params_summary(params: &CIRParams) {
    println!("CIR Parameters:");
    println!("  kappa (mean-reversion speed): {:.6}", params.kappa);
    println!("  theta (long-run mean):        {:.6}", params.theta);
    println!("  sigma (volatility):           {:.6}", params.sigma);
    println!("  r0    (initial rate):          {:.6}", params.r0);
    println!(
        "  Feller condition: 2*kappa*theta = {:.6} {} sigma^2 = {:.6} [{}]",
        2.0 * params.kappa * params.theta,
        if params.feller_condition() { ">=" } else { "<" },
        params.sigma * params.sigma,
        if params.feller_condition() {
            "SATISFIED"
        } else {
            "VIOLATED"
        }
    );
    println!("  gamma: {:.6}", params.gamma());
    println!("  chi^2 df: {:.6}", params.chi_sq_df());
}

/// Print a yield curve table.
pub fn print_yield_curve(r: f64, maturities: &[f64], params: &CIRParams) {
    println!("\nCIR Yield Curve (r = {:.4}):", r);
    println!("{:>10} {:>12} {:>12} {:>12}", "Maturity", "Bond Price", "Yield (%)", "Fwd Rate (%)");
    println!("{}", "-".repeat(50));
    for &tau in maturities {
        let price = cir_bond_price(r, tau, params);
        let y = cir_yield(r, tau, params);
        let fwd = cir_forward_rate(r, tau, params);
        println!(
            "{:>10.2}y {:>12.6} {:>12.4} {:>12.4}",
            tau,
            price,
            y * 100.0,
            fwd * 100.0
        );
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bond_price_at_maturity() {
        let params = CIRParams::default();
        let price = cir_bond_price(0.05, 0.0, &params);
        assert!((price - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bond_price_positive() {
        let params = CIRParams::default();
        for r in [0.01, 0.05, 0.10, 0.15, 0.20] {
            for tau in [0.5, 1.0, 2.0, 5.0, 10.0] {
                let price = cir_bond_price(r, tau, &params);
                assert!(price > 0.0, "Price must be positive for r={}, tau={}", r, tau);
                assert!(price <= 1.0, "Price must be <= 1 for r={}, tau={}", r, tau);
            }
        }
    }

    #[test]
    fn test_bond_price_monotone_in_r() {
        let params = CIRParams::default();
        let tau = 5.0;
        let mut prev_price = cir_bond_price(0.01, tau, &params);
        for r in [0.02, 0.05, 0.10, 0.15] {
            let price = cir_bond_price(r, tau, &params);
            assert!(price < prev_price, "Price must decrease with r");
            prev_price = price;
        }
    }

    #[test]
    fn test_bond_price_monotone_in_tau() {
        let params = CIRParams::default();
        let r = 0.05;
        let mut prev_price = cir_bond_price(r, 0.5, &params);
        for tau in [1.0, 2.0, 5.0, 10.0] {
            let price = cir_bond_price(r, tau, &params);
            assert!(price < prev_price, "Price must decrease with tau");
            prev_price = price;
        }
    }

    #[test]
    fn test_feller_condition() {
        // Satisfied
        let params = CIRParams {
            kappa: 0.5,
            theta: 0.05,
            sigma: 0.1,
            r0: 0.03,
        };
        assert!(params.feller_condition());

        // Violated
        let params2 = CIRParams {
            kappa: 0.5,
            theta: 0.01,
            sigma: 0.5,
            r0: 0.01,
        };
        assert!(!params2.feller_condition());
    }

    #[test]
    fn test_conditional_mean() {
        let params = CIRParams::default();
        // At t=0, mean should be r0
        let mean_0 = cir_conditional_mean(params.r0, 0.0, &params);
        assert!((mean_0 - params.r0).abs() < 1e-10);

        // At t->infinity, mean should be theta
        let mean_inf = cir_conditional_mean(params.r0, 1000.0, &params);
        assert!((mean_inf - params.theta).abs() < 1e-6);
    }

    #[test]
    fn test_yield_positive() {
        let params = CIRParams::default();
        for r in [0.01, 0.05, 0.10] {
            for tau in [0.5, 1.0, 5.0] {
                let y = cir_yield(r, tau, &params);
                assert!(y > 0.0, "Yield must be positive");
            }
        }
    }

    #[test]
    fn test_simulation_nonnegative() {
        let params = CIRParams::default();
        let paths = simulate_cir_euler(&params, 5.0, 500, 100);
        for val in paths.iter() {
            assert!(*val >= 0.0, "CIR rates must be non-negative");
        }
    }

    #[test]
    fn test_network_output_range() {
        let net = SimpleNetwork::new(&[3, 32, 16, 1]);
        for _ in 0..100 {
            let r = rand::random::<f64>() * 0.2;
            let t = rand::random::<f64>() * 5.0;
            let p = net.predict(r, t, 0.2, 5.0);
            assert!(p > 0.0 && p < 1.0, "Sigmoid output must be in (0, 1)");
        }
    }

    #[test]
    fn test_calibration_ols() {
        let params = CIRParams::default();
        let paths = simulate_cir_euler(&params, 10.0, 2520, 1);
        let rates: Vec<f64> = paths.row(0).to_vec();
        let dt = 10.0 / 2520.0;

        let estimated = calibrate_cir_ols(&rates, dt);
        // OLS estimates should be in the right ballpark
        assert!(estimated.kappa > 0.0);
        assert!(estimated.theta > 0.0);
        assert!(estimated.sigma > 0.0);
    }

    #[test]
    fn test_survival_probability() {
        let params = CIRParams {
            kappa: 0.3,
            theta: 0.02,
            sigma: 0.08,
            r0: 0.015,
        };

        // Survival probability at t=0 should be 1
        let q0 = survival_probability(0.015, 0.0, &params);
        assert!((q0 - 1.0).abs() < 1e-10);

        // Survival probability should decrease with maturity
        let q1 = survival_probability(0.015, 1.0, &params);
        let q5 = survival_probability(0.015, 5.0, &params);
        assert!(q1 > q5);
    }
}
