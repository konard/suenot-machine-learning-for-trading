//! Gaussian Process Trading Library
//!
//! Implements Gaussian Process regression with RBF and Matern 5/2 kernels
//! for uncertainty-aware trading predictions on Bybit BTCUSDT data.

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use serde::Deserialize;

// ── Kernel Trait and Implementations ──────────────────────────────────────────

/// Trait for GP kernel (covariance) functions.
pub trait Kernel: Send + Sync {
    /// Evaluate the kernel between two scalar inputs.
    fn evaluate(&self, x1: f64, x2: f64) -> f64;

    /// Build the full kernel matrix for a set of inputs.
    fn matrix(&self, xs: &[f64]) -> Array2<f64> {
        let n = xs.len();
        let mut k = Array2::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let v = self.evaluate(xs[i], xs[j]);
                k[[i, j]] = v;
                k[[j, i]] = v;
            }
        }
        k
    }

    /// Build the cross-kernel matrix between two sets of inputs.
    fn cross_matrix(&self, xs: &[f64], zs: &[f64]) -> Array2<f64> {
        let n = xs.len();
        let m = zs.len();
        let mut k = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                k[[i, j]] = self.evaluate(xs[i], zs[j]);
            }
        }
        k
    }
}

/// Radial Basis Function (Squared Exponential) kernel.
///
/// k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))
#[derive(Debug, Clone)]
pub struct RBFKernel {
    /// Length scale parameter.
    pub length_scale: f64,
    /// Signal variance (amplitude squared).
    pub signal_variance: f64,
}

impl RBFKernel {
    pub fn new(length_scale: f64, signal_variance: f64) -> Self {
        Self {
            length_scale,
            signal_variance,
        }
    }
}

impl Kernel for RBFKernel {
    fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        let diff = x1 - x2;
        let sq_dist = diff * diff;
        self.signal_variance * (-sq_dist / (2.0 * self.length_scale * self.length_scale)).exp()
    }
}

/// Matern 5/2 kernel.
///
/// k(x, x') = sigma_f^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
#[derive(Debug, Clone)]
pub struct Matern52Kernel {
    /// Length scale parameter.
    pub length_scale: f64,
    /// Signal variance (amplitude squared).
    pub signal_variance: f64,
}

impl Matern52Kernel {
    pub fn new(length_scale: f64, signal_variance: f64) -> Self {
        Self {
            length_scale,
            signal_variance,
        }
    }
}

impl Kernel for Matern52Kernel {
    fn evaluate(&self, x1: f64, x2: f64) -> f64 {
        let r = (x1 - x2).abs();
        let sqrt5 = 5.0_f64.sqrt();
        let scaled = sqrt5 * r / self.length_scale;
        let r_over_l = r / self.length_scale;
        self.signal_variance
            * (1.0 + scaled + 5.0 * r_over_l * r_over_l / 3.0)
            * (-scaled).exp()
    }
}

// ── Cholesky Decomposition ───────────────────────────────────────────────────

/// Compute the lower-triangular Cholesky factor L of a positive-definite matrix A,
/// such that A = L * L^T.
///
/// Adds a small jitter to the diagonal for numerical stability.
pub fn cholesky(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "Matrix must be square");
    let mut l = Array2::zeros((n, n));
    let jitter = 1e-8;

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diag = a[[i, i]] + jitter - sum;
                if diag <= 0.0 {
                    return Err(anyhow!(
                        "Matrix is not positive definite (diag={} at index {})",
                        diag,
                        i
                    ));
                }
                l[[i, j]] = diag.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Solve L x = b where L is lower-triangular (forward substitution).
pub fn solve_lower_triangular(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * x[j];
        }
        x[i] = (b[i] - sum) / l[[i, i]];
    }
    x
}

/// Solve L^T x = b where L is lower-triangular (backward substitution).
pub fn solve_upper_triangular(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j]; // L^T[i,j] = L[j,i]
        }
        x[i] = (b[i] - sum) / l[[i, i]];
    }
    x
}

// ── Gaussian Process ─────────────────────────────────────────────────────────

/// Enum to select kernel type for the GP.
#[derive(Debug, Clone)]
pub enum KernelType {
    RBF {
        length_scale: f64,
        signal_variance: f64,
    },
    Matern52 {
        length_scale: f64,
        signal_variance: f64,
    },
}

impl KernelType {
    fn as_kernel(&self) -> Box<dyn Kernel> {
        match self {
            KernelType::RBF {
                length_scale,
                signal_variance,
            } => Box::new(RBFKernel::new(*length_scale, *signal_variance)),
            KernelType::Matern52 {
                length_scale,
                signal_variance,
            } => Box::new(Matern52Kernel::new(*length_scale, *signal_variance)),
        }
    }
}

/// A trained Gaussian Process model.
pub struct GaussianProcess {
    /// Training inputs.
    pub x_train: Vec<f64>,
    /// Training targets.
    pub y_train: Array1<f64>,
    /// Kernel type and parameters.
    pub kernel_type: KernelType,
    /// Observation noise variance.
    pub noise_variance: f64,
    /// Cached Cholesky factor of K + sigma_n^2 I.
    l_factor: Array2<f64>,
    /// Cached alpha = (K + sigma_n^2 I)^{-1} y.
    alpha: Array1<f64>,
}

impl GaussianProcess {
    /// Fit a Gaussian Process to training data.
    pub fn fit(
        x_train: Vec<f64>,
        y_train: Vec<f64>,
        kernel_type: KernelType,
        noise_variance: f64,
    ) -> Result<Self> {
        let n = x_train.len();
        assert_eq!(n, y_train.len(), "x and y must have the same length");

        let kernel = kernel_type.as_kernel();
        let mut k = kernel.matrix(&x_train);

        // Add noise variance to diagonal
        for i in 0..n {
            k[[i, i]] += noise_variance;
        }

        let l = cholesky(&k)?;
        let y = Array1::from_vec(y_train.clone());
        let tmp = solve_lower_triangular(&l, &y);
        let alpha = solve_upper_triangular(&l, &tmp);

        Ok(Self {
            x_train,
            y_train: y,
            kernel_type,
            noise_variance,
            l_factor: l,
            alpha,
        })
    }

    /// Predict at test points, returning (means, variances).
    pub fn predict(&self, x_test: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let kernel = self.kernel_type.as_kernel();
        let k_star = kernel.cross_matrix(x_test, &self.x_train); // (m, n)
        let m = x_test.len();

        let mut means = Vec::with_capacity(m);
        let mut variances = Vec::with_capacity(m);

        for i in 0..m {
            // Mean: k_star_i . alpha
            let mut mu = 0.0;
            for j in 0..self.x_train.len() {
                mu += k_star[[i, j]] * self.alpha[j];
            }
            means.push(mu);

            // Variance: k(x*, x*) - v^T v where L v = k_star_i
            let k_star_row = Array1::from_iter(
                (0..self.x_train.len()).map(|j| k_star[[i, j]]),
            );
            let v = solve_lower_triangular(&self.l_factor, &k_star_row);
            let k_ss = kernel.evaluate(x_test[i], x_test[i]);
            let var = k_ss - v.dot(&v);
            variances.push(var.max(1e-10)); // Clamp to avoid negative variance
        }

        (means, variances)
    }

    /// Compute the log marginal likelihood.
    pub fn log_marginal_likelihood(&self) -> f64 {
        let n = self.x_train.len() as f64;

        // -0.5 * y^T alpha
        let data_fit = -0.5 * self.y_train.dot(&self.alpha);

        // -sum(log(diag(L)))
        let mut log_det = 0.0;
        for i in 0..self.x_train.len() {
            log_det += self.l_factor[[i, i]].ln();
        }
        let complexity = -log_det;

        // -(n/2) * log(2 * pi)
        let constant = -0.5 * n * (2.0 * std::f64::consts::PI).ln();

        data_fit + complexity + constant
    }
}

/// Optimize GP hyperparameters by grid search over log-space.
///
/// Returns (best_length_scale, best_signal_variance, best_noise_variance, best_lml).
pub fn optimize_hyperparameters(
    x_train: &[f64],
    y_train: &[f64],
    kernel_name: &str,
    n_candidates: usize,
) -> Result<(f64, f64, f64, f64)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut best_lml = f64::NEG_INFINITY;
    let mut best_params = (1.0, 1.0, 0.01);

    for _ in 0..n_candidates {
        let log_l: f64 = rng.gen_range(-2.0..2.0);
        let log_sf: f64 = rng.gen_range(-2.0..2.0);
        let log_sn: f64 = rng.gen_range(-4.0..0.0);

        let l = log_l.exp();
        let sf = log_sf.exp();
        let sn = log_sn.exp();

        let kernel_type = match kernel_name {
            "matern52" => KernelType::Matern52 {
                length_scale: l,
                signal_variance: sf,
            },
            _ => KernelType::RBF {
                length_scale: l,
                signal_variance: sf,
            },
        };

        if let Ok(gp) = GaussianProcess::fit(
            x_train.to_vec(),
            y_train.to_vec(),
            kernel_type,
            sn,
        ) {
            let lml = gp.log_marginal_likelihood();
            if lml > best_lml {
                best_lml = lml;
                best_params = (l, sf, sn);
            }
        }
    }

    Ok((best_params.0, best_params.1, best_params.2, best_lml))
}

// ── Bybit API ────────────────────────────────────────────────────────────────

/// Response from Bybit v5 kline endpoint.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// A single OHLCV candle.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch BTCUSDT daily candles from Bybit v5 API.
pub fn fetch_bybit_candles(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

    if resp.ret_code != 0 {
        return Err(anyhow!("Bybit API error: {}", resp.ret_msg));
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Generate a trading signal based on GP prediction.
pub fn trading_signal(predicted_mean: f64, predicted_std: f64, current_price: f64) -> &'static str {
    if predicted_mean > current_price + 2.0 * predicted_std {
        "STRONG BUY"
    } else if predicted_mean > current_price + predicted_std {
        "BUY"
    } else if predicted_mean < current_price - 2.0 * predicted_std {
        "STRONG SELL"
    } else if predicted_mean < current_price - predicted_std {
        "SELL"
    } else {
        "HOLD"
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_kernel_self() {
        let k = RBFKernel::new(1.0, 1.0);
        let val = k.evaluate(0.0, 0.0);
        assert!((val - 1.0).abs() < 1e-10, "k(x,x) should equal signal_variance");
    }

    #[test]
    fn test_rbf_kernel_symmetry() {
        let k = RBFKernel::new(0.5, 2.0);
        let v1 = k.evaluate(1.0, 2.0);
        let v2 = k.evaluate(2.0, 1.0);
        assert!((v1 - v2).abs() < 1e-10, "Kernel must be symmetric");
    }

    #[test]
    fn test_rbf_kernel_decay() {
        let k = RBFKernel::new(1.0, 1.0);
        let close = k.evaluate(0.0, 0.1);
        let far = k.evaluate(0.0, 5.0);
        assert!(close > far, "RBF kernel should decay with distance");
    }

    #[test]
    fn test_matern52_kernel_self() {
        let k = Matern52Kernel::new(1.0, 2.0);
        let val = k.evaluate(3.0, 3.0);
        assert!(
            (val - 2.0).abs() < 1e-10,
            "k(x,x) should equal signal_variance"
        );
    }

    #[test]
    fn test_matern52_kernel_symmetry() {
        let k = Matern52Kernel::new(0.5, 1.5);
        let v1 = k.evaluate(1.0, 3.0);
        let v2 = k.evaluate(3.0, 1.0);
        assert!((v1 - v2).abs() < 1e-10, "Kernel must be symmetric");
    }

    #[test]
    fn test_cholesky_identity() {
        let eye = Array2::eye(3);
        let l = cholesky(&eye).expect("Cholesky of identity should succeed");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (l[[i, j]] - expected).abs() < 1e-6,
                    "Cholesky of identity should be ~identity"
                );
            }
        }
    }

    #[test]
    fn test_cholesky_reconstruction() {
        // A = [[4, 2], [2, 3]]
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let l = cholesky(&a).expect("Cholesky should succeed");
        // Reconstruct: L * L^T should ≈ A (within jitter tolerance)
        let lt = l.t();
        let reconstructed = l.dot(&lt);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-6,
                    "L*L^T should reconstruct original matrix"
                );
            }
        }
    }

    #[test]
    fn test_gp_fit_predict() {
        // Simple test: fit a GP to y = sin(x) at a few points
        let x_train: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_train: Vec<f64> = x_train.iter().map(|x| x.sin()).collect();

        let gp = GaussianProcess::fit(
            x_train.clone(),
            y_train.clone(),
            KernelType::RBF {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            0.01,
        )
        .expect("GP fit should succeed");

        // Predict at training points: should be close to training values
        let (means, variances) = gp.predict(&x_train);
        for i in 0..x_train.len() {
            assert!(
                (means[i] - y_train[i]).abs() < 0.1,
                "GP prediction at training point should be close to target"
            );
            assert!(
                variances[i] < 0.1,
                "Variance at training point should be small"
            );
        }
    }

    #[test]
    fn test_gp_uncertainty_increases() {
        let x_train: Vec<f64> = vec![0.0, 1.0, 2.0];
        let y_train: Vec<f64> = vec![0.0, 1.0, 0.5];

        let gp = GaussianProcess::fit(
            x_train,
            y_train,
            KernelType::RBF {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            0.01,
        )
        .expect("GP fit should succeed");

        // Predict near and far from training data
        let (_, var_near) = gp.predict(&[1.0]);
        let (_, var_far) = gp.predict(&[10.0]);
        assert!(
            var_far[0] > var_near[0],
            "Uncertainty should be larger far from training data"
        );
    }

    #[test]
    fn test_log_marginal_likelihood() {
        let x_train: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
        let y_train: Vec<f64> = vec![0.0, 1.0, 0.0, -1.0];

        let gp = GaussianProcess::fit(
            x_train,
            y_train,
            KernelType::RBF {
                length_scale: 1.0,
                signal_variance: 1.0,
            },
            0.1,
        )
        .expect("GP fit should succeed");

        let lml = gp.log_marginal_likelihood();
        assert!(lml.is_finite(), "LML should be finite");
        assert!(lml < 0.0, "LML should typically be negative");
    }

    #[test]
    fn test_trading_signal() {
        assert_eq!(trading_signal(110.0, 2.0, 100.0), "STRONG BUY");
        assert_eq!(trading_signal(103.0, 2.0, 100.0), "BUY");
        assert_eq!(trading_signal(100.0, 2.0, 100.0), "HOLD");
        assert_eq!(trading_signal(97.0, 2.0, 100.0), "SELL");
        assert_eq!(trading_signal(90.0, 2.0, 100.0), "STRONG SELL");
    }
}
