//! # HHL Algorithm Finance
//!
//! Educational simulator for the Harrow-Hassidim-Lloyd (HHL) quantum algorithm
//! applied to financial linear systems (portfolio optimization, risk models).
//!
//! This crate provides:
//! - A classical simulation of the HHL algorithm for 2x2 Hermitian systems
//! - Eigenvalue decomposition for 2x2 symmetric matrices
//! - Phase estimation simulation
//! - Classical Gaussian elimination for comparison
//! - Portfolio optimization as a linear system
//! - Covariance matrix construction from asset returns
//! - Bybit API data fetching

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Bybit API types
// ---------------------------------------------------------------------------

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
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// A single kline (candlestick) record.
#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

// ---------------------------------------------------------------------------
// Bybit data fetching
// ---------------------------------------------------------------------------

/// Fetch kline data from Bybit public API for the given symbol.
///
/// Uses the v5 spot market endpoint. `interval` is in minutes (e.g. "60" for
/// 1-hour candles). `limit` controls how many candles to retrieve (max 200).
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: u32) -> Result<Vec<Kline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

    if resp.ret_code != 0 {
        return Err(anyhow!("Bybit API error: {}", resp.ret_msg));
    }

    let mut klines: Vec<Kline> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Kline {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order.
    klines.reverse();
    Ok(klines)
}

// ---------------------------------------------------------------------------
// Returns & covariance
// ---------------------------------------------------------------------------

/// Compute log returns from a series of closing prices.
pub fn log_returns(closes: &[f64]) -> Vec<f64> {
    closes
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Compute the mean of a slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the sample covariance between two return series of equal length.
pub fn covariance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Series must have equal length");
    let n = a.len() as f64;
    if n <= 1.0 {
        return 0.0;
    }
    let mean_a = mean(a);
    let mean_b = mean(b);
    let cov: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - mean_a) * (bi - mean_b))
        .sum();
    cov / (n - 1.0)
}

/// Build a covariance matrix from a list of return series.
///
/// `returns_matrix[i]` is the return series for asset i.
/// Returns an N x N `Array2<f64>`.
pub fn build_covariance_matrix(returns_matrix: &[Vec<f64>]) -> Array2<f64> {
    let n = returns_matrix.len();
    let mut cov = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            cov[[i, j]] = covariance(&returns_matrix[i], &returns_matrix[j]);
        }
    }
    cov
}

// ---------------------------------------------------------------------------
// Eigenvalue decomposition for 2x2 symmetric matrices
// ---------------------------------------------------------------------------

/// Eigendecompose a 2x2 real symmetric matrix.
///
/// Given:
/// ```text
///     | a  b |
/// A = | b  d |
/// ```
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors are columns.
/// Each eigenvector is returned as `[v0, v1]`.
pub fn eigendecompose_2x2(matrix: &[[f64; 2]; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
    let a = matrix[0][0];
    let b = matrix[0][1];
    let d = matrix[1][1];

    let trace = a + d;
    let det = a * d - b * b;
    let disc = ((trace * trace - 4.0 * det).max(0.0)).sqrt();

    let lambda1 = (trace + disc) / 2.0;
    let lambda2 = (trace - disc) / 2.0;

    // Eigenvectors
    let (v1, v2) = if b.abs() > 1e-14 {
        let v1_raw = [lambda1 - d, b];
        let v2_raw = [lambda2 - d, b];
        let norm1 = (v1_raw[0] * v1_raw[0] + v1_raw[1] * v1_raw[1]).sqrt();
        let norm2 = (v2_raw[0] * v2_raw[0] + v2_raw[1] * v2_raw[1]).sqrt();
        (
            [v1_raw[0] / norm1, v1_raw[1] / norm1],
            [v2_raw[0] / norm2, v2_raw[1] / norm2],
        )
    } else {
        // Already diagonal
        ([1.0, 0.0], [0.0, 1.0])
    };

    ([lambda1, lambda2], [v1, v2])
}

// ---------------------------------------------------------------------------
// Phase estimation simulation
// ---------------------------------------------------------------------------

/// Simulate quantum phase estimation for a 2x2 system.
///
/// In the real HHL algorithm QPE extracts eigenvalue information into an
/// ancilla register.  Here we simulate it classically by decomposing |b> in
/// the eigenbasis and returning the coefficients (beta_j) and eigenvalues.
///
/// Returns `(betas, eigenvalues)`.
pub fn simulate_phase_estimation(
    eigenvalues: &[f64; 2],
    eigenvectors: &[[f64; 2]; 2],
    b: &[f64; 2],
) -> ([f64; 2], [f64; 2]) {
    // Project b onto each eigenvector: beta_j = <u_j | b>
    let beta0 = eigenvectors[0][0] * b[0] + eigenvectors[0][1] * b[1];
    let beta1 = eigenvectors[1][0] * b[0] + eigenvectors[1][1] * b[1];
    ([beta0, beta1], *eigenvalues)
}

// ---------------------------------------------------------------------------
// Controlled rotation simulation
// ---------------------------------------------------------------------------

/// Simulate the controlled rotation step of HHL.
///
/// Given eigenvalues lambda_j and a normalisation constant C, compute the
/// amplitudes that encode 1/lambda_j.
///
/// For each eigenvalue lambda_j the ancilla amplitudes are:
///   |1> amplitude = C / lambda_j
///   |0> amplitude = sqrt(1 - C^2 / lambda_j^2)
///
/// `C` must satisfy C <= |lambda_min|.
///
/// Returns the |1> amplitudes (the inversion factors).
pub fn simulate_controlled_rotation(eigenvalues: &[f64; 2], c: f64) -> [f64; 2] {
    [c / eigenvalues[0], c / eigenvalues[1]]
}

// ---------------------------------------------------------------------------
// HHL simulation (2x2)
// ---------------------------------------------------------------------------

/// Simulate the full HHL algorithm for a 2x2 real symmetric system Ax = b.
///
/// Steps:
/// 1. Eigendecompose A
/// 2. Phase estimation: project b onto eigenbasis
/// 3. Controlled rotation: apply 1/lambda factors
/// 4. Uncompute and reconstruct x in computational basis
///
/// Returns the (unnormalised) solution vector x.
pub fn hhl_simulate_2x2(a: &[[f64; 2]; 2], b: &[f64; 2]) -> Result<[f64; 2]> {
    // Step 1: eigendecompose
    let (eigenvalues, eigenvectors) = eigendecompose_2x2(a);

    // Check that eigenvalues are non-zero
    for &lam in &eigenvalues {
        if lam.abs() < 1e-14 {
            return Err(anyhow!("Matrix is singular (zero eigenvalue)"));
        }
    }

    // Step 2: phase estimation (project b onto eigenbasis)
    let (betas, _eigenvalues) = simulate_phase_estimation(&eigenvalues, &eigenvectors, b);

    // Step 3: controlled rotation -- apply 1/lambda
    let c = eigenvalues
        .iter()
        .map(|l| l.abs())
        .fold(f64::INFINITY, f64::min);
    let rotation_amplitudes = simulate_controlled_rotation(&eigenvalues, c);

    // The post-selected state coefficients in the eigenbasis:
    //   x_j = beta_j * (C / lambda_j)   (proportional to beta_j / lambda_j)
    let x_eigen = [
        betas[0] * rotation_amplitudes[0],
        betas[1] * rotation_amplitudes[1],
    ];

    // Step 4: transform back to computational basis
    // |x> = sum_j x_eigen_j |u_j>
    let x0 = x_eigen[0] * eigenvectors[0][0] + x_eigen[1] * eigenvectors[1][0];
    let x1 = x_eigen[0] * eigenvectors[0][1] + x_eigen[1] * eigenvectors[1][1];

    // Rescale: the constant C was used for normalisation; undo it so the
    // result matches the classical solution.  x_true = (1/C) * x_hhl_raw
    // But since x_eigen already has C/lambda, we have x_raw proportional to
    // C * A^{-1} b.  Dividing by C gives A^{-1} b.
    Ok([x0 / c, x1 / c])
}

// ---------------------------------------------------------------------------
// Classical solver: Gaussian elimination with partial pivoting
// ---------------------------------------------------------------------------

/// Solve Ax = b using Gaussian elimination with partial pivoting.
///
/// Works for arbitrary N x N systems (given as Vec<Vec<f64>>).
pub fn gaussian_elimination(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Err(anyhow!("Empty system"));
    }
    for row in a {
        if row.len() != n {
            return Err(anyhow!("Matrix must be square"));
        }
    }
    if b.len() != n {
        return Err(anyhow!("Dimension mismatch between A and b"));
    }

    // Augmented matrix
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    // Forward elimination
    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(anyhow!("Matrix is singular or near-singular"));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Portfolio optimization helpers
// ---------------------------------------------------------------------------

/// Formulate the minimum-variance portfolio as a linear system.
///
/// The minimum-variance portfolio solves:  Sigma * w = 1
/// where Sigma is the covariance matrix and 1 is the all-ones vector.
/// After solving, weights are normalised to sum to 1.
pub fn minimum_variance_weights(cov: &Array2<f64>) -> Result<Array1<f64>> {
    let n = cov.nrows();
    let a_vec: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| cov[[i, j]]).collect())
        .collect();
    let b_vec: Vec<f64> = vec![1.0; n];

    let raw_weights = gaussian_elimination(&a_vec, &b_vec)?;
    let sum: f64 = raw_weights.iter().sum();
    if sum.abs() < 1e-14 {
        return Err(anyhow!("Weights sum to zero; cannot normalise"));
    }
    let normalised: Vec<f64> = raw_weights.iter().map(|w| w / sum).collect();
    Ok(Array1::from(normalised))
}

/// Solve minimum-variance portfolio using HHL simulation (only for 2 assets).
pub fn minimum_variance_weights_hhl(cov: &Array2<f64>) -> Result<Array1<f64>> {
    if cov.nrows() != 2 || cov.ncols() != 2 {
        return Err(anyhow!(
            "HHL simulation only supports 2x2 systems in this educational implementation"
        ));
    }
    let a = [
        [cov[[0, 0]], cov[[0, 1]]],
        [cov[[1, 0]], cov[[1, 1]]],
    ];
    let b = [1.0, 1.0];

    let raw = hhl_simulate_2x2(&a, &b)?;
    let sum = raw[0] + raw[1];
    if sum.abs() < 1e-14 {
        return Err(anyhow!("Weights sum to zero; cannot normalise"));
    }
    Ok(Array1::from(vec![raw[0] / sum, raw[1] / sum]))
}

// ---------------------------------------------------------------------------
// Benchmarking helper
// ---------------------------------------------------------------------------

/// Compare HHL simulation vs classical solver for a 2x2 portfolio system.
///
/// Returns `(hhl_weights, classical_weights, hhl_time_us, classical_time_us)`.
pub fn compare_solvers(
    cov: &Array2<f64>,
) -> Result<(Array1<f64>, Array1<f64>, u128, u128)> {
    let t0 = Instant::now();
    let w_hhl = minimum_variance_weights_hhl(cov)?;
    let hhl_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    let w_classical = minimum_variance_weights(cov)?;
    let classical_us = t1.elapsed().as_micros();

    Ok((w_hhl, w_classical, hhl_us, classical_us))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_eigendecompose_2x2_diagonal() {
        let m = [[3.0, 0.0], [0.0, 5.0]];
        let (evals, evecs) = eigendecompose_2x2(&m);
        // Eigenvalues should be 5 and 3 (sorted largest first)
        assert!(approx_eq(evals[0], 5.0, 1e-10));
        assert!(approx_eq(evals[1], 3.0, 1e-10));
        // Eigenvectors for diagonal matrix
        assert!(approx_eq(evecs[0][0].abs(), 0.0, 1e-10) || approx_eq(evecs[0][0].abs(), 1.0, 1e-10));
    }

    #[test]
    fn test_eigendecompose_2x2_symmetric() {
        let m = [[2.0, 1.0], [1.0, 3.0]];
        let (evals, _evecs) = eigendecompose_2x2(&m);
        // trace = 5, det = 5, disc = sqrt(25-20) = sqrt(5)
        let expected1 = (5.0 + 5.0_f64.sqrt()) / 2.0;
        let expected2 = (5.0 - 5.0_f64.sqrt()) / 2.0;
        assert!(approx_eq(evals[0], expected1, 1e-10));
        assert!(approx_eq(evals[1], expected2, 1e-10));
    }

    #[test]
    fn test_phase_estimation_simulation() {
        let m = [[2.0, 1.0], [1.0, 3.0]];
        let (evals, evecs) = eigendecompose_2x2(&m);
        let b = [1.0, 0.0];
        let (betas, ret_evals) = simulate_phase_estimation(&evals, &evecs, &b);
        // betas should reconstruct b when multiplied by eigenvectors
        let reconstructed_0 = betas[0] * evecs[0][0] + betas[1] * evecs[1][0];
        let reconstructed_1 = betas[0] * evecs[0][1] + betas[1] * evecs[1][1];
        assert!(approx_eq(reconstructed_0, 1.0, 1e-10));
        assert!(approx_eq(reconstructed_1, 0.0, 1e-10));
        assert!(approx_eq(ret_evals[0], evals[0], 1e-14));
    }

    #[test]
    fn test_controlled_rotation() {
        let evals = [4.0, 2.0];
        let c = 2.0; // min eigenvalue
        let amps = simulate_controlled_rotation(&evals, c);
        assert!(approx_eq(amps[0], 0.5, 1e-10)); // C/4 = 0.5
        assert!(approx_eq(amps[1], 1.0, 1e-10)); // C/2 = 1.0
    }

    #[test]
    fn test_hhl_simulate_2x2_identity() {
        let a = [[1.0, 0.0], [0.0, 1.0]];
        let b = [3.0, 7.0];
        let x = hhl_simulate_2x2(&a, &b).unwrap();
        assert!(approx_eq(x[0], 3.0, 1e-10));
        assert!(approx_eq(x[1], 7.0, 1e-10));
    }

    #[test]
    fn test_hhl_simulate_2x2_general() {
        let a = [[2.0, 1.0], [1.0, 3.0]];
        let b = [5.0, 7.0];
        let x = hhl_simulate_2x2(&a, &b).unwrap();
        // Verify: A * x should equal b
        let check0 = a[0][0] * x[0] + a[0][1] * x[1];
        let check1 = a[1][0] * x[0] + a[1][1] * x[1];
        assert!(approx_eq(check0, b[0], 1e-8));
        assert!(approx_eq(check1, b[1], 1e-8));
    }

    #[test]
    fn test_hhl_vs_classical() {
        let a_arr = [[4.0, 2.0], [2.0, 3.0]];
        let b_arr = [1.0, 1.0];
        let x_hhl = hhl_simulate_2x2(&a_arr, &b_arr).unwrap();
        let a_vec = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let b_vec = vec![1.0, 1.0];
        let x_classical = gaussian_elimination(&a_vec, &b_vec).unwrap();
        assert!(approx_eq(x_hhl[0], x_classical[0], 1e-8));
        assert!(approx_eq(x_hhl[1], x_classical[1], 1e-8));
    }

    #[test]
    fn test_gaussian_elimination_3x3() {
        // 3x3 system: x + y + z = 6, 2y + 5z = -4, 2x + 5y - z = 27
        let a = vec![
            vec![1.0, 1.0, 1.0],
            vec![0.0, 2.0, 5.0],
            vec![2.0, 5.0, -1.0],
        ];
        let b = vec![6.0, -4.0, 27.0];
        let x = gaussian_elimination(&a, &b).unwrap();
        assert!(approx_eq(x[0], 5.0, 1e-10));
        assert!(approx_eq(x[1], 3.0, 1e-10));
        assert!(approx_eq(x[2], -2.0, 1e-10));
    }

    #[test]
    fn test_gaussian_elimination_singular() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![3.0, 6.0];
        assert!(gaussian_elimination(&a, &b).is_err());
    }

    #[test]
    fn test_log_returns() {
        let closes = vec![100.0, 105.0, 110.0];
        let rets = log_returns(&closes);
        assert_eq!(rets.len(), 2);
        assert!(approx_eq(rets[0], (105.0_f64 / 100.0).ln(), 1e-12));
        assert!(approx_eq(rets[1], (110.0_f64 / 105.0).ln(), 1e-12));
    }

    #[test]
    fn test_covariance_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cov = covariance(&a, &a);
        // Variance of [1,2,3,4,5] = 2.5
        assert!(approx_eq(cov, 2.5, 1e-10));
    }

    #[test]
    fn test_build_covariance_matrix() {
        let r1 = vec![0.01, -0.02, 0.03, 0.01];
        let r2 = vec![0.02, -0.01, 0.02, 0.00];
        let cov = build_covariance_matrix(&[r1.clone(), r2.clone()]);
        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);
        // Symmetric
        assert!(approx_eq(cov[[0, 1]], cov[[1, 0]], 1e-14));
        // Diagonal should be variance
        assert!(approx_eq(cov[[0, 0]], covariance(&r1, &r1), 1e-14));
    }

    #[test]
    fn test_minimum_variance_weights() {
        // Simple 2x2 covariance
        let cov = Array2::from_shape_vec((2, 2), vec![0.04, 0.01, 0.01, 0.09]).unwrap();
        let w = minimum_variance_weights(&cov).unwrap();
        // Weights should sum to 1
        let sum: f64 = w.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
        // Higher variance asset should get lower weight
        assert!(w[0] > w[1]);
    }

    #[test]
    fn test_minimum_variance_weights_hhl_vs_classical() {
        let cov = Array2::from_shape_vec((2, 2), vec![0.04, 0.01, 0.01, 0.09]).unwrap();
        let w_hhl = minimum_variance_weights_hhl(&cov).unwrap();
        let w_classical = minimum_variance_weights(&cov).unwrap();
        assert!(approx_eq(w_hhl[0], w_classical[0], 1e-8));
        assert!(approx_eq(w_hhl[1], w_classical[1], 1e-8));
    }

    #[test]
    fn test_compare_solvers() {
        let cov = Array2::from_shape_vec((2, 2), vec![0.04, 0.01, 0.01, 0.09]).unwrap();
        let (w_hhl, w_classical, _, _) = compare_solvers(&cov).unwrap();
        assert!(approx_eq(w_hhl[0], w_classical[0], 1e-8));
    }
}
