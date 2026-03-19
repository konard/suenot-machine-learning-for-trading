//! # VQE Portfolio Optimization
//!
//! Classical simulation of the Variational Quantum Eigensolver (VQE) algorithm
//! applied to portfolio optimization. The portfolio selection problem is encoded
//! as a QUBO (Quadratic Unconstrained Binary Optimization) problem, which is then
//! mapped to an Ising Hamiltonian whose ground state is found via VQE.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Complex number helpers (avoid external dependency)
// ---------------------------------------------------------------------------

/// Minimal complex number type for statevector simulation.
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

// ---------------------------------------------------------------------------
// Quantum gate simulation
// ---------------------------------------------------------------------------

/// Apply a single-qubit gate (2x2 matrix) to qubit `target` in a statevector.
/// The gate is given as [[a, b], [c, d]] in row-major order.
pub fn apply_single_qubit_gate(
    state: &mut [Complex],
    n_qubits: usize,
    target: usize,
    gate: [[Complex; 2]; 2],
) {
    let dim = 1 << n_qubits;
    let mask = 1 << target;
    let mut i = 0;
    while i < dim {
        // Find pairs of indices that differ only in the target bit
        if i & mask == 0 {
            let j = i | mask;
            let a0 = state[i];
            let a1 = state[j];
            state[i] = gate[0][0] * a0 + gate[0][1] * a1;
            state[j] = gate[1][0] * a0 + gate[1][1] * a1;
        }
        i += 1;
    }
}

/// RY(theta) gate: rotation around Y-axis.
/// Matrix: [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
pub fn ry_gate(theta: f64) -> [[Complex; 2]; 2] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        [Complex::new(c, 0.0), Complex::new(-s, 0.0)],
        [Complex::new(s, 0.0), Complex::new(c, 0.0)],
    ]
}

/// RZ(theta) gate: rotation around Z-axis.
/// Matrix: [[e^{-i*t/2}, 0], [0, e^{i*t/2}]]
pub fn rz_gate(theta: f64) -> [[Complex; 2]; 2] {
    let phase_neg = Complex::from_polar(1.0, -theta / 2.0);
    let phase_pos = Complex::from_polar(1.0, theta / 2.0);
    [
        [phase_neg, Complex::zero()],
        [Complex::zero(), phase_pos],
    ]
}

/// Apply CNOT gate with given control and target qubits.
pub fn apply_cnot(state: &mut [Complex], n_qubits: usize, control: usize, target: usize) {
    let dim = 1 << n_qubits;
    let ctrl_mask = 1 << control;
    let tgt_mask = 1 << target;
    for i in 0..dim {
        // Only act when control bit is 1 and target bit is 0
        if (i & ctrl_mask) != 0 && (i & tgt_mask) == 0 {
            let j = i | tgt_mask;
            state.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// VQE Ansatz circuit
// ---------------------------------------------------------------------------

/// Number of parameters for the hardware-efficient ansatz.
/// Each layer has 2 params per qubit (RY + RZ), plus initial RY layer.
pub fn num_params(n_qubits: usize, n_layers: usize) -> usize {
    n_qubits + n_layers * 2 * n_qubits
}

/// Build the statevector by applying the hardware-efficient ansatz.
/// Parameter layout:
///   [0..n_qubits): initial RY rotations
///   For each layer l:
///     [base + 0..n_qubits): RY rotations
///     [base + n_qubits..2*n_qubits): RZ rotations
///     followed by CNOT ladder
pub fn build_ansatz_state(n_qubits: usize, n_layers: usize, params: &[f64]) -> Vec<Complex> {
    let dim = 1 << n_qubits;
    let mut state = vec![Complex::zero(); dim];
    state[0] = Complex::new(1.0, 0.0); // |00...0>

    // Initial RY layer
    for q in 0..n_qubits {
        let gate = ry_gate(params[q]);
        apply_single_qubit_gate(&mut state, n_qubits, q, gate);
    }

    // Variational layers
    let mut idx = n_qubits;
    for _layer in 0..n_layers {
        // RY rotations
        for q in 0..n_qubits {
            let gate = ry_gate(params[idx]);
            apply_single_qubit_gate(&mut state, n_qubits, q, gate);
            idx += 1;
        }
        // RZ rotations
        for q in 0..n_qubits {
            let gate = rz_gate(params[idx]);
            apply_single_qubit_gate(&mut state, n_qubits, q, gate);
            idx += 1;
        }
        // CNOT ladder
        for q in 0..n_qubits.saturating_sub(1) {
            apply_cnot(&mut state, n_qubits, q, q + 1);
        }
    }

    state
}

// ---------------------------------------------------------------------------
// QUBO matrix construction
// ---------------------------------------------------------------------------

/// Build the QUBO matrix for portfolio optimization.
///
/// # Arguments
/// * `cov_matrix` - N x N covariance matrix of asset returns
/// * `expected_returns` - N-vector of expected returns
/// * `k` - target number of assets to select
/// * `lambda_risk` - penalty weight for portfolio risk
/// * `lambda_return` - reward weight for expected return
/// * `lambda_budget` - penalty weight for budget constraint (select exactly k assets)
pub fn build_qubo_matrix(
    cov_matrix: &Array2<f64>,
    expected_returns: &Array1<f64>,
    k: usize,
    lambda_risk: f64,
    lambda_return: f64,
    lambda_budget: f64,
) -> Array2<f64> {
    let n = cov_matrix.nrows();
    let mut q = Array2::<f64>::zeros((n, n));
    let k_f = k as f64;

    // Risk term: lambda_risk * Sigma_ij / k^2
    for i in 0..n {
        for j in 0..n {
            q[[i, j]] += lambda_risk * cov_matrix[[i, j]] / (k_f * k_f);
        }
    }

    // Return term: -lambda_return * mu_i on diagonal
    for i in 0..n {
        q[[i, i]] -= lambda_return * expected_returns[i];
    }

    // Budget constraint: lambda_budget * (sum_i x_i - k)^2
    // Expanding: sum_i sum_j x_i*x_j - 2*k*sum_i x_i + k^2
    // The x_i*x_j terms: +lambda_budget for all (i,j)
    // The -2*k*x_i terms: -2*k*lambda_budget on diagonal
    // The k^2 constant is dropped (doesn't affect optimization)
    for i in 0..n {
        for j in 0..n {
            q[[i, j]] += lambda_budget;
        }
        q[[i, i]] -= 2.0 * k_f * lambda_budget;
    }

    q
}

// ---------------------------------------------------------------------------
// Cost function evaluation
// ---------------------------------------------------------------------------

/// Evaluate the QUBO cost for a given binary string.
pub fn qubo_cost(qubo: &Array2<f64>, bits: &[bool]) -> f64 {
    let n = qubo.nrows();
    let mut cost = 0.0;
    for i in 0..n {
        if !bits[i] {
            continue;
        }
        for j in 0..n {
            if bits[j] {
                cost += qubo[[i, j]];
            }
        }
    }
    cost
}

/// Evaluate the VQE cost function: expectation value of the QUBO Hamiltonian
/// with respect to the ansatz statevector.
///
/// For a diagonal Hamiltonian, <H> = sum_x |a_x|^2 * f(x).
pub fn vqe_cost(
    qubo: &Array2<f64>,
    n_qubits: usize,
    n_layers: usize,
    params: &[f64],
) -> f64 {
    let state = build_ansatz_state(n_qubits, n_layers, params);
    let dim = 1 << n_qubits;
    let mut energy = 0.0;

    for idx in 0..dim {
        let prob = state[idx].norm_sq();
        if prob < 1e-15 {
            continue;
        }
        // Convert index to bit string
        let bits: Vec<bool> = (0..n_qubits).map(|q| (idx >> q) & 1 == 1).collect();
        let cost = qubo_cost(qubo, &bits);
        energy += prob * cost;
    }
    energy
}

// ---------------------------------------------------------------------------
// Classical optimizer: coordinate descent with random restarts
// ---------------------------------------------------------------------------

/// Configuration for the VQE optimizer.
#[derive(Debug, Clone)]
pub struct VqeConfig {
    pub n_qubits: usize,
    pub n_layers: usize,
    pub n_restarts: usize,
    pub n_iterations: usize,
    pub perturbation_size: f64,
    pub n_perturbations: usize,
}

impl Default for VqeConfig {
    fn default() -> Self {
        Self {
            n_qubits: 3,
            n_layers: 2,
            n_restarts: 5,
            n_iterations: 100,
            perturbation_size: 0.3,
            n_perturbations: 8,
        }
    }
}

/// Result of VQE optimization.
#[derive(Debug, Clone)]
pub struct VqeResult {
    pub optimal_params: Vec<f64>,
    pub optimal_energy: f64,
    pub optimal_portfolio: Vec<bool>,
    pub portfolio_weights: Vec<f64>,
}

/// Run VQE optimization for portfolio selection.
pub fn run_vqe(qubo: &Array2<f64>, config: &VqeConfig) -> VqeResult {
    let n_params = num_params(config.n_qubits, config.n_layers);
    let mut rng = rand::thread_rng();

    let mut best_energy = f64::MAX;
    let mut best_params = vec![0.0; n_params];

    for _restart in 0..config.n_restarts {
        // Random initialization
        let mut params: Vec<f64> = (0..n_params)
            .map(|_| rng.gen_range(0.0..2.0 * PI))
            .collect();
        let mut current_energy = vqe_cost(qubo, config.n_qubits, config.n_layers, &params);

        for _iter in 0..config.n_iterations {
            let mut improved = false;

            // Coordinate descent: optimize each parameter independently
            for p in 0..n_params {
                let original = params[p];
                let mut best_val = original;
                let mut best_e = current_energy;

                // Try several perturbations
                for k in 0..config.n_perturbations {
                    let angle = (k as f64 / config.n_perturbations as f64) * 2.0 * PI;
                    let delta = config.perturbation_size * angle.cos();
                    params[p] = original + delta;
                    let e = vqe_cost(qubo, config.n_qubits, config.n_layers, &params);
                    if e < best_e {
                        best_e = e;
                        best_val = params[p];
                    }
                }

                if best_e < current_energy - 1e-10 {
                    params[p] = best_val;
                    current_energy = best_e;
                    improved = true;
                } else {
                    params[p] = original;
                }
            }

            if !improved {
                break;
            }
        }

        if current_energy < best_energy {
            best_energy = current_energy;
            best_params = params;
        }
    }

    // Extract optimal portfolio from the statevector
    let state = build_ansatz_state(config.n_qubits, config.n_layers, &best_params);
    let dim = 1 << config.n_qubits;

    // Find the most probable basis state
    let mut max_prob = 0.0;
    let mut max_idx = 0;
    for idx in 0..dim {
        let prob = state[idx].norm_sq();
        if prob > max_prob {
            max_prob = prob;
            max_idx = idx;
        }
    }

    let optimal_portfolio: Vec<bool> = (0..config.n_qubits)
        .map(|q| (max_idx >> q) & 1 == 1)
        .collect();

    let n_selected = optimal_portfolio.iter().filter(|&&b| b).count().max(1);
    let weight = 1.0 / n_selected as f64;
    let portfolio_weights: Vec<f64> = optimal_portfolio
        .iter()
        .map(|&b| if b { weight } else { 0.0 })
        .collect();

    VqeResult {
        optimal_params: best_params,
        optimal_energy: best_energy,
        optimal_portfolio,
        portfolio_weights,
    }
}

// ---------------------------------------------------------------------------
// Brute-force solver for verification
// ---------------------------------------------------------------------------

/// Find the optimal portfolio by brute-force enumeration of all 2^N possibilities.
pub fn brute_force_solve(qubo: &Array2<f64>, n: usize) -> (Vec<bool>, f64) {
    let dim = 1 << n;
    let mut best_cost = f64::MAX;
    let mut best_bits = vec![false; n];

    for idx in 0..dim {
        let bits: Vec<bool> = (0..n).map(|q| (idx >> q) & 1 == 1).collect();
        let cost = qubo_cost(qubo, &bits);
        if cost < best_cost {
            best_cost = cost;
            best_bits = bits;
        }
    }

    (best_bits, best_cost)
}

// ---------------------------------------------------------------------------
// Classical mean-variance optimization (for comparison)
// ---------------------------------------------------------------------------

/// Simple equal-weight portfolio among the selected assets.
pub fn classical_equal_weight_portfolio(
    cov_matrix: &Array2<f64>,
    expected_returns: &Array1<f64>,
    selected: &[bool],
) -> (f64, f64) {
    let n_selected = selected.iter().filter(|&&b| b).count();
    if n_selected == 0 {
        return (0.0, 0.0);
    }
    let w = 1.0 / n_selected as f64;

    // Expected return
    let mut ret = 0.0;
    for (i, &s) in selected.iter().enumerate() {
        if s {
            ret += w * expected_returns[i];
        }
    }

    // Portfolio variance
    let mut var = 0.0;
    for (i, &si) in selected.iter().enumerate() {
        if !si {
            continue;
        }
        for (j, &sj) in selected.iter().enumerate() {
            if sj {
                var += w * w * cov_matrix[[i, j]];
            }
        }
    }

    (ret, var)
}

/// Find the best subset of k assets by brute-force classical enumeration.
/// Returns (selected assets, expected return, variance).
pub fn classical_best_subset(
    cov_matrix: &Array2<f64>,
    expected_returns: &Array1<f64>,
    n: usize,
    k: usize,
    lambda_risk: f64,
    lambda_return: f64,
) -> (Vec<bool>, f64, f64) {
    let dim = 1 << n;
    let mut best_score = f64::MAX;
    let mut best_bits = vec![false; n];

    for idx in 0..dim {
        let bits: Vec<bool> = (0..n).map(|q| (idx >> q) & 1 == 1).collect();
        let count = bits.iter().filter(|&&b| b).count();
        if count != k {
            continue;
        }
        let (ret, var) = classical_equal_weight_portfolio(cov_matrix, expected_returns, &bits);
        let score = lambda_risk * var - lambda_return * ret;
        if score < best_score {
            best_score = score;
            best_bits = bits;
        }
    }

    let (ret, var) = classical_equal_weight_portfolio(cov_matrix, expected_returns, &best_bits);
    (best_bits, ret, var)
}

// ---------------------------------------------------------------------------
// Bybit API data fetching
// ---------------------------------------------------------------------------

/// Raw kline response from Bybit V5 API.
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
    pub list: Vec<Vec<String>>,
}

/// Fetch kline (candlestick) data from Bybit V5 API.
///
/// Returns a vector of closing prices (oldest first).
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> anyhow::Result<Vec<f64>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("User-Agent", "vqe-portfolio-optimizer/0.1")
        .send()?
        .json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
    }

    // Bybit returns data in reverse chronological order.
    // Each entry: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    let mut closes: Vec<f64> = resp
        .result
        .list
        .iter()
        .map(|candle| candle[4].parse::<f64>().unwrap_or(0.0))
        .collect();

    closes.reverse(); // oldest first
    Ok(closes)
}

/// Compute log returns from a price series.
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Compute the covariance matrix and mean returns from multiple return series.
///
/// All return series must have the same length. Returns (cov_matrix, mean_returns).
pub fn compute_covariance_and_means(
    returns: &[Vec<f64>],
) -> (Array2<f64>, Array1<f64>) {
    let n_assets = returns.len();
    let n_obs = returns[0].len();

    // Mean returns
    let means: Vec<f64> = returns
        .iter()
        .map(|r| r.iter().sum::<f64>() / n_obs as f64)
        .collect();

    // Covariance matrix
    let mut cov = Array2::<f64>::zeros((n_assets, n_assets));
    for i in 0..n_assets {
        for j in 0..n_assets {
            let mut sum = 0.0;
            for t in 0..n_obs {
                sum += (returns[i][t] - means[i]) * (returns[j][t] - means[j]);
            }
            cov[[i, j]] = sum / (n_obs as f64 - 1.0);
        }
    }

    let mean_arr = Array1::from(means);
    (cov, mean_arr)
}

// ---------------------------------------------------------------------------
// Utility: print helpers
// ---------------------------------------------------------------------------

/// Pretty-print a portfolio allocation.
pub fn print_portfolio(asset_names: &[&str], weights: &[f64]) {
    println!("\nPortfolio Allocation:");
    println!("{:-<40}", "");
    for (i, name) in asset_names.iter().enumerate() {
        if weights[i] > 1e-6 {
            println!("  {:<12} {:>6.2}%", name, weights[i] * 100.0);
        }
    }
    println!("{:-<40}", "");
}

/// Pretty-print a covariance matrix.
pub fn print_covariance_matrix(asset_names: &[&str], cov: &Array2<f64>) {
    println!("\nCovariance Matrix:");
    print!("{:>12}", "");
    for name in asset_names {
        print!("{:>12}", name);
    }
    println!();
    for (i, name) in asset_names.iter().enumerate() {
        print!("{:>12}", name);
        for j in 0..asset_names.len() {
            print!("{:>12.6}", cov[[i, j]]);
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ry_gate_identity() {
        // RY(0) should be identity
        let gate = ry_gate(0.0);
        assert!((gate[0][0].re - 1.0).abs() < 1e-10);
        assert!((gate[1][1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_statevector_normalization() {
        let n_qubits = 3;
        let n_layers = 2;
        let n_p = num_params(n_qubits, n_layers);
        let params: Vec<f64> = (0..n_p).map(|i| i as f64 * 0.5).collect();
        let state = build_ansatz_state(n_qubits, n_layers, &params);
        let norm: f64 = state.iter().map(|c| c.norm_sq()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Statevector not normalized: {}",
            norm
        );
    }

    #[test]
    fn test_qubo_cost_empty() {
        let qubo = Array2::<f64>::zeros((3, 3));
        let bits = vec![false, false, false];
        assert!((qubo_cost(&qubo, &bits)).abs() < 1e-10);
    }

    #[test]
    fn test_brute_force_finds_minimum() {
        // Simple 2-asset QUBO where selecting asset 0 only is optimal
        let mut qubo = Array2::<f64>::zeros((2, 2));
        qubo[[0, 0]] = -5.0; // Strong incentive to select asset 0
        qubo[[1, 1]] = 3.0; // Penalty for selecting asset 1
        qubo[[0, 1]] = 1.0;
        qubo[[1, 0]] = 1.0;

        let (bits, cost) = brute_force_solve(&qubo, 2);
        assert!(bits[0]); // Asset 0 should be selected
        assert!(!bits[1]); // Asset 1 should not be selected
        assert!((cost - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_vqe_simple_case() {
        // 2-qubit problem: select asset 0 only
        let mut qubo = Array2::<f64>::zeros((2, 2));
        qubo[[0, 0]] = -10.0;
        qubo[[1, 1]] = 5.0;
        qubo[[0, 1]] = 1.0;
        qubo[[1, 0]] = 1.0;

        let config = VqeConfig {
            n_qubits: 2,
            n_layers: 2,
            n_restarts: 10,
            n_iterations: 150,
            perturbation_size: 0.4,
            n_perturbations: 12,
        };

        let result = run_vqe(&qubo, &config);
        // VQE should find that selecting only asset 0 is optimal
        let (bf_bits, bf_cost) = brute_force_solve(&qubo, 2);
        println!(
            "VQE energy: {:.4}, Brute-force cost: {:.4}",
            result.optimal_energy, bf_cost
        );
        println!("VQE portfolio: {:?}, BF: {:?}", result.optimal_portfolio, bf_bits);
        // The VQE energy should be close to or below the brute-force minimum
        // (it's an expectation value, so can be slightly above)
        assert!(result.optimal_energy < bf_cost + 1.0);
    }
}
