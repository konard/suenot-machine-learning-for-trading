//! # VQE Portfolio Optimization
//!
//! Classical simulation of the Variational Quantum Eigensolver (VQE) algorithm
//! applied to portfolio optimization via QUBO formulation.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;
use std::f64::consts::PI;

// ============================================================================
// Complex number type
// ============================================================================

/// A minimal complex number represented as (real, imaginary) f64 pair.
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

    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    pub fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
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

impl std::ops::AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
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

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl std::ops::Mul<Complex> for f64 {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Complex {
        Complex {
            re: self * rhs.re,
            im: self * rhs.im,
        }
    }
}

// ============================================================================
// 2x2 Complex matrix type
// ============================================================================

/// A 2x2 complex matrix stored as [[row0], [row1]].
#[derive(Debug, Clone, Copy)]
pub struct Mat2x2 {
    pub m: [[Complex; 2]; 2],
}

impl Mat2x2 {
    pub fn new(m00: Complex, m01: Complex, m10: Complex, m11: Complex) -> Self {
        Self {
            m: [[m00, m01], [m10, m11]],
        }
    }

    pub fn identity() -> Self {
        Self::new(Complex::one(), Complex::zero(), Complex::zero(), Complex::one())
    }

    pub fn mul_vec(&self, v: [Complex; 2]) -> [Complex; 2] {
        [
            self.m[0][0] * v[0] + self.m[0][1] * v[1],
            self.m[1][0] * v[0] + self.m[1][1] * v[1],
        ]
    }

    pub fn matmul(&self, other: &Mat2x2) -> Mat2x2 {
        let mut result = [[Complex::zero(); 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    result[i][j] = result[i][j] + self.m[i][k] * other.m[k][j];
                }
            }
        }
        Mat2x2 { m: result }
    }
}

impl std::ops::Mul<f64> for Mat2x2 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Mat2x2::new(
            self.m[0][0] * rhs,
            self.m[0][1] * rhs,
            self.m[1][0] * rhs,
            self.m[1][1] * rhs,
        )
    }
}

impl std::ops::Add for Mat2x2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Mat2x2::new(
            self.m[0][0] + rhs.m[0][0],
            self.m[0][1] + rhs.m[0][1],
            self.m[1][0] + rhs.m[1][0],
            self.m[1][1] + rhs.m[1][1],
        )
    }
}

impl std::ops::Sub for Mat2x2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Mat2x2::new(
            self.m[0][0] - rhs.m[0][0],
            self.m[0][1] - rhs.m[0][1],
            self.m[1][0] - rhs.m[1][0],
            self.m[1][1] - rhs.m[1][1],
        )
    }
}

// ============================================================================
// Pauli matrices
// ============================================================================

/// Pauli-I (identity) matrix.
pub fn pauli_i() -> Mat2x2 {
    Mat2x2::identity()
}

/// Pauli-X matrix.
pub fn pauli_x() -> Mat2x2 {
    Mat2x2::new(
        Complex::zero(),
        Complex::one(),
        Complex::one(),
        Complex::zero(),
    )
}

/// Pauli-Y matrix.
pub fn pauli_y() -> Mat2x2 {
    Mat2x2::new(
        Complex::zero(),
        Complex::new(0.0, -1.0),
        Complex::new(0.0, 1.0),
        Complex::zero(),
    )
}

/// Pauli-Z matrix.
pub fn pauli_z() -> Mat2x2 {
    Mat2x2::new(
        Complex::one(),
        Complex::zero(),
        Complex::zero(),
        Complex::new(-1.0, 0.0),
    )
}

// ============================================================================
// Rotation gates
// ============================================================================

/// Rx(theta) = cos(theta/2)*I - i*sin(theta/2)*X
pub fn rx(theta: f64) -> Mat2x2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    Mat2x2::new(
        Complex::new(c, 0.0),
        Complex::new(0.0, -s),
        Complex::new(0.0, -s),
        Complex::new(c, 0.0),
    )
}

/// Ry(theta) = cos(theta/2)*I - i*sin(theta/2)*Y
pub fn ry(theta: f64) -> Mat2x2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    Mat2x2::new(
        Complex::new(c, 0.0),
        Complex::new(-s, 0.0),
        Complex::new(s, 0.0),
        Complex::new(c, 0.0),
    )
}

/// Rz(theta) = cos(theta/2)*I - i*sin(theta/2)*Z
pub fn rz(theta: f64) -> Mat2x2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    Mat2x2::new(
        Complex::new(c, -s),
        Complex::zero(),
        Complex::zero(),
        Complex::new(c, s),
    )
}

// ============================================================================
// State vector simulation
// ============================================================================

/// Apply a single-qubit gate to qubit `target` in an n-qubit state vector.
/// The state vector has length 2^n.
pub fn apply_single_qubit_gate(state: &mut [Complex], n_qubits: usize, target: usize, gate: &Mat2x2) {
    let dim = 1 << n_qubits;
    assert_eq!(state.len(), dim);
    let step = 1 << target;
    for block_start in (0..dim).step_by(step << 1) {
        for i in block_start..block_start + step {
            let j = i + step;
            let a = state[i];
            let b = state[j];
            state[i] = gate.m[0][0] * a + gate.m[0][1] * b;
            state[j] = gate.m[1][0] * a + gate.m[1][1] * b;
        }
    }
}

/// Apply a CNOT gate with `control` and `target` qubits.
pub fn apply_cnot(state: &mut [Complex], n_qubits: usize, control: usize, target: usize) {
    let dim = 1 << n_qubits;
    assert_eq!(state.len(), dim);
    for i in 0..dim {
        // If control bit is set and target bit is 0, swap with flipped target
        if (i >> control) & 1 == 1 && (i >> target) & 1 == 0 {
            let j = i | (1 << target);
            state.swap(i, j);
        }
    }
}

/// Initialize a state vector to |0...0>.
pub fn init_state(n_qubits: usize) -> Vec<Complex> {
    let dim = 1 << n_qubits;
    let mut state = vec![Complex::zero(); dim];
    state[0] = Complex::one();
    state
}

// ============================================================================
// VQE Ansatz circuit
// ============================================================================

/// Number of parameters for the hardware-efficient ansatz.
/// Each layer has 2 rotation angles per qubit (Ry, Rz).
pub fn num_params(n_qubits: usize, n_layers: usize) -> usize {
    n_qubits * 2 * n_layers
}

/// Apply the hardware-efficient ansatz to a state vector.
/// Parameters are ordered as: [layer0_qubit0_ry, layer0_qubit0_rz, layer0_qubit1_ry, ...]
pub fn apply_ansatz(state: &mut [Complex], n_qubits: usize, n_layers: usize, params: &[f64]) {
    assert_eq!(params.len(), num_params(n_qubits, n_layers));
    let mut idx = 0;
    for _layer in 0..n_layers {
        // Single-qubit rotations
        for qubit in 0..n_qubits {
            let ry_gate = ry(params[idx]);
            let rz_gate = rz(params[idx + 1]);
            apply_single_qubit_gate(state, n_qubits, qubit, &ry_gate);
            apply_single_qubit_gate(state, n_qubits, qubit, &rz_gate);
            idx += 2;
        }
        // Entangling layer: linear chain of CNOTs
        for qubit in 0..n_qubits.saturating_sub(1) {
            apply_cnot(state, n_qubits, qubit, qubit + 1);
        }
    }
}

// ============================================================================
// Ising Hamiltonian
// ============================================================================

/// Ising model: H = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + offset
#[derive(Debug, Clone)]
pub struct IsingHamiltonian {
    pub n_qubits: usize,
    pub h: Vec<f64>,           // local fields
    pub j_couplings: Vec<(usize, usize, f64)>, // (i, j, J_ij) with i < j
    pub offset: f64,
}

/// Compute <psi|H|psi> for an Ising Hamiltonian.
/// Uses the fact that Z_i|x> = (-1)^{x_i}|x> for computational basis states.
pub fn ising_expectation(state: &[Complex], hamiltonian: &IsingHamiltonian) -> f64 {
    let dim = state.len();
    let mut energy = 0.0;

    for basis in 0..dim {
        let prob = state[basis].norm_sq();
        if prob < 1e-15 {
            continue;
        }

        let mut basis_energy = 0.0;

        // Local field terms
        for (i, &hi) in hamiltonian.h.iter().enumerate() {
            let zi: f64 = if (basis >> i) & 1 == 1 { -1.0 } else { 1.0 };
            basis_energy += hi * zi;
        }

        // Coupling terms
        for &(i, j, jij) in &hamiltonian.j_couplings {
            let zi: f64 = if (basis >> i) & 1 == 1 { -1.0 } else { 1.0 };
            let zj: f64 = if (basis >> j) & 1 == 1 { -1.0 } else { 1.0 };
            basis_energy += jij * zi * zj;
        }

        energy += prob * basis_energy;
    }

    energy + hamiltonian.offset
}

// ============================================================================
// QUBO to Ising conversion
// ============================================================================

/// Convert a QUBO matrix Q to an Ising Hamiltonian.
/// QUBO: min x^T Q x, x_i in {0,1}
/// Substitution: x_i = (1 - z_i) / 2
pub fn qubo_to_ising(qubo: &Array2<f64>) -> IsingHamiltonian {
    let n = qubo.nrows();
    assert_eq!(n, qubo.ncols());

    let mut offset = 0.0;
    let mut h = vec![0.0; n];
    let mut j_couplings = Vec::new();

    // Expand x_i = (1 - z_i)/2 into the QUBO
    // x_i * x_j = (1 - z_i)(1 - z_j)/4 = (1 - z_i - z_j + z_i*z_j)/4
    // For diagonal (i==j): x_i^2 = x_i = (1 - z_i)/2

    for i in 0..n {
        // Diagonal terms: Q_ii * x_i = Q_ii * (1 - z_i)/2
        offset += qubo[[i, i]] / 2.0;
        h[i] -= qubo[[i, i]] / 2.0;

        for j in (i + 1)..n {
            let qij = qubo[[i, j]] + qubo[[j, i]]; // symmetrize
            // qij * x_i * x_j = qij * (1 - z_i - z_j + z_i*z_j) / 4
            offset += qij / 4.0;
            h[i] -= qij / 4.0;
            h[j] -= qij / 4.0;
            j_couplings.push((i, j, qij / 4.0));
        }
    }

    IsingHamiltonian {
        n_qubits: n,
        h,
        j_couplings,
        offset,
    }
}

// ============================================================================
// Portfolio QUBO construction
// ============================================================================

/// Build the QUBO matrix for portfolio optimization.
///
/// # Arguments
/// * `returns` - expected returns vector (n_assets)
/// * `cov_matrix` - covariance matrix (n_assets x n_assets)
/// * `risk_aversion` - gamma parameter
/// * `penalty` - lambda for budget constraint
/// * `bits_per_asset` - number of binary variables per asset weight
pub fn build_portfolio_qubo(
    returns: &Array1<f64>,
    cov_matrix: &Array2<f64>,
    risk_aversion: f64,
    penalty: f64,
    bits_per_asset: usize,
) -> Array2<f64> {
    let n_assets = returns.len();
    let n_bits = n_assets * bits_per_asset;
    let max_val = (1 << bits_per_asset) as f64 - 1.0;

    let mut qubo = Array2::<f64>::zeros((n_bits, n_bits));

    // Encode weights: w_i = sum_k (2^k / max_val) * x_{i*bits+k}
    // Build coefficient vectors for each asset
    let mut coeffs = vec![vec![0.0; bits_per_asset]; n_assets];
    for i in 0..n_assets {
        for k in 0..bits_per_asset {
            coeffs[i][k] = (1 << k) as f64 / max_val;
        }
    }

    // Risk term: gamma * sum_{i,j} cov[i,j] * w_i * w_j
    for i in 0..n_assets {
        for j in 0..n_assets {
            for ki in 0..bits_per_asset {
                for kj in 0..bits_per_asset {
                    let bi = i * bits_per_asset + ki;
                    let bj = j * bits_per_asset + kj;
                    qubo[[bi, bj]] += risk_aversion * cov_matrix[[i, j]] * coeffs[i][ki] * coeffs[j][kj];
                }
            }
        }
    }

    // Return term: -sum_i mu_i * w_i
    for i in 0..n_assets {
        for k in 0..bits_per_asset {
            let bi = i * bits_per_asset + k;
            qubo[[bi, bi]] -= returns[i] * coeffs[i][k];
        }
    }

    // Budget constraint penalty: lambda * (sum_i w_i - 1)^2
    // = lambda * (sum_i w_i)^2 - 2*lambda*(sum_i w_i) + lambda
    // The constant lambda doesn't affect optimization.
    // (sum_i w_i)^2 = sum over all bit pairs
    for i in 0..n_assets {
        for j in 0..n_assets {
            for ki in 0..bits_per_asset {
                for kj in 0..bits_per_asset {
                    let bi = i * bits_per_asset + ki;
                    let bj = j * bits_per_asset + kj;
                    qubo[[bi, bj]] += penalty * coeffs[i][ki] * coeffs[j][kj];
                }
            }
        }
    }

    // -2 * lambda * sum_i w_i
    for i in 0..n_assets {
        for k in 0..bits_per_asset {
            let bi = i * bits_per_asset + k;
            qubo[[bi, bi]] -= 2.0 * penalty * coeffs[i][k];
        }
    }

    qubo
}

/// Decode binary string to portfolio weights.
pub fn decode_weights(bitstring: usize, n_assets: usize, bits_per_asset: usize) -> Vec<f64> {
    let max_val = (1 << bits_per_asset) as f64 - 1.0;
    let mut weights = vec![0.0; n_assets];
    for i in 0..n_assets {
        let mut val = 0.0;
        for k in 0..bits_per_asset {
            let bit_idx = i * bits_per_asset + k;
            if (bitstring >> bit_idx) & 1 == 1 {
                val += (1 << k) as f64 / max_val;
            }
        }
        weights[i] = val;
    }
    weights
}

/// Normalize weights so they sum to 1.
pub fn normalize_weights(weights: &[f64]) -> Vec<f64> {
    let sum: f64 = weights.iter().sum();
    if sum < 1e-10 {
        vec![1.0 / weights.len() as f64; weights.len()]
    } else {
        weights.iter().map(|w| w / sum).collect()
    }
}

// ============================================================================
// VQE Optimization (Nelder-Mead)
// ============================================================================

/// Configuration for the VQE optimizer.
#[derive(Debug, Clone)]
pub struct VqeConfig {
    pub n_qubits: usize,
    pub n_layers: usize,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for VqeConfig {
    fn default() -> Self {
        Self {
            n_qubits: 8,
            n_layers: 2,
            max_iterations: 500,
            tolerance: 1e-8,
        }
    }
}

/// Result of VQE optimization.
#[derive(Debug, Clone)]
pub struct VqeResult {
    pub optimal_params: Vec<f64>,
    pub optimal_energy: f64,
    pub iterations: usize,
    pub optimal_bitstring: usize,
    pub weights: Vec<f64>,
}

/// Evaluate the VQE energy for given parameters.
pub fn vqe_energy(params: &[f64], config: &VqeConfig, hamiltonian: &IsingHamiltonian) -> f64 {
    let mut state = init_state(config.n_qubits);
    apply_ansatz(&mut state, config.n_qubits, config.n_layers, params);
    ising_expectation(&state, hamiltonian)
}

/// Find the most probable bitstring from the state vector.
pub fn most_probable_bitstring(state: &[Complex]) -> usize {
    let mut max_prob = 0.0;
    let mut max_idx = 0;
    for (i, c) in state.iter().enumerate() {
        let prob = c.norm_sq();
        if prob > max_prob {
            max_prob = prob;
            max_idx = i;
        }
    }
    max_idx
}

/// Run VQE optimization using Nelder-Mead simplex method.
pub fn run_vqe(hamiltonian: &IsingHamiltonian, config: &VqeConfig) -> VqeResult {
    let n_params = num_params(config.n_qubits, config.n_layers);
    let mut rng = rand::thread_rng();

    // Initialize simplex: n+1 points
    let n = n_params;
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut values: Vec<f64> = Vec::with_capacity(n + 1);

    // First point: random
    let p0: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..2.0 * PI)).collect();
    let v0 = vqe_energy(&p0, config, hamiltonian);
    simplex.push(p0);
    values.push(v0);

    // Remaining points: perturbed from p0
    for i in 0..n {
        let mut p = simplex[0].clone();
        p[i] += 0.5;
        let v = vqe_energy(&p, config, hamiltonian);
        simplex.push(p);
        values.push(v);
    }

    let alpha = 1.0; // reflection
    let gamma_nm = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma = 0.5; // shrink

    let mut iterations = 0;

    for _iter in 0..config.max_iterations {
        iterations = _iter + 1;

        // Sort simplex by values
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

        let sorted_simplex: Vec<Vec<f64>> = indices.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_values: Vec<f64> = indices.iter().map(|&i| values[i]).collect();
        simplex = sorted_simplex;
        values = sorted_values;

        // Check convergence
        let range = values[n] - values[0];
        if range < config.tolerance {
            break;
        }

        // Centroid of all points except the worst
        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let fr = vqe_energy(&reflected, config, hamiltonian);

        if fr < values[0] {
            // Expansion
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma_nm * (reflected[j] - centroid[j]))
                .collect();
            let fe = vqe_energy(&expanded, config, hamiltonian);
            if fe < fr {
                simplex[n] = expanded;
                values[n] = fe;
            } else {
                simplex[n] = reflected;
                values[n] = fr;
            }
        } else if fr < values[n - 1] {
            simplex[n] = reflected;
            values[n] = fr;
        } else {
            // Contraction
            let contracted: Vec<f64> = (0..n)
                .map(|j| centroid[j] + rho * (simplex[n][j] - centroid[j]))
                .collect();
            let fc = vqe_energy(&contracted, config, hamiltonian);
            if fc < values[n] {
                simplex[n] = contracted;
                values[n] = fc;
            } else {
                // Shrink
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                    }
                    values[i] = vqe_energy(&simplex[i], config, hamiltonian);
                }
            }
        }
    }

    // Find best
    let mut best_idx = 0;
    for i in 1..=n {
        if values[i] < values[best_idx] {
            best_idx = i;
        }
    }

    let optimal_params = simplex[best_idx].clone();
    let optimal_energy = values[best_idx];

    // Get the optimal bitstring
    let mut state = init_state(config.n_qubits);
    apply_ansatz(&mut state, config.n_qubits, config.n_layers, &optimal_params);
    let optimal_bitstring = most_probable_bitstring(&state);

    VqeResult {
        optimal_params,
        optimal_energy,
        iterations,
        optimal_bitstring,
        weights: vec![], // filled by caller
    }
}

// ============================================================================
// Portfolio analytics
// ============================================================================

/// Compute portfolio expected return.
pub fn portfolio_return(weights: &[f64], returns: &Array1<f64>) -> f64 {
    weights
        .iter()
        .zip(returns.iter())
        .map(|(w, r)| w * r)
        .sum()
}

/// Compute portfolio variance (risk^2).
pub fn portfolio_variance(weights: &[f64], cov_matrix: &Array2<f64>) -> f64 {
    let n = weights.len();
    let mut var = 0.0;
    for i in 0..n {
        for j in 0..n {
            var += weights[i] * weights[j] * cov_matrix[[i, j]];
        }
    }
    var
}

/// Compute portfolio risk (standard deviation).
pub fn portfolio_risk(weights: &[f64], cov_matrix: &Array2<f64>) -> f64 {
    portfolio_variance(weights, cov_matrix).sqrt()
}

/// Compute Sharpe ratio (assuming risk-free rate = 0).
pub fn sharpe_ratio(weights: &[f64], returns: &Array1<f64>, cov_matrix: &Array2<f64>) -> f64 {
    let ret = portfolio_return(weights, returns);
    let risk = portfolio_risk(weights, cov_matrix);
    if risk < 1e-10 {
        0.0
    } else {
        ret / risk
    }
}

/// Simple mean-variance optimization (analytical solution for unconstrained).
/// Solves: min gamma * w^T Sigma w - mu^T w, subject to sum(w)=1
/// Using Lagrangian: w = (1/(2*gamma)) * Sigma^{-1} * (mu + lambda*1)
/// This is a simplified version using equal-risk-contribution as fallback.
pub fn classical_mean_variance(
    returns: &Array1<f64>,
    cov_matrix: &Array2<f64>,
    risk_aversion: f64,
) -> Vec<f64> {
    let n = returns.len();

    // Simple approach: use inverse-variance weighting as proxy
    // For a proper solution we'd need matrix inversion, but this serves as comparison
    let mut inv_var = vec![0.0; n];
    for i in 0..n {
        let var = cov_matrix[[i, i]];
        if var > 1e-10 {
            inv_var[i] = 1.0 / var;
        }
    }

    // Adjust by expected returns
    let mut weights: Vec<f64> = (0..n)
        .map(|i| inv_var[i] * (1.0 + returns[i] / risk_aversion))
        .collect();

    // Ensure non-negative
    for w in weights.iter_mut() {
        if *w < 0.0 {
            *w = 0.0;
        }
    }

    normalize_weights(&weights)
}

// ============================================================================
// Bybit API integration
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitKlineResult {
    pub list: Vec<Vec<String>>,
}

/// Fetch kline (candlestick) data from Bybit API.
/// Returns a vector of closing prices (most recent first).
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> anyhow::Result<Vec<f64>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitKlineResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
    }

    let prices: Vec<f64> = resp
        .result
        .list
        .iter()
        .filter_map(|candle| {
            if candle.len() >= 5 {
                candle[4].parse::<f64>().ok() // Close price
            } else {
                None
            }
        })
        .collect();

    Ok(prices)
}

/// Compute log returns from a price series (assumes most-recent-first ordering).
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }
    // Reverse to chronological order, then compute returns
    let mut chronological: Vec<f64> = prices.to_vec();
    chronological.reverse();

    chronological
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Compute mean and covariance matrix from multiple return series.
pub fn compute_statistics(
    all_returns: &[Vec<f64>],
) -> (Array1<f64>, Array2<f64>) {
    let n_assets = all_returns.len();
    let min_len = all_returns.iter().map(|r| r.len()).min().unwrap_or(0);

    // Trim to same length
    let trimmed: Vec<&[f64]> = all_returns
        .iter()
        .map(|r| &r[..min_len])
        .collect();

    // Mean returns
    let mut means = Array1::<f64>::zeros(n_assets);
    for (i, returns) in trimmed.iter().enumerate() {
        means[i] = returns.iter().sum::<f64>() / min_len as f64;
    }

    // Covariance matrix
    let mut cov = Array2::<f64>::zeros((n_assets, n_assets));
    for i in 0..n_assets {
        for j in 0..n_assets {
            let mut cov_ij = 0.0;
            for t in 0..min_len {
                cov_ij += (trimmed[i][t] - means[i]) * (trimmed[j][t] - means[j]);
            }
            cov[[i, j]] = cov_ij / (min_len as f64 - 1.0);
        }
    }

    (means, cov)
}

/// Run the full VQE portfolio optimization pipeline.
pub fn run_portfolio_optimization(
    returns: &Array1<f64>,
    cov_matrix: &Array2<f64>,
    n_assets: usize,
    bits_per_asset: usize,
    risk_aversion: f64,
    penalty: f64,
    n_layers: usize,
    max_iterations: usize,
) -> VqeResult {
    let n_qubits = n_assets * bits_per_asset;
    let qubo = build_portfolio_qubo(returns, cov_matrix, risk_aversion, penalty, bits_per_asset);
    let hamiltonian = qubo_to_ising(&qubo);

    let config = VqeConfig {
        n_qubits,
        n_layers,
        max_iterations,
        tolerance: 1e-8,
    };

    let mut result = run_vqe(&hamiltonian, &config);
    let raw_weights = decode_weights(result.optimal_bitstring, n_assets, bits_per_asset);
    result.weights = normalize_weights(&raw_weights);
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);

        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 1.0).abs() < 1e-10);

        let prod = a * b;
        // (1+2i)(3-i) = 3 - i + 6i - 2i^2 = 3 + 5i + 2 = 5 + 5i
        assert!((prod.re - 5.0).abs() < 1e-10);
        assert!((prod.im - 5.0).abs() < 1e-10);

        let conj = a.conj();
        assert!((conj.re - 1.0).abs() < 1e-10);
        assert!((conj.im - (-2.0)).abs() < 1e-10);

        assert!((a.norm_sq() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_matrices() {
        let z = pauli_z();
        // Z|0> = |0>, Z|1> = -|1>
        let v0 = z.mul_vec([Complex::one(), Complex::zero()]);
        assert!((v0[0].re - 1.0).abs() < 1e-10);
        assert!(v0[1].norm() < 1e-10);

        let v1 = z.mul_vec([Complex::zero(), Complex::one()]);
        assert!(v1[0].norm() < 1e-10);
        assert!((v1[1].re - (-1.0)).abs() < 1e-10);

        // X|0> = |1>
        let x = pauli_x();
        let xv = x.mul_vec([Complex::one(), Complex::zero()]);
        assert!(xv[0].norm() < 1e-10);
        assert!((xv[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_gates() {
        // Ry(pi)|0> = |1>
        let gate = ry(PI);
        let result = gate.mul_vec([Complex::one(), Complex::zero()]);
        assert!(result[0].norm() < 1e-10);
        assert!((result[1].norm() - 1.0).abs() < 1e-10);

        // Rz(0) = I
        let gate_rz = rz(0.0);
        let result_rz = gate_rz.mul_vec([Complex::one(), Complex::zero()]);
        assert!((result_rz[0].re - 1.0).abs() < 1e-10);
        assert!(result_rz[1].norm() < 1e-10);
    }

    #[test]
    fn test_state_vector_operations() {
        // 2-qubit system: apply X to qubit 0 on |00> -> |01>
        let mut state = init_state(2);
        let x = pauli_x();
        apply_single_qubit_gate(&mut state, 2, 0, &x);
        // |01> = state[1]
        assert!((state[1].norm_sq() - 1.0).abs() < 1e-10);

        // Apply CNOT(0,1) on |01> -> |11> (state[3])
        apply_cnot(&mut state, 2, 0, 1);
        assert!((state[3].norm_sq() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_qubo_to_ising() {
        // Simple 2-variable QUBO
        let mut qubo = Array2::<f64>::zeros((2, 2));
        qubo[[0, 0]] = 1.0;
        qubo[[1, 1]] = 1.0;
        qubo[[0, 1]] = -2.0;

        let ising = qubo_to_ising(&qubo);
        assert_eq!(ising.n_qubits, 2);
        assert_eq!(ising.h.len(), 2);
        assert_eq!(ising.j_couplings.len(), 1);
    }

    #[test]
    fn test_decode_weights() {
        // 2 assets, 2 bits each
        // bitstring = 0b1001 -> asset0 bits: 01 = 1/3, asset1 bits: 10 = 2/3
        let weights = decode_weights(0b1001, 2, 2);
        assert!((weights[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((weights[1] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_metrics() {
        let weights = vec![0.5, 0.5];
        let returns = Array1::from_vec(vec![0.01, 0.02]);
        let mut cov = Array2::<f64>::zeros((2, 2));
        cov[[0, 0]] = 0.04;
        cov[[1, 1]] = 0.09;
        cov[[0, 1]] = 0.01;
        cov[[1, 0]] = 0.01;

        let ret = portfolio_return(&weights, &returns);
        assert!((ret - 0.015).abs() < 1e-10);

        let var = portfolio_variance(&weights, &cov);
        // 0.25*0.04 + 0.25*0.09 + 2*0.25*0.01 = 0.01 + 0.0225 + 0.005 = 0.0375
        assert!((var - 0.0375).abs() < 1e-10);

        let risk = portfolio_risk(&weights, &cov);
        assert!((risk - 0.0375_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_weights() {
        let w = vec![1.0, 2.0, 3.0];
        let norm = normalize_weights(&w);
        let sum: f64 = norm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((norm[0] - 1.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_log_returns() {
        // Most recent first: [110, 100, 90]
        let prices = vec![110.0, 100.0, 90.0];
        let returns = compute_log_returns(&prices);
        // Chronological: [90, 100, 110]
        // Returns: ln(100/90), ln(110/100)
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - (100.0_f64 / 90.0).ln()).abs() < 1e-10);
        assert!((returns[1] - (110.0_f64 / 100.0).ln()).abs() < 1e-10);
    }
}
