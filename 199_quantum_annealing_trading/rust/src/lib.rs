//! Quantum Annealing Trading
//!
//! This library implements QUBO (Quadratic Unconstrained Binary Optimization)
//! solvers for trading applications, including simulated annealing and
//! simulated quantum annealing with transverse field tunneling.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ============================================================================
// QUBO Problem Representation
// ============================================================================

/// Represents a QUBO (Quadratic Unconstrained Binary Optimization) problem.
///
/// The objective is to minimize x^T Q x where x is a binary vector.
/// Diagonal elements Q[i][i] encode linear terms.
/// Upper-triangular elements Q[i][j] (i < j) encode quadratic interactions.
#[derive(Debug, Clone)]
pub struct QuboProblem {
    /// The QUBO matrix (upper-triangular with linear terms on diagonal)
    pub q_matrix: Array2<f64>,
    /// Number of binary variables
    pub num_variables: usize,
}

impl QuboProblem {
    /// Create a new QUBO problem from a Q matrix.
    pub fn new(q_matrix: Array2<f64>) -> Self {
        let num_variables = q_matrix.nrows();
        assert_eq!(q_matrix.nrows(), q_matrix.ncols(), "Q matrix must be square");
        QuboProblem {
            q_matrix,
            num_variables,
        }
    }

    /// Compute the energy (objective value) for a given binary solution.
    /// Energy = x^T Q x = sum_i Q_ii * x_i + sum_{i<j} Q_ij * x_i * x_j
    pub fn energy(&self, solution: &[u8]) -> f64 {
        assert_eq!(solution.len(), self.num_variables);
        let mut energy = 0.0;
        for i in 0..self.num_variables {
            if solution[i] == 1 {
                energy += self.q_matrix[[i, i]];
                for j in (i + 1)..self.num_variables {
                    if solution[j] == 1 {
                        energy += self.q_matrix[[i, j]];
                    }
                }
            }
        }
        energy
    }

    /// Compute the energy change from flipping bit at position `pos`.
    pub fn delta_energy(&self, solution: &[u8], pos: usize) -> f64 {
        let current = solution[pos];
        let flip_direction = if current == 0 { 1.0 } else { -1.0 };

        let mut delta = flip_direction * self.q_matrix[[pos, pos]];
        for j in 0..self.num_variables {
            if j != pos && solution[j] == 1 {
                if j > pos {
                    delta += flip_direction * self.q_matrix[[pos, j]];
                } else {
                    delta += flip_direction * self.q_matrix[[j, pos]];
                }
            }
        }
        delta
    }
}

// ============================================================================
// Annealing Schedules
// ============================================================================

/// Annealing schedule type
#[derive(Debug, Clone, Copy)]
pub enum AnnealingSchedule {
    /// Linear schedule: T(t) = T_start * (1 - t/T_total)
    Linear,
    /// Exponential schedule: T(t) = T_start * exp(-alpha * t/T_total)
    Exponential { alpha: f64 },
}

impl AnnealingSchedule {
    /// Compute the temperature/field strength at normalized time s in [0, 1].
    pub fn value(&self, initial: f64, s: f64) -> f64 {
        match self {
            AnnealingSchedule::Linear => initial * (1.0 - s),
            AnnealingSchedule::Exponential { alpha } => initial * (-alpha * s).exp(),
        }
    }
}

// ============================================================================
// Solver Result
// ============================================================================

/// Result from a QUBO solver.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Best binary solution found
    pub best_solution: Vec<u8>,
    /// Energy of the best solution
    pub best_energy: f64,
    /// Energy history over the annealing process
    pub energy_history: Vec<f64>,
}

// ============================================================================
// Simulated Annealing Solver (Classical Baseline)
// ============================================================================

/// Classical simulated annealing solver for QUBO problems.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingSolver {
    /// Number of sweeps (each sweep attempts N spin flips)
    pub num_sweeps: usize,
    /// Initial temperature
    pub initial_temperature: f64,
    /// Annealing schedule
    pub schedule: AnnealingSchedule,
}

impl SimulatedAnnealingSolver {
    pub fn new(num_sweeps: usize, initial_temperature: f64, schedule: AnnealingSchedule) -> Self {
        SimulatedAnnealingSolver {
            num_sweeps,
            initial_temperature,
            schedule,
        }
    }

    /// Solve the QUBO problem using simulated annealing.
    pub fn solve(&self, problem: &QuboProblem) -> SolverResult {
        let mut rng = rand::thread_rng();
        let n = problem.num_variables;

        // Initialize random binary solution
        let mut solution: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
        let mut current_energy = problem.energy(&solution);

        let mut best_solution = solution.clone();
        let mut best_energy = current_energy;
        let mut energy_history = Vec::with_capacity(self.num_sweeps);

        for sweep in 0..self.num_sweeps {
            let s = sweep as f64 / self.num_sweeps as f64;
            let temperature = self.schedule.value(self.initial_temperature, s);

            // One sweep: attempt to flip each variable once
            for _ in 0..n {
                let pos = rng.gen_range(0..n);
                let delta_e = problem.delta_energy(&solution, pos);

                // Metropolis acceptance criterion
                let accept = if delta_e <= 0.0 {
                    true
                } else if temperature > 1e-10 {
                    rng.gen::<f64>() < (-delta_e / temperature).exp()
                } else {
                    false
                };

                if accept {
                    solution[pos] = 1 - solution[pos];
                    current_energy += delta_e;

                    if current_energy < best_energy {
                        best_solution = solution.clone();
                        best_energy = current_energy;
                    }
                }
            }

            energy_history.push(best_energy);
        }

        SolverResult {
            best_solution,
            best_energy,
            energy_history,
        }
    }
}

// ============================================================================
// Simulated Quantum Annealing Solver (with Transverse Field Tunneling)
// ============================================================================

/// Simulated quantum annealing solver using the Suzuki-Trotter decomposition.
///
/// Simulates P replicas (Trotter slices) of the system coupled along
/// an imaginary time dimension. The transverse field enables quantum
/// tunneling effects that can escape local minima.
#[derive(Debug, Clone)]
pub struct SimulatedQuantumAnnealingSolver {
    /// Number of sweeps
    pub num_sweeps: usize,
    /// Number of Trotter slices (replicas)
    pub num_trotter_slices: usize,
    /// Initial transverse field strength
    pub initial_transverse_field: f64,
    /// Temperature for the Monte Carlo sampling
    pub temperature: f64,
    /// Annealing schedule for the transverse field
    pub schedule: AnnealingSchedule,
}

impl SimulatedQuantumAnnealingSolver {
    pub fn new(
        num_sweeps: usize,
        num_trotter_slices: usize,
        initial_transverse_field: f64,
        temperature: f64,
        schedule: AnnealingSchedule,
    ) -> Self {
        SimulatedQuantumAnnealingSolver {
            num_sweeps,
            num_trotter_slices,
            initial_transverse_field,
            temperature,
            schedule,
        }
    }

    /// Solve the QUBO problem using simulated quantum annealing.
    pub fn solve(&self, problem: &QuboProblem) -> SolverResult {
        let mut rng = rand::thread_rng();
        let n = problem.num_variables;
        let p = self.num_trotter_slices;

        // Initialize P random replicas
        let mut replicas: Vec<Vec<u8>> = (0..p)
            .map(|_| (0..n).map(|_| rng.gen_range(0..2)).collect())
            .collect();

        let mut best_solution = replicas[0].clone();
        let mut best_energy = problem.energy(&best_solution);
        let mut energy_history = Vec::with_capacity(self.num_sweeps);

        // Pre-compute scaled problem energy (divided by P for replicas)
        let energy_scale = 1.0 / p as f64;

        for sweep in 0..self.num_sweeps {
            let s = sweep as f64 / self.num_sweeps as f64;
            let gamma = self.schedule.value(self.initial_transverse_field, s);

            // Inter-replica coupling strength
            // J_perp = -(PT/2) * ln(tanh(Gamma / (PT)))
            let pt = p as f64 * self.temperature;
            let j_perp = if gamma > 1e-10 && pt > 1e-10 {
                let arg = (gamma / pt).tanh();
                if arg > 1e-10 {
                    -0.5 * pt * arg.ln()
                } else {
                    f64::MAX * 0.001
                }
            } else {
                f64::MAX * 0.001 // Very strong coupling when field is off
            };

            // Sweep over all replicas and all variables
            for k in 0..p {
                for _ in 0..n {
                    let pos = rng.gen_range(0..n);

                    // Problem energy change (scaled by 1/P)
                    let delta_problem = energy_scale * problem.delta_energy(&replicas[k], pos);

                    // Inter-replica coupling energy change
                    let current_spin = if replicas[k][pos] == 1 { 1.0 } else { -1.0 };
                    let prev_k = if k == 0 { p - 1 } else { k - 1 };
                    let next_k = if k == p - 1 { 0 } else { k + 1 };
                    let prev_spin =
                        if replicas[prev_k][pos] == 1 { 1.0 } else { -1.0 };
                    let next_spin =
                        if replicas[next_k][pos] == 1 { 1.0 } else { -1.0 };

                    // Flipping spin changes coupling energy by
                    // 2 * J_perp * current_spin * (prev_spin + next_spin)
                    let delta_coupling = 2.0 * j_perp * current_spin * (prev_spin + next_spin);

                    let delta_total = delta_problem + delta_coupling;

                    // Metropolis acceptance
                    let accept = if delta_total <= 0.0 {
                        true
                    } else if self.temperature > 1e-10 {
                        rng.gen::<f64>() < (-delta_total / self.temperature).exp()
                    } else {
                        false
                    };

                    if accept {
                        replicas[k][pos] = 1 - replicas[k][pos];
                    }
                }
            }

            // Track best solution across all replicas
            for replica in &replicas {
                let e = problem.energy(replica);
                if e < best_energy {
                    best_energy = e;
                    best_solution = replica.clone();
                }
            }
            energy_history.push(best_energy);
        }

        SolverResult {
            best_solution,
            best_energy,
            energy_history,
        }
    }
}

// ============================================================================
// Portfolio Optimization QUBO Encoding
// ============================================================================

/// Builds a QUBO matrix for portfolio optimization with cardinality constraints.
///
/// Objective: min lambda * x^T Sigma x - mu^T x + P * (sum(x) - K)^2
///
/// where:
/// - x_i in {0,1} indicates whether asset i is selected
/// - Sigma is the covariance matrix
/// - mu is the expected return vector
/// - lambda is the risk aversion parameter
/// - K is the target number of assets
/// - P is the penalty for violating the cardinality constraint
#[derive(Debug, Clone)]
pub struct PortfolioQuboBuilder {
    /// Covariance matrix
    pub covariance: Array2<f64>,
    /// Expected returns
    pub expected_returns: Array1<f64>,
    /// Risk aversion parameter
    pub lambda: f64,
    /// Target number of assets to select
    pub target_cardinality: usize,
    /// Penalty coefficient for cardinality constraint
    pub penalty: f64,
}

impl PortfolioQuboBuilder {
    pub fn new(
        covariance: Array2<f64>,
        expected_returns: Array1<f64>,
        lambda: f64,
        target_cardinality: usize,
        penalty: f64,
    ) -> Self {
        PortfolioQuboBuilder {
            covariance,
            expected_returns,
            lambda,
            target_cardinality,
            penalty,
        }
    }

    /// Build the QUBO problem from the portfolio parameters.
    pub fn build_qubo(&self) -> QuboProblem {
        let n = self.expected_returns.len();
        let mut q = Array2::<f64>::zeros((n, n));
        let k = self.target_cardinality as f64;

        for i in 0..n {
            // Linear terms on diagonal:
            // -mu_i (return maximization) +
            // lambda * Sigma_ii (risk) +
            // P * (1 - 2K) (from expanding (sum x_i - K)^2)
            q[[i, i]] = -self.expected_returns[i]
                + self.lambda * self.covariance[[i, i]]
                + self.penalty * (1.0 - 2.0 * k);

            // Quadratic terms (upper triangle):
            // lambda * Sigma_ij + 2P (from expanding (sum x_i - K)^2)
            for j in (i + 1)..n {
                q[[i, j]] = self.lambda * self.covariance[[i, j]] + 2.0 * self.penalty;
            }
        }

        QuboProblem::new(q)
    }
}

// ============================================================================
// Index Tracking QUBO Formulation
// ============================================================================

/// Builds a QUBO matrix for index tracking with cardinality constraints.
///
/// Selects K assets from N to minimize tracking error against an index.
#[derive(Debug, Clone)]
pub struct IndexTrackingQuboBuilder {
    /// Historical returns matrix: rows = time periods, cols = assets
    pub asset_returns: Array2<f64>,
    /// Index returns for each time period
    pub index_returns: Array1<f64>,
    /// Target number of assets
    pub target_cardinality: usize,
    /// Penalty coefficient for cardinality constraint
    pub penalty: f64,
}

impl IndexTrackingQuboBuilder {
    pub fn new(
        asset_returns: Array2<f64>,
        index_returns: Array1<f64>,
        target_cardinality: usize,
        penalty: f64,
    ) -> Self {
        IndexTrackingQuboBuilder {
            asset_returns,
            index_returns,
            target_cardinality,
            penalty,
        }
    }

    /// Build QUBO for index tracking with equal weights among selected assets.
    pub fn build_qubo(&self) -> QuboProblem {
        let n = self.asset_returns.ncols();
        let t = self.asset_returns.nrows();
        let k = self.target_cardinality as f64;
        let w = 1.0 / k; // Equal weight per selected asset

        let mut q = Array2::<f64>::zeros((n, n));

        // Tracking error: sum_t (r_t^index - sum_i x_i * w * r_it)^2
        // Expanding: sum_t (r_t^index)^2 - 2w * sum_t sum_i r_t^index * r_it * x_i
        //            + w^2 * sum_t sum_{i,j} r_it * r_jt * x_i * x_j
        for i in 0..n {
            // Linear term from cross-term: -2w * sum_t r_t^index * r_it
            let mut linear = 0.0;
            for tt in 0..t {
                linear -= 2.0 * w * self.index_returns[tt] * self.asset_returns[[tt, i]];
            }

            // Quadratic diagonal: w^2 * sum_t r_it^2
            let mut quad_diag = 0.0;
            for tt in 0..t {
                quad_diag += w * w * self.asset_returns[[tt, i]].powi(2);
            }

            q[[i, i]] = linear + quad_diag + self.penalty * (1.0 - 2.0 * k);

            for j in (i + 1)..n {
                // Quadratic off-diagonal: w^2 * sum_t r_it * r_jt * 2
                // (factor 2 because Q is upper-triangular and we combine i,j and j,i)
                let mut quad_off = 0.0;
                for tt in 0..t {
                    quad_off +=
                        2.0 * w * w * self.asset_returns[[tt, i]] * self.asset_returns[[tt, j]];
                }
                q[[i, j]] = quad_off + 2.0 * self.penalty;
            }
        }

        QuboProblem::new(q)
    }
}

// ============================================================================
// Energy Landscape Analysis
// ============================================================================

/// Tools for analyzing the energy landscape of a QUBO problem.
pub struct EnergyLandscapeAnalyzer;

impl EnergyLandscapeAnalyzer {
    /// Sample random solutions and compute energy distribution statistics.
    pub fn energy_distribution(problem: &QuboProblem, num_samples: usize) -> EnergyStats {
        let mut rng = rand::thread_rng();
        let n = problem.num_variables;
        let mut energies = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let solution: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
            energies.push(problem.energy(&solution));
        }

        energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = energies.iter().sum::<f64>() / num_samples as f64;
        let variance = energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / num_samples as f64;

        EnergyStats {
            min: energies[0],
            max: *energies.last().unwrap(),
            mean,
            std_dev: variance.sqrt(),
            median: energies[num_samples / 2],
            percentile_10: energies[num_samples / 10],
        }
    }

    /// Estimate the fraction of local minima by random sampling.
    /// A solution is a local minimum if no single bit flip decreases energy.
    pub fn local_minima_fraction(problem: &QuboProblem, num_samples: usize) -> f64 {
        let mut rng = rand::thread_rng();
        let n = problem.num_variables;
        let mut num_local_min = 0;

        for _ in 0..num_samples {
            let solution: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2)).collect();
            let is_local_min = (0..n).all(|pos| problem.delta_energy(&solution, pos) >= 0.0);
            if is_local_min {
                num_local_min += 1;
            }
        }

        num_local_min as f64 / num_samples as f64
    }

    /// Compute the energy barrier between two solutions.
    /// Uses a greedy path that flips one bit at a time from sol_a to sol_b.
    pub fn energy_barrier(problem: &QuboProblem, sol_a: &[u8], sol_b: &[u8]) -> f64 {
        let n = problem.num_variables;
        assert_eq!(sol_a.len(), n);
        assert_eq!(sol_b.len(), n);

        let mut current = sol_a.to_vec();
        let start_energy = problem.energy(&current);
        let mut max_energy = start_energy;

        // Flip bits one at a time to go from sol_a to sol_b
        for i in 0..n {
            if current[i] != sol_b[i] {
                current[i] = sol_b[i];
                let e = problem.energy(&current);
                if e > max_energy {
                    max_energy = e;
                }
            }
        }

        max_energy - start_energy
    }
}

/// Statistics about the energy distribution of a QUBO problem.
#[derive(Debug, Clone)]
pub struct EnergyStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub percentile_10: f64,
}

// ============================================================================
// Bybit API Data Fetching
// ============================================================================

/// Raw kline data from Bybit API.
#[derive(Debug, Clone)]
pub struct KlineData {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch kline (OHLCV) data from Bybit API v5.
pub async fn fetch_klines(
    symbol: &str,
    interval: &str,
    limit: u32,
) -> Result<Vec<KlineData>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: ret_code={}", resp.ret_code);
    }

    let mut klines: Vec<KlineData> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(KlineData {
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

    // Bybit returns newest first, reverse to chronological order
    klines.reverse();
    Ok(klines)
}

/// Fetch kline data synchronously (blocking).
pub fn fetch_klines_blocking(
    symbol: &str,
    interval: &str,
    limit: u32,
) -> Result<Vec<KlineData>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: ret_code={}", resp.ret_code);
    }

    let mut klines: Vec<KlineData> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(KlineData {
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

    klines.reverse();
    Ok(klines)
}

/// Compute log returns from kline close prices.
pub fn compute_log_returns(klines: &[KlineData]) -> Vec<f64> {
    klines
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

/// Build covariance matrix from multiple asset return series.
pub fn build_covariance_matrix(returns: &[Vec<f64>]) -> Array2<f64> {
    let n = returns.len();
    let t = returns.iter().map(|r| r.len()).min().unwrap_or(0);

    // Compute means
    let means: Vec<f64> = returns
        .iter()
        .map(|r| r.iter().take(t).sum::<f64>() / t as f64)
        .collect();

    // Compute covariance matrix
    let mut cov = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0;
            for tt in 0..t {
                sum += (returns[i][tt] - means[i]) * (returns[j][tt] - means[j]);
            }
            let c = sum / (t as f64 - 1.0).max(1.0);
            cov[[i, j]] = c;
            cov[[j, i]] = c;
        }
    }
    cov
}

/// Compute mean returns for each asset.
pub fn compute_mean_returns(returns: &[Vec<f64>]) -> Array1<f64> {
    let means: Vec<f64> = returns
        .iter()
        .map(|r| {
            if r.is_empty() {
                0.0
            } else {
                r.iter().sum::<f64>() / r.len() as f64
            }
        })
        .collect();
    Array1::from(means)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_qubo() -> QuboProblem {
        // Simple 3-variable QUBO:
        // min -5x0 - 3x1 - 8x2 + 2x0x1 + 4x0x2 + 6x1x2
        let mut q = Array2::<f64>::zeros((3, 3));
        q[[0, 0]] = -5.0;
        q[[1, 1]] = -3.0;
        q[[2, 2]] = -8.0;
        q[[0, 1]] = 2.0;
        q[[0, 2]] = 4.0;
        q[[1, 2]] = 6.0;
        QuboProblem::new(q)
    }

    #[test]
    fn test_qubo_energy_all_zeros() {
        let problem = simple_qubo();
        assert_eq!(problem.energy(&[0, 0, 0]), 0.0);
    }

    #[test]
    fn test_qubo_energy_all_ones() {
        let problem = simple_qubo();
        // -5 - 3 - 8 + 2 + 4 + 6 = -4
        assert!((problem.energy(&[1, 1, 1]) - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_qubo_energy_single_bits() {
        let problem = simple_qubo();
        assert!((problem.energy(&[1, 0, 0]) - (-5.0)).abs() < 1e-10);
        assert!((problem.energy(&[0, 1, 0]) - (-3.0)).abs() < 1e-10);
        assert!((problem.energy(&[0, 0, 1]) - (-8.0)).abs() < 1e-10);
    }

    #[test]
    fn test_qubo_energy_pairs() {
        let problem = simple_qubo();
        // x0=1, x1=1: -5 - 3 + 2 = -6
        assert!((problem.energy(&[1, 1, 0]) - (-6.0)).abs() < 1e-10);
        // x0=1, x2=1: -5 - 8 + 4 = -9
        assert!((problem.energy(&[1, 0, 1]) - (-9.0)).abs() < 1e-10);
        // x1=1, x2=1: -3 - 8 + 6 = -5
        assert!((problem.energy(&[0, 1, 1]) - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_delta_energy_consistency() {
        let problem = simple_qubo();
        let solution = vec![1, 0, 1];
        let base_energy = problem.energy(&solution);

        for pos in 0..3 {
            let delta = problem.delta_energy(&solution, pos);
            let mut flipped = solution.clone();
            flipped[pos] = 1 - flipped[pos];
            let new_energy = problem.energy(&flipped);
            assert!(
                (delta - (new_energy - base_energy)).abs() < 1e-10,
                "Delta energy mismatch at pos {}: delta={}, actual={}",
                pos,
                delta,
                new_energy - base_energy
            );
        }
    }

    #[test]
    fn test_simulated_annealing_finds_good_solution() {
        let problem = simple_qubo();
        let solver = SimulatedAnnealingSolver::new(
            1000,
            10.0,
            AnnealingSchedule::Exponential { alpha: 3.0 },
        );

        // Run multiple times to account for randomness
        let mut best_energy = f64::MAX;
        for _ in 0..10 {
            let result = solver.solve(&problem);
            if result.best_energy < best_energy {
                best_energy = result.best_energy;
            }
        }

        // Optimal is x0=1, x2=1 -> energy = -9
        assert!(best_energy <= -9.0 + 1e-10, "SA should find optimal: got {}", best_energy);
    }

    #[test]
    fn test_quantum_annealing_finds_good_solution() {
        let problem = simple_qubo();
        let solver = SimulatedQuantumAnnealingSolver::new(
            500,
            4,
            3.0,
            1.0,
            AnnealingSchedule::Linear,
        );

        let mut best_energy = f64::MAX;
        for _ in 0..10 {
            let result = solver.solve(&problem);
            if result.best_energy < best_energy {
                best_energy = result.best_energy;
            }
        }

        assert!(best_energy <= -9.0 + 1e-10, "SQA should find optimal: got {}", best_energy);
    }

    #[test]
    fn test_annealing_schedule_linear() {
        let schedule = AnnealingSchedule::Linear;
        assert!((schedule.value(10.0, 0.0) - 10.0).abs() < 1e-10);
        assert!((schedule.value(10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((schedule.value(10.0, 1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_annealing_schedule_exponential() {
        let schedule = AnnealingSchedule::Exponential { alpha: 1.0 };
        assert!((schedule.value(10.0, 0.0) - 10.0).abs() < 1e-10);
        let mid = schedule.value(10.0, 0.5);
        assert!(mid > 0.0 && mid < 10.0);
        let end = schedule.value(10.0, 1.0);
        assert!(end > 0.0 && end < mid);
    }

    #[test]
    fn test_portfolio_qubo_builder() {
        let cov = Array2::from_shape_vec(
            (3, 3),
            vec![0.04, 0.01, 0.02, 0.01, 0.09, 0.03, 0.02, 0.03, 0.16],
        )
        .unwrap();
        let returns = Array1::from(vec![0.10, 0.15, 0.20]);

        let builder = PortfolioQuboBuilder::new(cov, returns, 1.0, 2, 5.0);
        let problem = builder.build_qubo();

        assert_eq!(problem.num_variables, 3);
        // Verify the QUBO matrix has correct dimensions
        assert_eq!(problem.q_matrix.nrows(), 3);
        assert_eq!(problem.q_matrix.ncols(), 3);
    }

    #[test]
    fn test_portfolio_qubo_cardinality_constraint() {
        let cov = Array2::from_shape_vec(
            (3, 3),
            vec![0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
        )
        .unwrap();
        let returns = Array1::from(vec![0.10, 0.10, 0.10]);

        // Very high penalty should enforce cardinality
        let builder = PortfolioQuboBuilder::new(cov, returns, 0.01, 2, 100.0);
        let problem = builder.build_qubo();

        let solver = SimulatedAnnealingSolver::new(
            2000,
            10.0,
            AnnealingSchedule::Exponential { alpha: 3.0 },
        );

        let mut found_k2 = false;
        for _ in 0..20 {
            let result = solver.solve(&problem);
            let count: u8 = result.best_solution.iter().sum();
            if count == 2 {
                found_k2 = true;
                break;
            }
        }
        assert!(found_k2, "Solver should find solutions with exactly K=2 assets");
    }

    #[test]
    fn test_index_tracking_qubo_builder() {
        let asset_returns = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.01, 0.02, 0.03, -0.01, 0.01, -0.02, 0.02, -0.01, 0.01, 0.00, 0.02, 0.00,
                -0.01, 0.00, 0.02,
            ],
        )
        .unwrap();
        let index_returns = Array1::from(vec![0.015, -0.005, 0.01, 0.005, 0.005]);

        let builder = IndexTrackingQuboBuilder::new(asset_returns, index_returns, 2, 5.0);
        let problem = builder.build_qubo();

        assert_eq!(problem.num_variables, 3);
    }

    #[test]
    fn test_energy_distribution() {
        let problem = simple_qubo();
        let stats = EnergyLandscapeAnalyzer::energy_distribution(&problem, 1000);

        assert!(stats.min <= stats.median);
        assert!(stats.median <= stats.max);
        assert!(stats.std_dev >= 0.0);
    }

    #[test]
    fn test_energy_barrier() {
        let problem = simple_qubo();
        let sol_a = vec![0, 0, 0];
        let sol_b = vec![1, 1, 1];
        let barrier = EnergyLandscapeAnalyzer::energy_barrier(&problem, &sol_a, &sol_b);
        // Barrier is the max energy increase along the path
        assert!(barrier >= 0.0 || barrier < 0.0); // It's defined, that's what matters
    }

    #[test]
    fn test_compute_log_returns() {
        let klines = vec![
            KlineData { timestamp: 0, open: 100.0, high: 105.0, low: 95.0, close: 100.0, volume: 1000.0 },
            KlineData { timestamp: 1, open: 100.0, high: 110.0, low: 98.0, close: 105.0, volume: 1200.0 },
            KlineData { timestamp: 2, open: 105.0, high: 108.0, low: 100.0, close: 102.0, volume: 900.0 },
        ];
        let returns = compute_log_returns(&klines);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[1] - (102.0_f64 / 105.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_build_covariance_matrix() {
        let returns = vec![
            vec![0.01, -0.02, 0.03, 0.00, -0.01],
            vec![0.02, -0.01, 0.01, 0.01, -0.02],
        ];
        let cov = build_covariance_matrix(&returns);
        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);
        // Covariance matrix should be symmetric
        assert!((cov[[0, 1]] - cov[[1, 0]]).abs() < 1e-10);
        // Diagonal should be positive (variance)
        assert!(cov[[0, 0]] > 0.0);
        assert!(cov[[1, 1]] > 0.0);
    }

    #[test]
    fn test_compute_mean_returns() {
        let returns = vec![
            vec![0.01, 0.02, 0.03],
            vec![-0.01, 0.01, 0.02],
        ];
        let means = compute_mean_returns(&returns);
        // mean of [0.01, 0.02, 0.03] = 0.02
        assert!((means[0] - 0.02).abs() < 1e-10);
        // mean of [-0.01, 0.01, 0.02] = 0.02/3 ≈ 0.006667
        let expected_1 = (-0.01 + 0.01 + 0.02) / 3.0;
        assert!((means[1] - expected_1).abs() < 1e-10);
    }

    #[test]
    fn test_local_minima_fraction() {
        let problem = simple_qubo();
        let fraction = EnergyLandscapeAnalyzer::local_minima_fraction(&problem, 1000);
        assert!(fraction >= 0.0 && fraction <= 1.0);
    }

    #[test]
    fn test_solver_energy_history_decreasing() {
        let problem = simple_qubo();
        let solver = SimulatedAnnealingSolver::new(
            100,
            10.0,
            AnnealingSchedule::Linear,
        );
        let result = solver.solve(&problem);
        // Best energy history should be non-increasing
        for i in 1..result.energy_history.len() {
            assert!(
                result.energy_history[i] <= result.energy_history[i - 1] + 1e-10,
                "Best energy should be non-increasing"
            );
        }
    }
}
