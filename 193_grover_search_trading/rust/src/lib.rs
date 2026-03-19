//! # Grover Search Trading
//!
//! A quantum-inspired Grover search simulator applied to trading problems.
//! Implements amplitude amplification for finding optimal portfolio configurations
//! from a combinatorial search space, with Bybit market data integration.

use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Complex number type
// ---------------------------------------------------------------------------

/// Minimal complex number for statevector simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
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

    /// Squared magnitude |z|^2 = re^2 + im^2.
    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Negate both components (phase flip by pi).
    pub fn negate(&self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
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

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.6}+{:.6}i", self.re, self.im)
        } else {
            write!(f, "{:.6}{:.6}i", self.re, self.im)
        }
    }
}

// ---------------------------------------------------------------------------
// Grover search engine
// ---------------------------------------------------------------------------

/// Core Grover search simulator operating on a full statevector.
pub struct GroverSearch {
    /// Number of qubits (search space = 2^num_qubits).
    pub num_qubits: usize,
    /// Full statevector of dimension 2^num_qubits.
    pub statevector: Vec<Complex>,
    /// Indices of marked (solution) states.
    pub marked_states: Vec<usize>,
}

impl GroverSearch {
    /// Create a new Grover search instance.
    ///
    /// * `num_qubits` - number of qubits (search space size = 2^num_qubits)
    /// * `marked_states` - indices of states that the oracle marks as solutions
    pub fn new(num_qubits: usize, marked_states: Vec<usize>) -> Self {
        let dim = 1 << num_qubits;
        let statevector = vec![Complex::zero(); dim];
        Self {
            num_qubits,
            statevector,
            marked_states,
        }
    }

    /// Dimension of the Hilbert space (2^n).
    pub fn dimension(&self) -> usize {
        1 << self.num_qubits
    }

    // -- Quantum operations ------------------------------------------------

    /// Apply Hadamard transform: set all amplitudes to 1/sqrt(N).
    /// This creates a uniform superposition from the |0...0> state.
    pub fn apply_hadamard(&mut self) {
        let n = self.dimension();
        let amp = 1.0 / (n as f64).sqrt();
        for state in self.statevector.iter_mut() {
            *state = Complex::new(amp, 0.0);
        }
    }

    /// Apply the oracle operator: negate the amplitude of every marked state.
    ///
    /// O_f |x> = (-1)^{f(x)} |x>
    pub fn apply_oracle(&mut self) {
        for &idx in &self.marked_states {
            if idx < self.statevector.len() {
                self.statevector[idx] = self.statevector[idx].negate();
            }
        }
    }

    /// Apply the diffusion (Grover diffuser) operator.
    ///
    /// D = 2|s><s| - I   (inversion about the mean)
    pub fn apply_diffusion(&mut self) {
        let n = self.dimension();
        // Compute mean amplitude (real part only for real-valued superpositions,
        // but we handle complex for generality).
        let mean_re: f64 = self.statevector.iter().map(|c| c.re).sum::<f64>() / n as f64;
        let mean_im: f64 = self.statevector.iter().map(|c| c.im).sum::<f64>() / n as f64;

        for state in self.statevector.iter_mut() {
            state.re = 2.0 * mean_re - state.re;
            state.im = 2.0 * mean_im - state.im;
        }
    }

    /// Perform one Grover iteration: oracle followed by diffusion.
    pub fn grover_iteration(&mut self) {
        self.apply_oracle();
        self.apply_diffusion();
    }

    /// Run the full Grover search with the optimal number of iterations.
    /// Returns the measurement probabilities for all states.
    pub fn run(&mut self) -> Vec<f64> {
        let iters = optimal_iterations(self.dimension(), self.marked_states.len());
        self.run_with_iterations(iters)
    }

    /// Run Grover search with a specified number of iterations.
    pub fn run_with_iterations(&mut self, iterations: usize) -> Vec<f64> {
        // Step 1: uniform superposition
        self.apply_hadamard();

        // Step 2: repeated Grover iterations
        for _ in 0..iterations {
            self.grover_iteration();
        }

        // Step 3: extract measurement probabilities
        self.get_probabilities()
    }

    // -- Measurement -------------------------------------------------------

    /// Extract measurement probabilities from the statevector.
    pub fn get_probabilities(&self) -> Vec<f64> {
        self.statevector.iter().map(|c| c.norm_sq()).collect()
    }

    /// Return the index of the state with highest measurement probability.
    pub fn measure_top(&self) -> (usize, f64) {
        let probs = self.get_probabilities();
        let (idx, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        (idx, prob)
    }

    /// Return the top-k states by probability.
    pub fn measure_top_k(&self, k: usize) -> Vec<(usize, f64)> {
        let mut probs: Vec<(usize, f64)> = self.get_probabilities().into_iter().enumerate().collect();
        probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        probs.truncate(k);
        probs
    }
}

// ---------------------------------------------------------------------------
// Optimal iteration count
// ---------------------------------------------------------------------------

/// Compute the optimal number of Grover iterations.
///
/// k_opt = round( (pi/4) * sqrt(N/M) - 1/2 )
///
/// where N = total_states and M = num_marked.
pub fn optimal_iterations(total_states: usize, num_marked: usize) -> usize {
    if num_marked == 0 || num_marked >= total_states {
        return 0;
    }
    let ratio = num_marked as f64 / total_states as f64;
    let theta = ratio.sqrt().asin();
    let k = (PI / (4.0 * theta) - 0.5).round() as isize;
    k.max(1) as usize
}

// ---------------------------------------------------------------------------
// Trading oracle
// ---------------------------------------------------------------------------

/// Asset data fetched from exchange.
#[derive(Debug, Clone)]
pub struct AssetData {
    pub symbol: String,
    pub last_price: f64,
    pub price_change_pct: f64,
    pub high_24h: f64,
    pub low_24h: f64,
    pub volume_24h: f64,
}

/// Trading criteria for portfolio selection.
#[derive(Debug, Clone)]
pub struct TradingCriteria {
    /// Minimum acceptable portfolio return (e.g. 0.01 = 1%).
    pub min_return: f64,
    /// Maximum acceptable portfolio risk / volatility.
    pub max_risk: f64,
}

/// Evaluate a portfolio subset and determine if it meets trading criteria.
///
/// * `mask` - bitmask where bit i = 1 means asset i is included
/// * `returns` - expected return for each asset
/// * `covariance` - covariance matrix (n x n)
/// * `criteria` - return/risk thresholds
pub fn evaluate_portfolio(
    mask: usize,
    returns: &[f64],
    covariance: &[Vec<f64>],
    criteria: &TradingCriteria,
) -> bool {
    let n = returns.len();
    let selected: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();

    if selected.is_empty() {
        return false;
    }

    let count = selected.len() as f64;
    let weight = 1.0 / count;

    // Equal-weighted portfolio return
    let portfolio_return: f64 = selected.iter().map(|&i| returns[i] * weight).sum();

    // Equal-weighted portfolio variance: w^T * Sigma * w
    let mut portfolio_variance = 0.0;
    for &i in &selected {
        for &j in &selected {
            portfolio_variance += weight * weight * covariance[i][j];
        }
    }
    let portfolio_risk = portfolio_variance.sqrt();

    portfolio_return >= criteria.min_return && portfolio_risk <= criteria.max_risk
}

/// Build a trading oracle by evaluating all 2^n portfolio subsets.
///
/// Returns the list of state indices (bitmasks) that satisfy the criteria.
pub fn build_trading_oracle(
    returns: &[f64],
    covariance: &[Vec<f64>],
    criteria: &TradingCriteria,
) -> Vec<usize> {
    let n = returns.len();
    let total = 1usize << n;
    let mut marked = Vec::new();

    for mask in 0..total {
        if evaluate_portfolio(mask, returns, covariance, criteria) {
            marked.push(mask);
        }
    }

    marked
}

/// Decode a portfolio bitmask into asset names.
pub fn decode_portfolio(mask: usize, symbols: &[String]) -> Vec<String> {
    symbols
        .iter()
        .enumerate()
        .filter(|(i, _)| mask & (1 << i) != 0)
        .map(|(_, s)| s.clone())
        .collect()
}

/// Build a simple covariance matrix from asset data.
///
/// Uses a diagonal model: variance estimated from 24h high-low range.
/// Off-diagonal elements set to a small positive correlation.
pub fn build_covariance_matrix(assets: &[AssetData]) -> Vec<Vec<f64>> {
    let n = assets.len();
    let mut cov = vec![vec![0.0; n]; n];

    for i in 0..n {
        // Estimate volatility from 24h range (Parkinson estimator, simplified)
        let range = assets[i].high_24h - assets[i].low_24h;
        let mid = (assets[i].high_24h + assets[i].low_24h) / 2.0;
        let volatility = if mid > 0.0 { range / mid } else { 0.01 };
        let variance = volatility * volatility;

        cov[i][i] = variance;

        // Off-diagonal: assume mild positive correlation (0.3) among crypto assets
        for j in 0..n {
            if i != j {
                let range_j = assets[j].high_24h - assets[j].low_24h;
                let mid_j = (assets[j].high_24h + assets[j].low_24h) / 2.0;
                let vol_j = if mid_j > 0.0 { range_j / mid_j } else { 0.01 };
                cov[i][j] = 0.3 * volatility * vol_j;
            }
        }
    }

    cov
}

// ---------------------------------------------------------------------------
// Bybit API integration
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BybitResponse {
    #[serde(rename = "retCode")]
    ret_code: i32,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<BybitTicker>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct BybitTicker {
    symbol: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
    #[serde(rename = "highPrice24h")]
    high_price_24h: String,
    #[serde(rename = "lowPrice24h")]
    low_price_24h: String,
    #[serde(rename = "volume24h")]
    volume_24h: String,
    #[serde(rename = "price24hPcnt")]
    price_24h_pcnt: String,
}

/// Fetch ticker data from Bybit for the given symbols.
///
/// Uses the public `/v5/market/tickers` endpoint (no authentication required).
pub async fn fetch_bybit_tickers(symbols: &[&str]) -> Result<Vec<AssetData>> {
    let client = reqwest::Client::new();
    let resp = client
        .get("https://api.bybit.com/v5/market/tickers")
        .query(&[("category", "spot")])
        .send()
        .await
        .map_err(|e| anyhow!("Failed to fetch Bybit tickers: {}", e))?;

    let body: BybitResponse = resp
        .json()
        .await
        .map_err(|e| anyhow!("Failed to parse Bybit response: {}", e))?;

    if body.ret_code != 0 {
        return Err(anyhow!("Bybit API error code: {}", body.ret_code));
    }

    let symbol_set: std::collections::HashSet<&str> = symbols.iter().copied().collect();

    let assets: Vec<AssetData> = body
        .result
        .list
        .into_iter()
        .filter(|t| symbol_set.contains(t.symbol.as_str()))
        .filter_map(|t| {
            Some(AssetData {
                symbol: t.symbol.clone(),
                last_price: t.last_price.parse().ok()?,
                price_change_pct: t.price_24h_pcnt.parse().ok()?,
                high_24h: t.high_price_24h.parse().ok()?,
                low_24h: t.low_price_24h.parse().ok()?,
                volume_24h: t.volume_24h.parse().ok()?,
            })
        })
        .collect();

    // Preserve requested order
    let mut ordered = Vec::new();
    for &sym in symbols {
        if let Some(a) = assets.iter().find(|a| a.symbol == sym) {
            ordered.push(a.clone());
        }
    }

    Ok(ordered)
}

/// Blocking version of fetch_bybit_tickers for non-async contexts.
pub fn fetch_bybit_tickers_blocking(symbols: &[&str]) -> Result<Vec<AssetData>> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get("https://api.bybit.com/v5/market/tickers")
        .query(&[("category", "spot")])
        .send()
        .map_err(|e| anyhow!("Failed to fetch Bybit tickers: {}", e))?;

    let body: BybitResponse = resp
        .json()
        .map_err(|e| anyhow!("Failed to parse Bybit response: {}", e))?;

    if body.ret_code != 0 {
        return Err(anyhow!("Bybit API error code: {}", body.ret_code));
    }

    let symbol_set: std::collections::HashSet<&str> = symbols.iter().copied().collect();

    let assets: Vec<AssetData> = body
        .result
        .list
        .into_iter()
        .filter(|t| symbol_set.contains(t.symbol.as_str()))
        .filter_map(|t| {
            Some(AssetData {
                symbol: t.symbol.clone(),
                last_price: t.last_price.parse().ok()?,
                price_change_pct: t.price_24h_pcnt.parse().ok()?,
                high_24h: t.high_price_24h.parse().ok()?,
                low_24h: t.low_price_24h.parse().ok()?,
                volume_24h: t.volume_24h.parse().ok()?,
            })
        })
        .collect();

    let mut ordered = Vec::new();
    for &sym in symbols {
        if let Some(a) = assets.iter().find(|a| a.symbol == sym) {
            ordered.push(a.clone());
        }
    }

    Ok(ordered)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(3.0, 4.0);
        assert!((a.norm_sq() - 25.0).abs() < 1e-10);

        let b = a.negate();
        assert!((b.re - (-3.0)).abs() < 1e-10);
        assert!((b.im - (-4.0)).abs() < 1e-10);

        let c = Complex::new(1.0, 2.0);
        let d = Complex::new(3.0, 4.0);
        let sum = c + d;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        let prod = c * d;
        // (1+2i)(3+4i) = 3+4i+6i+8i^2 = 3+10i-8 = -5+10i
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_creates_uniform_superposition() {
        let mut gs = GroverSearch::new(3, vec![0]);
        gs.apply_hadamard();

        let n = gs.dimension(); // 8
        let expected_amp = 1.0 / (n as f64).sqrt();

        for state in &gs.statevector {
            assert!((state.re - expected_amp).abs() < 1e-10);
            assert!(state.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_oracle_flips_marked_states() {
        let mut gs = GroverSearch::new(3, vec![2, 5]);
        gs.apply_hadamard();

        let amp_before = gs.statevector[2].re;
        gs.apply_oracle();

        assert!((gs.statevector[2].re - (-amp_before)).abs() < 1e-10);
        assert!((gs.statevector[5].re - (-amp_before)).abs() < 1e-10);
        // Unmarked states unchanged
        assert!((gs.statevector[0].re - amp_before).abs() < 1e-10);
    }

    #[test]
    fn test_diffusion_inverts_about_mean() {
        let mut gs = GroverSearch::new(2, vec![1]);
        gs.apply_hadamard();
        gs.apply_oracle();
        gs.apply_diffusion();

        // After one Grover iteration on 4 states with 1 marked,
        // the marked state should have higher amplitude.
        let probs = gs.get_probabilities();
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_optimal_iterations_single_marked() {
        // N=1024, M=1 => k ≈ (pi/4)*sqrt(1024) ≈ 25.1 => 25
        let k = optimal_iterations(1024, 1);
        assert!(k >= 24 && k <= 26, "Expected ~25, got {}", k);
    }

    #[test]
    fn test_optimal_iterations_multiple_marked() {
        // N=256, M=4 => k ≈ (pi/4)*sqrt(64) ≈ 6.3 => 6
        let k = optimal_iterations(256, 4);
        assert!(k >= 5 && k <= 7, "Expected ~6, got {}", k);
    }

    #[test]
    fn test_optimal_iterations_edge_cases() {
        assert_eq!(optimal_iterations(16, 0), 0);
        assert_eq!(optimal_iterations(16, 16), 0);
    }

    #[test]
    fn test_grover_finds_single_marked_state() {
        // 4 qubits = 16 states, mark state 7
        let mut gs = GroverSearch::new(4, vec![7]);
        let probs = gs.run();

        // State 7 should have the highest probability
        let (top_idx, top_prob) = gs.measure_top();
        assert_eq!(top_idx, 7);
        assert!(top_prob > 0.9, "Expected high prob, got {}", top_prob);

        // Verify probabilities sum to ~1
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_grover_finds_multiple_marked_states() {
        // 3 qubits = 8 states, mark states 1, 4, 6
        let marked = vec![1, 4, 6];
        let mut gs = GroverSearch::new(3, marked.clone());
        let probs = gs.run();

        // Marked states should have higher probability than unmarked
        let marked_prob: f64 = marked.iter().map(|&i| probs[i]).sum();
        assert!(
            marked_prob > 0.8,
            "Expected marked states to dominate, total prob = {}",
            marked_prob
        );
    }

    #[test]
    fn test_measure_top_k() {
        let mut gs = GroverSearch::new(3, vec![2, 5]);
        gs.run();

        let top2 = gs.measure_top_k(2);
        assert_eq!(top2.len(), 2);
        let top_indices: Vec<usize> = top2.iter().map(|(i, _)| *i).collect();
        assert!(top_indices.contains(&2));
        assert!(top_indices.contains(&5));
    }

    #[test]
    fn test_evaluate_portfolio() {
        let returns = vec![0.05, -0.02, 0.03, 0.04];
        let cov = vec![
            vec![0.01, 0.002, 0.001, 0.001],
            vec![0.002, 0.02, 0.001, 0.001],
            vec![0.001, 0.001, 0.015, 0.001],
            vec![0.001, 0.001, 0.001, 0.012],
        ];
        let criteria = TradingCriteria {
            min_return: 0.03,
            max_risk: 0.15,
        };

        // Asset 0 alone: return=0.05, risk=sqrt(0.01)=0.1 -> passes
        assert!(evaluate_portfolio(0b0001, &returns, &cov, &criteria));

        // Asset 1 alone: return=-0.02 -> fails min_return
        assert!(!evaluate_portfolio(0b0010, &returns, &cov, &criteria));

        // Empty portfolio -> fails
        assert!(!evaluate_portfolio(0b0000, &returns, &cov, &criteria));
    }

    #[test]
    fn test_build_trading_oracle() {
        let returns = vec![0.05, -0.02, 0.08];
        let cov = vec![
            vec![0.01, 0.002, 0.001],
            vec![0.002, 0.02, 0.001],
            vec![0.001, 0.001, 0.015],
        ];
        let criteria = TradingCriteria {
            min_return: 0.03,
            max_risk: 0.20,
        };

        let marked = build_trading_oracle(&returns, &cov, &criteria);
        // At least asset 0 alone (return=0.05) and asset 2 alone (return=0.08)
        // should be marked
        assert!(marked.contains(&0b001)); // asset 0
        assert!(marked.contains(&0b100)); // asset 2
        // Asset 1 alone has negative return, should not be marked
        assert!(!marked.contains(&0b010));
    }

    #[test]
    fn test_decode_portfolio() {
        let symbols = vec![
            "BTCUSDT".to_string(),
            "ETHUSDT".to_string(),
            "SOLUSDT".to_string(),
        ];
        let result = decode_portfolio(0b101, &symbols);
        assert_eq!(result, vec!["BTCUSDT", "SOLUSDT"]);
    }

    #[test]
    fn test_build_covariance_matrix() {
        let assets = vec![
            AssetData {
                symbol: "BTCUSDT".to_string(),
                last_price: 50000.0,
                price_change_pct: 0.02,
                high_24h: 51000.0,
                low_24h: 49000.0,
                volume_24h: 1000.0,
            },
            AssetData {
                symbol: "ETHUSDT".to_string(),
                last_price: 3000.0,
                price_change_pct: 0.03,
                high_24h: 3100.0,
                low_24h: 2900.0,
                volume_24h: 500.0,
            },
        ];

        let cov = build_covariance_matrix(&assets);
        assert_eq!(cov.len(), 2);
        assert_eq!(cov[0].len(), 2);
        // Diagonal should be positive
        assert!(cov[0][0] > 0.0);
        assert!(cov[1][1] > 0.0);
        // Off-diagonal should be positive (positive correlation)
        assert!(cov[0][1] > 0.0);
        assert!(cov[1][0] > 0.0);
        // Symmetric
        assert!((cov[0][1] - cov[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_grover_with_trading_oracle() {
        // Simulate a small trading scenario: 4 assets
        let returns = vec![0.05, -0.01, 0.07, 0.03];
        let cov = vec![
            vec![0.010, 0.002, 0.001, 0.001],
            vec![0.002, 0.020, 0.003, 0.001],
            vec![0.001, 0.003, 0.015, 0.002],
            vec![0.001, 0.001, 0.002, 0.012],
        ];
        let criteria = TradingCriteria {
            min_return: 0.04,
            max_risk: 0.20,
        };

        // Build oracle: find qualifying portfolios
        let marked = build_trading_oracle(&returns, &cov, &criteria);
        assert!(!marked.is_empty(), "Should have qualifying portfolios");

        // Run Grover search
        let mut gs = GroverSearch::new(4, marked.clone());
        gs.run();

        // Top result should be a marked state
        let (top_idx, _prob) = gs.measure_top();
        assert!(
            marked.contains(&top_idx),
            "Grover should find a marked state, found {}",
            top_idx
        );
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let mut gs = GroverSearch::new(5, vec![3, 10, 25]);
        let probs = gs.run();
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Probabilities should sum to 1, got {}",
            total
        );
    }
}
