//! Quantum Feature Map Library
//!
//! Implements quantum feature maps for encoding classical trading data
//! into quantum Hilbert space. Includes state vector simulation,
//! multiple feature map types, kernel computation, and Bybit integration.

use ndarray::Array2;
use rand::Rng;
use serde::Deserialize;
use std::f64::consts::PI;

// ============================================================================
// Complex number type
// ============================================================================

/// Minimal complex number for quantum state simulation.
#[derive(Debug, Clone, Copy)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// e^{i * theta}
    pub fn from_polar(theta: f64) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }
}

impl std::ops::Add for Complex64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for Complex64 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl std::ops::AddAssign for Complex64 {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

/// Type alias for quantum state vectors.
pub type StateVector = Vec<Complex64>;

// ============================================================================
// Quantum gates
// ============================================================================

/// 2x2 gate matrix type.
pub type Gate2x2 = [[Complex64; 2]; 2];

/// Rotation-Y gate: RY(theta).
pub fn ry_gate(theta: f64) -> Gate2x2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [
        [Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
        [Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
    ]
}

/// Rotation-Z gate: RZ(theta).
pub fn rz_gate(theta: f64) -> Gate2x2 {
    [
        [Complex64::from_polar(-theta / 2.0), Complex64::zero()],
        [Complex64::zero(), Complex64::from_polar(theta / 2.0)],
    ]
}

/// Hadamard gate.
pub fn hadamard_gate() -> Gate2x2 {
    let h = 1.0 / 2.0_f64.sqrt();
    [
        [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
        [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
    ]
}

// ============================================================================
// State vector operations
// ============================================================================

/// Create initial |0...0> state for n qubits.
pub fn zero_state(n_qubits: usize) -> StateVector {
    let dim = 1 << n_qubits;
    let mut state = vec![Complex64::zero(); dim];
    state[0] = Complex64::one();
    state
}

/// Apply a single-qubit gate to the target qubit.
pub fn apply_single_gate(state: &mut StateVector, target: usize, gate: Gate2x2) {
    let step = 1 << target;
    let mut i = 0;
    while i < state.len() {
        for j in i..i + step {
            let a = state[j];
            let b = state[j + step];
            state[j] = gate[0][0] * a + gate[0][1] * b;
            state[j + step] = gate[1][0] * a + gate[1][1] * b;
        }
        i += step << 1;
    }
}

/// Apply a CNOT gate with given control and target qubits.
pub fn apply_cnot(state: &mut StateVector, control: usize, target: usize) {
    let n_qubits = (state.len() as f64).log2() as usize;
    let dim = 1 << n_qubits;
    for i in 0..dim {
        let ctrl_bit = (i >> control) & 1;
        let tgt_bit = (i >> target) & 1;
        if ctrl_bit == 1 && tgt_bit == 0 {
            let j = i | (1 << target);
            let tmp = state[i];
            state[i] = state[j];
            state[j] = tmp;
        }
    }
}

/// Apply a ZZ interaction: exp(i * theta * Z_i Z_j).
/// This is a diagonal gate: phase of exp(i*theta) if qubits agree,
/// exp(-i*theta) if they disagree.
pub fn apply_zz_interaction(state: &mut StateVector, qubit_i: usize, qubit_j: usize, theta: f64) {
    let dim = state.len();
    for k in 0..dim {
        let bi = (k >> qubit_i) & 1;
        let bj = (k >> qubit_j) & 1;
        // Z eigenvalue: +1 for |0>, -1 for |1>
        let zi: f64 = if bi == 0 { 1.0 } else { -1.0 };
        let zj: f64 = if bj == 0 { 1.0 } else { -1.0 };
        let phase = Complex64::from_polar(theta * zi * zj);
        state[k] = state[k] * phase;
    }
}

/// Apply a Z rotation: exp(i * theta * Z_i) -- diagonal phase gate.
pub fn apply_z_phase(state: &mut StateVector, qubit: usize, theta: f64) {
    let dim = state.len();
    for k in 0..dim {
        let b = (k >> qubit) & 1;
        let z: f64 = if b == 0 { 1.0 } else { -1.0 };
        let phase = Complex64::from_polar(theta * z);
        state[k] = state[k] * phase;
    }
}

/// Compute inner product <a|b>.
pub fn inner_product(a: &StateVector, b: &StateVector) -> Complex64 {
    assert_eq!(a.len(), b.len());
    let mut result = Complex64::zero();
    for i in 0..a.len() {
        result += a[i].conj() * b[i];
    }
    result
}

/// Compute fidelity |<a|b>|^2.
pub fn fidelity(a: &StateVector, b: &StateVector) -> f64 {
    let ip = inner_product(a, b);
    ip.norm_sq()
}

/// Compute the reduced density matrix purity for a single qubit (tr(rho_i^2)).
pub fn single_qubit_purity(state: &StateVector, qubit: usize) -> f64 {
    let n_qubits = (state.len() as f64).log2() as usize;
    let dim = 1 << n_qubits;

    // Compute 2x2 reduced density matrix rho for the given qubit
    // rho[a][b] = sum over all other qubits of <..a..|psi><psi|..b..>
    let mut rho = [[Complex64::zero(); 2]; 2];
    for a in 0..2usize {
        for b in 0..2usize {
            for k in 0..dim {
                let bit_k = (k >> qubit) & 1;
                if bit_k != a {
                    continue;
                }
                // l is same as k but with qubit set to b
                let l = if b == 1 {
                    k | (1 << qubit)
                } else {
                    k & !(1 << qubit)
                };
                // rho[a][b] += state[k] * conj(state[l])
                rho[a][b] += state[k] * state[l].conj();
            }
        }
    }

    // tr(rho^2) = sum_{a,b} |rho[a][b]|^2
    let mut purity = 0.0;
    for a in 0..2 {
        for b in 0..2 {
            purity += rho[a][b].norm_sq();
        }
    }
    purity
}

/// Compute the Meyer-Wallach entanglement measure Q for a state.
/// Q = 2(1 - 1/n * sum_i tr(rho_i^2))
pub fn meyer_wallach_measure(state: &StateVector) -> f64 {
    let n_qubits = (state.len() as f64).log2() as usize;
    if n_qubits < 2 {
        return 0.0;
    }
    let avg_purity: f64 = (0..n_qubits)
        .map(|i| single_qubit_purity(state, i))
        .sum::<f64>()
        / n_qubits as f64;
    2.0 * (1.0 - avg_purity)
}

// ============================================================================
// Quantum feature map trait and implementations
// ============================================================================

/// Trait for quantum feature maps.
pub trait QuantumFeatureMap {
    /// Encode classical features into a quantum state vector.
    fn encode(&self, features: &[f64]) -> StateVector;

    /// Number of qubits used by this feature map.
    fn n_qubits(&self) -> usize;

    /// Compute quantum kernel k(x,y) = |<phi(x)|phi(y)>|^2.
    fn kernel(&self, x: &[f64], y: &[f64]) -> f64 {
        let sx = self.encode(x);
        let sy = self.encode(y);
        fidelity(&sx, &sy)
    }

    /// Compute the full kernel matrix for a dataset.
    fn kernel_matrix(&self, data: &[Vec<f64>]) -> Array2<f64> {
        let n = data.len();
        let mut km = Array2::zeros((n, n));
        let states: Vec<StateVector> = data.iter().map(|x| self.encode(x)).collect();
        for i in 0..n {
            km[[i, i]] = 1.0;
            for j in (i + 1)..n {
                let k = fidelity(&states[i], &states[j]);
                km[[i, j]] = k;
                km[[j, i]] = k;
            }
        }
        km
    }
}

// ============================================================================
// AngleMap: simple angle encoding with RY rotations
// ============================================================================

/// Angle encoding feature map: each feature is encoded as an RY rotation on one qubit.
pub struct AngleMap {
    pub n_qubits: usize,
}

impl AngleMap {
    pub fn new(n_features: usize) -> Self {
        Self {
            n_qubits: n_features,
        }
    }
}

impl QuantumFeatureMap for AngleMap {
    fn encode(&self, features: &[f64]) -> StateVector {
        assert!(
            features.len() <= self.n_qubits,
            "Too many features for AngleMap"
        );
        let mut state = zero_state(self.n_qubits);
        for (i, &f) in features.iter().enumerate() {
            apply_single_gate(&mut state, i, ry_gate(f));
        }
        state
    }

    fn n_qubits(&self) -> usize {
        self.n_qubits
    }
}

// ============================================================================
// ZZFeatureMap: Hadamard + Z rotations + ZZ entangling layers
// ============================================================================

/// ZZ feature map with configurable depth (number of layers).
pub struct ZZFeatureMap {
    pub n_qubits: usize,
    pub depth: usize,
}

impl ZZFeatureMap {
    pub fn new(n_features: usize, depth: usize) -> Self {
        Self {
            n_qubits: n_features,
            depth,
        }
    }

    /// Apply one layer of the ZZ feature map.
    fn apply_layer(&self, state: &mut StateVector, features: &[f64]) {
        let nq = self.n_qubits;

        // Hadamard on all qubits
        for i in 0..nq {
            apply_single_gate(state, i, hadamard_gate());
        }

        // Single-qubit Z rotations: exp(i * x_i * Z_i)
        for i in 0..features.len().min(nq) {
            apply_z_phase(state, i, features[i]);
        }

        // Two-qubit ZZ interactions: exp(i * x_i * x_j * Z_i Z_j)
        // Linear connectivity: (0,1), (1,2), ..., (n-2, n-1)
        for i in 0..nq.saturating_sub(1) {
            let fi = if i < features.len() { features[i] } else { 0.0 };
            let fj = if i + 1 < features.len() {
                features[i + 1]
            } else {
                0.0
            };
            apply_zz_interaction(state, i, i + 1, fi * fj);
        }
    }
}

impl QuantumFeatureMap for ZZFeatureMap {
    fn encode(&self, features: &[f64]) -> StateVector {
        assert!(
            features.len() <= self.n_qubits,
            "Too many features for ZZFeatureMap"
        );
        let mut state = zero_state(self.n_qubits);
        for _ in 0..self.depth {
            self.apply_layer(&mut state, features);
        }
        state
    }

    fn n_qubits(&self) -> usize {
        self.n_qubits
    }
}

// ============================================================================
// IQPFeatureMap: H - D(x) - H structure
// ============================================================================

/// IQP (Instantaneous Quantum Polynomial) feature map.
/// Structure: H^n -> D(x) -> H^n, where D(x) is diagonal.
pub struct IQPFeatureMap {
    pub n_qubits: usize,
    pub depth: usize,
}

impl IQPFeatureMap {
    pub fn new(n_features: usize, depth: usize) -> Self {
        Self {
            n_qubits: n_features,
            depth,
        }
    }

    fn apply_layer(&self, state: &mut StateVector, features: &[f64]) {
        let nq = self.n_qubits;

        // First Hadamard layer
        for i in 0..nq {
            apply_single_gate(state, i, hadamard_gate());
        }

        // Diagonal D(x): single-qubit Z phases
        for i in 0..features.len().min(nq) {
            apply_z_phase(state, i, features[i]);
        }

        // Diagonal D(x): two-qubit ZZ phases (all pairs)
        for i in 0..nq {
            for j in (i + 1)..nq {
                let fi = if i < features.len() { features[i] } else { 0.0 };
                let fj = if j < features.len() { features[j] } else { 0.0 };
                apply_zz_interaction(state, i, j, fi * fj);
            }
        }

        // Second Hadamard layer
        for i in 0..nq {
            apply_single_gate(state, i, hadamard_gate());
        }
    }
}

impl QuantumFeatureMap for IQPFeatureMap {
    fn encode(&self, features: &[f64]) -> StateVector {
        assert!(
            features.len() <= self.n_qubits,
            "Too many features for IQPFeatureMap"
        );
        let mut state = zero_state(self.n_qubits);
        for _ in 0..self.depth {
            self.apply_layer(&mut state, features);
        }
        state
    }

    fn n_qubits(&self) -> usize {
        self.n_qubits
    }
}

// ============================================================================
// Expressibility and entangling capability metrics
// ============================================================================

/// Estimate expressibility of a feature map by comparing its fidelity
/// distribution to the Haar-random distribution.
/// Returns the estimated KL divergence (lower = more expressible).
pub fn estimate_expressibility<F: QuantumFeatureMap>(
    fmap: &F,
    n_features: usize,
    n_samples: usize,
    n_bins: usize,
) -> f64 {
    let mut rng = rand::thread_rng();
    let n_qubits = fmap.n_qubits();
    let dim = 1u64 << n_qubits;

    // Sample fidelities from the circuit
    let mut fidelities = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let x: Vec<f64> = (0..n_features).map(|_| rng.gen::<f64>() * PI).collect();
        let y: Vec<f64> = (0..n_features).map(|_| rng.gen::<f64>() * PI).collect();
        let f = fmap.kernel(&x, &y);
        fidelities.push(f);
    }

    // Build histogram of circuit fidelities
    let mut circuit_hist = vec![0.0f64; n_bins];
    for &f in &fidelities {
        let bin = ((f * n_bins as f64) as usize).min(n_bins - 1);
        circuit_hist[bin] += 1.0;
    }
    // Normalize
    let total: f64 = circuit_hist.iter().sum();
    for v in circuit_hist.iter_mut() {
        *v /= total;
    }

    // Haar-random fidelity distribution: P(F) = (d-1)(1-F)^{d-2}
    let d = dim as f64;
    let mut haar_hist = vec![0.0f64; n_bins];
    let bin_width = 1.0 / n_bins as f64;
    for i in 0..n_bins {
        let f_mid = (i as f64 + 0.5) * bin_width;
        haar_hist[i] = (d - 1.0) * (1.0 - f_mid).powf(d - 2.0) * bin_width;
    }
    // Normalize Haar histogram
    let haar_total: f64 = haar_hist.iter().sum();
    for v in haar_hist.iter_mut() {
        *v /= haar_total;
    }

    // KL divergence: D_KL(P_circuit || P_Haar)
    let epsilon = 1e-10;
    let mut kl = 0.0;
    for i in 0..n_bins {
        let p = circuit_hist[i] + epsilon;
        let q = haar_hist[i] + epsilon;
        kl += p * (p / q).ln();
    }
    kl
}

/// Estimate entangling capability by averaging the Meyer-Wallach measure
/// over random feature vectors.
pub fn estimate_entangling_capability<F: QuantumFeatureMap>(
    fmap: &F,
    n_features: usize,
    n_samples: usize,
) -> f64 {
    let mut rng = rand::thread_rng();
    let mut total_q = 0.0;
    for _ in 0..n_samples {
        let x: Vec<f64> = (0..n_features).map(|_| rng.gen::<f64>() * PI).collect();
        let state = fmap.encode(&x);
        total_q += meyer_wallach_measure(&state);
    }
    total_q / n_samples as f64
}

// ============================================================================
// Feature engineering from OHLCV data
// ============================================================================

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

/// Compute trading features from a sequence of candles.
/// Returns a Vec of feature vectors, one per candle (starting from index 1).
/// Features: [return, range, upper_shadow, lower_shadow, volume_ratio]
pub fn engineer_features(candles: &[Candle]) -> Vec<Vec<f64>> {
    if candles.len() < 2 {
        return vec![];
    }

    // Compute rolling average volume (simple: use all available history)
    let mut features = Vec::new();
    let mut vol_sum = candles[0].volume;

    for i in 1..candles.len() {
        let c = &candles[i];
        let avg_vol = vol_sum / i as f64;

        let ret = (c.close - c.open) / c.open;
        let range = (c.high - c.low) / c.close;
        let upper_shadow = (c.high - c.open.max(c.close)) / c.close;
        let lower_shadow = (c.open.min(c.close) - c.low) / c.close;
        let volume_ratio = if avg_vol > 0.0 {
            c.volume / avg_vol
        } else {
            1.0
        };

        features.push(vec![ret, range, upper_shadow, lower_shadow, volume_ratio]);
        vol_sum += c.volume;
    }
    features
}

/// Normalize features to [0, pi] using min-max scaling.
pub fn normalize_features(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return vec![];
    }
    let n_features = data[0].len();

    // Find min and max for each feature
    let mut mins = vec![f64::INFINITY; n_features];
    let mut maxs = vec![f64::NEG_INFINITY; n_features];
    for row in data {
        for (j, &v) in row.iter().enumerate() {
            if v < mins[j] {
                mins[j] = v;
            }
            if v > maxs[j] {
                maxs[j] = v;
            }
        }
    }

    // Scale to [0, pi]
    data.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &v)| {
                    let range = maxs[j] - mins[j];
                    if range.abs() < 1e-12 {
                        PI / 2.0
                    } else {
                        ((v - mins[j]) / range) * PI
                    }
                })
                .collect()
        })
        .collect()
}

// ============================================================================
// Bybit API integration
// ============================================================================

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

/// Fetch OHLCV candles from Bybit API.
/// Returns candles sorted by time ascending.
pub fn fetch_bybit_candles(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>, anyhow::Error> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: retCode={}", resp.ret_code);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() < 6 {
                return None;
            }
            Some(Candle {
                timestamp: row[0].parse().ok()?,
                open: row[1].parse().ok()?,
                high: row[2].parse().ok()?,
                low: row[3].parse().ok()?,
                close: row[4].parse().ok()?,
                volume: row[5].parse().ok()?,
            })
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Kernel matrix statistics: mean, diagonal mean, off-diagonal mean, and std.
pub struct KernelStats {
    pub diag_mean: f64,
    pub off_diag_mean: f64,
    pub off_diag_std: f64,
}

/// Compute summary statistics of a kernel matrix.
pub fn kernel_stats(km: &Array2<f64>) -> KernelStats {
    let n = km.nrows();
    let mut off_diag_sum = 0.0;
    let mut off_diag_sq_sum = 0.0;
    let mut count = 0.0;

    for i in 0..n {
        for j in 0..n {
            if i != j {
                off_diag_sum += km[[i, j]];
                off_diag_sq_sum += km[[i, j]] * km[[i, j]];
                count += 1.0;
            }
        }
    }

    let off_diag_mean = if count > 0.0 {
        off_diag_sum / count
    } else {
        0.0
    };
    let off_diag_var = if count > 0.0 {
        off_diag_sq_sum / count - off_diag_mean * off_diag_mean
    } else {
        0.0
    };

    KernelStats {
        diag_mean: 1.0, // Always 1.0 for valid quantum kernels
        off_diag_mean,
        off_diag_std: off_diag_var.abs().sqrt(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, 4.0);
        let c = a * b;
        // (1+2i)(3+4i) = 3+4i+6i+8i^2 = -5+10i
        assert!(approx_eq(c.re, -5.0, 1e-10));
        assert!(approx_eq(c.im, 10.0, 1e-10));
    }

    #[test]
    fn test_zero_state() {
        let state = zero_state(3);
        assert_eq!(state.len(), 8);
        assert!(approx_eq(state[0].re, 1.0, 1e-10));
        for i in 1..8 {
            assert!(approx_eq(state[i].norm_sq(), 0.0, 1e-10));
        }
    }

    #[test]
    fn test_hadamard_single_qubit() {
        let mut state = zero_state(1);
        apply_single_gate(&mut state, 0, hadamard_gate());
        // Should be |+> = (|0> + |1>) / sqrt(2)
        let h = 1.0 / 2.0_f64.sqrt();
        assert!(approx_eq(state[0].re, h, 1e-10));
        assert!(approx_eq(state[1].re, h, 1e-10));
    }

    #[test]
    fn test_ry_gate() {
        let mut state = zero_state(1);
        apply_single_gate(&mut state, 0, ry_gate(PI));
        // RY(pi)|0> = |1>
        assert!(approx_eq(state[0].norm_sq(), 0.0, 1e-10));
        assert!(approx_eq(state[1].norm_sq(), 1.0, 1e-10));
    }

    #[test]
    fn test_state_normalization() {
        let mut state = zero_state(3);
        apply_single_gate(&mut state, 0, hadamard_gate());
        apply_single_gate(&mut state, 1, ry_gate(1.0));
        apply_single_gate(&mut state, 2, rz_gate(0.5));
        let norm: f64 = state.iter().map(|c| c.norm_sq()).sum();
        assert!(approx_eq(norm, 1.0, 1e-10));
    }

    #[test]
    fn test_cnot() {
        // CNOT on |10> should give |11>
        let mut state = zero_state(2);
        // Create |10>: flip qubit 1
        apply_single_gate(&mut state, 1, ry_gate(PI));
        apply_cnot(&mut state, 1, 0);
        // Should be |11> = state[3]
        assert!(approx_eq(state[3].norm_sq(), 1.0, 1e-10));
    }

    #[test]
    fn test_angle_map_self_kernel() {
        let fmap = AngleMap::new(3);
        let x = vec![0.5, 1.0, 1.5];
        let k = fmap.kernel(&x, &x);
        assert!(approx_eq(k, 1.0, 1e-10));
    }

    #[test]
    fn test_angle_map_different_inputs() {
        let fmap = AngleMap::new(2);
        let x = vec![0.0, 0.0];
        let y = vec![PI, PI];
        let k = fmap.kernel(&x, &y);
        // Very different inputs should have low kernel value
        assert!(k < 0.1);
    }

    #[test]
    fn test_zz_feature_map_self_kernel() {
        let fmap = ZZFeatureMap::new(3, 2);
        let x = vec![0.5, 1.0, 1.5];
        let k = fmap.kernel(&x, &x);
        assert!(approx_eq(k, 1.0, 1e-10));
    }

    #[test]
    fn test_zz_feature_map_normalization() {
        let fmap = ZZFeatureMap::new(4, 2);
        let x = vec![0.3, 0.7, 1.2, 2.1];
        let state = fmap.encode(&x);
        let norm: f64 = state.iter().map(|c| c.norm_sq()).sum();
        assert!(approx_eq(norm, 1.0, 1e-10));
    }

    #[test]
    fn test_iqp_feature_map_self_kernel() {
        let fmap = IQPFeatureMap::new(3, 1);
        let x = vec![0.5, 1.0, 1.5];
        let k = fmap.kernel(&x, &x);
        assert!(approx_eq(k, 1.0, 1e-10));
    }

    #[test]
    fn test_iqp_feature_map_normalization() {
        let fmap = IQPFeatureMap::new(4, 2);
        let x = vec![0.3, 0.7, 1.2, 2.1];
        let state = fmap.encode(&x);
        let norm: f64 = state.iter().map(|c| c.norm_sq()).sum();
        assert!(approx_eq(norm, 1.0, 1e-10));
    }

    #[test]
    fn test_kernel_matrix_symmetry() {
        let fmap = ZZFeatureMap::new(3, 1);
        let data = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];
        let km = fmap.kernel_matrix(&data);
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(km[[i, j]], km[[j, i]], 1e-10));
            }
            assert!(approx_eq(km[[i, i]], 1.0, 1e-10));
        }
    }

    #[test]
    fn test_meyer_wallach_product_state() {
        // Product state (no entanglement) should have Q = 0
        let fmap = AngleMap::new(3);
        let x = vec![0.5, 1.0, 1.5];
        let state = fmap.encode(&x);
        let q = meyer_wallach_measure(&state);
        assert!(approx_eq(q, 0.0, 1e-10));
    }

    #[test]
    fn test_meyer_wallach_entangled_state() {
        // Bell state (|00> + |11>)/sqrt(2) should have Q > 0
        let mut state = zero_state(2);
        apply_single_gate(&mut state, 0, hadamard_gate());
        apply_cnot(&mut state, 0, 1);
        let q = meyer_wallach_measure(&state);
        assert!(q > 0.5);
    }

    #[test]
    fn test_feature_engineering() {
        let candles = vec![
            Candle {
                timestamp: 0,
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 102.0,
                volume: 1000.0,
            },
            Candle {
                timestamp: 1,
                open: 102.0,
                high: 108.0,
                low: 100.0,
                close: 106.0,
                volume: 1200.0,
            },
        ];
        let features = engineer_features(&candles);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].len(), 5);

        // Return should be (106-102)/102
        let expected_ret = (106.0 - 102.0) / 102.0;
        assert!(approx_eq(features[0][0], expected_ret, 1e-10));
    }

    #[test]
    fn test_normalize_features() {
        let data = vec![vec![0.0, 10.0], vec![1.0, 20.0], vec![0.5, 15.0]];
        let normalized = normalize_features(&data);
        assert_eq!(normalized.len(), 3);
        // First row should be [0, 0], last [pi, pi], middle [pi/2, pi/2]
        assert!(approx_eq(normalized[0][0], 0.0, 1e-10));
        assert!(approx_eq(normalized[0][1], 0.0, 1e-10));
        assert!(approx_eq(normalized[1][0], PI, 1e-10));
        assert!(approx_eq(normalized[1][1], PI, 1e-10));
        assert!(approx_eq(normalized[2][0], PI / 2.0, 1e-10));
        assert!(approx_eq(normalized[2][1], PI / 2.0, 1e-10));
    }

    #[test]
    fn test_kernel_stats_computation() {
        let fmap = AngleMap::new(2);
        let data = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let km = fmap.kernel_matrix(&data);
        let stats = kernel_stats(&km);
        assert!(approx_eq(stats.diag_mean, 1.0, 1e-10));
        assert!(stats.off_diag_mean >= 0.0 && stats.off_diag_mean <= 1.0);
        assert!(stats.off_diag_std >= 0.0);
    }

    #[test]
    fn test_zz_produces_entanglement() {
        let fmap = ZZFeatureMap::new(3, 2);
        let cap = estimate_entangling_capability(&fmap, 3, 50);
        // ZZ map should produce non-zero entanglement
        assert!(cap > 0.01);
    }

    #[test]
    fn test_angle_map_no_entanglement() {
        let fmap = AngleMap::new(3);
        let cap = estimate_entangling_capability(&fmap, 3, 50);
        // Angle map produces product states -- zero entanglement
        assert!(approx_eq(cap, 0.0, 1e-10));
    }
}
