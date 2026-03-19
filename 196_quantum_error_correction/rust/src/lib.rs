//! Quantum Error Correction for Trading
//!
//! This library implements quantum noise channels, error correction codes,
//! and error mitigation techniques, demonstrating their impact on quantum
//! trading algorithms.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ============================================================================
// Quantum State Representation
// ============================================================================

/// Represents a quantum state as a vector of real amplitudes (simplified model).
/// For a single qubit: [alpha, beta] where |psi> = alpha|0> + beta|1>.
/// For n qubits: 2^n amplitudes.
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitudes: Array1<f64>,
    pub num_qubits: usize,
}

impl QuantumState {
    /// Create |0> state for a single qubit
    pub fn zero() -> Self {
        QuantumState {
            amplitudes: Array1::from_vec(vec![1.0, 0.0]),
            num_qubits: 1,
        }
    }

    /// Create |1> state for a single qubit
    pub fn one() -> Self {
        QuantumState {
            amplitudes: Array1::from_vec(vec![0.0, 1.0]),
            num_qubits: 1,
        }
    }

    /// Create a custom single-qubit state
    pub fn new(alpha: f64, beta: f64) -> Self {
        let norm = (alpha * alpha + beta * beta).sqrt();
        QuantumState {
            amplitudes: Array1::from_vec(vec![alpha / norm, beta / norm]),
            num_qubits: 1,
        }
    }

    /// Create an n-qubit zero state |00...0>
    pub fn zero_n(n: usize) -> Self {
        let size = 1 << n;
        let mut amps = Array1::zeros(size);
        amps[0] = 1.0;
        QuantumState {
            amplitudes: amps,
            num_qubits: n,
        }
    }

    /// Compute the probability of measuring each basis state
    pub fn probabilities(&self) -> Array1<f64> {
        self.amplitudes.mapv(|a| a * a)
    }

    /// Measure the state, returning the index of the measured basis state
    pub fn measure(&self) -> usize {
        let probs = self.probabilities();
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return i;
            }
        }
        probs.len() - 1
    }

    /// Get the fidelity (overlap) between two states
    pub fn fidelity(&self, other: &QuantumState) -> f64 {
        let dot: f64 = self.amplitudes.iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| a * b)
            .sum();
        dot * dot
    }
}

// ============================================================================
// Noise Channels
// ============================================================================

/// Apply a bit-flip (X) channel to a single qubit state with probability p.
/// With probability (1-p), state is unchanged; with probability p, |0> <-> |1>.
pub fn bit_flip_channel(state: &QuantumState, p: f64) -> QuantumState {
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    if r < p {
        // Apply X gate: swap amplitudes
        let mut new_amps = state.amplitudes.clone();
        let n = new_amps.len();
        for i in 0..n / 2 {
            new_amps.swap(i, i + n / 2);
        }
        QuantumState {
            amplitudes: new_amps,
            num_qubits: state.num_qubits,
        }
    } else {
        state.clone()
    }
}

/// Apply a phase-flip (Z) channel to a single qubit state with probability p.
/// With probability p, applies Z: |0> -> |0>, |1> -> -|1>.
pub fn phase_flip_channel(state: &QuantumState, p: f64) -> QuantumState {
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    if r < p {
        let mut new_amps = state.amplitudes.clone();
        let n = new_amps.len();
        // Flip phase of |1> components (odd indices in single qubit, upper half in multi)
        for i in n / 2..n {
            new_amps[i] = -new_amps[i];
        }
        QuantumState {
            amplitudes: new_amps,
            num_qubits: state.num_qubits,
        }
    } else {
        state.clone()
    }
}

/// Apply a depolarizing channel to a single qubit state with probability p.
/// With probability (1-p), state is unchanged.
/// With probability p/3 each, apply X, Y (=XZ), or Z.
pub fn depolarizing_channel(state: &QuantumState, p: f64) -> QuantumState {
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();

    if r < 1.0 - p {
        // No error
        state.clone()
    } else if r < 1.0 - 2.0 * p / 3.0 {
        // X error (bit flip)
        bit_flip_channel_deterministic(state)
    } else if r < 1.0 - p / 3.0 {
        // Z error (phase flip)
        phase_flip_channel_deterministic(state)
    } else {
        // Y error (both bit and phase flip)
        let flipped = bit_flip_channel_deterministic(state);
        phase_flip_channel_deterministic(&flipped)
    }
}

/// Deterministic bit flip (always applies X)
fn bit_flip_channel_deterministic(state: &QuantumState) -> QuantumState {
    let mut new_amps = state.amplitudes.clone();
    let n = new_amps.len();
    for i in 0..n / 2 {
        new_amps.swap(i, i + n / 2);
    }
    QuantumState {
        amplitudes: new_amps,
        num_qubits: state.num_qubits,
    }
}

/// Deterministic phase flip (always applies Z)
fn phase_flip_channel_deterministic(state: &QuantumState) -> QuantumState {
    let mut new_amps = state.amplitudes.clone();
    let n = new_amps.len();
    for i in n / 2..n {
        new_amps[i] = -new_amps[i];
    }
    QuantumState {
        amplitudes: new_amps,
        num_qubits: state.num_qubits,
    }
}

// ============================================================================
// 3-Qubit Bit Flip Code
// ============================================================================

/// Encode a single logical qubit into 3 physical qubits using the bit-flip code.
/// |0>_L = |000>, |1>_L = |111>
/// alpha|0> + beta|1> -> alpha|000> + beta|111>
pub fn encode_bit_flip_code(state: &QuantumState) -> QuantumState {
    assert_eq!(state.num_qubits, 1, "Input must be a single qubit");
    let alpha = state.amplitudes[0];
    let beta = state.amplitudes[1];

    // 3-qubit state: 8 amplitudes for |000> through |111>
    let mut encoded = Array1::zeros(8);
    encoded[0b000] = alpha; // |000>
    encoded[0b111] = beta;  // |111>

    QuantumState {
        amplitudes: encoded,
        num_qubits: 3,
    }
}

/// Apply bit-flip noise to a specific qubit in a 3-qubit state.
/// qubit_index: 0, 1, or 2 (from most significant to least significant)
pub fn apply_bit_flip_to_qubit(state: &QuantumState, qubit_index: usize, p: f64) -> QuantumState {
    assert_eq!(state.num_qubits, 3);
    assert!(qubit_index < 3);

    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();

    if r >= p {
        return state.clone();
    }

    let mut new_amps = state.amplitudes.clone();
    let bit = 2 - qubit_index; // Convert to bit position (qubit 0 = bit 2, etc.)

    for i in 0..8 {
        let j = i ^ (1 << bit); // Flip the target bit
        if i < j {
            new_amps.swap(i, j);
        }
    }

    QuantumState {
        amplitudes: new_amps,
        num_qubits: 3,
    }
}

/// Measure the error syndrome for the 3-qubit bit flip code.
/// Returns (s1, s2) where:
///   s1 = parity of qubit 0 and qubit 1
///   s2 = parity of qubit 1 and qubit 2
/// Syndrome interpretation:
///   (0,0) -> no error
///   (1,0) -> error on qubit 0
///   (1,1) -> error on qubit 1
///   (0,1) -> error on qubit 2
pub fn measure_syndrome(state: &QuantumState) -> (u8, u8) {
    assert_eq!(state.num_qubits, 3);

    // Find the most likely basis state
    let probs = state.probabilities();
    let most_likely = probs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let q0 = (most_likely >> 2) & 1;
    let q1 = (most_likely >> 1) & 1;
    let q2 = most_likely & 1;

    let s1 = (q0 ^ q1) as u8;
    let s2 = (q1 ^ q2) as u8;

    (s1, s2)
}

/// Correct a single bit-flip error based on the syndrome.
pub fn correct_bit_flip(state: &QuantumState, syndrome: (u8, u8)) -> QuantumState {
    let qubit_to_flip = match syndrome {
        (0, 0) => return state.clone(), // No error
        (1, 0) => 0, // Error on qubit 0
        (1, 1) => 1, // Error on qubit 1
        (0, 1) => 2, // Error on qubit 2
        _ => return state.clone(),
    };

    let mut new_amps = state.amplitudes.clone();
    let bit = 2 - qubit_to_flip;

    for i in 0..8 {
        let j = i ^ (1 << bit);
        if i < j {
            new_amps.swap(i, j);
        }
    }

    QuantumState {
        amplitudes: new_amps,
        num_qubits: 3,
    }
}

/// Decode a 3-qubit state back to a single logical qubit.
/// Assumes the state is in the code space: alpha|000> + beta|111>.
pub fn decode_bit_flip_code(state: &QuantumState) -> QuantumState {
    assert_eq!(state.num_qubits, 3);
    let alpha = state.amplitudes[0b000];
    let beta = state.amplitudes[0b111];

    let norm = (alpha * alpha + beta * beta).sqrt();
    QuantumState {
        amplitudes: Array1::from_vec(vec![alpha / norm, beta / norm]),
        num_qubits: 1,
    }
}

/// Full error correction pipeline: encode, apply noise, measure syndrome, correct, decode.
pub fn run_error_correction_pipeline(
    state: &QuantumState,
    noise_prob: f64,
) -> QuantumState {
    // Encode
    let encoded = encode_bit_flip_code(state);

    // Apply noise to each qubit independently
    let noisy = apply_bit_flip_to_qubit(&encoded, 0, noise_prob);
    let noisy = apply_bit_flip_to_qubit(&noisy, 1, noise_prob);
    let noisy = apply_bit_flip_to_qubit(&noisy, 2, noise_prob);

    // Measure syndrome
    let syndrome = measure_syndrome(&noisy);

    // Correct
    let corrected = correct_bit_flip(&noisy, syndrome);

    // Decode
    decode_bit_flip_code(&corrected)
}

// ============================================================================
// Zero-Noise Extrapolation
// ============================================================================

/// Perform zero-noise extrapolation using linear Richardson extrapolation.
/// Given expectation values at different noise scale factors,
/// extrapolate to zero noise.
///
/// noise_levels: scale factors (e.g., [1.0, 2.0, 3.0])
/// expectation_values: measured expectation values at each noise level
///
/// Returns: estimated zero-noise expectation value
pub fn zero_noise_extrapolation(
    noise_levels: &[f64],
    expectation_values: &[f64],
) -> f64 {
    assert_eq!(noise_levels.len(), expectation_values.len());
    assert!(!noise_levels.is_empty());

    match noise_levels.len() {
        1 => expectation_values[0],
        2 => {
            // Linear extrapolation to x=0
            let (x1, y1) = (noise_levels[0], expectation_values[0]);
            let (x2, y2) = (noise_levels[1], expectation_values[1]);
            let slope = (y2 - y1) / (x2 - x1);
            y1 - slope * x1
        }
        _ => {
            // Richardson extrapolation with polynomial fitting
            // For n points, fit polynomial of degree n-1 and evaluate at x=0
            let n = noise_levels.len();

            // Lagrange interpolation evaluated at x=0
            let mut result = 0.0;
            for i in 0..n {
                let mut basis = 1.0;
                for j in 0..n {
                    if i != j {
                        basis *= (0.0 - noise_levels[j]) / (noise_levels[i] - noise_levels[j]);
                    }
                }
                result += expectation_values[i] * basis;
            }
            result
        }
    }
}

// ============================================================================
// Quantum Kernel Computation
// ============================================================================

/// Compute a simplified quantum kernel value between two feature vectors.
/// K(x, y) = |<0|U^dag(x) U(y)|0>|^2
/// We use a simple rotation-based encoding: U(x)|0> = cos(x)|0> + sin(x)|1>
pub fn quantum_kernel_ideal(x: f64, y: f64) -> f64 {
    // |<0|U^dag(x)U(y)|0>|^2 = cos^2(x-y) for rotation encoding
    let diff = x - y;
    diff.cos().powi(2)
}

/// Compute quantum kernel with depolarizing noise.
/// Noise reduces the kernel value toward the maximally mixed state (0.5).
pub fn quantum_kernel_noisy(x: f64, y: f64, noise_prob: f64) -> f64 {
    let ideal = quantum_kernel_ideal(x, y);
    // Depolarizing noise shrinks the kernel toward 0.5
    let shrink_factor = (1.0 - noise_prob).powi(2); // Two layers of noise
    0.5 + shrink_factor * (ideal - 0.5)
}

/// Compute quantum kernel with error correction (reduced effective noise).
/// The bit-flip code reduces the effective error rate from p to O(p^2).
pub fn quantum_kernel_corrected(x: f64, y: f64, noise_prob: f64) -> f64 {
    // After error correction, effective error rate is ~3p^2 (probability of 2+ errors)
    let effective_noise = 3.0 * noise_prob * noise_prob;
    quantum_kernel_noisy(x, y, effective_noise)
}

/// Compute a kernel matrix for a set of data points
pub fn compute_kernel_matrix<F>(data: &[f64], kernel_fn: F) -> Array2<f64>
where
    F: Fn(f64, f64) -> f64,
{
    let n = data.len();
    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            matrix[[i, j]] = kernel_fn(data[i], data[j]);
        }
    }
    matrix
}

/// Simple kernel-based nearest-neighbor classifier.
/// Returns predicted labels for test data based on training data and labels.
pub fn kernel_classify(
    train_data: &[f64],
    train_labels: &[i32],
    test_data: &[f64],
    kernel_matrix_fn: &dyn Fn(f64, f64) -> f64,
) -> Vec<i32> {
    test_data
        .iter()
        .map(|&test_point| {
            let mut best_sim = f64::NEG_INFINITY;
            let mut best_label = 0;
            for (i, &train_point) in train_data.iter().enumerate() {
                let sim = kernel_matrix_fn(test_point, train_point);
                if sim > best_sim {
                    best_sim = sim;
                    best_label = train_labels[i];
                }
            }
            best_label
        })
        .collect()
}

/// Compute classification accuracy
pub fn accuracy(predictions: &[i32], labels: &[i32]) -> f64 {
    assert_eq!(predictions.len(), labels.len());
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(p, l)| p == l)
        .count();
    correct as f64 / predictions.len() as f64
}

// ============================================================================
// Noisy vs Error-Corrected Circuit Comparison
// ============================================================================

/// Run a simple computation (kernel evaluation) with noise many times
/// and return the average result.
pub fn run_noisy_kernel_average(
    x: f64,
    y: f64,
    noise_prob: f64,
    num_shots: usize,
) -> f64 {
    let mut total = 0.0;
    for _ in 0..num_shots {
        total += quantum_kernel_noisy(x, y, noise_prob);
    }
    total / num_shots as f64
}

/// Run a computation with error correction many times
/// and return the average result.
pub fn run_corrected_kernel_average(
    x: f64,
    y: f64,
    noise_prob: f64,
    num_shots: usize,
) -> f64 {
    let mut total = 0.0;
    for _ in 0..num_shots {
        total += quantum_kernel_corrected(x, y, noise_prob);
    }
    total / num_shots as f64
}

/// Compare noisy, corrected, and ideal kernel values
pub fn compare_kernel_methods(
    x: f64,
    y: f64,
    noise_prob: f64,
    num_shots: usize,
) -> KernelComparison {
    let ideal = quantum_kernel_ideal(x, y);
    let noisy = run_noisy_kernel_average(x, y, noise_prob, num_shots);
    let corrected = run_corrected_kernel_average(x, y, noise_prob, num_shots);

    // Zero-noise extrapolation: evaluate at noise_prob and 2*noise_prob
    let e1 = quantum_kernel_noisy(x, y, noise_prob);
    let e2 = quantum_kernel_noisy(x, y, 2.0 * noise_prob);
    let e3 = quantum_kernel_noisy(x, y, 3.0 * noise_prob);
    let mitigated = zero_noise_extrapolation(
        &[1.0, 2.0, 3.0],
        &[e1, e2, e3],
    );

    KernelComparison {
        ideal,
        noisy,
        corrected,
        mitigated,
        noisy_error: (noisy - ideal).abs(),
        corrected_error: (corrected - ideal).abs(),
        mitigated_error: (mitigated - ideal).abs(),
    }
}

#[derive(Debug, Clone)]
pub struct KernelComparison {
    pub ideal: f64,
    pub noisy: f64,
    pub corrected: f64,
    pub mitigated: f64,
    pub noisy_error: f64,
    pub corrected_error: f64,
    pub mitigated_error: f64,
}

// ============================================================================
// Bybit API Integration
// ============================================================================

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
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Kline {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline data from Bybit API
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Kline>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let klines: Vec<Kline> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Kline {
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

    Ok(klines)
}

/// Compute returns from kline data
pub fn compute_returns(klines: &[Kline]) -> Vec<f64> {
    klines
        .windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect()
}

/// Compute simple moving average volatility
pub fn compute_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window {
        return vec![];
    }
    returns
        .windows(window)
        .map(|w| {
            let mean: f64 = w.iter().sum::<f64>() / w.len() as f64;
            let variance: f64 = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / w.len() as f64;
            variance.sqrt()
        })
        .collect()
}

/// Generate binary labels: 1 if next return is positive, -1 otherwise
pub fn generate_labels(returns: &[f64]) -> Vec<i32> {
    returns.iter().map(|&r| if r > 0.0 { 1 } else { -1 }).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::zero();
        assert!((state.amplitudes[0] - 1.0).abs() < 1e-10);
        assert!((state.amplitudes[1] - 0.0).abs() < 1e-10);

        let state = QuantumState::one();
        assert!((state.amplitudes[0] - 0.0).abs() < 1e-10);
        assert!((state.amplitudes[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_state_normalization() {
        let state = QuantumState::new(3.0, 4.0);
        let norm: f64 = state.amplitudes.iter().map(|a| a * a).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fidelity() {
        let s1 = QuantumState::zero();
        let s2 = QuantumState::zero();
        assert!((s1.fidelity(&s2) - 1.0).abs() < 1e-10);

        let s3 = QuantumState::one();
        assert!(s1.fidelity(&s3).abs() < 1e-10);
    }

    #[test]
    fn test_bit_flip_code_no_error() {
        let state = QuantumState::new(0.6, 0.8);
        let encoded = encode_bit_flip_code(&state);
        assert_eq!(encoded.num_qubits, 3);

        let syndrome = measure_syndrome(&encoded);
        assert_eq!(syndrome, (0, 0));

        let decoded = decode_bit_flip_code(&encoded);
        assert!((decoded.fidelity(&state) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bit_flip_code_qubit0_error() {
        let state = QuantumState::zero();
        let encoded = encode_bit_flip_code(&state);

        // Manually flip qubit 0: |000> -> |100>
        let flipped = apply_bit_flip_to_qubit(&encoded, 0, 1.0);
        assert!((flipped.amplitudes[0b100] - 1.0).abs() < 1e-10);

        let syndrome = measure_syndrome(&flipped);
        assert_eq!(syndrome, (1, 0));

        let corrected = correct_bit_flip(&flipped, syndrome);
        assert!((corrected.amplitudes[0b000] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bit_flip_code_qubit1_error() {
        let state = QuantumState::zero();
        let encoded = encode_bit_flip_code(&state);

        let flipped = apply_bit_flip_to_qubit(&encoded, 1, 1.0);
        assert!((flipped.amplitudes[0b010] - 1.0).abs() < 1e-10);

        let syndrome = measure_syndrome(&flipped);
        assert_eq!(syndrome, (1, 1));

        let corrected = correct_bit_flip(&flipped, syndrome);
        assert!((corrected.amplitudes[0b000] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bit_flip_code_qubit2_error() {
        let state = QuantumState::zero();
        let encoded = encode_bit_flip_code(&state);

        let flipped = apply_bit_flip_to_qubit(&encoded, 2, 1.0);
        assert!((flipped.amplitudes[0b001] - 1.0).abs() < 1e-10);

        let syndrome = measure_syndrome(&flipped);
        assert_eq!(syndrome, (0, 1));

        let corrected = correct_bit_flip(&flipped, syndrome);
        assert!((corrected.amplitudes[0b000] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_kernel_ideal() {
        // Same point should give kernel = 1
        let k = quantum_kernel_ideal(0.5, 0.5);
        assert!((k - 1.0).abs() < 1e-10);

        // Orthogonal points
        let k = quantum_kernel_ideal(0.0, std::f64::consts::FRAC_PI_2);
        assert!(k.abs() < 1e-10);
    }

    #[test]
    fn test_quantum_kernel_noisy() {
        // Noisy kernel should be closer to 0.5 than ideal
        let ideal = quantum_kernel_ideal(0.3, 0.7);
        let noisy = quantum_kernel_noisy(0.3, 0.7, 0.1);

        let ideal_dist = (ideal - 0.5).abs();
        let noisy_dist = (noisy - 0.5).abs();
        assert!(noisy_dist < ideal_dist);
    }

    #[test]
    fn test_quantum_kernel_corrected() {
        // Corrected kernel should be between noisy and ideal
        let ideal = quantum_kernel_ideal(0.3, 0.7);
        let noisy = quantum_kernel_noisy(0.3, 0.7, 0.1);
        let corrected = quantum_kernel_corrected(0.3, 0.7, 0.1);

        let noisy_err = (noisy - ideal).abs();
        let corrected_err = (corrected - ideal).abs();
        assert!(corrected_err < noisy_err);
    }

    #[test]
    fn test_zero_noise_extrapolation_linear() {
        // f(x) = 1 - 0.5x, so f(0) = 1.0
        let noise_levels = vec![1.0, 2.0];
        let values = vec![0.5, 0.0];
        let result = zero_noise_extrapolation(&noise_levels, &values);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_noise_extrapolation_quadratic() {
        // f(x) = 1 - x + 0.5x^2, so f(0) = 1.0
        let noise_levels = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 1.0, 2.5];
        let result = zero_noise_extrapolation(&noise_levels, &values);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy() {
        let preds = vec![1, 1, -1, -1, 1];
        let labels = vec![1, -1, -1, -1, 1];
        let acc = accuracy(&preds, &labels);
        assert!((acc - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_compute_returns() {
        let klines = vec![
            Kline { timestamp: 0, open: 100.0, high: 110.0, low: 90.0, close: 100.0, volume: 1.0 },
            Kline { timestamp: 1, open: 100.0, high: 115.0, low: 95.0, close: 110.0, volume: 1.0 },
            Kline { timestamp: 2, open: 110.0, high: 120.0, low: 100.0, close: 105.0, volume: 1.0 },
        ];
        let returns = compute_returns(&klines);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.1).abs() < 1e-10);
        assert!((returns[1] - (-5.0 / 110.0)).abs() < 1e-10);
    }

    #[test]
    fn test_generate_labels() {
        let returns = vec![0.01, -0.02, 0.005, -0.001, 0.0];
        let labels = generate_labels(&returns);
        assert_eq!(labels, vec![1, -1, 1, -1, -1]);
    }

    #[test]
    fn test_kernel_comparison() {
        let comp = compare_kernel_methods(0.3, 0.7, 0.05, 100);
        assert!(comp.corrected_error < comp.noisy_error);
    }

    #[test]
    fn test_depolarizing_channel_no_noise() {
        let state = QuantumState::zero();
        // With p=0, state should be unchanged
        let result = depolarizing_channel(&state, 0.0);
        assert!((result.fidelity(&state) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_error_correction_pipeline_no_noise() {
        let state = QuantumState::new(0.6, 0.8);
        let result = run_error_correction_pipeline(&state, 0.0);
        assert!((result.fidelity(&state) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_kernel_matrix() {
        let data = vec![0.0, 1.0, 2.0];
        let matrix = compute_kernel_matrix(&data, quantum_kernel_ideal);
        // Diagonal should be 1.0
        for i in 0..3 {
            assert!((matrix[[i, i]] - 1.0).abs() < 1e-10);
        }
        // Should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[[i, j]] - matrix[[j, i]]).abs() < 1e-10);
            }
        }
    }
}
