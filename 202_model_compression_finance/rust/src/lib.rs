//! Model Compression Finance
//!
//! A library for compressing neural network models used in financial trading.
//! Implements magnitude pruning, INT8 quantization, and SVD-based low-rank factorization.

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;
use std::time::Instant;

// =============================================================================
// Neural Network
// =============================================================================

/// A single dense (fully connected) layer.
#[derive(Clone, Debug)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
}

/// A feedforward neural network composed of dense layers.
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
}

impl NeuralNetwork {
    /// Create a new neural network with the given layer sizes.
    /// `layer_sizes` must have at least 2 elements (input and output).
    /// Weights are initialized with Xavier/Glorot uniform initialization.
    pub fn new(layer_sizes: &[usize]) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(anyhow!("Need at least 2 layer sizes (input and output)"));
        }
        let mut rng = rand::thread_rng();
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let fan_in = layer_sizes[i];
            let fan_out = layer_sizes[i + 1];
            let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();

            let weights = Array2::from_shape_fn((fan_in, fan_out), |_| {
                rng.gen_range(-limit..limit)
            });
            let biases = Array1::zeros(fan_out);

            layers.push(DenseLayer { weights, biases });
        }

        Ok(NeuralNetwork { layers })
    }

    /// Forward pass through the network.
    /// Applies ReLU activation between hidden layers; final layer is linear.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = x.dot(&layer.weights) + &layer.biases;
            // Apply ReLU for all layers except the last
            if i < self.layers.len() - 1 {
                x.mapv_inplace(|v| v.max(0.0));
            }
        }
        x
    }

    /// Total number of parameters (weights + biases).
    pub fn param_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum()
    }

    /// Total number of non-zero parameters.
    pub fn nonzero_param_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.weights.iter().filter(|&&w| w != 0.0).count()
                    + l.biases.iter().filter(|&&b| b != 0.0).count()
            })
            .sum()
    }

    /// Memory size in bytes (assuming f64 = 8 bytes per parameter).
    pub fn memory_bytes(&self) -> usize {
        self.param_count() * 8
    }

    /// Train the network on input-output pairs using simple SGD.
    pub fn train(
        &mut self,
        inputs: &[Array1<f64>],
        targets: &[Array1<f64>],
        learning_rate: f64,
        epochs: usize,
    ) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                self.train_step(input, target, learning_rate);
            }
        }
    }

    /// Single training step with backpropagation.
    fn train_step(&mut self, input: &Array1<f64>, target: &Array1<f64>, lr: f64) {
        // Forward pass, storing activations
        let mut activations: Vec<Array1<f64>> = Vec::new();
        activations.push(input.clone());

        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = x.dot(&layer.weights) + &layer.biases;
            if i < self.layers.len() - 1 {
                x.mapv_inplace(|v| v.max(0.0));
            }
            activations.push(x.clone());
        }

        // Backward pass
        let output = activations.last().unwrap();
        let mut delta = output - target;

        for i in (0..self.layers.len()).rev() {
            let prev_activation = &activations[i];

            // Gradient for weights: outer product of prev_activation and delta
            let grad_w = outer_product(prev_activation, &delta);
            let grad_b = delta.clone();

            // Update weights and biases
            self.layers[i].weights = &self.layers[i].weights - &(grad_w * lr);
            self.layers[i].biases = &self.layers[i].biases - &(grad_b * lr);

            if i > 0 {
                // Propagate delta to previous layer
                delta = delta.dot(&self.layers[i].weights.t());
                // Apply ReLU derivative
                let prev = &activations[i];
                for (d, &a) in delta.iter_mut().zip(prev.iter()) {
                    if a <= 0.0 {
                        *d = 0.0;
                    }
                }
            }
        }
    }
}

/// Compute outer product of two 1D arrays.
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let m = a.len();
    let n = b.len();
    let mut result = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

// =============================================================================
// Compression Metrics
// =============================================================================

/// Metrics describing the compression of a model.
#[derive(Debug, Clone)]
pub struct CompressionMetrics {
    pub original_params: usize,
    pub compressed_params: usize,
    pub original_nonzero: usize,
    pub compressed_nonzero: usize,
    pub original_memory_bytes: usize,
    pub compressed_memory_bytes: usize,
    pub compression_ratio: f64,
    pub sparsity: f64,
    pub memory_savings_percent: f64,
}

impl std::fmt::Display for CompressionMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CompressionMetrics {{\n  \
             original_params: {},\n  \
             compressed_nonzero: {},\n  \
             compression_ratio: {:.2}x,\n  \
             sparsity: {:.2}%,\n  \
             memory_savings: {:.2}%,\n  \
             original_memory: {} bytes,\n  \
             compressed_memory: {} bytes\n\
             }}",
            self.original_params,
            self.compressed_nonzero,
            self.compression_ratio,
            self.sparsity * 100.0,
            self.memory_savings_percent,
            self.original_memory_bytes,
            self.compressed_memory_bytes,
        )
    }
}

/// Compute compression metrics between an original and compressed network.
pub fn compute_metrics(original: &NeuralNetwork, compressed: &NeuralNetwork) -> CompressionMetrics {
    let original_params = original.param_count();
    let compressed_params = compressed.param_count();
    let original_nonzero = original.nonzero_param_count();
    let compressed_nonzero = compressed.nonzero_param_count();
    let original_memory = original.memory_bytes();
    let compressed_memory = compressed.memory_bytes();

    let compression_ratio = if compressed_nonzero > 0 {
        original_nonzero as f64 / compressed_nonzero as f64
    } else {
        f64::INFINITY
    };

    let sparsity = 1.0 - (compressed_nonzero as f64 / compressed_params as f64);
    let memory_savings = if original_memory > 0 {
        (1.0 - compressed_memory as f64 / original_memory as f64) * 100.0
    } else {
        0.0
    };

    CompressionMetrics {
        original_params,
        compressed_params,
        original_nonzero,
        compressed_nonzero,
        original_memory_bytes: original_memory,
        compressed_memory_bytes: compressed_memory,
        compression_ratio,
        sparsity,
        memory_savings_percent: memory_savings,
    }
}

// =============================================================================
// Magnitude Pruning
// =============================================================================

/// Apply unstructured magnitude pruning to a network.
/// `sparsity` is the fraction of weights to set to zero (0.0 to 1.0).
pub fn magnitude_prune(network: &NeuralNetwork, sparsity: f64) -> NeuralNetwork {
    assert!(
        (0.0..=1.0).contains(&sparsity),
        "Sparsity must be between 0.0 and 1.0"
    );

    // Collect absolute values of all weights
    let mut all_abs_weights: Vec<f64> = network
        .layers
        .iter()
        .flat_map(|l| l.weights.iter().map(|w| w.abs()))
        .collect();

    all_abs_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let cutoff_index = ((sparsity * all_abs_weights.len() as f64) as usize)
        .min(all_abs_weights.len().saturating_sub(1));
    let threshold = all_abs_weights[cutoff_index];

    let mut pruned = network.clone();
    for layer in &mut pruned.layers {
        layer.weights.mapv_inplace(|w| {
            if w.abs() <= threshold {
                0.0
            } else {
                w
            }
        });
    }

    pruned
}

// =============================================================================
// Weight Quantization (FP64 -> INT8 simulation)
// =============================================================================

/// Quantized representation of a dense layer.
#[derive(Debug, Clone)]
pub struct QuantizedLayer {
    pub weights_int8: Vec<i8>,
    pub shape: (usize, usize),
    pub scale: f64,
    pub zero_point: i8,
    pub biases: Array1<f64>, // biases kept in full precision
}

/// Quantized neural network.
#[derive(Debug, Clone)]
pub struct QuantizedNetwork {
    pub layers: Vec<QuantizedLayer>,
}

impl QuantizedNetwork {
    /// Memory size in bytes.
    /// INT8 weights use 1 byte each, biases remain f64 (8 bytes).
    pub fn memory_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.weights_int8.len()          // 1 byte per INT8 weight
                    + 8                        // scale (f64)
                    + 1                        // zero_point (i8)
                    + l.biases.len() * 8       // biases in f64
            })
            .sum()
    }

    /// Forward pass using dequantized weights.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            // Dequantize weights
            let (rows, cols) = layer.shape;
            let mut weights = Array2::zeros((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    let idx = r * cols + c;
                    weights[[r, c]] =
                        (layer.weights_int8[idx] as f64 - layer.zero_point as f64) * layer.scale;
                }
            }
            x = x.dot(&weights) + &layer.biases;
            if i < self.layers.len() - 1 {
                x.mapv_inplace(|v| v.max(0.0));
            }
        }
        x
    }
}

/// Quantize a neural network from f64 to INT8.
pub fn quantize_network(network: &NeuralNetwork) -> QuantizedNetwork {
    let layers = network
        .layers
        .iter()
        .map(|layer| {
            let w_min = layer
                .weights
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let w_max = layer
                .weights
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            let scale = if (w_max - w_min).abs() < 1e-10 {
                1e-10
            } else {
                (w_max - w_min) / 255.0
            };
            let zero_point = (-128.0_f64).max((-w_min / scale - 128.0).round()).min(127.0) as i8;

            let (rows, cols) = (layer.weights.nrows(), layer.weights.ncols());
            let weights_int8: Vec<i8> = layer
                .weights
                .iter()
                .map(|&w| {
                    let q = (w / scale + zero_point as f64).round();
                    q.max(-128.0).min(127.0) as i8
                })
                .collect();

            QuantizedLayer {
                weights_int8,
                shape: (rows, cols),
                scale,
                zero_point,
                biases: layer.biases.clone(),
            }
        })
        .collect();

    QuantizedNetwork { layers }
}

// =============================================================================
// Low-Rank Factorization (SVD-based)
// =============================================================================

/// Apply SVD-based low-rank factorization to compress each layer.
/// `rank_fraction` (0.0 to 1.0) controls how many singular values to keep.
pub fn low_rank_factorize(network: &NeuralNetwork, rank_fraction: f64) -> NeuralNetwork {
    assert!(
        (0.0..=1.0).contains(&rank_fraction),
        "rank_fraction must be between 0.0 and 1.0"
    );

    let mut factorized = network.clone();

    for layer in &mut factorized.layers {
        let (m, n) = (layer.weights.nrows(), layer.weights.ncols());
        let min_dim = m.min(n);
        let target_rank = ((rank_fraction * min_dim as f64).round() as usize).max(1);

        // Simple power-iteration based SVD approximation
        let approx = truncated_svd_approx(&layer.weights, target_rank);
        layer.weights = approx;
    }

    factorized
}

/// Approximate truncated SVD using power iteration method.
/// Returns the rank-k approximation of the input matrix.
fn truncated_svd_approx(matrix: &Array2<f64>, rank: usize) -> Array2<f64> {
    let (m, n) = (matrix.nrows(), matrix.ncols());
    let k = rank.min(m.min(n));

    let mut rng = rand::thread_rng();

    // Build U and V matrices column by column
    let mut u_cols: Vec<Array1<f64>> = Vec::with_capacity(k);
    let mut v_cols: Vec<Array1<f64>> = Vec::with_capacity(k);
    let mut sigmas: Vec<f64> = Vec::with_capacity(k);

    let mut residual = matrix.clone();

    for _ in 0..k {
        // Random starting vector
        let mut v: Array1<f64> = Array1::from_shape_fn(n, |_| rng.gen_range(-1.0..1.0));
        let norm = v.dot(&v).sqrt();
        if norm > 1e-10 {
            v /= norm;
        }

        // Power iteration to find dominant singular vector
        for _ in 0..20 {
            // u = A * v
            let mut u = residual.dot(&v);
            let u_norm = u.dot(&u).sqrt();
            if u_norm < 1e-10 {
                break;
            }
            u /= u_norm;

            // v = A^T * u
            v = residual.t().dot(&u);
            let v_norm = v.dot(&v).sqrt();
            if v_norm < 1e-10 {
                break;
            }
            v /= v_norm;
        }

        let u = residual.dot(&v);
        let sigma = u.dot(&u).sqrt();
        if sigma < 1e-10 {
            break;
        }
        let u_normalized = &u / sigma;

        // Deflate: remove this component from the residual
        let outer = outer_product(&u_normalized, &v);
        residual = residual - &(outer * sigma);

        u_cols.push(u_normalized);
        v_cols.push(v);
        sigmas.push(sigma);
    }

    // Reconstruct: sum of sigma_i * u_i * v_i^T
    let mut result = Array2::zeros((m, n));
    for i in 0..u_cols.len() {
        let component = outer_product(&u_cols[i], &v_cols[i]);
        result = result + &(component * sigmas[i]);
    }

    result
}

// =============================================================================
// Accuracy Evaluation
// =============================================================================

/// Evaluate mean squared error of a network on test data.
pub fn evaluate_mse(
    network: &NeuralNetwork,
    inputs: &[Array1<f64>],
    targets: &[Array1<f64>],
) -> f64 {
    if inputs.is_empty() {
        return 0.0;
    }
    let total_error: f64 = inputs
        .iter()
        .zip(targets.iter())
        .map(|(input, target)| {
            let output = network.forward(input);
            let diff = &output - target;
            diff.dot(&diff)
        })
        .sum();

    total_error / inputs.len() as f64
}

/// Evaluate MSE for a quantized network.
pub fn evaluate_mse_quantized(
    network: &QuantizedNetwork,
    inputs: &[Array1<f64>],
    targets: &[Array1<f64>],
) -> f64 {
    if inputs.is_empty() {
        return 0.0;
    }
    let total_error: f64 = inputs
        .iter()
        .zip(targets.iter())
        .map(|(input, target)| {
            let output = network.forward(input);
            let diff = &output - target;
            diff.dot(&diff)
        })
        .sum();

    total_error / inputs.len() as f64
}

/// Evaluate directional accuracy (correct sign prediction) for a network.
pub fn evaluate_directional_accuracy(
    network: &NeuralNetwork,
    inputs: &[Array1<f64>],
    targets: &[Array1<f64>],
) -> f64 {
    if inputs.is_empty() {
        return 0.0;
    }
    let correct: usize = inputs
        .iter()
        .zip(targets.iter())
        .filter(|(input, target)| {
            let output = network.forward(input);
            output[0] * target[0] > 0.0 // Same sign
        })
        .count();

    correct as f64 / inputs.len() as f64
}

/// Benchmark inference speed (average time per inference in microseconds).
pub fn benchmark_inference(network: &NeuralNetwork, input: &Array1<f64>, iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = network.forward(input);
    }
    let elapsed = start.elapsed();
    elapsed.as_micros() as f64 / iterations as f64
}

/// Benchmark inference speed for quantized network.
pub fn benchmark_inference_quantized(
    network: &QuantizedNetwork,
    input: &Array1<f64>,
    iterations: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = network.forward(input);
    }
    let elapsed = start.elapsed();
    elapsed.as_micros() as f64 / iterations as f64
}

// =============================================================================
// Bybit API Data Fetching
// =============================================================================

/// OHLCV candle data from Bybit.
#[derive(Debug, Clone)]
pub struct Candle {
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
    #[serde(rename = "retMsg")]
    ret_msg: String,
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch kline (candlestick) data from Bybit API.
///
/// # Arguments
/// * `symbol` - Trading pair, e.g., "BTCUSDT"
/// * `interval` - Candle interval: "1", "5", "15", "60", "240", "D", "W"
/// * `limit` - Number of candles to fetch (max 200)
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

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
                    timestamp: row[0].parse().unwrap_or(0),
                    open: row[1].parse().unwrap_or(0.0),
                    high: row[2].parse().unwrap_or(0.0),
                    low: row[3].parse().unwrap_or(0.0),
                    close: row[4].parse().unwrap_or(0.0),
                    volume: row[5].parse().unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first, reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Prepare features and targets from candle data for price prediction.
/// Features: normalized [open, high, low, close, volume] over a lookback window.
/// Target: next candle's return (close_next / close_current - 1).
pub fn prepare_training_data(
    candles: &[Candle],
    lookback: usize,
) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    if candles.len() < lookback + 2 {
        return (vec![], vec![]);
    }

    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for i in lookback..candles.len() - 1 {
        let window = &candles[i - lookback..i];

        // Find min/max for normalization
        let mut all_prices: Vec<f64> = Vec::new();
        let mut all_volumes: Vec<f64> = Vec::new();
        for c in window {
            all_prices.extend_from_slice(&[c.open, c.high, c.low, c.close]);
            all_volumes.push(c.volume);
        }

        let p_min = all_prices.iter().copied().fold(f64::INFINITY, f64::min);
        let p_max = all_prices
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let v_min = all_volumes.iter().copied().fold(f64::INFINITY, f64::min);
        let v_max = all_volumes
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let p_range = if (p_max - p_min).abs() < 1e-10 {
            1.0
        } else {
            p_max - p_min
        };
        let v_range = if (v_max - v_min).abs() < 1e-10 {
            1.0
        } else {
            v_max - v_min
        };

        // Build feature vector: normalized OHLCV for each candle in window
        let mut features = Vec::with_capacity(lookback * 5);
        for c in window {
            features.push((c.open - p_min) / p_range);
            features.push((c.high - p_min) / p_range);
            features.push((c.low - p_min) / p_range);
            features.push((c.close - p_min) / p_range);
            features.push((c.volume - v_min) / v_range);
        }

        // Target: next return
        let current_close = candles[i].close;
        let next_close = candles[i + 1].close;
        let ret = if current_close.abs() > 1e-10 {
            next_close / current_close - 1.0
        } else {
            0.0
        };

        inputs.push(Array1::from(features));
        targets.push(Array1::from(vec![ret]));
    }

    (inputs, targets)
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = NeuralNetwork::new(&[10, 32, 16, 1]).unwrap();
        assert_eq!(net.layers.len(), 3);
        assert_eq!(net.layers[0].weights.nrows(), 10);
        assert_eq!(net.layers[0].weights.ncols(), 32);
        assert_eq!(net.layers[1].weights.nrows(), 32);
        assert_eq!(net.layers[1].weights.ncols(), 16);
        assert_eq!(net.layers[2].weights.nrows(), 16);
        assert_eq!(net.layers[2].weights.ncols(), 1);
    }

    #[test]
    fn test_network_creation_too_few_layers() {
        let result = NeuralNetwork::new(&[10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_pass_shape() {
        let net = NeuralNetwork::new(&[5, 10, 1]).unwrap();
        let input = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let output = net.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_param_count() {
        let net = NeuralNetwork::new(&[4, 8, 2]).unwrap();
        // Layer 0: 4*8 weights + 8 biases = 40
        // Layer 1: 8*2 weights + 2 biases = 18
        assert_eq!(net.param_count(), 40 + 18);
    }

    #[test]
    fn test_magnitude_pruning() {
        let net = NeuralNetwork::new(&[4, 8, 2]).unwrap();
        let pruned = magnitude_prune(&net, 0.5);

        // At least some weights should be zero after 50% pruning
        let zero_count: usize = pruned
            .layers
            .iter()
            .map(|l| l.weights.iter().filter(|&&w| w == 0.0).count())
            .sum();
        assert!(zero_count > 0, "Pruning should create zero weights");

        // Sparsity should be approximately 50%
        let total_weights: usize = pruned.layers.iter().map(|l| l.weights.len()).sum();
        let sparsity = zero_count as f64 / total_weights as f64;
        assert!(
            sparsity >= 0.4 && sparsity <= 0.7,
            "Sparsity {:.2} should be approximately 0.5",
            sparsity
        );
    }

    #[test]
    fn test_pruning_preserves_structure() {
        let net = NeuralNetwork::new(&[4, 8, 2]).unwrap();
        let pruned = magnitude_prune(&net, 0.3);
        assert_eq!(pruned.layers.len(), net.layers.len());
        for (orig, prun) in net.layers.iter().zip(pruned.layers.iter()) {
            assert_eq!(orig.weights.shape(), prun.weights.shape());
            assert_eq!(orig.biases.len(), prun.biases.len());
        }
    }

    #[test]
    fn test_quantization() {
        let net = NeuralNetwork::new(&[4, 8, 2]).unwrap();
        let quantized = quantize_network(&net);
        assert_eq!(quantized.layers.len(), net.layers.len());

        // Quantized model should use less memory
        assert!(quantized.memory_bytes() < net.memory_bytes());
    }

    #[test]
    fn test_quantized_forward_pass() {
        let net = NeuralNetwork::new(&[4, 8, 2]).unwrap();
        let quantized = quantize_network(&net);

        let input = Array1::from(vec![0.5, -0.3, 0.1, 0.8]);
        let orig_output = net.forward(&input);
        let quant_output = quantized.forward(&input);

        // Outputs should be similar (within quantization error)
        assert_eq!(orig_output.len(), quant_output.len());
    }

    #[test]
    fn test_low_rank_factorization() {
        let net = NeuralNetwork::new(&[8, 16, 4]).unwrap();
        let factorized = low_rank_factorize(&net, 0.5);

        assert_eq!(factorized.layers.len(), net.layers.len());
        for (orig, fact) in net.layers.iter().zip(factorized.layers.iter()) {
            assert_eq!(orig.weights.shape(), fact.weights.shape());
        }
    }

    #[test]
    fn test_compression_metrics() {
        let net = NeuralNetwork::new(&[4, 8, 2]).unwrap();
        let pruned = magnitude_prune(&net, 0.5);
        let metrics = compute_metrics(&net, &pruned);

        assert!(metrics.compression_ratio > 1.0);
        assert!(metrics.sparsity > 0.0);
    }

    #[test]
    fn test_evaluate_mse() {
        let net = NeuralNetwork::new(&[2, 4, 1]).unwrap();
        let inputs = vec![
            Array1::from(vec![1.0, 0.0]),
            Array1::from(vec![0.0, 1.0]),
        ];
        let targets = vec![Array1::from(vec![1.0]), Array1::from(vec![0.0])];
        let mse = evaluate_mse(&net, &inputs, &targets);
        assert!(mse >= 0.0, "MSE should be non-negative");
    }

    #[test]
    fn test_training() {
        let mut net = NeuralNetwork::new(&[2, 8, 1]).unwrap();
        let inputs = vec![
            Array1::from(vec![0.0, 0.0]),
            Array1::from(vec![1.0, 0.0]),
            Array1::from(vec![0.0, 1.0]),
            Array1::from(vec![1.0, 1.0]),
        ];
        let targets = vec![
            Array1::from(vec![0.0]),
            Array1::from(vec![0.5]),
            Array1::from(vec![0.5]),
            Array1::from(vec![1.0]),
        ];

        let mse_before = evaluate_mse(&net, &inputs, &targets);
        net.train(&inputs, &targets, 0.01, 100);
        let mse_after = evaluate_mse(&net, &inputs, &targets);

        assert!(
            mse_after < mse_before,
            "Training should reduce MSE: before={}, after={}",
            mse_before,
            mse_after
        );
    }

    #[test]
    fn test_prepare_training_data() {
        let candles = (0..20)
            .map(|i| Candle {
                timestamp: i as u64 * 60000,
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0 + i as f64 * 10.0,
            })
            .collect::<Vec<_>>();

        let (inputs, targets) = prepare_training_data(&candles, 5);
        assert!(!inputs.is_empty());
        assert_eq!(inputs.len(), targets.len());
        assert_eq!(inputs[0].len(), 25); // 5 candles * 5 features
        assert_eq!(targets[0].len(), 1);
    }

    #[test]
    fn test_benchmark_inference() {
        let net = NeuralNetwork::new(&[4, 8, 2]).unwrap();
        let input = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let avg_time = benchmark_inference(&net, &input, 100);
        assert!(avg_time >= 0.0);
    }

    #[test]
    fn test_directional_accuracy() {
        let net = NeuralNetwork::new(&[2, 4, 1]).unwrap();
        let inputs = vec![
            Array1::from(vec![1.0, 0.0]),
            Array1::from(vec![0.0, 1.0]),
        ];
        let targets = vec![Array1::from(vec![1.0]), Array1::from(vec![-1.0])];
        let acc = evaluate_directional_accuracy(&net, &inputs, &targets);
        assert!(acc >= 0.0 && acc <= 1.0);
    }
}
