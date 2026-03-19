//! # Quantization Trading Models
//!
//! A library for quantizing neural network trading models from FP32 to INT8/INT4
//! precision with support for symmetric/asymmetric and per-tensor/per-channel schemes.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Quantization scheme definitions
// ---------------------------------------------------------------------------

/// Supported quantization schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationScheme {
    SymmetricPerTensor,
    AsymmetricPerTensor,
    SymmetricPerChannel,
    AsymmetricPerChannel,
}

/// Target precision for quantization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    Int8,
    Int4,
}

/// Quantization parameters (scale and zero-point)
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    pub scale: Vec<f32>,
    pub zero_point: Vec<i32>,
    pub precision: Precision,
}

/// A tensor quantized to INT8
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub shape: Vec<usize>,
    pub params: QuantizationParams,
}

/// A tensor quantized to INT4 (stored packed: two values per byte)
#[derive(Debug, Clone)]
pub struct QuantizedTensorInt4 {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub params: QuantizationParams,
    pub num_elements: usize,
}

// ---------------------------------------------------------------------------
// Scale and zero-point computation
// ---------------------------------------------------------------------------

/// Compute quantization params using the min-max method (per-tensor, symmetric)
pub fn compute_params_symmetric(data: &[f32], precision: Precision) -> QuantizationParams {
    let q_max = match precision {
        Precision::Int8 => 127.0_f32,
        Precision::Int4 => 7.0_f32,
    };

    let abs_max = data
        .iter()
        .fold(0.0_f32, |acc, &v| acc.max(v.abs()))
        .max(1e-8);

    let scale = abs_max / q_max;

    QuantizationParams {
        scale: vec![scale],
        zero_point: vec![0],
        precision,
    }
}

/// Compute quantization params using the min-max method (per-tensor, asymmetric)
pub fn compute_params_asymmetric(data: &[f32], precision: Precision) -> QuantizationParams {
    let (q_min, q_max) = match precision {
        Precision::Int8 => (-128.0_f32, 127.0_f32),
        Precision::Int4 => (-8.0_f32, 7.0_f32),
    };

    let x_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (x_max - x_min).max(1e-8);

    let scale = range / (q_max - q_min);
    let zero_point = (q_min - x_min / scale).round() as i32;

    QuantizationParams {
        scale: vec![scale],
        zero_point: vec![zero_point],
        precision,
    }
}

/// Compute per-channel symmetric quantization params for a 2D weight matrix.
/// Each row (output channel) gets its own scale factor.
pub fn compute_params_symmetric_per_channel(
    weights: &Array2<f32>,
    precision: Precision,
) -> QuantizationParams {
    let q_max = match precision {
        Precision::Int8 => 127.0_f32,
        Precision::Int4 => 7.0_f32,
    };

    let n_channels = weights.nrows();
    let mut scales = Vec::with_capacity(n_channels);
    let mut zero_points = Vec::with_capacity(n_channels);

    for i in 0..n_channels {
        let row = weights.row(i);
        let abs_max = row.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs())).max(1e-8);
        scales.push(abs_max / q_max);
        zero_points.push(0);
    }

    QuantizationParams {
        scale: scales,
        zero_point: zero_points,
        precision,
    }
}

/// Compute per-channel asymmetric quantization params for a 2D weight matrix.
pub fn compute_params_asymmetric_per_channel(
    weights: &Array2<f32>,
    precision: Precision,
) -> QuantizationParams {
    let (q_min, q_max) = match precision {
        Precision::Int8 => (-128.0_f32, 127.0_f32),
        Precision::Int4 => (-8.0_f32, 7.0_f32),
    };

    let n_channels = weights.nrows();
    let mut scales = Vec::with_capacity(n_channels);
    let mut zero_points = Vec::with_capacity(n_channels);

    for i in 0..n_channels {
        let row = weights.row(i);
        let x_min = row.iter().cloned().fold(f32::INFINITY, f32::min);
        let x_max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (x_max - x_min).max(1e-8);
        let scale = range / (q_max - q_min);
        let zp = (q_min - x_min / scale).round() as i32;
        scales.push(scale);
        zero_points.push(zp);
    }

    QuantizationParams {
        scale: scales,
        zero_point: zero_points,
        precision,
    }
}

/// Compute quantization params using the percentile method (symmetric, per-tensor).
/// `percentile` is in [0, 100], e.g. 1.0 means clip at the 1st and 99th percentile.
pub fn compute_params_percentile(
    data: &[f32],
    percentile: f32,
    precision: Precision,
) -> QuantizationParams {
    let q_max = match precision {
        Precision::Int8 => 127.0_f32,
        Precision::Int4 => 7.0_f32,
    };

    let mut sorted: Vec<f32> = data.iter().map(|v| v.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((100.0 - percentile) / 100.0 * sorted.len() as f32) as usize;
    let idx = idx.min(sorted.len() - 1);
    let clip_val = sorted[idx].max(1e-8);

    let scale = clip_val / q_max;

    QuantizationParams {
        scale: vec![scale],
        zero_point: vec![0],
        precision,
    }
}

// ---------------------------------------------------------------------------
// Quantize / Dequantize
// ---------------------------------------------------------------------------

/// Quantize a float slice to INT8 values
pub fn quantize_to_int8(data: &[f32], params: &QuantizationParams) -> Vec<i8> {
    let scale = params.scale[0];
    let zp = params.zero_point[0];

    data.iter()
        .map(|&x| {
            let q = (x / scale + zp as f32).round();
            q.clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Quantize a float slice to INT8 using per-channel params (for a 2D weight matrix).
pub fn quantize_to_int8_per_channel(
    weights: &Array2<f32>,
    params: &QuantizationParams,
) -> Vec<i8> {
    let (rows, cols) = weights.dim();
    let mut result = Vec::with_capacity(rows * cols);

    for i in 0..rows {
        let scale = params.scale[i];
        let zp = params.zero_point[i];
        for j in 0..cols {
            let q = (weights[[i, j]] / scale + zp as f32).round();
            result.push(q.clamp(-128.0, 127.0) as i8);
        }
    }

    result
}

/// Dequantize INT8 values back to float (per-tensor)
pub fn dequantize_int8(data: &[i8], params: &QuantizationParams) -> Vec<f32> {
    let scale = params.scale[0];
    let zp = params.zero_point[0];

    data.iter()
        .map(|&q| scale * (q as f32 - zp as f32))
        .collect()
}

/// Dequantize INT8 values back to float (per-channel, for 2D matrix)
pub fn dequantize_int8_per_channel(
    data: &[i8],
    params: &QuantizationParams,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        let scale = params.scale[i];
        let zp = params.zero_point[i];
        for j in 0..cols {
            let q = data[i * cols + j];
            result.push(scale * (q as f32 - zp as f32));
        }
    }
    result
}

/// Quantize a float slice to INT4 (packed: two values per byte)
pub fn quantize_to_int4(data: &[f32], params: &QuantizationParams) -> Vec<u8> {
    let scale = params.scale[0];
    let zp = params.zero_point[0];

    let mut packed = Vec::with_capacity((data.len() + 1) / 2);
    let mut i = 0;
    while i < data.len() {
        let q_lo = ((data[i] / scale + zp as f32).round().clamp(-8.0, 7.0) as i8) & 0x0F;
        let q_hi = if i + 1 < data.len() {
            ((data[i + 1] / scale + zp as f32).round().clamp(-8.0, 7.0) as i8) & 0x0F
        } else {
            0
        };
        packed.push((q_hi as u8) << 4 | (q_lo as u8 & 0x0F));
        i += 2;
    }
    packed
}

/// Dequantize INT4 packed values back to float
pub fn dequantize_int4(packed: &[u8], num_elements: usize, params: &QuantizationParams) -> Vec<f32> {
    let scale = params.scale[0];
    let zp = params.zero_point[0];

    let mut result = Vec::with_capacity(num_elements);
    for (idx, &byte) in packed.iter().enumerate() {
        // Low nibble
        let lo = (byte & 0x0F) as i8;
        let lo = if lo & 0x08 != 0 { lo | !0x0F_u8 as i8 } else { lo }; // sign-extend
        result.push(scale * (lo as f32 - zp as f32));
        if idx * 2 + 1 >= num_elements {
            break;
        }
        // High nibble
        let hi = ((byte >> 4) & 0x0F) as i8;
        let hi = if hi & 0x08 != 0 { hi | !0x0F_u8 as i8 } else { hi }; // sign-extend
        result.push(scale * (hi as f32 - zp as f32));
    }
    result.truncate(num_elements);
    result
}

/// Create a QuantizedTensor from float data
pub fn quantize_tensor(
    data: &[f32],
    shape: Vec<usize>,
    scheme: QuantizationScheme,
    precision: Precision,
) -> QuantizedTensor {
    let params = match scheme {
        QuantizationScheme::SymmetricPerTensor => compute_params_symmetric(data, precision),
        QuantizationScheme::AsymmetricPerTensor => compute_params_asymmetric(data, precision),
        _ => compute_params_symmetric(data, precision), // fallback for non-2D
    };

    let quantized_data = match precision {
        Precision::Int8 => quantize_to_int8(data, &params),
        Precision::Int4 => {
            // For Int4 we still store as i8 in QuantizedTensor for simplicity
            let packed = quantize_to_int4(data, &params);
            // Unpack for uniform interface
            let unpacked = dequantize_int4(&packed, data.len(), &params);
            // Re-quantize to int8 range but with int4 precision loss
            let int4_params = compute_params_symmetric(data, Precision::Int8);
            quantize_to_int8(&unpacked, &int4_params)
        }
    };

    QuantizedTensor {
        data: quantized_data,
        shape,
        params,
    }
}

// ---------------------------------------------------------------------------
// Quantized matrix multiplication
// ---------------------------------------------------------------------------

/// Perform matrix-vector multiplication using INT8 arithmetic.
/// weights: [out_features x in_features] stored row-major as i8
/// input: [in_features] as i8
/// Returns: [out_features] as f32 (dequantized output)
pub fn quantized_matmul_int8(
    weights: &[i8],
    weight_params: &QuantizationParams,
    input: &[i8],
    input_params: &QuantizationParams,
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    let w_scale = weight_params.scale[0];
    let w_zp = weight_params.zero_point[0];
    let i_scale = input_params.scale[0];
    let i_zp = input_params.zero_point[0];

    let output_scale = w_scale * i_scale;
    let mut output = Vec::with_capacity(out_features);

    for o in 0..out_features {
        let mut acc: i32 = 0;
        for i in 0..in_features {
            let w = weights[o * in_features + i] as i32 - w_zp;
            let x = input[i] as i32 - i_zp;
            acc += w * x;
        }
        output.push(output_scale * acc as f32);
    }

    output
}

/// Per-channel quantized matrix-vector multiplication
pub fn quantized_matmul_int8_per_channel(
    weights: &[i8],
    weight_params: &QuantizationParams,
    input: &[i8],
    input_params: &QuantizationParams,
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    let i_scale = input_params.scale[0];
    let i_zp = input_params.zero_point[0];

    let mut output = Vec::with_capacity(out_features);

    for o in 0..out_features {
        let w_scale = weight_params.scale[o];
        let w_zp = weight_params.zero_point[o];
        let output_scale = w_scale * i_scale;

        let mut acc: i32 = 0;
        for i in 0..in_features {
            let w = weights[o * in_features + i] as i32 - w_zp;
            let x = input[i] as i32 - i_zp;
            acc += w * x;
        }
        output.push(output_scale * acc as f32);
    }

    output
}

// ---------------------------------------------------------------------------
// Quantization error metrics
// ---------------------------------------------------------------------------

/// Compute Mean Squared Error between original and quantized values
pub fn compute_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    assert_eq!(original.len(), reconstructed.len());
    let n = original.len() as f32;
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / n
}

/// Compute Signal-to-Noise Ratio in dB
pub fn compute_snr(original: &[f32], reconstructed: &[f32]) -> f32 {
    let signal_power = original.iter().map(|&x| x.powi(2)).sum::<f32>() / original.len() as f32;
    let mse = compute_mse(original, reconstructed);
    if mse < 1e-10 {
        return f32::INFINITY;
    }
    10.0 * (signal_power / mse).log10()
}

/// Compute max absolute error
pub fn compute_max_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

// ---------------------------------------------------------------------------
// Neural network layers
// ---------------------------------------------------------------------------

/// A fully-connected (linear) layer in FP32
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearLayer {
    /// Create a new linear layer with random Xavier initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / (in_features + out_features) as f64).sqrt() as f32;

        let weights = Array2::from_shape_fn((out_features, in_features), |_| {
            rng.gen_range(-std_dev..std_dev)
        });
        let bias = Array1::zeros(out_features);

        Self {
            weights,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass: output = weights @ input + bias
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.out_features);
        for o in 0..self.out_features {
            let mut sum = self.bias[o];
            for i in 0..self.in_features {
                sum += self.weights[[o, i]] * input[i];
            }
            output.push(sum);
        }
        output
    }

    /// Get model size in bytes (FP32)
    pub fn size_bytes(&self) -> usize {
        (self.in_features * self.out_features + self.out_features) * 4
    }
}

/// A quantized linear layer operating in INT8
#[derive(Debug, Clone)]
pub struct QuantizedLinearLayer {
    pub weights_int8: Vec<i8>,
    pub weight_params: QuantizationParams,
    pub bias: Array1<f32>, // bias stays FP32
    pub in_features: usize,
    pub out_features: usize,
    pub scheme: QuantizationScheme,
}

impl QuantizedLinearLayer {
    /// Create a quantized layer from an FP32 layer
    pub fn from_fp32(layer: &LinearLayer, scheme: QuantizationScheme) -> Self {
        let (weights_int8, weight_params) = match scheme {
            QuantizationScheme::SymmetricPerTensor => {
                let flat: Vec<f32> = layer.weights.iter().cloned().collect();
                let params = compute_params_symmetric(&flat, Precision::Int8);
                let q = quantize_to_int8(&flat, &params);
                (q, params)
            }
            QuantizationScheme::AsymmetricPerTensor => {
                let flat: Vec<f32> = layer.weights.iter().cloned().collect();
                let params = compute_params_asymmetric(&flat, Precision::Int8);
                let q = quantize_to_int8(&flat, &params);
                (q, params)
            }
            QuantizationScheme::SymmetricPerChannel => {
                let params =
                    compute_params_symmetric_per_channel(&layer.weights, Precision::Int8);
                let q = quantize_to_int8_per_channel(&layer.weights, &params);
                (q, params)
            }
            QuantizationScheme::AsymmetricPerChannel => {
                let params =
                    compute_params_asymmetric_per_channel(&layer.weights, Precision::Int8);
                let q = quantize_to_int8_per_channel(&layer.weights, &params);
                (q, params)
            }
        };

        Self {
            weights_int8,
            weight_params,
            bias: layer.bias.clone(),
            in_features: layer.in_features,
            out_features: layer.out_features,
            scheme,
        }
    }

    /// Forward pass using quantized integer arithmetic
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Quantize input on-the-fly (dynamic quantization for activations)
        let input_params = compute_params_symmetric(input, Precision::Int8);
        let input_int8 = quantize_to_int8(input, &input_params);

        let raw_output = match self.scheme {
            QuantizationScheme::SymmetricPerChannel | QuantizationScheme::AsymmetricPerChannel => {
                quantized_matmul_int8_per_channel(
                    &self.weights_int8,
                    &self.weight_params,
                    &input_int8,
                    &input_params,
                    self.out_features,
                    self.in_features,
                )
            }
            _ => quantized_matmul_int8(
                &self.weights_int8,
                &self.weight_params,
                &input_int8,
                &input_params,
                self.out_features,
                self.in_features,
            ),
        };

        // Add FP32 bias
        raw_output
            .iter()
            .zip(self.bias.iter())
            .map(|(&o, &b)| o + b)
            .collect()
    }

    /// Get model size in bytes (INT8 weights + FP32 bias + params)
    pub fn size_bytes(&self) -> usize {
        self.weights_int8.len() + self.out_features * 4 + self.weight_params.scale.len() * 8
    }
}

// ---------------------------------------------------------------------------
// Trading model (multi-layer perceptron)
// ---------------------------------------------------------------------------

/// A simple MLP trading model in FP32
#[derive(Debug, Clone)]
pub struct TradingModel {
    pub layers: Vec<LinearLayer>,
}

impl TradingModel {
    /// Create a new trading model with given layer sizes
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(LinearLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        Self { layers }
    }

    /// Forward pass with ReLU activations between layers
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut x = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            // Apply ReLU for all layers except the last
            if i < self.layers.len() - 1 {
                x.iter_mut().for_each(|v| *v = v.max(0.0));
            }
        }
        x
    }

    /// Simple training step using MSE loss and gradient-free optimization.
    /// Uses a simple perturbation-based method for demonstration purposes.
    pub fn train(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], epochs: usize, lr: f32) {
        let mut rng = rand::thread_rng();

        for _epoch in 0..epochs {
            let n_layers = self.layers.len();
            for layer_idx in 0..n_layers {
                let (rows, cols) = self.layers[layer_idx].weights.dim();
                for r in 0..rows {
                    for c in 0..cols {
                        // Compute current loss
                        let current_loss: f32 = inputs
                            .iter()
                            .zip(targets.iter())
                            .map(|(inp, tgt)| {
                                let pred = self.forward(inp);
                                pred.iter()
                                    .zip(tgt.iter())
                                    .map(|(p, t)| (p - t).powi(2))
                                    .sum::<f32>()
                            })
                            .sum::<f32>()
                            / inputs.len() as f32;

                        // Perturb weight
                        let delta = rng.gen_range(-0.01_f32..0.01);
                        self.layers[layer_idx].weights[[r, c]] += delta;

                        // Compute new loss
                        let new_loss: f32 = inputs
                            .iter()
                            .zip(targets.iter())
                            .map(|(inp, tgt)| {
                                let pred = self.forward(inp);
                                pred.iter()
                                    .zip(tgt.iter())
                                    .map(|(p, t)| (p - t).powi(2))
                                    .sum::<f32>()
                            })
                            .sum::<f32>()
                            / inputs.len() as f32;

                        // Approximate gradient
                        let grad = (new_loss - current_loss) / delta;
                        self.layers[layer_idx].weights[[r, c]] -= delta; // undo perturbation
                        self.layers[layer_idx].weights[[r, c]] -= lr * grad;
                    }
                }
            }
        }
    }

    /// Get total model size in bytes
    pub fn size_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.size_bytes()).sum()
    }
}

/// A quantized version of the trading model
#[derive(Debug, Clone)]
pub struct QuantizedTradingModel {
    pub layers: Vec<QuantizedLinearLayer>,
}

impl QuantizedTradingModel {
    /// Create a quantized model from an FP32 model
    pub fn from_fp32(model: &TradingModel, scheme: QuantizationScheme) -> Self {
        let layers = model
            .layers
            .iter()
            .map(|l| QuantizedLinearLayer::from_fp32(l, scheme))
            .collect();
        Self { layers }
    }

    /// Forward pass with ReLU activations between layers
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut x = input.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < self.layers.len() - 1 {
                x.iter_mut().for_each(|v| *v = v.max(0.0));
            }
        }
        x
    }

    /// Get total model size in bytes
    pub fn size_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.size_bytes()).sum()
    }
}

// ---------------------------------------------------------------------------
// Post-training quantization pipeline
// ---------------------------------------------------------------------------

/// Post-training quantization pipeline: calibrate and quantize an FP32 model
pub fn post_training_quantization(
    model: &TradingModel,
    calibration_data: &[Vec<f32>],
    scheme: QuantizationScheme,
) -> QuantizedTradingModel {
    // Run calibration data through the model to collect activation statistics
    // (in a full implementation, we'd collect per-layer statistics)
    let _activations: Vec<Vec<f32>> = calibration_data
        .iter()
        .map(|input| model.forward(input))
        .collect();

    // Quantize the model
    QuantizedTradingModel::from_fp32(model, scheme)
}

// ---------------------------------------------------------------------------
// INT4 Trading Model (simulated with reduced precision)
// ---------------------------------------------------------------------------

/// A trading model quantized to INT4 precision
#[derive(Debug, Clone)]
pub struct Int4TradingModel {
    pub weights_packed: Vec<Vec<u8>>,
    pub weight_params: Vec<QuantizationParams>,
    pub biases: Vec<Array1<f32>>,
    pub layer_dims: Vec<(usize, usize)>, // (in, out) per layer
}

impl Int4TradingModel {
    /// Create an INT4 model from FP32
    pub fn from_fp32(model: &TradingModel) -> Self {
        let mut weights_packed = Vec::new();
        let mut weight_params = Vec::new();
        let mut biases = Vec::new();
        let mut layer_dims = Vec::new();

        for layer in &model.layers {
            let flat: Vec<f32> = layer.weights.iter().cloned().collect();
            let params = compute_params_symmetric(&flat, Precision::Int4);
            let packed = quantize_to_int4(&flat, &params);
            weights_packed.push(packed);
            weight_params.push(params);
            biases.push(layer.bias.clone());
            layer_dims.push((layer.in_features, layer.out_features));
        }

        Self {
            weights_packed,
            weight_params,
            biases,
            layer_dims,
        }
    }

    /// Forward pass: dequantize weights on-the-fly and compute
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut x = input.to_vec();
        for (i, (packed, params)) in self
            .weights_packed
            .iter()
            .zip(self.weight_params.iter())
            .enumerate()
        {
            let (in_f, out_f) = self.layer_dims[i];
            let num_elements = in_f * out_f;
            let weights_fp32 = dequantize_int4(packed, num_elements, params);

            let mut output = Vec::with_capacity(out_f);
            for o in 0..out_f {
                let mut sum = self.biases[i][o];
                for j in 0..in_f {
                    sum += weights_fp32[o * in_f + j] * x[j];
                }
                output.push(sum);
            }
            x = output;

            // ReLU for all but last layer
            if i < self.layer_dims.len() - 1 {
                x.iter_mut().for_each(|v| *v = v.max(0.0));
            }
        }
        x
    }

    /// Get total model size in bytes
    pub fn size_bytes(&self) -> usize {
        let weight_size: usize = self.weights_packed.iter().map(|w| w.len()).sum();
        let bias_size: usize = self.biases.iter().map(|b| b.len() * 4).sum();
        let param_size: usize = self.weight_params.iter().map(|p| p.scale.len() * 8).sum();
        weight_size + bias_size + param_size
    }
}

// ---------------------------------------------------------------------------
// Bybit API integration
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(alias = "retCode")]
    pub ret_code: i32,
    #[serde(alias = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// OHLCV candle data
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
}

/// Fetch kline data from Bybit API
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
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

/// Convert candles to feature vectors for the trading model.
/// Features: [return, high_low_range, volume_change]
pub fn candles_to_features(candles: &[Candle]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut features = Vec::new();
    let mut targets = Vec::new();

    // Compute volume moving average
    let vol_ma_period = 10;

    for i in vol_ma_period..candles.len() - 1 {
        let c = &candles[i];

        // Features
        let ret = (c.close - c.open) / c.open.max(1e-8);
        let hl_range = (c.high - c.low) / c.open.max(1e-8);
        let vol_ma: f32 = candles[i - vol_ma_period..i]
            .iter()
            .map(|c| c.volume)
            .sum::<f32>()
            / vol_ma_period as f32;
        let vol_change = (c.volume - vol_ma) / vol_ma.max(1e-8);

        features.push(vec![ret, hl_range, vol_change]);

        // Target: next candle return
        let next_ret = (candles[i + 1].close - candles[i + 1].open) / candles[i + 1].open.max(1e-8);
        targets.push(vec![next_ret]);
    }

    (features, targets)
}

/// Normalize features to [-1, 1] range
pub fn normalize_features(features: &mut [Vec<f32>]) {
    if features.is_empty() {
        return;
    }
    let n_features = features[0].len();

    for f in 0..n_features {
        let vals: Vec<f32> = features.iter().map(|row| row[f]).collect();
        let min_val = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-8);

        for row in features.iter_mut() {
            row[f] = 2.0 * (row[f] - min_val) / range - 1.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_quantization_int8() {
        let data = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.75];
        let params = compute_params_symmetric(&data, Precision::Int8);
        let quantized = quantize_to_int8(&data, &params);
        let dequantized = dequantize_int8(&quantized, &params);

        let mse = compute_mse(&data, &dequantized);
        assert!(mse < 0.001, "MSE too high: {}", mse);
    }

    #[test]
    fn test_asymmetric_quantization_int8() {
        let data = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let params = compute_params_asymmetric(&data, Precision::Int8);
        let quantized = quantize_to_int8(&data, &params);
        let dequantized = dequantize_int8(&quantized, &params);

        let mse = compute_mse(&data, &dequantized);
        assert!(mse < 0.001, "MSE too high: {}", mse);
    }

    #[test]
    fn test_int4_quantization() {
        let data = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let params = compute_params_symmetric(&data, Precision::Int4);
        let packed = quantize_to_int4(&data, &params);
        let dequantized = dequantize_int4(&packed, data.len(), &params);

        // INT4 has much lower precision, allow larger error
        let mse = compute_mse(&data, &dequantized);
        assert!(mse < 0.1, "INT4 MSE too high: {}", mse);
    }

    #[test]
    fn test_per_channel_quantization() {
        let weights = Array2::from_shape_vec(
            (2, 3),
            vec![0.1, 0.2, 0.3, -1.0, -2.0, -3.0],
        )
        .unwrap();

        let params = compute_params_symmetric_per_channel(&weights, Precision::Int8);
        assert_eq!(params.scale.len(), 2);
        assert!(params.scale[0] < params.scale[1]); // second row has larger values
    }

    #[test]
    fn test_percentile_params() {
        let mut data: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        data.push(100.0); // outlier

        let params_minmax = compute_params_symmetric(&data, Precision::Int8);
        let params_pct = compute_params_percentile(&data, 1.0, Precision::Int8);

        // Percentile should give smaller scale (ignoring outlier)
        assert!(params_pct.scale[0] < params_minmax.scale[0]);
    }

    #[test]
    fn test_quantized_matmul() {
        let weights_fp32 = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2x2
        let input_fp32 = vec![0.5_f32, 0.5];

        let w_params = compute_params_symmetric(&weights_fp32, Precision::Int8);
        let i_params = compute_params_symmetric(&input_fp32, Precision::Int8);
        let w_q = quantize_to_int8(&weights_fp32, &w_params);
        let i_q = quantize_to_int8(&input_fp32, &i_params);

        let result = quantized_matmul_int8(&w_q, &w_params, &i_q, &i_params, 2, 2);

        // Expected: [1.0*0.5 + 2.0*0.5, 3.0*0.5 + 4.0*0.5] = [1.5, 3.5]
        assert!((result[0] - 1.5).abs() < 0.2, "Got {}", result[0]);
        assert!((result[1] - 3.5).abs() < 0.2, "Got {}", result[1]);
    }

    #[test]
    fn test_snr_computation() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let noisy = vec![1.01, 2.02, 3.01, 4.03, 5.01];
        let snr = compute_snr(&original, &noisy);
        assert!(snr > 30.0, "SNR should be high for small noise: {}", snr);
    }

    #[test]
    fn test_linear_layer_forward() {
        let layer = LinearLayer::new(3, 2);
        let input = vec![1.0, 0.5, -0.5];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_quantized_layer_matches_fp32() {
        let layer = LinearLayer::new(4, 3);
        let input = vec![0.5, -0.3, 0.8, -0.1];

        let fp32_output = layer.forward(&input);
        let q_layer = QuantizedLinearLayer::from_fp32(&layer, QuantizationScheme::SymmetricPerTensor);
        let q_output = q_layer.forward(&input);

        // Quantized output should be close to FP32
        for (fp, q) in fp32_output.iter().zip(q_output.iter()) {
            assert!(
                (fp - q).abs() < 0.5,
                "FP32={} vs Q={} differ too much",
                fp,
                q
            );
        }
    }

    #[test]
    fn test_trading_model_forward() {
        let model = TradingModel::new(&[3, 8, 4, 1]);
        let input = vec![0.1, -0.2, 0.3];
        let output = model.forward(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_quantized_trading_model() {
        let model = TradingModel::new(&[3, 8, 4, 1]);
        let q_model =
            QuantizedTradingModel::from_fp32(&model, QuantizationScheme::SymmetricPerTensor);

        let input = vec![0.1, -0.2, 0.3];
        let fp32_out = model.forward(&input);
        let q_out = q_model.forward(&input);

        assert_eq!(fp32_out.len(), q_out.len());
        // Model size should be smaller
        assert!(q_model.size_bytes() < model.size_bytes());
    }

    #[test]
    fn test_int4_trading_model() {
        let model = TradingModel::new(&[3, 8, 4, 1]);
        let int4_model = Int4TradingModel::from_fp32(&model);

        let input = vec![0.1, -0.2, 0.3];
        let output = int4_model.forward(&input);
        assert_eq!(output.len(), 1);

        // INT4 model should be significantly smaller
        assert!(int4_model.size_bytes() < model.size_bytes());
    }

    #[test]
    fn test_candle_feature_extraction() {
        let candles: Vec<Candle> = (0..20)
            .map(|i| Candle {
                timestamp: i as u64,
                open: 100.0 + i as f32,
                high: 102.0 + i as f32,
                low: 99.0 + i as f32,
                close: 101.0 + i as f32,
                volume: 1000.0 + (i as f32 * 10.0),
            })
            .collect();

        let (features, targets) = candles_to_features(&candles);
        assert!(!features.is_empty());
        assert_eq!(features.len(), targets.len());
        assert_eq!(features[0].len(), 3); // 3 features
        assert_eq!(targets[0].len(), 1); // 1 target
    }

    #[test]
    fn test_normalize_features() {
        let mut features = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
        ];
        normalize_features(&mut features);

        // After normalization, values should be in [-1, 1]
        for row in &features {
            for &v in row {
                assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6, "Value {} out of range", v);
            }
        }
    }

    #[test]
    fn test_max_error() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.2, 2.9];
        let max_err = compute_max_error(&a, &b);
        assert!((max_err - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_post_training_quantization_pipeline() {
        let model = TradingModel::new(&[3, 8, 4, 1]);
        let calibration: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![0.1, -0.2, 0.3])
            .collect();

        let q_model = post_training_quantization(
            &model,
            &calibration,
            QuantizationScheme::SymmetricPerChannel,
        );

        let input = vec![0.5, -0.1, 0.7];
        let output = q_model.forward(&input);
        assert_eq!(output.len(), 1);
    }
}
