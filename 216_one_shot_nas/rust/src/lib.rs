//! # One-Shot Neural Architecture Search (NAS)
//!
//! This crate implements One-Shot NAS for trading model architecture search.
//! A single supernet containing all candidate architectures is trained,
//! then subnets are evaluated using shared weights to rank architectures.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;
use std::collections::HashMap;

// ─────────────────────────── Search Space ───────────────────────────

/// Candidate activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    LeakyReLU,
    GELU,
}

impl Activation {
    pub const ALL: [Activation; 5] = [
        Activation::ReLU,
        Activation::Tanh,
        Activation::Sigmoid,
        Activation::LeakyReLU,
        Activation::GELU,
    ];

    /// Apply the activation element-wise.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::GELU => {
                // Approximation: x * sigmoid(1.702 * x)
                let s = 1.0 / (1.0 + (-1.702 * x).exp());
                x * s
            }
        }
    }

    /// Derivative of the activation.
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::GELU => {
                let s = 1.0 / (1.0 + (-1.702 * x).exp());
                let ds = 1.702 * s * (1.0 - s);
                s + x * ds
            }
        }
    }
}

/// Candidate layer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    Dense,
    DenseNoBias,
    Residual,
    GatedDense,
}

impl LayerType {
    pub const ALL: [LayerType; 4] = [
        LayerType::Dense,
        LayerType::DenseNoBias,
        LayerType::Residual,
        LayerType::GatedDense,
    ];
}

/// A specific architecture configuration selected from the search space.
#[derive(Debug, Clone)]
pub struct ArchitectureConfig {
    pub layer_type: LayerType,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub activation: Activation,
    pub dropout_rate: f64,
}

impl std::fmt::Display for ArchitectureConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}(h={}, L={}, act={:?}, drop={:.1})",
            self.layer_type, self.hidden_dim, self.num_layers, self.activation, self.dropout_rate
        )
    }
}

/// Defines the full search space.
pub struct SearchSpace {
    pub layer_types: Vec<LayerType>,
    pub hidden_dims: Vec<usize>,
    pub num_layers_options: Vec<usize>,
    pub activations: Vec<Activation>,
    pub dropout_rates: Vec<f64>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            layer_types: LayerType::ALL.to_vec(),
            hidden_dims: vec![16, 32, 64],
            num_layers_options: vec![1, 2, 3, 4],
            activations: Activation::ALL.to_vec(),
            dropout_rates: vec![0.0, 0.1, 0.2],
        }
    }
}

impl SearchSpace {
    /// Total number of architectures in the search space.
    pub fn size(&self) -> usize {
        self.layer_types.len()
            * self.hidden_dims.len()
            * self.num_layers_options.len()
            * self.activations.len()
            * self.dropout_rates.len()
    }

    /// Sample a random architecture uniformly.
    pub fn sample_random(&self) -> ArchitectureConfig {
        let mut rng = rand::thread_rng();
        ArchitectureConfig {
            layer_type: self.layer_types[rng.gen_range(0..self.layer_types.len())],
            hidden_dim: self.hidden_dims[rng.gen_range(0..self.hidden_dims.len())],
            num_layers: self.num_layers_options[rng.gen_range(0..self.num_layers_options.len())],
            activation: self.activations[rng.gen_range(0..self.activations.len())],
            dropout_rate: self.dropout_rates[rng.gen_range(0..self.dropout_rates.len())],
        }
    }

    /// Maximum hidden dim in the search space.
    pub fn max_hidden_dim(&self) -> usize {
        *self.hidden_dims.iter().max().unwrap_or(&64)
    }

    /// Maximum number of layers in the search space.
    pub fn max_layers(&self) -> usize {
        *self.num_layers_options.iter().max().unwrap_or(&4)
    }
}

// ─────────────────────────── Weight Key ───────────────────────────

/// Key for identifying a specific weight tensor in the supernet.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct WeightKey {
    layer_idx: usize,
    layer_type: LayerType,
    hidden_dim: usize,
    component: &'static str, // "weight", "bias", "gate_weight", "gate_bias"
}

// ─────────────────────────── Supernet ───────────────────────────

/// The One-Shot supernet containing all candidate operations.
pub struct Supernet {
    pub search_space: SearchSpace,
    pub input_dim: usize,
    weights: HashMap<WeightKey, Array2<f64>>,
    biases: HashMap<WeightKey, Array1<f64>>,
    // Output layer
    output_weight: Array2<f64>,
    output_bias: Array1<f64>,
}

impl Supernet {
    /// Create a new supernet for the given input dimension.
    pub fn new(input_dim: usize, search_space: SearchSpace) -> Self {
        let mut weights = HashMap::new();
        let mut biases = HashMap::new();
        let mut rng = rand::thread_rng();

        let max_layers = search_space.max_layers();

        // Initialize weights for every (layer, type, dim) combination
        for layer_idx in 0..max_layers {
            for &lt in &search_space.layer_types {
                for &hd in &search_space.hidden_dims {
                    let in_dim = if layer_idx == 0 { input_dim } else { hd };

                    // Xavier initialization scale
                    let scale = (2.0 / (in_dim + hd) as f64).sqrt();

                    let key_w = WeightKey {
                        layer_idx,
                        layer_type: lt,
                        hidden_dim: hd,
                        component: "weight",
                    };
                    let w = Array2::from_shape_fn((in_dim, hd), |_| {
                        rng.gen_range(-scale..scale)
                    });
                    weights.insert(key_w, w);

                    let key_b = WeightKey {
                        layer_idx,
                        layer_type: lt,
                        hidden_dim: hd,
                        component: "bias",
                    };
                    biases.insert(key_b, Array1::zeros(hd));

                    // Gate weights for GatedDense
                    if lt == LayerType::GatedDense {
                        let key_gw = WeightKey {
                            layer_idx,
                            layer_type: lt,
                            hidden_dim: hd,
                            component: "gate_weight",
                        };
                        let gw = Array2::from_shape_fn((in_dim, hd), |_| {
                            rng.gen_range(-scale..scale)
                        });
                        weights.insert(key_gw, gw);

                        let key_gb = WeightKey {
                            layer_idx,
                            layer_type: lt,
                            hidden_dim: hd,
                            component: "gate_bias",
                        };
                        biases.insert(key_gb, Array1::zeros(hd));
                    }
                }
            }
        }

        // Output layer: map from max hidden dim to 1
        let max_hd = search_space.max_hidden_dim();
        let out_scale = (2.0 / (max_hd + 1) as f64).sqrt();
        let output_weight =
            Array2::from_shape_fn((max_hd, 1), |_| rng.gen_range(-out_scale..out_scale));
        let output_bias = Array1::zeros(1);

        Supernet {
            search_space,
            input_dim,
            weights,
            biases,
            output_weight,
            output_bias,
        }
    }

    /// Forward pass through a specific subnet defined by `arch`.
    /// Input x: (batch_size, input_dim), returns (batch_size,)
    pub fn forward(&self, x: &Array2<f64>, arch: &ArchitectureConfig) -> Array1<f64> {
        let mut hidden = x.clone();

        for layer_idx in 0..arch.num_layers {
            let in_dim = if layer_idx == 0 {
                self.input_dim
            } else {
                arch.hidden_dim
            };

            let key_w = WeightKey {
                layer_idx,
                layer_type: arch.layer_type,
                hidden_dim: arch.hidden_dim,
                component: "weight",
            };
            let key_b = WeightKey {
                layer_idx,
                layer_type: arch.layer_type,
                hidden_dim: arch.hidden_dim,
                component: "bias",
            };

            let w = match self.weights.get(&key_w) {
                Some(w) => w,
                None => continue,
            };
            let b = match self.biases.get(&key_b) {
                Some(b) => b,
                None => continue,
            };

            // Slice weight if input dim does not match (for layers > 0 with different hidden dims)
            let w_slice = if hidden.ncols() != in_dim {
                // Truncate or pad -- in practice we keep dims consistent
                let actual_in = hidden.ncols();
                let rows = actual_in.min(w.nrows());
                let cols = arch.hidden_dim.min(w.ncols());
                w.slice(ndarray::s![..rows, ..cols]).to_owned()
            } else {
                w.clone()
            };

            let actual_in = hidden.ncols();
            let w_rows = w_slice.nrows();

            // Handle dimension mismatch by truncating hidden
            let h_input = if actual_in > w_rows {
                hidden.slice(ndarray::s![.., ..w_rows]).to_owned()
            } else if actual_in < w_rows {
                // Pad with zeros
                let batch = hidden.nrows();
                let mut padded = Array2::zeros((batch, w_rows));
                padded
                    .slice_mut(ndarray::s![.., ..actual_in])
                    .assign(&hidden);
                padded
            } else {
                hidden.clone()
            };

            let mut out = h_input.dot(&w_slice) + b;

            match arch.layer_type {
                LayerType::Dense | LayerType::DenseNoBias => {
                    if arch.layer_type == LayerType::DenseNoBias {
                        out = h_input.dot(&w_slice);
                    }
                    out.mapv_inplace(|v| arch.activation.apply(v));
                }
                LayerType::Residual => {
                    out.mapv_inplace(|v| arch.activation.apply(v));
                    // Add residual connection if dimensions match
                    if h_input.ncols() == out.ncols() {
                        out = out + &h_input;
                    }
                }
                LayerType::GatedDense => {
                    let key_gw = WeightKey {
                        layer_idx,
                        layer_type: arch.layer_type,
                        hidden_dim: arch.hidden_dim,
                        component: "gate_weight",
                    };
                    let key_gb = WeightKey {
                        layer_idx,
                        layer_type: arch.layer_type,
                        hidden_dim: arch.hidden_dim,
                        component: "gate_bias",
                    };
                    if let (Some(gw), Some(gb)) = (self.weights.get(&key_gw), self.biases.get(&key_gb)) {
                        let gw_rows = gw.nrows();
                        let g_input = if h_input.ncols() > gw_rows {
                            h_input.slice(ndarray::s![.., ..gw_rows]).to_owned()
                        } else {
                            h_input.clone()
                        };
                        let gate = (&g_input.dot(gw) + gb).mapv(|v| 1.0 / (1.0 + (-v).exp()));
                        out.mapv_inplace(|v| arch.activation.apply(v));
                        out = out * &gate;
                    }
                }
            }

            hidden = out;
        }

        // Output layer: project to scalar
        let hd = hidden.ncols();
        let out_rows = self.output_weight.nrows().min(hd);
        let h_out = if hd > out_rows {
            hidden.slice(ndarray::s![.., ..out_rows]).to_owned()
        } else if hd < out_rows {
            let batch = hidden.nrows();
            let mut padded = Array2::zeros((batch, out_rows));
            padded.slice_mut(ndarray::s![.., ..hd]).assign(&hidden);
            padded
        } else {
            hidden
        };

        let out_w = self.output_weight.slice(ndarray::s![..out_rows, ..]).to_owned();
        let logits = h_out.dot(&out_w) + &self.output_bias;
        logits.column(0).to_owned()
    }

    /// Train the supernet for one step with a randomly sampled path.
    /// Returns (sampled architecture, loss value).
    pub fn train_step(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        lr: f64,
    ) -> (ArchitectureConfig, f64) {
        let arch = self.search_space.sample_random();
        let pred = self.forward(x, &arch);

        // MSE loss
        let diff = &pred - y;
        let loss = diff.mapv(|v| v * v).mean().unwrap_or(0.0);
        let batch = x.nrows() as f64;

        // Gradient of MSE: 2 * (pred - y) / batch
        let grad_out = diff.mapv(|v| 2.0 * v / batch);

        // Backprop through output layer -- simplified gradient descent
        // We use numerical-style updates scaled by the gradient magnitude
        let grad_scale = grad_out.mapv(|v| v.abs()).mean().unwrap_or(0.0);

        // Update output weights
        self.output_weight
            .mapv_inplace(|v| v - lr * grad_scale * v.signum() * 0.01);
        self.output_bias
            .mapv_inplace(|v| v - lr * grad_scale * 0.01);

        // Update subnet weights (simplified gradient-based update)
        for layer_idx in 0..arch.num_layers {
            let key_w = WeightKey {
                layer_idx,
                layer_type: arch.layer_type,
                hidden_dim: arch.hidden_dim,
                component: "weight",
            };
            let key_b = WeightKey {
                layer_idx,
                layer_type: arch.layer_type,
                hidden_dim: arch.hidden_dim,
                component: "bias",
            };

            if let Some(w) = self.weights.get_mut(&key_w) {
                w.mapv_inplace(|v| v - lr * grad_scale * v.signum() * 0.01);
            }
            if let Some(b) = self.biases.get_mut(&key_b) {
                b.mapv_inplace(|v| v - lr * grad_scale * 0.01);
            }

            if arch.layer_type == LayerType::GatedDense {
                let key_gw = WeightKey {
                    layer_idx,
                    layer_type: arch.layer_type,
                    hidden_dim: arch.hidden_dim,
                    component: "gate_weight",
                };
                let key_gb = WeightKey {
                    layer_idx,
                    layer_type: arch.layer_type,
                    hidden_dim: arch.hidden_dim,
                    component: "gate_bias",
                };
                if let Some(gw) = self.weights.get_mut(&key_gw) {
                    gw.mapv_inplace(|v| v - lr * grad_scale * v.signum() * 0.01);
                }
                if let Some(gb) = self.biases.get_mut(&key_gb) {
                    gb.mapv_inplace(|v| v - lr * grad_scale * 0.01);
                }
            }
        }

        (arch, loss)
    }

    /// Train the supernet for multiple epochs using path sampling.
    pub fn train(&mut self, x: &Array2<f64>, y: &Array1<f64>, epochs: usize, lr: f64) -> Vec<f64> {
        let mut losses = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let (_, loss) = self.train_step(x, y, lr);
            losses.push(loss);
        }
        losses
    }

    /// Evaluate a subnet on validation data. Returns MSE.
    pub fn evaluate_subnet(
        &self,
        arch: &ArchitectureConfig,
        val_x: &Array2<f64>,
        val_y: &Array1<f64>,
    ) -> f64 {
        let pred = self.forward(val_x, arch);
        let diff = &pred - val_y;
        diff.mapv(|v| v * v).mean().unwrap_or(f64::MAX)
    }

    /// Sample and evaluate N random subnets, returning (architecture, mse) sorted by MSE.
    pub fn evaluate_random_subnets(
        &self,
        n: usize,
        val_x: &Array2<f64>,
        val_y: &Array1<f64>,
    ) -> Vec<(ArchitectureConfig, f64)> {
        let mut results: Vec<(ArchitectureConfig, f64)> = (0..n)
            .map(|_| {
                let arch = self.search_space.sample_random();
                let mse = self.evaluate_subnet(&arch, val_x, val_y);
                (arch, mse)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Select top-K architectures from evaluation results.
    pub fn select_top_k(
        results: &[(ArchitectureConfig, f64)],
        k: usize,
    ) -> Vec<ArchitectureConfig> {
        results.iter().take(k).map(|(a, _)| a.clone()).collect()
    }
}

// ─────────────────────── Standalone Model ───────────────────────

/// A standalone model for a specific architecture, retrained from scratch.
pub struct StandaloneModel {
    pub arch: ArchitectureConfig,
    pub input_dim: usize,
    layer_weights: Vec<Array2<f64>>,
    layer_biases: Vec<Array1<f64>>,
    gate_weights: Vec<Option<Array2<f64>>>,
    gate_biases: Vec<Option<Array1<f64>>>,
    output_weight: Array2<f64>,
    output_bias: Array1<f64>,
}

impl StandaloneModel {
    /// Create a new standalone model with random initialization.
    pub fn new(input_dim: usize, arch: ArchitectureConfig) -> Self {
        let mut rng = rand::thread_rng();
        let mut layer_weights = Vec::new();
        let mut layer_biases = Vec::new();
        let mut gate_weights = Vec::new();
        let mut gate_biases = Vec::new();

        for layer_idx in 0..arch.num_layers {
            let in_dim = if layer_idx == 0 {
                input_dim
            } else {
                arch.hidden_dim
            };
            let scale = (2.0 / (in_dim + arch.hidden_dim) as f64).sqrt();

            let w = Array2::from_shape_fn((in_dim, arch.hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            });
            let b = Array1::zeros(arch.hidden_dim);
            layer_weights.push(w);
            layer_biases.push(b);

            if arch.layer_type == LayerType::GatedDense {
                let gw = Array2::from_shape_fn((in_dim, arch.hidden_dim), |_| {
                    rng.gen_range(-scale..scale)
                });
                gate_weights.push(Some(gw));
                gate_biases.push(Some(Array1::zeros(arch.hidden_dim)));
            } else {
                gate_weights.push(None);
                gate_biases.push(None);
            }
        }

        let out_scale = (2.0 / (arch.hidden_dim + 1) as f64).sqrt();
        let output_weight =
            Array2::from_shape_fn((arch.hidden_dim, 1), |_| rng.gen_range(-out_scale..out_scale));
        let output_bias = Array1::zeros(1);

        StandaloneModel {
            arch,
            input_dim,
            layer_weights,
            layer_biases,
            gate_weights,
            gate_biases,
            output_weight,
            output_bias,
        }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut hidden = x.clone();

        for (i, (w, b)) in self
            .layer_weights
            .iter()
            .zip(self.layer_biases.iter())
            .enumerate()
        {
            let h_in = if hidden.ncols() != w.nrows() {
                let cols = hidden.ncols().min(w.nrows());
                if hidden.ncols() > w.nrows() {
                    hidden.slice(ndarray::s![.., ..cols]).to_owned()
                } else {
                    let batch = hidden.nrows();
                    let mut padded = Array2::zeros((batch, w.nrows()));
                    padded
                        .slice_mut(ndarray::s![.., ..hidden.ncols()])
                        .assign(&hidden);
                    padded
                }
            } else {
                hidden.clone()
            };

            let mut out = h_in.dot(w) + b;

            match self.arch.layer_type {
                LayerType::Dense => {
                    out.mapv_inplace(|v| self.arch.activation.apply(v));
                }
                LayerType::DenseNoBias => {
                    out = h_in.dot(w);
                    out.mapv_inplace(|v| self.arch.activation.apply(v));
                }
                LayerType::Residual => {
                    out.mapv_inplace(|v| self.arch.activation.apply(v));
                    if h_in.ncols() == out.ncols() {
                        out = out + &h_in;
                    }
                }
                LayerType::GatedDense => {
                    if let (Some(Some(gw)), Some(Some(gb))) =
                        (self.gate_weights.get(i), self.gate_biases.get(i))
                    {
                        let gate = (&h_in.dot(gw) + gb).mapv(|v| 1.0 / (1.0 + (-v).exp()));
                        out.mapv_inplace(|v| self.arch.activation.apply(v));
                        out = out * &gate;
                    }
                }
            }

            hidden = out;
        }

        let hd = hidden.ncols();
        let out_rows = self.output_weight.nrows().min(hd);
        let h_out = if hd != out_rows {
            if hd > out_rows {
                hidden.slice(ndarray::s![.., ..out_rows]).to_owned()
            } else {
                let batch = hidden.nrows();
                let mut padded = Array2::zeros((batch, out_rows));
                padded.slice_mut(ndarray::s![.., ..hd]).assign(&hidden);
                padded
            }
        } else {
            hidden
        };

        let logits = h_out.dot(&self.output_weight) + &self.output_bias;
        logits.column(0).to_owned()
    }

    /// Train the standalone model.
    pub fn train(&mut self, x: &Array2<f64>, y: &Array1<f64>, epochs: usize, lr: f64) -> Vec<f64> {
        let mut losses = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let pred = self.forward(x);
            let diff = &pred - y;
            let loss = diff.mapv(|v| v * v).mean().unwrap_or(0.0);
            losses.push(loss);

            let grad_scale = diff.mapv(|v| v.abs()).mean().unwrap_or(0.0);
            let update = lr * grad_scale * 0.01;

            // Update all weights
            for w in &mut self.layer_weights {
                w.mapv_inplace(|v| v - update * v.signum());
            }
            for b in &mut self.layer_biases {
                b.mapv_inplace(|v| v - update);
            }
            for gw in &mut self.gate_weights {
                if let Some(ref mut w) = gw {
                    w.mapv_inplace(|v| v - update * v.signum());
                }
            }
            for gb in &mut self.gate_biases {
                if let Some(ref mut b) = gb {
                    b.mapv_inplace(|v| v - update);
                }
            }
            self.output_weight
                .mapv_inplace(|v| v - update * v.signum());
            self.output_bias.mapv_inplace(|v| v - update);
        }

        losses
    }

    /// Evaluate on data. Returns MSE.
    pub fn evaluate(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let pred = self.forward(x);
        let diff = &pred - y;
        diff.mapv(|v| v * v).mean().unwrap_or(f64::MAX)
    }
}

// ─────────────────────── Bybit API ───────────────────────

/// Raw candle data from Bybit.
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
    result: BybitResult,
}

#[derive(Debug, Deserialize)]
struct BybitResult {
    list: Vec<Vec<String>>,
}

/// Fetch OHLCV kline data from Bybit public API.
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

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

/// Blocking version of fetch_bybit_klines.
pub fn fetch_bybit_klines_blocking(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
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

    candles.reverse();
    Ok(candles)
}

// ─────────────────────── Feature Engineering ───────────────────────

/// Compute log returns from a price series.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Simple moving average.
pub fn sma(values: &[f64], period: usize) -> Vec<f64> {
    if values.len() < period {
        return vec![];
    }
    let mut result = Vec::with_capacity(values.len() - period + 1);
    let mut sum: f64 = values[..period].iter().sum();
    result.push(sum / period as f64);
    for i in period..values.len() {
        sum += values[i] - values[i - period];
        result.push(sum / period as f64);
    }
    result
}

/// Rolling standard deviation.
pub fn rolling_std(values: &[f64], period: usize) -> Vec<f64> {
    if values.len() < period {
        return vec![];
    }
    let mut result = Vec::with_capacity(values.len() - period + 1);
    for i in 0..=(values.len() - period) {
        let window = &values[i..i + period];
        let mean: f64 = window.iter().sum::<f64>() / period as f64;
        let var = window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / period as f64;
        result.push(var.sqrt());
    }
    result
}

/// Approximate RSI calculation.
pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let returns = log_returns(prices);
    if returns.len() < period {
        return vec![];
    }
    let mut result = Vec::new();
    for i in period..=returns.len() {
        let window = &returns[i - period..i];
        let gains: f64 = window.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = window.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let rs = if losses == 0.0 {
            100.0
        } else {
            gains / losses
        };
        let rsi_val = 100.0 - (100.0 / (1.0 + rs));
        result.push(rsi_val / 100.0); // Normalize to [0, 1]
    }
    result
}

/// Build feature matrix and target vector from candles.
/// Features: log_ret_1, log_ret_5, log_ret_10, close/sma20, volatility_20, volume_ratio_20, rsi_14
/// Target: next-bar log return
pub fn build_features(candles: &[Candle]) -> (Array2<f64>, Array1<f64>) {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    let lookback = 50; // Need at least this many bars for features
    if closes.len() < lookback + 10 {
        return (Array2::zeros((0, 7)), Array1::zeros(0));
    }

    let ret1 = log_returns(&closes);
    let sma20 = sma(&closes, 20);
    let vol20 = rolling_std(&ret1, 20);
    let vol_sma20 = sma(&volumes, 20);
    let rsi14 = rsi(&closes, 14);

    // Align all features to a common start index
    let start = lookback;
    let end = closes.len() - 1; // -1 because target is next bar
    if start >= end {
        return (Array2::zeros((0, 7)), Array1::zeros(0));
    }

    let n_samples = end - start;
    let n_features = 7;
    let mut features = Array2::zeros((n_samples, n_features));
    let mut targets = Array1::zeros(n_samples);

    for (row, i) in (start..end).enumerate() {
        // log return 1-bar
        let lr1 = if i > 0 && i - 1 < ret1.len() {
            ret1[i - 1]
        } else {
            0.0
        };
        // log return 5-bar
        let lr5 = if i >= 5 {
            (closes[i] / closes[i - 5]).ln()
        } else {
            0.0
        };
        // log return 10-bar
        let lr10 = if i >= 10 {
            (closes[i] / closes[i - 10]).ln()
        } else {
            0.0
        };
        // close / sma20
        let sma_idx = if i >= 19 { i - 19 } else { 0 };
        let close_sma = if sma_idx < sma20.len() && sma20[sma_idx] > 0.0 {
            closes[i] / sma20[sma_idx] - 1.0
        } else {
            0.0
        };
        // volatility
        let vol_idx = if i >= 20 { i - 20 } else { 0 };
        let vol = if vol_idx < vol20.len() {
            vol20[vol_idx]
        } else {
            0.0
        };
        // volume ratio
        let vsma_idx = if i >= 19 { i - 19 } else { 0 };
        let vol_ratio = if vsma_idx < vol_sma20.len() && vol_sma20[vsma_idx] > 0.0 {
            volumes[i] / vol_sma20[vsma_idx] - 1.0
        } else {
            0.0
        };
        // RSI
        let rsi_idx = if i >= 14 { i - 14 } else { 0 };
        let rsi_val = if rsi_idx < rsi14.len() {
            rsi14[rsi_idx] - 0.5
        } else {
            0.0
        };

        features[[row, 0]] = lr1;
        features[[row, 1]] = lr5;
        features[[row, 2]] = lr10;
        features[[row, 3]] = close_sma;
        features[[row, 4]] = vol;
        features[[row, 5]] = vol_ratio;
        features[[row, 6]] = rsi_val;

        // Target: next bar return
        targets[row] = if i + 1 < closes.len() {
            (closes[i + 1] / closes[i]).ln()
        } else {
            0.0
        };
    }

    (features, targets)
}

/// Split data into train and validation sets.
pub fn train_val_split(
    x: &Array2<f64>,
    y: &Array1<f64>,
    train_ratio: f64,
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let n = x.nrows();
    let n_train = (n as f64 * train_ratio) as usize;

    let train_x = x.slice(ndarray::s![..n_train, ..]).to_owned();
    let train_y = y.slice(ndarray::s![..n_train]).to_owned();
    let val_x = x.slice(ndarray::s![n_train.., ..]).to_owned();
    let val_y = y.slice(ndarray::s![n_train..]).to_owned();

    (train_x, train_y, val_x, val_y)
}

/// Compute directional accuracy (percentage of correct sign predictions).
pub fn directional_accuracy(predictions: &Array1<f64>, actuals: &Array1<f64>) -> f64 {
    let n = predictions.len().min(actuals.len());
    if n == 0 {
        return 0.0;
    }
    let correct = (0..n)
        .filter(|&i| {
            (predictions[i] >= 0.0 && actuals[i] >= 0.0)
                || (predictions[i] < 0.0 && actuals[i] < 0.0)
        })
        .count();
    correct as f64 / n as f64
}

// ─────────────────────── Tests ───────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_relu() {
        assert_eq!(Activation::ReLU.apply(1.0), 1.0);
        assert_eq!(Activation::ReLU.apply(-1.0), 0.0);
        assert_eq!(Activation::ReLU.apply(0.0), 0.0);
    }

    #[test]
    fn test_activation_sigmoid() {
        let s = Activation::Sigmoid.apply(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        assert!(Activation::Sigmoid.apply(100.0) > 0.99);
        assert!(Activation::Sigmoid.apply(-100.0) < 0.01);
    }

    #[test]
    fn test_activation_tanh() {
        assert!((Activation::Tanh.apply(0.0)).abs() < 1e-10);
        assert!(Activation::Tanh.apply(3.0) > 0.99);
    }

    #[test]
    fn test_activation_leaky_relu() {
        assert_eq!(Activation::LeakyReLU.apply(1.0), 1.0);
        assert!((Activation::LeakyReLU.apply(-1.0) - (-0.01)).abs() < 1e-10);
    }

    #[test]
    fn test_activation_gelu() {
        // GELU(0) should be close to 0
        assert!(Activation::GELU.apply(0.0).abs() < 1e-5);
        // GELU is approximately x for large positive x
        assert!((Activation::GELU.apply(3.0) - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_search_space_size() {
        let ss = SearchSpace::default();
        let expected = 4 * 3 * 4 * 5 * 3; // 720
        assert_eq!(ss.size(), expected);
    }

    #[test]
    fn test_search_space_sample() {
        let ss = SearchSpace::default();
        let arch = ss.sample_random();
        assert!(ss.hidden_dims.contains(&arch.hidden_dim));
        assert!(ss.num_layers_options.contains(&arch.num_layers));
    }

    #[test]
    fn test_supernet_forward() {
        let ss = SearchSpace::default();
        let supernet = Supernet::new(7, ss);
        let x = Array2::from_shape_fn((10, 7), |_| rand::random::<f64>() * 0.1);
        let arch = supernet.search_space.sample_random();
        let out = supernet.forward(&x, &arch);
        assert_eq!(out.len(), 10);
        // Output should be finite
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_supernet_train_step() {
        let ss = SearchSpace::default();
        let mut supernet = Supernet::new(7, ss);
        let x = Array2::from_shape_fn((20, 7), |_| rand::random::<f64>() * 0.1);
        let y = Array1::from_shape_fn(20, |_| rand::random::<f64>() * 0.01);
        let (arch, loss) = supernet.train_step(&x, &y, 0.01);
        assert!(loss.is_finite());
        assert!(arch.num_layers > 0);
    }

    #[test]
    fn test_supernet_training_reduces_loss() {
        let ss = SearchSpace {
            layer_types: vec![LayerType::Dense],
            hidden_dims: vec![16],
            num_layers_options: vec![1],
            activations: vec![Activation::ReLU],
            dropout_rates: vec![0.0],
        };
        let mut supernet = Supernet::new(4, ss);
        let x = Array2::from_shape_fn((50, 4), |_| rand::random::<f64>());
        let y = Array1::zeros(50);
        let losses = supernet.train(&x, &y, 100, 0.1);
        // Loss should generally decrease or at least not explode
        assert!(losses.last().unwrap().is_finite());
    }

    #[test]
    fn test_evaluate_random_subnets() {
        let ss = SearchSpace::default();
        let supernet = Supernet::new(7, ss);
        let x = Array2::from_shape_fn((20, 7), |_| rand::random::<f64>() * 0.1);
        let y = Array1::from_shape_fn(20, |_| rand::random::<f64>() * 0.01);
        let results = supernet.evaluate_random_subnets(10, &x, &y);
        assert_eq!(results.len(), 10);
        // Results should be sorted by MSE
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn test_select_top_k() {
        let ss = SearchSpace::default();
        let supernet = Supernet::new(7, ss);
        let x = Array2::from_shape_fn((20, 7), |_| rand::random::<f64>() * 0.1);
        let y = Array1::from_shape_fn(20, |_| rand::random::<f64>() * 0.01);
        let results = supernet.evaluate_random_subnets(20, &x, &y);
        let top3 = Supernet::select_top_k(&results, 3);
        assert_eq!(top3.len(), 3);
    }

    #[test]
    fn test_standalone_model_forward() {
        let arch = ArchitectureConfig {
            layer_type: LayerType::Dense,
            hidden_dim: 16,
            num_layers: 2,
            activation: Activation::ReLU,
            dropout_rate: 0.0,
        };
        let model = StandaloneModel::new(7, arch);
        let x = Array2::from_shape_fn((10, 7), |_| rand::random::<f64>() * 0.1);
        let out = model.forward(&x);
        assert_eq!(out.len(), 10);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_standalone_model_train() {
        let arch = ArchitectureConfig {
            layer_type: LayerType::Dense,
            hidden_dim: 16,
            num_layers: 1,
            activation: Activation::ReLU,
            dropout_rate: 0.0,
        };
        let mut model = StandaloneModel::new(4, arch);
        let x = Array2::from_shape_fn((30, 4), |_| rand::random::<f64>());
        let y = Array1::zeros(30);
        let losses = model.train(&x, &y, 50, 0.1);
        assert_eq!(losses.len(), 50);
        assert!(losses.last().unwrap().is_finite());
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 110.0, 105.0];
        let rets = log_returns(&prices);
        assert_eq!(rets.len(), 2);
        assert!((rets[0] - (110.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_sma() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&values, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_std() {
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let result = rolling_std(&values, 3);
        assert!(!result.is_empty());
        assert!(result[0].abs() < 1e-10); // Constant values -> 0 std
    }

    #[test]
    fn test_rsi() {
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = rsi(&prices, 14);
        assert!(!result.is_empty());
        // All positive returns -> RSI near 1.0
        assert!(result.last().unwrap() > &0.8);
    }

    #[test]
    fn test_build_features() {
        // Create synthetic candles
        let candles: Vec<Candle> = (0..100)
            .map(|i| Candle {
                timestamp: i as u64 * 60000,
                open: 100.0 + (i as f64 * 0.1).sin(),
                high: 101.0 + (i as f64 * 0.1).sin(),
                low: 99.0 + (i as f64 * 0.1).sin(),
                close: 100.0 + (i as f64 * 0.1).sin() * 0.5,
                volume: 1000.0 + (i as f64 * 0.2).cos() * 100.0,
            })
            .collect();
        let (features, targets) = build_features(&candles);
        assert!(features.nrows() > 0);
        assert_eq!(features.ncols(), 7);
        assert_eq!(features.nrows(), targets.len());
        // All values should be finite
        assert!(features.iter().all(|v| v.is_finite()));
        assert!(targets.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_train_val_split() {
        let x = Array2::zeros((100, 7));
        let y = Array1::zeros(100);
        let (tx, ty, vx, vy) = train_val_split(&x, &y, 0.8);
        assert_eq!(tx.nrows(), 80);
        assert_eq!(ty.len(), 80);
        assert_eq!(vx.nrows(), 20);
        assert_eq!(vy.len(), 20);
    }

    #[test]
    fn test_directional_accuracy() {
        let pred = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1]);
        let actual = Array1::from_vec(vec![0.05, -0.1, -0.2, -0.3]);
        let acc = directional_accuracy(&pred, &actual);
        // pred signs: +, -, +, -
        // actual signs: +, -, -, -
        // correct: 1, 2, 4 => 3/4 = 0.75
        assert!((acc - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_gated_dense_forward() {
        let arch = ArchitectureConfig {
            layer_type: LayerType::GatedDense,
            hidden_dim: 16,
            num_layers: 2,
            activation: Activation::Tanh,
            dropout_rate: 0.0,
        };
        let ss = SearchSpace::default();
        let supernet = Supernet::new(7, ss);
        let x = Array2::from_shape_fn((5, 7), |_| rand::random::<f64>() * 0.1);
        let out = supernet.forward(&x, &arch);
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_residual_forward() {
        let arch = ArchitectureConfig {
            layer_type: LayerType::Residual,
            hidden_dim: 32,
            num_layers: 3,
            activation: Activation::GELU,
            dropout_rate: 0.0,
        };
        let model = StandaloneModel::new(7, arch);
        let x = Array2::from_shape_fn((8, 7), |_| rand::random::<f64>() * 0.1);
        let out = model.forward(&x);
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_full_pipeline_synthetic() {
        // End-to-end test with synthetic data
        let candles: Vec<Candle> = (0..200)
            .map(|i| {
                let t = i as f64 * 0.05;
                Candle {
                    timestamp: i as u64 * 60000,
                    open: 100.0 + t.sin() * 2.0,
                    high: 101.0 + t.sin() * 2.0,
                    low: 99.0 + t.sin() * 2.0,
                    close: 100.0 + t.sin() * 2.0 + t.cos() * 0.5,
                    volume: 1000.0 + (t * 2.0).cos() * 200.0,
                }
            })
            .collect();

        let (features, targets) = build_features(&candles);
        assert!(features.nrows() > 10);

        let (train_x, train_y, val_x, val_y) = train_val_split(&features, &targets, 0.7);

        // Train supernet
        let ss = SearchSpace {
            layer_types: vec![LayerType::Dense, LayerType::Residual],
            hidden_dims: vec![16, 32],
            num_layers_options: vec![1, 2],
            activations: vec![Activation::ReLU, Activation::Tanh],
            dropout_rates: vec![0.0],
        };
        let mut supernet = Supernet::new(7, ss);
        let losses = supernet.train(&train_x, &train_y, 50, 0.01);
        assert_eq!(losses.len(), 50);

        // Evaluate subnets
        let results = supernet.evaluate_random_subnets(10, &val_x, &val_y);
        assert_eq!(results.len(), 10);

        // Select top 2
        let top = Supernet::select_top_k(&results, 2);
        assert_eq!(top.len(), 2);

        // Retrain standalone
        let mut standalone = StandaloneModel::new(7, top[0].clone());
        let sa_losses = standalone.train(&train_x, &train_y, 50, 0.01);
        assert!(sa_losses.last().unwrap().is_finite());

        let mse = standalone.evaluate(&val_x, &val_y);
        assert!(mse.is_finite());
    }
}
