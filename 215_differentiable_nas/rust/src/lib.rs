//! Differentiable Architecture Search (DARTS) for Trading
//!
//! This crate implements DARTS - a method for making neural architecture search
//! continuous and differentiable, applied to financial time-series prediction.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Operation Primitives
// ---------------------------------------------------------------------------

/// Enumeration of candidate operations in the DARTS search space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    Identity,
    Zero,
    Dense,
    Conv1DLike,
    SkipConnection,
}

impl OpType {
    pub fn all() -> Vec<OpType> {
        vec![
            OpType::Identity,
            OpType::Zero,
            OpType::Dense,
            OpType::Conv1DLike,
            OpType::SkipConnection,
        ]
    }

    pub fn name(&self) -> &str {
        match self {
            OpType::Identity => "identity",
            OpType::Zero => "zero",
            OpType::Dense => "dense",
            OpType::Conv1DLike => "conv1d_like",
            OpType::SkipConnection => "skip_connection",
        }
    }
}

/// A concrete operation with learnable weights.
#[derive(Debug, Clone)]
pub struct Operation {
    pub op_type: OpType,
    pub weights: Option<Array2<f64>>,
    pub bias: Option<Array1<f64>>,
    input_dim: usize,
    output_dim: usize,
}

impl Operation {
    pub fn new(op_type: OpType, input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let (weights, bias) = match op_type {
            OpType::Identity | OpType::Zero | OpType::SkipConnection => (None, None),
            OpType::Dense => {
                let scale = (2.0 / input_dim as f64).sqrt();
                let w = Array2::from_shape_fn((output_dim, input_dim), |_| {
                    rng.gen_range(-scale..scale)
                });
                let b = Array1::zeros(output_dim);
                (Some(w), Some(b))
            }
            OpType::Conv1DLike => {
                // 1D convolution approximated as a banded weight matrix (kernel_size=3)
                let kernel_size = 3.min(input_dim);
                let scale = (2.0 / kernel_size as f64).sqrt();
                let mut w = Array2::zeros((output_dim, input_dim));
                for i in 0..output_dim {
                    let start = if i >= kernel_size / 2 {
                        i - kernel_size / 2
                    } else {
                        0
                    };
                    let end = (i + kernel_size / 2 + 1).min(input_dim);
                    for j in start..end {
                        w[[i, j]] = rng.gen_range(-scale..scale);
                    }
                }
                let b = Array1::zeros(output_dim);
                (Some(w), Some(b))
            }
        };
        Self {
            op_type,
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass for this operation.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        match self.op_type {
            OpType::Identity | OpType::SkipConnection => {
                if x.len() == self.output_dim {
                    x.clone()
                } else {
                    // Truncate or pad
                    let mut out = Array1::zeros(self.output_dim);
                    let copy_len = x.len().min(self.output_dim);
                    for i in 0..copy_len {
                        out[i] = x[i];
                    }
                    out
                }
            }
            OpType::Zero => Array1::zeros(self.output_dim),
            OpType::Dense => {
                let w = self.weights.as_ref().unwrap();
                let b = self.bias.as_ref().unwrap();
                let out = w.dot(x) + b;
                // ReLU activation
                out.mapv(|v| v.max(0.0))
            }
            OpType::Conv1DLike => {
                let w = self.weights.as_ref().unwrap();
                let b = self.bias.as_ref().unwrap();
                let out = w.dot(x) + b;
                out.mapv(|v| v.max(0.0))
            }
        }
    }

    /// Compute gradient of loss w.r.t. operation weights (simplified numerical gradient).
    pub fn weight_gradient(
        &self,
        x: &Array1<f64>,
        grad_output: &Array1<f64>,
    ) -> Option<Array2<f64>> {
        match self.op_type {
            OpType::Dense | OpType::Conv1DLike => {
                let w = self.weights.as_ref().unwrap();
                let b = self.bias.as_ref().unwrap();
                let pre_act = w.dot(x) + b;
                // ReLU mask
                let mask: Array1<f64> = pre_act.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
                let masked_grad = grad_output * &mask;
                // dL/dW = masked_grad outer x
                let grad_w = outer_product(&masked_grad, x);
                Some(grad_w)
            }
            _ => None,
        }
    }

    /// Update weights with gradient descent.
    pub fn update_weights(&mut self, grad_w: &Array2<f64>, lr: f64) {
        if let Some(ref mut w) = self.weights {
            *w = &*w - &(grad_w * lr);
        }
    }
}

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let m = a.len();
    let n = b.len();
    Array2::from_shape_fn((m, n), |(i, j)| a[i] * b[j])
}

// ---------------------------------------------------------------------------
// Mixed Operation (softmax over candidate operations)
// ---------------------------------------------------------------------------

/// A mixed operation that maintains architecture weights alpha over candidate ops.
#[derive(Debug, Clone)]
pub struct MixedOperation {
    pub operations: Vec<Operation>,
    pub alphas: Vec<f64>,
    pub op_types: Vec<OpType>,
}

impl MixedOperation {
    pub fn new(op_types: &[OpType], input_dim: usize, output_dim: usize) -> Self {
        let operations: Vec<Operation> = op_types
            .iter()
            .map(|&ot| Operation::new(ot, input_dim, output_dim))
            .collect();
        let alphas = vec![0.0; op_types.len()]; // uniform init
        Self {
            operations,
            alphas,
            op_types: op_types.to_vec(),
        }
    }

    /// Compute softmax weights from alphas.
    pub fn softmax_weights(&self) -> Vec<f64> {
        softmax(&self.alphas)
    }

    /// Forward pass: softmax-weighted mixture of all operations.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let weights = self.softmax_weights();
        let outputs: Vec<Array1<f64>> = self.operations.iter().map(|op| op.forward(x)).collect();
        let mut result = Array1::zeros(outputs[0].len());
        for (w, out) in weights.iter().zip(outputs.iter()) {
            result = result + &(out * *w);
        }
        result
    }

    /// Select the operation with the highest alpha (discretization).
    pub fn best_op_index(&self) -> usize {
        self.alphas
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }

    pub fn best_op_type(&self) -> OpType {
        self.op_types[self.best_op_index()]
    }

    /// Update architecture parameters alpha given gradient.
    pub fn update_alphas(&mut self, grad_alpha: &[f64], lr: f64) {
        for (a, g) in self.alphas.iter_mut().zip(grad_alpha.iter()) {
            *a -= lr * g;
        }
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// DARTS Cell
// ---------------------------------------------------------------------------

/// A DARTS cell composed of multiple mixed operations forming a small DAG.
/// Nodes: 0 = input, 1..num_intermediate = intermediate, last = output
/// Each intermediate node j receives edges from all previous nodes i < j.
#[derive(Debug, Clone)]
pub struct DARTSCell {
    pub num_nodes: usize,
    /// edges[j][i] = MixedOperation on edge from node i to node j
    pub edges: Vec<Vec<MixedOperation>>,
    pub feature_dim: usize,
}

impl DARTSCell {
    pub fn new(num_intermediate_nodes: usize, feature_dim: usize, op_types: &[OpType]) -> Self {
        let num_nodes = num_intermediate_nodes + 2; // input + intermediate + output
        let mut edges = Vec::new();
        // Node 0 is input, has no incoming edges
        edges.push(Vec::new());
        // Intermediate and output nodes
        for j in 1..num_nodes {
            let mut incoming = Vec::new();
            for _i in 0..j {
                incoming.push(MixedOperation::new(op_types, feature_dim, feature_dim));
            }
            edges.push(incoming);
        }
        Self {
            num_nodes,
            edges,
            feature_dim,
        }
    }

    /// Forward pass through the cell.
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let mut node_outputs: Vec<Array1<f64>> = Vec::with_capacity(self.num_nodes);
        node_outputs.push(input.clone());

        for j in 1..self.num_nodes {
            let mut node_sum = Array1::zeros(self.feature_dim);
            for (i, mixed_op) in self.edges[j].iter().enumerate() {
                let edge_out = mixed_op.forward(&node_outputs[i]);
                node_sum = node_sum + edge_out;
            }
            node_outputs.push(node_sum);
        }
        // Output is the last node
        node_outputs.last().unwrap().clone()
    }

    /// Collect all architecture parameters.
    pub fn collect_alphas(&self) -> Vec<Vec<Vec<f64>>> {
        self.edges
            .iter()
            .map(|node_edges| {
                node_edges.iter().map(|mo| mo.alphas.clone()).collect()
            })
            .collect()
    }

    /// Discretize: extract the best operation per edge.
    pub fn discretize(&self) -> Vec<Vec<(usize, usize, OpType)>> {
        let mut result = Vec::new();
        for (j, node_edges) in self.edges.iter().enumerate() {
            let mut node_result = Vec::new();
            for (i, mixed_op) in node_edges.iter().enumerate() {
                let best = mixed_op.best_op_type();
                if best != OpType::Zero {
                    node_result.push((i, j, best));
                }
            }
            result.push(node_result);
        }
        result
    }

    /// Pretty-print the discovered architecture.
    pub fn print_architecture(&self) {
        println!("Discovered Architecture:");
        println!("========================");
        for (j, node_edges) in self.edges.iter().enumerate() {
            for (i, mixed_op) in node_edges.iter().enumerate() {
                let best = mixed_op.best_op_type();
                let weights = mixed_op.softmax_weights();
                let best_idx = mixed_op.best_op_index();
                println!(
                    "  Edge ({} -> {}): {} (weight: {:.4})",
                    i, j, best.name(), weights[best_idx]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bi-level Optimization / DARTS Search
// ---------------------------------------------------------------------------

/// Configuration for DARTS search.
#[derive(Debug, Clone)]
pub struct DARTSConfig {
    pub num_intermediate_nodes: usize,
    pub feature_dim: usize,
    pub search_epochs: usize,
    pub weight_lr: f64,
    pub arch_lr: f64,
    pub op_types: Vec<OpType>,
}

impl Default for DARTSConfig {
    fn default() -> Self {
        Self {
            num_intermediate_nodes: 2,
            feature_dim: 5,
            search_epochs: 50,
            weight_lr: 0.01,
            arch_lr: 0.02,
            op_types: OpType::all(),
        }
    }
}

/// Result of a DARTS search.
#[derive(Debug)]
pub struct SearchResult {
    pub cell: DARTSCell,
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub architecture: Vec<Vec<(usize, usize, OpType)>>,
}

/// Simple MSE loss for a single sample.
fn mse_loss(prediction: &Array1<f64>, target: &Array1<f64>) -> f64 {
    let diff = prediction - target;
    diff.mapv(|v| v * v).mean().unwrap_or(0.0)
}

/// Compute gradient of MSE loss w.r.t. prediction.
fn mse_grad(prediction: &Array1<f64>, target: &Array1<f64>) -> Array1<f64> {
    let n = prediction.len() as f64;
    (prediction - target) * (2.0 / n)
}

/// Numerical gradient of loss w.r.t. a single alpha parameter.
fn numerical_alpha_gradient(
    cell: &DARTSCell,
    node_j: usize,
    edge_i: usize,
    alpha_k: usize,
    input: &Array1<f64>,
    target: &Array1<f64>,
    epsilon: f64,
) -> f64 {
    // f(alpha + eps)
    let mut cell_plus = cell.clone();
    cell_plus.edges[node_j][edge_i].alphas[alpha_k] += epsilon;
    let out_plus = cell_plus.forward(input);
    let loss_plus = mse_loss(&out_plus, target);

    // f(alpha - eps)
    let mut cell_minus = cell.clone();
    cell_minus.edges[node_j][edge_i].alphas[alpha_k] -= epsilon;
    let out_minus = cell_minus.forward(input);
    let loss_minus = mse_loss(&out_minus, target);

    (loss_plus - loss_minus) / (2.0 * epsilon)
}

/// Run DARTS bi-level optimization search.
pub fn darts_search(
    config: &DARTSConfig,
    train_data: &[(Array1<f64>, Array1<f64>)],
    val_data: &[(Array1<f64>, Array1<f64>)],
) -> SearchResult {
    let mut cell = DARTSCell::new(
        config.num_intermediate_nodes,
        config.feature_dim,
        &config.op_types,
    );
    let mut train_losses = Vec::new();
    let mut val_losses = Vec::new();
    let mut rng = rand::thread_rng();
    let epsilon = 1e-4;

    for epoch in 0..config.search_epochs {
        // --- Step 1: Weight update on training data ---
        let train_idx = rng.gen_range(0..train_data.len());
        let (ref train_x, ref train_y) = train_data[train_idx];
        let train_pred = cell.forward(train_x);
        let train_loss = mse_loss(&train_pred, train_y);
        let grad_out = mse_grad(&train_pred, train_y);

        // Update weights of all operations in all mixed ops
        for j in 1..cell.num_nodes {
            for i in 0..cell.edges[j].len() {
                let sw = cell.edges[j][i].softmax_weights();
                for (k, op) in cell.edges[j][i].operations.iter_mut().enumerate() {
                    if let Some(gw) =
                        op.weight_gradient(if i == 0 { train_x } else { train_x }, &grad_out)
                    {
                        op.update_weights(&gw, config.weight_lr * sw[k]);
                    }
                }
            }
        }

        // --- Step 2: Architecture update on validation data ---
        let val_idx = rng.gen_range(0..val_data.len());
        let (ref val_x, ref val_y) = val_data[val_idx];

        for j in 1..cell.num_nodes {
            for i in 0..cell.edges[j].len() {
                let num_ops = cell.edges[j][i].alphas.len();
                let mut grad_alpha = vec![0.0; num_ops];
                for k in 0..num_ops {
                    grad_alpha[k] =
                        numerical_alpha_gradient(&cell, j, i, k, val_x, val_y, epsilon);
                }
                cell.edges[j][i].update_alphas(&grad_alpha, config.arch_lr);
            }
        }

        // Record losses
        let val_pred = cell.forward(val_x);
        let val_loss = mse_loss(&val_pred, val_y);
        train_losses.push(train_loss);
        val_losses.push(val_loss);

        if epoch % 10 == 0 || epoch == config.search_epochs - 1 {
            println!(
                "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}",
                epoch + 1,
                config.search_epochs,
                train_loss,
                val_loss
            );
        }
    }

    let architecture = cell.discretize();
    SearchResult {
        cell,
        train_losses,
        val_losses,
        architecture,
    }
}

// ---------------------------------------------------------------------------
// Architecture Evaluation
// ---------------------------------------------------------------------------

/// Evaluate a discovered architecture on test data.
pub fn evaluate_architecture(
    cell: &DARTSCell,
    test_data: &[(Array1<f64>, Array1<f64>)],
) -> f64 {
    if test_data.is_empty() {
        return 0.0;
    }
    let total_loss: f64 = test_data
        .iter()
        .map(|(x, y)| {
            let pred = cell.forward(x);
            mse_loss(&pred, y)
        })
        .sum();
    total_loss / test_data.len() as f64
}

/// A simple hand-designed baseline (single dense layer).
pub fn baseline_forward(weights: &Array2<f64>, bias: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    let out = weights.dot(x) + bias;
    out.mapv(|v| v.max(0.0))
}

/// Train and evaluate a simple baseline model.
pub fn train_baseline(
    feature_dim: usize,
    train_data: &[(Array1<f64>, Array1<f64>)],
    test_data: &[(Array1<f64>, Array1<f64>)],
    epochs: usize,
    lr: f64,
) -> f64 {
    let mut rng = rand::thread_rng();
    let scale = (2.0 / feature_dim as f64).sqrt();
    let mut weights = Array2::from_shape_fn((feature_dim, feature_dim), |_| {
        rng.gen_range(-scale..scale)
    });
    let mut bias = Array1::zeros(feature_dim);

    for _epoch in 0..epochs {
        let idx = rng.gen_range(0..train_data.len());
        let (ref x, ref y) = train_data[idx];
        let pred = baseline_forward(&weights, &bias, x);
        let pre_act = weights.dot(x) + &bias;
        let mask: Array1<f64> = pre_act.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let grad_out = mse_grad(&pred, y);
        let masked = &grad_out * &mask;
        let grad_w = outer_product(&masked, x);
        weights = weights - &(grad_w * lr);
        bias = bias - &(&masked * lr);
    }

    // Evaluate
    if test_data.is_empty() {
        return 0.0;
    }
    let total_loss: f64 = test_data
        .iter()
        .map(|(x, y)| {
            let pred = baseline_forward(&weights, &bias, x);
            mse_loss(&pred, y)
        })
        .sum();
    total_loss / test_data.len() as f64
}

// ---------------------------------------------------------------------------
// Bybit API Data Fetching
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

/// Fetch kline data from Bybit API.
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

/// Convert candles to feature vectors (normalized OHLCV).
pub fn candles_to_features(candles: &[Candle]) -> Vec<Array1<f64>> {
    if candles.is_empty() {
        return Vec::new();
    }

    // Compute means and stds for z-score normalization
    let n = candles.len() as f64;
    let mean_o: f64 = candles.iter().map(|c| c.open).sum::<f64>() / n;
    let mean_h: f64 = candles.iter().map(|c| c.high).sum::<f64>() / n;
    let mean_l: f64 = candles.iter().map(|c| c.low).sum::<f64>() / n;
    let mean_c: f64 = candles.iter().map(|c| c.close).sum::<f64>() / n;
    let mean_v: f64 = candles.iter().map(|c| c.volume).sum::<f64>() / n;

    let std_o = (candles.iter().map(|c| (c.open - mean_o).powi(2)).sum::<f64>() / n).sqrt().max(1e-8);
    let std_h = (candles.iter().map(|c| (c.high - mean_h).powi(2)).sum::<f64>() / n).sqrt().max(1e-8);
    let std_l = (candles.iter().map(|c| (c.low - mean_l).powi(2)).sum::<f64>() / n).sqrt().max(1e-8);
    let std_c = (candles.iter().map(|c| (c.close - mean_c).powi(2)).sum::<f64>() / n).sqrt().max(1e-8);
    let std_v = (candles.iter().map(|c| (c.volume - mean_v).powi(2)).sum::<f64>() / n).sqrt().max(1e-8);

    candles
        .iter()
        .map(|c| {
            Array1::from(vec![
                (c.open - mean_o) / std_o,
                (c.high - mean_h) / std_h,
                (c.low - mean_l) / std_l,
                (c.close - mean_c) / std_c,
                (c.volume - mean_v) / std_v,
            ])
        })
        .collect()
}

/// Create sliding window samples: input = features[t], target = features[t+1].
pub fn create_samples(features: &[Array1<f64>]) -> Vec<(Array1<f64>, Array1<f64>)> {
    features
        .windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect()
}

/// Split data into train and validation sets.
pub fn train_val_split(
    samples: &[(Array1<f64>, Array1<f64>)],
    train_ratio: f64,
) -> (Vec<(Array1<f64>, Array1<f64>)>, Vec<(Array1<f64>, Array1<f64>)>) {
    let split = (samples.len() as f64 * train_ratio) as usize;
    let train = samples[..split].to_vec();
    let val = samples[split..].to_vec();
    (train, val)
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_equal() {
        let logits = vec![0.0, 0.0, 0.0];
        let probs = softmax(&logits);
        for p in &probs {
            assert!((*p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_identity_operation() {
        let op = Operation::new(OpType::Identity, 5, 5);
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = op.forward(&x);
        assert_eq!(x, y);
    }

    #[test]
    fn test_zero_operation() {
        let op = Operation::new(OpType::Zero, 5, 5);
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = op.forward(&x);
        assert_eq!(y, Array1::<f64>::zeros(5));
    }

    #[test]
    fn test_dense_operation_shape() {
        let op = Operation::new(OpType::Dense, 5, 5);
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = op.forward(&x);
        assert_eq!(y.len(), 5);
    }

    #[test]
    fn test_conv1d_operation_shape() {
        let op = Operation::new(OpType::Conv1DLike, 5, 5);
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = op.forward(&x);
        assert_eq!(y.len(), 5);
    }

    #[test]
    fn test_mixed_operation_forward() {
        let op_types = OpType::all();
        let mixed = MixedOperation::new(&op_types, 5, 5);
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = mixed.forward(&x);
        assert_eq!(y.len(), 5);
    }

    #[test]
    fn test_mixed_operation_best_op() {
        let op_types = OpType::all();
        let mut mixed = MixedOperation::new(&op_types, 5, 5);
        // Set dense to have highest alpha
        mixed.alphas[2] = 10.0;
        assert_eq!(mixed.best_op_type(), OpType::Dense);
    }

    #[test]
    fn test_darts_cell_forward() {
        let op_types = OpType::all();
        let cell = DARTSCell::new(2, 5, &op_types);
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = cell.forward(&x);
        assert_eq!(y.len(), 5);
    }

    #[test]
    fn test_darts_cell_discretize() {
        let op_types = OpType::all();
        let cell = DARTSCell::new(2, 5, &op_types);
        let arch = cell.discretize();
        assert!(!arch.is_empty());
    }

    #[test]
    fn test_darts_search_runs() {
        let config = DARTSConfig {
            num_intermediate_nodes: 1,
            feature_dim: 3,
            search_epochs: 5,
            weight_lr: 0.01,
            arch_lr: 0.02,
            op_types: OpType::all(),
        };
        let train_data: Vec<(Array1<f64>, Array1<f64>)> = (0..10)
            .map(|i| {
                let x = Array1::from(vec![i as f64 * 0.1, (i as f64 * 0.1).sin(), 0.5]);
                let y = Array1::from(vec![(i + 1) as f64 * 0.1, ((i + 1) as f64 * 0.1).sin(), 0.5]);
                (x, y)
            })
            .collect();
        let val_data = train_data.clone();

        let result = darts_search(&config, &train_data, &val_data);
        assert_eq!(result.train_losses.len(), 5);
        assert_eq!(result.val_losses.len(), 5);
    }

    #[test]
    fn test_evaluate_architecture() {
        let cell = DARTSCell::new(1, 3, &OpType::all());
        let test_data: Vec<(Array1<f64>, Array1<f64>)> = vec![
            (Array1::from(vec![1.0, 0.0, 0.0]), Array1::from(vec![0.0, 1.0, 0.0])),
        ];
        let loss = evaluate_architecture(&cell, &test_data);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_create_samples() {
        let features = vec![
            Array1::from(vec![1.0, 2.0]),
            Array1::from(vec![3.0, 4.0]),
            Array1::from(vec![5.0, 6.0]),
        ];
        let samples = create_samples(&features);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].0, Array1::from(vec![1.0, 2.0]));
        assert_eq!(samples[0].1, Array1::from(vec![3.0, 4.0]));
    }

    #[test]
    fn test_train_val_split() {
        let samples: Vec<(Array1<f64>, Array1<f64>)> = (0..10)
            .map(|i| (Array1::from(vec![i as f64]), Array1::from(vec![i as f64 + 1.0])))
            .collect();
        let (train, val) = train_val_split(&samples, 0.8);
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    #[test]
    fn test_candles_to_features() {
        let candles = vec![
            Candle { timestamp: 1, open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 1000.0 },
            Candle { timestamp: 2, open: 102.0, high: 107.0, low: 98.0, close: 106.0, volume: 1200.0 },
            Candle { timestamp: 3, open: 106.0, high: 110.0, low: 100.0, close: 108.0, volume: 1100.0 },
        ];
        let features = candles_to_features(&candles);
        assert_eq!(features.len(), 3);
        assert_eq!(features[0].len(), 5);
    }

    #[test]
    fn test_alpha_update() {
        let mut mixed = MixedOperation::new(&OpType::all(), 3, 3);
        let grad = vec![0.1, -0.2, 0.3, -0.1, 0.0];
        mixed.update_alphas(&grad, 0.1);
        assert!((mixed.alphas[0] - (-0.01)).abs() < 1e-10);
        assert!((mixed.alphas[1] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_outer_product() {
        let a = Array1::from(vec![1.0, 2.0]);
        let b = Array1::from(vec![3.0, 4.0, 5.0]);
        let result = outer_product(&a, &b);
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 3.0);
        assert_eq!(result[[1, 2]], 10.0);
    }
}
