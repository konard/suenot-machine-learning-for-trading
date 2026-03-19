//! # DARTS Trading
//!
//! Differentiable Architecture Search (DARTS) adapted for financial time series.
//! Automatically discovers optimal neural network architectures for trading tasks
//! such as price prediction and volatility forecasting.
//!
//! ## Key Components
//! - Time-series specific operations (1D conv, dilated conv, moving average, skip)
//! - DARTS cell with architecture weights and continuous relaxation
//! - Gumbel-Softmax for differentiable discrete selection
//! - Bi-level optimization (alternating architecture/weight updates)
//! - Architecture discretization and extraction
//! - Bybit API integration for live market data

use anyhow::{Context, Result};

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Operation types for the time-series search space
// ---------------------------------------------------------------------------

/// Enumeration of all candidate operations in the DARTS search space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpType {
    /// 1D convolution with given kernel size
    Conv1D { kernel_size: usize },
    /// Dilated 1D convolution with given kernel size and dilation factor
    DilatedConv1D { kernel_size: usize, dilation: usize },
    /// Moving average with given window size
    MovingAverage { window: usize },
    /// Identity / skip connection
    Skip,
    /// Zero operation (prunes the edge)
    Zero,
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpType::Conv1D { kernel_size } => write!(f, "Conv1D(k={})", kernel_size),
            OpType::DilatedConv1D {
                kernel_size,
                dilation,
            } => write!(f, "DilConv1D(k={}, d={})", kernel_size, dilation),
            OpType::MovingAverage { window } => write!(f, "MovAvg(w={})", window),
            OpType::Skip => write!(f, "Skip"),
            OpType::Zero => write!(f, "Zero"),
        }
    }
}

/// Default operation space for trading time series.
pub fn default_op_space() -> Vec<OpType> {
    vec![
        OpType::Conv1D { kernel_size: 3 },
        OpType::Conv1D { kernel_size: 5 },
        OpType::DilatedConv1D {
            kernel_size: 3,
            dilation: 2,
        },
        OpType::DilatedConv1D {
            kernel_size: 3,
            dilation: 4,
        },
        OpType::MovingAverage { window: 5 },
        OpType::MovingAverage { window: 10 },
        OpType::Skip,
        OpType::Zero,
    ]
}

// ---------------------------------------------------------------------------
// Operation forward passes
// ---------------------------------------------------------------------------

/// Apply a single operation to a 1-D time-series input.
/// Returns a vector of the same length (zero-padded where necessary).
pub fn apply_operation(op: &OpType, input: &[f64], weights: &[f64]) -> Vec<f64> {
    let n = input.len();
    match op {
        OpType::Conv1D { kernel_size } => conv1d(input, weights, *kernel_size, 1),
        OpType::DilatedConv1D {
            kernel_size,
            dilation,
        } => conv1d(input, weights, *kernel_size, *dilation),
        OpType::MovingAverage { window } => moving_average(input, *window),
        OpType::Skip => input.to_vec(),
        OpType::Zero => vec![0.0; n],
    }
}

/// 1-D convolution with dilation, producing same-length output via zero-padding.
fn conv1d(input: &[f64], kernel: &[f64], kernel_size: usize, dilation: usize) -> Vec<f64> {
    let n = input.len();
    let effective_size = (kernel_size - 1) * dilation + 1;
    let pad = effective_size / 2;
    let mut output = vec![0.0; n];

    // Use provided kernel weights (only first kernel_size elements)
    let k: Vec<f64> = if kernel.len() >= kernel_size {
        kernel[..kernel_size].to_vec()
    } else {
        // Pad kernel with zeros if too short
        let mut k = kernel.to_vec();
        k.resize(kernel_size, 0.0);
        k
    };

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..kernel_size {
            let idx = i as isize + (j as isize * dilation as isize) - pad as isize;
            if idx >= 0 && (idx as usize) < n {
                sum += input[idx as usize] * k[j];
            }
        }
        output[i] = sum;
    }
    output
}

/// Moving average filter of given window size.
fn moving_average(input: &[f64], window: usize) -> Vec<f64> {
    let n = input.len();
    if n == 0 || window == 0 {
        return vec![0.0; n];
    }
    let mut output = vec![0.0; n];
    let mut cumsum = 0.0;
    for i in 0..n {
        cumsum += input[i];
        if i >= window {
            cumsum -= input[i - window];
        }
        let w = (i + 1).min(window) as f64;
        output[i] = cumsum / w;
    }
    output
}

// ---------------------------------------------------------------------------
// Softmax & Gumbel-Softmax utilities
// ---------------------------------------------------------------------------

/// Standard softmax over a slice of logits.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Gumbel-Softmax: differentiable approximation to categorical sampling.
///
/// - `logits`: unnormalized log-probabilities (architecture weights alpha)
/// - `temperature`: controls sharpness (lower = more discrete)
/// - `rng`: random number generator
pub fn gumbel_softmax(logits: &[f64], temperature: f64, rng: &mut impl Rng) -> Vec<f64> {
    let gumbel_noise: Vec<f64> = logits
        .iter()
        .map(|_| {
            let u: f64 = rng.gen_range(1e-10..1.0f64);
            -(-u.ln()).ln()
        })
        .collect();
    let scaled: Vec<f64> = logits
        .iter()
        .zip(&gumbel_noise)
        .map(|(l, g)| (l + g) / temperature)
        .collect();
    softmax(&scaled)
}

/// Sample Gumbel(0,1) noise.
pub fn sample_gumbel(rng: &mut impl Rng) -> f64 {
    let u: f64 = rng.gen_range(1e-10..1.0f64);
    -(-u.ln()).ln()
}

// ---------------------------------------------------------------------------
// DARTS Cell
// ---------------------------------------------------------------------------

/// Architecture parameters for a single edge (one per candidate operation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeAlpha {
    /// Logits for each operation on this edge
    pub logits: Vec<f64>,
}

impl EdgeAlpha {
    pub fn new(num_ops: usize) -> Self {
        Self {
            logits: vec![0.0; num_ops], // uniform initialization
        }
    }

    /// Softmax probabilities for this edge.
    pub fn probabilities(&self) -> Vec<f64> {
        softmax(&self.logits)
    }

    /// Index of the strongest operation.
    pub fn best_op(&self) -> usize {
        self.logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Strength of the best operation (its softmax probability).
    pub fn best_strength(&self) -> f64 {
        let probs = self.probabilities();
        probs[self.best_op()]
    }
}

/// A DARTS cell: a DAG with intermediate nodes, edges carrying mixed operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DartsCell {
    /// Number of intermediate nodes in the cell
    pub num_nodes: usize,
    /// Operation space
    pub ops: Vec<OpType>,
    /// Architecture parameters: alpha[i][j] for edge from node i to node j
    /// Only edges where i < j exist. Stored as flat vec indexed by edge_index(i,j).
    pub alphas: Vec<EdgeAlpha>,
    /// Learnable weights for parameterized operations, indexed per edge per operation.
    /// Shape: num_edges x num_ops x max_kernel_size
    pub op_weights: Vec<Vec<Vec<f64>>>,
}

impl DartsCell {
    /// Create a new DARTS cell.
    ///
    /// - `num_nodes`: number of intermediate nodes (typically 4)
    /// - `ops`: candidate operations
    pub fn new(num_nodes: usize, ops: Vec<OpType>, rng: &mut impl Rng) -> Self {
        let num_input_nodes = 2; // two inputs from previous cells
        let total_nodes = num_input_nodes + num_nodes;
        let num_ops = ops.len();

        // Count edges: from every earlier node to every later intermediate node
        let mut num_edges = 0;
        for j in num_input_nodes..total_nodes {
            for _i in 0..j {
                num_edges += 1;
            }
        }

        let alphas: Vec<EdgeAlpha> = (0..num_edges).map(|_| EdgeAlpha::new(num_ops)).collect();

        // Initialize operation weights (small random)
        let max_kernel = 5;
        let op_weights: Vec<Vec<Vec<f64>>> = (0..num_edges)
            .map(|_| {
                (0..num_ops)
                    .map(|_| (0..max_kernel).map(|_| rng.gen_range(-0.1..0.1)).collect())
                    .collect()
            })
            .collect();

        Self {
            num_nodes,
            ops,
            alphas,
            op_weights,
        }
    }

    /// Number of input nodes (fixed at 2 for DARTS).
    pub fn num_input_nodes(&self) -> usize {
        2
    }

    /// Total nodes including inputs.
    pub fn total_nodes(&self) -> usize {
        self.num_input_nodes() + self.num_nodes
    }

    /// Compute edge index for edge (i -> j) where i < j.
    pub fn edge_index(&self, i: usize, j: usize) -> usize {
        assert!(i < j);
        let num_input = self.num_input_nodes();
        let mut idx = 0;
        for jj in num_input..self.total_nodes() {
            for ii in 0..jj {
                if ii == i && jj == j {
                    return idx;
                }
                idx += 1;
            }
        }
        panic!("Edge ({}, {}) not found in cell", i, j);
    }

    /// Forward pass through the cell with continuous relaxation.
    ///
    /// - `input0`, `input1`: outputs from two previous cells (1-D sequences)
    /// - `temperature`: Gumbel-Softmax temperature
    /// - `rng`: for Gumbel noise
    ///
    /// Returns the cell output (average of intermediate node outputs).
    pub fn forward(
        &self,
        input0: &[f64],
        input1: &[f64],
        temperature: f64,
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        let n = input0.len();
        let total = self.total_nodes();
        let num_input = self.num_input_nodes();

        // Node outputs
        let mut node_outputs: Vec<Vec<f64>> = Vec::with_capacity(total);
        node_outputs.push(input0.to_vec());
        node_outputs.push(input1.to_vec());

        // Compute intermediate nodes
        for j in num_input..total {
            let mut node_out = vec![0.0; n];
            for i in 0..j {
                let eidx = self.edge_index(i, j);
                let weights = gumbel_softmax(&self.alphas[eidx].logits, temperature, rng);

                // Mixed operation: weighted sum of all operations
                for (k, (op, w)) in self.ops.iter().zip(&weights).enumerate() {
                    let op_out = apply_operation(op, &node_outputs[i], &self.op_weights[eidx][k]);
                    for t in 0..n {
                        node_out[t] += w * op_out[t];
                    }
                }
            }
            node_outputs.push(node_out);
        }

        // Output: average of all intermediate node outputs
        let mut output = vec![0.0; n];
        for j in num_input..total {
            for t in 0..n {
                output[t] += node_outputs[j][t];
            }
        }
        let scale = 1.0 / self.num_nodes as f64;
        output.iter_mut().for_each(|v| *v *= scale);
        output
    }

    /// Forward pass with standard softmax (no Gumbel noise), for testing.
    pub fn forward_deterministic(&self, input0: &[f64], input1: &[f64]) -> Vec<f64> {
        let n = input0.len();
        let total = self.total_nodes();
        let num_input = self.num_input_nodes();

        let mut node_outputs: Vec<Vec<f64>> = Vec::with_capacity(total);
        node_outputs.push(input0.to_vec());
        node_outputs.push(input1.to_vec());

        for j in num_input..total {
            let mut node_out = vec![0.0; n];
            for i in 0..j {
                let eidx = self.edge_index(i, j);
                let weights = softmax(&self.alphas[eidx].logits);

                for (k, (op, w)) in self.ops.iter().zip(&weights).enumerate() {
                    let op_out = apply_operation(op, &node_outputs[i], &self.op_weights[eidx][k]);
                    for t in 0..n {
                        node_out[t] += w * op_out[t];
                    }
                }
            }
            node_outputs.push(node_out);
        }

        let mut output = vec![0.0; n];
        for j in num_input..total {
            for t in 0..n {
                output[t] += node_outputs[j][t];
            }
        }
        let scale = 1.0 / self.num_nodes as f64;
        output.iter_mut().for_each(|v| *v *= scale);
        output
    }

    /// Entropy of architecture weights (averaged over all edges).
    /// Low entropy = architecture is converging; very low = potential collapse.
    pub fn architecture_entropy(&self) -> f64 {
        let mut total_entropy = 0.0;
        for alpha in &self.alphas {
            let probs = alpha.probabilities();
            for p in &probs {
                if *p > 1e-10 {
                    total_entropy -= p * p.ln();
                }
            }
        }
        total_entropy / self.alphas.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Architecture discretization
// ---------------------------------------------------------------------------

/// A single edge in a discrete (final) architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteEdge {
    pub from: usize,
    pub to: usize,
    pub op: OpType,
    pub strength: f64,
}

/// A fully discretized architecture extracted from DARTS search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteArchitecture {
    pub edges: Vec<DiscreteEdge>,
    pub num_nodes: usize,
}

impl fmt::Display for DiscreteArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Discrete Architecture ({} nodes):", self.num_nodes)?;
        for edge in &self.edges {
            writeln!(
                f,
                "  {} -> {}: {} (strength: {:.4})",
                edge.from, edge.to, edge.op, edge.strength
            )?;
        }
        Ok(())
    }
}

/// Discretize a DARTS cell into a final architecture.
///
/// For each intermediate node, selects the top-2 incoming edges
/// (by strength of the best operation on that edge).
pub fn discretize_architecture(cell: &DartsCell) -> DiscreteArchitecture {
    let num_input = cell.num_input_nodes();
    let total = cell.total_nodes();
    let ops = &cell.ops;
    let mut edges = Vec::new();

    for j in num_input..total {
        // Collect all incoming edges for node j with their best operation
        let mut candidates: Vec<(usize, usize, f64)> = Vec::new(); // (from, op_idx, strength)
        for i in 0..j {
            let eidx = cell.edge_index(i, j);
            let best_op = cell.alphas[eidx].best_op();
            let strength = cell.alphas[eidx].best_strength();
            candidates.push((i, best_op, strength));
        }

        // Sort by strength descending, keep top-2
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        for &(from, op_idx, strength) in candidates.iter().take(2) {
            // Skip Zero operations
            if ops[op_idx] != OpType::Zero {
                edges.push(DiscreteEdge {
                    from,
                    to: j,
                    op: ops[op_idx],
                    strength,
                });
            }
        }
    }

    DiscreteArchitecture {
        edges,
        num_nodes: cell.num_nodes,
    }
}

// ---------------------------------------------------------------------------
// Bi-level optimization (DARTS search)
// ---------------------------------------------------------------------------

/// Configuration for the DARTS search.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of intermediate nodes in each cell
    pub num_nodes: usize,
    /// Number of search epochs
    pub num_epochs: usize,
    /// Learning rate for network weights
    pub weight_lr: f64,
    /// Learning rate for architecture parameters
    pub arch_lr: f64,
    /// Initial Gumbel-Softmax temperature
    pub initial_temperature: f64,
    /// Final Gumbel-Softmax temperature
    pub final_temperature: f64,
    /// Number of warm-up epochs (weight-only training before arch updates)
    pub warmup_epochs: usize,
    /// Entropy threshold for early stopping (collapse detection)
    pub min_entropy: f64,
    /// L2 regularization on skip/zero architecture weights
    pub skip_penalty: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            num_nodes: 4,
            num_epochs: 50,
            weight_lr: 0.01,
            arch_lr: 0.003,
            initial_temperature: 1.0,
            final_temperature: 0.1,
            warmup_epochs: 5,
            min_entropy: 0.1,
            skip_penalty: 0.01,
        }
    }
}

/// Result of the DARTS search process.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub cell: DartsCell,
    pub architecture: DiscreteArchitecture,
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub entropies: Vec<f64>,
    pub stopped_early: bool,
}

/// Mean squared error loss.
pub fn mse_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len().min(targets.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = predictions[..n]
        .iter()
        .zip(&targets[..n])
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    sum / n as f64
}

/// Numerical gradient of loss w.r.t. a parameter.
fn numerical_gradient(
    cell: &mut DartsCell,
    input0: &[f64],
    input1: &[f64],
    targets: &[f64],
    param_type: ParamType,
    edge_idx: usize,
    param_idx: usize,
    sub_idx: usize,
    eps: f64,
) -> f64 {
    // Forward with +eps
    let original = get_param(cell, param_type, edge_idx, param_idx, sub_idx);
    set_param(cell, param_type, edge_idx, param_idx, sub_idx, original + eps);
    let out_plus = cell.forward_deterministic(input0, input1);
    let loss_plus = mse_loss(&out_plus, targets);

    // Forward with -eps
    set_param(cell, param_type, edge_idx, param_idx, sub_idx, original - eps);
    let out_minus = cell.forward_deterministic(input0, input1);
    let loss_minus = mse_loss(&out_minus, targets);

    // Restore
    set_param(cell, param_type, edge_idx, param_idx, sub_idx, original);

    (loss_plus - loss_minus) / (2.0 * eps)
}

#[derive(Debug, Clone, Copy)]
enum ParamType {
    Alpha,
    Weight,
}

fn get_param(
    cell: &DartsCell,
    ptype: ParamType,
    edge: usize,
    pidx: usize,
    sub: usize,
) -> f64 {
    match ptype {
        ParamType::Alpha => cell.alphas[edge].logits[pidx],
        ParamType::Weight => cell.op_weights[edge][pidx][sub],
    }
}

fn set_param(
    cell: &mut DartsCell,
    ptype: ParamType,
    edge: usize,
    pidx: usize,
    sub: usize,
    val: f64,
) {
    match ptype {
        ParamType::Alpha => cell.alphas[edge].logits[pidx] = val,
        ParamType::Weight => cell.op_weights[edge][pidx][sub] = val,
    }
}

/// Run the DARTS bi-level search.
///
/// - `train_data`: training sequences (1-D feature vector)
/// - `val_data`: validation sequences (1-D feature vector)
/// - `targets_train`: training targets (1-D)
/// - `targets_val`: validation targets (1-D)
/// - `config`: search configuration
pub fn darts_search(
    train_data: &[f64],
    val_data: &[f64],
    targets_train: &[f64],
    targets_val: &[f64],
    config: &SearchConfig,
) -> SearchResult {
    let mut rng = rand::thread_rng();
    let ops = default_op_space();
    let num_ops = ops.len();
    let mut cell = DartsCell::new(config.num_nodes, ops, &mut rng);
    let num_edges = cell.alphas.len();
    let max_kernel = 5;

    let mut train_losses = Vec::new();
    let mut val_losses = Vec::new();
    let mut entropies = Vec::new();
    let mut stopped_early = false;

    let eps = 1e-4;

    for epoch in 0..config.num_epochs {
        // Temperature annealing
        let progress = epoch as f64 / config.num_epochs as f64;
        let temperature = config.initial_temperature
            + (config.final_temperature - config.initial_temperature) * progress;

        // --- Step 1: Update network weights on training data ---
        let train_out = cell.forward(&train_data, &train_data, temperature, &mut rng);
        let t_loss = mse_loss(&train_out, targets_train);
        train_losses.push(t_loss);

        // Numerical gradient descent on operation weights
        for e in 0..num_edges {
            for op in 0..num_ops {
                for s in 0..max_kernel {
                    let grad = numerical_gradient(
                        &mut cell,
                        train_data,
                        train_data,
                        targets_train,
                        ParamType::Weight,
                        e,
                        op,
                        s,
                        eps,
                    );
                    cell.op_weights[e][op][s] -= config.weight_lr * grad;
                }
            }
        }

        // --- Step 2: Update architecture parameters on validation data ---
        if epoch >= config.warmup_epochs {
            let val_out = cell.forward_deterministic(val_data, val_data);
            let v_loss = mse_loss(&val_out, targets_val);
            val_losses.push(v_loss);

            for e in 0..num_edges {
                for op_idx in 0..num_ops {
                    let grad = numerical_gradient(
                        &mut cell,
                        val_data,
                        val_data,
                        targets_val,
                        ParamType::Alpha,
                        e,
                        op_idx,
                        0, // sub_idx not used for alpha
                        eps,
                    );

                    // Add penalty for skip/zero to prevent collapse
                    let penalty = match cell.ops[op_idx] {
                        OpType::Skip | OpType::Zero | OpType::MovingAverage { .. } => {
                            config.skip_penalty * cell.alphas[e].logits[op_idx]
                        }
                        _ => 0.0,
                    };

                    cell.alphas[e].logits[op_idx] -= config.arch_lr * (grad + penalty);
                }
            }
        } else {
            // During warmup, just record val loss
            let val_out = cell.forward_deterministic(val_data, val_data);
            let v_loss = mse_loss(&val_out, targets_val);
            val_losses.push(v_loss);
        }

        // --- Monitor entropy for collapse detection ---
        let entropy = cell.architecture_entropy();
        entropies.push(entropy);

        if epoch >= config.warmup_epochs && entropy < config.min_entropy {
            stopped_early = true;
            break;
        }
    }

    let architecture = discretize_architecture(&cell);

    SearchResult {
        cell,
        architecture,
        train_losses,
        val_losses,
        entropies,
        stopped_early,
    }
}

// ---------------------------------------------------------------------------
// Assessment of final (discrete) architecture
// ---------------------------------------------------------------------------

/// Assess a discrete architecture on test data.
/// Uses only the selected operations (no mixing).
pub fn assess_discrete(
    arch: &DiscreteArchitecture,
    cell: &DartsCell,
    input: &[f64],
    targets: &[f64],
) -> f64 {
    let n = input.len();
    let total = cell.total_nodes();
    let num_input = cell.num_input_nodes();

    let mut node_outputs: Vec<Vec<f64>> = Vec::with_capacity(total);
    node_outputs.push(input.to_vec());
    node_outputs.push(input.to_vec());

    for j in num_input..total {
        let mut node_out = vec![0.0; n];
        // Find edges targeting this node
        for edge in &arch.edges {
            if edge.to == j {
                let eidx = cell.edge_index(edge.from, j);
                let op_idx = cell
                    .ops
                    .iter()
                    .position(|o| *o == edge.op)
                    .unwrap_or(0);
                let op_out =
                    apply_operation(&edge.op, &node_outputs[edge.from], &cell.op_weights[eidx][op_idx]);
                for t in 0..n {
                    node_out[t] += op_out[t];
                }
            }
        }
        node_outputs.push(node_out);
    }

    let mut output = vec![0.0; n];
    for j in num_input..total {
        for t in 0..n {
            output[t] += node_outputs[j][t];
        }
    }
    let scale = 1.0 / arch.num_nodes as f64;
    output.iter_mut().for_each(|v| *v *= scale);

    mse_loss(&output, targets)
}

/// Simple baseline: moving average predictor.
pub fn baseline_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    moving_average(data, window)
}

// ---------------------------------------------------------------------------
// Feature engineering for financial data
// ---------------------------------------------------------------------------

/// Compute log returns from a price series.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

/// Compute rolling standard deviation (volatility proxy).
pub fn rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = if i >= window { i - window + 1 } else { 0 };
        let slice = &data[start..=i];
        let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
        let var: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        result[i] = var.sqrt();
    }
    result
}

/// Normalize data using rolling z-score.
pub fn rolling_zscore(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let start = if i >= window { i - window + 1 } else { 0 };
        let slice = &data[start..=i];
        let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
        let var: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        let std = var.sqrt();
        result[i] = if std > 1e-10 {
            (data[i] - mean) / std
        } else {
            0.0
        };
    }
    result
}

/// Compute momentum (rate of change) over a given period.
pub fn momentum(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut result = vec![0.0; n];
    for i in period..n {
        result[i] = (prices[i] - prices[i - period]) / prices[i - period];
    }
    result
}

// ---------------------------------------------------------------------------
// Bybit API integration
// ---------------------------------------------------------------------------

/// A single OHLCV candlestick from Bybit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Raw Bybit API response structures.
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
    list: Vec<Vec<serde_json::Value>>,
}

/// Fetch kline (candlestick) data from Bybit public API.
///
/// - `symbol`: e.g., "BTCUSDT"
/// - `interval`: e.g., "60" for 1-hour candles
/// - `limit`: number of candles (max 1000)
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .header("User-Agent", "darts-trading-rust/0.1")
        .send()
        .context("Failed to send request to Bybit API")?
        .json()
        .context("Failed to parse Bybit API response")?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {} (code {})", resp.ret_msg, resp.ret_code);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    timestamp: row[0].as_str()?.parse().ok()?,
                    open: row[1].as_str()?.parse().ok()?,
                    high: row[2].as_str()?.parse().ok()?,
                    low: row[3].as_str()?.parse().ok()?,
                    close: row[4].as_str()?.parse().ok()?,
                    volume: row[5].as_str()?.parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Convert candles to a feature matrix suitable for DARTS.
///
/// Returns (features, close_prices) where features are normalized log returns.
pub fn candles_to_features(candles: &[Candle]) -> (Vec<f64>, Vec<f64>) {
    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let returns = log_returns(&closes);
    let normalized = rolling_zscore(&returns, 20);
    (normalized, closes)
}

/// Split data chronologically into train/val/test.
/// Returns (train, val, test) slices.
pub fn split_data(data: &[f64], train_frac: f64, val_frac: f64) -> (&[f64], &[f64], &[f64]) {
    let n = data.len();
    let train_end = (n as f64 * train_frac) as usize;
    let val_end = (n as f64 * (train_frac + val_frac)) as usize;
    (&data[..train_end], &data[train_end..val_end], &data[val_end..])
}

// ---------------------------------------------------------------------------
// Unit tests
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
    fn test_gumbel_softmax_sums_to_one() {
        let mut rng = rand::thread_rng();
        let logits = vec![0.0, 1.0, -1.0, 2.0];
        let result = gumbel_softmax(&logits, 0.5, &mut rng);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_softmax_low_temp_is_sharp() {
        let mut rng = rand::thread_rng();
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        let result = gumbel_softmax(&logits, 0.01, &mut rng);
        // With very low temperature, the max logit should dominate
        assert!(result[2] > 0.9);
    }

    #[test]
    fn test_conv1d_identity_kernel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![0.0, 1.0, 0.0]; // identity-like kernel
        let output = conv1d(&input, &kernel, 3, 1);
        assert_eq!(output.len(), input.len());
        // Middle elements should be approximately preserved
        assert!((output[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_moving_average() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = moving_average(&input, 3);
        assert_eq!(output.len(), 5);
        // At index 2, avg of [1,2,3] = 2.0
        assert!((output[2] - 2.0).abs() < 1e-10);
        // At index 4, avg of [3,4,5] = 4.0
        assert!((output[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_skip_operation() {
        let input = vec![1.0, 2.0, 3.0];
        let output = apply_operation(&OpType::Skip, &input, &[]);
        assert_eq!(output, input);
    }

    #[test]
    fn test_zero_operation() {
        let input = vec![1.0, 2.0, 3.0];
        let output = apply_operation(&OpType::Zero, &input, &[]);
        assert_eq!(output, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 110.0, 105.0];
        let returns = log_returns(&prices);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - (110.0f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[1] - (105.0f64 / 110.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_std() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let std = rolling_std(&data, 3);
        // Constant data should have zero std
        assert!(std[3].abs() < 1e-10);
    }

    #[test]
    fn test_rolling_zscore() {
        let data = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let z = rolling_zscore(&data, 3);
        for v in &z {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_momentum() {
        let prices = vec![100.0, 110.0, 120.0, 115.0];
        let mom = momentum(&prices, 2);
        assert!((mom[2] - 0.2).abs() < 1e-10); // (120-100)/100
    }

    #[test]
    fn test_darts_cell_creation() {
        let mut rng = rand::thread_rng();
        let ops = default_op_space();
        let cell = DartsCell::new(4, ops.clone(), &mut rng);
        assert_eq!(cell.num_nodes, 4);
        assert_eq!(cell.total_nodes(), 6); // 2 inputs + 4 intermediate
        // Number of edges: node2 gets 2 edges, node3 gets 3, node4 gets 4, node5 gets 5 => 14
        assert_eq!(cell.alphas.len(), 14);
    }

    #[test]
    fn test_darts_cell_forward() {
        let mut rng = rand::thread_rng();
        let ops = default_op_space();
        let cell = DartsCell::new(2, ops, &mut rng);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = cell.forward(&input, &input, 1.0, &mut rng);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_discretize_architecture() {
        let mut rng = rand::thread_rng();
        let ops = default_op_space();
        let mut cell = DartsCell::new(2, ops, &mut rng);
        // Boost Conv1D-3 on the first edge
        cell.alphas[0].logits[0] = 5.0;
        let arch = discretize_architecture(&cell);
        assert!(!arch.edges.is_empty());
        // First edge should favor Conv1D-3
        let first_to_node2: Vec<_> = arch.edges.iter().filter(|e| e.to == 2).collect();
        assert!(!first_to_node2.is_empty());
    }

    #[test]
    fn test_architecture_entropy() {
        let mut rng = rand::thread_rng();
        let ops = default_op_space();
        let cell = DartsCell::new(2, ops, &mut rng);
        let entropy = cell.architecture_entropy();
        // Uniform initialization should give maximum entropy
        assert!(entropy > 1.0);
    }

    #[test]
    fn test_mse_loss() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        assert!(mse_loss(&pred, &target).abs() < 1e-10);

        let pred2 = vec![2.0, 3.0, 4.0];
        assert!((mse_loss(&pred2, &target) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_data() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let (train, val, test) = split_data(&data, 0.6, 0.2);
        assert_eq!(train.len(), 60);
        assert_eq!(val.len(), 20);
        assert_eq!(test.len(), 20);
    }

    #[test]
    fn test_darts_search_runs() {
        // Small test to verify the search loop runs without panic
        let train: Vec<f64> = (0..20).map(|x| (x as f64 * 0.1).sin()).collect();
        let val: Vec<f64> = (20..30).map(|x| (x as f64 * 0.1).sin()).collect();
        let targets_train: Vec<f64> = (1..21).map(|x| (x as f64 * 0.1).sin()).collect();
        let targets_val: Vec<f64> = (21..31).map(|x| (x as f64 * 0.1).sin()).collect();

        let config = SearchConfig {
            num_nodes: 2,
            num_epochs: 3,
            warmup_epochs: 1,
            ..Default::default()
        };

        let result = darts_search(&train, &val, &targets_train, &targets_val, &config);
        assert!(!result.train_losses.is_empty());
        assert!(!result.architecture.edges.is_empty());
    }

    #[test]
    fn test_op_display() {
        assert_eq!(
            format!("{}", OpType::Conv1D { kernel_size: 3 }),
            "Conv1D(k=3)"
        );
        assert_eq!(
            format!(
                "{}",
                OpType::DilatedConv1D {
                    kernel_size: 3,
                    dilation: 2
                }
            ),
            "DilConv1D(k=3, d=2)"
        );
        assert_eq!(format!("{}", OpType::Skip), "Skip");
    }

    #[test]
    fn test_edge_alpha_best_op() {
        let mut alpha = EdgeAlpha::new(4);
        alpha.logits = vec![1.0, 5.0, 2.0, 0.5];
        assert_eq!(alpha.best_op(), 1);
        assert!(alpha.best_strength() > 0.5);
    }

    #[test]
    fn test_baseline_moving_average() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let baseline = baseline_moving_average(&data, 3);
        assert_eq!(baseline.len(), 5);
        assert!((baseline[4] - 7.0).abs() < 1e-10); // avg(5,7,9) = 7
    }

    #[test]
    fn test_dilated_conv() {
        let input = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0];
        let kernel = vec![1.0, 0.0, 1.0];
        let output = conv1d(&input, &kernel, 3, 2);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_assess_discrete() {
        let mut rng = rand::thread_rng();
        let ops = default_op_space();
        let cell = DartsCell::new(2, ops, &mut rng);
        let arch = discretize_architecture(&cell);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let loss = assess_discrete(&arch, &cell, &input, &targets);
        assert!(loss >= 0.0);
    }
}
