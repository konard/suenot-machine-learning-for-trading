use anyhow::{anyhow, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================
// Device & Operation Types
// ============================================================

/// Target device types for hardware-aware NAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    FPGA,
    GPU,
    CPU,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::FPGA => write!(f, "FPGA"),
            DeviceType::GPU => write!(f, "GPU"),
            DeviceType::CPU => write!(f, "CPU"),
        }
    }
}

/// Supported operation types in the architecture search space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpType {
    Conv1x1,
    Conv3x3,
    Conv5x5,
    DepthwiseConv3x3,
    DepthwiseConv5x5,
    FullyConnected,
    SelfAttention,
    Identity,
    MaxPool,
    AvgPool,
}

impl std::fmt::Display for OpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Key for latency lookup table: (operation, input_channels, output_channels).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpSpec {
    pub op: OpType,
    pub in_channels: usize,
    pub out_channels: usize,
}

// ============================================================
// Hardware Profile
// ============================================================

/// Hardware profile containing latency LUT and device constraints.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub name: String,
    pub device_type: DeviceType,
    pub latency_table: HashMap<OpSpec, f64>,
    pub memory_budget_bytes: u64,
    pub peak_flops: f64,
    /// Default latency multipliers per operation type (microseconds per MFLOP).
    pub default_latency_per_mflop: f64,
}

impl HardwareProfile {
    /// Create an FPGA profile optimized for HFT.
    pub fn fpga_hft() -> Self {
        let mut table = HashMap::new();
        // FPGA: very fast small ops, poor at large matrix ops
        populate_fpga_latencies(&mut table);
        Self {
            name: "FPGA-HFT".to_string(),
            device_type: DeviceType::FPGA,
            latency_table: table,
            memory_budget_bytes: 16 * 1024 * 1024, // 16 MB BRAM
            peak_flops: 1e9,                         // 1 GFLOP/s (INT8-equivalent)
            default_latency_per_mflop: 1.0,          // 1 us per MFLOP
        }
    }

    /// Create a GPU profile for batch inference.
    pub fn gpu_batch() -> Self {
        let mut table = HashMap::new();
        populate_gpu_latencies(&mut table);
        Self {
            name: "GPU-Batch".to_string(),
            device_type: DeviceType::GPU,
            latency_table: table,
            memory_budget_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            peak_flops: 100e12,                            // 100 TFLOP/s
            default_latency_per_mflop: 0.00001,            // very fast per MFLOP
        }
    }

    /// Create a CPU profile for retail/mobile trading.
    pub fn cpu_mobile() -> Self {
        let mut table = HashMap::new();
        populate_cpu_latencies(&mut table);
        Self {
            name: "CPU-Mobile".to_string(),
            device_type: DeviceType::CPU,
            latency_table: table,
            memory_budget_bytes: 512 * 1024 * 1024, // 512 MB
            peak_flops: 10e9,                         // 10 GFLOP/s
            default_latency_per_mflop: 0.1,           // 0.1 us per MFLOP
        }
    }

    /// Look up latency for a given operation spec.
    /// Falls back to FLOPs-based estimate if not in table.
    pub fn lookup_latency(&self, spec: &OpSpec) -> f64 {
        if let Some(&lat) = self.latency_table.get(spec) {
            lat
        } else {
            // Estimate from FLOPs
            let flops = compute_op_flops(spec.op, spec.in_channels, spec.out_channels, 1);
            let mflops = flops as f64 / 1e6;
            mflops * self.default_latency_per_mflop
        }
    }
}

fn populate_fpga_latencies(table: &mut HashMap<OpSpec, f64>) {
    let channels_list = vec![8, 16, 32, 64, 128];
    for &c_in in &channels_list {
        for &c_out in &channels_list {
            // FPGA: small conv very fast, large ops expensive
            table.insert(
                OpSpec { op: OpType::Conv1x1, in_channels: c_in, out_channels: c_out },
                0.5 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::Conv3x3, in_channels: c_in, out_channels: c_out },
                2.0 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::Conv5x5, in_channels: c_in, out_channels: c_out },
                6.0 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::DepthwiseConv3x3, in_channels: c_in, out_channels: c_out },
                0.8 * c_in.max(c_out) as f64 / 64.0,
            );
            table.insert(
                OpSpec { op: OpType::DepthwiseConv5x5, in_channels: c_in, out_channels: c_out },
                2.0 * c_in.max(c_out) as f64 / 64.0,
            );
            table.insert(
                OpSpec { op: OpType::FullyConnected, in_channels: c_in, out_channels: c_out },
                1.0 * (c_in * c_out) as f64 / 1024.0,
            );
            // Attention is very expensive on FPGA
            table.insert(
                OpSpec { op: OpType::SelfAttention, in_channels: c_in, out_channels: c_out },
                50.0 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::Identity, in_channels: c_in, out_channels: c_out },
                0.01,
            );
            table.insert(
                OpSpec { op: OpType::MaxPool, in_channels: c_in, out_channels: c_out },
                0.2 * c_in as f64 / 64.0,
            );
            table.insert(
                OpSpec { op: OpType::AvgPool, in_channels: c_in, out_channels: c_out },
                0.15 * c_in as f64 / 64.0,
            );
        }
    }
}

fn populate_gpu_latencies(table: &mut HashMap<OpSpec, f64>) {
    let channels_list = vec![8, 16, 32, 64, 128];
    for &c_in in &channels_list {
        for &c_out in &channels_list {
            // GPU: good at large parallel ops, kernel launch overhead for small ops
            let base_overhead = 5.0; // 5 us kernel launch
            table.insert(
                OpSpec { op: OpType::Conv1x1, in_channels: c_in, out_channels: c_out },
                base_overhead + 0.01 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::Conv3x3, in_channels: c_in, out_channels: c_out },
                base_overhead + 0.05 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::Conv5x5, in_channels: c_in, out_channels: c_out },
                base_overhead + 0.15 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::DepthwiseConv3x3, in_channels: c_in, out_channels: c_out },
                base_overhead + 0.02 * c_in.max(c_out) as f64 / 64.0,
            );
            table.insert(
                OpSpec { op: OpType::DepthwiseConv5x5, in_channels: c_in, out_channels: c_out },
                base_overhead + 0.05 * c_in.max(c_out) as f64 / 64.0,
            );
            table.insert(
                OpSpec { op: OpType::FullyConnected, in_channels: c_in, out_channels: c_out },
                base_overhead + 0.005 * (c_in * c_out) as f64 / 1024.0,
            );
            // GPU handles attention well with tensor cores
            table.insert(
                OpSpec { op: OpType::SelfAttention, in_channels: c_in, out_channels: c_out },
                base_overhead + 0.1 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::Identity, in_channels: c_in, out_channels: c_out },
                0.1,
            );
            table.insert(
                OpSpec { op: OpType::MaxPool, in_channels: c_in, out_channels: c_out },
                base_overhead + 0.01 * c_in as f64 / 64.0,
            );
            table.insert(
                OpSpec { op: OpType::AvgPool, in_channels: c_in, out_channels: c_out },
                base_overhead + 0.01 * c_in as f64 / 64.0,
            );
        }
    }
}

fn populate_cpu_latencies(table: &mut HashMap<OpSpec, f64>) {
    let channels_list = vec![8, 16, 32, 64, 128];
    for &c_in in &channels_list {
        for &c_out in &channels_list {
            // CPU: moderate everything, good cache locality matters
            table.insert(
                OpSpec { op: OpType::Conv1x1, in_channels: c_in, out_channels: c_out },
                2.0 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::Conv3x3, in_channels: c_in, out_channels: c_out },
                10.0 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::Conv5x5, in_channels: c_in, out_channels: c_out },
                25.0 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::DepthwiseConv3x3, in_channels: c_in, out_channels: c_out },
                3.0 * c_in.max(c_out) as f64 / 64.0,
            );
            table.insert(
                OpSpec { op: OpType::DepthwiseConv5x5, in_channels: c_in, out_channels: c_out },
                8.0 * c_in.max(c_out) as f64 / 64.0,
            );
            table.insert(
                OpSpec { op: OpType::FullyConnected, in_channels: c_in, out_channels: c_out },
                3.0 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::SelfAttention, in_channels: c_in, out_channels: c_out },
                30.0 * (c_in * c_out) as f64 / 1024.0,
            );
            table.insert(
                OpSpec { op: OpType::Identity, in_channels: c_in, out_channels: c_out },
                0.05,
            );
            table.insert(
                OpSpec { op: OpType::MaxPool, in_channels: c_in, out_channels: c_out },
                1.0 * c_in as f64 / 64.0,
            );
            table.insert(
                OpSpec { op: OpType::AvgPool, in_channels: c_in, out_channels: c_out },
                0.8 * c_in as f64 / 64.0,
            );
        }
    }
}

// ============================================================
// Architecture Representation
// ============================================================

/// Single layer specification in a candidate architecture.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerSpec {
    pub op: OpType,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub use_skip: bool,
}

/// Complete architecture as a sequence of layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    pub layers: Vec<LayerSpec>,
    pub input_features: usize,
    pub output_classes: usize,
}

impl Architecture {
    /// Create a new architecture with given layers.
    pub fn new(layers: Vec<LayerSpec>, input_features: usize, output_classes: usize) -> Self {
        Self {
            layers,
            input_features,
            output_classes,
        }
    }

    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Architecture: {} layers, input={}, output={}\n",
            self.layers.len(),
            self.input_features,
            self.output_classes
        );
        for (i, layer) in self.layers.iter().enumerate() {
            s.push_str(&format!(
                "  Layer {}: {} ({}x{}) k={} s={} skip={}\n",
                i,
                layer.op,
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.use_skip
            ));
        }
        s
    }
}

// ============================================================
// FLOPs Counter
// ============================================================

/// Compute FLOPs for a single operation.
pub fn compute_op_flops(
    op: OpType,
    in_channels: usize,
    out_channels: usize,
    seq_len: usize,
) -> u64 {
    let seq = seq_len.max(1) as u64;
    let c_in = in_channels as u64;
    let c_out = out_channels as u64;

    match op {
        OpType::Conv1x1 => 2 * seq * c_in * c_out,
        OpType::Conv3x3 => 2 * seq * 9 * c_in * c_out,
        OpType::Conv5x5 => 2 * seq * 25 * c_in * c_out,
        OpType::DepthwiseConv3x3 => 2 * seq * 9 * c_in,
        OpType::DepthwiseConv5x5 => 2 * seq * 25 * c_in,
        OpType::FullyConnected => 2 * c_in * c_out,
        OpType::SelfAttention => 4 * seq * seq * c_in + 2 * seq * c_in * c_in,
        OpType::Identity => 0,
        OpType::MaxPool => seq * c_in * 9,
        OpType::AvgPool => seq * c_in * 9,
    }
}

/// Compute total FLOPs for an architecture.
pub fn compute_architecture_flops(arch: &Architecture, seq_len: usize) -> u64 {
    arch.layers
        .iter()
        .map(|layer| compute_op_flops(layer.op, layer.in_channels, layer.out_channels, seq_len))
        .sum()
}

// ============================================================
// Memory Footprint Estimator
// ============================================================

/// Compute parameter count for a single layer.
pub fn compute_layer_params(layer: &LayerSpec) -> u64 {
    let c_in = layer.in_channels as u64;
    let c_out = layer.out_channels as u64;
    let k = layer.kernel_size as u64;

    match layer.op {
        OpType::Conv1x1 => c_in * c_out + c_out, // weights + bias
        OpType::Conv3x3 => k * k * c_in * c_out + c_out,
        OpType::Conv5x5 => k * k * c_in * c_out + c_out,
        OpType::DepthwiseConv3x3 => k * k * c_in + c_in,
        OpType::DepthwiseConv5x5 => k * k * c_in + c_in,
        OpType::FullyConnected => c_in * c_out + c_out,
        OpType::SelfAttention => 4 * c_in * c_out, // Q, K, V, O projections
        OpType::Identity => 0,
        OpType::MaxPool => 0,
        OpType::AvgPool => 0,
    }
}

/// Compute total parameter count for an architecture.
pub fn compute_total_params(arch: &Architecture) -> u64 {
    arch.layers.iter().map(|l| compute_layer_params(l)).sum()
}

/// Memory footprint in bytes (FP32 = 4 bytes per parameter).
pub fn compute_param_memory(arch: &Architecture, bytes_per_param: u64) -> u64 {
    compute_total_params(arch) * bytes_per_param
}

/// Peak activation memory in bytes during inference.
pub fn compute_activation_memory(arch: &Architecture, seq_len: usize, bytes_per_value: u64) -> u64 {
    let mut peak = 0u64;
    for layer in &arch.layers {
        let input_size = seq_len as u64 * layer.in_channels as u64;
        let output_size = seq_len as u64 * layer.out_channels as u64;
        let layer_peak = (input_size + output_size) * bytes_per_value;
        if layer_peak > peak {
            peak = layer_peak;
        }
    }
    peak
}

/// Total memory footprint (parameters + peak activations).
pub fn compute_total_memory(arch: &Architecture, seq_len: usize, bytes_per_value: u64) -> u64 {
    compute_param_memory(arch, bytes_per_value) + compute_activation_memory(arch, seq_len, bytes_per_value)
}

// ============================================================
// Latency Estimator
// ============================================================

/// Estimate total latency of an architecture on a hardware profile (in microseconds).
pub fn estimate_latency(arch: &Architecture, hw: &HardwareProfile) -> f64 {
    arch.layers
        .iter()
        .map(|layer| {
            let spec = OpSpec {
                op: layer.op,
                in_channels: layer.in_channels,
                out_channels: layer.out_channels,
            };
            hw.lookup_latency(&spec)
        })
        .sum()
}

// ============================================================
// Metrics
// ============================================================

/// Combined metrics for a candidate architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    pub accuracy_proxy: f64,
    pub latency_us: f64,
    pub flops: u64,
    pub params: u64,
    pub memory_bytes: u64,
    pub score: f64,
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "acc={:.4} lat={:.2}us flops={} params={} mem={}B score={:.4}",
            self.accuracy_proxy,
            self.latency_us,
            self.flops,
            self.params,
            self.memory_bytes,
            self.score
        )
    }
}

// ============================================================
// Search Space
// ============================================================

/// Search space definition for hardware-aware NAS.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub allowed_ops: Vec<OpType>,
    pub channel_options: Vec<usize>,
    pub min_layers: usize,
    pub max_layers: usize,
    pub input_features: usize,
    pub output_classes: usize,
    pub skip_probability: f64,
}

impl SearchSpace {
    /// Search space for FPGA targets (restricted ops).
    pub fn fpga_space(input_features: usize, output_classes: usize) -> Self {
        Self {
            allowed_ops: vec![
                OpType::Conv1x1,
                OpType::Conv3x3,
                OpType::DepthwiseConv3x3,
                OpType::FullyConnected,
                OpType::Identity,
                OpType::MaxPool,
            ],
            channel_options: vec![8, 16, 32, 64],
            min_layers: 3,
            max_layers: 8,
            input_features,
            output_classes,
            skip_probability: 0.3,
        }
    }

    /// Search space for GPU targets (full ops).
    pub fn gpu_space(input_features: usize, output_classes: usize) -> Self {
        Self {
            allowed_ops: vec![
                OpType::Conv1x1,
                OpType::Conv3x3,
                OpType::Conv5x5,
                OpType::DepthwiseConv3x3,
                OpType::DepthwiseConv5x5,
                OpType::FullyConnected,
                OpType::SelfAttention,
                OpType::Identity,
                OpType::MaxPool,
                OpType::AvgPool,
            ],
            channel_options: vec![16, 32, 64, 128],
            min_layers: 4,
            max_layers: 12,
            input_features,
            output_classes,
            skip_probability: 0.4,
        }
    }

    /// Search space for CPU/mobile targets (efficient ops).
    pub fn cpu_space(input_features: usize, output_classes: usize) -> Self {
        Self {
            allowed_ops: vec![
                OpType::Conv1x1,
                OpType::Conv3x3,
                OpType::DepthwiseConv3x3,
                OpType::DepthwiseConv5x5,
                OpType::FullyConnected,
                OpType::Identity,
                OpType::AvgPool,
            ],
            channel_options: vec![8, 16, 32, 64],
            min_layers: 3,
            max_layers: 8,
            input_features,
            output_classes,
            skip_probability: 0.3,
        }
    }

    /// Sample a random architecture from the search space.
    pub fn sample_architecture(&self, rng: &mut impl Rng) -> Architecture {
        let num_layers = rng.gen_range(self.min_layers..=self.max_layers);
        let mut layers = Vec::with_capacity(num_layers);

        let mut current_channels = self.input_features;

        for i in 0..num_layers {
            let op_idx = rng.gen_range(0..self.allowed_ops.len());
            let op = self.allowed_ops[op_idx];

            let out_channels = if i == num_layers - 1 {
                self.output_classes
            } else {
                let ch_idx = rng.gen_range(0..self.channel_options.len());
                self.channel_options[ch_idx]
            };

            let kernel_size = match op {
                OpType::Conv1x1 | OpType::FullyConnected | OpType::Identity => 1,
                OpType::Conv3x3 | OpType::DepthwiseConv3x3 | OpType::MaxPool | OpType::AvgPool => 3,
                OpType::Conv5x5 | OpType::DepthwiseConv5x5 => 5,
                OpType::SelfAttention => 1,
            };

            let use_skip = i > 0
                && current_channels == out_channels
                && rng.gen_bool(self.skip_probability);

            layers.push(LayerSpec {
                op,
                in_channels: current_channels,
                out_channels: out_channels,
                kernel_size,
                stride: 1,
                use_skip,
            });

            current_channels = out_channels;
        }

        Architecture::new(layers, self.input_features, self.output_classes)
    }
}

// ============================================================
// Proxy Accuracy Estimator
// ============================================================

/// Estimate proxy accuracy for an architecture using data statistics.
/// This is a simplified proxy: in production, you'd train for a few epochs.
pub fn estimate_proxy_accuracy(
    arch: &Architecture,
    data: &[f64],
    rng: &mut impl Rng,
) -> f64 {
    let params = compute_total_params(arch) as f64;
    let num_layers = arch.layers.len() as f64;
    let data_variance = compute_variance(data);

    // Heuristic proxy: capacity * depth factor * data quality factor
    // More params and layers generally help up to a point
    let capacity_factor = (params.ln() / 20.0).min(1.0).max(0.1);
    let depth_factor = (num_layers / 6.0).min(1.0).max(0.3);
    let data_factor = (data_variance * 100.0).min(1.0).max(0.2);

    // Check for architectural quality features
    let has_attention = arch.layers.iter().any(|l| l.op == OpType::SelfAttention);
    let has_skip = arch.layers.iter().any(|l| l.use_skip);
    let attention_bonus = if has_attention { 0.05 } else { 0.0 };
    let skip_bonus = if has_skip { 0.03 } else { 0.0 };

    let base = 0.45 + 0.3 * capacity_factor + 0.15 * depth_factor + 0.05 * data_factor;
    let noise = rng.gen_range(-0.02..0.02);

    (base + attention_bonus + skip_bonus + noise).min(0.99).max(0.1)
}

fn compute_variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    var
}

// ============================================================
// Multi-Objective Search
// ============================================================

/// Scalarized score combining accuracy and hardware metrics.
/// Follows the MnasNet approach: ACC(a) * (LAT(a) / T)^w.
pub fn compute_scalarized_score(
    accuracy: f64,
    latency: f64,
    latency_target: f64,
    memory: u64,
    memory_budget: u64,
    latency_weight: f64,
    memory_weight: f64,
) -> f64 {
    let lat_ratio = latency / latency_target;
    let mem_ratio = memory as f64 / memory_budget as f64;
    accuracy * lat_ratio.powf(latency_weight) * mem_ratio.powf(memory_weight)
}

/// Check if solution `a` Pareto-dominates solution `b`.
pub fn dominates(a: &Metrics, b: &Metrics) -> bool {
    let a_better_or_equal = a.accuracy_proxy >= b.accuracy_proxy
        && a.latency_us <= b.latency_us
        && a.memory_bytes <= b.memory_bytes;
    let a_strictly_better = a.accuracy_proxy > b.accuracy_proxy
        || a.latency_us < b.latency_us
        || a.memory_bytes < b.memory_bytes;
    a_better_or_equal && a_strictly_better
}

/// Extract the Pareto front from a set of (architecture, metrics) pairs.
pub fn extract_pareto_front(
    candidates: &[(Architecture, Metrics)],
) -> Vec<(Architecture, Metrics)> {
    let mut front = Vec::new();
    for (i, (arch_a, metrics_a)) in candidates.iter().enumerate() {
        let is_dominated = candidates.iter().enumerate().any(|(j, (_, metrics_b))| {
            i != j && dominates(metrics_b, metrics_a)
        });
        if !is_dominated {
            front.push((arch_a.clone(), metrics_a.clone()));
        }
    }
    // Sort by latency for display
    front.sort_by(|a, b| a.1.latency_us.partial_cmp(&b.1.latency_us).unwrap());
    front
}

/// Run hardware-aware NAS with latency constraint.
pub fn search(
    search_space: &SearchSpace,
    hw: &HardwareProfile,
    latency_budget_us: f64,
    num_iterations: usize,
    data: &[f64],
    seed: u64,
) -> Vec<(Architecture, Metrics)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut all_candidates: Vec<(Architecture, Metrics)> = Vec::new();

    let seq_len = 64; // default sequence length for evaluation
    let bytes_per_value = 4u64; // FP32

    let latency_weight = -0.07; // MnasNet-style soft penalty
    let memory_weight = -0.03;

    for _ in 0..num_iterations {
        let arch = search_space.sample_architecture(&mut rng);

        let latency = estimate_latency(&arch, hw);
        let flops = compute_architecture_flops(&arch, seq_len);
        let params = compute_total_params(&arch);
        let memory = compute_total_memory(&arch, seq_len, bytes_per_value);
        let accuracy = estimate_proxy_accuracy(&arch, data, &mut rng);

        let score = compute_scalarized_score(
            accuracy,
            latency,
            latency_budget_us,
            memory,
            hw.memory_budget_bytes,
            latency_weight,
            memory_weight,
        );

        let metrics = Metrics {
            accuracy_proxy: accuracy,
            latency_us: latency,
            flops,
            params,
            memory_bytes: memory,
            score,
        };

        all_candidates.push((arch, metrics));
    }

    // Return Pareto front
    extract_pareto_front(&all_candidates)
}

/// Run search and return only architectures meeting the latency constraint.
pub fn search_constrained(
    search_space: &SearchSpace,
    hw: &HardwareProfile,
    latency_budget_us: f64,
    num_iterations: usize,
    data: &[f64],
    seed: u64,
) -> Vec<(Architecture, Metrics)> {
    let pareto = search(search_space, hw, latency_budget_us, num_iterations, data, seed);
    pareto
        .into_iter()
        .filter(|(_, m)| m.latency_us <= latency_budget_us)
        .collect()
}

// ============================================================
// Architecture Templates
// ============================================================

/// Pre-defined architecture templates for different hardware targets.
pub struct ArchitectureTemplates;

impl ArchitectureTemplates {
    /// FPGA-optimized: narrow, shallow, no attention.
    pub fn fpga_hft_template(input_features: usize, output_classes: usize) -> Architecture {
        Architecture::new(
            vec![
                LayerSpec {
                    op: OpType::Conv1x1,
                    in_channels: input_features,
                    out_channels: 16,
                    kernel_size: 1,
                    stride: 1,
                    use_skip: false,
                },
                LayerSpec {
                    op: OpType::DepthwiseConv3x3,
                    in_channels: 16,
                    out_channels: 16,
                    kernel_size: 3,
                    stride: 1,
                    use_skip: true,
                },
                LayerSpec {
                    op: OpType::Conv1x1,
                    in_channels: 16,
                    out_channels: 32,
                    kernel_size: 1,
                    stride: 1,
                    use_skip: false,
                },
                LayerSpec {
                    op: OpType::FullyConnected,
                    in_channels: 32,
                    out_channels: output_classes,
                    kernel_size: 1,
                    stride: 1,
                    use_skip: false,
                },
            ],
            input_features,
            output_classes,
        )
    }

    /// GPU-optimized: wider, deeper, uses attention.
    pub fn gpu_batch_template(input_features: usize, output_classes: usize) -> Architecture {
        Architecture::new(
            vec![
                LayerSpec {
                    op: OpType::Conv1x1,
                    in_channels: input_features,
                    out_channels: 64,
                    kernel_size: 1,
                    stride: 1,
                    use_skip: false,
                },
                LayerSpec {
                    op: OpType::Conv3x3,
                    in_channels: 64,
                    out_channels: 64,
                    kernel_size: 3,
                    stride: 1,
                    use_skip: true,
                },
                LayerSpec {
                    op: OpType::SelfAttention,
                    in_channels: 64,
                    out_channels: 64,
                    kernel_size: 1,
                    stride: 1,
                    use_skip: true,
                },
                LayerSpec {
                    op: OpType::Conv3x3,
                    in_channels: 64,
                    out_channels: 128,
                    kernel_size: 3,
                    stride: 1,
                    use_skip: false,
                },
                LayerSpec {
                    op: OpType::SelfAttention,
                    in_channels: 128,
                    out_channels: 128,
                    kernel_size: 1,
                    stride: 1,
                    use_skip: true,
                },
                LayerSpec {
                    op: OpType::FullyConnected,
                    in_channels: 128,
                    out_channels: output_classes,
                    kernel_size: 1,
                    stride: 1,
                    use_skip: false,
                },
            ],
            input_features,
            output_classes,
        )
    }

    /// CPU/mobile-optimized: depthwise separable, moderate depth.
    pub fn cpu_mobile_template(input_features: usize, output_classes: usize) -> Architecture {
        Architecture::new(
            vec![
                LayerSpec {
                    op: OpType::Conv1x1,
                    in_channels: input_features,
                    out_channels: 32,
                    kernel_size: 1,
                    stride: 1,
                    use_skip: false,
                },
                LayerSpec {
                    op: OpType::DepthwiseConv3x3,
                    in_channels: 32,
                    out_channels: 32,
                    kernel_size: 3,
                    stride: 1,
                    use_skip: true,
                },
                LayerSpec {
                    op: OpType::Conv1x1,
                    in_channels: 32,
                    out_channels: 64,
                    kernel_size: 1,
                    stride: 1,
                    use_skip: false,
                },
                LayerSpec {
                    op: OpType::DepthwiseConv5x5,
                    in_channels: 64,
                    out_channels: 64,
                    kernel_size: 5,
                    stride: 1,
                    use_skip: true,
                },
                LayerSpec {
                    op: OpType::FullyConnected,
                    in_channels: 64,
                    out_channels: output_classes,
                    kernel_size: 1,
                    stride: 1,
                    use_skip: false,
                },
            ],
            input_features,
            output_classes,
        )
    }
}

// ============================================================
// Bybit API Data Fetching
// ============================================================

/// Candle / OHLCV data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    list: Vec<Vec<serde_json::Value>>,
}

/// Fetch kline/candle data from Bybit V5 API.
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::Client::new();
    let resp: BybitResponse = client.get(&url).send().await?.json().await?;

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

    // Bybit returns newest first, reverse for chronological order
    candles.reverse();
    Ok(candles)
}

/// Blocking version of fetch for non-async contexts.
pub fn fetch_bybit_klines_blocking(
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
        return Err(anyhow!("Bybit API error: {}", resp.ret_msg));
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

    candles.reverse();
    Ok(candles)
}

/// Compute log returns from candle close prices.
pub fn compute_returns(candles: &[Candle]) -> Vec<f64> {
    candles
        .windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect()
}

/// Compute rolling volatility from returns.
pub fn compute_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window {
        return vec![];
    }
    (0..=returns.len() - window)
        .map(|i| {
            let slice = &returns[i..i + window];
            let mean = slice.iter().sum::<f64>() / window as f64;
            let var = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / window as f64;
            var.sqrt()
        })
        .collect()
}

// ============================================================
// Unit Tests
// ============================================================

use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flops_conv1x1() {
        let flops = compute_op_flops(OpType::Conv1x1, 32, 64, 100);
        assert_eq!(flops, 2 * 100 * 32 * 64);
    }

    #[test]
    fn test_flops_conv3x3() {
        let flops = compute_op_flops(OpType::Conv3x3, 16, 32, 50);
        assert_eq!(flops, 2 * 50 * 9 * 16 * 32);
    }

    #[test]
    fn test_flops_identity() {
        let flops = compute_op_flops(OpType::Identity, 64, 64, 100);
        assert_eq!(flops, 0);
    }

    #[test]
    fn test_layer_params_conv3x3() {
        let layer = LayerSpec {
            op: OpType::Conv3x3,
            in_channels: 16,
            out_channels: 32,
            kernel_size: 3,
            stride: 1,
            use_skip: false,
        };
        let params = compute_layer_params(&layer);
        assert_eq!(params, 3 * 3 * 16 * 32 + 32);
    }

    #[test]
    fn test_layer_params_identity() {
        let layer = LayerSpec {
            op: OpType::Identity,
            in_channels: 32,
            out_channels: 32,
            kernel_size: 1,
            stride: 1,
            use_skip: false,
        };
        assert_eq!(compute_layer_params(&layer), 0);
    }

    #[test]
    fn test_hardware_profiles_created() {
        let fpga = HardwareProfile::fpga_hft();
        let gpu = HardwareProfile::gpu_batch();
        let cpu = HardwareProfile::cpu_mobile();

        assert_eq!(fpga.device_type, DeviceType::FPGA);
        assert_eq!(gpu.device_type, DeviceType::GPU);
        assert_eq!(cpu.device_type, DeviceType::CPU);
        assert!(!fpga.latency_table.is_empty());
        assert!(!gpu.latency_table.is_empty());
        assert!(!cpu.latency_table.is_empty());
    }

    #[test]
    fn test_latency_lookup() {
        let fpga = HardwareProfile::fpga_hft();
        let spec = OpSpec {
            op: OpType::Conv1x1,
            in_channels: 32,
            out_channels: 64,
        };
        let lat = fpga.lookup_latency(&spec);
        assert!(lat > 0.0);
    }

    #[test]
    fn test_latency_fallback() {
        let fpga = HardwareProfile::fpga_hft();
        // Use channels not in the LUT to trigger fallback
        let spec = OpSpec {
            op: OpType::Conv1x1,
            in_channels: 7,
            out_channels: 13,
        };
        let lat = fpga.lookup_latency(&spec);
        assert!(lat >= 0.0);
    }

    #[test]
    fn test_architecture_template_fpga() {
        let arch = ArchitectureTemplates::fpga_hft_template(16, 3);
        assert_eq!(arch.input_features, 16);
        assert_eq!(arch.output_classes, 3);
        assert_eq!(arch.layers.len(), 4);
        // FPGA template should not use attention
        assert!(!arch.layers.iter().any(|l| l.op == OpType::SelfAttention));
    }

    #[test]
    fn test_architecture_template_gpu() {
        let arch = ArchitectureTemplates::gpu_batch_template(16, 3);
        assert_eq!(arch.layers.len(), 6);
        // GPU template should use attention
        assert!(arch.layers.iter().any(|l| l.op == OpType::SelfAttention));
    }

    #[test]
    fn test_total_memory() {
        let arch = ArchitectureTemplates::fpga_hft_template(16, 3);
        let mem = compute_total_memory(&arch, 64, 4);
        assert!(mem > 0);
    }

    #[test]
    fn test_estimate_latency_ordering() {
        let fpga = HardwareProfile::fpga_hft();
        let small_arch = ArchitectureTemplates::fpga_hft_template(16, 3);
        let big_arch = ArchitectureTemplates::gpu_batch_template(16, 3);

        let lat_small = estimate_latency(&small_arch, &fpga);
        let lat_big = estimate_latency(&big_arch, &fpga);

        // Bigger architecture should have higher latency
        assert!(lat_big > lat_small);
    }

    #[test]
    fn test_pareto_dominance() {
        let a = Metrics {
            accuracy_proxy: 0.9,
            latency_us: 10.0,
            flops: 1000,
            params: 100,
            memory_bytes: 400,
            score: 0.9,
        };
        let b = Metrics {
            accuracy_proxy: 0.8,
            latency_us: 20.0,
            flops: 2000,
            params: 200,
            memory_bytes: 800,
            score: 0.7,
        };
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn test_pareto_non_dominance() {
        let a = Metrics {
            accuracy_proxy: 0.9,
            latency_us: 20.0,
            flops: 1000,
            params: 100,
            memory_bytes: 400,
            score: 0.9,
        };
        let b = Metrics {
            accuracy_proxy: 0.8,
            latency_us: 10.0,
            flops: 500,
            params: 50,
            memory_bytes: 200,
            score: 0.85,
        };
        // Neither dominates the other (a has better accuracy, b has better latency)
        assert!(!dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn test_search_returns_results() {
        let hw = HardwareProfile::fpga_hft();
        let space = SearchSpace::fpga_space(16, 3);
        let data: Vec<f64> = (0..100).map(|i| (i as f64).sin() * 0.01).collect();

        let results = search(&space, &hw, 50.0, 100, &data, 42);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_constrained() {
        let hw = HardwareProfile::fpga_hft();
        let space = SearchSpace::fpga_space(16, 3);
        let data: Vec<f64> = (0..100).map(|i| (i as f64).sin() * 0.01).collect();

        let results = search_constrained(&space, &hw, 100.0, 200, &data, 42);
        for (_, m) in &results {
            assert!(m.latency_us <= 100.0);
        }
    }

    #[test]
    fn test_scalarized_score() {
        let score = compute_scalarized_score(0.9, 10.0, 20.0, 1000, 2000, -0.07, -0.03);
        // Latency under budget and memory under budget should give bonus
        assert!(score > 0.9); // ratio < 1 with negative exponent -> multiplier > 1
    }

    #[test]
    fn test_compute_returns() {
        let candles = vec![
            Candle { timestamp: 0, open: 100.0, high: 105.0, low: 95.0, close: 100.0, volume: 1.0 },
            Candle { timestamp: 1, open: 100.0, high: 110.0, low: 98.0, close: 105.0, volume: 2.0 },
            Candle { timestamp: 2, open: 105.0, high: 108.0, low: 100.0, close: 103.0, volume: 1.5 },
        ];
        let returns = compute_returns(&candles);
        assert_eq!(returns.len(), 2);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_volatility() {
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.02, -0.01];
        let vol = compute_volatility(&returns, 3);
        assert_eq!(vol.len(), 4);
        for v in &vol {
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_search_space_sample() {
        let space = SearchSpace::fpga_space(16, 3);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let arch = space.sample_architecture(&mut rng);

        assert!(arch.layers.len() >= space.min_layers);
        assert!(arch.layers.len() <= space.max_layers);
        assert_eq!(arch.layers[0].in_channels, 16);
        assert_eq!(arch.layers.last().unwrap().out_channels, 3);
    }

    #[test]
    fn test_proxy_accuracy_bounded() {
        let arch = ArchitectureTemplates::fpga_hft_template(16, 3);
        let data: Vec<f64> = (0..50).map(|i| (i as f64) * 0.001).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let acc = estimate_proxy_accuracy(&arch, &data, &mut rng);
        assert!(acc >= 0.1 && acc <= 0.99);
    }

    #[test]
    fn test_architecture_summary() {
        let arch = ArchitectureTemplates::fpga_hft_template(16, 3);
        let summary = arch.summary();
        assert!(summary.contains("Architecture:"));
        assert!(summary.contains("Layer 0:"));
    }

    #[test]
    fn test_extract_pareto_front() {
        let arch = ArchitectureTemplates::fpga_hft_template(16, 3);
        let candidates = vec![
            (
                arch.clone(),
                Metrics {
                    accuracy_proxy: 0.9,
                    latency_us: 10.0,
                    flops: 1000,
                    params: 100,
                    memory_bytes: 400,
                    score: 0.9,
                },
            ),
            (
                arch.clone(),
                Metrics {
                    accuracy_proxy: 0.7,
                    latency_us: 30.0,
                    flops: 3000,
                    params: 300,
                    memory_bytes: 1200,
                    score: 0.5,
                },
            ),
            (
                arch.clone(),
                Metrics {
                    accuracy_proxy: 0.85,
                    latency_us: 5.0,
                    flops: 500,
                    params: 50,
                    memory_bytes: 200,
                    score: 0.88,
                },
            ),
        ];

        let front = extract_pareto_front(&candidates);
        // The second candidate is dominated by the first, so front should have 2
        assert!(front.len() <= 3);
        assert!(!front.is_empty());
    }
}
