//! Grouped Query Attention Model Implementation
//!
//! This module contains the core GQA implementation and trading model.

use ndarray::{Array1, Array2, Array3, Axis, s};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f32::consts::PI;

/// Grouped Query Attention implementation.
///
/// GQA groups query heads to share key-value heads, reducing memory
/// usage while maintaining model quality.
#[derive(Clone)]
pub struct GroupedQueryAttention {
    d_model: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_groups: usize,
    scale: f32,

    // Weight matrices
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,
}

impl GroupedQueryAttention {
    /// Create a new GQA layer.
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key-value heads
    ///
    /// # Panics
    ///
    /// Panics if `d_model` is not divisible by `num_heads` or
    /// `num_heads` is not divisible by `num_kv_heads`.
    pub fn new(d_model: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );
        assert!(
            num_heads % num_kv_heads == 0,
            "num_heads must be divisible by num_kv_heads"
        );

        let head_dim = d_model / num_heads;
        let num_groups = num_heads / num_kv_heads;
        let scale = (head_dim as f32).powf(-0.5);

        // Initialize weights with Xavier initialization
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let w_q = Array2::from_shape_fn((d_model, num_heads * head_dim), |_| {
            normal.sample(&mut rng)
        });
        let w_k = Array2::from_shape_fn((d_model, num_kv_heads * head_dim), |_| {
            normal.sample(&mut rng)
        });
        let w_v = Array2::from_shape_fn((d_model, num_kv_heads * head_dim), |_| {
            normal.sample(&mut rng)
        });
        let w_o = Array2::from_shape_fn((num_heads * head_dim, d_model), |_| {
            normal.sample(&mut rng)
        });

        Self {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            num_groups,
            scale,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    /// Forward pass through the GQA layer.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (seq_len, d_model)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (seq_len, d_model)
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let seq_len = x.shape()[0];

        // Project queries, keys, values
        let q = x.dot(&self.w_q); // (seq_len, num_heads * head_dim)
        let k = x.dot(&self.w_k); // (seq_len, num_kv_heads * head_dim)
        let v = x.dot(&self.w_v); // (seq_len, num_kv_heads * head_dim)

        // Reshape for multi-head attention
        let q = q.into_shape((seq_len, self.num_heads, self.head_dim)).unwrap();
        let k = k.into_shape((seq_len, self.num_kv_heads, self.head_dim)).unwrap();
        let v = v.into_shape((seq_len, self.num_kv_heads, self.head_dim)).unwrap();

        // Compute attention for each head
        let mut outputs = Vec::with_capacity(self.num_heads);

        for head in 0..self.num_heads {
            // Determine which KV head this query head uses
            let kv_head = head / self.num_groups;

            // Extract head-specific Q, K, V
            let q_head = q.slice(s![.., head, ..]).to_owned();
            let k_head = k.slice(s![.., kv_head, ..]).to_owned();
            let v_head = v.slice(s![.., kv_head, ..]).to_owned();

            // Compute attention scores: Q @ K^T * scale
            let scores = q_head.dot(&k_head.t()) * self.scale;

            // Apply softmax
            let attn_weights = softmax(&scores);

            // Apply attention to values
            let head_output = attn_weights.dot(&v_head);
            outputs.push(head_output);
        }

        // Concatenate heads
        let mut concat = Array2::zeros((seq_len, self.num_heads * self.head_dim));
        for (i, output) in outputs.iter().enumerate() {
            concat
                .slice_mut(s![.., i * self.head_dim..(i + 1) * self.head_dim])
                .assign(output);
        }

        // Output projection
        concat.dot(&self.w_o)
    }

    /// Get memory usage statistics.
    pub fn memory_stats(&self, seq_len: usize) -> MemoryStats {
        let kv_cache_elements = 2 * self.num_kv_heads * seq_len * self.head_dim;
        let mha_cache_elements = 2 * self.num_heads * seq_len * self.head_dim;

        MemoryStats {
            gqa_kv_cache_bytes: kv_cache_elements * 4,
            mha_kv_cache_bytes: mha_cache_elements * 4,
            memory_savings: 1.0 - (kv_cache_elements as f32 / mha_cache_elements as f32),
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub gqa_kv_cache_bytes: usize,
    pub mha_kv_cache_bytes: usize,
    pub memory_savings: f32,
}

/// GQA Transformer block
#[derive(Clone)]
pub struct GQABlock {
    attention: GroupedQueryAttention,
    ff_w1: Array2<f32>,
    ff_w2: Array2<f32>,
    ln1_gamma: Array1<f32>,
    ln1_beta: Array1<f32>,
    ln2_gamma: Array1<f32>,
    ln2_beta: Array1<f32>,
    d_model: usize,
    d_ff: usize,
}

impl GQABlock {
    /// Create a new GQA block.
    pub fn new(d_model: usize, num_heads: usize, num_kv_heads: usize, d_ff: usize) -> Self {
        let attention = GroupedQueryAttention::new(d_model, num_heads, num_kv_heads);

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let ff_w1 = Array2::from_shape_fn((d_model, d_ff), |_| normal.sample(&mut rng));
        let ff_w2 = Array2::from_shape_fn((d_ff, d_model), |_| normal.sample(&mut rng));

        let ln1_gamma = Array1::ones(d_model);
        let ln1_beta = Array1::zeros(d_model);
        let ln2_gamma = Array1::ones(d_model);
        let ln2_beta = Array1::zeros(d_model);

        Self {
            attention,
            ff_w1,
            ff_w2,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
            d_model,
            d_ff,
        }
    }

    /// Forward pass through the block.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Self-attention with residual
        let normed = layer_norm(x, &self.ln1_gamma, &self.ln1_beta);
        let attn_out = self.attention.forward(&normed);
        let x = x + &attn_out;

        // Feed-forward with residual
        let normed = layer_norm(&x, &self.ln2_gamma, &self.ln2_beta);
        let ff_out = self.feed_forward(&normed);
        &x + &ff_out
    }

    fn feed_forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Linear -> GELU -> Linear
        let hidden = x.dot(&self.ff_w1);
        let activated = gelu(&hidden);
        activated.dot(&self.ff_w2)
    }
}

/// Complete GQA Trading Model
#[derive(Clone)]
pub struct GQATrader {
    input_proj: Array2<f32>,
    pos_encoding: Array2<f32>,
    layers: Vec<GQABlock>,
    output_ln_gamma: Array1<f32>,
    output_ln_beta: Array1<f32>,
    classifier_w1: Array2<f32>,
    classifier_w2: Array2<f32>,
    d_model: usize,
    max_seq_len: usize,
    num_classes: usize,
}

impl GQATrader {
    /// Create a new GQA trading model.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Number of input features (e.g., 5 for OHLCV)
    /// * `d_model` - Model hidden dimension
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads
    /// * `num_layers` - Number of transformer layers
    pub fn new(
        input_dim: usize,
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        num_layers: usize,
    ) -> Self {
        let max_seq_len = 512;
        let d_ff = 4 * d_model;
        let num_classes = 3;

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        // Input projection
        let input_proj =
            Array2::from_shape_fn((input_dim, d_model), |_| normal.sample(&mut rng));

        // Positional encoding
        let pos_encoding = create_positional_encoding(max_seq_len, d_model);

        // Transformer layers
        let layers: Vec<_> = (0..num_layers)
            .map(|_| GQABlock::new(d_model, num_heads, num_kv_heads, d_ff))
            .collect();

        // Output layer norm
        let output_ln_gamma = Array1::ones(d_model);
        let output_ln_beta = Array1::zeros(d_model);

        // Classifier
        let classifier_w1 =
            Array2::from_shape_fn((d_model, d_model / 2), |_| normal.sample(&mut rng));
        let classifier_w2 =
            Array2::from_shape_fn((d_model / 2, num_classes), |_| normal.sample(&mut rng));

        Self {
            input_proj,
            pos_encoding,
            layers,
            output_ln_gamma,
            output_ln_beta,
            classifier_w1,
            classifier_w2,
            d_model,
            max_seq_len,
            num_classes,
        }
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (seq_len, input_dim)
    ///
    /// # Returns
    ///
    /// Logits tensor of shape (num_classes,)
    pub fn forward(&self, x: &Array2<f32>) -> Array1<f32> {
        let seq_len = x.shape()[0];
        assert!(seq_len <= self.max_seq_len, "Sequence too long");

        // Project input
        let mut hidden = x.dot(&self.input_proj);

        // Add positional encoding
        hidden = &hidden + &self.pos_encoding.slice(s![..seq_len, ..]);

        // Pass through layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden);
        }

        // Final layer norm
        hidden = layer_norm(&hidden, &self.output_ln_gamma, &self.output_ln_beta);

        // Global average pooling
        let pooled = hidden.mean_axis(Axis(0)).unwrap();

        // Classifier
        let hidden = pooled.dot(&self.classifier_w1);
        let hidden = gelu_1d(&hidden);
        hidden.dot(&self.classifier_w2)
    }

    /// Predict class from input sequence.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (seq_len, input_dim)
    ///
    /// # Returns
    ///
    /// Predicted class (0=down, 1=neutral, 2=up)
    pub fn predict(&self, x: &Array2<f32>) -> usize {
        let logits = self.forward(x);
        argmax(&logits)
    }

    /// Predict with probabilities.
    pub fn predict_with_probs(&self, x: &Array2<f32>) -> (usize, Array1<f32>) {
        let logits = self.forward(x);
        let probs = softmax_1d(&logits);
        let prediction = argmax(&logits);
        (prediction, probs)
    }

    /// Get model parameter count.
    pub fn param_count(&self) -> usize {
        let mut count = 0;

        // Input projection
        count += self.input_proj.len();

        // Positional encoding
        count += self.pos_encoding.len();

        // Layers (approximate)
        for _ in &self.layers {
            // GQA weights + FF weights + LN params
            count += self.d_model * self.d_model * 4; // Q, K, V, O projections
            count += self.d_model * 4 * self.d_model * 2; // FF
            count += self.d_model * 4; // LN
        }

        // Classifier
        count += self.classifier_w1.len();
        count += self.classifier_w2.len();

        count
    }
}

// Helper functions

fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let max = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp = x.mapv(|v| (v - max).exp());
    let sum = exp.sum_axis(Axis(1));
    let mut result = exp;
    for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
        row.mapv_inplace(|v| v / sum[i]);
    }
    result
}

fn softmax_1d(x: &Array1<f32>) -> Array1<f32> {
    let max = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp = x.mapv(|v| (v - max).exp());
    let sum: f32 = exp.sum();
    exp / sum
}

fn layer_norm(x: &Array2<f32>, gamma: &Array1<f32>, beta: &Array1<f32>) -> Array2<f32> {
    let mean = x.mean_axis(Axis(1)).unwrap();
    let var = x.var_axis(Axis(1), 0.0);

    let mut result = x.clone();
    for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
        let std = (var[i] + 1e-5).sqrt();
        row.mapv_inplace(|v| (v - mean[i]) / std);
    }

    &result * gamma + beta
}

fn gelu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| {
        0.5 * v * (1.0 + ((2.0 / PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
    })
}

fn gelu_1d(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| {
        0.5 * v * (1.0 + ((2.0 / PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
    })
}

fn argmax(x: &Array1<f32>) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn create_positional_encoding(max_len: usize, d_model: usize) -> Array2<f32> {
    let mut pe = Array2::zeros((max_len, d_model));

    for pos in 0..max_len {
        for i in 0..d_model / 2 {
            let angle = pos as f32 / 10000_f32.powf((2 * i) as f32 / d_model as f32);
            pe[[pos, 2 * i]] = angle.sin();
            pe[[pos, 2 * i + 1]] = angle.cos();
        }
    }

    pe
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_gqa_attention() {
        let gqa = GroupedQueryAttention::new(64, 8, 2);
        let x = Array2::from_shape_fn((10, 64), |_| rand::random::<f32>());

        let output = gqa.forward(&x);
        assert_eq!(output.shape(), &[10, 64]);
    }

    #[test]
    fn test_gqa_trader() {
        let model = GQATrader::new(5, 64, 8, 2, 4);
        let x = Array2::from_shape_fn((60, 5), |_| rand::random::<f32>());

        let logits = model.forward(&x);
        assert_eq!(logits.shape(), &[3]);

        let prediction = model.predict(&x);
        assert!(prediction < 3);
    }

    #[test]
    fn test_memory_stats() {
        let gqa = GroupedQueryAttention::new(64, 8, 2);
        let stats = gqa.memory_stats(100);

        assert!(stats.memory_savings > 0.0);
        assert!(stats.gqa_kv_cache_bytes < stats.mha_kv_cache_bytes);
    }
}
