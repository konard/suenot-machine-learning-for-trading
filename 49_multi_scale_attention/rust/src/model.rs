//! Multi-Scale Attention Model for Financial Time Series
//!
//! This module implements the core multi-scale attention architecture that
//! processes financial data at multiple temporal resolutions simultaneously.

use ndarray::{Array1, Array2, Array3, Axis, s};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::collections::HashMap;

/// Configuration for a single scale encoder
#[derive(Debug, Clone)]
pub struct ScaleConfig {
    /// Name of the scale (e.g., "1min", "1H")
    pub name: String,
    /// Number of input features
    pub input_dim: usize,
    /// Expected sequence length
    pub seq_len: usize,
}

/// Configuration for the multi-scale attention model
#[derive(Debug, Clone)]
pub struct MultiScaleAttentionConfig {
    /// Configuration for each scale
    pub scale_configs: Vec<ScaleConfig>,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of encoder layers per scale
    pub n_encoder_layers: usize,
    /// Dropout rate (for training, not used in inference)
    pub dropout: f64,
    /// Output dimension
    pub output_dim: usize,
}

impl Default for MultiScaleAttentionConfig {
    fn default() -> Self {
        Self {
            scale_configs: vec![
                ScaleConfig {
                    name: "1min".to_string(),
                    input_dim: 15,
                    seq_len: 60,
                },
                ScaleConfig {
                    name: "5min".to_string(),
                    input_dim: 15,
                    seq_len: 24,
                },
                ScaleConfig {
                    name: "1H".to_string(),
                    input_dim: 15,
                    seq_len: 12,
                },
            ],
            d_model: 64,
            n_heads: 4,
            n_encoder_layers: 2,
            dropout: 0.1,
            output_dim: 1,
        }
    }
}

/// Linear layer implementation
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Array2<f64>,
    bias: Array1<f64>,
}

impl Linear {
    /// Create a new linear layer with Xavier initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let scale = (2.0 / (in_features + out_features) as f64).sqrt();
        let weight = Array2::random((out_features, in_features), Uniform::new(-scale, scale));
        let bias = Array1::zeros(out_features);
        Self { weight, bias }
    }

    /// Forward pass: y = Wx + b
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        x.dot(&self.weight.t()) + &self.bias
    }

    /// Forward pass for 3D input [batch, seq, features]
    pub fn forward_3d(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, _) = x.dim();
        let out_features = self.weight.nrows();
        let mut output = Array3::zeros((batch, seq_len, out_features));

        for b in 0..batch {
            for s in 0..seq_len {
                let input = x.slice(s![b, s, ..]).to_owned();
                let result = input.dot(&self.weight.t()) + &self.bias;
                output.slice_mut(s![b, s, ..]).assign(&result);
            }
        }
        output
    }
}

/// Layer normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    gamma: Array1<f64>,
    beta: Array1<f64>,
    eps: f64,
}

impl LayerNorm {
    pub fn new(features: usize) -> Self {
        Self {
            gamma: Array1::ones(features),
            beta: Array1::zeros(features),
            eps: 1e-5,
        }
    }

    /// Normalize along the last axis
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mean = x.mean_axis(Axis(1)).unwrap();
        let var = x.var_axis(Axis(1), 0.0);

        let mut output = x.clone();
        for (i, mut row) in output.axis_iter_mut(Axis(0)).enumerate() {
            let normalized = (&row.to_owned() - mean[i]) / (var[i] + self.eps).sqrt();
            row.assign(&(&normalized * &self.gamma + &self.beta));
        }
        output
    }

    /// Normalize 3D tensor along the last axis
    pub fn forward_3d(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, features) = x.dim();
        let mut output = Array3::zeros((batch, seq_len, features));

        for b in 0..batch {
            let slice = x.slice(s![b, .., ..]).to_owned();
            let normalized = self.forward(&slice);
            output.slice_mut(s![b, .., ..]).assign(&normalized);
        }
        output
    }
}

/// GELU activation function
pub fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
}

/// Apply GELU to array
pub fn gelu_array(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(gelu)
}

/// Softmax along axis 1
pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let max_vals = x.map_axis(Axis(1), |row| {
        row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });

    let mut exp_x = x.clone();
    for (i, mut row) in exp_x.axis_iter_mut(Axis(0)).enumerate() {
        row.mapv_inplace(|v| (v - max_vals[i]).exp());
    }

    let sum_exp = exp_x.sum_axis(Axis(1));
    for (i, mut row) in exp_x.axis_iter_mut(Axis(0)).enumerate() {
        row.mapv_inplace(|v| v / sum_exp[i]);
    }

    exp_x
}

/// Sigmoid activation
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus activation
pub fn softplus(x: f64) -> f64 {
    (1.0 + x.exp()).ln()
}

/// Multi-head attention mechanism
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    d_model: usize,
    #[allow(dead_code)]
    n_heads: usize,
    head_dim: usize,
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    output_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let head_dim = d_model / n_heads;
        Self {
            d_model,
            n_heads,
            head_dim,
            query_proj: Linear::new(d_model, d_model),
            key_proj: Linear::new(d_model, d_model),
            value_proj: Linear::new(d_model, d_model),
            output_proj: Linear::new(d_model, d_model),
        }
    }

    /// Scaled dot-product attention
    pub fn attention(&self, query: &Array3<f64>, key: &Array3<f64>, value: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_q, _) = query.dim();

        // Compute attention scores
        let scale = (self.head_dim as f64).sqrt();
        let mut output = Array3::zeros((batch, seq_q, self.d_model));

        for b in 0..batch {
            let q = query.slice(s![b, .., ..]).to_owned();
            let k = key.slice(s![b, .., ..]).to_owned();
            let v = value.slice(s![b, .., ..]).to_owned();

            // Q * K^T / sqrt(d_k)
            let scores = q.dot(&k.t()) / scale;

            // Softmax
            let attention_weights = softmax(&scores);

            // Attention * V
            let attended = attention_weights.dot(&v);
            output.slice_mut(s![b, .., ..]).assign(&attended);
        }

        output
    }

    /// Forward pass
    pub fn forward(&self, query: &Array3<f64>, key: &Array3<f64>, value: &Array3<f64>) -> Array3<f64> {
        let q = self.query_proj.forward_3d(query);
        let k = self.key_proj.forward_3d(key);
        let v = self.value_proj.forward_3d(value);

        let attended = self.attention(&q, &k, &v);
        self.output_proj.forward_3d(&attended)
    }
}

/// Transformer encoder layer
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer {
    attention: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, n_heads),
            linear1: Linear::new(d_model, d_model * 4),
            linear2: Linear::new(d_model * 4, d_model),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
        }
    }

    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        // Pre-norm architecture
        let (batch, seq_len, features) = x.dim();

        // Self-attention with residual
        let normed1 = self.norm1.forward_3d(x);
        let attended = self.attention.forward(&normed1, &normed1, &normed1);
        let x1 = x + &attended;

        // FFN with residual
        let normed2 = self.norm2.forward_3d(&x1);
        let mut ffn_out = Array3::zeros((batch, seq_len, features));

        for b in 0..batch {
            for s in 0..seq_len {
                let input = normed2.slice(s![b, s, ..]).to_owned().insert_axis(Axis(0));
                let hidden = gelu_array(&self.linear1.forward(&input));
                let output = self.linear2.forward(&hidden);
                ffn_out.slice_mut(s![b, s, ..]).assign(&output.slice(s![0, ..]));
            }
        }

        &x1 + &ffn_out
    }
}

/// Scale-specific encoder
#[derive(Debug, Clone)]
pub struct ScaleEncoder {
    pub scale_name: String,
    input_proj: Linear,
    input_norm: LayerNorm,
    positional_encoding: Array2<f64>,
    encoder_layers: Vec<TransformerEncoderLayer>,
    output_norm: LayerNorm,
}

impl ScaleEncoder {
    pub fn new(scale_name: String, input_dim: usize, d_model: usize, n_heads: usize, n_layers: usize, max_len: usize) -> Self {
        // Create learnable positional encoding
        let scale = 0.02;
        let positional_encoding = Array2::random((max_len, d_model), Uniform::new(-scale, scale));

        let encoder_layers = (0..n_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, n_heads))
            .collect();

        Self {
            scale_name,
            input_proj: Linear::new(input_dim, d_model),
            input_norm: LayerNorm::new(d_model),
            positional_encoding,
            encoder_layers,
            output_norm: LayerNorm::new(d_model),
        }
    }

    /// Encode a sequence at this scale
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch, seq_len, _) = x.dim();

        // Project input to model dimension
        let projected = self.input_proj.forward_3d(x);
        let normed = self.input_norm.forward_3d(&projected);

        // Add positional encoding
        let pos_enc = self.positional_encoding.slice(s![..seq_len, ..]).to_owned();
        let mut encoded = normed.clone();
        for b in 0..batch {
            for s in 0..seq_len {
                let mut slice = encoded.slice_mut(s![b, s, ..]);
                slice += &pos_enc.slice(s![s, ..]);
            }
        }

        // Apply transformer encoder layers
        let mut output = encoded;
        for layer in &self.encoder_layers {
            output = layer.forward(&output);
        }

        self.output_norm.forward_3d(&output)
    }
}

/// Cross-scale fusion module
#[derive(Debug, Clone)]
pub struct CrossScaleFusion {
    n_scales: usize,
    scale_weights: Array1<f64>,
    gate_linear1: Linear,
    gate_linear2: Linear,
    scale_transforms: Vec<Linear>,
    fusion_linear1: Linear,
    fusion_linear2: Linear,
    fusion_norm: LayerNorm,
}

impl CrossScaleFusion {
    pub fn new(d_model: usize, n_scales: usize) -> Self {
        // Initialize uniform scale weights
        let scale_weights = Array1::from_elem(n_scales, 1.0 / n_scales as f64);

        let scale_transforms = (0..n_scales)
            .map(|_| Linear::new(d_model, d_model))
            .collect();

        Self {
            n_scales,
            scale_weights,
            gate_linear1: Linear::new(d_model * n_scales, d_model),
            gate_linear2: Linear::new(d_model, n_scales),
            scale_transforms,
            fusion_linear1: Linear::new(d_model, d_model * 2),
            fusion_linear2: Linear::new(d_model * 2, d_model),
            fusion_norm: LayerNorm::new(d_model),
        }
    }

    /// Fuse multi-scale features
    pub fn forward(&self, scale_features: &[Array2<f64>]) -> (Array2<f64>, Array1<f64>) {
        let batch_size = scale_features[0].nrows();
        let d_model = scale_features[0].ncols();

        // Transform each scale
        let transformed: Vec<Array2<f64>> = scale_features
            .iter()
            .zip(&self.scale_transforms)
            .map(|(feat, transform)| gelu_array(&transform.forward(feat)))
            .collect();

        // Concatenate for gating
        let mut concat = Array2::zeros((batch_size, d_model * self.n_scales));
        for (i, feat) in scale_features.iter().enumerate() {
            concat.slice_mut(s![.., i*d_model..(i+1)*d_model]).assign(feat);
        }

        // Compute dynamic gates
        let gate_hidden = gelu_array(&self.gate_linear1.forward(&concat));
        let gate_logits = self.gate_linear2.forward(&gate_hidden);
        let dynamic_gates = softmax(&gate_logits);

        // Compute static weights (softmax of learnable weights)
        let static_weights = {
            let max_val = self.scale_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_weights: Array1<f64> = self.scale_weights.mapv(|w| (w - max_val).exp());
            let sum_exp = exp_weights.sum();
            exp_weights / sum_exp
        };

        // Combine static and dynamic weights
        let mut combined_weights = Array2::zeros((batch_size, self.n_scales));
        for b in 0..batch_size {
            for s in 0..self.n_scales {
                combined_weights[[b, s]] = static_weights[s] * dynamic_gates[[b, s]];
            }
        }
        // Normalize
        let weight_sums = combined_weights.sum_axis(Axis(1));
        for (i, mut row) in combined_weights.axis_iter_mut(Axis(0)).enumerate() {
            row.mapv_inplace(|w| w / (weight_sums[i] + 1e-8));
        }

        // Weighted sum of transformed features
        let mut weighted = Array2::zeros((batch_size, d_model));
        for (s, trans) in transformed.iter().enumerate() {
            for b in 0..batch_size {
                let weight = combined_weights[[b, s]];
                for d in 0..d_model {
                    weighted[[b, d]] += weight * trans[[b, d]];
                }
            }
        }

        // Final fusion
        let fusion_hidden = gelu_array(&self.fusion_linear1.forward(&weighted));
        let fusion_out = self.fusion_linear2.forward(&fusion_hidden);
        let output = self.fusion_norm.forward(&fusion_out);

        (output, static_weights)
    }
}

/// Model predictions
#[derive(Debug, Clone)]
pub struct Predictions {
    /// Short-term prediction
    pub short_term: Array2<f64>,
    /// Medium-term prediction
    pub medium_term: Array2<f64>,
    /// Long-term prediction
    pub long_term: Array2<f64>,
    /// Direction probability (0-1)
    pub direction: Array2<f64>,
    /// Prediction uncertainty
    pub uncertainty: Array2<f64>,
}

/// Multi-Scale Attention Network for financial time series prediction
#[derive(Debug)]
pub struct MultiScaleAttention {
    scale_names: Vec<String>,
    #[allow(dead_code)]
    n_scales: usize,
    #[allow(dead_code)]
    d_model: usize,
    scale_encoders: HashMap<String, ScaleEncoder>,
    cross_scale_fusion: CrossScaleFusion,
    short_term_head: (Linear, Linear),
    medium_term_head: (Linear, Linear),
    long_term_head: (Linear, Linear),
    direction_head: (Linear, Linear),
    uncertainty_head: (Linear, Linear),
}

impl MultiScaleAttention {
    /// Create a new multi-scale attention model
    pub fn new(config: MultiScaleAttentionConfig) -> Self {
        let scale_names: Vec<String> = config.scale_configs.iter().map(|c| c.name.clone()).collect();
        let n_scales = scale_names.len();
        let d_model = config.d_model;

        // Create scale-specific encoders
        let mut scale_encoders = HashMap::new();
        for scale_config in &config.scale_configs {
            let encoder = ScaleEncoder::new(
                scale_config.name.clone(),
                scale_config.input_dim,
                d_model,
                config.n_heads,
                config.n_encoder_layers,
                scale_config.seq_len + 100,
            );
            scale_encoders.insert(scale_config.name.clone(), encoder);
        }

        // Prediction heads
        let make_head = |d_model: usize, output_dim: usize| {
            (Linear::new(d_model, d_model / 2), Linear::new(d_model / 2, output_dim))
        };

        Self {
            scale_names,
            n_scales,
            d_model,
            scale_encoders,
            cross_scale_fusion: CrossScaleFusion::new(d_model, n_scales),
            short_term_head: make_head(d_model, config.output_dim),
            medium_term_head: make_head(d_model, config.output_dim),
            long_term_head: make_head(d_model, config.output_dim),
            direction_head: make_head(d_model, 1),
            uncertainty_head: make_head(d_model, 1),
        }
    }

    /// Forward pass
    pub fn forward(&self, scale_inputs: &HashMap<String, Array3<f64>>) -> Predictions {
        // Encode each scale
        let mut scale_encodings = Vec::new();
        for name in &self.scale_names {
            let input = scale_inputs.get(name).expect(&format!("Missing input for scale {}", name));
            let encoded = self.scale_encoders.get(name).unwrap().forward(input);
            scale_encodings.push(encoded);
        }

        // Get last position from each scale (most recent)
        let scale_summaries: Vec<Array2<f64>> = scale_encodings
            .iter()
            .map(|enc| {
                let seq_len = enc.dim().1;
                enc.slice(s![.., seq_len - 1, ..]).to_owned()
            })
            .collect();

        // Cross-scale fusion
        let (fused, _weights) = self.cross_scale_fusion.forward(&scale_summaries);

        // Generate predictions
        let apply_head = |head: &(Linear, Linear), x: &Array2<f64>| -> Array2<f64> {
            let hidden = gelu_array(&head.0.forward(x));
            head.1.forward(&hidden)
        };

        let short_term = apply_head(&self.short_term_head, &fused);
        let medium_term = apply_head(&self.medium_term_head, &fused);
        let long_term = apply_head(&self.long_term_head, &fused);

        let direction_logits = apply_head(&self.direction_head, &fused);
        let direction = direction_logits.mapv(sigmoid);

        let uncertainty_logits = apply_head(&self.uncertainty_head, &fused);
        let uncertainty = uncertainty_logits.mapv(softplus);

        Predictions {
            short_term,
            medium_term,
            long_term,
            direction,
            uncertainty,
        }
    }

    /// Get learned scale importance weights
    pub fn get_scale_importance(&self) -> HashMap<String, f64> {
        let weights = &self.cross_scale_fusion.scale_weights;
        let max_val = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_weights: Array1<f64> = weights.mapv(|w| (w - max_val).exp());
        let sum_exp = exp_weights.sum();
        let normalized = exp_weights / sum_exp;

        self.scale_names
            .iter()
            .zip(normalized.iter())
            .map(|(name, &weight)| (name.clone(), weight))
            .collect()
    }

    /// Get scale names
    pub fn scale_names(&self) -> &[String] {
        &self.scale_names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let linear = Linear::new(10, 5);
        let input = Array2::ones((2, 10));
        let output = linear.forward(&input);
        assert_eq!(output.dim(), (2, 5));
    }

    #[test]
    fn test_layer_norm() {
        let norm = LayerNorm::new(10);
        let input = Array2::random((2, 10), Uniform::new(-1.0, 1.0));
        let output = norm.forward(&input);
        assert_eq!(output.dim(), (2, 10));
    }

    #[test]
    fn test_multi_scale_attention() {
        let config = MultiScaleAttentionConfig {
            scale_configs: vec![
                ScaleConfig {
                    name: "1min".to_string(),
                    input_dim: 15,
                    seq_len: 10,
                },
                ScaleConfig {
                    name: "5min".to_string(),
                    input_dim: 15,
                    seq_len: 8,
                },
            ],
            d_model: 32,
            n_heads: 4,
            n_encoder_layers: 1,
            dropout: 0.1,
            output_dim: 1,
        };

        let model = MultiScaleAttention::new(config);

        let mut inputs = HashMap::new();
        inputs.insert("1min".to_string(), Array3::random((2, 10, 15), Uniform::new(-1.0, 1.0)));
        inputs.insert("5min".to_string(), Array3::random((2, 8, 15), Uniform::new(-1.0, 1.0)));

        let predictions = model.forward(&inputs);

        assert_eq!(predictions.short_term.dim(), (2, 1));
        assert_eq!(predictions.direction.dim(), (2, 1));

        // Check direction is in [0, 1]
        for &d in predictions.direction.iter() {
            assert!(d >= 0.0 && d <= 1.0);
        }

        // Check uncertainty is positive
        for &u in predictions.uncertainty.iter() {
            assert!(u > 0.0);
        }
    }

    #[test]
    fn test_scale_importance() {
        let config = MultiScaleAttentionConfig::default();
        let model = MultiScaleAttention::new(config);
        let importance = model.get_scale_importance();

        // Should have 3 scales by default
        assert_eq!(importance.len(), 3);

        // Weights should sum to ~1
        let sum: f64 = importance.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
