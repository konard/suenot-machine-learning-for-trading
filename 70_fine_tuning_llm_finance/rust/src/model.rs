//! Neural network model implementations for financial sentiment analysis.
//!
//! This module provides:
//! - `LoraLayer`: Low-Rank Adaptation layer for parameter-efficient fine-tuning
//! - `PrefixTuningLayer`: Prefix-tuning implementation
//! - `FinancialSentimentModel`: Complete sentiment analysis model with LoRA

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use std::f32::consts::PI;

/// LoRA (Low-Rank Adaptation) Layer
///
/// Implements the LoRA technique where W = W_base + (α/r) * B * A
/// where B and A are low-rank matrices.
///
/// # Example
///
/// ```
/// use llm_finance::LoraLayer;
///
/// let lora = LoraLayer::new(768, 768, 8, 16.0);
/// let input = ndarray::Array1::zeros(768);
/// let output = lora.forward(&input);
/// ```
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// Input dimension
    pub in_features: usize,
    /// Output dimension
    pub out_features: usize,
    /// LoRA rank (r)
    pub rank: usize,
    /// Scaling factor (alpha)
    pub alpha: f32,
    /// Computed scaling = alpha / rank
    scaling: f32,
    /// Low-rank matrix A (r x in_features)
    pub lora_a: Array2<f32>,
    /// Low-rank matrix B (out_features x r)
    pub lora_b: Array2<f32>,
    /// Optional base weight matrix
    pub base_weight: Option<Array2<f32>>,
    /// Dropout probability
    pub dropout: f32,
}

impl LoraLayer {
    /// Create a new LoRA layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `rank` - LoRA rank (typically 4-64)
    /// * `alpha` - Scaling factor (typically 16-32)
    ///
    /// # Returns
    ///
    /// A new `LoraLayer` instance with randomly initialized matrices.
    pub fn new(in_features: usize, out_features: usize, rank: usize, alpha: f32) -> Self {
        // Initialize A with Kaiming/He initialization
        let std_a = (2.0 / in_features as f32).sqrt();
        let normal_a = Normal::new(0.0, std_a).unwrap();

        // Initialize B with zeros (as per LoRA paper)
        let lora_a = Array2::random((rank, in_features), normal_a);
        let lora_b = Array2::zeros((out_features, rank));

        Self {
            in_features,
            out_features,
            rank,
            alpha,
            scaling: alpha / rank as f32,
            lora_a,
            lora_b,
            base_weight: None,
            dropout: 0.0,
        }
    }

    /// Create a LoRA layer with a base weight matrix.
    pub fn with_base_weight(mut self, base_weight: Array2<f32>) -> Self {
        assert_eq!(base_weight.shape(), &[self.out_features, self.in_features]);
        self.base_weight = Some(base_weight);
        self
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout.clamp(0.0, 1.0);
        self
    }

    /// Forward pass through the LoRA layer.
    ///
    /// Computes: output = x @ W_base^T + scaling * x @ A^T @ B^T
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        assert_eq!(input.len(), self.in_features);

        // Compute LoRA contribution: x @ A^T @ B^T
        let hidden = self.lora_a.dot(input); // (rank,)
        let lora_output = self.lora_b.dot(&hidden); // (out_features,)

        // Scale the LoRA output
        let scaled_output = &lora_output * self.scaling;

        // Add base weight contribution if present
        match &self.base_weight {
            Some(base) => {
                let base_output = base.dot(input);
                base_output + scaled_output
            }
            None => scaled_output,
        }
    }

    /// Forward pass for batched input.
    pub fn forward_batch(&self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(input.ncols(), self.in_features);

        // input: (batch, in_features)
        // lora_a: (rank, in_features)
        // lora_b: (out_features, rank)

        // hidden = input @ lora_a^T -> (batch, rank)
        let hidden = input.dot(&self.lora_a.t());

        // lora_output = hidden @ lora_b^T -> (batch, out_features)
        let lora_output = hidden.dot(&self.lora_b.t());

        // Scale
        let scaled_output = &lora_output * self.scaling;

        // Add base weight if present
        match &self.base_weight {
            Some(base) => {
                // base_output = input @ base^T -> (batch, out_features)
                let base_output = input.dot(&base.t());
                base_output + scaled_output
            }
            None => scaled_output,
        }
    }

    /// Get the number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.rank * self.in_features + self.out_features * self.rank
    }

    /// Get total parameters including base weight.
    pub fn num_total_params(&self) -> usize {
        let base_params = self
            .base_weight
            .as_ref()
            .map(|w| w.len())
            .unwrap_or(0);
        base_params + self.num_trainable_params()
    }

    /// Merge LoRA weights into base weight for inference.
    pub fn merge(&mut self) {
        // Compute merged weight: W = W_base + scaling * B @ A
        let lora_weight = self.lora_b.dot(&self.lora_a) * self.scaling;

        self.base_weight = Some(match &self.base_weight {
            Some(base) => base + &lora_weight,
            None => lora_weight,
        });

        // Reset LoRA matrices
        self.lora_a = Array2::zeros((self.rank, self.in_features));
        self.lora_b = Array2::zeros((self.out_features, self.rank));
    }
}

/// Prefix-Tuning Layer
///
/// Implements prefix-tuning where learnable prefix vectors are prepended
/// to the input sequence to steer model behavior.
#[derive(Debug, Clone)]
pub struct PrefixTuningLayer {
    /// Number of prefix tokens
    pub num_prefix_tokens: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Whether to use prefix projection
    pub prefix_projection: bool,
    /// Prefix embeddings for each layer
    pub prefix_embeddings: Vec<Array2<f32>>,
    /// Optional projection layer
    pub projection: Option<Array2<f32>>,
}

impl PrefixTuningLayer {
    /// Create a new prefix-tuning layer.
    pub fn new(
        num_prefix_tokens: usize,
        hidden_dim: usize,
        num_layers: usize,
        prefix_projection: bool,
    ) -> Self {
        let normal = Normal::new(0.0, 0.02).unwrap();

        // Initialize prefix embeddings for each layer
        let prefix_embeddings: Vec<Array2<f32>> = (0..num_layers)
            .map(|_| Array2::random((num_prefix_tokens, hidden_dim), normal.clone()))
            .collect();

        // Optional projection layer
        let projection = if prefix_projection {
            Some(Array2::random((hidden_dim, hidden_dim), normal))
        } else {
            None
        };

        Self {
            num_prefix_tokens,
            hidden_dim,
            num_layers,
            prefix_projection,
            prefix_embeddings,
            projection,
        }
    }

    /// Get prefix for a specific layer.
    pub fn get_prefix(&self, layer_idx: usize) -> Option<&Array2<f32>> {
        self.prefix_embeddings.get(layer_idx)
    }

    /// Get prefix with optional projection.
    pub fn get_projected_prefix(&self, layer_idx: usize) -> Option<Array2<f32>> {
        let prefix = self.prefix_embeddings.get(layer_idx)?;

        Some(match &self.projection {
            Some(proj) => prefix.dot(proj),
            None => prefix.clone(),
        })
    }

    /// Get total trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        let prefix_params = self.num_layers * self.num_prefix_tokens * self.hidden_dim;
        let proj_params = self
            .projection
            .as_ref()
            .map(|p| p.len())
            .unwrap_or(0);
        prefix_params + proj_params
    }
}

/// Financial Sentiment Analysis Model with LoRA
///
/// A complete model for financial sentiment classification using LoRA
/// for parameter-efficient fine-tuning.
#[derive(Debug, Clone)]
pub struct FinancialSentimentModel {
    /// Base feature dimension
    pub base_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// LoRA rank
    pub lora_rank: usize,
    /// LoRA alpha
    pub lora_alpha: f32,
    /// Number of output classes
    pub num_classes: usize,
    /// Dropout rate
    pub dropout: f32,

    /// Query projection with LoRA
    pub query_lora: LoraLayer,
    /// Key projection with LoRA
    pub key_lora: LoraLayer,
    /// Value projection with LoRA
    pub value_lora: LoraLayer,
    /// Output projection with LoRA
    pub output_lora: LoraLayer,

    /// Feed-forward layer 1
    pub ff1: Array2<f32>,
    /// Feed-forward layer 2
    pub ff2: Array2<f32>,

    /// Classification head
    pub classifier: Array2<f32>,
    /// Classification bias
    pub classifier_bias: Array1<f32>,

    /// Layer normalization parameters
    pub ln_gamma: Array1<f32>,
    pub ln_beta: Array1<f32>,
}

impl FinancialSentimentModel {
    /// Create a new financial sentiment model.
    ///
    /// # Arguments
    ///
    /// * `base_dim` - Base feature dimension (typically 768 for BERT-like models)
    /// * `num_heads` - Number of attention heads
    /// * `lora_rank` - LoRA rank
    /// * `lora_alpha` - LoRA scaling factor
    /// * `num_classes` - Number of output classes (typically 3: bullish/neutral/bearish)
    pub fn new(
        base_dim: usize,
        num_heads: usize,
        lora_rank: usize,
        lora_alpha: f32,
        num_classes: usize,
    ) -> Self {
        let normal = Normal::new(0.0, 0.02).unwrap();

        // Create LoRA layers for attention
        let query_lora = LoraLayer::new(base_dim, base_dim, lora_rank, lora_alpha);
        let key_lora = LoraLayer::new(base_dim, base_dim, lora_rank, lora_alpha);
        let value_lora = LoraLayer::new(base_dim, base_dim, lora_rank, lora_alpha);
        let output_lora = LoraLayer::new(base_dim, base_dim, lora_rank, lora_alpha);

        // Feed-forward layers
        let ff_hidden = base_dim * 4;
        let ff1 = Array2::random((ff_hidden, base_dim), normal.clone());
        let ff2 = Array2::random((base_dim, ff_hidden), normal.clone());

        // Classification head
        let classifier = Array2::random((num_classes, base_dim), normal);
        let classifier_bias = Array1::zeros(num_classes);

        // Layer normalization
        let ln_gamma = Array1::ones(base_dim);
        let ln_beta = Array1::zeros(base_dim);

        Self {
            base_dim,
            num_heads,
            lora_rank,
            lora_alpha,
            num_classes,
            dropout: 0.1,
            query_lora,
            key_lora,
            value_lora,
            output_lora,
            ff1,
            ff2,
            classifier,
            classifier_bias,
            ln_gamma,
            ln_beta,
        }
    }

    /// Set dropout rate.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout.clamp(0.0, 1.0);
        self
    }

    /// Layer normalization.
    fn layer_norm(&self, x: &Array1<f32>) -> Array1<f32> {
        let mean = x.mean().unwrap_or(0.0);
        let variance = x.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
        let std = (variance + 1e-5).sqrt();

        let normalized = x.mapv(|v| (v - mean) / std);
        &normalized * &self.ln_gamma + &self.ln_beta
    }

    /// GELU activation function.
    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// Softmax function.
    fn softmax(x: &Array1<f32>) -> Array1<f32> {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input` - Input features of shape (base_dim,)
    ///
    /// # Returns
    ///
    /// Output logits of shape (num_classes,)
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        assert_eq!(input.len(), self.base_dim);

        // Apply LoRA attention projections
        let query = self.query_lora.forward(input);
        let key = self.key_lora.forward(input);
        let value = self.value_lora.forward(input);

        // Simplified self-attention (single token)
        let scale = (self.base_dim as f32 / self.num_heads as f32).sqrt();
        let attention_score = query.dot(&key) / scale;
        let attention_weight = (attention_score).exp(); // Simplified softmax for single token
        let attended = &value * attention_weight;

        // Output projection
        let attention_output = self.output_lora.forward(&attended);

        // Residual connection + layer norm
        let x = self.layer_norm(&(input + &attention_output));

        // Feed-forward network
        let ff_hidden = self.ff1.dot(&x);
        let ff_activated = ff_hidden.mapv(Self::gelu);
        let ff_output = self.ff2.dot(&ff_activated);

        // Residual connection + layer norm
        let x = self.layer_norm(&(&x + &ff_output));

        // Classification head
        let logits = self.classifier.dot(&x) + &self.classifier_bias;

        logits
    }

    /// Forward pass with batched input.
    pub fn forward_batch(&self, input: &Array2<f32>) -> Array2<f32> {
        let batch_size = input.nrows();
        let mut outputs = Array2::zeros((batch_size, self.num_classes));

        for (i, row) in input.outer_iter().enumerate() {
            let row_arr = row.to_owned();
            let output = self.forward(&row_arr);
            outputs.row_mut(i).assign(&output);
        }

        outputs
    }

    /// Get class probabilities using softmax.
    pub fn predict_proba(&self, input: &Array1<f32>) -> Array1<f32> {
        let logits = self.forward(input);
        Self::softmax(&logits)
    }

    /// Get predicted class.
    pub fn predict(&self, input: &Array1<f32>) -> usize {
        let logits = self.forward(input);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.query_lora.num_trainable_params()
            + self.key_lora.num_trainable_params()
            + self.value_lora.num_trainable_params()
            + self.output_lora.num_trainable_params()
    }

    /// Get total parameters.
    pub fn num_total_params(&self) -> usize {
        self.num_trainable_params()
            + self.ff1.len()
            + self.ff2.len()
            + self.classifier.len()
            + self.classifier_bias.len()
            + self.ln_gamma.len()
            + self.ln_beta.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_layer_creation() {
        let lora = LoraLayer::new(768, 768, 8, 16.0);
        assert_eq!(lora.in_features, 768);
        assert_eq!(lora.out_features, 768);
        assert_eq!(lora.rank, 8);
        assert_eq!(lora.alpha, 16.0);
    }

    #[test]
    fn test_lora_forward() {
        let lora = LoraLayer::new(768, 256, 8, 16.0);
        let input = Array1::zeros(768);
        let output = lora.forward(&input);
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_model_forward() {
        let model = FinancialSentimentModel::new(768, 12, 8, 16.0, 3);
        let input = Array1::zeros(768);
        let output = model.forward(&input);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_model_predict() {
        let model = FinancialSentimentModel::new(768, 12, 8, 16.0, 3);
        let input = Array1::zeros(768);
        let pred = model.predict(&input);
        assert!(pred < 3);
    }

    #[test]
    fn test_prefix_tuning() {
        let prefix = PrefixTuningLayer::new(20, 768, 12, true);
        assert_eq!(prefix.num_prefix_tokens, 20);
        assert_eq!(prefix.hidden_dim, 768);
        assert_eq!(prefix.num_layers, 12);

        let p = prefix.get_projected_prefix(0);
        assert!(p.is_some());
        assert_eq!(p.unwrap().shape(), &[20, 768]);
    }
}
