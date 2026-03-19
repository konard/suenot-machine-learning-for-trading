//! Neural network models for sentiment-based trading
//!
//! This module provides lightweight neural network implementations
//! for financial sentiment classification and trading signal generation.

use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

/// Sentiment classification result
#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub negative: f64,
    pub neutral: f64,
    pub positive: f64,
    pub label: i32,
    pub confidence: f64,
}

impl SentimentResult {
    /// Get trading signal: positive - negative sentiment
    pub fn signal(&self) -> f64 {
        self.positive - self.negative
    }
}

/// Simple feed-forward network for sentiment classification
///
/// This serves as a lightweight inference model for production use.
/// In practice, weights would be loaded from a pre-trained ELECTRA model
/// exported to ONNX format.
pub struct SentimentNetwork {
    embedding_dim: usize,
    hidden_dim: usize,
    num_classes: usize,
    embeddings: Array2<f64>,
    fc1_weights: Array2<f64>,
    fc1_bias: Array1<f64>,
    fc2_weights: Array2<f64>,
    fc2_bias: Array1<f64>,
}

impl SentimentNetwork {
    /// Create a new sentiment network with random initialization
    ///
    /// # Arguments
    /// * `vocab_size` - Size of the vocabulary
    /// * `embedding_dim` - Dimension of token embeddings
    /// * `hidden_dim` - Dimension of hidden layer
    /// * `num_classes` - Number of output classes (3 for sentiment)
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        hidden_dim: usize,
        num_classes: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / embedding_dim as f64).sqrt();

        let embeddings = Array2::from_shape_fn((vocab_size, embedding_dim), |_| {
            rng.gen::<f64>() * scale - scale / 2.0
        });

        let fc1_scale = (2.0 / (embedding_dim + hidden_dim) as f64).sqrt();
        let fc1_weights = Array2::from_shape_fn((embedding_dim, hidden_dim), |_| {
            rng.gen::<f64>() * fc1_scale - fc1_scale / 2.0
        });
        let fc1_bias = Array1::zeros(hidden_dim);

        let fc2_scale = (2.0 / (hidden_dim + num_classes) as f64).sqrt();
        let fc2_weights = Array2::from_shape_fn((hidden_dim, num_classes), |_| {
            rng.gen::<f64>() * fc2_scale - fc2_scale / 2.0
        });
        let fc2_bias = Array1::zeros(num_classes);

        Self {
            embedding_dim,
            hidden_dim,
            num_classes,
            embeddings,
            fc1_weights,
            fc1_bias,
            fc2_weights,
            fc2_bias,
        }
    }

    /// Forward pass for sentiment prediction
    ///
    /// Takes token IDs and attention mask, returns class probabilities.
    pub fn predict(&self, token_ids: &[u32], attention_mask: &[u32]) -> SentimentResult {
        // Compute mean embedding (CLS-like pooling)
        let embedding = self.mean_pool_embedding(token_ids, attention_mask);

        // Hidden layer with GELU activation
        let hidden = self.linear(&embedding, &self.fc1_weights, &self.fc1_bias);
        let hidden = self.gelu(&hidden);

        // Output layer with softmax
        let logits = self.linear(&hidden, &self.fc2_weights, &self.fc2_bias);
        let probs = self.softmax(&logits);

        let label = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(1);

        let confidence = probs.iter().cloned().fold(0.0f64, f64::max);

        SentimentResult {
            negative: probs[0],
            neutral: probs[1],
            positive: probs[2],
            label,
            confidence,
        }
    }

    /// Predict sentiment for a batch of tokenized texts
    pub fn predict_batch(
        &self,
        batch_ids: &[Vec<u32>],
        batch_masks: &[Vec<u32>],
    ) -> Vec<SentimentResult> {
        batch_ids
            .iter()
            .zip(batch_masks.iter())
            .map(|(ids, mask)| self.predict(ids, mask))
            .collect()
    }

    /// Generate trading signal from multiple sentiment results
    pub fn aggregate_signal(
        &self,
        results: &[SentimentResult],
        confidence_threshold: f64,
    ) -> (f64, f64) {
        if results.is_empty() {
            return (0.0, 0.0);
        }

        let n = results.len() as f64;
        let avg_positive: f64 = results.iter().map(|r| r.positive).sum::<f64>() / n;
        let avg_negative: f64 = results.iter().map(|r| r.negative).sum::<f64>() / n;
        let avg_confidence: f64 = results.iter().map(|r| r.confidence).sum::<f64>() / n;

        let signal = avg_positive - avg_negative;

        if avg_confidence < confidence_threshold {
            return (0.0, avg_confidence);
        }

        (signal, avg_confidence)
    }

    /// Compute mean pooled embedding from token IDs
    fn mean_pool_embedding(&self, token_ids: &[u32], attention_mask: &[u32]) -> Array1<f64> {
        let mut pooled = Array1::zeros(self.embedding_dim);
        let mut count = 0.0;

        for (i, &id) in token_ids.iter().enumerate() {
            if attention_mask[i] == 1 && (id as usize) < self.embeddings.nrows() {
                let row = self.embeddings.row(id as usize);
                pooled = pooled + &row;
                count += 1.0;
            }
        }

        if count > 0.0 {
            pooled /= count;
        }

        pooled
    }

    /// Linear transformation: y = x @ W + b
    fn linear(&self, x: &Array1<f64>, weights: &Array2<f64>, bias: &Array1<f64>) -> Array1<f64> {
        let mut output = bias.clone();
        for j in 0..weights.ncols() {
            let mut sum = 0.0;
            for i in 0..weights.nrows() {
                sum += x[i] * weights[[i, j]];
            }
            output[j] += sum;
        }
        output
    }

    /// GELU activation function
    fn gelu(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|v| {
            0.5 * v * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v.powi(3)) / std::f64::consts::SQRT_2).tanh())
        })
    }

    /// Softmax function
    fn softmax(&self, x: &Array1<f64>) -> Vec<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = x.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();
        exp_vals.iter().map(|v| v / sum).collect()
    }
}
