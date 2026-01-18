//! Complete DCT Model

use super::config::DCTConfig;
use super::encoder::TransformerEncoder;
use super::inception::InceptionEmbedding;
use ndarray::{Array1, Array2, Array3};

/// Movement classification result
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Class probabilities [up, down, stable]
    pub probabilities: Vec<f64>,
    /// Predicted class (0=Up, 1=Down, 2=Stable)
    pub predicted_class: usize,
    /// Confidence score
    pub confidence: f64,
}

/// Complete DCT Model
#[derive(Debug, Clone)]
pub struct DCTModel {
    config: DCTConfig,
    inception: InceptionEmbedding,
    encoder: TransformerEncoder,
    /// Classification head weights
    classifier_w: Array2<f64>,
    classifier_b: Array1<f64>,
}

impl DCTModel {
    /// Create a new DCT model
    pub fn new(config: DCTConfig) -> Self {
        let inception = InceptionEmbedding::new(config.input_features, config.d_model);
        let encoder = TransformerEncoder::new(
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.num_encoder_layers,
        );

        // Initialize classifier
        let scale = (2.0 / (config.d_model + config.num_classes) as f64).sqrt();
        let classifier_w = random_matrix(config.d_model, config.num_classes, scale);
        let classifier_b = Array1::zeros(config.num_classes);

        Self {
            config,
            inception,
            encoder,
            classifier_w,
            classifier_b,
        }
    }

    /// Forward pass
    /// Input: (batch, seq_len, input_features)
    /// Output: (batch, num_classes)
    pub fn forward(&self, x: &Array3<f64>) -> Array2<f64> {
        // Inception embedding
        let embedded = self.inception.forward(x);

        // Transformer encoder
        let encoded = self.encoder.forward(&embedded);

        // Global average pooling over sequence dimension
        let pooled = global_avg_pool(&encoded);

        // Classification head
        let logits = linear_2d(&pooled, &self.classifier_w, &self.classifier_b);

        logits
    }

    /// Predict with probabilities
    pub fn predict(&self, x: &Array3<f64>) -> Vec<Prediction> {
        let logits = self.forward(x);
        let batch_size = logits.nrows();

        let mut predictions = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let row: Vec<f64> = (0..self.config.num_classes)
                .map(|c| logits[[b, c]])
                .collect();

            let probs = softmax(&row);
            let (predicted_class, confidence) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &v)| (i, v))
                .unwrap();

            predictions.push(Prediction {
                probabilities: probs,
                predicted_class,
                confidence,
            });
        }

        predictions
    }

    /// Get model configuration
    pub fn config(&self) -> &DCTConfig {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        // Approximate count (simplified)
        let inception_params = self.config.input_features * self.config.d_model * 4 * 3; // Rough estimate
        let encoder_params = self.config.num_encoder_layers
            * (4 * self.config.d_model * self.config.d_model
                + 2 * self.config.d_model * self.config.d_ff);
        let classifier_params = self.config.d_model * self.config.num_classes;

        inception_params + encoder_params + classifier_params
    }
}

/// Global average pooling over sequence dimension
fn global_avg_pool(x: &Array3<f64>) -> Array2<f64> {
    let (batch, seq_len, d_model) = x.dim();
    let mut output = Array2::zeros((batch, d_model));

    for b in 0..batch {
        for d in 0..d_model {
            let mut sum = 0.0;
            for t in 0..seq_len {
                sum += x[[b, t, d]];
            }
            output[[b, d]] = sum / seq_len as f64;
        }
    }

    output
}

/// Linear transformation with bias
fn linear_2d(x: &Array2<f64>, weight: &Array2<f64>, bias: &Array1<f64>) -> Array2<f64> {
    let (batch, in_dim) = x.dim();
    let out_dim = weight.ncols();

    let mut output = Array2::zeros((batch, out_dim));

    for b in 0..batch {
        for o in 0..out_dim {
            let mut sum = bias[o];
            for i in 0..in_dim {
                sum += x[[b, i]] * weight[[i, o]];
            }
            output[[b, o]] = sum;
        }
    }

    output
}

/// Softmax function
fn softmax(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&v| v / sum).collect()
}

/// Random matrix initialization
fn random_matrix(rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(99999);

    Array2::from_shape_fn((rows, cols), |_| {
        let s = SEED.fetch_add(1, Ordering::Relaxed);
        let u1 = ((s.wrapping_mul(1103515245).wrapping_add(12345) % (1 << 31)) as f64)
            / (1u64 << 31) as f64;
        let u2 = ((s.wrapping_mul(1103515245).wrapping_add(54321) % (1 << 31)) as f64)
            / (1u64 << 31) as f64;

        let u1 = u1.max(1e-10);
        let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        normal * scale
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_model() {
        let config = DCTConfig::default();
        let model = DCTModel::new(config);

        let input = Array3::from_shape_fn((4, 30, 13), |_| 0.1);
        let predictions = model.predict(&input);

        assert_eq!(predictions.len(), 4);
        for pred in &predictions {
            assert_eq!(pred.probabilities.len(), 3);
            assert!(pred.predicted_class < 3);
            assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
        }
    }

    #[test]
    fn test_model_forward() {
        let config = DCTConfig::default();
        let model = DCTModel::new(config);

        let input = Array3::from_shape_fn((2, 30, 13), |_| 0.1);
        let logits = model.forward(&input);

        assert_eq!(logits.dim(), (2, 3));
    }
}
