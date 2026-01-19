//! Complete Linformer model implementation.

use ndarray::{Array1, Array2, Array3, s};
use rand::Rng;

use super::attention::LinformerAttention;
use super::config::LinformerConfig;

/// Feed-forward network layer.
pub struct FeedForward {
    /// First linear layer weights [d_model, d_ff]
    pub w1: Array2<f64>,
    /// First linear layer bias [d_ff]
    pub b1: Array1<f64>,
    /// Second linear layer weights [d_ff, d_model]
    pub w2: Array2<f64>,
    /// Second linear layer bias [d_model]
    pub b2: Array1<f64>,
}

impl FeedForward {
    /// Create a new feed-forward layer.
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let scale2 = (2.0 / (d_ff + d_model) as f64).sqrt();

        Self {
            w1: Array2::from_shape_fn((d_model, d_ff), |_| {
                rng.gen::<f64>() * scale1 * 2.0 - scale1
            }),
            b1: Array1::zeros(d_ff),
            w2: Array2::from_shape_fn((d_ff, d_model), |_| {
                rng.gen::<f64>() * scale2 * 2.0 - scale2
            }),
            b2: Array1::zeros(d_model),
        }
    }

    /// Forward pass with GELU activation.
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // First linear + GELU
        let h1 = x.dot(&self.w1);
        let h1_gelu = self.gelu(&h1);

        // Second linear
        h1_gelu.dot(&self.w2)
    }

    /// GELU activation function.
    fn gelu(&self, x: &Array2<f64>) -> Array2<f64> {
        // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
        x.mapv(|v| {
            let inner = sqrt_2_pi * (v + 0.044715 * v.powi(3));
            v * 0.5 * (1.0 + inner.tanh())
        })
    }
}

/// Linformer encoder layer.
pub struct LinformerEncoderLayer {
    /// Self-attention layer
    pub attention: LinformerAttention,
    /// Feed-forward layer
    pub feed_forward: FeedForward,
    /// Layer norm parameters for attention
    pub norm1_gamma: Array1<f64>,
    pub norm1_beta: Array1<f64>,
    /// Layer norm parameters for feed-forward
    pub norm2_gamma: Array1<f64>,
    pub norm2_beta: Array1<f64>,
}

impl LinformerEncoderLayer {
    /// Create a new encoder layer.
    pub fn new(config: &LinformerConfig) -> Self {
        Self {
            attention: LinformerAttention::new(
                config.d_model,
                config.n_heads,
                config.seq_len,
                config.k,
                config.share_kv,
            ),
            feed_forward: FeedForward::new(config.d_model, config.d_ff),
            norm1_gamma: Array1::ones(config.d_model),
            norm1_beta: Array1::zeros(config.d_model),
            norm2_gamma: Array1::ones(config.d_model),
            norm2_beta: Array1::zeros(config.d_model),
        }
    }

    /// Forward pass with residual connections and layer normalization.
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Self-attention with residual
        let attn_out = self.attention.forward(x);
        let x1 = x + &attn_out;
        let x1_norm = self.layer_norm(&x1, &self.norm1_gamma, &self.norm1_beta);

        // Feed-forward with residual
        let ff_out = self.feed_forward.forward(&x1_norm);
        let x2 = &x1_norm + &ff_out;
        self.layer_norm(&x2, &self.norm2_gamma, &self.norm2_beta)
    }

    /// Layer normalization.
    fn layer_norm(
        &self,
        x: &Array2<f64>,
        gamma: &Array1<f64>,
        beta: &Array1<f64>,
    ) -> Array2<f64> {
        let (rows, cols) = x.dim();
        let mut result = Array2::zeros((rows, cols));
        let eps = 1e-6;

        for i in 0..rows {
            let row = x.row(i);
            let mean = row.mean().unwrap_or(0.0);
            let variance: f64 = row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / cols as f64;
            let std = (variance + eps).sqrt();

            for j in 0..cols {
                result[[i, j]] = gamma[j] * (x[[i, j]] - mean) / std + beta[j];
            }
        }

        result
    }
}

/// Complete Linformer model for sequence prediction.
pub struct Linformer {
    /// Model configuration
    pub config: LinformerConfig,
    /// Input projection [n_features, d_model]
    pub input_projection: Array2<f64>,
    /// Positional encoding [seq_len, d_model]
    pub positional_encoding: Array2<f64>,
    /// Encoder layers
    pub layers: Vec<LinformerEncoderLayer>,
    /// Output projection [d_model, n_outputs]
    pub output_projection: Array2<f64>,
}

impl Linformer {
    /// Create a new Linformer model.
    pub fn new(config: LinformerConfig) -> Result<Self, String> {
        config.validate()?;

        let mut rng = rand::thread_rng();

        // Input projection
        let scale_in = (2.0 / (config.n_features + config.d_model) as f64).sqrt();
        let input_projection = Array2::from_shape_fn(
            (config.n_features, config.d_model),
            |_| rng.gen::<f64>() * scale_in * 2.0 - scale_in,
        );

        // Positional encoding (sinusoidal)
        let positional_encoding = Self::create_positional_encoding(config.seq_len, config.d_model);

        // Create encoder layers
        let layers: Vec<LinformerEncoderLayer> = (0..config.n_layers)
            .map(|_| LinformerEncoderLayer::new(&config))
            .collect();

        // Output projection
        let scale_out = (2.0 / (config.d_model + config.n_outputs) as f64).sqrt();
        let output_projection = Array2::from_shape_fn(
            (config.d_model, config.n_outputs),
            |_| rng.gen::<f64>() * scale_out * 2.0 - scale_out,
        );

        Ok(Self {
            config,
            input_projection,
            positional_encoding,
            layers,
            output_projection,
        })
    }

    /// Create sinusoidal positional encoding.
    fn create_positional_encoding(seq_len: usize, d_model: usize) -> Array2<f64> {
        let mut pe = Array2::zeros((seq_len, d_model));

        for pos in 0..seq_len {
            for i in 0..d_model / 2 {
                let angle = pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
                pe[[pos, 2 * i]] = angle.sin();
                pe[[pos, 2 * i + 1]] = angle.cos();
            }
        }

        pe
    }

    /// Forward pass for single sequence.
    /// Input: [seq_len, n_features]
    /// Output: [n_outputs] (using last token)
    pub fn forward(&self, x: &Array2<f64>) -> Array1<f64> {
        let (seq_len, _) = x.dim();
        let actual_seq_len = seq_len.min(self.config.seq_len);

        // Input projection
        let mut hidden = x.dot(&self.input_projection);

        // Add positional encoding
        let pe_slice = self.positional_encoding.slice(s![..actual_seq_len, ..]);
        hidden = &hidden + &pe_slice;

        // Pass through encoder layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden);
        }

        // Take the last token and project to output
        let last_hidden = hidden.row(actual_seq_len - 1).to_owned();
        last_hidden.dot(&self.output_projection)
    }

    /// Forward pass for batch of sequences.
    /// Input: [batch_size, seq_len, n_features]
    /// Output: [batch_size, n_outputs]
    pub fn forward_batch(&self, batch: &Array3<f64>) -> Array2<f64> {
        let (batch_size, _, _) = batch.dim();
        let mut outputs = Array2::zeros((batch_size, self.config.n_outputs));

        for b in 0..batch_size {
            let x = batch.slice(s![b, .., ..]).to_owned();
            let out = self.forward(&x);
            for i in 0..self.config.n_outputs {
                outputs[[b, i]] = out[i];
            }
        }

        outputs
    }

    /// Get model summary as string.
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "Linformer Model Summary".to_string(),
            "=".repeat(50),
            format!("Model dimension: {}", self.config.d_model),
            format!("Number of heads: {}", self.config.n_heads),
            format!("Head dimension: {}", self.config.head_dim()),
            format!("Sequence length: {}", self.config.seq_len),
            format!("Projection k: {}", self.config.k),
            format!("Number of layers: {}", self.config.n_layers),
            format!("Feed-forward dim: {}", self.config.d_ff),
            format!("Input features: {}", self.config.n_features),
            format!("Output size: {}", self.config.n_outputs),
            "=".repeat(50),
            format!(
                "Memory: {}",
                self.layers[0].attention.memory_complexity()
            ),
        ];

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feed_forward() {
        let ff = FeedForward::new(64, 256);
        let input = Array2::from_shape_fn((10, 64), |(i, j)| (i * j) as f64 * 0.01);
        let output = ff.forward(&input);
        assert_eq!(output.dim(), (10, 64));
    }

    #[test]
    fn test_encoder_layer() {
        let config = LinformerConfig::new(64, 4, 128, 32, 1);
        let layer = LinformerEncoderLayer::new(&config);
        let input = Array2::from_shape_fn((128, 64), |(i, j)| ((i + j) as f64 * 0.01).sin());
        let output = layer.forward(&input);
        assert_eq!(output.dim(), (128, 64));
    }

    #[test]
    fn test_linformer_creation() {
        let config = LinformerConfig::new(64, 4, 128, 32, 2)
            .with_n_features(6)
            .with_n_outputs(1);

        let model = Linformer::new(config).unwrap();
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_linformer_forward() {
        let config = LinformerConfig::new(32, 2, 64, 16, 2)
            .with_n_features(4)
            .with_n_outputs(1);

        let model = Linformer::new(config).unwrap();
        let input = Array2::from_shape_fn((64, 4), |(i, j)| (i * 4 + j) as f64 * 0.01);
        let output = model.forward(&input);

        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_linformer_batch() {
        let config = LinformerConfig::new(32, 2, 64, 16, 2)
            .with_n_features(4)
            .with_n_outputs(2);

        let model = Linformer::new(config).unwrap();
        let batch = Array3::from_shape_fn((4, 64, 4), |(b, i, j)| {
            (b * 64 * 4 + i * 4 + j) as f64 * 0.001
        });
        let output = model.forward_batch(&batch);

        assert_eq!(output.dim(), (4, 2));
    }

    #[test]
    fn test_positional_encoding() {
        let pe = Linformer::create_positional_encoding(100, 64);
        assert_eq!(pe.dim(), (100, 64));

        // Check that values are bounded
        for val in pe.iter() {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_model_summary() {
        let config = LinformerConfig::default();
        let model = Linformer::new(config).unwrap();
        let summary = model.summary();
        assert!(summary.contains("Linformer"));
        assert!(summary.contains("Memory"));
    }
}
