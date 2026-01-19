//! Complete BigBird Model for Trading
//!
//! This module provides the complete BigBird model combining:
//! - Input projection
//! - Positional encoding
//! - BigBird encoder stack
//! - Output head for predictions

use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
};

use super::{config::BigBirdConfig, encoder::BigBirdEncoder};

/// Sinusoidal Positional Encoding
#[derive(Module, Debug)]
pub struct PositionalEncoding<B: Backend> {
    encoding: Tensor<B, 2>,
    dropout: Dropout,
}

impl<B: Backend> PositionalEncoding<B> {
    pub fn new(device: &B::Device, d_model: usize, max_len: usize, dropout: f32) -> Self {
        // Create positional encoding matrix
        let mut pe = vec![vec![0.0f32; d_model]; max_len];

        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let angle = pos as f32 / (10000.0f32).powf(2.0 * i as f32 / d_model as f32);
                pe[pos][2 * i] = angle.sin();
                pe[pos][2 * i + 1] = angle.cos();
            }
        }

        let flat: Vec<f32> = pe.into_iter().flatten().collect();
        let encoding = Tensor::from_floats(flat.as_slice(), device).reshape([max_len, d_model]);

        Self {
            encoding,
            dropout: DropoutConfig::new(dropout as f64).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, seq_len, _d_model] = x.dims();
        let pe = self.encoding.clone().slice([0..seq_len]).unsqueeze::<3>();
        let x = x + pe;
        self.dropout.forward(x)
    }
}

/// Complete BigBird Model for Trading
#[derive(Module, Debug)]
pub struct BigBirdModel<B: Backend> {
    /// Input projection: input_dim -> d_model
    input_projection: Linear<B>,
    /// Positional encoding
    positional_encoding: PositionalEncoding<B>,
    /// BigBird encoder stack
    encoder: BigBirdEncoder<B>,
    /// Output head for predictions
    output_head: OutputHead<B>,
    /// Configuration
    #[module(skip)]
    config: BigBirdConfig,
}

/// Output head for different prediction tasks
#[derive(Module, Debug)]
pub struct OutputHead<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> OutputHead<B> {
    pub fn new(device: &B::Device, d_model: usize, output_dim: usize, dropout: f32) -> Self {
        Self {
            linear1: LinearConfig::new(d_model, d_model / 2).init(device),
            linear2: LinearConfig::new(d_model / 2, output_dim).init(device),
            dropout: DropoutConfig::new(dropout as f64).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::gelu(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

impl<B: Backend> BigBirdModel<B> {
    /// Create a new BigBird model from configuration
    pub fn new(device: &B::Device, config: &BigBirdConfig) -> Self {
        config.validate().expect("Invalid configuration");

        let input_projection = LinearConfig::new(config.input_dim, config.d_model).init(device);

        let positional_encoding =
            PositionalEncoding::new(device, config.d_model, config.seq_len, config.dropout);

        let encoder = BigBirdEncoder::new(
            device,
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.n_layers,
            config.seq_len,
            config.window_size,
            config.num_random,
            config.num_global,
            config.dropout,
            config.pre_norm,
            config.seed,
        );

        let output_head = OutputHead::new(device, config.d_model, config.output_dim, config.dropout);

        Self {
            input_projection,
            positional_encoding,
            encoder,
            output_head,
            config: config.clone(),
        }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, seq_len, input_dim]
    ///
    /// # Returns
    /// * Output tensor of shape [batch_size, output_dim]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // Project input to d_model dimension
        let x = self.input_projection.forward(x);

        // Add positional encoding
        let x = self.positional_encoding.forward(x);

        // Pass through BigBird encoder
        let x = self.encoder.forward(x);

        // Use the last position for prediction (or could pool)
        let [batch_size, seq_len, d_model] = x.dims();
        let last_hidden = x.slice([0..batch_size, (seq_len - 1)..seq_len, 0..d_model]);
        let last_hidden = last_hidden.reshape([batch_size, d_model]);

        // Output prediction
        self.output_head.forward(last_hidden)
    }

    /// Forward pass with attention analysis
    pub fn forward_with_analysis(&self, x: Tensor<B, 3>) -> ModelOutput<B> {
        let predictions = self.forward(x);

        ModelOutput {
            predictions,
            attention_stats: None, // Could add attention weight tracking
        }
    }

    /// Get model configuration
    pub fn config(&self) -> &BigBirdConfig {
        &self.config
    }

    /// Count total parameters
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;

        // Input projection
        count += self.config.input_dim * self.config.d_model + self.config.d_model;

        // Encoder layers
        let attention_params = 4 * self.config.d_model * self.config.d_model; // Q, K, V, O
        let ff_params =
            self.config.d_model * self.config.d_ff + self.config.d_ff * self.config.d_model;
        let norm_params = 4 * self.config.d_model; // 2 norms per layer
        count += self.config.n_layers * (attention_params + ff_params + norm_params);

        // Output head
        count += self.config.d_model * (self.config.d_model / 2);
        count += (self.config.d_model / 2) * self.config.output_dim;

        count
    }
}

/// Model output structure
#[derive(Debug)]
pub struct ModelOutput<B: Backend> {
    pub predictions: Tensor<B, 2>,
    pub attention_stats: Option<Vec<AttentionLayerStats>>,
}

/// Statistics for a single attention layer
#[derive(Debug, Clone)]
pub struct AttentionLayerStats {
    pub layer_idx: usize,
    pub mean_attention: f32,
    pub max_attention: f32,
    pub entropy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_model_creation() {
        let device = Default::default();
        let config = BigBirdConfig {
            seq_len: 32,
            input_dim: 6,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 256,
            window_size: 7,
            num_random: 3,
            num_global: 2,
            dropout: 0.0,
            output_dim: 1,
            pre_norm: true,
            activation: "gelu".to_string(),
            seed: 42,
        };

        let model = BigBirdModel::<TestBackend>::new(&device, &config);

        println!("Model parameters: {}", model.num_parameters());
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_forward_pass() {
        let device = Default::default();
        let config = BigBirdConfig {
            seq_len: 32,
            input_dim: 6,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 256,
            window_size: 7,
            num_random: 3,
            num_global: 2,
            dropout: 0.0,
            output_dim: 1,
            pre_norm: true,
            activation: "gelu".to_string(),
            seed: 42,
        };

        let model = BigBirdModel::<TestBackend>::new(&device, &config);

        // Create random input: [batch_size=2, seq_len=32, input_dim=6]
        let x = Tensor::random(
            [2, 32, 6],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = model.forward(x);

        // Output should be [batch_size=2, output_dim=1]
        assert_eq!(output.dims(), [2, 1]);
    }

    #[test]
    fn test_classification_output() {
        let device = Default::default();
        let config = BigBirdConfig {
            seq_len: 32,
            input_dim: 6,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            d_ff: 256,
            window_size: 7,
            num_random: 3,
            num_global: 2,
            dropout: 0.0,
            output_dim: 3, // Classification: up, down, neutral
            pre_norm: true,
            activation: "gelu".to_string(),
            seed: 42,
        };

        let model = BigBirdModel::<TestBackend>::new(&device, &config);

        let x = Tensor::random(
            [4, 32, 6],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = model.forward(x);
        assert_eq!(output.dims(), [4, 3]);
    }
}
