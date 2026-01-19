//! Model configuration
//!
//! Configuration structures for BigBird model hyperparameters.

use serde::{Deserialize, Serialize};

/// Configuration for BigBird sparse attention model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigBirdConfig {
    /// Input sequence length
    pub seq_len: usize,
    /// Number of input features (OHLCV + technical indicators)
    pub input_dim: usize,
    /// Model hidden dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of encoder layers
    pub n_layers: usize,
    /// Feed-forward network dimension
    pub d_ff: usize,
    /// Window size for local attention
    pub window_size: usize,
    /// Number of random attention connections per query
    pub num_random: usize,
    /// Number of global tokens
    pub num_global: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Output dimension (1 for regression, 3 for classification)
    pub output_dim: usize,
    /// Whether to use pre-layer normalization
    pub pre_norm: bool,
    /// Activation function: "relu", "gelu", "silu"
    pub activation: String,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for BigBirdConfig {
    fn default() -> Self {
        Self {
            seq_len: 256,
            input_dim: 6,
            d_model: 128,
            n_heads: 8,
            n_layers: 4,
            d_ff: 512,
            window_size: 7,
            num_random: 3,
            num_global: 2,
            dropout: 0.1,
            output_dim: 1,
            pre_norm: true,
            activation: "gelu".to_string(),
            seed: 42,
        }
    }
}

impl BigBirdConfig {
    /// Create a new configuration builder
    pub fn builder() -> BigBirdConfigBuilder {
        BigBirdConfigBuilder::default()
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }
        if self.window_size % 2 == 0 {
            return Err("window_size should be odd for symmetric local attention".to_string());
        }
        if self.num_global >= self.seq_len {
            return Err("num_global must be less than seq_len".to_string());
        }
        Ok(())
    }

    /// Get the dimension per attention head
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Estimate memory usage in bytes for a given batch size
    pub fn estimate_memory(&self, batch_size: usize) -> usize {
        let attention_memory = batch_size
            * self.n_layers
            * self.n_heads
            * self.seq_len
            * (self.window_size + self.num_random + self.num_global);
        let embedding_memory = batch_size * self.seq_len * self.d_model;
        let ff_memory = batch_size * self.seq_len * self.d_ff * self.n_layers;

        // Approximate total (multiply by 4 for f32)
        (attention_memory + embedding_memory + ff_memory) * 4
    }
}

/// Builder for BigBirdConfig
#[derive(Debug, Default)]
pub struct BigBirdConfigBuilder {
    config: BigBirdConfig,
}

impl BigBirdConfigBuilder {
    pub fn seq_len(mut self, seq_len: usize) -> Self {
        self.config.seq_len = seq_len;
        self
    }

    pub fn input_dim(mut self, input_dim: usize) -> Self {
        self.config.input_dim = input_dim;
        self
    }

    pub fn d_model(mut self, d_model: usize) -> Self {
        self.config.d_model = d_model;
        self
    }

    pub fn n_heads(mut self, n_heads: usize) -> Self {
        self.config.n_heads = n_heads;
        self
    }

    pub fn n_layers(mut self, n_layers: usize) -> Self {
        self.config.n_layers = n_layers;
        self
    }

    pub fn d_ff(mut self, d_ff: usize) -> Self {
        self.config.d_ff = d_ff;
        self
    }

    pub fn window_size(mut self, window_size: usize) -> Self {
        self.config.window_size = window_size;
        self
    }

    pub fn num_random(mut self, num_random: usize) -> Self {
        self.config.num_random = num_random;
        self
    }

    pub fn num_global(mut self, num_global: usize) -> Self {
        self.config.num_global = num_global;
        self
    }

    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.dropout = dropout;
        self
    }

    pub fn output_dim(mut self, output_dim: usize) -> Self {
        self.config.output_dim = output_dim;
        self
    }

    pub fn build(self) -> Result<BigBirdConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BigBirdConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder() {
        let config = BigBirdConfig::builder()
            .seq_len(128)
            .d_model(64)
            .n_heads(4)
            .build()
            .unwrap();

        assert_eq!(config.seq_len, 128);
        assert_eq!(config.d_model, 64);
        assert_eq!(config.n_heads, 4);
    }

    #[test]
    fn test_invalid_config() {
        let config = BigBirdConfig::builder()
            .d_model(100)
            .n_heads(8) // 100 not divisible by 8
            .build();

        assert!(config.is_err());
    }
}
