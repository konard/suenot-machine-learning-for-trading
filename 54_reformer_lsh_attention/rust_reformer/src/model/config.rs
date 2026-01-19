//! Reformer model configuration
//!
//! Configuration for the Reformer architecture including LSH attention parameters.

use serde::{Deserialize, Serialize};

/// Type of attention mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionType {
    /// Full O(L²) attention
    Full,
    /// LSH attention O(L·log(L))
    LSH,
    /// Local window attention O(L·w)
    Local,
}

impl Default for AttentionType {
    fn default() -> Self {
        Self::LSH
    }
}

/// Type of output layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// Regression - predict continuous values
    Regression,
    /// Classification - predict direction (up/down/neutral)
    Direction,
    /// Portfolio allocation weights
    Portfolio,
    /// Quantile regression for uncertainty intervals
    Quantile,
}

impl Default for OutputType {
    fn default() -> Self {
        Self::Regression
    }
}

/// Reformer model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReformerConfig {
    /// Input sequence length
    pub seq_len: usize,

    /// Number of input features
    pub n_features: usize,

    /// Model dimension (d_model)
    pub d_model: usize,

    /// Number of attention heads
    pub n_heads: usize,

    /// Feed-forward dimension
    pub d_ff: usize,

    /// Number of encoder layers
    pub n_layers: usize,

    /// Number of LSH hash rounds
    pub n_hash_rounds: usize,

    /// Number of hash buckets
    pub n_buckets: usize,

    /// Chunk size for chunked attention
    pub chunk_size: usize,

    /// Dropout probability
    pub dropout: f64,

    /// Attention type
    pub attention_type: AttentionType,

    /// Output type
    pub output_type: OutputType,

    /// Prediction horizon
    pub prediction_horizon: usize,

    /// Use reversible layers for memory efficiency
    pub use_reversible_layers: bool,

    /// Use positional encoding
    pub use_positional_encoding: bool,

    /// Quantiles for quantile regression
    pub quantiles: Vec<f64>,

    /// Causal masking (for autoregressive tasks)
    pub causal: bool,
}

impl Default for ReformerConfig {
    fn default() -> Self {
        Self {
            seq_len: 168,
            n_features: 10,
            d_model: 128,
            n_heads: 8,
            d_ff: 512,
            n_layers: 6,
            n_hash_rounds: 4,
            n_buckets: 32,
            chunk_size: 32,
            dropout: 0.1,
            attention_type: AttentionType::LSH,
            output_type: OutputType::Regression,
            prediction_horizon: 1,
            use_reversible_layers: true,
            use_positional_encoding: true,
            quantiles: vec![0.1, 0.5, 0.9],
            causal: false,
        }
    }
}

impl ReformerConfig {
    /// Create a small configuration (for testing)
    pub fn small() -> Self {
        Self {
            d_model: 32,
            n_heads: 2,
            d_ff: 128,
            n_layers: 2,
            n_hash_rounds: 2,
            n_buckets: 8,
            chunk_size: 16,
            ..Default::default()
        }
    }

    /// Create a medium configuration
    pub fn medium() -> Self {
        Self::default()
    }

    /// Create a large configuration
    pub fn large() -> Self {
        Self {
            d_model: 256,
            n_heads: 16,
            d_ff: 1024,
            n_layers: 8,
            n_hash_rounds: 8,
            n_buckets: 64,
            chunk_size: 64,
            ..Default::default()
        }
    }

    /// Configuration optimized for long sequences
    pub fn long_sequence(seq_len: usize) -> Self {
        Self {
            seq_len,
            d_model: 128,
            n_heads: 8,
            d_ff: 512,
            n_layers: 6,
            n_hash_rounds: 8,
            n_buckets: 64,
            chunk_size: 64,
            attention_type: AttentionType::LSH,
            use_reversible_layers: true,
            ..Default::default()
        }
    }

    /// Configuration for cryptocurrency prediction
    pub fn crypto() -> Self {
        Self {
            seq_len: 168, // 7 days of hourly data
            n_features: 10,
            d_model: 128,
            n_heads: 8,
            d_ff: 512,
            n_layers: 4,
            n_hash_rounds: 4,
            n_buckets: 32,
            prediction_horizon: 24, // 24 hours ahead
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.seq_len == 0 {
            return Err("seq_len must be > 0".to_string());
        }
        if self.d_model == 0 {
            return Err("d_model must be > 0".to_string());
        }
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }
        if self.n_buckets == 0 || (self.n_buckets & (self.n_buckets - 1)) != 0 {
            return Err("n_buckets must be a power of 2".to_string());
        }
        if self.chunk_size == 0 {
            return Err("chunk_size must be > 0".to_string());
        }
        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err("dropout must be in [0, 1]".to_string());
        }
        if self.n_hash_rounds == 0 {
            return Err("n_hash_rounds must be > 0".to_string());
        }

        // Validate quantiles
        for q in &self.quantiles {
            if *q <= 0.0 || *q >= 1.0 {
                return Err("quantiles must be in (0, 1)".to_string());
            }
        }

        Ok(())
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Output dimension based on output type
    pub fn output_dim(&self) -> usize {
        match self.output_type {
            OutputType::Regression => self.prediction_horizon,
            OutputType::Direction => 3, // up, down, neutral
            OutputType::Portfolio => self.prediction_horizon,
            OutputType::Quantile => self.prediction_horizon * self.quantiles.len(),
        }
    }

    /// Estimated memory usage in bytes (approximate)
    pub fn estimated_memory(&self) -> usize {
        let param_size = 8; // f64

        // Embedding layer
        let embedding = self.n_features * self.d_model * param_size;

        // Positional encoding
        let pos_encoding = self.seq_len * self.d_model * param_size;

        // Attention layers (per layer)
        let attention_per_layer = 4 * self.d_model * self.d_model * param_size; // Q, K, V, O
        let ff_per_layer = 2 * self.d_model * self.d_ff * param_size;
        let layer_norm = 2 * self.d_model * param_size;

        let per_layer = attention_per_layer + ff_per_layer + 2 * layer_norm;
        let all_layers = self.n_layers * per_layer;

        // LSH random rotations
        let lsh_rotations = self.n_hash_rounds * self.head_dim() * self.n_buckets * param_size;

        // Output layer
        let output = self.d_model * self.output_dim() * param_size;

        embedding + pos_encoding + all_layers + lsh_rotations + output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ReformerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_small_config() {
        let config = ReformerConfig::small();
        assert!(config.validate().is_ok());
        assert_eq!(config.d_model, 32);
    }

    #[test]
    fn test_large_config() {
        let config = ReformerConfig::large();
        assert!(config.validate().is_ok());
        assert_eq!(config.d_model, 256);
    }

    #[test]
    fn test_long_sequence_config() {
        let config = ReformerConfig::long_sequence(8760);
        assert!(config.validate().is_ok());
        assert_eq!(config.seq_len, 8760);
    }

    #[test]
    fn test_invalid_config() {
        let mut config = ReformerConfig::default();

        // Invalid d_model
        config.d_model = 65; // Not divisible by n_heads=8
        assert!(config.validate().is_err());

        // Invalid n_buckets
        config = ReformerConfig::default();
        config.n_buckets = 33; // Not power of 2
        assert!(config.validate().is_err());

        // Invalid dropout
        config = ReformerConfig::default();
        config.dropout = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_head_dim() {
        let config = ReformerConfig::default();
        assert_eq!(config.head_dim(), 128 / 8);
    }

    #[test]
    fn test_output_dim() {
        let mut config = ReformerConfig::default();
        config.prediction_horizon = 10;

        config.output_type = OutputType::Regression;
        assert_eq!(config.output_dim(), 10);

        config.output_type = OutputType::Direction;
        assert_eq!(config.output_dim(), 3);

        config.output_type = OutputType::Quantile;
        config.quantiles = vec![0.1, 0.5, 0.9];
        assert_eq!(config.output_dim(), 30);
    }

    #[test]
    fn test_serialization() {
        let config = ReformerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ReformerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.d_model, deserialized.d_model);
        assert_eq!(config.n_heads, deserialized.n_heads);
        assert_eq!(config.n_hash_rounds, deserialized.n_hash_rounds);
    }
}
