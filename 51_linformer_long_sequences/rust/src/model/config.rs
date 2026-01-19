//! Configuration for Linformer model.

/// Configuration parameters for Linformer model.
#[derive(Debug, Clone)]
pub struct LinformerConfig {
    /// Model dimension (embedding size)
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Maximum sequence length
    pub seq_len: usize,
    /// Projection dimension (k in paper)
    pub k: usize,
    /// Number of encoder layers
    pub n_layers: usize,
    /// Feed-forward hidden dimension
    pub d_ff: usize,
    /// Number of input features
    pub n_features: usize,
    /// Number of output classes/values
    pub n_outputs: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Whether to share key-value projections
    pub share_kv: bool,
}

impl Default for LinformerConfig {
    fn default() -> Self {
        Self {
            d_model: 128,
            n_heads: 4,
            seq_len: 512,
            k: 64,
            n_layers: 4,
            d_ff: 256,
            n_features: 6,
            n_outputs: 1,
            dropout: 0.1,
            share_kv: true,
        }
    }
}

impl LinformerConfig {
    /// Create a new configuration.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        seq_len: usize,
        k: usize,
        n_layers: usize,
    ) -> Self {
        Self {
            d_model,
            n_heads,
            seq_len,
            k,
            n_layers,
            ..Default::default()
        }
    }

    /// Set feed-forward dimension.
    pub fn with_d_ff(mut self, d_ff: usize) -> Self {
        self.d_ff = d_ff;
        self
    }

    /// Set number of input features.
    pub fn with_n_features(mut self, n_features: usize) -> Self {
        self.n_features = n_features;
        self
    }

    /// Set number of output values.
    pub fn with_n_outputs(mut self, n_outputs: usize) -> Self {
        self.n_outputs = n_outputs;
        self
    }

    /// Set dropout rate.
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set key-value projection sharing.
    pub fn with_share_kv(mut self, share_kv: bool) -> Self {
        self.share_kv = share_kv;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model % self.n_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }

        if self.k > self.seq_len {
            return Err(format!(
                "k ({}) cannot be larger than seq_len ({})",
                self.k, self.seq_len
            ));
        }

        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err(format!(
                "dropout ({}) must be between 0.0 and 1.0",
                self.dropout
            ));
        }

        Ok(())
    }

    /// Calculate head dimension.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LinformerConfig::default();
        assert_eq!(config.d_model, 128);
        assert_eq!(config.n_heads, 4);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let config = LinformerConfig::new(128, 5, 512, 64, 4); // 128 not divisible by 5
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = LinformerConfig::new(256, 8, 1024, 128, 6)
            .with_d_ff(512)
            .with_n_features(10)
            .with_dropout(0.2);

        assert_eq!(config.d_model, 256);
        assert_eq!(config.d_ff, 512);
        assert_eq!(config.n_features, 10);
        assert!((config.dropout - 0.2).abs() < 0.001);
    }
}
