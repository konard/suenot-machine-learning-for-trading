//! DCT Model configuration

/// Configuration for the DCT model
#[derive(Debug, Clone)]
pub struct DCTConfig {
    /// Sequence length (lookback window)
    pub seq_len: usize,
    /// Number of input features
    pub input_features: usize,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Number of output classes
    pub num_classes: usize,
}

impl Default for DCTConfig {
    fn default() -> Self {
        Self {
            seq_len: 30,
            input_features: 13,
            d_model: 64,
            num_heads: 4,
            num_encoder_layers: 2,
            d_ff: 256,
            dropout: 0.1,
            num_classes: 3,
        }
    }
}

impl DCTConfig {
    /// Create a new configuration with custom parameters
    pub fn new(
        seq_len: usize,
        input_features: usize,
        d_model: usize,
        num_heads: usize,
        num_encoder_layers: usize,
    ) -> Self {
        Self {
            seq_len,
            input_features,
            d_model,
            num_heads,
            num_encoder_layers,
            d_ff: d_model * 4,
            ..Default::default()
        }
    }
}
