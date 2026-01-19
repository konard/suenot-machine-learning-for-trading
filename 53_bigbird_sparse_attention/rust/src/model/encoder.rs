//! BigBird Encoder Layer
//!
//! Transformer encoder layer with BigBird sparse attention.

use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
};

use super::attention::BigBirdSparseAttention;

/// Feed-Forward Network
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
    activation: Gelu,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(device: &B::Device, d_model: usize, d_ff: usize, dropout: f32) -> Self {
        Self {
            linear1: LinearConfig::new(d_model, d_ff).init(device),
            linear2: LinearConfig::new(d_ff, d_model).init(device),
            dropout: DropoutConfig::new(dropout as f64).init(),
            activation: Gelu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

/// BigBird Encoder Layer
///
/// A single encoder layer consisting of:
/// 1. BigBird sparse self-attention
/// 2. Feed-forward network
/// 3. Residual connections and layer normalization
#[derive(Module, Debug)]
pub struct BigBirdEncoderLayer<B: Backend> {
    attention: BigBirdSparseAttention<B>,
    feed_forward: FeedForward<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
    pre_norm: bool,
}

impl<B: Backend> BigBirdEncoderLayer<B> {
    pub fn new(
        device: &B::Device,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        seq_len: usize,
        window_size: usize,
        num_random: usize,
        num_global: usize,
        dropout: f32,
        pre_norm: bool,
        seed: u64,
    ) -> Self {
        Self {
            attention: BigBirdSparseAttention::new(
                device, d_model, n_heads, seq_len, window_size, num_random, num_global, dropout, seed,
            ),
            feed_forward: FeedForward::new(device, d_model, d_ff, dropout),
            norm1: LayerNormConfig::new(d_model).init(device),
            norm2: LayerNormConfig::new(d_model).init(device),
            dropout: DropoutConfig::new(dropout as f64).init(),
            pre_norm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.pre_norm {
            // Pre-LN: Norm -> Attention/FFN -> Residual
            let residual = x.clone();
            let x = self.norm1.forward(x);
            let x = self.attention.forward(x);
            let x = self.dropout.forward(x) + residual;

            let residual = x.clone();
            let x = self.norm2.forward(x);
            let x = self.feed_forward.forward(x);
            self.dropout.forward(x) + residual
        } else {
            // Post-LN: Attention/FFN -> Residual -> Norm
            let residual = x.clone();
            let x = self.attention.forward(x);
            let x = self.norm1.forward(self.dropout.forward(x) + residual);

            let residual = x.clone();
            let x = self.feed_forward.forward(x);
            self.norm2.forward(self.dropout.forward(x) + residual)
        }
    }
}

/// Complete BigBird Encoder (stack of encoder layers)
#[derive(Module, Debug)]
pub struct BigBirdEncoder<B: Backend> {
    layers: Vec<BigBirdEncoderLayer<B>>,
    final_norm: LayerNorm<B>,
}

impl<B: Backend> BigBirdEncoder<B> {
    pub fn new(
        device: &B::Device,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        seq_len: usize,
        window_size: usize,
        num_random: usize,
        num_global: usize,
        dropout: f32,
        pre_norm: bool,
        seed: u64,
    ) -> Self {
        let layers = (0..n_layers)
            .map(|i| {
                BigBirdEncoderLayer::new(
                    device,
                    d_model,
                    n_heads,
                    d_ff,
                    seq_len,
                    window_size,
                    num_random,
                    num_global,
                    dropout,
                    pre_norm,
                    seed + i as u64, // Different seed per layer for varied random patterns
                )
            })
            .collect();

        Self {
            layers,
            final_norm: LayerNormConfig::new(d_model).init(device),
        }
    }

    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        self.final_norm.forward(x)
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_encoder_layer() {
        let device = Default::default();
        let layer = BigBirdEncoderLayer::<TestBackend>::new(
            &device,
            64,    // d_model
            4,     // n_heads
            256,   // d_ff
            32,    // seq_len
            7,     // window_size
            3,     // num_random
            2,     // num_global
            0.0,   // dropout
            true,  // pre_norm
            42,    // seed
        );

        let x = Tensor::random([2, 32, 64], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let output = layer.forward(x);

        assert_eq!(output.dims(), [2, 32, 64]);
    }

    #[test]
    fn test_encoder_stack() {
        let device = Default::default();
        let encoder = BigBirdEncoder::<TestBackend>::new(
            &device,
            64,    // d_model
            4,     // n_heads
            256,   // d_ff
            3,     // n_layers
            32,    // seq_len
            7,     // window_size
            3,     // num_random
            2,     // num_global
            0.0,   // dropout
            true,  // pre_norm
            42,    // seed
        );

        let x = Tensor::random([2, 32, 64], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let output = encoder.forward(x);

        assert_eq!(output.dims(), [2, 32, 64]);
        assert_eq!(encoder.num_layers(), 3);
    }
}
