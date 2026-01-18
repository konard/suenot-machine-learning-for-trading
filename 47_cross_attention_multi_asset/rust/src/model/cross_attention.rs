//! Cross-Attention Multi-Asset Model
//!
//! Main model implementation combining all components.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, layer_norm, Linear, LayerNorm, Module, VarBuilder};

use super::attention::{CrossAssetAttention, TemporalAttention};
use super::embedding::CombinedEmbedding;
use super::{ModelConfig, OutputType};

/// Feed-Forward Network
pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    dropout: f64,
}

impl FeedForward {
    /// Create a new feed-forward network
    pub fn new(d_model: usize, d_ff: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = linear(d_ff, d_model, vb.pp("linear2"))?;

        Ok(Self {
            linear1,
            linear2,
            dropout,
        })
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.gelu_erf()?;

        let x = if training && self.dropout > 0.0 {
            candle_nn::ops::dropout(&x, self.dropout as f32)?
        } else {
            x
        };

        self.linear2.forward(&x)
    }
}

/// Cross-Attention Encoder Layer
///
/// One layer of the encoder, containing:
/// - Temporal self-attention
/// - Cross-asset attention
/// - Feed-forward network
pub struct CrossAttentionEncoderLayer {
    temporal_attention: TemporalAttention,
    cross_asset_attention: CrossAssetAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: f64,
}

impl CrossAttentionEncoderLayer {
    /// Create a new encoder layer
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let temporal_attention = TemporalAttention::new(
            config.d_model,
            config.n_heads,
            config.dropout,
            vb.pp("temporal_attn"),
        )?;

        let cross_asset_attention = CrossAssetAttention::new(
            config.d_model,
            config.n_heads,
            config.dropout,
            vb.pp("cross_asset_attn"),
        )?;

        let feed_forward = FeedForward::new(
            config.d_model,
            config.d_ff,
            config.dropout,
            vb.pp("ffn"),
        )?;

        let norm1 = layer_norm(config.d_model, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(config.d_model, 1e-5, vb.pp("norm2"))?;
        let norm3 = layer_norm(config.d_model, 1e-5, vb.pp("norm3"))?;

        Ok(Self {
            temporal_attention,
            cross_asset_attention,
            feed_forward,
            norm1,
            norm2,
            norm3,
            dropout: config.dropout,
        })
    }

    /// Forward pass
    ///
    /// Input shape: [batch, n_assets, seq_len, d_model]
    /// Output shape: [batch, n_assets, seq_len, d_model]
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        training: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Temporal self-attention with residual connection
        let (temporal_out, temporal_attn) = self.temporal_attention.forward(x, mask, training)?;
        let temporal_out = if training && self.dropout > 0.0 {
            candle_nn::ops::dropout(&temporal_out, self.dropout as f32)?
        } else {
            temporal_out
        };
        let x = (x + temporal_out)?;
        let x = self.norm1.forward(&x)?;

        // Cross-asset attention with residual connection
        let (cross_out, cross_attn) = self.cross_asset_attention.forward(&x, training)?;
        let cross_out = if training && self.dropout > 0.0 {
            candle_nn::ops::dropout(&cross_out, self.dropout as f32)?
        } else {
            cross_out
        };
        let x = (&x + cross_out)?;
        let x = self.norm2.forward(&x)?;

        // Feed-forward with residual connection
        let ff_out = self.feed_forward.forward(&x, training)?;
        let ff_out = if training && self.dropout > 0.0 {
            candle_nn::ops::dropout(&ff_out, self.dropout as f32)?
        } else {
            ff_out
        };
        let x = (&x + ff_out)?;
        let x = self.norm3.forward(&x)?;

        Ok((x, temporal_attn, cross_attn))
    }
}

/// Output head for the model
pub struct OutputHead {
    pooling_linear: Linear,
    output_linear: Linear,
    output_type: OutputType,
    n_assets: usize,
}

impl OutputHead {
    /// Create a new output head
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let output_dim = match config.output_type {
            OutputType::Regression => config.n_assets,
            OutputType::Classification => config.n_assets * 2, // up/down per asset
            OutputType::Portfolio => config.n_assets,
        };

        let pooling_linear = linear(config.d_model, config.d_model, vb.pp("pooling"))?;
        let output_linear = linear(config.d_model * config.n_assets, output_dim, vb.pp("output"))?;

        Ok(Self {
            pooling_linear,
            output_linear,
            output_type: config.output_type,
            n_assets: config.n_assets,
        })
    }

    /// Forward pass
    ///
    /// Input shape: [batch, n_assets, seq_len, d_model]
    /// Output shape: [batch, n_assets] or [batch, n_assets * 2]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dim(0)?;
        let n_assets = x.dim(1)?;
        let d_model = x.dim(3)?;

        // Pool over time (take last timestep or mean)
        let x = x.narrow(2, x.dim(2)? - 1, 1)?.squeeze(2)?; // [batch, n_assets, d_model]

        // Apply pooling projection
        let x = self.pooling_linear.forward(&x)?;
        let x = x.gelu_erf()?;

        // Flatten assets and features
        let x = x.reshape((batch_size, n_assets * d_model))?;

        // Output projection
        let output = self.output_linear.forward(&x)?;

        // Apply output-specific activation
        match self.output_type {
            OutputType::Regression => Ok(output),
            OutputType::Classification => {
                // Sigmoid for binary classification per asset
                candle_nn::ops::sigmoid(&output)
            }
            OutputType::Portfolio => {
                // Softmax for portfolio weights (sum to 1)
                candle_nn::ops::softmax(&output, candle_core::D::Minus1)
            }
        }
    }
}

/// Cross-Attention Multi-Asset Model
///
/// Main model combining embeddings, encoder layers, and output head.
pub struct CrossAttentionMultiAsset {
    embedding: CombinedEmbedding,
    layers: Vec<CrossAttentionEncoderLayer>,
    output_head: OutputHead,
    config: ModelConfig,
}

impl CrossAttentionMultiAsset {
    /// Create a new model
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        config.validate().map_err(|e| candle_core::Error::Msg(e))?;

        let embedding = CombinedEmbedding::new(
            config.n_features,
            config.d_model,
            config.n_assets,
            config.seq_len * 2,
            config.dropout,
            vb.pp("embedding"),
        )?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = CrossAttentionEncoderLayer::new(config, vb.pp(format!("layer_{}", i)))?;
            layers.push(layer);
        }

        let output_head = OutputHead::new(config, vb.pp("output_head"))?;

        Ok(Self {
            embedding,
            layers,
            output_head,
            config: config.clone(),
        })
    }

    /// Forward pass
    ///
    /// Input shape: [batch, n_assets, seq_len, n_features]
    /// Output: (predictions, attention_weights)
    pub fn forward(
        &self,
        x: &Tensor,
        training: bool,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        // Embedding
        let mut x = self.embedding.forward(x, training)?;

        // Encoder layers
        let mut attention_weights = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (output, temporal_attn, cross_attn) = layer.forward(&x, None, training)?;
            x = output;
            attention_weights.push((temporal_attn, cross_attn));
        }

        // Output head
        let predictions = self.output_head.forward(&x)?;

        Ok((predictions, attention_weights))
    }

    /// Get configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Save model weights
    pub fn save<P: AsRef<std::path::Path>>(&self, _path: P) -> Result<()> {
        // Note: In a real implementation, you would serialize the VarMap
        // This is a placeholder for the actual save logic
        Ok(())
    }

    /// Load model weights
    pub fn load<P: AsRef<std::path::Path>>(&mut self, _path: P) -> Result<()> {
        // Note: In a real implementation, you would deserialize into VarMap
        // This is a placeholder for the actual load logic
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn create_test_vb() -> (VarMap, VarBuilder<'static>) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        (varmap, vb)
    }

    #[test]
    fn test_feed_forward() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let ff = FeedForward::new(64, 256, 0.1, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 64), &Device::Cpu)?;
        let output = ff.forward(&x, false)?;

        assert_eq!(output.dims(), &[2, 5, 10, 64]);

        Ok(())
    }

    #[test]
    fn test_encoder_layer() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let config = ModelConfig::default();
        let layer = CrossAttentionEncoderLayer::new(&config, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 64), &Device::Cpu)?;
        let (output, _, _) = layer.forward(&x, None, false)?;

        assert_eq!(output.dims(), &[2, 5, 10, 64]);

        Ok(())
    }

    #[test]
    fn test_output_head_portfolio() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let config = ModelConfig::default().with_output_type(OutputType::Portfolio);
        let head = OutputHead::new(&config, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 64), &Device::Cpu)?;
        let output = head.forward(&x)?;

        assert_eq!(output.dims(), &[2, 5]);

        // Check that portfolio weights sum to 1
        let sums = output.sum(1)?;
        let sums_vec: Vec<f32> = sums.to_vec1()?;
        for sum in sums_vec {
            assert!((sum - 1.0).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_full_model() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let config = ModelConfig {
            n_assets: 5,
            n_features: 6,
            seq_len: 10,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            d_ff: 128,
            dropout: 0.1,
            output_type: OutputType::Portfolio,
        };

        let model = CrossAttentionMultiAsset::new(&config, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 6), &Device::Cpu)?;
        let (predictions, attention_weights) = model.forward(&x, false)?;

        assert_eq!(predictions.dims(), &[2, 5]);
        assert_eq!(attention_weights.len(), 2); // 2 layers

        Ok(())
    }

    #[test]
    fn test_model_regression_output() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let config = ModelConfig::default()
            .with_output_type(OutputType::Regression);

        let model = CrossAttentionMultiAsset::new(&config, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 6), &Device::Cpu)?;
        let (predictions, _) = model.forward(&x, false)?;

        assert_eq!(predictions.dims(), &[2, 5]);

        Ok(())
    }
}
