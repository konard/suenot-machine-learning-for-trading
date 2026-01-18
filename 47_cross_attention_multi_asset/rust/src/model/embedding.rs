//! Embedding layers for Cross-Attention Multi-Asset Trading
//!
//! Implements:
//! - Token Embedding (projects input features to model dimension)
//! - Positional Encoding (adds temporal position information)
//! - Asset Embedding (distinguishes between different assets)

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use std::f64::consts::PI;

/// Token Embedding layer
///
/// Projects input features to the model dimension.
pub struct TokenEmbedding {
    projection: Linear,
}

impl TokenEmbedding {
    /// Create a new token embedding layer
    pub fn new(n_features: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        let projection = linear(n_features, d_model, vb.pp("projection"))?;
        Ok(Self { projection })
    }

    /// Forward pass
    ///
    /// Input shape: [batch, n_assets, seq_len, n_features]
    /// Output shape: [batch, n_assets, seq_len, d_model]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.projection.forward(x)
    }
}

/// Positional Encoding
///
/// Adds sinusoidal position information to token embeddings.
pub struct PositionalEncoding {
    encoding: Tensor,
    dropout: f64,
}

impl PositionalEncoding {
    /// Create a new positional encoding layer
    pub fn new(d_model: usize, max_len: usize, dropout: f64, device: &Device) -> Result<Self> {
        let encoding = Self::create_encoding(d_model, max_len, device)?;
        Ok(Self { encoding, dropout })
    }

    /// Create sinusoidal positional encoding
    fn create_encoding(d_model: usize, max_len: usize, device: &Device) -> Result<Tensor> {
        let mut encoding = vec![0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f64 / (10000f64.powf((2 * (i / 2)) as f64 / d_model as f64));
                encoding[pos * d_model + i] = if i % 2 == 0 {
                    angle.sin() as f32
                } else {
                    angle.cos() as f32
                };
            }
        }

        Tensor::from_vec(encoding, (max_len, d_model), device)
    }

    /// Forward pass
    ///
    /// Input shape: [batch, n_assets, seq_len, d_model]
    /// Output shape: [batch, n_assets, seq_len, d_model]
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let seq_len = x.dim(2)?;

        // Get positional encoding for the sequence length
        let pe = self.encoding.narrow(0, 0, seq_len)?;

        // Add to input
        let output = x.broadcast_add(&pe)?;

        // Apply dropout during training
        if training && self.dropout > 0.0 {
            candle_nn::ops::dropout(&output, self.dropout as f32)
        } else {
            Ok(output)
        }
    }
}

/// Asset Embedding
///
/// Learnable embeddings to distinguish between different assets.
pub struct AssetEmbedding {
    embedding: Tensor,
}

impl AssetEmbedding {
    /// Create a new asset embedding layer
    pub fn new(n_assets: usize, d_model: usize, vb: VarBuilder) -> Result<Self> {
        // Initialize with small random values using candle's init
        let init = candle_nn::init::Init::Randn { mean: 0.0, stdev: 0.02 };
        let embedding = vb.get_with_hints((n_assets, d_model), "embedding", init)?;

        Ok(Self { embedding })
    }

    /// Forward pass
    ///
    /// Input shape: [batch, n_assets, seq_len, d_model]
    /// Output shape: [batch, n_assets, seq_len, d_model]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Expand asset embeddings to match input shape
        // [n_assets, d_model] -> [1, n_assets, 1, d_model] -> broadcast
        let asset_emb = self.embedding.unsqueeze(0)?.unsqueeze(2)?;

        // Add asset embeddings
        x.broadcast_add(&asset_emb)
    }

    /// Get asset embeddings for analysis
    pub fn get_embeddings(&self) -> &Tensor {
        &self.embedding
    }
}

/// Combined embedding layer
///
/// Combines token embedding, positional encoding, and asset embedding.
pub struct CombinedEmbedding {
    token_embedding: TokenEmbedding,
    positional_encoding: PositionalEncoding,
    asset_embedding: AssetEmbedding,
    dropout: f64,
}

impl CombinedEmbedding {
    /// Create a new combined embedding layer
    pub fn new(
        n_features: usize,
        d_model: usize,
        n_assets: usize,
        max_len: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let token_embedding = TokenEmbedding::new(n_features, d_model, vb.pp("token"))?;
        let positional_encoding =
            PositionalEncoding::new(d_model, max_len, dropout, vb.device())?;
        let asset_embedding = AssetEmbedding::new(n_assets, d_model, vb.pp("asset"))?;

        Ok(Self {
            token_embedding,
            positional_encoding,
            asset_embedding,
            dropout,
        })
    }

    /// Forward pass
    ///
    /// Input shape: [batch, n_assets, seq_len, n_features]
    /// Output shape: [batch, n_assets, seq_len, d_model]
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        // Token embedding
        let x = self.token_embedding.forward(x)?;

        // Positional encoding
        let x = self.positional_encoding.forward(&x, training)?;

        // Asset embedding
        let x = self.asset_embedding.forward(&x)?;

        // Final dropout
        if training && self.dropout > 0.0 {
            candle_nn::ops::dropout(&x, self.dropout as f32)
        } else {
            Ok(x)
        }
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
    fn test_token_embedding() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let emb = TokenEmbedding::new(6, 64, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 6), &Device::Cpu)?;
        let output = emb.forward(&x)?;

        assert_eq!(output.dims(), &[2, 5, 10, 64]);

        Ok(())
    }

    #[test]
    fn test_positional_encoding() -> Result<()> {
        let pe = PositionalEncoding::new(64, 512, 0.1, &Device::Cpu)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 64), &Device::Cpu)?;
        let output = pe.forward(&x, false)?;

        assert_eq!(output.dims(), &[2, 5, 10, 64]);

        Ok(())
    }

    #[test]
    fn test_asset_embedding() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let emb = AssetEmbedding::new(5, 64, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 64), &Device::Cpu)?;
        let output = emb.forward(&x)?;

        assert_eq!(output.dims(), &[2, 5, 10, 64]);

        Ok(())
    }

    #[test]
    fn test_combined_embedding() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let emb = CombinedEmbedding::new(6, 64, 5, 512, 0.1, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 6), &Device::Cpu)?;
        let output = emb.forward(&x, false)?;

        assert_eq!(output.dims(), &[2, 5, 10, 64]);

        Ok(())
    }
}
