//! Attention mechanisms for Cross-Attention Multi-Asset Trading
//!
//! Implements:
//! - Multi-Head Self-Attention
//! - Cross-Asset Attention
//! - Temporal Attention

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Scaled dot-product attention
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    dropout: f64,
    training: bool,
) -> Result<(Tensor, Tensor)> {
    let d_k = query.dim(candle_core::D::Minus1)? as f64;
    let scale = 1.0 / d_k.sqrt();

    // Compute attention scores: Q @ K^T / sqrt(d_k)
    let scores = query.matmul(&key.transpose(candle_core::D::Minus2, candle_core::D::Minus1)?)?;
    let scores = (scores * scale)?;

    // Apply mask if provided
    let scores = if let Some(m) = mask {
        let neg_inf = Tensor::new(f32::NEG_INFINITY, query.device())?;
        scores.broadcast_add(&(m.to_dtype(DType::F32)? * neg_inf)?)?
    } else {
        scores
    };

    // Softmax
    let attention_weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;

    // Apply dropout during training
    let attention_weights = if training && dropout > 0.0 {
        candle_nn::ops::dropout(&attention_weights, dropout as f32)?
    } else {
        attention_weights
    };

    // Apply attention to values
    let output = attention_weights.matmul(value)?;

    Ok((output, attention_weights))
}

/// Multi-Head Self-Attention
pub struct MultiHeadAttention {
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    output_proj: Linear,
    n_heads: usize,
    head_dim: usize,
    dropout: f64,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(d_model: usize, n_heads: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        assert!(
            d_model % n_heads == 0,
            "d_model must be divisible by n_heads"
        );

        let head_dim = d_model / n_heads;

        let query_proj = linear(d_model, d_model, vb.pp("query"))?;
        let key_proj = linear(d_model, d_model, vb.pp("key"))?;
        let value_proj = linear(d_model, d_model, vb.pp("value"))?;
        let output_proj = linear(d_model, d_model, vb.pp("output"))?;

        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            n_heads,
            head_dim,
            dropout,
        })
    }

    /// Forward pass with optional mask
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
        training: bool,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = query.dim(0)?;
        let seq_len = query.dim(1)?;

        // Project Q, K, V
        let q = self.query_proj.forward(query)?;
        let k = self.key_proj.forward(key)?;
        let v = self.value_proj.forward(value)?;

        // Reshape for multi-head attention: [batch, seq, n_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.n_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.n_heads, self.head_dim))?;

        // Transpose: [batch, n_heads, seq, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Compute attention
        let (output, attention_weights) =
            scaled_dot_product_attention(&q, &k, &v, mask, self.dropout, training)?;

        // Transpose back: [batch, seq, n_heads, head_dim]
        let output = output.transpose(1, 2)?;

        // Concatenate heads: [batch, seq, d_model]
        let output = output.reshape((batch_size, seq_len, self.n_heads * self.head_dim))?;

        // Output projection
        let output = self.output_proj.forward(&output)?;

        Ok((output, attention_weights))
    }
}

/// Cross-Asset Attention
///
/// Captures relationships between different assets at each time step.
pub struct CrossAssetAttention {
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    output_proj: Linear,
    n_heads: usize,
    head_dim: usize,
    dropout: f64,
}

impl CrossAssetAttention {
    /// Create a new cross-asset attention layer
    pub fn new(d_model: usize, n_heads: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        assert!(
            d_model % n_heads == 0,
            "d_model must be divisible by n_heads"
        );

        let head_dim = d_model / n_heads;

        let query_proj = linear(d_model, d_model, vb.pp("query"))?;
        let key_proj = linear(d_model, d_model, vb.pp("key"))?;
        let value_proj = linear(d_model, d_model, vb.pp("value"))?;
        let output_proj = linear(d_model, d_model, vb.pp("output"))?;

        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            n_heads,
            head_dim,
            dropout,
        })
    }

    /// Forward pass
    ///
    /// Input shape: [batch, n_assets, seq_len, d_model]
    /// Output shape: [batch, n_assets, seq_len, d_model]
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<(Tensor, Tensor)> {
        let batch_size = x.dim(0)?;
        let n_assets = x.dim(1)?;
        let seq_len = x.dim(2)?;
        let d_model = x.dim(3)?;

        // Pool temporal dimension: [batch, n_assets, d_model]
        let x_pooled = x.mean(2)?;

        // Project Q, K, V
        let q = self.query_proj.forward(&x_pooled)?;
        let k = self.key_proj.forward(&x_pooled)?;
        let v = self.value_proj.forward(&x_pooled)?;

        // Reshape for multi-head: [batch, n_assets, n_heads, head_dim]
        let q = q.reshape((batch_size, n_assets, self.n_heads, self.head_dim))?;
        let k = k.reshape((batch_size, n_assets, self.n_heads, self.head_dim))?;
        let v = v.reshape((batch_size, n_assets, self.n_heads, self.head_dim))?;

        // Transpose: [batch, n_heads, n_assets, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Compute cross-asset attention
        let (context, attention_weights) =
            scaled_dot_product_attention(&q, &k, &v, None, self.dropout, training)?;

        // Transpose back: [batch, n_assets, n_heads, head_dim]
        let context = context.transpose(1, 2)?;

        // Concatenate heads: [batch, n_assets, d_model]
        let context = context.reshape((batch_size, n_assets, d_model))?;

        // Output projection
        let context = self.output_proj.forward(&context)?;

        // Broadcast back to full sequence: [batch, n_assets, seq_len, d_model]
        let context = context.unsqueeze(2)?;
        let context = context.broadcast_as((batch_size, n_assets, seq_len, d_model))?;

        Ok((context, attention_weights))
    }
}

/// Temporal Attention
///
/// Captures temporal patterns within each asset's time series.
pub struct TemporalAttention {
    attention: MultiHeadAttention,
}

impl TemporalAttention {
    /// Create a new temporal attention layer
    pub fn new(d_model: usize, n_heads: usize, dropout: f64, vb: VarBuilder) -> Result<Self> {
        let attention = MultiHeadAttention::new(d_model, n_heads, dropout, vb)?;
        Ok(Self { attention })
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
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = x.dim(0)?;
        let n_assets = x.dim(1)?;
        let seq_len = x.dim(2)?;
        let d_model = x.dim(3)?;

        // Reshape: [batch * n_assets, seq_len, d_model]
        let x_reshaped = x.reshape((batch_size * n_assets, seq_len, d_model))?;

        // Apply self-attention over time
        let (output, attention_weights) =
            self.attention
                .forward(&x_reshaped, &x_reshaped, &x_reshaped, mask, training)?;

        // Reshape back: [batch, n_assets, seq_len, d_model]
        let output = output.reshape((batch_size, n_assets, seq_len, d_model))?;

        Ok((output, attention_weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn create_test_vb() -> (VarMap, VarBuilder<'static>) {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        (varmap, vb)
    }

    #[test]
    fn test_multi_head_attention() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let attn = MultiHeadAttention::new(64, 4, 0.1, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 10, 64), &Device::Cpu)?;
        let (output, weights) = attn.forward(&x, &x, &x, None, false)?;

        assert_eq!(output.dims(), &[2, 10, 64]);
        assert_eq!(weights.dims(), &[2, 4, 10, 10]);

        Ok(())
    }

    #[test]
    fn test_cross_asset_attention() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let attn = CrossAssetAttention::new(64, 4, 0.1, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 64), &Device::Cpu)?;
        let (output, weights) = attn.forward(&x, false)?;

        assert_eq!(output.dims(), &[2, 5, 10, 64]);
        assert_eq!(weights.dims(), &[2, 4, 5, 5]);

        Ok(())
    }

    #[test]
    fn test_temporal_attention() -> Result<()> {
        let (_varmap, vb) = create_test_vb();
        let attn = TemporalAttention::new(64, 4, 0.1, vb)?;

        let x = Tensor::randn(0f32, 1f32, (2, 5, 10, 64), &Device::Cpu)?;
        let (output, _weights) = attn.forward(&x, None, false)?;

        assert_eq!(output.dims(), &[2, 5, 10, 64]);

        Ok(())
    }
}
