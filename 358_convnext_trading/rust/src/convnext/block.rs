//! ConvNeXt Block implementation

use ndarray::Array3;
use serde::{Deserialize, Serialize};

use super::layers::{Conv1d, DropPath, LayerNorm, gelu_array};

/// ConvNeXt Block
///
/// The core building block of ConvNeXt architecture.
/// Implements: DWConv -> LayerNorm -> PWConv -> GELU -> PWConv -> Residual
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConvNeXtBlock {
    /// Depthwise convolution (7x1 kernel)
    dwconv: Conv1d,
    /// Layer normalization
    norm: LayerNorm,
    /// First pointwise convolution (expand channels by 4x)
    pwconv1: Conv1d,
    /// Second pointwise convolution (contract back)
    pwconv2: Conv1d,
    /// Layer scale parameter
    layer_scale: Option<f64>,
    /// Drop path for stochastic depth
    drop_path: DropPath,
    /// Number of input/output channels
    dim: usize,
}

impl ConvNeXtBlock {
    /// Create a new ConvNeXt block
    ///
    /// # Arguments
    /// * `dim` - Number of input/output channels
    /// * `drop_path_rate` - Probability of dropping the path
    /// * `layer_scale_init` - Initial value for layer scale (None to disable)
    pub fn new(dim: usize, drop_path_rate: f64, layer_scale_init: Option<f64>) -> Self {
        // Depthwise conv with 7x1 kernel
        let dwconv = Conv1d::depthwise(dim, 7, 3);

        // Layer normalization
        let norm = LayerNorm::new(dim);

        // Pointwise convolutions (expand 4x then contract)
        let hidden_dim = dim * 4;
        let pwconv1 = Conv1d::pointwise(dim, hidden_dim);
        let pwconv2 = Conv1d::pointwise(hidden_dim, dim);

        let drop_path = DropPath::new(drop_path_rate);

        Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            layer_scale: layer_scale_init,
            drop_path,
            dim,
        }
    }

    /// Forward pass
    ///
    /// Input shape: [batch, channels, length]
    /// Output shape: [batch, channels, length]
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let residual = x.clone();

        // Depthwise convolution
        let mut out = self.dwconv.forward(x);

        // Permute to [batch, length, channels] for layer norm
        // Then apply layer norm over channels
        out = self.norm.forward_3d(&out);

        // First pointwise conv (expand)
        out = self.pwconv1.forward(&out);

        // GELU activation
        out = gelu_array(&out);

        // Second pointwise conv (contract)
        out = self.pwconv2.forward(&out);

        // Apply layer scale if enabled
        if let Some(scale) = self.layer_scale {
            out = out * scale;
        }

        // Residual connection with optional drop path
        if self.drop_path.training && self.drop_path.drop_prob > 0.0 {
            let (batch, channels, length) = out.dim();
            let mut result = Array3::zeros((batch, channels, length));

            for b in 0..batch {
                let keep = rand::random::<f64>() >= self.drop_path.drop_prob;
                for c in 0..channels {
                    for l in 0..length {
                        if keep {
                            let scale = 1.0 / (1.0 - self.drop_path.drop_prob);
                            result[[b, c, l]] = residual[[b, c, l]] + out[[b, c, l]] * scale;
                        } else {
                            result[[b, c, l]] = residual[[b, c, l]];
                        }
                    }
                }
            }
            result
        } else {
            &residual + &out
        }
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.drop_path.set_training(training);
    }

    /// Get the dimension (number of channels)
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Downsampling layer between stages
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Downsample {
    /// Layer normalization
    norm: LayerNorm,
    /// Strided convolution for downsampling
    conv: Conv1d,
}

impl Downsample {
    /// Create a new downsampling layer
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `out_channels` - Output channels
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        let norm = LayerNorm::new(in_channels);
        let conv = Conv1d::new(in_channels, out_channels, 2, 2, 0, 1);

        Self { norm, conv }
    }

    /// Forward pass
    ///
    /// Input shape: [batch, in_channels, length]
    /// Output shape: [batch, out_channels, length/2]
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let out = self.norm.forward_3d(x);
        self.conv.forward(&out)
    }
}

/// Stem layer (initial patchify)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stem {
    /// Convolution for patchifying
    conv: Conv1d,
    /// Layer normalization
    norm: LayerNorm,
}

impl Stem {
    /// Create a new stem layer
    ///
    /// # Arguments
    /// * `in_channels` - Input channels (e.g., 20 for OHLCV + indicators)
    /// * `out_channels` - Output channels (first stage channels)
    /// * `patch_size` - Size of each patch (stride and kernel size)
    pub fn new(in_channels: usize, out_channels: usize, patch_size: usize) -> Self {
        let conv = Conv1d::new(in_channels, out_channels, patch_size, patch_size, 0, 1);
        let norm = LayerNorm::new(out_channels);

        Self { conv, norm }
    }

    /// Forward pass
    ///
    /// Input shape: [batch, in_channels, length]
    /// Output shape: [batch, out_channels, length/patch_size]
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let out = self.conv.forward(x);
        self.norm.forward_3d(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_convnext_block() {
        let block = ConvNeXtBlock::new(64, 0.0, Some(1e-6));
        let input = Array3::ones((2, 64, 32));
        let output = block.forward(&input);
        assert_eq!(output.dim(), (2, 64, 32));
    }

    #[test]
    fn test_downsample() {
        let ds = Downsample::new(64, 128);
        let input = Array3::ones((2, 64, 32));
        let output = ds.forward(&input);
        assert_eq!(output.dim(), (2, 128, 16));
    }

    #[test]
    fn test_stem() {
        let stem = Stem::new(20, 96, 4);
        let input = Array3::ones((2, 20, 256));
        let output = stem.forward(&input);
        assert_eq!(output.dim(), (2, 96, 64));
    }
}
