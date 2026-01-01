//! Core Squeeze-and-Excitation Block Implementation
//!
//! This module provides the fundamental SE block that can adaptively
//! recalibrate channel-wise feature responses for time series data.

use ndarray::{Array1, Array2, Axis};
use rand::Rng;

use super::activation::{relu, sigmoid};

/// Type of squeeze operation to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SqueezeType {
    /// Global Average Pooling - mean across time dimension
    GlobalAveragePooling,
    /// Global Max Pooling - max across time dimension
    GlobalMaxPooling,
    /// Use only the last time step value
    LastValue,
    /// Exponentially weighted average (more weight to recent)
    ExponentialWeighted { alpha: f64 },
}

impl Default for SqueezeType {
    fn default() -> Self {
        Self::GlobalAveragePooling
    }
}

/// Squeeze-and-Excitation Block for time series
///
/// The SE block learns to dynamically weight different features (channels)
/// based on the current input, allowing the model to focus on more relevant
/// features for the given market conditions.
#[derive(Debug, Clone)]
pub struct SEBlock {
    /// Number of input channels (features)
    channels: usize,
    /// Reduction ratio for the bottleneck layer
    reduction_ratio: usize,
    /// Reduced channel dimension
    reduced_channels: usize,
    /// Type of squeeze operation
    squeeze_type: SqueezeType,
    /// Weights for first FC layer (dimensionality reduction)
    weights_fc1: Array2<f64>,
    /// Weights for second FC layer (dimensionality expansion)
    weights_fc2: Array2<f64>,
    /// Bias for first FC layer
    bias_fc1: Array1<f64>,
    /// Bias for second FC layer
    bias_fc2: Array1<f64>,
}

impl SEBlock {
    /// Create a new SE block with given parameters
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of input channels (features)
    /// * `reduction_ratio` - Reduction ratio for bottleneck (typically 4 or 16)
    ///
    /// # Example
    ///
    /// ```
    /// use se_trading::models::se_block::SEBlock;
    ///
    /// let se_block = SEBlock::new(10, 4);
    /// ```
    pub fn new(channels: usize, reduction_ratio: usize) -> Self {
        Self::with_squeeze_type(channels, reduction_ratio, SqueezeType::default())
    }

    /// Create a new SE block with a specific squeeze type
    pub fn with_squeeze_type(
        channels: usize,
        reduction_ratio: usize,
        squeeze_type: SqueezeType,
    ) -> Self {
        let reduced_channels = (channels / reduction_ratio).max(1);
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization
        let scale1 = (2.0 / (channels + reduced_channels) as f64).sqrt();
        let scale2 = (2.0 / (reduced_channels + channels) as f64).sqrt();

        let weights_fc1 = Array2::from_shape_fn((reduced_channels, channels), |_| {
            rng.gen_range(-scale1..scale1)
        });

        let weights_fc2 = Array2::from_shape_fn((channels, reduced_channels), |_| {
            rng.gen_range(-scale2..scale2)
        });

        Self {
            channels,
            reduction_ratio,
            reduced_channels,
            squeeze_type,
            weights_fc1,
            weights_fc2,
            bias_fc1: Array1::zeros(reduced_channels),
            bias_fc2: Array1::zeros(channels),
        }
    }

    /// Get the number of channels
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Get the reduction ratio
    pub fn reduction_ratio(&self) -> usize {
        self.reduction_ratio
    }

    /// Squeeze operation: Aggregate temporal information into channel descriptors
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (time_steps, channels)
    ///
    /// # Returns
    ///
    /// Channel descriptors of shape (channels,)
    fn squeeze(&self, x: &Array2<f64>) -> Array1<f64> {
        match self.squeeze_type {
            SqueezeType::GlobalAveragePooling => {
                x.mean_axis(Axis(0)).expect("Mean calculation failed")
            }
            SqueezeType::GlobalMaxPooling => {
                let mut max_vals = Array1::zeros(self.channels);
                for (i, col) in x.columns().into_iter().enumerate() {
                    max_vals[i] = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                }
                max_vals
            }
            SqueezeType::LastValue => {
                x.row(x.nrows() - 1).to_owned()
            }
            SqueezeType::ExponentialWeighted { alpha } => {
                let mut result = Array1::zeros(self.channels);
                let n = x.nrows();

                for (i, col) in x.columns().into_iter().enumerate() {
                    let mut ema = col[0];
                    for j in 1..n {
                        ema = alpha * col[j] + (1.0 - alpha) * ema;
                    }
                    result[i] = ema;
                }
                result
            }
        }
    }

    /// Excitation operation: Learn channel-wise dependencies
    ///
    /// Applies FC -> ReLU -> FC -> Sigmoid to produce channel weights
    ///
    /// # Arguments
    ///
    /// * `z` - Channel descriptors of shape (channels,)
    ///
    /// # Returns
    ///
    /// Channel attention weights of shape (channels,) in range [0, 1]
    fn excitation(&self, z: &Array1<f64>) -> Array1<f64> {
        // First FC layer with ReLU activation
        let fc1_out = self.weights_fc1.dot(z) + &self.bias_fc1;
        let relu_out = relu(&fc1_out);

        // Second FC layer with Sigmoid activation
        let fc2_out = self.weights_fc2.dot(&relu_out) + &self.bias_fc2;
        sigmoid(&fc2_out)
    }

    /// Forward pass through the SE block
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (time_steps, channels)
    ///
    /// # Returns
    ///
    /// Scaled output tensor of shape (time_steps, channels)
    ///
    /// # Example
    ///
    /// ```
    /// use se_trading::models::se_block::SEBlock;
    /// use ndarray::Array2;
    ///
    /// let se_block = SEBlock::new(5, 2);
    /// let input = Array2::ones((100, 5)); // 100 time steps, 5 features
    /// let output = se_block.forward(&input);
    /// ```
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // Squeeze: aggregate temporal info
        let z = self.squeeze(x);

        // Excitation: learn channel weights
        let s = self.excitation(&z);

        // Scale: multiply each channel by its weight
        let mut output = x.clone();
        for (i, &weight) in s.iter().enumerate() {
            output.column_mut(i).mapv_inplace(|v| v * weight);
        }

        output
    }

    /// Get the attention weights for interpretability
    ///
    /// This method allows you to see how much each feature is being
    /// weighted for the current input.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (time_steps, channels)
    ///
    /// # Returns
    ///
    /// Attention weights of shape (channels,) in range [0, 1]
    pub fn get_attention_weights(&self, x: &Array2<f64>) -> Array1<f64> {
        let z = self.squeeze(x);
        self.excitation(&z)
    }

    /// Update weights manually (for training or loading pre-trained weights)
    pub fn set_weights(
        &mut self,
        weights_fc1: Array2<f64>,
        weights_fc2: Array2<f64>,
        bias_fc1: Array1<f64>,
        bias_fc2: Array1<f64>,
    ) {
        assert_eq!(weights_fc1.shape(), self.weights_fc1.shape());
        assert_eq!(weights_fc2.shape(), self.weights_fc2.shape());
        assert_eq!(bias_fc1.len(), self.bias_fc1.len());
        assert_eq!(bias_fc2.len(), self.bias_fc2.len());

        self.weights_fc1 = weights_fc1;
        self.weights_fc2 = weights_fc2;
        self.bias_fc1 = bias_fc1;
        self.bias_fc2 = bias_fc2;
    }

    /// Get the current weights (for saving or inspection)
    pub fn get_weights(&self) -> (&Array2<f64>, &Array2<f64>, &Array1<f64>, &Array1<f64>) {
        (&self.weights_fc1, &self.weights_fc2, &self.bias_fc1, &self.bias_fc2)
    }
}

/// Multi-scale SE block that operates at different time resolutions
#[derive(Debug, Clone)]
pub struct MultiScaleSEBlock {
    /// SE blocks for each time scale
    se_blocks: Vec<SEBlock>,
    /// Time scales (in number of bars)
    time_scales: Vec<usize>,
    /// Fusion weights for combining multi-scale outputs
    fusion_weights: Array1<f64>,
}

impl MultiScaleSEBlock {
    /// Create a new multi-scale SE block
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of input channels
    /// * `reduction_ratio` - Reduction ratio for all SE blocks
    /// * `time_scales` - List of time scales to use (e.g., [5, 15, 60])
    pub fn new(channels: usize, reduction_ratio: usize, time_scales: Vec<usize>) -> Self {
        let n_scales = time_scales.len();
        let se_blocks: Vec<SEBlock> = time_scales
            .iter()
            .map(|_| SEBlock::new(channels, reduction_ratio))
            .collect();

        // Equal weights initially
        let fusion_weights = Array1::from_elem(n_scales, 1.0 / n_scales as f64);

        Self {
            se_blocks,
            time_scales,
            fusion_weights,
        }
    }

    /// Forward pass through all SE blocks at different scales
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (time_steps, channels)
    ///
    /// # Returns
    ///
    /// Fused output tensor of shape (time_steps, channels)
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut outputs: Vec<Array2<f64>> = Vec::new();

        for (se_block, &scale) in self.se_blocks.iter().zip(self.time_scales.iter()) {
            // Downsample to the time scale
            let downsampled = self.downsample(x, scale);

            // Apply SE block
            let se_output = se_block.forward(&downsampled);

            // Upsample back to original resolution
            let upsampled = self.upsample(&se_output, x.nrows());

            outputs.push(upsampled);
        }

        // Fuse outputs with weighted sum
        self.fuse_outputs(&outputs)
    }

    /// Downsample the input by averaging over windows
    fn downsample(&self, x: &Array2<f64>, scale: usize) -> Array2<f64> {
        let n_rows = x.nrows();
        let new_rows = (n_rows / scale).max(1);

        let mut result = Array2::zeros((new_rows, x.ncols()));

        for i in 0..new_rows {
            let start = i * scale;
            let end = ((i + 1) * scale).min(n_rows);

            for j in 0..x.ncols() {
                let sum: f64 = (start..end).map(|k| x[[k, j]]).sum();
                result[[i, j]] = sum / (end - start) as f64;
            }
        }

        result
    }

    /// Upsample by repeating values
    fn upsample(&self, x: &Array2<f64>, target_rows: usize) -> Array2<f64> {
        let mut result = Array2::zeros((target_rows, x.ncols()));
        let scale = target_rows / x.nrows();

        for i in 0..target_rows {
            let src_idx = (i / scale).min(x.nrows() - 1);
            for j in 0..x.ncols() {
                result[[i, j]] = x[[src_idx, j]];
            }
        }

        result
    }

    /// Fuse multiple outputs with weighted average
    fn fuse_outputs(&self, outputs: &[Array2<f64>]) -> Array2<f64> {
        let shape = outputs[0].shape();
        let mut result = Array2::zeros((shape[0], shape[1]));

        for (output, &weight) in outputs.iter().zip(self.fusion_weights.iter()) {
            result = result + output * weight;
        }

        result
    }

    /// Get attention weights from all scales
    pub fn get_multiscale_attention(&self, x: &Array2<f64>) -> Vec<(usize, Array1<f64>)> {
        self.se_blocks
            .iter()
            .zip(self.time_scales.iter())
            .map(|(block, &scale)| {
                let downsampled = self.downsample(x, scale);
                (scale, block.get_attention_weights(&downsampled))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_se_block_creation() {
        let se = SEBlock::new(10, 4);
        assert_eq!(se.channels(), 10);
        assert_eq!(se.reduction_ratio(), 4);
    }

    #[test]
    fn test_se_block_forward() {
        let se = SEBlock::new(5, 2);
        let input = Array2::ones((100, 5));
        let output = se.forward(&input);

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_attention_weights_range() {
        let se = SEBlock::new(5, 2);
        let input = Array2::from_shape_fn((50, 5), |(i, j)| (i * j) as f64 * 0.01);
        let weights = se.get_attention_weights(&input);

        // All weights should be between 0 and 1 (sigmoid output)
        for &w in weights.iter() {
            assert!(w >= 0.0 && w <= 1.0);
        }
    }

    #[test]
    fn test_different_squeeze_types() {
        let input = Array2::from_shape_fn((10, 3), |(i, _)| i as f64);

        let se_avg = SEBlock::with_squeeze_type(3, 1, SqueezeType::GlobalAveragePooling);
        let se_max = SEBlock::with_squeeze_type(3, 1, SqueezeType::GlobalMaxPooling);
        let se_last = SEBlock::with_squeeze_type(3, 1, SqueezeType::LastValue);

        let w_avg = se_avg.get_attention_weights(&input);
        let w_max = se_max.get_attention_weights(&input);
        let w_last = se_last.get_attention_weights(&input);

        // Different squeeze types should produce different results
        assert!(w_avg != w_max || w_avg != w_last);
    }

    #[test]
    fn test_multiscale_se_block() {
        let mse = MultiScaleSEBlock::new(4, 2, vec![1, 5, 10]);
        let input = Array2::ones((100, 4));
        let output = mse.forward(&input);

        assert_eq!(output.shape(), input.shape());
    }
}
