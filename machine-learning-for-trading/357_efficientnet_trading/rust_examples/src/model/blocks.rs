//! EfficientNet building blocks

/// MBConv block configuration
#[derive(Debug, Clone)]
pub struct MBConvConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub expand_ratio: usize,
    pub se_ratio: f64,
    pub drop_connect_rate: f64,
}

impl MBConvConfig {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        expand_ratio: usize,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            expand_ratio,
            se_ratio: 0.25,
            drop_connect_rate: 0.0,
        }
    }

    /// Check if this block uses residual connection
    pub fn use_residual(&self) -> bool {
        self.stride == 1 && self.in_channels == self.out_channels
    }

    /// Get expanded channels
    pub fn expanded_channels(&self) -> usize {
        self.in_channels * self.expand_ratio
    }

    /// Get SE squeeze channels
    pub fn se_channels(&self) -> usize {
        ((self.in_channels as f64 * self.se_ratio).ceil() as usize).max(1)
    }
}

/// Swish activation function implementation
#[derive(Debug, Clone, Copy, Default)]
pub struct SwishActivation;

impl SwishActivation {
    /// Apply swish activation: x * sigmoid(x)
    pub fn forward(&self, x: f64) -> f64 {
        x * sigmoid(x)
    }

    /// Apply swish to a vector
    pub fn forward_vec(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.forward(v)).collect()
    }

    /// Apply swish to a 2D array
    pub fn forward_2d(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| self.forward_vec(row))
            .collect()
    }
}

/// Sigmoid function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax function
pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = x.iter().map(|&v| (v - max).exp()).sum();
    x.iter().map(|&v| (v - max).exp() / exp_sum).collect()
}

/// ReLU activation
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Hardswish activation (used in EfficientNetV2)
pub fn hardswish(x: f64) -> f64 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}

/// Squeeze-and-Excitation block configuration
#[derive(Debug, Clone)]
pub struct SEConfig {
    pub in_channels: usize,
    pub squeeze_channels: usize,
}

impl SEConfig {
    pub fn new(in_channels: usize, reduction_ratio: f64) -> Self {
        let squeeze_channels = ((in_channels as f64 * reduction_ratio).ceil() as usize).max(1);
        Self {
            in_channels,
            squeeze_channels,
        }
    }
}

/// Drop connect (stochastic depth) helper
#[derive(Debug, Clone)]
pub struct DropConnect {
    pub drop_rate: f64,
    pub training: bool,
}

impl DropConnect {
    pub fn new(drop_rate: f64) -> Self {
        Self {
            drop_rate,
            training: true,
        }
    }

    /// Get keep probability
    pub fn keep_prob(&self) -> f64 {
        1.0 - self.drop_rate
    }

    /// Check if should apply drop during training
    pub fn should_drop(&self, random_val: f64) -> bool {
        self.training && random_val < self.drop_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swish() {
        let swish = SwishActivation;

        // swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((swish.forward(0.0) - 0.0).abs() < 0.001);

        // swish(x) should be positive for positive x
        assert!(swish.forward(1.0) > 0.0);

        // swish(x) should be negative for some negative x (not monotonic)
        assert!(swish.forward(-5.0) < 0.0);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Higher logit should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_mbconv_config() {
        let config = MBConvConfig::new(32, 32, 3, 1, 6);
        assert!(config.use_residual());
        assert_eq!(config.expanded_channels(), 192);

        let config2 = MBConvConfig::new(32, 64, 3, 2, 6);
        assert!(!config2.use_residual());
    }
}
