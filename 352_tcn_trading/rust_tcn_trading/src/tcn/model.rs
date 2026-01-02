//! Full TCN Model Implementation

use ndarray::{Array1, Array2};

use super::block::TCNResidualBlock;

/// TCN Model Configuration
#[derive(Debug, Clone)]
pub struct TCNConfig {
    /// Number of input features
    pub input_size: usize,
    /// Number of output classes/values
    pub output_size: usize,
    /// Number of channels in each layer
    pub num_channels: Vec<usize>,
    /// Convolution kernel size
    pub kernel_size: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl Default for TCNConfig {
    fn default() -> Self {
        Self {
            input_size: 10,
            output_size: 3, // Up, Down, Neutral
            num_channels: vec![64, 64, 64, 64],
            kernel_size: 3,
            dropout: 0.2,
        }
    }
}

impl TCNConfig {
    /// Create config for binary classification (up/down)
    pub fn binary_classification(input_size: usize) -> Self {
        Self {
            input_size,
            output_size: 2,
            ..Default::default()
        }
    }

    /// Create config for regression (predict return value)
    pub fn regression(input_size: usize) -> Self {
        Self {
            input_size,
            output_size: 1,
            ..Default::default()
        }
    }

    /// Create config for multi-class classification
    pub fn multi_class(input_size: usize, num_classes: usize) -> Self {
        Self {
            input_size,
            output_size: num_classes,
            ..Default::default()
        }
    }

    /// Calculate total receptive field
    pub fn receptive_field(&self) -> usize {
        let num_layers = self.num_channels.len();
        // Receptive field = 1 + 2 * (kernel_size - 1) * sum(2^i for i in 0..num_layers)
        let dilation_sum: usize = (0..num_layers).map(|i| 1 << i).sum();
        1 + 2 * (self.kernel_size - 1) * dilation_sum
    }
}

/// Temporal Convolutional Network
#[derive(Debug)]
pub struct TCN {
    /// Configuration
    pub config: TCNConfig,
    /// Stack of residual blocks
    pub blocks: Vec<TCNResidualBlock>,
    /// Output projection weights
    pub output_weights: Array2<f64>,
    /// Output projection bias
    pub output_bias: Array1<f64>,
}

impl TCN {
    /// Create a new TCN model
    pub fn new(config: TCNConfig) -> Self {
        let mut blocks = Vec::new();
        let mut in_channels = config.input_size;

        // Create residual blocks with exponentially increasing dilation
        for (i, &out_channels) in config.num_channels.iter().enumerate() {
            let dilation = 1 << i; // 1, 2, 4, 8, ...
            let block = TCNResidualBlock::new(
                in_channels,
                out_channels,
                config.kernel_size,
                dilation,
                config.dropout,
            );
            blocks.push(block);
            in_channels = out_channels;
        }

        // Output projection
        let last_channels = *config.num_channels.last().unwrap_or(&config.input_size);
        let output_weights = Array2::zeros((config.output_size, last_channels));
        let output_bias = Array1::zeros(config.output_size);

        Self {
            config,
            blocks,
            output_weights,
            output_bias,
        }
    }

    /// Forward pass through the TCN
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [input_size, seq_len]
    /// * `training` - Whether in training mode
    ///
    /// # Returns
    /// Output tensor of shape [output_size]
    pub fn forward(&self, input: &Array2<f64>, training: bool) -> Array1<f64> {
        let mut x = input.clone();

        // Pass through all residual blocks
        for block in &self.blocks {
            x = block.forward(&x, training);
        }

        // Global average pooling over the sequence dimension
        let pooled = x.mean_axis(ndarray::Axis(1)).unwrap();

        // Output projection
        let mut output = self.output_bias.clone();
        for i in 0..self.config.output_size {
            for j in 0..pooled.len() {
                output[i] += self.output_weights[[i, j]] * pooled[j];
            }
        }

        output
    }

    /// Predict class probabilities (softmax output)
    pub fn predict_proba(&self, input: &Array2<f64>) -> Array1<f64> {
        let logits = self.forward(input, false);
        Self::softmax(&logits)
    }

    /// Predict class label
    pub fn predict_class(&self, input: &Array2<f64>) -> usize {
        let proba = self.predict_proba(input);
        proba
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Softmax function
    fn softmax(logits: &Array1<f64>) -> Array1<f64> {
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: Array1<f64> = logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        exp_logits / sum_exp
    }

    /// Get receptive field of the model
    pub fn receptive_field(&self) -> usize {
        self.config.receptive_field()
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        let block_params: usize = self.blocks.iter().map(|b| b.num_parameters()).sum();
        let output_params = self.output_weights.len() + self.output_bias.len();
        block_params + output_params
    }

    /// Get model summary as string
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("TCN Model Summary\n");
        s.push_str("=================\n");
        s.push_str(&format!("Input size: {}\n", self.config.input_size));
        s.push_str(&format!("Output size: {}\n", self.config.output_size));
        s.push_str(&format!("Num layers: {}\n", self.blocks.len()));
        s.push_str(&format!("Channels: {:?}\n", self.config.num_channels));
        s.push_str(&format!("Kernel size: {}\n", self.config.kernel_size));
        s.push_str(&format!("Dropout: {}\n", self.config.dropout));
        s.push_str(&format!("Receptive field: {}\n", self.receptive_field()));
        s.push_str(&format!("Total parameters: {}\n", self.num_parameters()));
        s
    }
}

/// Simple training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Validation split ratio
    pub validation_split: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            patience: 10,
            validation_split: 0.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcn_creation() {
        let config = TCNConfig::default();
        let tcn = TCN::new(config);

        assert_eq!(tcn.blocks.len(), 4);
        assert!(tcn.num_parameters() > 0);
    }

    #[test]
    fn test_receptive_field() {
        let config = TCNConfig {
            num_channels: vec![64, 64, 64, 64],
            kernel_size: 3,
            ..Default::default()
        };

        // RF = 1 + 2 * (3-1) * (1+2+4+8) = 1 + 4 * 15 = 61
        assert_eq!(config.receptive_field(), 61);
    }

    #[test]
    fn test_forward_pass() {
        let config = TCNConfig {
            input_size: 5,
            output_size: 3,
            num_channels: vec![16, 16],
            kernel_size: 3,
            dropout: 0.0,
        };
        let tcn = TCN::new(config);

        let input = Array2::ones((5, 50));
        let output = tcn.forward(&input, false);

        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_predict_proba() {
        let config = TCNConfig {
            input_size: 5,
            output_size: 3,
            num_channels: vec![16, 16],
            kernel_size: 3,
            dropout: 0.0,
        };
        let tcn = TCN::new(config);

        let input = Array2::ones((5, 50));
        let proba = tcn.predict_proba(&input);

        // Probabilities should sum to 1
        assert!((proba.sum() - 1.0).abs() < 1e-6);

        // All probabilities should be positive
        for &p in proba.iter() {
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn test_model_summary() {
        let config = TCNConfig::default();
        let tcn = TCN::new(config);
        let summary = tcn.summary();

        assert!(summary.contains("TCN Model Summary"));
        assert!(summary.contains("Receptive field"));
    }
}
